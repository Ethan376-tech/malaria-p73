# -*- coding: utf-8 -*-
"""Phase 2B flagship (3-seed confirm): does the reverse-schedule (loose->strict) dynamic filter robustly beat the
baselines on YOLOv8n teacher-student? 3 seeds x {naive-matched, static-multi, dyn-single-rev, dyn-multi-rev}. Reuses
build_y3 fixed base D1. Custom AP. Output -> y5_dynamic_3seed/results.json"""
import os, sys, json, glob, shutil
os.environ['XFORMERS_DISABLED']='1'; os.environ['PYTORCH_CUDA_ALLOC_CONF']='expandable_segments:True'
import numpy as np, torch, torch.nn as nn
import torchvision.transforms.functional as TF, torchvision.transforms as T
from PIL import Image; Image.MAX_IMAGE_PIXELS=None
from ultralytics import YOLO
from sklearn.decomposition import PCA; from sklearn.covariance import LedoitWolf
sys.path.insert(0,'/root/autodl-tmp/A_Dissertation/references/RedDino/train'); sys.path.append('/root/A_Dissertation')
from common.paths import OUTPUTS, CHECKPOINTS
from dinov2.models.vision_transformer import vit_small
dev='cuda'; CROP=640; TILE=224; T_MC=12; MC_CAP=40; ROUNDS=3; EP=8; LR0=0.002; C_HI=0.20; C_LO=0.05
YO=OUTPUTS/'experiments'/'s5_yolo'; S3=OUTPUTS/'experiments'/'s3_filter'; RES=OUTPUTS/'experiments'/'y5_dynamic_3seed'; RES.mkdir(parents=True,exist_ok=True)
FIN=json.load(open(S3/'final.json'))['calib']; b_protect=FIN['b_protect']; a_star=FIN['a_star']
pb=np.load(S3/'perbox_multi2.npz'); dv=pb['field_id'].astype(int)<95; tpm=pb['is_tp']==1
faint=dv&tpm&(pb['p_mean']<=np.quantile(pb['p_mean'][dv&tpm],1/3)); b_tta=float(np.quantile(pb['tta_var'][faint],0.10))
_E=pb['emb'][dv&tpm]; _mu=_E.mean(0); _sd=_E.std(0)+1e-6; _pca=PCA(40,random_state=0).fit((_E-_mu)/_sd); _lw=LedoitWolf().fit(_pca.transform((_E-_mu)/_sd))
def maha_of(emb): return _lw.mahalanobis(_pca.transform((emb-_mu)/_sd)) if len(emb) else np.array([])
norm=T.Compose([T.ToTensor(),T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
ftm=vit_small(patch_size=16,img_size=224,block_chunks=0,init_values=1.0,drop_path_rate=0.1)
ftm.load_state_dict(torch.load(CHECKPOINTS/'ssl_vits_thickdino_ft_ibadan'/'encoder.pth',map_location='cpu'),strict=False); ftm=ftm.to(dev).eval()
fth=nn.Sequential(nn.Dropout(0.2),nn.Linear(384,1)).to(dev); fth.load_state_dict(torch.load(CHECKPOINTS/'ssl_vits_thickdino_ft_ibadan'/'head.pth',map_location='cpu')); fth.eval()
@torch.no_grad()
def mc(tiles,bs=48):
    for mod in list(ftm.modules())+list(fth.modules()):
        if isinstance(mod,nn.Dropout): mod.train()
    for mod in ftm.modules():
        if mod.__class__.__name__=='DropPath': mod.train()
    M=[];V=[]
    for i in range(0,len(tiles),bs):
        xb=torch.stack([norm(Image.fromarray(t)) for t in tiles[i:i+bs]]).to(dev); pr=[]
        for _ in range(T_MC): pr.append(torch.sigmoid(fth(ftm.forward_features(xb)['x_norm_clstoken']).squeeze(-1)).cpu().numpy())
        A=np.stack(pr); M.append(A.mean(0)); V.append(A.var(0))
    ftm.eval(); fth.eval(); return (np.concatenate(M),np.concatenate(V)) if M else (np.array([]),np.array([]))
@torch.no_grad()
def emb_tta(tiles,bs=48):
    ftm.eval(); fth.eval(); EM=[];TV=[]
    for i in range(0,len(tiles),bs):
        xb=torch.stack([norm(Image.fromarray(t)) for t in tiles[i:i+bs]]).to(dev)
        EM.append(ftm.forward_features(xb)['x_norm_clstoken'].cpu().numpy())
        views=[xb,torch.flip(xb,[3]),torch.flip(xb,[2]),torch.rot90(xb,1,[2,3]),torch.rot90(xb,2,[2,3]),torch.rot90(xb,3,[2,3])]
        pv=[torch.sigmoid(fth(ftm.forward_features(v)['x_norm_clstoken']).squeeze(-1)).cpu().numpy() for v in views]
        TV.append(np.stack(pv).var(0))
    return (np.concatenate(EM),np.concatenate(TV)) if EM else (np.zeros((0,384),np.float32),np.array([]))
def yolo_predict(model, ip, conf=0.15):
    r=model.predict(ip, imgsz=CROP, conf=conf, verbose=False, device=0)[0]; return r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()
def yolo_line(b): cx=(b[0]+b[2])/2/CROP; cy=(b[1]+b[3])/2/CROP; w=(b[2]-b[0])/CROP; h=(b[3]-b[1])/CROP; return f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"
train_imgs=sorted(glob.glob(str(YO/'ds_a0'/'images'/'train'/'*.jpg')))
def _cap(boxes,scores):
    if len(boxes)==0: return boxes,scores
    order=np.argsort(-scores)[:MC_CAP]; return boxes[order],scores[order]
def score_dets(model, need_multi):
    DET={}
    for ip in train_imgs:
        arr=np.asarray(Image.open(ip).convert('RGB')); boxes,scores=_cap(*yolo_predict(model,ip)); base=os.path.basename(ip)[:-4]
        if len(boxes)==0: DET[base]=[]; continue
        tiles=[]
        for b in boxes:
            cx=(b[0]+b[2])/2; cy=(b[1]+b[3])/2; x0=int(min(max(cx-TILE/2,0),CROP-TILE)); y0=int(min(max(cy-TILE/2,0),CROP-TILE)); tiles.append(arr[y0:y0+TILE,x0:x0+TILE])
        pm,pv=mc(tiles)
        if need_multi: em,tta=emb_tta(tiles); mh=maha_of(em)
        else: tta=np.zeros(len(boxes)); mh=np.zeros(len(boxes))
        DET[base]=[(yolo_line(boxes[j]),float(scores[j]),float(pm[j]),float(pv[j]),float(tta[j]),float(mh[j])) for j in range(len(boxes))]
    return DET
def keep_single(d): return (d[3]>=b_protect) or (d[2]>=a_star)
def keep_multi(d,tau): return ((d[3]>=b_protect) or (d[4]>=b_tta) or (d[2]>=a_star)) and (d[5]<=tau)
def write_labels(DET, labdir, config, r, tau, N=None):
    labdir.mkdir(parents=True,exist_ok=True)
    if config=='naive':
        alls=sorted([d[1] for v in DET.values() for d in v],reverse=True); thr=alls[min(N,len(alls))-1] if (N and alls) else 1.1; nlab=0
        for base,v in DET.items():
            sel=[d for d in v if d[1]>=thr]; (labdir/(base+'.txt')).write_text("\n".join(d[0] for d in sel)); nlab+=len(sel)
        return nlab
    c = (np.linspace(C_LO,C_HI,ROUNDS) if 'rev' in config else np.linspace(C_HI,C_LO,ROUNDS))[r] if 'dyn' in config else -1.0
    nlab=0
    for base,v in DET.items():
        sel=[d for d in v if (keep_multi(d,tau) if 'multi' in config else keep_single(d)) and (c<0 or d[1]>=c)]
        (labdir/(base+'.txt')).write_text("\n".join(d[0] for d in sel)); nlab+=len(sel)
    return nlab
def _populate(srcdir, dstdir, ext, copy=False):
    for f in glob.glob(str(srcdir)+f'/*.{ext}'):
        d=str(dstdir/os.path.basename(f))
        if copy: shutil.copy(f,d)
        else:
            try: os.link(f,d)
            except OSError: shutil.copy(f,d)
def make_ds(name, labdir):
    root=RES/name
    for sub in ['images/train','images/val','labels/train','labels/val']:
        d=root/sub
        if d.exists(): shutil.rmtree(d)
        d.mkdir(parents=True,exist_ok=True)
    _populate(YO/'ds_a0'/'images'/'train', root/'images'/'train', 'jpg'); _populate(YO/'ds_a0'/'images'/'val', root/'images'/'val', 'jpg')
    _populate(labdir, root/'labels'/'train', 'txt', copy=True); _populate(YO/'ds_a0'/'labels'/'val', root/'labels'/'val', 'txt', copy=True)
    for cc in glob.glob(str(root)+'/labels/*.cache'): os.remove(cc)
    (root/'data.yaml').write_text(f"path: {root}\ntrain: images/train\nval: images/val\nnc: 1\nnames: ['parasite']\n"); return str(root/'data.yaml')
val_imgs=sorted(glob.glob(str(YO/'ds_a0'/'images'/'val'/'*.jpg'))); VAL_LAB=YO/'ds_a0'/'labels'/'val'
def load_gt(ip):
    lp=VAL_LAB/(os.path.basename(ip)[:-4]+'.txt'); bx=[]
    if lp.exists():
        for ln in lp.read_text().splitlines():
            f=ln.split()
            if len(f)==5: _,cx,cy,w,h=f; cx,cy,w,h=float(cx)*CROP,float(cy)*CROP,float(w)*CROP,float(h)*CROP; bx.append([cx-w/2,cy-h/2,cx+w/2,cy+h/2])
    return bx
def iou(a,b):
    ix1,iy1=max(a[0],b[0]),max(a[1],b[1]); ix2,iy2=min(a[2],b[2]),min(a[3],b[3]); iw,ih=max(0,ix2-ix1),max(0,iy2-iy1); it=iw*ih
    ua=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-it; return it/ua if ua>0 else 0.0
gts=[load_gt(ip) for ip in val_imgs]; cropgt=[len(g) for g in gts]; med=np.median([c for c in cropgt if c>0]) if any(cropgt) else 0
def evaluate(model):
    at={0.3:[],0.5:[]}; sc=[]; ngt=0; btp={'low':[],'high':[]}; bsc={'low':[],'high':[]}; bgt={'low':0,'high':0}
    for idx,ip in enumerate(val_imgs):
        gt=gts[idx]; ngt+=len(gt); bd='low' if cropgt[idx]<=med else 'high'; bgt[bd]+=len(gt)
        bxs,scs=yolo_predict(model,ip,conf=0.05); order=np.argsort(-scs)
        for th in (0.3,0.5):
            matched=set()
            for j in order:
                best=-1;bj=-1
                for jg,g in enumerate(gt):
                    if jg in matched: continue
                    v=iou(bxs[j],g)
                    if v>best: best,bj=v,jg
                tp=1 if best>=th and bj>=0 else 0
                if tp: matched.add(bj)
                at[th].append(tp)
                if th==0.3: sc.append(float(scs[j])); btp[bd].append(tp); bsc[bd].append(float(scs[j]))
    def ap(tps,scs,ng):
        if not tps: return 0.0
        o=np.argsort(-np.array(scs)); tps=np.array(tps)[o]; ctp=np.cumsum(tps); cfp=np.cumsum(1-tps); rec=ctp/max(1,ng); pre=ctp/np.maximum(1,ctp+cfp)
        mr=np.concatenate([[0],rec,[1]]); mp=np.concatenate([[0],pre,[0]])
        for i in range(len(mp)-1,0,-1): mp[i-1]=max(mp[i-1],mp[i])
        k=np.where(mr[1:]!=mr[:-1])[0]; return float(np.sum((mr[k+1]-mr[k])*mp[k+1]))
    return dict(**{f'AP@{t}':round(ap(at[t],sc,ngt),3) for t in (0.3,0.5)}, low=round(ap(btp['low'],bsc['low'],bgt['low']),3), high=round(ap(btp['high'],bsc['high'],bgt['high']),3))
def train_yolo(init_wts, yaml, name, seed, epochs, lr0):
    m=YOLO(init_wts); m.train(data=yaml, epochs=epochs, imgsz=CROP, batch=16, device=0, workers=8, lr0=lr0, warmup_epochs=1,
        project=str(RES/'runs'), name=name, exist_ok=True, verbose=False, plots=False, seed=seed); return str(RES/'runs'/name/'weights'/'best.pt')
D1=str(OUTPUTS/'experiments'/'y3_yolo_ts'/'runs'/'d1_base'/'weights'/'best.pt'); print(f'reuse D1 {D1}',flush=True)
DET0=score_dets(YOLO(D1), True); tau=float(np.quantile(np.array([d[5] for v in DET0.values() for d in v]),0.99))
N=sum(1 for v in DET0.values() for d in v if keep_single(d)); print(f'tau99 {tau:.1f} naive_N {N}',flush=True)
CONFIGS=['naive','static-multi','dyn-single-rev','dyn-multi-rev']
def ts_run(config, seed):
    torch.manual_seed(seed); np.random.seed(seed); teacher_wts=D1
    for r in range(ROUNDS):
        need_multi=('multi' in config) or (config=='naive' and False)
        DET=DET0 if (r==0 and seed==0) else score_dets(YOLO(teacher_wts), 'multi' in config)
        n=write_labels(DET, RES/f'lab_{config}', config, r, tau, N=N)
        yaml=make_ds(f'ds_{config}', RES/f'lab_{config}'); teacher_wts=train_yolo(teacher_wts, yaml, f'{config}_s{seed}_r{r}', seed, EP, LR0)
    return evaluate(YOLO(teacher_wts))
runs={c:[] for c in CONFIGS}
for seed in [0,1,2]:
    for config in CONFIGS:
        print(f'=== seed {seed} {config} ===',flush=True); r=ts_run(config,seed); runs[config].append(r); print(f'  {config} s{seed} -> {r}',flush=True)
    json.dump(runs,open(RES/'results.json','w'),indent=2)
agg={}
for c in CONFIGS:
    ks=list(runs[c][0].keys()); agg[c]={m:{'mean':round(float(np.mean([r[m] for r in runs[c]])),3),'std':round(float(np.std([r[m] for r in runs[c]])),3)} for m in ks}
json.dump(dict(agg=agg,runs=runs,tau99=round(tau,1),naive_N=N),open(RES/'results.json','w'),indent=2)
print('\n=== reverse-dynamic 3-seed (mean±std, YOLO TS, custom AP) ===')
for c in CONFIGS: print(f'  {c:16s} {agg[c]}')
print('Y5_DONE',flush=True)
