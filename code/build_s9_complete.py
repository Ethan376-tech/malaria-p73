# -*- coding: utf-8 -*-
"""Completes the Table 4.9 weak-detector (RetinaNet) teacher-student retain grid so it matches the MicroYOLO grid.
naive (conf>0.7) and single (faintness single-signal) already exist at 3 seeds; this fills the three MISSING rows:
  multi      = faintness-protected multi-signal ((MC-var OR TTA-var OR MC-mean) AND typicality Mahalanobis<=tau) [Eq 3.8]
  dyn-single = dynamic single-signal: single retain PLUS a per-round confidence floor c_r on a reverse (loose->strict)
               schedule c_r = linspace(C_LO,C_HI,#rounds)[r]  [Eq 3.10]
  dyn-multi  = dynamic multi-signal: multi retain PLUS the same c_r floor
Same EMA-teacher loop / base / eval as build_s9_teacher_student.py; multi signals + tau identical to build_y14
(shared dev calibration s3_filter/perbox_multi2.npz; tau = 0.99-quantile of the RetinaNet base Mahalanobis distances;
C_LO/C_HI = 20th/55th percentile of the RetinaNet base detection scores, matching the y14 MicroYOLO gate recipe).
3 SEEDS (== main-table protocol). PER-RUN try/except with incremental save -> a mid-run failure never loses the
completed runs. Output -> outputs/experiments/s9_teacher_student/ts_complete3_results.json . Does NOT overwrite the
existing naive/single results (teacher_student_results.json)."""
import os, sys, json, glob, copy, traceback
os.environ['XFORMERS_DISABLED']='1'; os.environ['PYTORCH_CUDA_ALLOC_CONF']='expandable_segments:True'
import numpy as np, torch, torch.nn as nn
import torchvision.transforms.functional as TF
import torchvision.transforms as T
from PIL import Image; Image.MAX_IMAGE_PIXELS=None
from torchvision.models.detection import retinanet_resnet50_fpn
from sklearn.decomposition import PCA; from sklearn.covariance import LedoitWolf
sys.path.insert(0,'/root/autodl-tmp/A_Dissertation/references/RedDino/train'); sys.path.append('/root/A_Dissertation')
from common.paths import OUTPUTS, CHECKPOINTS
from dinov2.models.vision_transformer import vit_small
dev='cuda'
CROP=640; TILE=224; T_MC=15; EPOCHS=10; REFRESH=[0,5]; EMA=0.95; LR=0.002; BATCH=6; MC_CAP=40
MODES=['multi','dyn-single','dyn-multi']; SEEDS=[0,1,2]
YO=OUTPUTS/'experiments'/'s5_yolo'; S3=OUTPUTS/'experiments'/'s3_filter'; S6=OUTPUTS/'experiments'/'s6_retinanet'
RES=OUTPUTS/'experiments'/'s9_teacher_student'; RES.mkdir(parents=True,exist_ok=True)
OUTJSON=RES/'ts_complete3_results.json'
FIN=json.load(open(S3/'final.json'))['calib']; b_protect=FIN['b_protect']; a_star=FIN['a_star']
# ---- shared multi-signal calibration (identical to build_y14_microyolo_grid.py) ----
pb=np.load(S3/'perbox_multi2.npz'); dv=pb['field_id'].astype(int)<95; tpm=pb['is_tp']==1
faint=dv&tpm&(pb['p_mean']<=np.quantile(pb['p_mean'][dv&tpm],1/3)); b_tta=float(np.quantile(pb['tta_var'][faint],0.10))
_E=pb['emb'][dv&tpm]; _mu=_E.mean(0); _sd=_E.std(0)+1e-6; _pca=PCA(40,random_state=0).fit((_E-_mu)/_sd); _lw=LedoitWolf().fit(_pca.transform((_E-_mu)/_sd))
def maha_of(emb): return _lw.mahalanobis(_pca.transform((emb-_mu)/_sd)) if len(emb) else np.array([])
norm=T.Compose([T.ToTensor(),T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
def build_ret(): return retinanet_resnet50_fpn(num_classes=2,weights=None,weights_backbone='IMAGENET1K_V1').to(dev)
ftm=vit_small(patch_size=16,img_size=224,block_chunks=0,init_values=1.0,drop_path_rate=0.1)
ftm.load_state_dict(torch.load(CHECKPOINTS/'ssl_vits_thickdino_ft_ibadan'/'encoder.pth',map_location='cpu'),strict=False); ftm=ftm.to(dev).eval()
fth=nn.Sequential(nn.Dropout(0.2),nn.Linear(384,1)).to(dev); fth.load_state_dict(torch.load(CHECKPOINTS/'ssl_vits_thickdino_ft_ibadan'/'head.pth',map_location='cpu')); fth.eval()
@torch.no_grad()
def mc_boxes(tiles,bs=48):
    for mod in list(ftm.modules())+list(fth.modules()):
        if isinstance(mod,nn.Dropout): mod.train()
    for mod in ftm.modules():
        if mod.__class__.__name__=='DropPath': mod.train()
    M=[];V=[]
    for i in range(0,len(tiles),bs):
        xb=torch.stack([norm(Image.fromarray(t)) for t in tiles[i:i+bs]]).to(dev); pr=[]
        for _ in range(T_MC): ff=ftm.forward_features(xb); pr.append(torch.sigmoid(fth(ff['x_norm_clstoken']).squeeze(-1)).cpu().numpy())
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
train_imgs=sorted(glob.glob(str(YO/'ds_a0'/'images'/'train'/'*.jpg')))
def _tiles_for(arr, boxes, keep):
    tiles=[]
    for j in keep:
        cx=(boxes[j][0]+boxes[j][2])/2; cy=(boxes[j][1]+boxes[j][3])/2
        x0=int(min(max(cx-TILE/2,0),CROP-TILE)); y0=int(min(max(cy-TILE/2,0),CROP-TILE)); tiles.append(arr[y0:y0+TILE,x0:x0+TILE])
    return tiles
def yolo_line(b): cx=(b[0]+b[2])/2/CROP; cy=(b[1]+b[3])/2/CROP; w=(b[2]-b[0])/CROP; h=(b[3]-b[1])/CROP; return f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"
@torch.no_grad()
def compute_tau():
    base=build_ret(); base.load_state_dict(torch.load(S6/'retinanet_r50_protected.pth')); base.eval(); allmh=[]
    for ip in train_imgs:
        im=Image.open(ip).convert('RGB'); arr=np.asarray(im); x=TF.to_tensor(im).to(dev)
        pred=base([x])[0]; boxes=pred['boxes'].cpu().numpy(); scores=pred['scores'].cpu().numpy()
        keep=np.where(scores>0.15)[0]; keep=keep[np.argsort(-scores[keep])][:MC_CAP]
        if len(keep)==0: continue
        em,_=emb_tta(_tiles_for(arr,boxes,keep)); allmh.extend(list(maha_of(em)))
    del base; torch.cuda.empty_cache(); return float(np.quantile(np.array(allmh),0.99))
@torch.no_grad()
def compute_gate():
    """C_LO/C_HI = 20th/55th percentile of the RetinaNet base detection scores on the train crops (y14 recipe)."""
    base=build_ret(); base.load_state_dict(torch.load(S6/'retinanet_r50_protected.pth')); base.eval(); allsc=[]
    for ip in train_imgs:
        x=TF.to_tensor(Image.open(ip).convert('RGB')).to(dev); pred=base([x])[0]; scores=pred['scores'].cpu().numpy()
        keep=np.where(scores>0.15)[0]; keep=keep[np.argsort(-scores[keep])][:MC_CAP]; allsc.extend(list(scores[keep]))
    del base; torch.cuda.empty_cache(); a=np.array(allsc)
    return float(np.percentile(a,20)), float(np.percentile(a,55))
NEED_MULTI=any('multi' in m for m in MODES); NEED_DYN=any('dyn' in m for m in MODES)
TAU=compute_tau() if NEED_MULTI else None
C_LO,C_HI=compute_gate() if NEED_DYN else (None,None)
NROUNDS=len(REFRESH)
print(f'RetinaNet base tau99 = {TAU} ; C_LO = {C_LO} ; C_HI = {C_HI} ; b_tta = {b_tta:.3e} ; b_protect = {b_protect:.3e} ; a_star = {a_star:.4f}',flush=True)
@torch.no_grad()
def teacher_generate(teacher, dsroot, mode, round_idx):
    teacher.eval(); (dsroot/'labels'/'train').mkdir(parents=True,exist_ok=True); nlab=0
    is_dyn='dyn' in mode; use_multi='multi' in mode
    c_r=float(np.linspace(C_LO,C_HI,NROUNDS)[round_idx]) if is_dyn else -1.0   # reverse (loose->strict) confidence floor, Eq 3.10
    for ip in train_imgs:
        im=Image.open(ip).convert('RGB'); arr=np.asarray(im); x=TF.to_tensor(im).to(dev)
        pred=teacher([x])[0]; boxes=pred['boxes'].cpu().numpy(); scores=pred['scores'].cpu().numpy()
        keep=np.where(scores>0.15)[0]; keep=keep[np.argsort(-scores[keep])][:MC_CAP]; base=os.path.basename(ip)[:-4]
        if len(keep)==0: (dsroot/'labels'/'train'/(base+'.txt')).write_text(""); continue
        tiles=_tiles_for(arr,boxes,keep); pm,pv=mc_boxes(tiles)
        if use_multi:
            em,tta=emb_tta(tiles); mh=maha_of(em)
            base_keep=[((pv[ji]>=b_protect) or (tta[ji]>=b_tta) or (pm[ji]>=a_star)) and (mh[ji]<=TAU) for ji in range(len(keep))]
        else:
            base_keep=[(pv[ji]>=b_protect or pm[ji]>=a_star) for ji in range(len(keep))]
        sel=[keep[ji] for ji in range(len(keep)) if base_keep[ji] and (c_r<0 or scores[keep[ji]]>=c_r)]
        (dsroot/'labels'/'train'/(base+'.txt')).write_text("\n".join(yolo_line(boxes[j]) for j in sel)); nlab+=len(sel)
    return nlab
class DetDS(torch.utils.data.Dataset):
    def __init__(self,imgdir,labdir): self.imgs=sorted(glob.glob(str(imgdir/'*.jpg'))); self.labdir=labdir
    def __len__(self): return len(self.imgs)
    def __getitem__(self,i):
        ip=self.imgs[i]; x=TF.to_tensor(Image.open(ip).convert('RGB')); lp=self.labdir/(os.path.basename(ip)[:-4]+'.txt'); bx=[]
        if lp.exists():
            for ln in lp.read_text().splitlines():
                f=ln.split()
                if len(f)==5: _,cx,cy,w,h=f; cx,cy,w,h=float(cx)*CROP,float(cy)*CROP,float(w)*CROP,float(h)*CROP; bx.append([cx-w/2,cy-h/2,cx+w/2,cy+h/2])
        bx=torch.tensor(bx,dtype=torch.float32).reshape(-1,4)
        if len(bx): bx[:,[0,2]]=bx[:,[0,2]].clamp(0,CROP); bx[:,[1,3]]=bx[:,[1,3]].clamp(0,CROP); k=(bx[:,2]-bx[:,0]>=1)&(bx[:,3]-bx[:,1]>=1); bx=bx[k]
        return x,{'boxes':bx,'labels':torch.ones(len(bx),dtype=torch.int64)}
def collate(b): return tuple(zip(*b))
def train_epoch(model,imgdir,labdir,opt):
    dl=torch.utils.data.DataLoader(DetDS(imgdir,labdir),batch_size=BATCH,shuffle=True,collate_fn=collate,num_workers=6)
    params=[p for p in model.parameters() if p.requires_grad]; model.train()
    for imgs,tgts in dl:
        imgs=[im.to(dev) for im in imgs]; tgts=[{k:v.to(dev) for k,v in t.items()} for t in tgts]
        if sum(len(t['boxes']) for t in tgts)==0: continue
        loss=sum(model(imgs,tgts).values()); opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(params,5.0); opt.step()
def ema(teacher,student,a):
    with torch.no_grad():
        for tp,sp in zip(teacher.parameters(),student.parameters()): tp.mul_(a).add_(sp,alpha=1-a)
        for tb,sb in zip(teacher.buffers(),student.buffers()): tb.copy_(sb)
def iou(a,b):
    ix1,iy1=max(a[0],b[0]),max(a[1],b[1]); ix2,iy2=min(a[2],b[2]),min(a[3],b[3]); iw,ih=max(0,ix2-ix1),max(0,iy2-iy1); it=iw*ih
    ua=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-it; return it/ua if ua>0 else 0.0
TR_IMG=YO/'ds_a0'/'images'/'train'; VAL_IMG=YO/'ds_a0'/'images'/'val'; VAL_LAB=YO/'ds_a0'/'labels'/'val'
@torch.no_grad()
def evaluate(model):
    ds=DetDS(VAL_IMG,VAL_LAB); model.eval(); cropgt=[len(ds[i][1]['boxes']) for i in range(len(ds))]; med=np.median([c for c in cropgt if c>0]) if any(cropgt) else 0
    dl=torch.utils.data.DataLoader(ds,batch_size=BATCH,shuffle=False,collate_fn=collate,num_workers=6)
    all_tp={0.3:[],0.5:[]}; sc=[]; ngt=0; btp={'low':[],'high':[]}; bsc={'low':[],'high':[]}; bgt={'low':0,'high':0}; idx=0
    for imgs,tgts in dl:
        preds=model([im.to(dev) for im in imgs])
        for pred,tgt in zip(preds,tgts):
            gt=tgt['boxes'].numpy().tolist(); ngt+=len(gt); bd='low' if cropgt[idx]<=med else 'high'; bgt[bd]+=len(gt); idx+=1
            bxs=pred['boxes'].cpu().numpy(); scs=pred['scores'].cpu().numpy(); order=np.argsort(-scs)
            for iouth in (0.3,0.5):
                matched=set()
                for j in order:
                    b=bxs[j]; best=-1;bj=-1
                    for jg,g in enumerate(gt):
                        if jg in matched: continue
                        v=iou(b,g)
                        if v>best: best,bj=v,jg
                    tp=1 if best>=iouth and bj>=0 else 0
                    if tp: matched.add(bj)
                    all_tp[iouth].append(tp)
                    if iouth==0.3: sc.append(float(scs[j])); btp[bd].append(tp); bsc[bd].append(float(scs[j]))
    def ap(tps,scs,ng):
        if not tps: return 0.0
        o=np.argsort(-np.array(scs)); tps=np.array(tps)[o]; ctp=np.cumsum(tps); cfp=np.cumsum(1-tps); rec=ctp/max(1,ng); pre=ctp/np.maximum(1,ctp+cfp)
        mr=np.concatenate([[0],rec,[1]]); mp=np.concatenate([[0],pre,[0]])
        for i in range(len(mp)-1,0,-1): mp[i-1]=max(mp[i-1],mp[i])
        k=np.where(mr[1:]!=mr[:-1])[0]; return float(np.sum((mr[k+1]-mr[k])*mp[k+1]))
    return dict(**{f'AP@{t}':round(ap(all_tp[t],sc,ngt),3) for t in (0.3,0.5)}, AP03_low=round(ap(btp['low'],bsc['low'],bgt['low']),3), AP03_high=round(ap(btp['high'],bsc['high'],bgt['high']),3))
def run(mode):
    student=build_ret(); student.load_state_dict(torch.load(S6/'retinanet_r50_protected.pth')); teacher=copy.deepcopy(student)
    opt=torch.optim.SGD([p for p in student.parameters() if p.requires_grad],lr=LR,momentum=0.9,weight_decay=1e-4)
    dsroot=RES/f'ds_cmp_{mode}'; (dsroot).mkdir(parents=True,exist_ok=True)
    for ep in range(EPOCHS):
        if ep in REFRESH:
            n=teacher_generate(teacher,dsroot,mode,REFRESH.index(ep)); print(f'  [{mode}] ep{ep} (round {REFRESH.index(ep)}) refreshed pseudo-labels: {n}',flush=True)
        train_epoch(student,TR_IMG,dsroot/'labels'/'train',opt); ema(teacher,student,EMA)
    r=evaluate(teacher); return r
def save(runs):
    out={'seeds':SEEDS,'modes':MODES,'tau99':(round(TAU,2) if TAU else None),'C_LO':(round(C_LO,4) if C_LO else None),'C_HI':(round(C_HI,4) if C_HI else None),'b_tta':b_tta,'runs':runs,'agg':{}}
    for m in MODES:
        if runs.get(m):
            ks=list(runs[m][0].keys())
            out['agg'][m]={k:{'mean':round(float(np.mean([r[k] for r in runs[m]])),3),'std':round(float(np.std([r[k] for r in runs[m]])),3),'n':len(runs[m])} for k in ks}
    json.dump(out,open(OUTJSON,'w'),indent=2)
runs={m:[] for m in MODES}
for seed in SEEDS:
    for mode in MODES:
        try:
            torch.manual_seed(seed); np.random.seed(seed)
            print(f'=== seed {seed} mode {mode} ===',flush=True); r=run(mode); runs[mode].append(r)
            print(f'  -> seed {seed} {mode}: {r}',flush=True); save(runs); torch.cuda.empty_cache()
        except Exception as e:
            print(f'!!! seed {seed} mode {mode} FAILED: {e}',flush=True); traceback.print_exc(); torch.cuda.empty_cache(); continue
save(runs)
print('\n=== RetinaNet TS completion (multi + dyn-single + dyn-multi, 3-seed) ==='); print(json.dumps(json.load(open(OUTJSON))['agg'],indent=2)); print('COMPLETE_DONE',flush=True)
