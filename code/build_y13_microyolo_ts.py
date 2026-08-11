# -*- coding: utf-8 -*-
"""Phase 2A switch-gate (teacher-student): YOLOv8n teacher-student refinement, YOLO analog of Table 4.9. Because
ultralytics wraps training, we use ROUND-BASED teacher-student (iterated EMA self-training): each round the current
teacher (ultralytics keeps an internal EMA -> best.pt) regenerates pseudo-labels, they are retained, and a student is
fine-tuned; the student becomes next round's teacher. 2 refresh rounds (matches build_s9 REFRESH=[0,5]). Retain: the
same external ThickDINO-MC faintness-protected(single) vs naive-matched (global conf threshold, retention-matched;
MILCA conf>0.7 collapses on YOLO). Custom AP, comparable to Table 4.9. 3 seeds. Output -> y13_microyolo_ts/results.json"""
import os, sys, json, glob, shutil
os.environ['XFORMERS_DISABLED']='1'; os.environ['PYTORCH_CUDA_ALLOC_CONF']='expandable_segments:True'
import numpy as np, torch, torch.nn as nn
import torchvision.transforms.functional as TF, torchvision.transforms as T
from PIL import Image; Image.MAX_IMAGE_PIXELS=None
from ultralytics import YOLO
sys.path.insert(0,'/root/autodl-tmp/A_Dissertation/references/RedDino/train'); sys.path.append('/root/A_Dissertation')
from common.paths import OUTPUTS, CHECKPOINTS
from dinov2.models.vision_transformer import vit_small
dev='cuda'; CROP=640; TILE=224; T_MC=15; MC_CAP=40; ROUNDS=2; EP=10; LR0=0.002
YO=OUTPUTS/'experiments'/'s5_yolo'; RES=OUTPUTS/'experiments'/'y13_microyolo_ts'; RES.mkdir(parents=True,exist_ok=True)
FIN=json.load(open(OUTPUTS/'experiments'/'s3_filter'/'final.json'))['calib']; b_protect=FIN['b_protect']; a_star=FIN['a_star']
norm=T.Compose([T.ToTensor(),T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
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
def yolo_predict(model, ip, conf=0.05):
    r=model.predict(ip, imgsz=CROP, conf=conf, verbose=False, device=0)[0]
    return r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()
def yolo_line(b): cx=(b[0]+b[2])/2/CROP; cy=(b[1]+b[3])/2/CROP; w=(b[2]-b[0])/CROP; h=(b[3]-b[1])/CROP; return f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"
train_imgs=sorted(glob.glob(str(YO/'ds_a0'/'images'/'train'/'*.jpg')))
def _cap(boxes,scores):
    if len(boxes)==0: return boxes,scores
    order=np.argsort(-scores)[:MC_CAP]; return boxes[order],scores[order]
def regen(model, labdir, mode, N=None):
    labdir.mkdir(parents=True,exist_ok=True)
    if mode=='naive':
        cache={}; alls=[]
        for ip in train_imgs:
            b,s=_cap(*yolo_predict(model,ip)); cache[ip]=(b,s); alls.extend(s.tolist())
        alls=np.sort(np.array(alls))[::-1]; thr=float(alls[min(N,len(alls))-1]) if (N and len(alls)) else 1.1; nlab=0
        for ip,(b,s) in cache.items():
            base=os.path.basename(ip)[:-4]; sel=[j for j in range(len(b)) if s[j]>=thr]
            (labdir/(base+'.txt')).write_text("\n".join(yolo_line(b[j]) for j in sel)); nlab+=len(sel)
        return nlab
    nlab=0
    for ip in train_imgs:
        arr=np.asarray(Image.open(ip).convert('RGB')); b,s=_cap(*yolo_predict(model,ip)); base=os.path.basename(ip)[:-4]
        if len(b)==0: (labdir/(base+'.txt')).write_text(""); continue
        tiles=[]
        for bb in b:
            cx=(bb[0]+bb[2])/2; cy=(bb[1]+bb[3])/2; x0=int(min(max(cx-TILE/2,0),CROP-TILE)); y0=int(min(max(cy-TILE/2,0),CROP-TILE)); tiles.append(arr[y0:y0+TILE,x0:x0+TILE])
        pm,pv=mc_boxes(tiles); sel=[j for j in range(len(b)) if (pv[j]>=b_protect or pm[j]>=a_star)]
        (labdir/(base+'.txt')).write_text("\n".join(yolo_line(b[j]) for j in sel)); nlab+=len(sel)
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
    _populate(YO/'ds_a0'/'images'/'train', root/'images'/'train', 'jpg')
    _populate(YO/'ds_a0'/'images'/'val',   root/'images'/'val',   'jpg')
    _populate(labdir,                       root/'labels'/'train', 'txt', copy=True)
    _populate(YO/'ds_a0'/'labels'/'val',   root/'labels'/'val',   'txt', copy=True)
    for c in glob.glob(str(root)+'/labels/*.cache'): os.remove(c)
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
    return dict(**{f'AP@{t}':round(ap(at[t],sc,ngt),3) for t in (0.3,0.5)}, AP03_low=round(ap(btp['low'],bsc['low'],bgt['low']),3), AP03_high=round(ap(btp['high'],bsc['high'],bgt['high']),3))
def train_yolo(init_wts, yaml, name, seed, epochs, lr0):
    if init_wts=='yolov8n.pt':
        m=YOLO('yolov8n-p2.yaml')
        try: m=m.load('yolov8n.pt')
        except Exception: pass
    else: m=YOLO(init_wts)
    m.train(data=yaml, epochs=epochs, imgsz=CROP, batch=16, box=9.0, device=0, workers=8, lr0=lr0, warmup_epochs=1,
        project=str(RES/'runs'), name=name, exist_ok=True, verbose=False, plots=False, seed=seed)
    return str(RES/'runs'/name/'weights'/'best.pt')
# ---- fixed base D1 (trained once on A0, matches build_s9 fixed base) ----
print('=== train base D1 (YOLOv8n on A0, once) ===',flush=True)
D1=train_yolo('yolov8n.pt', make_ds('ds_base', YO/'ds_a0'/'labels'/'train'),'d1_base',0,80,0.005)
res={'D1_base':evaluate(YOLO(D1))}; print('  D1_base',res['D1_base'],flush=True)
def ts_run(mode, seed):
    torch.manual_seed(seed); np.random.seed(seed); teacher_wts=D1; N=None
    for r in range(ROUNDS):
        tm=YOLO(teacher_wts)
        if mode=='protected': n=regen(tm, RES/f'lab_{mode}', 'protected')
        else:
            if N is None:  # match naive retention to protected's round-0 count (computed once from D1)
                N=regen(YOLO(D1), RES/'lab_protcount','protected')
            n=regen(tm, RES/f'lab_{mode}', 'naive', N=N)
        yaml=make_ds(f'ds_{mode}', RES/f'lab_{mode}')
        teacher_wts=train_yolo(teacher_wts, yaml, f'{mode}_s{seed}_r{r}', seed, EP, LR0)
        print(f'    [{mode} s{seed} r{r}] labels={n}',flush=True)
    return evaluate(YOLO(teacher_wts))
runs={'ts_naive':[], 'ts_protected':[]}
for seed in [0,1,2]:
    for mode in ['naive','protected']:
        print(f'=== seed {seed} teacher-student {mode} ===',flush=True)
        r=ts_run(mode,seed); runs[f'ts_{mode}'].append(r); print(f'  ts_{mode} s{seed} -> {r}',flush=True)
    json.dump(runs,open(RES/'results.json','w'),indent=2)
agg={}
for k in runs:
    ks=[x for x in runs[k][0].keys() if x.startswith('AP')]
    agg[k]={m:{'mean':round(float(np.mean([r[m] for r in runs[k]])),3),'std':round(float(np.std([r[m] for r in runs[k]])),3)} for m in ks}
json.dump(dict(agg=agg,runs=runs,D1_base=res['D1_base'],
    ref={'yolo_oneshot_bootstrap_protected_AP@0.5':0.358,'retinanet_TS_Table4.9_note':'RetinaNet TS ours best AP@0.3 ~0.374'}),
    open(RES/'results.json','w'),indent=2)
print('\n=== MicroYOLO teacher-student 3-seed (mean±std, custom AP) ===')
print('  D1_base',res['D1_base'])
for k in runs: print(f'  {k:14s} {agg[k]}')
print('Y13_DONE',flush=True)
