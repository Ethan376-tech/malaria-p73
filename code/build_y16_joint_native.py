# -*- coding: utf-8 -*-
"""Joint MIL-detector on MicroYOLO — OPTION 3 (proper): subclass DetectionModel (bag head + joint loss) + DetectionTrainer,
train NATIVELY (full ultralytics machinery: aug/EMA/optimizer; mosaic OFF since sample-level bag labels ⊥ mosaic).
From-scratch from yolov8n.pt (matches the RetinaNet joint's from-scratch setup). Three arms x 3 seeds: sequential
baseline / +negatives-no-bag / joint(+bag). Custom AP@0.3/0.5 + low/high-para bands (== Table 4.11). Output ->
outputs/experiments/y16_joint_native/."""
import os, sys, json, glob, shutil
os.environ['XFORMERS_DISABLED']='1'; os.environ['PYTORCH_CUDA_ALLOC_CONF']='expandable_segments:True'
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from PIL import Image; Image.MAX_IMAGE_PIXELS=None
from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.tasks import DetectionModel
sys.path.append('/root/A_Dissertation'); from common.paths import OUTPUTS
dev='cuda'; CROP=640; LAM=8.0; EPOCHS=80
YO=OUTPUTS/'experiments'/'s5_yolo'; RES=OUTPUTS/'experiments'/'y16_joint_native'; RES.mkdir(parents=True,exist_ok=True)
_BAGFEAT={}
def _sppf_hook(module, inp, out): _BAGFEAT['f']=out
class JointDetectionModel(DetectionModel):
    def __init__(self, cfg='yolov8n-p2.yaml', ch=3, nc=None, verbose=True):
        super().__init__(cfg, ch, nc, verbose)
        self._sppf=[i for i,m in enumerate(self.model) if type(m).__name__=='SPPF'][-1]
        self.bag_head=nn.Sequential(nn.AdaptiveAvgPool2d(1),nn.Flatten(),nn.Linear(256,1))
        self.model[self._sppf].register_forward_hook(_sppf_hook)
    def loss(self, batch, preds=None):
        out=super().loss(batch, preds); L,items=out if isinstance(out,tuple) else (out,None)
        feat=_BAGFEAT.get('f')
        if feat is not None and self.training:
            blog=self.bag_head(feat).squeeze(-1); B=batch['img'].shape[0]
            has=torch.zeros(B,device=blog.device); bi=batch['batch_idx'].long()
            if len(bi): has[bi.unique()]=1.0
            L=L+LAM*F.binary_cross_entropy_with_logits(blog,has)
        return (L,items) if items is not None else L
class JointTrainer(DetectionTrainer):
    def get_model(self, cfg=None, weights=None, verbose=True):
        m=JointDetectionModel(cfg or 'yolov8n-p2.yaml', nc=self.data['nc'], verbose=verbose)
        if weights: m.load(weights)
        return m
POS=sorted(glob.glob(str(YO/'ds_a0'/'images'/'train'/'*.jpg'))); POS_LAB=YO/'ds_protected'/'labels'/'train'
NEG=sorted(glob.glob(str(OUTPUTS/'experiments'/'s12_joint'/'neg_crops'/'*.jpg')))
def _link(s,d):
    try: os.link(s,d)
    except OSError: shutil.copy(s,d)
def make_ds(name, with_neg):
    root=RES/name
    for sub in ['images/train','images/val','labels/train','labels/val']:
        d=root/sub
        if d.exists(): shutil.rmtree(d)
        d.mkdir(parents=True,exist_ok=True)
    for p in POS:
        b=os.path.basename(p); _link(p,str(root/'images/train'/b)); lp=POS_LAB/(b[:-4]+'.txt')
        shutil.copy(str(lp),str(root/'labels/train'/(b[:-4]+'.txt'))) if lp.exists() else open(str(root/'labels/train'/(b[:-4]+'.txt')),'w').write('')
    if with_neg:
        for p in NEG:
            b=os.path.basename(p); _link(p,str(root/'images/train'/b)); open(str(root/'labels/train'/(b[:-4]+'.txt')),'w').write('')
    for p in glob.glob(str(YO/'ds_gtsup'/'images/val'/'*.jpg')):
        b=os.path.basename(p); _link(p,str(root/'images/val'/b)); shutil.copy(str(YO/'ds_gtsup'/'labels/val'/(b[:-4]+'.txt')),str(root/'labels/val'/(b[:-4]+'.txt')))
    yaml=str(root/'data.yaml'); open(yaml,'w').write(f"path: {root}\ntrain: images/train\nval: images/val\nnc: 1\nnames: ['parasite']\n"); return yaml
YAML_POS=make_ds('ds_pos', False); YAML_POSNEG=make_ds('ds_posneg', True)
AUG=dict(mosaic=0.0, close_mosaic=0, mixup=0.0, copy_paste=0.0)
# ---- eval ----
val_imgs=sorted(glob.glob(str(YO/'ds_gtsup'/'images'/'val'/'*.jpg'))); VLAB=YO/'ds_gtsup'/'labels'/'val'
def load_gt(ip):
    lp=VLAB/(os.path.basename(ip)[:-4]+'.txt'); bx=[]
    if lp.exists():
        for ln in lp.read_text().splitlines():
            f=ln.split()
            if len(f)==5: _,cx,cy,w,h=[float(z) for z in f]; cx,cy,w,h=cx*CROP,cy*CROP,w*CROP,h*CROP; bx.append([cx-w/2,cy-h/2,cx+w/2,cy+h/2])
    return bx
gts=[load_gt(ip) for ip in val_imgs]; cropgt=[len(g) for g in gts]; med=np.median([c for c in cropgt if c>0]) if any(cropgt) else 0
def iou(a,b):
    ix1,iy1=max(a[0],b[0]),max(a[1],b[1]); ix2,iy2=min(a[2],b[2]),min(a[3],b[3]); iw,ih=max(0,ix2-ix1),max(0,iy2-iy1); it=iw*ih
    ua=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-it; return it/ua if ua>0 else 0
def evaluate(best):
    ck=torch.load(str(best), weights_only=False); ye=YOLO('yolov8n-p2.yaml'); ye.model=ck['model'].float().to(dev).eval()
    at={0.3:[],0.5:[]}; sc=[]; ng=0; btp={'low':[],'high':[]}; bsc={'low':[],'high':[]}; bg={'low':0,'high':0}
    for idx,ip in enumerate(val_imgs):
        gt=gts[idx]; ng+=len(gt); bd='low' if cropgt[idx]<=med else 'high'; bg[bd]+=len(gt)
        r=ye.predict(ip,imgsz=CROP,conf=0.05,verbose=False,device=0)[0]; bxs=r.boxes.xyxy.cpu().numpy(); scs=r.boxes.conf.cpu().numpy(); o=np.argsort(-scs)
        for th in (0.3,0.5):
            m=set()
            for j in o:
                bst=-1;bj=-1
                for jg,g in enumerate(gt):
                    if jg in m: continue
                    v=iou(bxs[j],g)
                    if v>bst: bst,bj=v,jg
                tp=1 if bst>=th and bj>=0 else 0
                if tp:m.add(bj)
                at[th].append(tp)
                if th==0.3: sc.append(float(scs[j])); btp[bd].append(tp); bsc[bd].append(float(scs[j]))
    def ap(t,s,n):
        if not t:return 0
        oo=np.argsort(-np.array(s)); t=np.array(t)[oo]; ct=np.cumsum(t);cf=np.cumsum(1-t);rc=ct/max(1,n);pr=ct/np.maximum(1,ct+cf)
        mr=np.r_[0,rc,1];mp=np.r_[0,pr,0]
        for i in range(len(mp)-1,0,-1):mp[i-1]=max(mp[i-1],mp[i])
        k=np.where(mr[1:]!=mr[:-1])[0];return float(np.sum((mr[k+1]-mr[k])*mp[k+1]))
    return dict(**{f'AP@{t}':round(ap(at[t],sc,ng),3) for t in (0.3,0.5)}, AP03_low=round(ap(btp['low'],bsc['low'],bg['low']),3), AP03_high=round(ap(btp['high'],bsc['high'],bg['high']),3))
def run_arm(name, yaml, joint, seed):
    y=YOLO('yolov8n-p2.yaml')
    kw=dict(data=yaml, epochs=EPOCHS, imgsz=CROP, batch=16, device=0, workers=8, box=9.0, pretrained='yolov8n.pt',
            project=str(RES/'runs'), name=f'{name}_s{seed}', exist_ok=True, verbose=False, plots=False, seed=seed, **AUG)
    if joint: y.train(trainer=JointTrainer, **kw)
    else: y.train(**kw)
    return evaluate(RES/'runs'/f'{name}_s{seed}'/'weights'/'best.pt')
ARMS=[('baseline_seq', YAML_POS, False),('neg_no_bag', YAML_POSNEG, False),('joint_mil', YAML_POSNEG, True)]
runs={a[0]:[] for a in ARMS}
for seed in [0,1,2]:
    for name,yaml,joint in ARMS:
        print(f'=== seed {seed} {name} (joint={joint}) ===',flush=True); r=run_arm(name,yaml,joint,seed); runs[name].append(r); print(f'  {name} s{seed} -> {r}',flush=True)
    json.dump(runs,open(RES/'results.json','w'),indent=2)
agg={}
for k in runs:
    ks=list(runs[k][0].keys()); agg[k]={m:{'mean':round(float(np.mean([r[m] for r in runs[k]])),3),'std':round(float(np.std([r[m] for r in runs[k]])),3)} for m in ks}
json.dump(dict(agg=agg,runs=runs,lam=LAM,note='native (option 3), from-scratch, mosaic-off'),open(RES/'results.json','w'),indent=2)
print('\n=== MicroYOLO joint MIL-detector (option 3 native, 3-seed mean±std) ===')
for k in ['baseline_seq','neg_no_bag','joint_mil']: print(f'  {k:14s} {agg[k]}')
print('Y16_DONE',flush=True)
