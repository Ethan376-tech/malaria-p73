# -*- coding: utf-8 -*-
"""T4: transformer detector ablation. Train RT-DETR (transformer detector) on domain-A pseudo-labels (protected + A0),
3 seeds, and compare to MicroYOLO (Table 4.7). Does a transformer detector beat the specialised CNN MicroYOLO on the
noisy pseudo-labels? Custom AP@0.3/0.5 + low/high bands (== Table 4.9). Output -> outputs/experiments/t4_rtdetr/.
Heartbeat -> t4_status.log. Each training wrapped in try/except so one failure does not abort the sweep."""
import os, sys, json, glob, time, traceback
os.environ['XFORMERS_DISABLED']='1'; os.environ['PYTORCH_CUDA_ALLOC_CONF']='expandable_segments:True'
import numpy as np, torch
from PIL import Image; Image.MAX_IMAGE_PIXELS=None
from ultralytics import RTDETR
sys.path.append('/root/A_Dissertation'); from common.paths import OUTPUTS
dev='cuda'; CROP=640; EPOCHS=80
YO=OUTPUTS/'experiments'/'s5_yolo'; RES=OUTPUTS/'experiments'/'t4_rtdetr'; RES.mkdir(parents=True,exist_ok=True)
STAT=OUTPUTS/'experiments'/'t4_status.log'
def hb(m):
    with open(STAT,'a') as f: f.write(f'{int(time.time())} {m}\n')
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
    m=RTDETR(str(best))
    at={0.3:[],0.5:[]}; sc=[]; ng=0; btp={'low':[],'high':[]}; bsc={'low':[],'high':[]}; bg={'low':0,'high':0}
    for idx,ip in enumerate(val_imgs):
        gt=gts[idx]; ng+=len(gt); bd='low' if cropgt[idx]<=med else 'high'; bg[bd]+=len(gt)
        r=m.predict(ip,imgsz=CROP,conf=0.05,verbose=False,device=0)[0]; bxs=r.boxes.xyxy.cpu().numpy(); scs=r.boxes.conf.cpu().numpy(); o=np.argsort(-scs)
        for th in (0.3,0.5):
            mm=set()
            for j in o:
                bst=-1;bj=-1
                for jg,g in enumerate(gt):
                    if jg in mm: continue
                    v=iou(bxs[j],g)
                    if v>bst: bst,bj=v,jg
                tp=1 if bst>=th and bj>=0 else 0
                if tp:mm.add(bj)
                at[th].append(tp)
                if th==0.3: sc.append(float(scs[j])); btp[bd].append(tp); bsc[bd].append(float(scs[j]))
    def ap(t,s,n):
        if not t:return 0.0
        oo=np.argsort(-np.array(s)); t=np.array(t)[oo]; ct=np.cumsum(t);cf=np.cumsum(1-t);rc=ct/max(1,n);pr=ct/np.maximum(1,ct+cf)
        mr=np.r_[0,rc,1];mp=np.r_[0,pr,0]
        for i in range(len(mp)-1,0,-1):mp[i-1]=max(mp[i-1],mp[i])
        k=np.where(mr[1:]!=mr[:-1])[0];return float(np.sum((mr[k+1]-mr[k])*mp[k+1]))
    return dict(**{f'AP@{t}':round(ap(at[t],sc,ng),3) for t in (0.3,0.5)}, AP03_low=round(ap(btp['low'],bsc['low'],bg['low']),3), AP03_high=round(ap(btp['high'],bsc['high'],bg['high']),3))
def run(arm, yaml, seed):
    m=RTDETR('rtdetr-l.yaml')
    m.train(data=yaml, epochs=EPOCHS, imgsz=CROP, batch=8, device=0, workers=8, pretrained=True,
            project=str(RES/'runs'), name=f'{arm}_s{seed}', exist_ok=True, verbose=False, plots=False, seed=seed,
            mosaic=0.0, close_mosaic=0)
    return evaluate(RES/'runs'/f'{arm}_s{seed}'/'weights'/'best.pt')
ARMS=[('protected', str(YO/'data_protected.yaml')), ('a0', str(YO/'data_a0.yaml'))]
hb('T4 START RT-DETR (rtdetr-l, batch8, 80ep)')
runs={a[0]:[] for a in ARMS}
for seed in [0,1,2]:
    for arm,yaml in ARMS:
        try:
            r=run(arm,yaml,seed); runs[arm].append(r); hb(f'{arm} s{seed} -> {r}')
        except Exception as e:
            hb(f'{arm} s{seed} FAILED: {type(e).__name__}: {str(e)[:150]}')
    json.dump(runs,open(RES/'results.json','w'),indent=2)
agg={}
for k in runs:
    if runs[k]:
        ks=list(runs[k][0].keys()); agg[k]={m:{'mean':round(float(np.mean([r[m] for r in runs[k]])),3),'std':round(float(np.std([r[m] for r in runs[k]])),3)} for m in ks}
json.dump(dict(agg=agg,runs=runs),open(RES/'results.json','w'),indent=2)
hb(f'T4_DONE {agg}')
