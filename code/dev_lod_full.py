# -*- coding: utf-8 -*-
"""BLOCK C (accurate): recompute per-sample parasite count over K=40 fields (vs block-D's 6) for a more reliable
LoD, esp. at low parasitaemia. All 241 Ibadan samples incl. negatives; gtsup RetinaNet. Then LoD (Mehanian)."""
import os, sys, glob, random
import numpy as np, pandas as pd, torch
from pathlib import Path
from PIL import Image; Image.MAX_IMAGE_PIXELS=None
import torchvision.transforms.functional as TF
from torchvision.models.detection import retinanet_resnet50_fpn
sys.path.append('/root/A_Dissertation'); from common.paths import OUTPUTS, RAW
dev='cuda'; random.seed(0); CROP=640; TAU=0.5; KFIELD=40
S6=OUTPUTS/'experiments'/'s6_retinanet'
meta=pd.read_csv(OUTPUTS/'experiments'/'stageD_bag'/'bag_meta_v2.csv')
IB=meta[meta.dataset=='ibadan'].reset_index(drop=True)
sp=pd.read_csv(OUTPUTS.parent/'data'/'splits'/'splits.csv')[['sample_id','parasites_per_ul','rel_path']]
IB=IB.merge(sp,on='sample_id',how='left')
y=IB.label01.values.astype(int); ppul=IB.parasites_per_ul.values.astype(float)
det=retinanet_resnet50_fpn(num_classes=2,weights=None,weights_backbone='IMAGENET1K_V1').to(dev)
det.load_state_dict(torch.load(S6/'retinanet_r50_gtsup.pth',map_location='cpu')); det.eval()
@torch.no_grad()
def fcount(imgpath):
    im=Image.open(imgpath).convert('RGB'); W,H=im.size; crops=[]
    xs=sorted(set(list(range(0,max(1,W-CROP+1),CROP))+[max(0,W-CROP)])); ys=sorted(set(list(range(0,max(1,H-CROP+1),CROP))+[max(0,H-CROP)]))
    for ty in ys:
        for tx in xs: crops.append(TF.to_tensor(im.crop((tx,ty,tx+CROP,ty+CROP))))
    c=0
    for i in range(0,len(crops),8):
        for p in det([cc.to(dev) for cc in crops[i:i+8]]): c+=int((p['scores'].cpu().numpy()>=TAU).sum())
    return c
cnt=np.zeros(len(IB))
for k,r in IB.iterrows():
    fields=sorted(glob.glob(str(RAW/r.rel_path)+'/*.tiff')); random.Random(k).shuffle(fields)
    cnt[k]=sum(fcount(f) for f in fields[:KFIELD])
    if (k+1)%30==0: print(f'  {k+1}/{len(IB)}',flush=True)
np.save('/root/lod_counts_k40.npy', cnt)
neg=y==0; pos=y==1
print(f'\nIbadan {len(IB)}: neg {neg.sum()} pos {pos.sum()}; K={KFIELD} count neg-mean {cnt[neg].mean():.1f} pos-mean {cnt[pos].mean():.1f}',flush=True)
for SPEC in (0.95,0.90):
    T=np.quantile(cnt[neg],SPEC)
    print(f'\n=== spec {SPEC:.0%} -> T={T:.0f} (actual {float((cnt[neg]<=T).mean()):.2f}) ===')
    lod=None
    for a,b,nm in [(0,100,'<100'),(100,500,'100-500'),(500,1000,'500-1k'),(1000,5000,'1k-5k'),(5000,1e5,'5k-100k'),(1e5,1e12,'>100k')]:
        m=pos&(ppul>=a)&(ppul<b)
        if m.sum()==0: continue
        sens=float((cnt[m]>T).mean()); print(f'  {nm:10s} n={int(m.sum()):3d} sens={sens:.2f}',flush=True)
        if lod is None and sens>=0.5: lod=nm
    print(f'  overall pos sens {float((cnt[pos]>T).mean()):.2f} | LoD(>=50%) {lod}',flush=True)
print('DONE',flush=True)
