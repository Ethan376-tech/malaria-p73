# -*- coding: utf-8 -*-
"""BLOCK B step 2: parasitaemia = 8000 * (MP_detected / WBC_detected) on FASTMAL (Manescu 2020, AJH).
parasite detector = in-domain gtsup RetinaNet; WBC detector = Lacuna-trained (cross-site). Per Ibadan sample,
count MP + WBC over K fields, form ratio*8000. Validate against manual (splits.csv): (1) WBC_det vs manual WBC,
(2) MP_det vs manual parasite count, (3) predicted vs manual parasitaemia (MP/uL). Correlation R2 + Pearson."""
import os, sys, glob, random, json
import numpy as np, pandas as pd, torch
from pathlib import Path
from PIL import Image; Image.MAX_IMAGE_PIXELS=None
import torchvision.transforms.functional as TF
from torchvision.models.detection import retinanet_resnet50_fpn
from scipy.stats import pearsonr, spearmanr
sys.path.append('/root/A_Dissertation'); from common.paths import OUTPUTS, RAW
dev='cuda'; random.seed(0); CROP=640; KFIELD=15; TAU_P=0.5; TAU_W=0.3
S6=OUTPUTS/'experiments'/'s6_retinanet'
sp=pd.read_csv(OUTPUTS.parent/'data'/'splits'/'splits.csv')
ib=sp[(sp.dataset=='ibadan') & sp.parasite_count.notna() & sp.wbc_count.notna() & (sp.wbc_count>0)].copy()
ib['man_para']=8000.0*ib.parasite_count/ib.wbc_count                          # manual parasitaemia MP/uL
print(f'FASTMAL samples with manual MP+WBC: {len(ib)}  (pos {int((ib.parasite_count>0).sum())})',flush=True)

def load(name):
    m=retinanet_resnet50_fpn(num_classes=2,weights=None,weights_backbone='IMAGENET1K_V1').to(dev)
    m.load_state_dict(torch.load(S6/name,map_location='cpu')); m.eval(); return m
pdet=load('retinanet_r50_gtsup.pth'); wdet=load('wbc_retinanet.pth')
@torch.no_grad()
def counts(imgpath):
    im=Image.open(imgpath).convert('RGB'); W,H=im.size; crops=[]
    xs=sorted(set(list(range(0,max(1,W-CROP+1),CROP))+[max(0,W-CROP)])); ys=sorted(set(list(range(0,max(1,H-CROP+1),CROP))+[max(0,H-CROP)]))
    for ty in ys:
        for tx in xs: crops.append(TF.to_tensor(im.crop((tx,ty,tx+CROP,ty+CROP))))
    mp=wb=0
    for i in range(0,len(crops),8):
        batch=[c.to(dev) for c in crops[i:i+8]]
        for p in pdet(batch): mp+=int((p['scores'].cpu().numpy()>=TAU_P).sum())
        for p in wdet(batch): wb+=int((p['scores'].cpu().numpy()>=TAU_W).sum())
    return mp,wb
MP=np.zeros(len(ib)); WB=np.zeros(len(ib))
for k,(_,r) in enumerate(ib.iterrows()):
    fdir=RAW/r.rel_path; fields=sorted(glob.glob(str(fdir)+'/*.tiff')); random.Random(k).shuffle(fields)
    tm=tw=0
    for f in fields[:KFIELD]:
        a,b=counts(f); tm+=a; tw+=b
    MP[k]=tm; WB[k]=tw
    if (k+1)%20==0: print(f'  {k+1}/{len(ib)}',flush=True)
ib['MP_det']=MP; ib['WB_det']=WB; ib['pred_para']=8000.0*MP/np.maximum(WB,1)
np.save('/root/parasitemia_arrays.npy', ib[['parasite_count','wbc_count','man_para','MP_det','WB_det','pred_para']].values)

def rep(a,b,tag,logx=False):
    m=(np.asarray(a)>0)|(np.asarray(b)>0) if not logx else (np.asarray(a)>0)&(np.asarray(b)>0)
    x=np.asarray(a)[m].astype(float); y=np.asarray(b)[m].astype(float)
    if logx: x,y=np.log10(x),np.log10(y)
    r=pearsonr(x,y)[0] if len(x)>2 else float('nan'); rs=spearmanr(x,y)[0] if len(x)>2 else float('nan')
    ss=np.sum((y-x)**2); st=np.sum((y-y.mean())**2); r2=1-ss/st if st>0 else float('nan')
    print(f'{tag:34s} n={len(x):3d} Pearson r={r:.3f} Spearman={rs:.3f}',flush=True); return r,rs
print('\n=== validation (predicted vs manual) ===')
rep(ib.WB_det, ib.wbc_count, 'WBC count (det vs manual)')
rep(ib.MP_det, ib.parasite_count, 'parasite count (det vs manual)')
posm=ib.parasite_count>0
rep(ib.pred_para[posm], ib.man_para[posm], 'parasitaemia MP/uL (pos, log10)', logx=True)
print('\nmanual para median/range:', round(float(ib.man_para[posm].median()),0), [round(float(ib.man_para[posm].min()),0),round(float(ib.man_para[posm].max()),0)],flush=True)
print('DONE',flush=True)
