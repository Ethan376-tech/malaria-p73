# -*- coding: utf-8 -*-
"""BLOCK D: sample-level decision — MIL-only vs Detection-only vs MIL+Detection (supervisor request).
In-domain Ibadan (where the detector works). MIL-only = max-pooled bag embedding logistic (in-domain 5-fold OOF).
Detection-only = per-sample parasite count from the GT-supervised RetinaNet over K fields. MIL+Det = logistic on
[mil_oof_score, log1p(count)]. Report sample-level AUC overall and on the low-parasitaemia band — tests whether the
object-level detection signal rescues the bag decision that §4.2 shows is washed out by pooling at low parasitaemia."""
import os, sys, glob, json, random
os.environ['XFORMERS_DISABLED']='1'
sys.path.append('/root/A_Dissertation')
import numpy as np, pandas as pd, torch
from pathlib import Path
from PIL import Image; Image.MAX_IMAGE_PIXELS=None
import torchvision.transforms.functional as TF
from torchvision.models.detection import retinanet_resnet50_fpn
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from common.paths import OUTPUTS, RAW
random.seed(0); np.random.seed(0); dev='cuda'; CROP=640; TAU=0.5; KFIELD=6
D=OUTPUTS/'experiments'/'stageD_bag'; S6=OUTPUTS/'experiments'/'s6_retinanet'
meta=pd.read_csv(D/'bag_meta_v2.csv'); splits=pd.read_csv(OUTPUTS.parent/'data'/'splits'/'splits.csv')
IB=meta[meta.dataset=='ibadan'].reset_index(drop=True)
y=IB.label01.values; band=IB.para_band.values; lo=(band=='neg')|(band=='pos_lo')
print(f'Ibadan samples {len(IB)} pos-rate {y.mean():.3f} | low-para {lo.sum()} pos-rate {y[lo].mean():.3f}',flush=True)

# ---------- MIL-only: max-pool bag embedding, in-domain 5-fold OOF ----------
EMB=np.load(D/'emb_thickdino_v2.npy')                        # (441,256,384) aligned to meta rows
ib_idx=meta.index[meta.dataset=='ibadan'].values
Xbag=EMB[ib_idx].max(1)                                      # max-pool -> (nIb,384)
def oof_scores(X,yv,reps=1):
    sc=np.zeros(len(yv))
    skf=StratifiedKFold(5,shuffle=True,random_state=0)
    for tr,va in skf.split(X,yv):
        s=StandardScaler().fit(X[tr]); clf=LogisticRegression(max_iter=3000,C=0.3,class_weight='balanced').fit(s.transform(X[tr]),yv[tr])
        sc[va]=clf.predict_proba(s.transform(X[va]))[:,1]
    return sc
mil=oof_scores(Xbag,y)
print(f'MIL-only done (AUC {roc_auc_score(y,mil):.3f})',flush=True)

# ---------- Detection-only: per-sample parasite count over K fields ----------
det=retinanet_resnet50_fpn(num_classes=2,weights=None,weights_backbone='IMAGENET1K_V1').to(dev)
det.load_state_dict(torch.load(S6/'retinanet_r50_gtsup.pth',map_location='cpu')); det.eval()
@torch.no_grad()
def field_count(imgpath):
    im=Image.open(imgpath).convert('RGB'); W,H=im.size; crops=[]; offs=[]
    xs=list(range(0,max(1,W-CROP+1),CROP))+[max(0,W-CROP)]; ys=list(range(0,max(1,H-CROP+1),CROP))+[max(0,H-CROP)]
    for ty in sorted(set(ys)):
        for tx in sorted(set(xs)):
            crops.append(TF.to_tensor(im.crop((tx,ty,tx+CROP,ty+CROP)))); offs.append((tx,ty))
    cnt=0
    for i in range(0,len(crops),8):
        preds=det([c.to(dev) for c in crops[i:i+8]])
        for p in preds: cnt+=int((p['scores'].cpu().numpy()>=TAU).sum())
    return cnt
counts=np.zeros(len(IB))
for i,r in IB.iterrows():
    rp=splits[splits.sample_id==r.sample_id].rel_path.values
    if not len(rp): continue
    fdir=RAW/rp[0]; fields=sorted(glob.glob(str(fdir)+'/*.tiff'))
    if not fields: continue
    random.Random(i).shuffle(fields)
    counts[i]=sum(field_count(f) for f in fields[:KFIELD])
    if (i+1)%40==0: print(f'  det {i+1}/{len(IB)}',flush=True)
np.save('/root/block_d_counts.npy', counts)
detsc=np.log1p(counts)
print(f'Detection-only done (AUC {roc_auc_score(y,detsc):.3f}); count pos-mean {counts[y==1].mean():.1f} neg-mean {counts[y==0].mean():.1f}',flush=True)

# ---------- MIL+Detection: logistic on [mil_oof, log1p(count)] ----------
F=np.column_stack([mil, detsc]); comb=oof_scores(F,y)

def auc(s,mask=None):
    yy=y if mask is None else y[mask]; ss=s if mask is None else s[mask]
    return roc_auc_score(yy,ss) if len(set(yy))>1 else float('nan')
print('\n=== SAMPLE-LEVEL AUC (in-domain Ibadan) ===')
print(f'{"decision":16s} {"overall":>8s} {"low-para":>9s}')
for nm,s in [('MIL-only',mil),('Detection-only',detsc),('MIL+Detection',comb)]:
    print(f'{nm:16s} {auc(s):8.3f} {auc(s,lo):9.3f}',flush=True)
json.dump({'overall':{k:round(float(auc(v)),3) for k,v in [('mil',mil),('det',detsc),('comb',comb)]},
           'lowpara':{k:round(float(auc(v,lo)),3) for k,v in [('mil',mil),('det',detsc),('comb',comb)]}},
          open('/root/block_d_results.json','w'),indent=2)
print('\nDONE',flush=True)
