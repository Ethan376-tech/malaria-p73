# -*- coding: utf-8 -*-
"""LoD by the Mehanian (ICCV 2017) definition + parasitaemia by the Manescu (AJH 2020) recall/precision-corrected
formula, on FASTMAL/Ibadan. Replaces the earlier count-threshold LoD (which was K-invariant by construction).
 - parasitaemia  pp = 8000 * (MP_det * pr_mp/re_mp) / (WBC_det * pr_wbc/re_wbc)          [Manescu, per-µL via WHO 8000 WBC/µL]
 - LoD (per-µL)  = 1.65 * std(f) / sensitivity, f = pp on NEGATIVE samples, sensitivity = MP object recall   [Mehanian Eq 8]
 - K sweep {6,15,40,100 fields}: with the per-µL formula, std(f) falls as ~1/sqrt(K) so LoD FALLS with K if sampling
   is the limit; if LoD stays high even at K=100, the wall is the detector's low-parasitaemia sensitivity (re_mp), not
   sampling. Both detectors reuse s6_retinanet weights. Output -> outputs/experiments/lod_mehanian/results.json"""
import os, sys, glob, json, random
import numpy as np, pandas as pd, torch
from pathlib import Path
from PIL import Image; Image.MAX_IMAGE_PIXELS=None
import torchvision.transforms.functional as TF
from torchvision.models.detection import retinanet_resnet50_fpn
from scipy.stats import pearsonr
sys.path.append('/root/A_Dissertation'); from common.paths import OUTPUTS, RAW
dev='cuda'; random.seed(0); CROP=640; BS=16; TAU_P=0.5; TAU_W=0.3; KMAX=100; KS=[6,15,40,100]; Z=1.645
S6=OUTPUTS/'experiments'/'s6_retinanet'; RES=OUTPUTS/'experiments'/'lod_mehanian'; RES.mkdir(parents=True,exist_ok=True)
def load(name):
    m=retinanet_resnet50_fpn(num_classes=2,weights=None,weights_backbone='IMAGENET1K_V1').to(dev)
    m.load_state_dict(torch.load(S6/name,map_location='cpu')); m.eval(); return m
pdet=load('retinanet_r50_gtsup.pth'); wdet=load('wbc_retinanet.pth')
def crops_of(imgpath):
    im=Image.open(imgpath).convert('RGB'); W,H=im.size
    xs=sorted(set(list(range(0,max(1,W-CROP+1),CROP))+[max(0,W-CROP)])); ys=sorted(set(list(range(0,max(1,H-CROP+1),CROP))+[max(0,H-CROP)]))
    cr=[]; off=[]
    for ty in ys:
        for tx in xs: cr.append(TF.to_tensor(im.crop((tx,ty,tx+CROP,ty+CROP)))); off.append((tx,ty))
    return cr,off
@torch.no_grad()
def detect(det, imgpath, tau):
    cr,off=crops_of(imgpath); bs=[]
    for i in range(0,len(cr),BS):
        for p,(tx,ty) in zip(det([c.to(dev) for c in cr[i:i+BS]]),off[i:i+BS]):
            b=p['boxes'].cpu().numpy(); sc=p['scores'].cpu().numpy()
            for k in np.where(sc>=tau)[0]:
                x1,y1,x2,y2=b[k]; bs.append(((float(x1+tx),float(y1+ty),float(x2+tx),float(y2+ty)),float(sc[k])))
    return bs
@torch.no_grad()
def count_both(imgpath):
    cr,off=crops_of(imgpath); mp=wb=0
    for i in range(0,len(cr),BS):
        batch=[c.to(dev) for c in cr[i:i+BS]]
        for p in pdet(batch): mp+=int((p['scores'].cpu().numpy()>=TAU_P).sum())
        for p in wdet(batch): wb+=int((p['scores'].cpu().numpy()>=TAU_W).sum())
    return mp,wb
def iou(a,b):
    ix1,iy1=max(a[0],b[0]),max(a[1],b[1]); ix2,iy2=min(a[2],b[2]),min(a[3],b[3]); iw,ih=max(0,ix2-ix1),max(0,iy2-iy1); it=iw*ih
    ua=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-it; return it/ua if ua>0 else 0.0
def fastmal_ann(kind):
    fm=RAW/'ibadan'/'FASTMAL_THICK_FILMS'; out=[]
    for samp in sorted(p for p in fm.iterdir() if p.is_dir()):
        js=list(samp.glob('*.json'))
        if not js: continue
        for fld in json.loads(js[0].read_text())['rois']:
            ip=samp/fld['image_name']
            if not ip.exists(): continue
            gb=[(float(r['x']),float(r['y']),float(r['x'])+float(r['width']),float(r['y'])+float(r['height'])) for r in fld['roi'] if kind in r.get('type','') and 'CROWD' not in r.get('type','')]
            if gb: out.append((ip,gb))
    return out
def re_pr(det, fields, tau):
    TP=FP=NGT=0
    for ip,gb in fields:
        bs=sorted(detect(det,ip,tau),key=lambda x:-x[1]); matched=[False]*len(gb)
        for b,_ in bs:
            best=-1; bj=-1
            for j,g in enumerate(gb):
                if matched[j]: continue
                v=iou(b,g)
                if v>best: best,bj=v,j
            if best>=0.3 and bj>=0: TP+=1; matched[bj]=True
            else: FP+=1
        NGT+=len(gb)
    return TP/max(1,NGT), TP/max(1,TP+FP), TP, FP, NGT
print('=== detector recall/precision on FASTMAL annotated fields ===',flush=True)
re_mp,pr_mp,tp_m,fp_m,ng_m=re_pr(pdet, fastmal_ann('PARASITE'), TAU_P)
re_wb,pr_wb,tp_w,fp_w,ng_w=re_pr(wdet, fastmal_ann('WHITE_CELL'), TAU_W)
print(f'MP  recall {re_mp:.3f} precision {pr_mp:.3f} (TP {tp_m} FP {fp_m} GT {ng_m})',flush=True)
print(f'WBC recall {re_wb:.3f} precision {pr_wb:.3f} (TP {tp_w} FP {fp_w} GT {ng_w})',flush=True)
cf_mp=pr_mp/max(re_mp,1e-6); cf_wb=pr_wb/max(re_wb,1e-6)
# ---- per-sample per-field MP+WBC counts over up to KMAX fields ----
sp=pd.read_csv(OUTPUTS.parent/'data'/'splits'/'splits.csv'); ib=sp[sp.dataset=='ibadan'].reset_index(drop=True)
per_mp=[]; per_wb=[]; ppul=ib.parasites_per_ul.values.astype(float); pcnt=ib.parasite_count.values.astype(float)
print(f'\n=== counting MP+WBC over up to {KMAX} fields for {len(ib)} Ibadan samples ===',flush=True)
for k,r in ib.iterrows():
    fields=sorted(glob.glob(str(RAW/r.rel_path)+'/*.tiff')); random.Random(k).shuffle(fields); fields=fields[:KMAX]
    mps=[]; wbs=[]
    for f in fields:
        a,b=count_both(f); mps.append(a); wbs.append(b)
    per_mp.append(mps); per_wb.append(wbs)
    if (k+1)%20==0: print(f'  {k+1}/{len(ib)}',flush=True)
np.save('/root/lod_meh_permp.npy',np.array(per_mp,dtype=object),allow_pickle=True); np.save('/root/lod_meh_perwb.npy',np.array(per_wb,dtype=object),allow_pickle=True)
pos=pcnt>0; neg=~pos
def acc(per,K): return np.array([sum(v[:K]) for v in per],float)
out={'re_mp':round(re_mp,4),'pr_mp':round(pr_mp,4),'re_wb':round(re_wb,4),'pr_wb':round(pr_wb,4),
     'cf_mp':round(cf_mp,4),'cf_wb':round(cf_wb,4),'n_pos':int(pos.sum()),'n_neg':int(neg.sum()),'K_sweep':{}}
print('\n=== Mehanian LoD (Manescu-corrected parasitaemia), K sweep ===',flush=True)
for K in KS:
    MP=acc(per_mp,K); WB=acc(per_wb,K)
    pp=8000.0*(MP*cf_mp)/np.maximum(WB*cf_wb,1e-6)               # Manescu-corrected parasitaemia (MP/µL)
    pp_raw=8000.0*MP/np.maximum(WB,1)                            # uncorrected, for comparison
    f=pp[neg]; mean_f=float(f.mean()); std_f=float(f.std())
    thr=mean_f+Z*std_f; sens=re_mp; LoD=Z*std_f/max(sens,1e-6)   # Mehanian Eq 8 (per-µL)
    # empirical band sensitivity at this 95%-spec threshold
    bands=[('<1k',0,1000),('1k-5k',1000,5000),('5k-100k',5000,1e5),('>100k',1e5,1e12)]
    bsens={}
    for nm,lo,hi in bands:
        m=pos&(ppul>=lo)&(ppul<hi)
        bsens[nm]=[int(m.sum()), (round(float((pp[m]>thr).mean()),3) if m.sum() else None)]
    # corrected vs raw parasitaemia correlation (pos, log10)
    pm=pos&(ppul>0)&(pp>0); r_corr=float(pearsonr(np.log10(pp[pm]),np.log10(ppul[pm]))[0]) if pm.sum()>2 else None
    pmr=pos&(ppul>0)&(pp_raw>0); r_raw=float(pearsonr(np.log10(pp_raw[pmr]),np.log10(ppul[pmr]))[0]) if pmr.sum()>2 else None
    print(f'K={K:3d}: mean_f {mean_f:8.1f}  std_f {std_f:8.1f}  thr {thr:8.1f}  sens(re_mp) {sens:.3f}  LoD {LoD:8.1f} MP/µL | para r corr {r_corr} raw {r_raw}',flush=True)
    for nm,_,_ in bands: print(f'     band {nm:8s} n={bsens[nm][0]:3d} sens@95spec={bsens[nm][1]}',flush=True)
    out['K_sweep'][K]=dict(mean_f=round(mean_f,1),std_f=round(std_f,1),thr=round(thr,1),LoD=round(LoD,1),
                           band_sens=bsens,r_para_corrected=(round(r_corr,3) if r_corr else None),r_para_raw=(round(r_raw,3) if r_raw else None))
json.dump(out,open(RES/'results.json','w'),indent=2); print('\nWROTE',RES/'results.json'); print('LOD_MEH_DONE',flush=True)
