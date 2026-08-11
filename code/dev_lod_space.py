# -*- coding: utf-8 -*-
"""Decision-time SPACE filter for the Mehanian LoD: does aggressively dropping dense (clustered-FP) parasite detections
LOWER the per-µL limit of detection? LoD = Z*std_f/sens, std_f = std of parasitaemia on NEGATIVE samples (FP-driven
noise = the report's 'FP-heterogeneity wall'). Cutting clustered FPs should drop std_f -> drop LoD, if it hurts recall
(sens) less. We detect ONCE (store field-level MP detection coords+conf), then sweep space thresholds cheaply.
Same detector/protocol as dev_lod_mehanian.py. Output -> outputs/experiments/lod_mehanian/space_results.json"""
import os, sys, glob, json, random
import numpy as np, pandas as pd, torch
from PIL import Image; Image.MAX_IMAGE_PIXELS=None
import torchvision.transforms.functional as TF
from torchvision.models.detection import retinanet_resnet50_fpn
sys.path.append('/root/A_Dissertation'); from common.paths import OUTPUTS, RAW
dev='cuda'; random.seed(0); CROP=640; BS=16; TAU_P=0.5; TAU_W=0.3; KMAX=40; KS=[6,15,40]; Z=1.645
NMS_D=21; RSP=200.0; NDMAXES=[10**9, 5, 3, 2]         # 1e9 = unfiltered baseline; then increasingly aggressive
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
def nms_field(dets, md):                                # dets: [(x1,y1,x2,y2,conf)] field coords
    dets=sorted(dets,key=lambda z:-z[4]); k=[]
    for d in dets:
        cx,cy=(d[0]+d[2])/2,(d[1]+d[3])/2
        if all(((cx-(kd[0]+kd[2])/2)**2+(cy-(kd[1]+kd[3])/2)**2)>=md*md for kd in k): k.append(d)
    return k
@torch.no_grad()
def detect_field(det, imgpath, tau):                    # -> [(x1,y1,x2,y2,conf)] field coords, cross-crop NMS
    cr,off=crops_of(imgpath); pts=[]
    for i in range(0,len(cr),BS):
        for p,(tx,ty) in zip(det([c.to(dev) for c in cr[i:i+BS]]),off[i:i+BS]):
            b=p['boxes'].cpu().numpy(); sc=p['scores'].cpu().numpy()
            for k in np.where(sc>=tau)[0]:
                x1,y1,x2,y2=b[k]; pts.append((float(x1+tx),float(y1+ty),float(x2+tx),float(y2+ty),float(sc[k])))
    return nms_field(pts, NMS_D)
def space_keep(dets, ndmax):                            # drop a det with > ndmax neighbours within RSP px
    if ndmax>=10**9 or len(dets)<2: return dets
    C=np.array([[(d[0]+d[2])/2,(d[1]+d[3])/2] for d in dets],float); D=np.sqrt(((C[:,None,:]-C[None,:,:])**2).sum(-1)); nd=(D<=RSP).sum(1)-1
    return [dets[i] for i in range(len(dets)) if nd[i]<=ndmax]
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
# ---- pass 1: MP detections on FASTMAL annotated fields (store once) ----
print('=== detect MP on FASTMAL annotated fields (for recall/precision) ===',flush=True)
ann=fastmal_ann('PARASITE'); ann_det=[(detect_field(pdet,ip,TAU_P),gb) for ip,gb in ann]
# WBC recall/precision (unchanged by space; compute once, no space)
def re_pr_boxes(det_fields):
    TP=FP=NGT=0
    for dets,gb in det_fields:
        bs=sorted(dets,key=lambda x:-x[4]); matched=[False]*len(gb)
        for b in bs:
            best=-1;bj=-1
            for j,g in enumerate(gb):
                if matched[j]: continue
                v=iou(b[:4],g)
                if v>best: best,bj=v,j
            if best>=0.3 and bj>=0: TP+=1; matched[bj]=True
            else: FP+=1
        NGT+=len(gb)
    return TP/max(1,NGT), TP/max(1,TP+FP)
wann=fastmal_ann('WHITE_CELL'); wann_det=[(detect_field(wdet,ip,TAU_W),gb) for ip,gb in wann]
re_wb,pr_wb=re_pr_boxes(wann_det); cf_wb=pr_wb/max(re_wb,1e-6)
# ---- pass 2: MP detections on Ibadan sample fields (store coords once) ----
sp=pd.read_csv(OUTPUTS.parent/'data'/'splits'/'splits.csv'); ib=sp[sp.dataset=='ibadan'].reset_index(drop=True)
ppul=ib.parasites_per_ul.values.astype(float); pcnt=ib.parasite_count.values.astype(float)
print(f'\n=== detect MP over up to {KMAX} fields for {len(ib)} Ibadan samples (once) ===',flush=True)
sample_field_dets=[]  # per sample: list of per-field detection lists
per_wb=[]
@torch.no_grad()
def wbc_count(imgpath):
    cr,off=crops_of(imgpath); wb=0
    for i in range(0,len(cr),BS):
        for p in wdet([c.to(dev) for c in cr[i:i+BS]]): wb+=int((p['scores'].cpu().numpy()>=TAU_W).sum())
    return wb
for k,r in ib.iterrows():
    fields=sorted(glob.glob(str(RAW/r.rel_path)+'/*.tiff')); random.Random(k).shuffle(fields); fields=fields[:KMAX]
    fd=[]; wbs=[]
    for f in fields:
        fd.append(detect_field(pdet,f,TAU_P)); wbs.append(wbc_count(f))
    sample_field_dets.append(fd); per_wb.append(wbs)
    if (k+1)%20==0: print(f'  {k+1}/{len(ib)}',flush=True)
print('detection done; sweeping space thresholds...',flush=True)
# ---- sweep NDMAX: recompute re_mp/pr_mp, per-sample MP counts, LoD ----
pos=pcnt>0; neg=~pos
def acc_wb(K): return np.array([sum(v[:K]) for v in per_wb],float)
out={'RSP':RSP,'KS':KS,'re_wb':round(re_wb,4),'pr_wb':round(pr_wb,4),'n_pos':int(pos.sum()),'n_neg':int(neg.sum()),'by_ndmax':{}}
for ND in NDMAXES:
    re_mp,pr_mp=re_pr_boxes([(space_keep(d,ND),gb) for d,gb in ann_det]); cf_mp=pr_mp/max(re_mp,1e-6)
    per_mp=[[len(space_keep(fd,ND)) for fd in fdlist] for fdlist in sample_field_dets]
    def acc_mp(K): return np.array([sum(v[:K]) for v in per_mp],float)
    ksw={}
    for K in KS:
        MP=acc_mp(K); WB=acc_wb(K); pp=8000.0*(MP*cf_mp)/np.maximum(WB*cf_wb,1e-6)
        f=pp[neg]; mean_f=float(f.mean()); std_f=float(f.std()); thr=mean_f+Z*std_f; LoD=Z*std_f/max(re_mp,1e-6)
        bands=[('<1k',0,1000),('1k-5k',1000,5000),('5k-100k',5000,1e5),('>100k',1e5,1e12)]; bsens={}
        for nm,lo,hi in bands:
            m=pos&(ppul>=lo)&(ppul<hi); bsens[nm]=[int(m.sum()),(round(float((pp[m]>thr).mean()),3) if m.sum() else None)]
        ksw[K]=dict(mean_f=round(mean_f,1),std_f=round(std_f,1),thr=round(thr,1),LoD=round(LoD,1),band_sens=bsens)
    tag='unfiltered' if ND>=10**9 else f'space_nd{ND}'
    out['by_ndmax'][tag]=dict(re_mp=round(re_mp,4),pr_mp=round(pr_mp,4),cf_mp=round(cf_mp,4),K_sweep=ksw)
    print(f'[{tag}] re_mp {re_mp:.3f} pr_mp {pr_mp:.3f} | ' + ' '.join(f'K{K}:LoD {ksw[K]["LoD"]:.0f}(std_f {ksw[K]["std_f"]:.0f})' for K in KS),flush=True)
json.dump(out,open(RES/'space_results.json','w'),indent=2); print('\nWROTE',RES/'space_results.json'); print('LOD_SPACE_DONE',flush=True)
