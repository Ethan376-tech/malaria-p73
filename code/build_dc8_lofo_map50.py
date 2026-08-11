# -*- coding: utf-8 -*-
"""LOFO-clean GT-supervised for Table 4.7 (mAP50 metric). Compute AP@0.5 (all-point, IoU0.5, single class, conf~0.001
to match ultralytics val) with (a) the LEAKED GT-sup (y11 gtsup) as a method check vs report 0.719, and (b) the LOFO
fold weights (each on its held-out film's crops, pooled). Same 640 report test crops. Output -> dc_lofo_gtsup/map50.json"""
import os, sys, json, glob, random
os.environ['XFORMERS_DISABLED']='1'
import numpy as np
from PIL import Image; Image.MAX_IMAGE_PIXELS=None
from ultralytics import YOLO
sys.path.append('/root/A_Dissertation'); from common.paths import OUTPUTS, RAW
CROP=640; CONF=0.001
YO=OUTPUTS/'experiments'/'s5_yolo'; LOFO=OUTPUTS/'experiments'/'dc_lofo_gtsup'
GTSUP=OUTPUTS/'experiments'/'y11_microyolo_table47'/'runs'/'gtsup'/'weights'/'best.pt'
fm=RAW/'ibadan'/'FASTMAL_THICK_FILMS'
def fastmal_fields():
    o=[]
    for samp in sorted(p for p in fm.iterdir() if p.is_dir()):
        js=list(samp.glob('*.json'))
        if not js: continue
        for fld in json.loads(js[0].read_text())['rois']:
            ip=samp/fld['image_name']
            if not ip.exists(): continue
            b=[r for r in fld['roi'] if 'PARASITE' in r.get('type','') and 'CROWD' not in r.get('type','')]
            if b: o.append((samp.name, ip.name))
    return o
fields=sorted(fastmal_fields(),key=lambda z:str(z)); random.Random(0).shuffle(fields); testF=fields[1::2][:40]
k2film={k:testF[k][0] for k in range(len(testF))}
val_imgs=sorted(glob.glob(str(YO/'ds_a0'/'images'/'val'/'*.jpg'))); VLAB=YO/'ds_a0'/'labels'/'val'
def crop_film(ip):
    b=os.path.basename(ip)
    try: k=int(b.split('_')[0][2:])
    except Exception: return None
    return k2film.get(k)
def load_gt(ip):
    lp=VLAB/(os.path.basename(ip)[:-4]+'.txt'); bx=[]
    if lp.exists():
        for ln in lp.read_text().splitlines():
            f=ln.split()
            if len(f)==5: _,cx,cy,w,h=[float(z) for z in f]; cx,cy,w,h=cx*CROP,cy*CROP,w*CROP,h*CROP; bx.append([cx-w/2,cy-h/2,cx+w/2,cy+h/2])
    return bx
gts={ip:load_gt(ip) for ip in val_imgs}; films={ip:crop_film(ip) for ip in val_imgs}
def iou(a,b):
    ix1,iy1=max(a[0],b[0]),max(a[1],b[1]); ix2,iy2=min(a[2],b[2]),min(a[3],b[3]); iw,ih=max(0,ix2-ix1),max(0,iy2-iy1); it=iw*ih
    ua=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-it; return it/ua if ua>0 else 0
def ap50(preds):
    # preds: dict ip->(boxes,scores). all-point AP at IoU0.5, greedy per-image match.
    all_tp=[]; all_sc=[]; ng=0
    for ip in val_imgs:
        gt=gts[ip]; ng+=len(gt); bxs,scs=preds[ip]; o=np.argsort(-scs); m=set()
        for j in o:
            bst=-1;bj=-1
            for jg,g in enumerate(gt):
                if jg in m: continue
                v=iou(bxs[j],g)
                if v>bst: bst,bj=v,jg
            tp=1 if bst>=0.5 and bj>=0 else 0
            if tp:m.add(bj)
            all_tp.append(tp); all_sc.append(float(scs[j]))
    if not all_tp: return 0.0
    o=np.argsort(-np.array(all_sc)); t=np.array(all_tp)[o]; ct=np.cumsum(t);cf=np.cumsum(1-t);rc=ct/max(1,ng);pr=ct/np.maximum(1,ct+cf)
    mr=np.r_[0,rc,1];mp=np.r_[0,pr,0]
    for i in range(len(mp)-1,0,-1):mp[i-1]=max(mp[i-1],mp[i])
    k=np.where(mr[1:]!=mr[:-1])[0]; return round(float(np.sum((mr[k+1]-mr[k])*mp[k+1])),3)
# (a) leaked GT-sup
mg=YOLO(str(GTSUP)); pl={}
for ip in val_imgs:
    r=mg.predict(ip,imgsz=CROP,conf=CONF,verbose=False,device=0)[0]; pl[ip]=(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy())
leaked=ap50(pl); print(f'LEAKED GT-sup AP@0.5(conf{CONF}) = {leaked}  (report Table 4.7 mAP50 = 0.719)',flush=True)
# (b) LOFO: per crop use the fold trained WITHOUT that crop's film
fold_models={}
for f in sorted(set(films.values())):
    if f is None: continue
    w=LOFO/'runs'/f'excl_{f}'/'weights'/'best.pt'
    if w.exists(): fold_models[f]=YOLO(str(w))
pc={}
for ip in val_imgs:
    f=films[ip]; m=fold_models.get(f)
    if m is None: pc[ip]=(np.zeros((0,4)),np.array([])); continue
    r=m.predict(ip,imgsz=CROP,conf=CONF,verbose=False,device=0)[0]; pc[ip]=(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy())
clean=ap50(pc); print(f'LOFO clean GT-sup AP@0.5(conf{CONF}) = {clean}',flush=True)
json.dump(dict(leaked_ap50=leaked, lofo_clean_ap50=clean, report_map50=0.719, conf=CONF), open(LOFO/'map50.json','w'), indent=2)
print(f'\n=== Table 4.7 GT-supervised mAP50: leaked {leaked} (~report 0.719) -> LOFO clean {clean} ===')
print('DC8_DONE',flush=True)
