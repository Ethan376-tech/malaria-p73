# -*- coding: utf-8 -*-
"""Part-2 (2): sample-level DIAGNOSTIC DECISION from DETECTION COUNT vs bag aggregation, at low parasitaemia.
For each domain-A Ibadan bag sample, run the refined MicroYOLO (fixed pipeline) over its fields, and record
detection-based signals per sample (max detection confidence; #detections at conf thresholds; sum of top-k det confs).
These become a sample-level decision score to compare against the MIL bag decision (report Table 4.12/§4.1: overall
BA 0.651 / AUC 0.68; low-parasitaemia BA 0.58 / AUC ~0.45). Output -> outputs/experiments/dc_detcount/persample.csv"""
import os, sys, json, glob
os.environ['XFORMERS_DISABLED']='1'; os.environ['PYTORCH_CUDA_ALLOC_CONF']='expandable_segments:True'
import numpy as np, pandas as pd
from PIL import Image; Image.MAX_IMAGE_PIXELS=None
from ultralytics import YOLO
sys.path.append('/root/A_Dissertation'); from common.paths import OUTPUTS, IBADAN_PART1, IBADAN_PART2
dev='cuda'; TILE=640; NMS_D=21; CAP=60   # up to 60 fields/sample (median 100); more fields only help low-para
REF=OUTPUTS/'experiments'/'y13_microyolo_ts'/'runs'/'protected_s0_r1'/'weights'/'best.pt'
RES=OUTPUTS/'experiments'/'dc_detcount'; RES.mkdir(parents=True,exist_ok=True)
import random
bm=pd.read_csv(OUTPUTS/'experiments'/'stageD_bag'/'bag_meta_v2.csv'); A=bm[bm.domain=='A_Nigeria'].copy()
ib=pd.read_csv('/root/autodl-tmp/A_Dissertation/data/raw/ibadan/sample_codes_parasite_diagnosis_crosscheck.csv')
sampdir={r.sample_code:(IBADAN_PART1 if 'part1' in r.input_file else IBADAN_PART2)/r.sample_code for _,r in ib.iterrows()}
EXT={'.tif','.tiff','.png','.jpg','.jpeg'}
model=YOLO(str(REF))
def nms_pts(pts, md):
    pts=sorted(pts,key=lambda z:-z[2]); k=[]
    for x,y,s in pts:
        if all((x-kx)**2+(y-ky)**2>=md*md for kx,ky,_ in k): k.append((x,y,s))
    return k
def field_confs(im):
    W,H=im.size; arr=np.asarray(im); xs=sorted(set(list(range(0,max(1,W-TILE+1),TILE))+[max(0,W-TILE)])); ys=sorted(set(list(range(0,max(1,H-TILE+1),TILE))+[max(0,H-TILE)]))
    pts=[]
    for ty in ys:
        for tx in xs:
            sub=arr[ty:ty+TILE,tx:tx+TILE]
            if sub.shape[0]<TILE or sub.shape[1]<TILE:
                pad=np.zeros((TILE,TILE,3),np.uint8); pad[:sub.shape[0],:sub.shape[1]]=sub; sub=pad
            p=model.predict(Image.fromarray(sub), imgsz=TILE, conf=0.05, verbose=False, device=0)[0]
            b=p.boxes.xyxy.cpu().numpy(); s=p.boxes.conf.cpu().numpy()
            for j in range(len(s)): pts.append((tx+(b[j][0]+b[j][2])/2, ty+(b[j][1]+b[j][3])/2, float(s[j])))
    kept=nms_pts(pts, NMS_D); return np.array([s for (_,_,s) in kept]) if kept else np.array([])
rows=[]
for i,(_,r) in enumerate(A.iterrows()):
    d=sampdir.get(r.sample_id)
    if d is None or not d.exists(): continue
    fs=sorted(p for p in d.rglob('*') if p.suffix.lower() in EXT)
    random.Random(0).shuffle(fs); fs=fs[:CAP]
    allc=[]
    for fp in fs:
        try: im=Image.open(fp).convert('RGB')
        except Exception: continue
        c=field_confs(im); allc.append(c)
    allc=np.concatenate(allc) if allc else np.array([])
    sc=np.sort(allc)[::-1]
    rows.append(dict(sample_id=r.sample_id, label=int(r.label01), band=r.para_band, fold=int(r.e1_fold),
        nfields=len(fs), maxconf=float(sc[0]) if len(sc) else 0.0,
        n03=int((allc>=0.3).sum()), n05=int((allc>=0.5).sum()), n07=int((allc>=0.7).sum()),
        top5sum=float(sc[:5].sum()), top1=float(sc[0]) if len(sc) else 0.0, ndet=int(len(allc))))
    if (i+1)%30==0:
        pd.DataFrame(rows).to_csv(RES/'persample.csv',index=False); print(f'{i+1}/{len(A)} done',flush=True)
df=pd.DataFrame(rows); df.to_csv(RES/'persample.csv',index=False)
print(f'DONE persample: {len(df)} samples',flush=True); print('DC1_DONE',flush=True)
