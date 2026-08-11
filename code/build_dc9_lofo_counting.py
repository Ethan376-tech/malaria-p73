# -*- coding: utf-8 -*-
"""LOFO-clean GT-supervised for Table 4.15 (counting). For each report test field (film f), count with the fold model
trained WITHOUT film f; tau tuned on the other films' fields (GT counts). Aggregate predicted vs GT counts over the test
fields -> Pearson r / R2 / MAE + low-density r. Compare to the leaked GT-sup counting (report: r 0.85, low-density 0.80).
Output -> dc_lofo_gtsup/counting.json"""
import os, sys, json, glob, random
os.environ['XFORMERS_DISABLED']='1'
import numpy as np
from PIL import Image; Image.MAX_IMAGE_PIXELS=None
from ultralytics import YOLO
from scipy.stats import pearsonr
sys.path.append('/root/A_Dissertation'); from common.paths import OUTPUTS, RAW
TILE=640; NMS_D=21; PRED_CONF=0.01
LOFO=OUTPUTS/'experiments'/'dc_lofo_gtsup'
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
            b=[(float(r['x']),float(r['y']),float(r['width']),float(r['height'])) for r in fld['roi'] if 'PARASITE' in r.get('type','') and 'CROWD' not in r.get('type','')]
            if b: o.append((samp.name, ip, len(b)))
    return o
allf=fastmal_fields()
keyed=sorted(allf, key=lambda z:(z[0], z[1].name)); random.Random(0).shuffle(keyed)
testF=keyed[1::2][:40]  # (film, path, gtcount)
def nms_pts(pts, md):
    pts=sorted(pts,key=lambda z:-z[2]); k=[]
    for x,y,s in pts:
        if all((x-kx)**2+(y-ky)**2>=md*md for kx,ky,_ in k): k.append((x,y,s))
    return k
def field_dets(model, ip):
    im=Image.open(ip).convert('RGB'); W,H=im.size; arr=np.asarray(im)
    xs=sorted(set(list(range(0,max(1,W-TILE+1),TILE))+[max(0,W-TILE)])); ys=sorted(set(list(range(0,max(1,H-TILE+1),TILE))+[max(0,H-TILE)]))
    pts=[]
    for ty in ys:
        for tx in xs:
            sub=arr[ty:ty+TILE,tx:tx+TILE]
            if sub.shape[0]<TILE or sub.shape[1]<TILE:
                pad=np.zeros((TILE,TILE,3),np.uint8); pad[:sub.shape[0],:sub.shape[1]]=sub; sub=pad
            p=model.predict(Image.fromarray(sub),imgsz=TILE,conf=PRED_CONF,verbose=False,device=0)[0]
            b=p.boxes.xyxy.cpu().numpy(); s=p.boxes.conf.cpu().numpy()
            for j in range(len(s)): pts.append((tx+(b[j][0]+b[j][2])/2, ty+(b[j][1]+b[j][3])/2, float(s[j])))
    return nms_pts(pts, NMS_D)
def counts_at(detsets, tau): return np.array([sum(1 for (_,_,s) in d if s>=tau) for d in detsets])
def metrics(pred,gt):
    pred=np.asarray(pred,float); gt=np.asarray(gt,float)
    ss=np.sum((gt-pred)**2); st=np.sum((gt-gt.mean())**2); r2=1-ss/st if st>0 else float('nan')
    r=pearsonr(pred,gt)[0] if np.std(pred)>0 and np.std(gt)>0 else float('nan'); mae=np.mean(np.abs(pred-gt))
    return dict(R2=round(float(r2),3), pearson=round(float(r),3), MAE=round(float(mae),2))
# fold models
films_test=sorted(set(f for f,_,_ in testF))
fold_model={f:YOLO(str(LOFO/'runs'/f'excl_{f}'/'weights'/'best.pt')) for f in films_test if (LOFO/'runs'/f'excl_{f}'/'weights'/'best.pt').exists()}
_testkeys=set((a,b.name) for a,b,_ in testF)
devfields=[(f,ip,gc) for (f,ip,gc) in keyed if (f,ip.name) not in _testkeys]
def eval_counting(get_model, tau_model):
    # dets for each test field (per-field model policy)
    dets=[]; gts=[]
    for (film,ip,gc) in testF:
        dets.append(field_dets(get_model(film),ip)); gts.append(gc)
    gts=np.array(gts)
    # tau tuned on dev fields, all scored by tau_model (single, available for both policies) via MAE
    ddets=[field_dets(tau_model,ip) for (f,ip,gc) in devfields]; dgt=np.array([gc for _,_,gc in devfields])
    best=(1e9,0.1)
    for t in np.linspace(0.02,0.9,45):
        mae=np.mean(np.abs(counts_at(ddets,t)-dgt))
        if mae<best[0]: best=(mae,float(t))
    tau=best[1]
    pred=counts_at(dets,tau); med=np.median(gts); low=gts<=med; high=gts>med
    return dict(tau=round(tau,3), overall=metrics(pred,gts), low_density=metrics(pred[low],gts[low]),
                high_density=metrics(pred[high],gts[high]))
mg=YOLO(str(GTSUP))  # leaked single model, also used to tune tau on dev for both
# LOFO GT-sup counting (per test field -> fold model that excluded its film)
lofo=eval_counting(lambda film: fold_model[film], mg); print('LOFO GT-sup counting:',lofo,flush=True)
# leaked GT-sup counting (single model) for method-check vs report r0.85
leaked=eval_counting(lambda film: mg, mg); print('LEAKED GT-sup counting:',leaked,flush=True)
json.dump(dict(lofo=lofo, leaked=leaked, report_ref={'r':0.85,'low_r':0.80,'R2':0.65,'MAE':3.80}), open(LOFO/'counting.json','w'), indent=2)
print('DC9_DONE',flush=True)
