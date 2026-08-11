# -*- coding: utf-8 -*-
"""Propagate MicroYOLO to the parasite COUNT estimation (Table 4.15). Same procedure as build_s11_counting.py (tile
field into 640-crops, detect, map to field coords, NMS-dedup 21px, calibrate tau on dev by min count-MAE, report on
test; R2/Pearson/MAE overall + by density band) but the detector is MicroYOLO (YOLOv8n-P2 + box9): weakly-supervised =
MicroYOLO trained on Faintness-protected pseudo-labels, GT-supervised = MicroYOLO on real boxes (y11 checkpoints).
Output -> outputs/experiments/s11_counting_microyolo/."""
import os, sys, json, glob
os.environ['XFORMERS_DISABLED']='1'; os.environ['PYTORCH_CUDA_ALLOC_CONF']='expandable_segments:True'
import numpy as np
from PIL import Image; Image.MAX_IMAGE_PIXELS=None
from ultralytics import YOLO
sys.path.append('/root/A_Dissertation'); from common.paths import RAW, OUTPUTS
from scipy.stats import pearsonr
TILE=640; NMS_D=21; PRED_CONF=0.01
Y11=OUTPUTS/'experiments'/'y11_microyolo_table47'/'runs'; RES=OUTPUTS/'experiments'/'s11_counting_microyolo'; RES.mkdir(parents=True,exist_ok=True)
import random; random.seed(0)
def fastmal_fields():
    fm=RAW/'ibadan'/'FASTMAL_THICK_FILMS'; o=[]
    for samp in sorted(p for p in fm.iterdir() if p.is_dir()):
        js=list(samp.glob('*.json'))
        if not js: continue
        for fld in json.loads(js[0].read_text())['rois']:
            ip=samp/fld['image_name']
            if not ip.exists(): continue
            b=[(float(r['x']),float(r['y']),float(r['width']),float(r['height'])) for r in fld['roi'] if 'PARASITE' in r.get('type','') and 'CROWD' not in r.get('type','')]
            if b: o.append((ip,b))
    return o
fields=sorted(fastmal_fields(),key=lambda z:str(z[0])); random.Random(0).shuffle(fields)
devF=fields[0::2][:20]; testF=fields[1::2][:40]   # identical split to build_s11 (seed 0)
def nms_pts(pts, md):
    pts=sorted(pts,key=lambda z:-z[2]); k=[]
    for x,y,s in pts:
        if all((x-kx)**2+(y-ky)**2>=md*md for kx,ky,_ in k): k.append((x,y,s))
    return k
def field_dets(model, im):
    W,H=im.size; arr=np.asarray(im); xs=sorted(set(list(range(0,max(1,W-TILE+1),TILE))+[max(0,W-TILE)])); ys=sorted(set(list(range(0,max(1,H-TILE+1),TILE))+[max(0,H-TILE)]))
    pts=[]
    for ty in ys:
        for tx in xs:
            sub=arr[ty:ty+TILE,tx:tx+TILE]
            if sub.shape[0]<TILE or sub.shape[1]<TILE:
                pad=np.zeros((TILE,TILE,3),np.uint8); pad[:sub.shape[0],:sub.shape[1]]=sub; sub=pad
            p=model.predict(Image.fromarray(sub), imgsz=TILE, conf=PRED_CONF, verbose=False, device=0)[0]
            b=p.boxes.xyxy.cpu().numpy(); s=p.boxes.conf.cpu().numpy()
            for j in range(len(s)): pts.append((tx+(b[j][0]+b[j][2])/2, ty+(b[j][1]+b[j][3])/2, float(s[j])))
    return nms_pts(pts, NMS_D)
def counts_at(detsets, tau): return np.array([sum(1 for (_,_,s) in d if s>=tau) for d in detsets])
def metrics(pred,gt):
    pred=np.asarray(pred,float); gt=np.asarray(gt,float)
    ss_res=np.sum((gt-pred)**2); ss_tot=np.sum((gt-gt.mean())**2); r2=1-ss_res/ss_tot if ss_tot>0 else float('nan')
    r=pearsonr(pred,gt)[0] if np.std(pred)>0 and np.std(gt)>0 else float('nan'); mae=np.mean(np.abs(pred-gt))
    return dict(R2=round(float(r2),3), pearson=round(float(r),3), MAE=round(float(mae),2))
res={}; scatter={}
for name,ckpt in [('weakly_supervised', Y11/'protected'/'weights'/'best.pt'), ('GT_supervised', Y11/'gtsup'/'weights'/'best.pt')]:
    m=YOLO(str(ckpt))
    dev_d=[field_dets(m,Image.open(ip).convert('RGB')) for ip,_ in devF]; dev_gt=[len(b) for _,b in devF]
    taus=np.linspace(0.02,0.9,45); best=(1e9,0.1)
    for t in taus:
        mae=np.mean(np.abs(counts_at(dev_d,t)-np.array(dev_gt)))
        if mae<best[0]: best=(mae,float(t))
    tau=best[1]
    test_d=[field_dets(m,Image.open(ip).convert('RGB')) for ip,_ in testF]; test_gt=np.array([len(b) for _,b in testF])
    pred=counts_at(test_d,tau)
    med=np.median(test_gt); low=test_gt<=med; high=test_gt>med
    res[name]=dict(tau=round(tau,3), overall=metrics(pred,test_gt),
                   low_density=metrics(pred[low],test_gt[low]), high_density=metrics(pred[high],test_gt[high]),
                   mean_gt=round(float(test_gt.mean()),1), mean_pred=round(float(pred.mean()),1))
    scatter[name]=(test_gt.tolist(), pred.tolist())
    print(name, res[name], flush=True)
json.dump(res, open(RES/'counting_results.json','w'), indent=2)
print('\n=== MicroYOLO counting (cf RetinaNet Table 4.15: weakly r0.47/low0.70, GT r0.80) ===')
for name in res: print(f"  {name:18s} overall r={res[name]['overall']['pearson']} MAE={res[name]['overall']['MAE']} | low-density r={res[name]['low_density']['pearson']} | tau={res[name]['tau']} mean_pred={res[name]['mean_pred']}/gt{res[name]['mean_gt']}")
print('COUNTING_MICRO_DONE',flush=True)
