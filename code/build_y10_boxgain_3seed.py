# -*- coding: utf-8 -*-
"""Confirm + tune the box-loss gain for MicroYOLO (YOLOv8n-P2). 3 seeds x {p2 (box7.5 default), box9, box11} on A0,
mAP50. The robust winner is the MicroYOLO version adopted downstream. Output -> y10_boxgain_3seed/results.json"""
import os, sys, json, traceback
os.environ['XFORMERS_DISABLED']='1'; os.environ['PYTORCH_CUDA_ALLOC_CONF']='expandable_segments:True'
import numpy as np
sys.path.append('/root/A_Dissertation'); from common.paths import OUTPUTS
from ultralytics import YOLO
OUT=OUTPUTS/'experiments'/'s5_yolo'; yaml=str(OUT/'data_a0.yaml'); RES=OUTPUTS/'experiments'/'y10_boxgain_3seed'; RES.mkdir(parents=True,exist_ok=True)
CONFIGS=[{'name':'p2_box7.5','box':7.5},{'name':'p2_box9','box':9.0},{'name':'p2_box11','box':11.0}]
runs={c['name']:[] for c in CONFIGS}
for seed in [0,1,2]:
    for cfg in CONFIGS:
        try:
            print(f"=== seed {seed} {cfg['name']} ===",flush=True)
            m=YOLO('yolov8n-p2.yaml')
            try: m=m.load('yolov8n.pt')
            except Exception as e: print('  transfer failed:',str(e)[:120],flush=True)
            m.train(data=yaml, epochs=80, imgsz=640, batch=16, device=0, workers=8, box=cfg['box'],
                    project=str(RES/'runs'), name=f"{cfg['name']}_s{seed}", exist_ok=True, verbose=False, plots=False, seed=seed)
            mt=m.val(data=yaml, split='val', imgsz=640, verbose=False)
            r=dict(mAP50=round(float(mt.box.map50),4), mAP50_95=round(float(mt.box.map),4), precision=round(float(mt.box.mp),4), recall=round(float(mt.box.mr),4))
            runs[cfg['name']].append(r); print('  ->',cfg['name'],seed,r,flush=True)
        except Exception as e:
            print('  !! FAILED',cfg['name'],seed,str(e)[:200],flush=True); traceback.print_exc()
    json.dump(runs,open(RES/'results.json','w'),indent=2)
agg={}
for c in CONFIGS:
    nm=c['name']
    if runs[nm]: agg[nm]={k:{'mean':round(float(np.mean([r[k] for r in runs[nm]])),4),'std':round(float(np.std([r[k] for r in runs[nm]])),4)} for k in runs[nm][0]}
json.dump(dict(agg=agg,runs=runs),open(RES/'results.json','w'),indent=2)
print('\n=== MicroYOLO box-gain 3-seed (mAP50 mean±std) ===')
for c in CONFIGS:
    a=agg.get(c['name'],{}).get('mAP50',{}); print(f"  {c['name']:10s} mAP50 {a.get('mean')} ± {a.get('std')}")
print('Y10_DONE',flush=True)
