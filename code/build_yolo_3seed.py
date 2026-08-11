# -*- coding: utf-8 -*-
"""Phase 1b: 3-seed the top YOLO contenders (v9t leader, yolo11n runner-up, v8n incumbent) on A0 pseudo-labels to
confirm the version-screen ordering is not single-seed luck. Same recipe (80ep/640/batch16). Output -> s5_yolo/yolo_3seed.json"""
import os, sys, json, traceback
os.environ['XFORMERS_DISABLED']='1'; os.environ['PYTORCH_CUDA_ALLOC_CONF']='expandable_segments:True'
sys.path.append('/root/A_Dissertation'); from common.paths import OUTPUTS
import numpy as np
from ultralytics import YOLO
OUT=OUTPUTS/'experiments'/'s5_yolo'; yaml=str(OUT/'data_a0.yaml')
VERSIONS=['yolov8n','yolov9t','yolo11n']
runs={v:[] for v in VERSIONS}
for s in [0,1,2]:
    for v in VERSIONS:
        try:
            print(f'=== {v} seed {s} ===',flush=True)
            m=YOLO(f'{v}.pt')
            m.train(data=yaml, epochs=80, imgsz=640, batch=16, device=0, workers=8,
                    project=str(OUT/'runs_3seed'), name=f'{v}_s{s}', exist_ok=True, verbose=False, plots=False, seed=s)
            mt=m.val(data=yaml, split='val', verbose=False)
            r=dict(mAP50=round(float(mt.box.map50),4), mAP50_95=round(float(mt.box.map),4),
                   precision=round(float(mt.box.mp),4), recall=round(float(mt.box.mr),4))
            runs[v].append(r); print('  ->',v,s,r,flush=True)
        except Exception as e:
            print('  !! FAILED',v,s,str(e)[:200],flush=True); traceback.print_exc()
    json.dump(runs, open(OUT/'yolo_3seed.json','w'), indent=2)
agg={}
for v in VERSIONS:
    if runs[v]:
        ks=list(runs[v][0].keys()); agg[v]={k:{'mean':round(float(np.mean([r[k] for r in runs[v]])),4),'std':round(float(np.std([r[k] for r in runs[v]])),4)} for k in ks}
json.dump(dict(agg=agg,runs=runs), open(OUT/'yolo_3seed.json','w'), indent=2)
print('\n=== YOLO version 3-seed on A0 (mAP50 mean±std) ===')
for v in VERSIONS:
    a=agg.get(v,{}).get('mAP50',{}); print(f'  {v:10s} mAP50 {a.get("mean")} ± {a.get("std")}')
print('YOLO_3SEED_DONE', flush=True)
