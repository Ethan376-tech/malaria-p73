# -*- coding: utf-8 -*-
"""Phase 1a screen: does a newer YOLO beat YOLOv8n on the A0 pseudo-labels? Train each version (seed 0, same recipe as
Table 4.7: 80ep/640/batch16) on data_a0.yaml, eval mAP50 on the same FASTMAL val split. Per-version try/except so a
failed weight download does not abort the sweep. Output -> s5_yolo/yolo_versions.json  (3-seed the top contenders next)."""
import os, sys, json, traceback
os.environ['XFORMERS_DISABLED']='1'; os.environ['PYTORCH_CUDA_ALLOC_CONF']='expandable_segments:True'
sys.path.append('/root/A_Dissertation'); from common.paths import OUTPUTS
from ultralytics import YOLO
OUT=OUTPUTS/'experiments'/'s5_yolo'; yaml=str(OUT/'data_a0.yaml')
VERSIONS=['yolov8n','yolov9t','yolov10n','yolo11n','yolo12n','yolo26n']
res={}
for v in VERSIONS:
    try:
        print(f'=== train {v} (seed 0, a0) ===',flush=True)
        m=YOLO(f'{v}.pt')
        m.train(data=yaml, epochs=80, imgsz=640, batch=16, device=0, workers=8,
                project=str(OUT/'runs_ver'), name=v, exist_ok=True, verbose=False, plots=False, seed=0)
        mt=m.val(data=yaml, split='val', verbose=False)
        res[v]=dict(mAP50=round(float(mt.box.map50),3), mAP50_95=round(float(mt.box.map),3),
                    precision=round(float(mt.box.mp),3), recall=round(float(mt.box.mr),3))
        print('  ->',v,res[v],flush=True)
    except Exception as e:
        res[v]=dict(error=str(e)[:200]); print('  !! FAILED',v,str(e)[:200],flush=True); traceback.print_exc()
    json.dump(res, open(OUT/'yolo_versions.json','w'), indent=2)
print('\n=== YOLO version screen on A0 (mAP50, seed 0) ===')
for v in VERSIONS:
    r=res.get(v,{}); print(f'  {v:10s} {r}')
print('YOLO_VERSIONS_DONE', flush=True)
