# -*- coding: utf-8 -*-
"""Stage 3 — train YOLOv8 on the §2.3 pseudo-labels (A0 vs faintness-protected) and a GT-supervised baseline; validate
detection mAP on FASTMAL-test crops. Box regression should lift absolute AP (vs the patch detector's patch-quantised
boxes) and the A0-vs-protected ordering should hold. Output -> outputs/experiments/s5_yolo/yolo_results.json"""
import os, sys, json
os.environ['XFORMERS_DISABLED']='1'; os.environ['PYTORCH_CUDA_ALLOC_CONF']='expandable_segments:True'
sys.path.append('/root/A_Dissertation'); from common.paths import OUTPUTS
from ultralytics import YOLO
OUT=OUTPUTS/'experiments'/'s5_yolo'; res={}
for nm in ['a0','protected','gtsup']:
    yaml=OUT/f'data_{nm}.yaml'
    if not yaml.exists(): print('missing',yaml); continue
    m=YOLO('yolov8n.pt')
    m.train(data=str(yaml), epochs=80, imgsz=640, batch=16, device=0, workers=8,
            project=str(OUT/'runs'), name=nm, exist_ok=True, verbose=False, plots=False, seed=0)
    mt=m.val(data=str(yaml), split='val', verbose=False)
    res[nm]=dict(mAP50=round(float(mt.box.map50),3), mAP50_95=round(float(mt.box.map),3),
                 precision=round(float(mt.box.mp),3), recall=round(float(mt.box.mr),3))
    print('===', nm, res[nm], flush=True)
json.dump(res, open(OUT/'yolo_results.json','w'), indent=2)
print('\n=== YOLO detection mAP on FASTMAL (val) ==='); print(json.dumps(res,indent=2))
print('YOLO_TRAIN_DONE', flush=True)
