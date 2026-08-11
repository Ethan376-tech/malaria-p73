# -*- coding: utf-8 -*-
"""Complete the Table 4.7 MicroYOLO row: MicroYOLO (YOLOv8n-P2 + box-loss gain 9.0) trained on A0 / Faintness-protected
/ GT-supervised pseudo-labels, eval ultralytics mAP50 — same protocol as the YOLOv8n row (build_s5_yolo_train, seed 0).
Output -> y11_microyolo_table47/results.json"""
import os, sys, json
os.environ['XFORMERS_DISABLED']='1'; os.environ['PYTORCH_CUDA_ALLOC_CONF']='expandable_segments:True'
sys.path.append('/root/A_Dissertation'); from common.paths import OUTPUTS
from ultralytics import YOLO
OUT=OUTPUTS/'experiments'/'s5_yolo'; RES=OUTPUTS/'experiments'/'y11_microyolo_table47'; RES.mkdir(parents=True,exist_ok=True); res={}
for nm in ['a0','protected','gtsup']:
    yaml=OUT/f'data_{nm}.yaml'
    if not yaml.exists(): print('missing',yaml); continue
    print('=== MicroYOLO on',nm,'===',flush=True)
    m=YOLO('yolov8n-p2.yaml')
    try: m=m.load('yolov8n.pt')
    except Exception as e: print('  transfer failed:',str(e)[:120],flush=True)
    m.train(data=str(yaml), epochs=80, imgsz=640, batch=16, device=0, workers=8, box=9.0,
            project=str(RES/'runs'), name=nm, exist_ok=True, verbose=False, plots=False, seed=0)
    mt=m.val(data=str(yaml), split='val', verbose=False)
    res[nm]=dict(mAP50=round(float(mt.box.map50),3), mAP50_95=round(float(mt.box.map),3),
                 precision=round(float(mt.box.mp),3), recall=round(float(mt.box.mr),3))
    print('===',nm,res[nm],flush=True)
    json.dump(res,open(RES/'results.json','w'),indent=2)
print('\n=== MicroYOLO Table 4.7 row (mAP50; cf YOLOv8n 0.259/0.258/0.704) ===')
for nm in ['a0','protected','gtsup']: print(f'  {nm:10s} {res.get(nm,{})}')
print('Y11_DONE',flush=True)
