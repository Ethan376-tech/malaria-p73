# -*- coding: utf-8 -*-
"""Phase 2C continued (1-seed screen): further-specialize MicroYOLO (YOLOv8n + P2 stride-4 small-object head) for tiny
sparse parasites. On top of the confirmed P2 head, test: reduced augmentation (mosaic hurts sparse tiny objects),
higher box-loss gain (tighter tiny-box localisation), moderate resolution (768). Reference = plain P2 (3-seed 0.311,
seed-0 0.298). Train on A0, eval mAP50. Output -> y9_microyolo/results.json. 3-seed the winner vs P2 next."""
import os, sys, json, traceback
os.environ['XFORMERS_DISABLED']='1'; os.environ['PYTORCH_CUDA_ALLOC_CONF']='expandable_segments:True'
sys.path.append('/root/A_Dissertation'); from common.paths import OUTPUTS
from ultralytics import YOLO
OUT=OUTPUTS/'experiments'/'s5_yolo'; yaml=str(OUT/'data_a0.yaml'); RES=OUTPUTS/'experiments'/'y9_microyolo'; RES.mkdir(parents=True,exist_ok=True)
LOWAUG={'mosaic':0.0,'scale':0.2,'close_mosaic':0,'translate':0.05}
CONFIGS=[
    {'name':'p2_ref',        'imgsz':640, 'batch':16, 'kw':{}},
    {'name':'p2_lowaug',     'imgsz':640, 'batch':16, 'kw':dict(LOWAUG)},
    {'name':'p2_boxgain',    'imgsz':640, 'batch':16, 'kw':{'box':9.0}},
    {'name':'p2_768',        'imgsz':768, 'batch':12, 'kw':{}},
    {'name':'p2_lowaug_box', 'imgsz':640, 'batch':16, 'kw':dict(LOWAUG, box=9.0)},
]
res={}
for cfg in CONFIGS:
    try:
        print(f"=== {cfg['name']} (imgsz={cfg['imgsz']} kw={cfg['kw']}) ===",flush=True)
        m=YOLO('yolov8n-p2.yaml')
        try: m=m.load('yolov8n.pt')
        except Exception as e: print('  transfer failed:',str(e)[:120],flush=True)
        m.train(data=yaml, epochs=80, imgsz=cfg['imgsz'], batch=cfg['batch'], device=0, workers=8,
                project=str(RES/'runs'), name=cfg['name'], exist_ok=True, verbose=False, plots=False, seed=0, **cfg['kw'])
        mt=m.val(data=yaml, split='val', imgsz=cfg['imgsz'], verbose=False)
        res[cfg['name']]=dict(mAP50=round(float(mt.box.map50),4), mAP50_95=round(float(mt.box.map),4),
                              precision=round(float(mt.box.mp),4), recall=round(float(mt.box.mr),4))
        print('  ->',cfg['name'],res[cfg['name']],flush=True)
    except Exception as e:
        res[cfg['name']]=dict(error=str(e)[:200]); print('  !! FAILED',cfg['name'],str(e)[:200],flush=True); traceback.print_exc()
    json.dump(res,open(RES/'results.json','w'),indent=2)
print('\n=== MicroYOLO further-specialization screen (mAP50 on A0, 1-seed; P2 ref seed-0 0.298 / 3-seed 0.311) ===')
for cfg in CONFIGS: print(f"  {cfg['name']:16s} {res.get(cfg['name'],{})}")
print('Y9_DONE',flush=True)
