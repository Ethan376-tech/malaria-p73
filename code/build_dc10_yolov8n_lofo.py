# -*- coding: utf-8 -*-
"""LOFO GT-supervised YOLOv8n (stock, not P2) for Table 4.7. Reuses dc7 pool crops; trains one yolov8n per test film
(on all OTHER films' positive GT crops); saves fold weights to dc_lofo_yolov8n/runs/excl_{f}. mAP50 aggregation is done
in dc13."""
import os, sys, glob, shutil
os.environ['XFORMERS_DISABLED']='1'; os.environ['PYTORCH_CUDA_ALLOC_CONF']='expandable_segments:True'
from ultralytics import YOLO
sys.path.append('/root/A_Dissertation'); from common.paths import OUTPUTS, RAW
import json, random
POOL=OUTPUTS/'experiments'/'dc_lofo_gtsup'/'pool'; RES=OUTPUTS/'experiments'/'dc_lofo_yolov8n'; RES.mkdir(parents=True,exist_ok=True)
CROP=640; EP=60
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
test_films=sorted(set(f for f,_ in testF))
all_films=sorted(p.name for p in POOL.iterdir() if p.is_dir())
def make_train(excl):
    root=RES/f'train_excl_{excl}'
    for sub in ['images','labels']:
        d=root/sub
        if d.exists(): shutil.rmtree(d)
        d.mkdir(parents=True,exist_ok=True)
    n=0
    for film in all_films:
        if film==excl: continue
        for imgp in glob.glob(str(POOL/film/'images'/'*.jpg')):
            lp=str(POOL/film/'labels'/(os.path.basename(imgp)[:-4]+'.txt'))
            if not os.path.exists(lp) or not open(lp).read().strip(): continue
            b=os.path.basename(imgp)
            try: os.link(imgp,str(root/'images'/b))
            except OSError: shutil.copy(imgp,str(root/'images'/b))
            shutil.copy(lp,str(root/'labels'/(b[:-4]+'.txt'))); n+=1
    yaml=str(root/'data.yaml'); open(yaml,'w').write(f"path: {root}\ntrain: images\nval: images\nnc: 1\nnames: ['parasite']\n")
    return yaml, n
for f in test_films:
    yaml,n=make_train(f)
    y=YOLO('yolov8n.yaml')
    try: y=y.load('yolov8n.pt')
    except Exception: pass
    y.train(data=yaml, epochs=EP, imgsz=CROP, batch=16, device=0, workers=8, box=7.5, pretrained='yolov8n.pt',
            project=str(RES/'runs'), name=f'excl_{f}', exist_ok=True, verbose=False, plots=False, seed=0)
    print(f'  yolov8n fold excl {f}: trained on {n} crops',flush=True)
print('DC10_DONE',flush=True)
