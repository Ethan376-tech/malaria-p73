# -*- coding: utf-8 -*-
"""Aggregate ultralytics mAP50 for the LOFO YOLO detectors (MicroYOLO folds=dc7, YOLOv8n folds=dc10). For each test
film f, run the fold-f model's ultralytics val on that film's test crops -> mAP50_f + n_instances; instance-weighted mean
= clean GT-sup mAP50 (report-metric). Also runs the LEAKED MicroYOLO (y11 gtsup) full val as a method check vs 0.719.
Output -> dc_lofo_gtsup/map50_agg.json"""
import os, sys, glob, json, random, shutil
os.environ['XFORMERS_DISABLED']='1'
import numpy as np
from ultralytics import YOLO
sys.path.append('/root/A_Dissertation'); from common.paths import OUTPUTS, RAW
CROP=640
YO=OUTPUTS/'experiments'/'s5_yolo'; LOFO=OUTPUTS/'experiments'/'dc_lofo_gtsup'
DC7=LOFO/'runs'; DC10=OUTPUTS/'experiments'/'dc_lofo_yolov8n'/'runs'
GTSUP=OUTPUTS/'experiments'/'y11_microyolo_table47'/'runs'/'gtsup'/'weights'/'best.pt'
WORK=OUTPUTS/'experiments'/'dc_lofo_gtsup'/'perfilm_val'; WORK.mkdir(parents=True,exist_ok=True)
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
k2film={k:testF[k][0] for k in range(len(testF))}
val_imgs=sorted(glob.glob(str(YO/'ds_a0'/'images'/'val'/'*.jpg'))); VLAB=YO/'ds_a0'/'labels'/'val'
def cfilm(ip):
    try: k=int(os.path.basename(ip).split('_')[0][2:])
    except Exception: return None
    return k2film.get(k)
films={ip:cfilm(ip) for ip in val_imgs}
test_films=sorted(set(f for f in films.values() if f))
# build per-film val dirs
def build_film_val(f):
    d=WORK/f
    for sub in ['images','labels']:
        p=d/sub
        if p.exists(): shutil.rmtree(p)
        p.mkdir(parents=True,exist_ok=True)
    ninst=0
    for ip in [p for p in val_imgs if films[p]==f]:
        b=os.path.basename(ip)
        try: os.link(ip,str(d/'images'/b))
        except OSError: shutil.copy(ip,str(d/'images'/b))
        lp=VLAB/(b[:-4]+'.txt'); shutil.copy(str(lp),str(d/'labels'/(b[:-4]+'.txt'))) if lp.exists() else open(str(d/'labels'/(b[:-4]+'.txt')),'w').write('')
        if lp.exists(): ninst+=len([1 for ln in lp.read_text().splitlines() if len(ln.split())==5])
    yaml=str(d/'data.yaml'); open(yaml,'w').write(f"path: {d}\ntrain: images\nval: images\nnc: 1\nnames: ['parasite']\n")
    return yaml, ninst
film_yaml={}; film_ninst={}
for f in test_films: film_yaml[f],film_ninst[f]=build_film_val(f)
def lofo_map50(runs_dir, cfg):
    num=0.0; den=0
    per={}
    for f in test_films:
        w=runs_dir/f'excl_{f}'/'weights'/'best.pt'
        if not w.exists(): print(f'  MISSING {w}',flush=True); continue
        m=YOLO(str(w)); mt=m.val(data=film_yaml[f], split='val', verbose=False, plots=False)
        ap=float(mt.box.map50); per[f]=round(ap,3); num+=ap*film_ninst[f]; den+=film_ninst[f]
    return round(num/max(1,den),3), per
mm, mper=lofo_map50(DC7, 'yolov8n-p2.yaml'); print(f'MicroYOLO LOFO mAP50 (inst-weighted) = {mm}  per-film {mper}',flush=True)
ym, yper=lofo_map50(DC10, 'yolov8n.yaml'); print(f'YOLOv8n LOFO mAP50 (inst-weighted) = {ym}  per-film {yper}',flush=True)
# leaked check (MicroYOLO gtsup full val)
allyaml=str(WORK/'full'); os.makedirs(WORK/'full'/'images',exist_ok=True); os.makedirs(WORK/'full'/'labels',exist_ok=True)
for ip in val_imgs:
    b=os.path.basename(ip)
    try: os.link(ip,str(WORK/'full'/'images'/b))
    except OSError: pass
    lp=VLAB/(b[:-4]+'.txt')
    if lp.exists(): shutil.copy(str(lp),str(WORK/'full'/'labels'/(b[:-4]+'.txt')))
open(str(WORK/'full'/'data.yaml'),'w').write(f"path: {WORK/'full'}\ntrain: images\nval: images\nnc: 1\nnames: ['parasite']\n")
lk=float(YOLO(str(GTSUP)).val(data=str(WORK/'full'/'data.yaml'),split='val',verbose=False,plots=False).box.map50)
print(f'LEAKED MicroYOLO gtsup mAP50 (full val) = {round(lk,3)}  (report 0.719)',flush=True)
json.dump(dict(microyolo_lofo=mm, yolov8n_lofo=ym, leaked_microyolo=round(lk,3), micro_perfilm=mper, yolo_perfilm=yper), open(LOFO/'map50_agg.json','w'), indent=2)
print('DC13_DONE',flush=True)
