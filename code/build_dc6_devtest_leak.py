# -*- coding: utf-8 -*-
"""Quantify the FASTMAL dev/test FILM-overlap leakage. Re-evaluate the GT-supervised detector (trained on FASTMAL-dev)
and the weakly-supervised refined detector (trained on Ibadan bag, disjoint from FASTMAL) on three subsets of the
FASTMAL-test crops: FULL (current), CLEAN (test films disjoint from dev), LEAKED (test films shared with dev).
Custom greedy AP@0.3/0.5 (== Table 4.9 metric). No retraining. Output -> outputs/experiments/dc_devtest_leak.json"""
import os, sys, json, glob, random
os.environ['XFORMERS_DISABLED']='1'
import numpy as np
from PIL import Image; Image.MAX_IMAGE_PIXELS=None
from ultralytics import YOLO
sys.path.append('/root/A_Dissertation'); from common.paths import OUTPUTS, RAW
dev='cuda'; CROP=640
YO=OUTPUTS/'experiments'/'s5_yolo'
GTSUP=OUTPUTS/'experiments'/'y11_microyolo_table47'/'runs'/'gtsup'/'weights'/'best.pt'
WEAK=OUTPUTS/'experiments'/'y13_microyolo_ts'/'runs'/'protected_s0_r1'/'weights'/'best.pt'
# reconstruct testF -> film mapping (exactly as build_s5_yolo_data.py)
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
fields=sorted(fastmal_fields(),key=lambda z:str(z)); random.Random(0).shuffle(fields)
devF=fields[0::2][:20]; testF=fields[1::2][:40]
dev_films=set(f for f,_ in devF); test_films=set(f for f,_ in testF)
leaked_films=dev_films & test_films; clean_films=test_films - dev_films
k2film={k:testF[k][0] for k in range(len(testF))}
print(f'dev films {sorted(dev_films)}',flush=True)
print(f'test clean films {sorted(clean_films)}  leaked films {sorted(leaked_films)}',flush=True)
# val crops (shared GT val) -> tag by film via te{k}
val_imgs=sorted(glob.glob(str(YO/'ds_a0'/'images'/'val'/'*.jpg'))); VLAB=YO/'ds_a0'/'labels'/'val'
def crop_film(ip):
    b=os.path.basename(ip)
    try: k=int(b.split('_')[0][2:])  # te{k}_...
    except Exception: return None
    return k2film.get(k)
def load_gt(ip):
    lp=VLAB/(os.path.basename(ip)[:-4]+'.txt'); bx=[]
    if lp.exists():
        for ln in lp.read_text().splitlines():
            f=ln.split()
            if len(f)==5: _,cx,cy,w,h=[float(z) for z in f]; cx,cy,w,h=cx*CROP,cy*CROP,w*CROP,h*CROP; bx.append([cx-w/2,cy-h/2,cx+w/2,cy+h/2])
    return bx
films_of={ip:crop_film(ip) for ip in val_imgs}
gts={ip:load_gt(ip) for ip in val_imgs}; cropgt={ip:len(gts[ip]) for ip in val_imgs}
med=np.median([c for c in cropgt.values() if c>0]) if any(cropgt.values()) else 0
def iou(a,b):
    ix1,iy1=max(a[0],b[0]),max(a[1],b[1]); ix2,iy2=min(a[2],b[2]),min(a[3],b[3]); iw,ih=max(0,ix2-ix1),max(0,iy2-iy1); it=iw*ih
    ua=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-it; return it/ua if ua>0 else 0
def ap(t,s,n):
    if not t:return 0.0
    o=np.argsort(-np.array(s)); t=np.array(t)[o]; ct=np.cumsum(t);cf=np.cumsum(1-t);rc=ct/max(1,n);pr=ct/np.maximum(1,ct+cf)
    mr=np.r_[0,rc,1];mp=np.r_[0,pr,0]
    for i in range(len(mp)-1,0,-1):mp[i-1]=max(mp[i-1],mp[i])
    k=np.where(mr[1:]!=mr[:-1])[0];return float(np.sum((mr[k+1]-mr[k])*mp[k+1]))
def evaluate(model, imgs):
    at={0.3:[],0.5:[]}; sc=[]; ng=0; btp={'low':[],'high':[]}; bsc={'low':[],'high':[]}; bg={'low':0,'high':0}
    for ip in imgs:
        gt=gts[ip]; ng+=len(gt); bd='low' if cropgt[ip]<=med else 'high'; bg[bd]+=len(gt)
        r=model.predict(ip,imgsz=CROP,conf=0.05,verbose=False,device=0)[0]; bxs=r.boxes.xyxy.cpu().numpy(); scs=r.boxes.conf.cpu().numpy(); o=np.argsort(-scs)
        for th in (0.3,0.5):
            m=set()
            for j in o:
                bst=-1;bj=-1
                for jg,g in enumerate(gt):
                    if jg in m: continue
                    v=iou(bxs[j],g)
                    if v>bst: bst,bj=v,jg
                tp=1 if bst>=th and bj>=0 else 0
                if tp:m.add(bj)
                at[th].append(tp)
                if th==0.3: sc.append(float(scs[j])); btp[bd].append(tp); bsc[bd].append(float(scs[j]))
    return dict(n_img=len(imgs), n_gt=ng, **{f'AP@{t}':round(ap(at[t],sc,ng),3) for t in (0.3,0.5)},
                AP03_low=round(ap(btp['low'],bsc['low'],bg['low']),3), AP03_high=round(ap(btp['high'],bsc['high'],bg['high']),3))
full=val_imgs
clean=[ip for ip in val_imgs if films_of[ip] in clean_films]
leaked=[ip for ip in val_imgs if films_of[ip] in leaked_films]
print(f'val crops: full {len(full)} | clean {len(clean)} | leaked {len(leaked)}',flush=True)
out={}
for name,wts in [('GT-supervised (trained on FASTMAL-dev)',GTSUP),('weakly-supervised refined (trained on Ibadan bag)',WEAK)]:
    m=YOLO(str(wts)); out[name]={sub:evaluate(m,imgs) for sub,imgs in [('full',full),('clean',clean),('leaked',leaked)]}
    print(f'\n{name}')
    for sub in ['full','clean','leaked']:
        r=out[name][sub]; print(f'  {sub:7s} n_img={r["n_img"]} AP@0.3={r["AP@0.3"]} AP@0.5={r["AP@0.5"]} low={r["AP03_low"]} high={r["AP03_high"]}')
json.dump({'dev_films':sorted(dev_films),'clean_films':sorted(clean_films),'leaked_films':sorted(leaked_films),'results':out},
          open(OUTPUTS/'experiments'/'dc_devtest_leak.json','w'), indent=2)
print('DC6_DONE',flush=True)
