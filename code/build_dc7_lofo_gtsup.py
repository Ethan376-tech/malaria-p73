# -*- coding: utf-8 -*-
"""Clean the GT-supervised reference via leave-one-FILM-out (LOFO). For each FASTMAL film f that appears in the report's
test set, train a GT-supervised MicroYOLO on the real boxes of all OTHER films and evaluate on film f's test crops.
Aggregating over the full report test set gives a film-DISJOINT (clean) GT-supervised AP, directly comparable to the
weakly-supervised numbers (which stay on the same full test and are already clean). Output -> outputs/experiments/
dc_lofo_gtsup/results.json"""
import os, sys, json, glob, shutil, random
os.environ['XFORMERS_DISABLED']='1'; os.environ['PYTORCH_CUDA_ALLOC_CONF']='expandable_segments:True'
import numpy as np
from PIL import Image; Image.MAX_IMAGE_PIXELS=None
from ultralytics import YOLO
sys.path.append('/root/A_Dissertation'); from common.paths import OUTPUTS, RAW
dev='cuda'; CROP=640; EP=60
RES=OUTPUTS/'experiments'/'dc_lofo_gtsup'; RES.mkdir(parents=True,exist_ok=True)
POOL=RES/'pool';
fm=RAW/'ibadan'/'FASTMAL_THICK_FILMS'
def fastmal_fields():
    o=[]
    for samp in sorted(p for p in fm.iterdir() if p.is_dir()):
        js=list(samp.glob('*.json'))
        if not js: continue
        for fld in json.loads(js[0].read_text())['rois']:
            ip=samp/fld['image_name']
            if not ip.exists(): continue
            b=[(float(r['x']),float(r['y']),float(r['width']),float(r['height'])) for r in fld['roi'] if 'PARASITE' in r.get('type','') and 'CROWD' not in r.get('type','')]
            if b: o.append((samp.name, ip, b))
    return o
allf=fastmal_fields()
# reconstruct report test fields (build_s5): shuffle by (film,imgname), test=fields[1::2][:40]
keyed=sorted(allf, key=lambda z:(z[0], z[1].name)); random.Random(0).shuffle(keyed)
testset=set((f,ip.name) for (f,ip,_) in keyed[1::2][:40])
def crop_grid(W,H):
    xs=list(range(0,max(1,W-CROP+1),CROP)) + ([W-CROP] if W>CROP else [0]); ys=list(range(0,max(1,H-CROP+1),CROP)) + ([H-CROP] if H>CROP else [0])
    return sorted(set(xs)),sorted(set(ys))
def gen_crops(film, ip, boxes, outdir, tag, only_boxes=True):
    arr=np.asarray(Image.open(ip).convert('RGB')); H,W=arr.shape[:2]; xs,ys=crop_grid(W,H); n=0
    (outdir/'images').mkdir(parents=True,exist_ok=True); (outdir/'labels').mkdir(parents=True,exist_ok=True)
    bx=[(x+w/2,y+h/2,w,h) for (x,y,w,h) in boxes]
    for cxo in xs:
        for cyo in ys:
            sub=arr[cyo:cyo+CROP, cxo:cxo+CROP]
            if sub.shape[0]<CROP or sub.shape[1]<CROP:
                pad=np.zeros((CROP,CROP,3),np.uint8); pad[:sub.shape[0],:sub.shape[1]]=sub; sub=pad
            labs=[]
            for (cx,cy,w,h) in bx:
                lx,ly=cx-cxo,cy-cyo
                if 0<=lx<CROP and 0<=ly<CROP: labs.append(f"0 {lx/CROP:.6f} {ly/CROP:.6f} {w/CROP:.6f} {h/CROP:.6f}")
            name=f"{tag}_{cxo}_{cyo}"; Image.fromarray(sub).save(outdir/'images'/f'{name}.jpg',quality=92)
            (outdir/'labels'/f'{name}.txt').write_text("\n".join(labs)); n+=1
    return n
# generate all crops per film, mark which crops are TEST (from report test fields)
if POOL.exists(): shutil.rmtree(POOL)
crop_film={}; crop_istest={}
for i,(film,ip,boxes) in enumerate(allf):
    d=POOL/film; nf=gen_crops(film,ip,boxes,d,f'{film}_{i}')
for film in sorted(set(f for f,_,_ in allf)):
    for imgp in glob.glob(str(POOL/film/'images'/'*.jpg')):
        crop_film[imgp]=film
# tag test crops: a crop is test if its source field (film,imgname) in testset. We re-map by regenerating test crops separately.
# simpler: rebuild mapping tag->(film,imgname)
tagmap={}
for i,(film,ip,boxes) in enumerate(allf): tagmap[f'{film}_{i}']=(film, ip.name)
def is_test_crop(imgp):
    b=os.path.basename(imgp); parts=b.split('_'); tag='_'.join(parts[:2]); fi=tagmap.get(tag); return fi in testset if fi else False
test_films=sorted(set(f for (f,n) in testset))
print(f'test fields: {len(testset)}  test films: {test_films}',flush=True)
print(f'total crops: {len(crop_film)}',flush=True)
def iou(a,b):
    ix1,iy1=max(a[0],b[0]),max(a[1],b[1]); ix2,iy2=min(a[2],b[2]),min(a[3],b[3]); iw,ih=max(0,ix2-ix1),max(0,iy2-iy1); it=iw*ih
    ua=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-it; return it/ua if ua>0 else 0
def load_gt(lp):
    bx=[]
    if os.path.exists(lp):
        for ln in open(lp).read().splitlines():
            f=ln.split()
            if len(f)==5: _,cx,cy,w,h=[float(z) for z in f]; cx,cy,w,h=cx*CROP,cy*CROP,w*CROP,h*CROP; bx.append([cx-w/2,cy-h/2,cx+w/2,cy+h/2])
    return bx
# gather per-crop gt counts for band split (median over test crops)
test_crops=[p for p in crop_film if is_test_crop(p)]
gtc={p:load_gt(str(POOL/crop_film[p]/'labels'/(os.path.basename(p)[:-4]+'.txt'))) for p in test_crops}
med=np.median([len(g) for g in gtc.values() if len(g)>0]) if any(len(g)>0 for g in gtc.values()) else 0
print(f'test crops: {len(test_crops)}  median gt/crop(pos): {med}',flush=True)
def make_train(fold_excl):
    root=RES/f'train_excl_{fold_excl}'
    for sub in ['images','labels']:
        d=root/sub
        if d.exists(): shutil.rmtree(d)
        d.mkdir(parents=True,exist_ok=True)
    n=0
    for film in sorted(set(crop_film.values())):
        if film==fold_excl: continue
        for imgp in glob.glob(str(POOL/film/'images'/'*.jpg')):
            b=os.path.basename(imgp)
            lp=str(POOL/film/'labels'/(b[:-4]+'.txt'))
            if not open(lp).read().strip(): continue  # train only on crops with a box (positive crops)
            try: os.link(imgp,str(root/'images'/b))
            except OSError: shutil.copy(imgp,str(root/'images'/b))
            shutil.copy(lp,str(root/'labels'/(b[:-4]+'.txt'))); n+=1
    # small val = a few of its own train (dummy, not used for our AP)
    (root/'val_images').mkdir(exist_ok=True)
    yaml=str(root/'data.yaml'); open(yaml,'w').write(f"path: {root}\ntrain: images\nval: images\nnc: 1\nnames: ['parasite']\n")
    return yaml, n
# LOFO: for each test film, train excluding it, predict on its test crops
preds={}  # crop -> (boxes, scores)
for f in test_films:
    yaml,ntr=make_train(f)
    y=YOLO('yolov8n-p2.yaml')
    y.train(data=yaml, epochs=EP, imgsz=CROP, batch=16, device=0, workers=8, box=9.0, pretrained='yolov8n.pt',
            project=str(RES/'runs'), name=f'excl_{f}', exist_ok=True, verbose=False, plots=False, seed=0)
    best=RES/'runs'/f'excl_{f}'/'weights'/'best.pt'; m=YOLO(str(best))
    fcrops=[p for p in test_crops if crop_film[p]==f]
    for p in fcrops:
        r=m.predict(p,imgsz=CROP,conf=0.05,verbose=False,device=0)[0]; preds[p]=(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy())
    print(f'  fold excl {f}: trained on {ntr} crops, predicted {len(fcrops)} test crops',flush=True)
# aggregate custom AP over all test crops
def ap(t,s,n):
    if not t:return 0.0
    o=np.argsort(-np.array(s)); t=np.array(t)[o]; ct=np.cumsum(t);cf=np.cumsum(1-t);rc=ct/max(1,n);pr=ct/np.maximum(1,ct+cf)
    mr=np.r_[0,rc,1];mp=np.r_[0,pr,0]
    for i in range(len(mp)-1,0,-1):mp[i-1]=max(mp[i-1],mp[i])
    k=np.where(mr[1:]!=mr[:-1])[0];return float(np.sum((mr[k+1]-mr[k])*mp[k+1]))
at={0.3:[],0.5:[]}; sc=[]; ng=0; btp={'low':[],'high':[]}; bsc={'low':[],'high':[]}; bg={'low':0,'high':0}
for p in test_crops:
    gt=gtc[p]; ng+=len(gt); bd='low' if len(gt)<=med else 'high'; bg[bd]+=len(gt)
    bxs,scs=preds.get(p,(np.zeros((0,4)),np.array([]))); o=np.argsort(-scs)
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
res=dict(**{f'AP@{t}':round(ap(at[t],sc,ng),3) for t in (0.3,0.5)}, AP03_low=round(ap(btp['low'],bsc['low'],bg['low']),3),
         AP03_high=round(ap(btp['high'],bsc['high'],bg['high']),3), n_test_crops=len(test_crops), n_gt=ng)
json.dump(dict(lofo_gtsup=res, leaked_gtsup_ref={'AP@0.5':0.681,'AP@0.3':0.771,'low':0.701},
               weakly_sup_ref={'AP@0.5':0.398,'AP@0.3':0.611,'low':0.497}), open(RES/'results.json','w'), indent=2)
print('\n=== CLEAN (LOFO) GT-supervised on full report test ===')
print('  ', res)
print('  vs leaked GT-sup (report): AP@0.5 0.681 / AP@0.3 0.771 / low 0.701')
print('  vs weakly-sup (clean, report): AP@0.5 0.398 / AP@0.3 0.611 / low 0.497')
print('DC7_DONE',flush=True)
