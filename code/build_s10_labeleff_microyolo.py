# -*- coding: utf-8 -*-
"""Table 4.14 label-efficiency with MicroYOLO (YOLOv8n-P2 + box9). At annotation budget k FASTMAL-dev fields with real
boxes: (supervised) MicroYOLO from scratch on k; (hybrid, ours) MicroYOLO pretrained on Faintness-protected pseudo-labels
(y11) fine-tuned on k; (semisl) supervised(k) + self-training on unlabeled Ibadan crops. Custom AP on GT val, comparable
to the RetinaNet Table 4.14. 3 random k-field subsets -> mean±std. Output -> outputs/experiments/s10_labeleff_microyolo/."""
import os, sys, json, glob, shutil, re
os.environ['XFORMERS_DISABLED']='1'; os.environ['PYTORCH_CUDA_ALLOC_CONF']='expandable_segments:True'
import numpy as np
from PIL import Image; Image.MAX_IMAGE_PIXELS=None
from ultralytics import YOLO
sys.path.append('/root/A_Dissertation'); from common.paths import OUTPUTS
CROP=640; BUDGETS=[2,5,10,20]; UNLAB_CAP=400; MICRO='yolov8n-p2.yaml'; BOX=9.0; SSL_CONF=0.3
YO=OUTPUTS/'experiments'/'s5_yolo'; Y11P=OUTPUTS/'experiments'/'y11_microyolo_table47'/'runs'/'protected'/'weights'/'best.pt'
RES=OUTPUTS/'experiments'/'s10_labeleff_microyolo'; RES.mkdir(parents=True,exist_ok=True)
GT_TR=YO/'ds_gtsup'/'images'/'train'; GT_LAB=YO/'ds_gtsup'/'labels'/'train'; VAL_IMG=YO/'ds_gtsup'/'images'/'val'; VAL_LAB=YO/'ds_gtsup'/'labels'/'val'
UNLAB=sorted(glob.glob(str(YO/'ds_a0'/'images'/'train'/'*.jpg')))[:UNLAB_CAP]
def field_idx(p):
    m=re.match(r'gt(\d+)_', os.path.basename(p)); return int(m.group(1)) if m else 999
gt_train=sorted(glob.glob(str(GT_TR/'*.jpg'))); ALL_FIELDS=sorted(set(field_idx(p) for p in gt_train))
def labeled(k, rng):
    chosen=set(rng.choice(ALL_FIELDS, size=min(k,len(ALL_FIELDS)), replace=False))
    return [p for p in gt_train if field_idx(p) in chosen]
def _link(src,dst):
    try: os.link(src,dst)
    except OSError: shutil.copy(src,dst)
def make_ds(name, items):   # items: list of (img_path, label_txt_path_or_None)
    root=RES/name
    for sub in ['images/train','images/val','labels/train','labels/val']:
        d=root/sub
        if d.exists(): shutil.rmtree(d)
        d.mkdir(parents=True,exist_ok=True)
    for ip,lp in items:
        b=os.path.basename(ip); _link(ip,str(root/'images'/'train'/b))
        if lp and os.path.exists(lp): shutil.copy(lp,str(root/'labels'/'train'/(b[:-4]+'.txt')))
        else: open(str(root/'labels'/'train'/(b[:-4]+'.txt')),'w').write("")
    for ip in sorted(glob.glob(str(VAL_IMG/'*.jpg'))):
        b=os.path.basename(ip); _link(ip,str(root/'images'/'val'/b)); shutil.copy(str(VAL_LAB/(b[:-4]+'.txt')),str(root/'labels'/'val'/(b[:-4]+'.txt')))
    for c in glob.glob(str(root)+'/labels/*.cache'): os.remove(c)
    (root/'data.yaml').write_text(f"path: {root}\ntrain: images/train\nval: images/val\nnc: 1\nnames: ['parasite']\n"); return str(root/'data.yaml')
def train_micro(init, yaml, name, seed, epochs, lr0, batch=16):
    if init=='scratch':
        m=YOLO(MICRO)
        try: m=m.load('yolov8n.pt')
        except Exception: pass
    else: m=YOLO(str(init))
    m.train(data=yaml, epochs=epochs, imgsz=CROP, batch=batch, device=0, workers=8, box=BOX, lr0=lr0, warmup_epochs=1,
            project=str(RES/'runs'), name=name, exist_ok=True, verbose=False, plots=False, seed=seed)
    return str(RES/'runs'/name/'weights'/'best.pt')
def yolo_predict(mp, ip, conf):
    r=YOLO(mp).predict(ip, imgsz=CROP, conf=conf, verbose=False, device=0)[0]; return r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()
def self_train(model_path, outdir, conf):
    os.makedirs(outdir,exist_ok=True); m=YOLO(model_path); items=[]
    for ip in UNLAB:
        r=m.predict(ip, imgsz=CROP, conf=conf, verbose=False, device=0)[0]; b=r.boxes.xyxy.cpu().numpy()
        if len(b)==0: continue
        labs=[f"0 {((bb[0]+bb[2])/2/CROP):.6f} {((bb[1]+bb[3])/2/CROP):.6f} {((bb[2]-bb[0])/CROP):.6f} {((bb[3]-bb[1])/CROP):.6f}" for bb in b]
        lp=os.path.join(outdir,os.path.basename(ip)[:-4]+'.txt'); open(lp,'w').write("\n".join(labs)); items.append((ip,lp))
    return items
val_imgs=sorted(glob.glob(str(VAL_IMG/'*.jpg')))
def load_gt(ip):
    lp=VAL_LAB/(os.path.basename(ip)[:-4]+'.txt'); bx=[]
    if lp.exists():
        for ln in lp.read_text().splitlines():
            f=ln.split()
            if len(f)==5: _,cx,cy,w,h=f; cx,cy,w,h=float(cx)*CROP,float(cy)*CROP,float(w)*CROP,float(h)*CROP; bx.append([cx-w/2,cy-h/2,cx+w/2,cy+h/2])
    return bx
gts=[load_gt(ip) for ip in val_imgs]
def iou(a,b):
    ix1,iy1=max(a[0],b[0]),max(a[1],b[1]); ix2,iy2=min(a[2],b[2]),min(a[3],b[3]); iw,ih=max(0,ix2-ix1),max(0,iy2-iy1); it=iw*ih
    ua=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-it; return it/ua if ua>0 else 0.0
def evaluate(mp):
    m=YOLO(mp); at={0.3:[],0.5:[]}; sc=[]; ngt=0
    for idx,ip in enumerate(val_imgs):
        gt=gts[idx]; ngt+=len(gt)
        r=m.predict(ip, imgsz=CROP, conf=0.05, verbose=False, device=0)[0]; bxs=r.boxes.xyxy.cpu().numpy(); scs=r.boxes.conf.cpu().numpy(); order=np.argsort(-scs)
        for th in (0.3,0.5):
            matched=set()
            for j in order:
                best=-1;bj=-1
                for jg,g in enumerate(gt):
                    if jg in matched: continue
                    v=iou(bxs[j],g)
                    if v>best: best,bj=v,jg
                tp=1 if best>=th and bj>=0 else 0
                if tp: matched.add(bj)
                at[th].append(tp)
                if th==0.3: sc.append(float(scs[j]))
    def ap(tps,scs,ng):
        if not tps: return 0.0
        o=np.argsort(-np.array(scs)); tps=np.array(tps)[o]; ctp=np.cumsum(tps); cfp=np.cumsum(1-tps); rec=ctp/max(1,ng); pre=ctp/np.maximum(1,ctp+cfp)
        mr=np.concatenate([[0],rec,[1]]); mp=np.concatenate([[0],pre,[0]])
        for i in range(len(mp)-1,0,-1): mp[i-1]=max(mp[i-1],mp[i])
        k=np.where(mr[1:]!=mr[:-1])[0]; return float(np.sum((mr[k+1]-mr[k])*mp[k+1]))
    return {f'AP@{t}':round(ap(at[t],sc,ngt),3) for t in (0.3,0.5)}
def gt_items(paths): return [(p, str(GT_LAB/(os.path.basename(p)[:-4]+'.txt'))) for p in paths]
SEEDS=[0,1,2]; allres=[]
for seed in SEEDS:
    rng=np.random.RandomState(seed); res={'supervised':{}, 'semisl':{}, 'hybrid':{}}
    res['hybrid']['0']=evaluate(str(Y11P)); print(f'=== seed {seed} k=0 hybrid(base) {res["hybrid"]["0"]}',flush=True)
    for k in BUDGETS:
        lab=labeled(k,rng); print(f'=== seed {seed} budget k={k} ({len(lab)} crops) ===',flush=True)
        ys=make_ds(f's{seed}_k{k}_sup', gt_items(lab)); sp=train_micro('scratch',ys,f'sup_s{seed}_k{k}',seed,60,0.005); res['supervised'][str(k)]=evaluate(sp); print(f'  supervised {res["supervised"][str(k)]}',flush=True)
        yh=make_ds(f's{seed}_k{k}_hyb', gt_items(lab)); hp=train_micro(Y11P,yh,f'hyb_s{seed}_k{k}',seed,40,0.001); res['hybrid'][str(k)]=evaluate(hp); print(f'  hybrid {res["hybrid"][str(k)]}',flush=True)
        up=self_train(sp, str(RES/f'ssl_s{seed}_k{k}'), SSL_CONF); ysem=make_ds(f's{seed}_k{k}_sem', gt_items(lab)+up); ssp=train_micro('scratch',ysem,f'sem_s{seed}_k{k}',seed,60,0.005); res['semisl'][str(k)]=evaluate(ssp); print(f'  semisl {res["semisl"][str(k)]} (pseudo {len(up)})',flush=True)
    allres.append(res); json.dump({'per_run':allres}, open(RES/'labeleff_results.json','w'), indent=2)
agg={'supervised':{}, 'semisl':{}, 'hybrid':{}}
for arm in agg:
    for kk in allres[0][arm]:
        agg[arm][kk]={m:{'mean':round(float(np.mean([allres[s][arm][kk][m] for s in range(len(SEEDS))])),3),'std':round(float(np.std([allres[s][arm][kk][m] for s in range(len(SEEDS))])),3)} for m in allres[0][arm][kk]}
json.dump({'seeds':SEEDS,'per_run':allres,'agg':agg}, open(RES/'labeleff_results.json','w'), indent=2)
print('\n=== MicroYOLO label-efficiency (3-run mean±std, AP@0.3) ===')
for k in ['0']+[str(b) for b in BUDGETS]:
    row=' '.join(f"{arm}={agg[arm].get(k,{}).get('AP@0.3',{}).get('mean','-')}" for arm in ['supervised','semisl','hybrid'])
    print(f'  k={k:3s} {row}')
print('LABELEFF_MICRO_DONE',flush=True)
