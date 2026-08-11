# -*- coding: utf-8 -*-
"""Phase 2A switch-gate (3-seed): lock the YOLOv8n bootstrap numbers. Three retain rules, custom AP (comparable to
Table 4.8): D1 un-refined / naive-matched (global confidence threshold retention-matched to protected — the YOLO-fair
analog of MILCA conf>0.7, since literal 0.7 collapses on YOLO) / faintness-protected (single-signal, external
ThickDINO-MC). 3 seeds -> mean±std. Output -> y12_microyolo_bootstrap/results.json"""
import os, sys, json, glob, shutil
os.environ['XFORMERS_DISABLED']='1'; os.environ['PYTORCH_CUDA_ALLOC_CONF']='expandable_segments:True'
import numpy as np, torch, torch.nn as nn
import torchvision.transforms.functional as TF, torchvision.transforms as T
from PIL import Image; Image.MAX_IMAGE_PIXELS=None
from ultralytics import YOLO
sys.path.insert(0,'/root/autodl-tmp/A_Dissertation/references/RedDino/train'); sys.path.append('/root/A_Dissertation')
from common.paths import OUTPUTS, CHECKPOINTS
from dinov2.models.vision_transformer import vit_small
dev='cuda'; CROP=640; TILE=224; T_MC=15; MC_CAP=40
YO=OUTPUTS/'experiments'/'s5_yolo'; RES=OUTPUTS/'experiments'/'y12_microyolo_bootstrap'; RES.mkdir(parents=True,exist_ok=True)
FIN=json.load(open(OUTPUTS/'experiments'/'s3_filter'/'final.json'))['calib']; b_protect=FIN['b_protect']; a_star=FIN['a_star']
norm=T.Compose([T.ToTensor(),T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
ftm=vit_small(patch_size=16,img_size=224,block_chunks=0,init_values=1.0,drop_path_rate=0.1)
ftm.load_state_dict(torch.load(CHECKPOINTS/'ssl_vits_thickdino_ft_ibadan'/'encoder.pth',map_location='cpu'),strict=False); ftm=ftm.to(dev).eval()
fth=nn.Sequential(nn.Dropout(0.2),nn.Linear(384,1)).to(dev); fth.load_state_dict(torch.load(CHECKPOINTS/'ssl_vits_thickdino_ft_ibadan'/'head.pth',map_location='cpu')); fth.eval()
@torch.no_grad()
def mc_boxes(tiles,bs=48):
    for mod in list(ftm.modules())+list(fth.modules()):
        if isinstance(mod,nn.Dropout): mod.train()
    for mod in ftm.modules():
        if mod.__class__.__name__=='DropPath': mod.train()
    M=[];V=[]
    for i in range(0,len(tiles),bs):
        xb=torch.stack([norm(Image.fromarray(t)) for t in tiles[i:i+bs]]).to(dev); pr=[]
        for _ in range(T_MC): ff=ftm.forward_features(xb); pr.append(torch.sigmoid(fth(ff['x_norm_clstoken']).squeeze(-1)).cpu().numpy())
        A=np.stack(pr); M.append(A.mean(0)); V.append(A.var(0))
    ftm.eval(); fth.eval(); return (np.concatenate(M),np.concatenate(V)) if M else (np.array([]),np.array([]))
def yolo_predict(model, ip, conf=0.05):
    r=model.predict(ip, imgsz=CROP, conf=conf, verbose=False, device=0)[0]
    return r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()
def yolo_line(b): cx=(b[0]+b[2])/2/CROP; cy=(b[1]+b[3])/2/CROP; w=(b[2]-b[0])/CROP; h=(b[3]-b[1])/CROP; return f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"
train_imgs=sorted(glob.glob(str(YO/'ds_a0'/'images'/'train'/'*.jpg')))
def _cap(boxes,scores):
    if len(boxes)==0: return boxes,scores
    order=np.argsort(-scores)[:MC_CAP]; return boxes[order],scores[order]
def regen_protected(model, labdir):
    labdir.mkdir(parents=True,exist_ok=True); nlab=0
    for ip in train_imgs:
        arr=np.asarray(Image.open(ip).convert('RGB')); boxes,scores=_cap(*yolo_predict(model,ip)); base=os.path.basename(ip)[:-4]
        if len(boxes)==0: (labdir/(base+'.txt')).write_text(""); continue
        tiles=[]
        for b in boxes:
            cx=(b[0]+b[2])/2; cy=(b[1]+b[3])/2; x0=int(min(max(cx-TILE/2,0),CROP-TILE)); y0=int(min(max(cy-TILE/2,0),CROP-TILE)); tiles.append(arr[y0:y0+TILE,x0:x0+TILE])
        pm,pv=mc_boxes(tiles); sel=[j for j in range(len(boxes)) if (pv[j]>=b_protect or pm[j]>=a_star)]
        (labdir/(base+'.txt')).write_text("\n".join(yolo_line(boxes[j]) for j in sel)); nlab+=len(sel)
    return nlab
def regen_naive_matched(model, labdir, N):
    labdir.mkdir(parents=True,exist_ok=True); cache={}; alls=[]
    for ip in train_imgs:
        boxes,scores=_cap(*yolo_predict(model,ip)); cache[ip]=(boxes,scores); alls.extend(scores.tolist())
    alls=np.sort(np.array(alls))[::-1]; thr=float(alls[min(N,len(alls))-1]) if (N>0 and len(alls)>0) else 1.1; nlab=0
    for ip,(boxes,scores) in cache.items():
        base=os.path.basename(ip)[:-4]; sel=[j for j in range(len(boxes)) if scores[j]>=thr]
        (labdir/(base+'.txt')).write_text("\n".join(yolo_line(boxes[j]) for j in sel)); nlab+=len(sel)
    return nlab, round(thr,4)
def _populate(srcdir, dstdir, ext, copy=False):
    for f in glob.glob(str(srcdir)+f'/*.{ext}'):
        d=str(dstdir/os.path.basename(f))
        if copy: shutil.copy(f,d)
        else:
            try: os.link(f,d)
            except OSError: shutil.copy(f,d)
def make_ds(name, labdir):
    root=RES/name
    for sub in ['images/train','images/val','labels/train','labels/val']:
        d=root/sub
        if d.exists(): shutil.rmtree(d)
        d.mkdir(parents=True,exist_ok=True)
    _populate(YO/'ds_a0'/'images'/'train', root/'images'/'train', 'jpg')
    _populate(YO/'ds_a0'/'images'/'val',   root/'images'/'val',   'jpg')
    _populate(labdir,                       root/'labels'/'train', 'txt', copy=True)
    _populate(YO/'ds_a0'/'labels'/'val',   root/'labels'/'val',   'txt', copy=True)
    for c in glob.glob(str(root)+'/labels/*.cache'): os.remove(c)
    yaml=root/'data.yaml'; yaml.write_text(f"path: {root}\ntrain: images/train\nval: images/val\nnc: 1\nnames: ['parasite']\n"); return str(yaml)
val_imgs=sorted(glob.glob(str(YO/'ds_a0'/'images'/'val'/'*.jpg'))); VAL_LAB=YO/'ds_a0'/'labels'/'val'
def load_gt(ip):
    lp=VAL_LAB/(os.path.basename(ip)[:-4]+'.txt'); bx=[]
    if lp.exists():
        for ln in lp.read_text().splitlines():
            f=ln.split()
            if len(f)==5: _,cx,cy,w,h=f; cx,cy,w,h=float(cx)*CROP,float(cy)*CROP,float(w)*CROP,float(h)*CROP; bx.append([cx-w/2,cy-h/2,cx+w/2,cy+h/2])
    return bx
def iou(a,b):
    ix1,iy1=max(a[0],b[0]),max(a[1],b[1]); ix2,iy2=min(a[2],b[2]),min(a[3],b[3]); iw,ih=max(0,ix2-ix1),max(0,iy2-iy1); it=iw*ih
    ua=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-it; return it/ua if ua>0 else 0.0
gts=[load_gt(ip) for ip in val_imgs]; cropgt=[len(g) for g in gts]; med=np.median([c for c in cropgt if c>0]) if any(cropgt) else 0
def evaluate(model):
    at={0.3:[],0.5:[]}; sc=[]; ngt=0; btp={'low':[],'high':[]}; bsc={'low':[],'high':[]}; bgt={'low':0,'high':0}
    for idx,ip in enumerate(val_imgs):
        gt=gts[idx]; ngt+=len(gt); bd='low' if cropgt[idx]<=med else 'high'; bgt[bd]+=len(gt)
        bxs,scs=yolo_predict(model,ip,conf=0.05); order=np.argsort(-scs)
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
                if th==0.3: sc.append(float(scs[j])); btp[bd].append(tp); bsc[bd].append(float(scs[j]))
    def ap(tps,scs,ng):
        if not tps: return 0.0
        o=np.argsort(-np.array(scs)); tps=np.array(tps)[o]; ctp=np.cumsum(tps); cfp=np.cumsum(1-tps); rec=ctp/max(1,ng); pre=ctp/np.maximum(1,ctp+cfp)
        mr=np.concatenate([[0],rec,[1]]); mp=np.concatenate([[0],pre,[0]])
        for i in range(len(mp)-1,0,-1): mp[i-1]=max(mp[i-1],mp[i])
        k=np.where(mr[1:]!=mr[:-1])[0]; return float(np.sum((mr[k+1]-mr[k])*mp[k+1]))
    return dict(**{f'AP@{t}':round(ap(at[t],sc,ngt),3) for t in (0.3,0.5)}, AP03_low=round(ap(btp['low'],bsc['low'],bgt['low']),3), AP03_high=round(ap(btp['high'],bsc['high'],bgt['high']),3))
def train_yolo(yaml, name, seed):
    m=YOLO('yolov8n-p2.yaml')
    try: m=m.load('yolov8n.pt')
    except Exception: pass
    m.train(data=yaml, epochs=80, imgsz=CROP, batch=16, box=9.0, device=0, workers=8,
        project=str(RES/'runs'), name=name, exist_ok=True, verbose=False, plots=False, seed=seed); return m
runs={'D1':[], 'naive_matched':[], 'protected':[]}
for seed in [0,1,2]:
    print(f'######## SEED {seed} ########',flush=True)
    yaml_a0=make_ds('ds_base', YO/'ds_a0'/'labels'/'train'); d1=train_yolo(yaml_a0,f'd1_s{seed}',seed)
    r1=evaluate(d1); runs['D1'].append(r1); print(f'  [s{seed}] D1 {r1}',flush=True)
    n_prot=regen_protected(d1, RES/'lab_prot'); yp=make_ds('ds_prot', RES/'lab_prot'); d2p=train_yolo(yp,f'd2p_s{seed}',seed)
    rp=dict(n=n_prot,**evaluate(d2p)); runs['protected'].append(rp); print(f'  [s{seed}] protected {rp}',flush=True)
    n_nv,thr=regen_naive_matched(d1, RES/'lab_naive', n_prot); yn=make_ds('ds_naive', RES/'lab_naive'); d2n=train_yolo(yn,f'd2n_s{seed}',seed)
    rn=dict(n=n_nv,thr=thr,**evaluate(d2n)); runs['naive_matched'].append(rn); print(f'  [s{seed}] naive_matched {rn}',flush=True)
    json.dump(runs,open(RES/'results.json','w'),indent=2)
agg={}
for k in runs:
    ks=[x for x in runs[k][0].keys() if x.startswith('AP')]
    agg[k]={m:{'mean':round(float(np.mean([r[m] for r in runs[k]])),3),'std':round(float(np.std([r[m] for r in runs[k]])),3)} for m in ks}
json.dump(dict(agg=agg,runs=runs,ref_retinanet={'D1_AP@0.5':0.125,'naive_AP@0.5':0.134,'protected_AP@0.5':0.206}),open(RES/'results.json','w'),indent=2)
print('\n=== MicroYOLO bootstrap 3-seed (mean±std, custom AP) ===')
for k in runs: print(f'  {k:14s} {agg[k]}')
print('Y12_DONE',flush=True)
