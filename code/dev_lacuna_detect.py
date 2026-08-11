# -*- coding: utf-8 -*-
"""BLOCK A (detection): cross-site parasite detection on Lacuna (Ghana). Take the domain-A-trained RetinaNet
detectors (A0 / faintness-protected / GT-supervised) and evaluate them ZERO-SHOT on Ghana Lacuna 640-crops,
vs an in-Lacuna RetinaNet upper bound. Mirrors §4.3 evaluate(); directly comparable to the Ghana MIDL-2026
paper's finding that detectors overfit to acquisition and fail cross-site."""
import os, sys, glob, json, random
import numpy as np, torch
from PIL import Image; Image.MAX_IMAGE_PIXELS=None
import torchvision.transforms.functional as TF
from torchvision.models.detection import retinanet_resnet50_fpn
sys.path.append('/root/A_Dissertation'); from common.paths import OUTPUTS
dev='cuda'; torch.manual_seed(0); np.random.seed(0); random.seed(0)
CROP=640; BATCH=6
S6=OUTPUTS/'experiments'/'s6_retinanet'
LAC=__import__('pathlib').Path('/root/autodl-tmp/A_Dissertation/data/raw/lacuna/Ghana/Thick')
DS=OUTPUTS/'experiments'/'ds_lacuna'

# ---------- build Lacuna 640-crop detection dataset (parasite class 0 only) ----------
def make_ds(nimg=360, crops_per=12, val_frac=0.3):
    if (DS/'images'/'val').exists() and len(glob.glob(str(DS/'images'/'val'/'*.jpg')))>500:
        print('reuse ds_lacuna',flush=True); return
    for sp in ('train','val'):
        (DS/'images'/sp).mkdir(parents=True,exist_ok=True); (DS/'labels'/sp).mkdir(parents=True,exist_ok=True)
    imgs=sorted(glob.glob(str(LAC/'images'/'*.jpg')), key=lambda p:int(os.path.basename(p)[:-4])); random.Random(2).shuffle(imgs); imgs=imgs[:nimg]
    nval=int(nimg*val_frac); ntr=0; nva=0
    for idx,ip in enumerate(imgs):
        sp='val' if idx<nval else 'train'
        lp=LAC/'labels_yolo'/(os.path.basename(ip)[:-4]+'.txt')
        if not lp.exists(): continue
        im=Image.open(ip).convert('RGB'); W,H=im.size; boxes=[]
        for ln in lp.read_text().splitlines():
            f=ln.split()
            if len(f)==5 and f[0]=='0':                        # parasite only
                boxes.append((float(f[1])*W,float(f[2])*H,float(f[3])*W,float(f[4])*H))
        if not boxes: continue
        for c in range(crops_per):
            x0=random.randint(0,W-CROP); y0=random.randint(0,H-CROP); lab=[]
            for (cx,cy,w,h) in boxes:
                if x0<=cx<x0+CROP and y0<=cy<y0+CROP:
                    lab.append(f'0 {(cx-x0)/CROP:.6f} {(cy-y0)/CROP:.6f} {w/CROP:.6f} {h/CROP:.6f}')
            if not lab: continue
            im.crop((x0,y0,x0+CROP,y0+CROP)).save(DS/'images'/sp/f'{idx}_{c}.jpg',quality=90)
            (DS/'labels'/sp/f'{idx}_{c}.txt').write_text('\n'.join(lab))
            ntr+=sp=='train'; nva+=sp=='val'
    print(f'ds_lacuna built: train {ntr} crops, val {nva} crops',flush=True)

class DetDS(torch.utils.data.Dataset):
    def __init__(self, root, split): self.imgs=sorted(glob.glob(str(root/'images'/split/'*.jpg')))
    def __len__(self): return len(self.imgs)
    def __getitem__(self,i):
        ip=self.imgs[i]; x=TF.to_tensor(Image.open(ip).convert('RGB'))
        lp=ip.replace('/images/','/labels/').replace('.jpg','.txt'); boxes=[]
        if os.path.exists(lp):
            for ln in open(lp).read().splitlines():
                f=ln.split()
                if len(f)==5:
                    _,cx,cy,w,h=f; cx,cy,w,h=float(cx)*CROP,float(cy)*CROP,float(w)*CROP,float(h)*CROP
                    if w>=1 and h>=1: boxes.append([cx-w/2,cy-h/2,cx+w/2,cy+h/2])   # drop degenerate boxes
        boxes=torch.tensor(boxes,dtype=torch.float32).reshape(-1,4)
        return x,{'boxes':boxes,'labels':torch.ones(len(boxes),dtype=torch.int64)}
def collate(b): return tuple(zip(*b))
def iou(a,b):
    ix1,iy1=max(a[0],b[0]),max(a[1],b[1]); ix2,iy2=min(a[2],b[2]),min(a[3],b[3]); iw,ih=max(0,ix2-ix1),max(0,iy2-iy1); it=iw*ih
    ua=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-it; return it/ua if ua>0 else 0.0
@torch.no_grad()
def evaluate(model, root):
    dl=torch.utils.data.DataLoader(DetDS(root,'val'),batch_size=BATCH,shuffle=False,collate_fn=collate,num_workers=6)
    model.eval(); all_tp={0.3:[],0.5:[]}; sc=[]; n_gt=0
    for imgs,tgts in dl:
        preds=model([im.to(dev) for im in imgs])
        for pred,tgt in zip(preds,tgts):
            gt=tgt['boxes'].numpy().tolist(); n_gt+=len(gt)
            boxes=pred['boxes'].cpu().numpy(); scores=pred['scores'].cpu().numpy(); order=np.argsort(-scores)
            for iouth in (0.3,0.5):
                matched=set()
                for j in order:
                    b=boxes[j]; best=-1; bj=-1
                    for jg,g in enumerate(gt):
                        if jg in matched: continue
                        v=iou(b,g);
                        if v>best: best,bj=v,jg
                    tp=1 if best>=iouth and bj>=0 else 0
                    if tp: matched.add(bj)
                    all_tp[iouth].append(tp)
                    if iouth==0.3: sc.append(float(scores[j]))
    def ap(tps,scs,ng):
        if not tps: return 0.0
        o=np.argsort(-np.array(scs)); tps=np.array(tps)[o]; ctp=np.cumsum(tps); cfp=np.cumsum(1-tps); rec=ctp/max(1,ng); pre=ctp/np.maximum(1,ctp+cfp)
        mr=np.concatenate([[0],rec,[1]]); mp=np.concatenate([[0],pre,[0]])
        for i in range(len(mp)-1,0,-1): mp[i-1]=max(mp[i-1],mp[i])
        idx=np.where(mr[1:]!=mr[:-1])[0]; return float(np.sum((mr[idx+1]-mr[idx])*mp[idx+1]))
    return {f'AP@{t}':round(ap(all_tp[t],sc,n_gt),3) for t in (0.3,0.5)}
def build():
    return retinanet_resnet50_fpn(num_classes=2, weights=None, weights_backbone='IMAGENET1K_V1').to(dev)

make_ds()
# ---- cross-site: A-trained detectors zero-shot on Lacuna ----
print('\n=== cross-site A-trained -> Lacuna (zero-shot) ===',flush=True)
fastmal=json.load(open(S6/'retinanet_results.json')) if (S6/'retinanet_results.json').exists() else {}
for tag in ['a0','protected','gtsup']:
    m=build(); m.load_state_dict(torch.load(S6/f'retinanet_r50_{tag}.pth',map_location='cpu')); m.eval()
    r=evaluate(m, DS); fm=fastmal.get(f'retinanet_r50|{tag}',{})
    print(f'  {tag:10s} Lacuna {r}   | FASTMAL-val AP@0.5 {fm.get("AP@0.5","?")} AP@0.3 {fm.get("AP@0.3","?")}',flush=True)
    del m; torch.cuda.empty_cache()

# ---- in-Lacuna upper bound (train on Lacuna train split) ----
print('\n=== in-Lacuna RetinaNet (upper bound) ===',flush=True)
EPOCHS=50; BASE_LR=0.005; WARMUP=500
m=build(); dl=torch.utils.data.DataLoader(DetDS(DS,'train'),batch_size=BATCH,shuffle=True,collate_fn=collate,num_workers=6)
params=[p for p in m.parameters() if p.requires_grad]; opt=torch.optim.SGD(params,lr=BASE_LR,momentum=0.9,weight_decay=1e-4)
sched=torch.optim.lr_scheduler.MultiStepLR(opt,milestones=[30,42],gamma=0.1); it=0
for ep in range(EPOCHS):
    m.train(); tot=0
    for imgs,tgts in dl:
        if it<WARMUP:
            for g in opt.param_groups: g['lr']=BASE_LR*(0.01+0.99*it/WARMUP)
        imgs=[im.to(dev) for im in imgs]; tgts=[{k:v.to(dev) for k,v in t.items()} for t in tgts]
        ld=m(imgs,tgts); loss=sum(ld.values()); opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(params,5.0); opt.step(); it+=1; tot+=float(loss)
    if it>=WARMUP: sched.step()
    if (ep+1)%5==0: print(f'    ep{ep+1}/{EPOCHS} loss {tot/max(1,len(dl)):.3f}',flush=True)
print('  in-Lacuna ->', evaluate(m, DS),flush=True)
print('\nDONE',flush=True)
