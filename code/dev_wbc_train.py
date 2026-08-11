# -*- coding: utf-8 -*-
"""BLOCK B step 1: train a WBC detector on Lacuna WBC boxes (class 1). WBC are large (~200px) and stain-dark,
so far easier + more site-robust than parasites. Saves wbc_retinanet.pth for parasitaemia = 8000*(MP/WBC)."""
import os, sys, glob, random, shutil
import numpy as np, torch
from pathlib import Path
from PIL import Image; Image.MAX_IMAGE_PIXELS=None
import torchvision.transforms.functional as TF
from torchvision.models.detection import retinanet_resnet50_fpn
sys.path.append('/root/A_Dissertation'); from common.paths import OUTPUTS
dev='cuda'; torch.manual_seed(0); np.random.seed(0); random.seed(0); CROP=640; BATCH=6
LAC=Path('/root/autodl-tmp/A_Dissertation/data/raw/lacuna/Ghana/Thick')
DS=OUTPUTS/'experiments'/'ds_lacuna_wbc'; RES=OUTPUTS/'experiments'/'s6_retinanet'

def make_ds(nimg=360, crops_per=10, val_frac=0.3):
    if (DS/'images'/'val').exists() and len(glob.glob(str(DS/'images'/'val'/'*.jpg')))>400:
        print('reuse ds_lacuna_wbc',flush=True); return
    if DS.exists(): shutil.rmtree(DS)
    for sp in ('train','val'):
        (DS/'images'/sp).mkdir(parents=True,exist_ok=True); (DS/'labels'/sp).mkdir(parents=True,exist_ok=True)
    imgs=sorted(glob.glob(str(LAC/'images'/'*.jpg')), key=lambda p:int(os.path.basename(p)[:-4])); random.Random(2).shuffle(imgs); imgs=imgs[:nimg]
    nval=int(nimg*val_frac); ntr=nva=0
    for idx,ip in enumerate(imgs):
        sp='val' if idx<nval else 'train'; lp=LAC/'labels_yolo'/(os.path.basename(ip)[:-4]+'.txt')
        if not lp.exists(): continue
        im=Image.open(ip).convert('RGB'); W,H=im.size; boxes=[]
        for ln in lp.read_text().splitlines():
            f=ln.split()
            if len(f)==5 and f[0]=='1':                                 # class 1 = WBC
                boxes.append((float(f[1])*W,float(f[2])*H,float(f[3])*W,float(f[4])*H))
        if not boxes: continue
        for c in range(crops_per):
            x0=random.randint(0,W-CROP); y0=random.randint(0,H-CROP); lab=[]
            for (cx,cy,w,h) in boxes:
                if x0<=cx<x0+CROP and y0<=cy<y0+CROP and w*CROP/W>=1 and h*CROP/H>=1:
                    lab.append(f'0 {(cx-x0)/CROP:.6f} {(cy-y0)/CROP:.6f} {w/CROP:.6f} {h/CROP:.6f}')
            if not lab: continue
            im.crop((x0,y0,x0+CROP,y0+CROP)).save(DS/'images'/sp/f'{idx}_{c}.jpg',quality=90)
            (DS/'labels'/sp/f'{idx}_{c}.txt').write_text('\n'.join(lab)); ntr+=sp=='train'; nva+=sp=='val'
    print(f'ds_lacuna_wbc: train {ntr} val {nva}',flush=True)

class DetDS(torch.utils.data.Dataset):
    def __init__(s,root,split): s.imgs=sorted(glob.glob(str(root/'images'/split/'*.jpg')))
    def __len__(s): return len(s.imgs)
    def __getitem__(s,i):
        ip=s.imgs[i]; x=TF.to_tensor(Image.open(ip).convert('RGB'))
        lp=ip.replace('/images/','/labels/').replace('.jpg','.txt'); bx=[]
        if os.path.exists(lp):
            for ln in open(lp).read().splitlines():
                f=ln.split()
                if len(f)==5:
                    _,cx,cy,w,h=f; cx,cy,w,h=float(cx)*CROP,float(cy)*CROP,float(w)*CROP,float(h)*CROP
                    if w>=1 and h>=1: bx.append([cx-w/2,cy-h/2,cx+w/2,cy+h/2])
        bx=torch.tensor(bx,dtype=torch.float32).reshape(-1,4)
        return x,{'boxes':bx,'labels':torch.ones(len(bx),dtype=torch.int64)}
def collate(b): return tuple(zip(*b))
def iou(a,b):
    ix1,iy1=max(a[0],b[0]),max(a[1],b[1]); ix2,iy2=min(a[2],b[2]),min(a[3],b[3]); iw,ih=max(0,ix2-ix1),max(0,iy2-iy1); it=iw*ih
    ua=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-it; return it/ua if ua>0 else 0.0
@torch.no_grad()
def evaluate(model,root):
    dl=torch.utils.data.DataLoader(DetDS(root,'val'),batch_size=BATCH,shuffle=False,collate_fn=collate,num_workers=6)
    model.eval(); tp={0.3:[],0.5:[]}; sc=[]; ng=0
    for imgs,tgts in dl:
        for pred,tgt in zip(model([im.to(dev) for im in imgs]),tgts):
            gt=tgt['boxes'].numpy().tolist(); ng+=len(gt); bxs=pred['boxes'].cpu().numpy(); scs=pred['scores'].cpu().numpy(); order=np.argsort(-scs)
            for th in (0.3,0.5):
                mt=set()
                for j in order:
                    b=bxs[j]; best=-1; bj=-1
                    for jg,g in enumerate(gt):
                        if jg in mt: continue
                        v=iou(b,g)
                        if v>best: best,bj=v,jg
                    t=1 if best>=th and bj>=0 else 0
                    if t: mt.add(bj)
                    tp[th].append(t)
                    if th==0.3: sc.append(float(scs[j]))
    def ap(tps,scs,n):
        if not tps: return 0.0
        o=np.argsort(-np.array(scs)); tps=np.array(tps)[o]; ctp=np.cumsum(tps); cfp=np.cumsum(1-tps); rec=ctp/max(1,n); pre=ctp/np.maximum(1,ctp+cfp)
        mr=np.concatenate([[0],rec,[1]]); mp=np.concatenate([[0],pre,[0]])
        for i in range(len(mp)-1,0,-1): mp[i-1]=max(mp[i-1],mp[i])
        idx=np.where(mr[1:]!=mr[:-1])[0]; return float(np.sum((mr[idx+1]-mr[idx])*mp[idx+1]))
    return {f'AP@{t}':round(ap(tp[t],sc,ng),3) for t in (0.3,0.5)}

make_ds()
m=retinanet_resnet50_fpn(num_classes=2,weights=None,weights_backbone='IMAGENET1K_V1').to(dev)
dl=torch.utils.data.DataLoader(DetDS(DS,'train'),batch_size=BATCH,shuffle=True,collate_fn=collate,num_workers=6)
params=[p for p in m.parameters() if p.requires_grad]; opt=torch.optim.SGD(params,lr=0.005,momentum=0.9,weight_decay=1e-4)
sched=torch.optim.lr_scheduler.MultiStepLR(opt,milestones=[18,25],gamma=0.1); EPOCHS=30; WARMUP=400; it=0
for ep in range(EPOCHS):
    m.train(); tot=0
    for imgs,tgts in dl:
        if it<WARMUP:
            for g in opt.param_groups: g['lr']=0.005*(0.01+0.99*it/WARMUP)
        imgs=[im.to(dev) for im in imgs]; tgts=[{k:v.to(dev) for k,v in t.items()} for t in tgts]
        ld=m(imgs,tgts); loss=sum(ld.values()); opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(params,5.0); opt.step(); it+=1; tot+=float(loss)
    if it>=WARMUP: sched.step()
    if (ep+1)%5==0: print(f'  ep{ep+1}/{EPOCHS} loss {tot/max(1,len(dl)):.3f}',flush=True)
torch.save(m.state_dict(), RES/'wbc_retinanet.pth')
print('in-Lacuna WBC AP ->', evaluate(m,DS),flush=True)
print('SAVED', RES/'wbc_retinanet.pth','\nDONE',flush=True)
