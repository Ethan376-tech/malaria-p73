# -*- coding: utf-8 -*-
"""Matched 3-seed A0 baseline (no HNM), same band eval as build_hnm_train.py, for a FAIR comparison vs A0+HNM.
Output -> s6_retinanet/notyp_3seed_results.json"""
import os, sys, json, glob
os.environ['XFORMERS_DISABLED']='1'; os.environ['PYTORCH_CUDA_ALLOC_CONF']='expandable_segments:True'
import numpy as np, torch
from PIL import Image; Image.MAX_IMAGE_PIXELS=None
import torchvision.transforms.functional as TF
from torchvision.models.detection import retinanet_resnet50_fpn
sys.path.append('/root/A_Dissertation'); from common.paths import OUTPUTS
dev='cuda'; YO=OUTPUTS/'experiments'/'s5_yolo'; RES=OUTPUTS/'experiments'/'s6_retinanet'
EPOCHS=50; BASE_LR=0.005; WARMUP=500; BATCH=6; CROP=640
class DetDS(torch.utils.data.Dataset):
    def __init__(self, root, split): self.imgs=sorted(glob.glob(str(root/'images'/split/'*.jpg')))
    def __len__(self): return len(self.imgs)
    def __getitem__(self,i):
        ip=self.imgs[i]; x=TF.to_tensor(Image.open(ip).convert('RGB')); lp=ip.replace('/images/','/labels/').replace('.jpg','.txt'); bx=[]
        if os.path.exists(lp):
            for ln in open(lp).read().splitlines():
                f=ln.split()
                if len(f)==5: _,cx,cy,w,h=f; cx,cy,w,h=float(cx)*CROP,float(cy)*CROP,float(w)*CROP,float(h)*CROP; bx.append([cx-w/2,cy-h/2,cx+w/2,cy+h/2])
        bx=torch.tensor(bx,dtype=torch.float32).reshape(-1,4); return x,{'boxes':bx,'labels':torch.ones(len(bx),dtype=torch.int64)}
def collate(b): return tuple(zip(*b))
def build(): return retinanet_resnet50_fpn(num_classes=2,weights=None,weights_backbone='IMAGENET1K_V1').to(dev)
def train(model, root):
    dl=torch.utils.data.DataLoader(DetDS(root,'train'),batch_size=BATCH,shuffle=True,collate_fn=collate,num_workers=6)
    params=[p for p in model.parameters() if p.requires_grad]; opt=torch.optim.SGD(params,lr=BASE_LR,momentum=0.9,weight_decay=1e-4)
    sched=torch.optim.lr_scheduler.MultiStepLR(opt,milestones=[30,42],gamma=0.1); it=0
    for ep in range(EPOCHS):
        model.train()
        for imgs,tgts in dl:
            if it<WARMUP:
                f=it/WARMUP
                for g in opt.param_groups: g['lr']=BASE_LR*(0.01+0.99*f)
            imgs=[im.to(dev) for im in imgs]; tgts=[{k:v.to(dev) for k,v in t.items()} for t in tgts]
            loss=sum(model(imgs,tgts).values()); opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(params,5.0); opt.step(); it+=1
        if it>=WARMUP: sched.step()
        if (ep+1)%10==0: print(f'  ep{ep+1}/{EPOCHS}',flush=True)
    return model
def iou(a,b):
    ix1,iy1=max(a[0],b[0]),max(a[1],b[1]); ix2,iy2=min(a[2],b[2]),min(a[3],b[3]); iw,ih=max(0,ix2-ix1),max(0,iy2-iy1); it=iw*ih
    ua=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-it; return it/ua if ua>0 else 0.0
@torch.no_grad()
def evaluate(model, root):
    ds=DetDS(root,'val'); model.eval(); cropgt=[len(ds[i][1]['boxes']) for i in range(len(ds))]; med=np.median([c for c in cropgt if c>0]) if any(cropgt) else 0
    all_tp={0.3:[],0.5:[]}; sc=[]; ngt=0; btp={'low':[],'high':[]}; bsc={'low':[],'high':[]}; bgt={'low':0,'high':0}
    dl=torch.utils.data.DataLoader(ds,batch_size=BATCH,shuffle=False,collate_fn=collate,num_workers=6); idx=0
    for imgs,tgts in dl:
        preds=model([im.to(dev) for im in imgs])
        for pred,tgt in zip(preds,tgts):
            gt=tgt['boxes'].numpy().tolist(); ngt+=len(gt); bd='low' if cropgt[idx]<=med else 'high'; bgt[bd]+=len(gt); idx+=1
            bxs=pred['boxes'].cpu().numpy(); scs=pred['scores'].cpu().numpy(); order=np.argsort(-scs)
            for iouth in (0.3,0.5):
                matched=set()
                for j in order:
                    b=bxs[j]; best=-1;bj=-1
                    for jg,g in enumerate(gt):
                        if jg in matched: continue
                        v=iou(b,g)
                        if v>best: best,bj=v,jg
                    tp=1 if best>=iouth and bj>=0 else 0
                    if tp: matched.add(bj)
                    all_tp[iouth].append(tp)
                    if iouth==0.3: sc.append(float(scs[j])); btp[bd].append(tp); bsc[bd].append(float(scs[j]))
    def ap(tps,scs,ng):
        if not tps: return 0.0
        o=np.argsort(-np.array(scs)); tps=np.array(tps)[o]; ctp=np.cumsum(tps); cfp=np.cumsum(1-tps); rec=ctp/max(1,ng); pre=ctp/np.maximum(1,ctp+cfp)
        mr=np.concatenate([[0],rec,[1]]); mp=np.concatenate([[0],pre,[0]])
        for i in range(len(mp)-1,0,-1): mp[i-1]=max(mp[i-1],mp[i])
        idx=np.where(mr[1:]!=mr[:-1])[0]; return float(np.sum((mr[idx+1]-mr[idx])*mp[idx+1]))
    return dict(**{f'AP@{t}':round(ap(all_tp[t],sc,ngt),3) for t in (0.3,0.5)},
                AP03_low=round(ap(btp['low'],bsc['low'],bgt['low']),3), AP03_high=round(ap(btp['high'],bsc['high'],bgt['high']),3))
SEEDS=[0,1,2]; runs=[]
for seed in SEEDS:
    torch.manual_seed(seed); np.random.seed(seed)
    print(f'=== seed {seed}: train notyp (protect-only, no typicality) ===',flush=True); m=train(build(), YO/"ds_sweep_notyp"); r=evaluate(m, YO/'ds_a0'); runs.append(r); print('  ->',r,flush=True)
agg={k:{'mean':round(float(np.mean([r[k] for r in runs])),3),'std':round(float(np.std([r[k] for r in runs])),3)} for k in runs[0]}
json.dump({'notyp_3seed':agg,'runs':runs},open(RES/'notyp_3seed_results.json','w'),indent=2)
print('\n=== notyp / protect-only (matched 3-seed) ===\n ',agg); print('NOTYP_3SEED_DONE',flush=True)
