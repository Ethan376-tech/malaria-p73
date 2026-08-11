# -*- coding: utf-8 -*-
"""WS3 — resolution-aware detection. Domain B (Chittagong) parasites are ~19px; the default RetinaNet (min anchor 32px,
min_size 800) cannot match them -> even the GT-supervised detector gives AP=0 (build_s8_chitt2). Engineering fix:
custom SMALL anchors and/or higher input resolution (upscale). Reuses the cached GT-supervised crops (train = C1 GT
boxes, val = C2 GT boxes, 640px native-res). Compares configs; question: does AP move off 0?
Output -> outputs/experiments/ws3_resdet/results.json
"""
import os, sys, json, glob
os.environ['XFORMERS_DISABLED']='1'
sys.path.append('/root/A_Dissertation')
import numpy as np, torch
import torchvision.transforms.functional as TF
from PIL import Image; Image.MAX_IMAGE_PIXELS=None
from common.paths import OUTPUTS
from torchvision.models.detection import retinanet_resnet50_fpn
from torchvision.models.detection.retinanet import RetinaNetHead
from torchvision.models.detection.anchor_utils import AnchorGenerator
dev='cuda'; CROP=640; EPOCHS=30; BASE_LR=0.005; WARMUP=400; BATCH=6
DG=OUTPUTS/'experiments'/'s8_chitt2'/'ds_gtsup'                      # cached: train=C1 GT boxes, val=C2 GT boxes
RES=OUTPUTS/'experiments'/'ws3_resdet'; RES.mkdir(parents=True,exist_ok=True)

class DetDS(torch.utils.data.Dataset):
    def __init__(self,root,split): self.imgs=sorted(glob.glob(str(root/'images'/split/'*.jpg')))
    def __len__(self): return len(self.imgs)
    def __getitem__(self,i):
        ip=self.imgs[i]; x=TF.to_tensor(Image.open(ip).convert('RGB')); lp=ip.replace('/images/','/labels/').replace('.jpg','.txt'); bx=[]
        if os.path.exists(lp):
            for ln in open(lp).read().splitlines():
                f=ln.split()
                if len(f)==5: _,cx,cy,w,h=f; cx,cy,w,h=float(cx)*CROP,float(cy)*CROP,float(w)*CROP,float(h)*CROP; bx.append([cx-w/2,cy-h/2,cx+w/2,cy+h/2])
        bx=torch.tensor(bx,dtype=torch.float32).reshape(-1,4)
        if len(bx):
            bx[:,[0,2]]=bx[:,[0,2]].clamp(0,CROP); bx[:,[1,3]]=bx[:,[1,3]].clamp(0,CROP)
            keep=(bx[:,2]-bx[:,0]>=1)&(bx[:,3]-bx[:,1]>=1); bx=bx[keep]
        return x,{'boxes':bx,'labels':torch.ones(len(bx),dtype=torch.int64)}
def collate(b): return tuple(zip(*b))

def build_model(cfg):
    m=retinanet_resnet50_fpn(num_classes=2,weights=None,weights_backbone='IMAGENET1K_V1')
    if cfg['anchors']=='small':
        sizes=tuple((s,int(s*1.26),int(s*1.587)) for s in (8,16,32,64,128))   # shifted down from default 32..512
        ag=AnchorGenerator(sizes=sizes, aspect_ratios=((0.5,1.0,2.0),)*len(sizes))
        m.anchor_generator=ag; m.head=RetinaNetHead(256, ag.num_anchors_per_location()[0], 2)
    m.transform.min_size=(cfg['min_size'],); m.transform.max_size=cfg['max_size']
    return m.to(dev)

def train(model,root):
    dl=torch.utils.data.DataLoader(DetDS(root,'train'),batch_size=BATCH,shuffle=True,collate_fn=collate,num_workers=6)
    params=[p for p in model.parameters() if p.requires_grad]; opt=torch.optim.SGD(params,lr=BASE_LR,momentum=0.9,weight_decay=1e-4)
    sched=torch.optim.lr_scheduler.MultiStepLR(opt,milestones=[20,27],gamma=0.1); it=0
    for ep in range(EPOCHS):
        model.train()
        for imgs,tgts in dl:
            if it<WARMUP:
                f=it/WARMUP
                for g in opt.param_groups: g['lr']=BASE_LR*(0.01+0.99*f)
            imgs=[im.to(dev) for im in imgs]; tgts=[{k:v.to(dev) for k,v in t.items()} for t in tgts]
            loss=sum(model(imgs,tgts).values()); opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(params,5.0); opt.step(); it+=1
        if it>=WARMUP: sched.step()
    return model
def iou(a,b):
    ix1,iy1=max(a[0],b[0]),max(a[1],b[1]); ix2,iy2=min(a[2],b[2]),min(a[3],b[3]); iw,ih=max(0,ix2-ix1),max(0,iy2-iy1); inr=iw*ih
    ua=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inr; return inr/ua if ua>0 else 0.0
@torch.no_grad()
def evaluate(model,root):
    dl=torch.utils.data.DataLoader(DetDS(root,'val'),batch_size=BATCH,shuffle=False,collate_fn=collate,num_workers=6); model.eval()
    all_tp={0.3:[],0.5:[]}; sc=[]; ngt=0
    for imgs,tgts in dl:
        preds=model([im.to(dev) for im in imgs])
        for pred,tgt in zip(preds,tgts):
            gt=tgt['boxes'].numpy().tolist(); ngt+=len(gt); bxs=pred['boxes'].cpu().numpy(); scs=pred['scores'].cpu().numpy(); order=np.argsort(-scs)
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
                    if iouth==0.3: sc.append(float(scs[j]))
    def ap(tps,scs,ng):
        if not tps: return 0.0
        o=np.argsort(-np.array(scs)); tps=np.array(tps)[o]; ctp=np.cumsum(tps); cfp=np.cumsum(1-tps); rec=ctp/max(1,ng); pre=ctp/np.maximum(1,ctp+cfp)
        mr=np.concatenate([[0],rec,[1]]); mp=np.concatenate([[0],pre,[0]])
        for i in range(len(mp)-1,0,-1): mp[i-1]=max(mp[i-1],mp[i])
        idx=np.where(mr[1:]!=mr[:-1])[0]; return float(np.sum((mr[idx+1]-mr[idx])*mp[idx+1]))
    return dict(**{f'AP@{t}':round(ap(all_tp[t],sc,ngt),3) for t in (0.3,0.5)}, n_gt=ngt)

CONFIGS={
 'baseline_default':   dict(anchors='default', min_size=800,  max_size=1333),   # reproduces AP=0
 'small_anchors':      dict(anchors='small',   min_size=640,  max_size=640),    # small anchors, native res (19px)
 'small_anchors_2x':   dict(anchors='small',   min_size=1280, max_size=1280),   # small anchors + 2x upscale (38px)
}
ntr=len(glob.glob(str(DG/'images'/'train'/'*.jpg'))); nva=len(glob.glob(str(DG/'images'/'val'/'*.jpg')))
print(f'GT-sup crops: train {ntr} | val {nva}',flush=True)
res={}
for nm,cfg in CONFIGS.items():
    torch.manual_seed(0); m=train(build_model(cfg),DG); r=evaluate(m,DG); res[nm]={**r,'cfg':cfg}
    print(f'{nm}: {r}',flush=True); json.dump(res,open(RES/'results.json','w'),indent=2)
print('\n=== WS3 resolution-aware detection (Chittagong-2, GT-supervised) ==='); print(json.dumps(res,indent=2)); print('WS3_DONE',flush=True)
