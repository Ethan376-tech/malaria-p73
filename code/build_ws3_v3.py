# -*- coding: utf-8 -*-
"""WS3 v3 — decisive test: is 21px detection fundamentally hard, or is v2's low AP just the C1->C2 scale gap
(train 40px, test 21px)? Run IN-DOMAIN, matched scale: C2->C2 (21px, train/test on disjoint C2 fields) and
C1->C1 (40px, reference). Correct boxes (WS3 v2 fix: center=midpoint of the 2 circle points, side=distance).
Output -> outputs/experiments/ws3_v3/results.json
"""
import os, sys, json, glob, random
os.environ['XFORMERS_DISABLED']='1'
sys.path.append('/root/A_Dissertation')
import numpy as np, torch
import torchvision.transforms.functional as TF
from pathlib import Path
from PIL import Image; Image.MAX_IMAGE_PIXELS=None
from common.paths import RAW, CHITTAGONG2, OUTPUTS
from torchvision.models.detection import retinanet_resnet50_fpn
from torchvision.models.detection.retinanet import RetinaNetHead
from torchvision.models.detection.anchor_utils import AnchorGenerator
dev='cuda'; CROP=640; EPOCHS=30; BASE_LR=0.005; WARMUP=400; BATCH=6; MINSIDE=12
RES=OUTPUTS/'experiments'/'ws3_v3'; RES.mkdir(parents=True,exist_ok=True); DS=RES/'ds'; random.seed(0)

def parse_circles(fp, key):
    out=[]
    for ln in open(fp,errors='ignore').read().splitlines()[1:]:
        f=ln.split(',')
        if len(f)>=9 and key in f[1]:
            try:
                x1,y1,x2,y2=map(float,f[5:9]); cx,cy=(x1+x2)/2,(y1+y2)/2; side=max(((x2-x1)**2+(y2-y1)**2)**0.5,MINSIDE); out.append((cx,cy,side))
            except ValueError: pass
    return out
def c1_fields():
    o=[]
    for fp in glob.glob(str(RAW/'**'/'All_annotations'/'**'/'*.txt'),recursive=True):
        bx=parse_circles(fp,'Parasitized'); ip=fp.replace('All_annotations','All_PvTk').replace('.txt','.jpg')
        if bx and os.path.exists(ip): o.append((ip,bx))
    return o
def c2_fields():
    o=[]; GT=CHITTAGONG2/'GT_updated'
    for tf in sorted(d for d in CHITTAGONG2.iterdir() if d.is_dir() and d.name!='GT_updated'):
        for ip in sorted(tf.glob('*.jpg')):
            bf=GT/tf.name/(ip.stem+'.txt')
            if bf.exists():
                bx=parse_circles(bf,'Parasite')
                if bx: o.append((str(ip),bx))
    return o
def crop_grid(W,H):
    xs=list(range(0,max(1,W-CROP+1),CROP))+([W-CROP] if W>CROP else [0]); ys=list(range(0,max(1,H-CROP+1),CROP))+([H-CROP] if H>CROP else [0]); return sorted(set(xs)),sorted(set(ys))
def write_crops(fields, split):
    (DS/'images'/split).mkdir(parents=True,exist_ok=True); (DS/'labels'/split).mkdir(parents=True,exist_ok=True); n=0
    for k,(ip,bx) in enumerate(fields):
        try: arr=np.asarray(Image.open(ip).convert('RGB'))
        except Exception: continue
        H,W=arr.shape[:2]; xs,ys=crop_grid(W,H)
        for cxo in xs:
            for cyo in ys:
                sub=arr[cyo:cyo+CROP,cxo:cxo+CROP]
                if sub.shape[0]<CROP or sub.shape[1]<CROP:
                    pad=np.zeros((CROP,CROP,3),np.uint8); pad[:sub.shape[0],:sub.shape[1]]=sub; sub=pad
                labs=[f"0 {(cx-cxo)/CROP:.6f} {(cy-cyo)/CROP:.6f} {side/CROP:.6f} {side/CROP:.6f}"
                      for (cx,cy,side) in bx if 0<=cx-cxo<CROP and 0<=cy-cyo<CROP]
                if not labs: continue
                Image.fromarray(sub).save(DS/'images'/split/f'{split}_{k}_{cxo}_{cyo}.jpg',quality=90)
                open(DS/'labels'/split/f'{split}_{k}_{cxo}_{cyo}.txt','w').write('\n'.join(labs)); n+=1
    return n

if not (DS/'images').exists():
    c1=c1_fields(); random.Random(1).shuffle(c1); c2=c2_fields(); random.Random(1).shuffle(c2)
    print(f'C1 fields {len(c1)} | C2 fields {len(c2)}',flush=True)
    a=write_crops(c2[:60],'c2train'); b=write_crops(c2[60:100],'c2test')
    c=write_crops(c1[:40],'c1train'); d=write_crops(c1[40:60],'c1test')
    print(f'crops C2 tr {a} te {b} | C1 tr {c} te {d}',flush=True)
else: print('reusing crops',flush=True)

class DetDS(torch.utils.data.Dataset):
    def __init__(self,split): self.imgs=sorted(glob.glob(str(DS/'images'/split/'*.jpg')))
    def __len__(self): return len(self.imgs)
    def __getitem__(self,i):
        ip=self.imgs[i]; x=TF.to_tensor(Image.open(ip).convert('RGB')); lp=ip.replace('/images/','/labels/').replace('.jpg','.txt'); bx=[]
        for ln in open(lp).read().splitlines():
            f=ln.split(); _,cx,cy,w,h=f; cx,cy,w,h=float(cx)*CROP,float(cy)*CROP,float(w)*CROP,float(h)*CROP; bx.append([cx-w/2,cy-h/2,cx+w/2,cy+h/2])
        bx=torch.tensor(bx,dtype=torch.float32).reshape(-1,4); bx[:,[0,2]]=bx[:,[0,2]].clamp(0,CROP); bx[:,[1,3]]=bx[:,[1,3]].clamp(0,CROP)
        keep=(bx[:,2]-bx[:,0]>=1)&(bx[:,3]-bx[:,1]>=1); bx=bx[keep]
        return x,{'boxes':bx,'labels':torch.ones(len(bx),dtype=torch.int64)}
def collate(b): return tuple(zip(*b))
def build_model(cfg):
    m=retinanet_resnet50_fpn(num_classes=2,weights=None,weights_backbone='IMAGENET1K_V1')
    if cfg['anchors']=='small':
        sizes=tuple((s,int(s*1.26),int(s*1.587)) for s in (8,16,32,64,128))
        ag=AnchorGenerator(sizes=sizes, aspect_ratios=((0.5,1.0,2.0),)*len(sizes))
        m.anchor_generator=ag; m.head=RetinaNetHead(256, ag.num_anchors_per_location()[0], 2)
    m.transform.min_size=(cfg['min_size'],); m.transform.max_size=cfg['max_size']; return m.to(dev)
def train(model,TR):
    dl=torch.utils.data.DataLoader(DetDS(TR),batch_size=BATCH,shuffle=True,collate_fn=collate,num_workers=6)
    params=[p for p in model.parameters() if p.requires_grad]; opt=torch.optim.SGD(params,lr=BASE_LR,momentum=0.9,weight_decay=1e-4)
    sched=torch.optim.lr_scheduler.MultiStepLR(opt,milestones=[20,27],gamma=0.1); it=0
    for ep in range(EPOCHS):
        model.train()
        for imgs,tgts in dl:
            if it<WARMUP:
                fr=it/WARMUP
                for g in opt.param_groups: g['lr']=BASE_LR*(0.01+0.99*fr)
            imgs=[im.to(dev) for im in imgs]; tgts=[{k:v.to(dev) for k,v in t.items()} for t in tgts]
            loss=sum(model(imgs,tgts).values()); opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(params,5.0); opt.step(); it+=1
        if it>=WARMUP: sched.step()
    return model
def iou(a,b):
    ix1,iy1=max(a[0],b[0]),max(a[1],b[1]); ix2,iy2=min(a[2],b[2]),min(a[3],b[3]); iw,ih=max(0,ix2-ix1),max(0,iy2-iy1); inr=iw*ih
    ua=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inr; return inr/ua if ua>0 else 0.0
@torch.no_grad()
def evaluate(model,TE):
    dl=torch.utils.data.DataLoader(DetDS(TE),batch_size=BATCH,shuffle=False,collate_fn=collate,num_workers=6); model.eval()
    all_tp={0.3:[],0.5:[]}; sc=[]; ngt=0; npred=0
    for imgs,tgts in dl:
        preds=model([im.to(dev) for im in imgs])
        for pred,tgt in zip(preds,tgts):
            gt=tgt['boxes'].numpy().tolist(); ngt+=len(gt); bxs=pred['boxes'].cpu().numpy(); scs=pred['scores'].cpu().numpy(); npred+=len(scs); order=np.argsort(-scs)
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
    return dict(**{f'AP@{t}':round(ap(all_tp[t],sc,ngt),3) for t in (0.3,0.5)}, n_gt=ngt, n_pred=npred)

RUNS={
 'C2_indomain_small':    dict(TR='c2train',TE='c2test', cfg=dict(anchors='small',min_size=640, max_size=640)),
 'C2_indomain_small_2x': dict(TR='c2train',TE='c2test', cfg=dict(anchors='small',min_size=1280,max_size=1280)),
 'C1_indomain_small':    dict(TR='c1train',TE='c1test', cfg=dict(anchors='small',min_size=640, max_size=640)),
}
res={}
for nm,r in RUNS.items():
    torch.manual_seed(0); m=train(build_model(r['cfg']),r['TR']); rr=evaluate(m,r['TE']); res[nm]={**rr,'cfg':r['cfg']}
    print(f"{nm}: {rr}",flush=True); json.dump(res,open(RES/'results.json','w'),indent=2)
print('\n=== WS3 v3 in-domain detection (correct boxes) ==='); print(json.dumps(res,indent=2)); print('WS3V3_DONE',flush=True)
