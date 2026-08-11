# -*- coding: utf-8 -*-
"""LOFO-clean GT-supervised for the torchvision detectors of Table 4.7 (RetinaNet-R50, RetinaNet-v2, FCOS-R50), custom
AP@0.5 (== build_s6_phase3 metric). For each report test film f, train the detector on all OTHER films' GT crops
(dc7 pool), predict on film f's test crops; aggregate custom AP over the full report test set. Also reproduces the
LEAKED value (train on dc7 pool minus nothing... actually train on report ds_gtsup) as a method check. Output ->
dc_lofo_gtsup/torchvision.json"""
import os, sys, json, glob, random
os.environ['XFORMERS_DISABLED']='1'; os.environ['PYTORCH_CUDA_ALLOC_CONF']='expandable_segments:True'
import numpy as np, torch
from PIL import Image; Image.MAX_IMAGE_PIXELS=None
import torchvision.transforms.functional as TF
from torchvision.models.detection import (retinanet_resnet50_fpn, retinanet_resnet50_fpn_v2, fcos_resnet50_fpn)
sys.path.append('/root/A_Dissertation'); from common.paths import OUTPUTS, RAW
dev='cuda'; torch.manual_seed(0); np.random.seed(0); random.seed(0)
EPOCHS=50; BASE_LR=0.005; WARMUP=500; BATCH=6; CROP=640  # exact build_s6 recipe (proven, gives report 0.683)
YO=OUTPUTS/'experiments'/'s5_yolo'; LOFO=OUTPUTS/'experiments'/'dc_lofo_gtsup'; POOL=LOFO/'pool'
# ---- test crops (report testF) + film map + gt + band (== dc8) ----
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
def crop_film(ip):
    b=os.path.basename(ip)
    try: k=int(b.split('_')[0][2:])
    except Exception: return None
    return k2film.get(k)
def load_gt(ip):
    lp=VLAB/(os.path.basename(ip)[:-4]+'.txt'); bx=[]
    if lp.exists():
        for ln in lp.read_text().splitlines():
            f=ln.split()
            if len(f)==5: _,cx,cy,w,h=[float(z) for z in f]; cx,cy,w,h=cx*CROP,cy*CROP,w*CROP,h*CROP; bx.append([cx-w/2,cy-h/2,cx+w/2,cy+h/2])
    return bx
films={ip:crop_film(ip) for ip in val_imgs}; gts={ip:load_gt(ip) for ip in val_imgs}
med=np.median([len(g) for g in gts.values() if len(g)>0]) if any(len(g)>0 for g in gts.values()) else 0
test_films=sorted(set(f for f in films.values() if f))
# ---- pool positive crops per film (for training) ----
def pool_pos(film):
    out=[]
    for imgp in glob.glob(str(POOL/film/'images'/'*.jpg')):
        lp=str(POOL/film/'labels'/(os.path.basename(imgp)[:-4]+'.txt'))
        if os.path.exists(lp) and open(lp).read().strip(): out.append((imgp,lp))
    return out
pool_by_film={f:pool_pos(f) for f in sorted(set(g for g in [p.name for p in POOL.iterdir() if p.is_dir()]))} if POOL.exists() else {}
class ListDS(torch.utils.data.Dataset):
    def __init__(self, items): self.items=items
    def __len__(self): return len(self.items)
    def __getitem__(self,i):
        ip,lp=self.items[i]; im=Image.open(ip).convert('RGB'); x=TF.to_tensor(im); boxes=[]
        for ln in open(lp).read().splitlines():
            f=ln.split()
            if len(f)==5: _,cx,cy,w,h=f; cx,cy,w,h=float(cx)*CROP,float(cy)*CROP,float(w)*CROP,float(h)*CROP; boxes.append([cx-w/2,cy-h/2,cx+w/2,cy+h/2])
        boxes=torch.tensor(boxes,dtype=torch.float32).reshape(-1,4)
        return x,{'boxes':boxes,'labels':torch.ones(len(boxes),dtype=torch.int64)}
def collate(b): return tuple(zip(*b))
def build(arch):
    kw=dict(num_classes=2, weights=None, weights_backbone='IMAGENET1K_V1')
    return {'retinanet_r50':retinanet_resnet50_fpn,'retinanet_r50_v2':retinanet_resnet50_fpn_v2,'fcos_r50':fcos_resnet50_fpn}[arch](**kw).to(dev)
def train(model, items):
    dl=torch.utils.data.DataLoader(ListDS(items),batch_size=BATCH,shuffle=True,collate_fn=collate,num_workers=0)
    params=[p for p in model.parameters() if p.requires_grad]
    opt=torch.optim.SGD(params,lr=BASE_LR,momentum=0.9,weight_decay=1e-4); sched=torch.optim.lr_scheduler.MultiStepLR(opt,milestones=[30,42],gamma=0.1); it=0
    for ep in range(EPOCHS):
        model.train()
        for imgs,tgts in dl:
            if it<WARMUP:
                fr=it/WARMUP
                for g in opt.param_groups: g['lr']=BASE_LR*(0.01+0.99*fr)
            imgs=[im.to(dev) for im in imgs]; tgts=[{k:v.to(dev) for k,v in t.items()} for t in tgts]
            ld=model(imgs,tgts); loss=sum(ld.values()); opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(params,5.0); opt.step(); it+=1
        if it>=WARMUP: sched.step()
    return model
def iou(a,b):
    ix1,iy1=max(a[0],b[0]),max(a[1],b[1]); ix2,iy2=min(a[2],b[2]),min(a[3],b[3]); iw,ih=max(0,ix2-ix1),max(0,iy2-iy1); it=iw*ih
    ua=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-it; return it/ua if ua>0 else 0
@torch.no_grad()
def predict(model, ip):
    model.eval(); x=TF.to_tensor(Image.open(ip).convert('RGB')).to(dev); p=model([x])[0]
    return p['boxes'].cpu().numpy(), p['scores'].cpu().numpy()
def ap(t,s,n):
    if not t:return 0.0
    o=np.argsort(-np.array(s)); t=np.array(t)[o]; ct=np.cumsum(t);cf=np.cumsum(1-t);rc=ct/max(1,n);pr=ct/np.maximum(1,ct+cf)
    mr=np.r_[0,rc,1];mp=np.r_[0,pr,0]
    for i in range(len(mp)-1,0,-1):mp[i-1]=max(mp[i-1],mp[i])
    k=np.where(mr[1:]!=mr[:-1])[0];return round(float(np.sum((mr[k+1]-mr[k])*mp[k+1])),3)
def aggregate(preds):
    at={0.3:[],0.5:[]}; sc=[]; ng=0; btp={'low':[],'high':[]}; bsc={'low':[],'high':[]}; bg={'low':0,'high':0}
    for ip in val_imgs:
        gt=gts[ip]; ng+=len(gt); bd='low' if len(gt)<=med else 'high'; bg[bd]+=len(gt)
        bxs,scs=preds.get(ip,(np.zeros((0,4)),np.array([]))); o=np.argsort(-scs)
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
    return dict(**{f'AP@{t}':ap(at[t],sc,ng) for t in (0.3,0.5)}, AP03_low=ap(btp['low'],bsc['low'],bg['low']), AP03_high=ap(btp['high'],bsc['high'],bg['high']))
print('DC11 START ok',flush=True)
res={}
for arch in ['retinanet_r50','retinanet_r50_v2','fcos_r50']:
    preds={}
    for f in test_films:
        items=[it for film,lst in pool_by_film.items() if film!=f for it in lst]
        torch.manual_seed(0); np.random.seed(0); m=build(arch); m=train(m,items)
        for ip in [p for p in val_imgs if films[p]==f]: preds[ip]=predict(m,ip)
        del m; torch.cuda.empty_cache()
        print(f'  {arch} fold excl {f}: trained on {len(items)} crops',flush=True)
    res[arch]=aggregate(preds); print(f'=== {arch} LOFO clean GT-sup: {res[arch]} ===',flush=True)
    json.dump(res, open(LOFO/'torchvision.json','w'), indent=2)
print('DC11_DONE',flush=True)
