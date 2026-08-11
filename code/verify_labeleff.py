# -*- coding: utf-8 -*-
"""VERIFY §4.7 label-efficiency: is 'supervised > hybrid at k>=2' real or a train-schedule artifact?
Controlled test — for each (seed,k) the SAME k fields and the SAME batch order are used, and the ONLY
difference between 'supervised' and 'hybrid_matched' is the initialisation (ImageNet vs weak-pretrained base).
Also runs 'hybrid_ft' (the original 25ep/lr0.001 fine-tune schedule) to reproduce the reported hybrid.
Writes s10_labeleff/labeleff_verify.json (does NOT overwrite labeleff_results.json)."""
import os, sys, json, glob, re
os.environ['XFORMERS_DISABLED']='1'; os.environ['PYTORCH_CUDA_ALLOC_CONF']='expandable_segments:True'
import numpy as np, torch
import torchvision.transforms.functional as TF
from PIL import Image; Image.MAX_IMAGE_PIXELS=None
from torchvision.models.detection import retinanet_resnet50_fpn
sys.path.append('/root/A_Dissertation'); from common.paths import OUTPUTS
dev='cuda'; CROP=640; BATCH=6; BUDGETS=[2,5,10,20]
YO=OUTPUTS/'experiments'/'s5_yolo'; S6=OUTPUTS/'experiments'/'s6_retinanet'; RES=OUTPUTS/'experiments'/'s10_labeleff'
GT_TR=YO/'ds_gtsup'/'images'/'train'; GT_LAB=YO/'ds_gtsup'/'labels'/'train'
VAL_IMG=YO/'ds_gtsup'/'images'/'val'; VAL_LAB=YO/'ds_gtsup'/'labels'/'val'
CKPT=S6/'retinanet_r50_protected.pth'
def field_idx(p):
    m=re.match(r'gt(\d+)_',os.path.basename(p)); return int(m.group(1)) if m else 999
gt_train=sorted(glob.glob(str(GT_TR/'*.jpg')))
ALL_FIELDS=sorted(set(field_idx(p) for p in gt_train))
def labeled(k,rng):
    chosen=set(rng.choice(ALL_FIELDS,size=min(k,len(ALL_FIELDS)),replace=False))
    return [p for p in gt_train if field_idx(p) in chosen], sorted(chosen)
def build_ret(): return retinanet_resnet50_fpn(num_classes=2,weights=None,weights_backbone='IMAGENET1K_V1').to(dev)
class DS(torch.utils.data.Dataset):
    def __init__(self,paths,labdir): self.paths=paths; self.labdir=labdir
    def __len__(self): return len(self.paths)
    def __getitem__(self,i):
        ip=self.paths[i]; x=TF.to_tensor(Image.open(ip).convert('RGB')); lp=os.path.join(str(self.labdir),os.path.basename(ip)[:-4]+'.txt'); bx=[]
        if os.path.exists(lp):
            for ln in open(lp).read().splitlines():
                f=ln.split()
                if len(f)==5: _,cx,cy,w,h=f; cx,cy,w,h=float(cx)*CROP,float(cy)*CROP,float(w)*CROP,float(h)*CROP; bx.append([cx-w/2,cy-h/2,cx+w/2,cy+h/2])
        bx=torch.tensor(bx,dtype=torch.float32).reshape(-1,4)
        if len(bx): bx[:,[0,2]]=bx[:,[0,2]].clamp(0,CROP); bx[:,[1,3]]=bx[:,[1,3]].clamp(0,CROP); k=(bx[:,2]-bx[:,0]>=1)&(bx[:,3]-bx[:,1]>=1); bx=bx[k]
        return x,{'boxes':bx,'labels':torch.ones(len(bx),dtype=torch.int64)}
def collate(b): return tuple(zip(*b))
def train(model,paths,labdir,epochs,lr,warm,seed):
    torch.manual_seed(seed); np.random.seed(seed)          # identical batch order for matched arms
    dl=torch.utils.data.DataLoader(DS(paths,labdir),batch_size=BATCH,shuffle=True,collate_fn=collate,num_workers=6)
    if len(dl)==0: return model
    params=[p for p in model.parameters() if p.requires_grad]; opt=torch.optim.SGD(params,lr=lr,momentum=0.9,weight_decay=1e-4)
    sched=torch.optim.lr_scheduler.MultiStepLR(opt,milestones=[int(epochs*0.6),int(epochs*0.85)],gamma=0.1); it=0
    for ep in range(epochs):
        model.train()
        for imgs,tgts in dl:
            if it<warm:
                f=it/max(1,warm)
                for g in opt.param_groups: g['lr']=lr*(0.01+0.99*f)
            imgs=[im.to(dev) for im in imgs]; tgts=[{k:v.to(dev) for k,v in t.items()} for t in tgts]
            if sum(len(t['boxes']) for t in tgts)==0: continue
            loss=sum(model(imgs,tgts).values()); opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(params,5.0); opt.step(); it+=1
        if it>=warm: sched.step()
    return model
def iou(a,b):
    ix1,iy1=max(a[0],b[0]),max(a[1],b[1]); ix2,iy2=min(a[2],b[2]),min(a[3],b[3]); iw,ih=max(0,ix2-ix1),max(0,iy2-iy1); it=iw*ih
    ua=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-it; return it/ua if ua>0 else 0.0
@torch.no_grad()
def evaluate(model):
    dl=torch.utils.data.DataLoader(DS(sorted(glob.glob(str(VAL_IMG/'*.jpg'))),VAL_LAB),batch_size=BATCH,shuffle=False,collate_fn=collate,num_workers=6); model.eval()
    all_tp={0.3:[],0.5:[]}; sc=[]; ngt=0
    for imgs,tgts in dl:
        preds=model([im.to(dev) for im in imgs])
        for pred,tgt in zip(preds,tgts):
            gt=tgt['boxes'].numpy().tolist(); ngt+=len(gt); bxs=pred['boxes'].cpu().numpy(); scs=pred['scores'].cpu().numpy(); order=np.argsort(-scs)
            for iouth in (0.3,0.5):
                matched=set()
                for j in order:
                    bb=bxs[j]; best=-1;bj=-1
                    for jg,g in enumerate(gt):
                        if jg in matched: continue
                        v=iou(bb,g)
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
    return {f'AP@{t}':round(ap(all_tp[t],sc,ngt),3) for t in (0.3,0.5)}
def sig(model):
    with torch.no_grad(): return round(float(torch.cat([p.detach().flatten() for p in model.parameters()])[:200000].abs().mean()),6)

# ---- sanity: prove the weak base actually loads and differs from a fresh ImageNet init ----
fresh=build_ret(); base=build_ret(); base.load_state_dict(torch.load(CKPT))
print(f'SANITY  fresh_sig={sig(fresh)}  weakbase_sig={sig(base)}  differ={sig(fresh)!=sig(base)}',flush=True)
print(f'SANITY  val_images={len(glob.glob(str(VAL_IMG/"*.jpg")))}  train_fields={ALL_FIELDS}',flush=True)
print(f'SANITY  weakbase AP (k=0) = {evaluate(base)}',flush=True)

SEEDS=[0,1,2]; per=[]
for seed in SEEDS:
    rng=np.random.RandomState(seed)
    res={'supervised':{}, 'hybrid_matched':{}, 'hybrid_ft':{}}
    b0=build_ret(); b0.load_state_dict(torch.load(CKPT)); e0=evaluate(b0)
    res['hybrid_matched']['0']=e0; res['hybrid_ft']['0']=e0
    for k in BUDGETS:
        lab,ids=labeled(k,rng); ts=1000*seed+k
        print(f'== seed{seed} k{k}  fields={ids}  crops={len(lab)} ==',flush=True)
        s=train(build_ret(),lab,GT_LAB,40,0.005,200,ts); res['supervised'][str(k)]=evaluate(s); print(f'   supervised     {res["supervised"][str(k)]}',flush=True)
        hm=build_ret(); hm.load_state_dict(torch.load(CKPT)); hm=train(hm,lab,GT_LAB,40,0.005,200,ts); res['hybrid_matched'][str(k)]=evaluate(hm); print(f'   hybrid_matched {res["hybrid_matched"][str(k)]}',flush=True)
        hf=build_ret(); hf.load_state_dict(torch.load(CKPT)); hf=train(hf,lab,GT_LAB,25,0.001,50,ts); res['hybrid_ft'][str(k)]=evaluate(hf); print(f'   hybrid_ft      {res["hybrid_ft"][str(k)]}',flush=True)
    per.append(res)
agg={}
for arm in per[0]:
    agg[arm]={}
    for kk in per[0][arm]:
        agg[arm][kk]={m:{'mean':round(float(np.mean([per[s][arm][kk][m] for s in range(len(SEEDS))])),3),
                         'std':round(float(np.std([per[s][arm][kk][m] for s in range(len(SEEDS))])),3)} for m in per[0][arm][kk]}
json.dump({'seeds':SEEDS,'per_run':per,'agg':agg},open(RES/'labeleff_verify.json','w'),indent=2)
print('\n=== VERIFY agg AP@0.3 (mean over 3 seeds) ===',flush=True)
print('arm             k2       k5       k10      k20',flush=True)
for arm in ['supervised','hybrid_matched','hybrid_ft']:
    print(arm.ljust(15)+'  '.join(str(agg[arm][k]['AP@0.3']['mean']).ljust(7) for k in ['2','5','10','20']),flush=True)
print('VERIFY_DONE',flush=True)
