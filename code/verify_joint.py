# -*- coding: utf-8 -*-
"""VERIFY §3.5.3/§4.4 joint MIL-detector negative — does joint training (bag loss) hurt, or is it just the
negative-sample crops? Adds the missing CONTROL arm 'det+neg' (detection-only trained on the SAME pos+neg crops,
no bag loss). Decomposition:  baseline(pos) -> det+neg(pos+neg): effect of negative crops;  det+neg -> joint:
effect of the bag loss (the actual MILCA joint idea). 3 seeds, matched 40ep/lr0.005 schedule for all arms.
Writes s12_joint/joint_verify.json (does not overwrite joint_results.json)."""
import os, sys, json, glob, random
os.environ['XFORMERS_DISABLED']='1'; os.environ['PYTORCH_CUDA_ALLOC_CONF']='expandable_segments:True'
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image; Image.MAX_IMAGE_PIXELS=None
from torchvision.models.detection import retinanet_resnet50_fpn
sys.path.append('/root/A_Dissertation'); from common.paths import OUTPUTS, IBADAN_PART1, IBADAN_PART2
dev='cuda'; CROP=640; EPOCHS=40; BASE_LR=0.005; WARMUP=200; BATCH=6; LAM=1.0
YO=OUTPUTS/'experiments'/'s5_yolo'; RES=OUTPUTS/'experiments'/'s12_joint'; RES.mkdir(parents=True,exist_ok=True)
POS_IMG=YO/'ds_a0'/'images'/'train'; POS_LAB=YO/'ds_protected'/'labels'/'train'
VAL_IMG=YO/'ds_gtsup'/'images'/'val'; VAL_LAB=YO/'ds_gtsup'/'labels'/'val'
NEGDIR=RES/'neg_crops'
EXT={'.tif','.tiff','.png','.jpg','.jpeg'}
if len(glob.glob(str(NEGDIR/'*.jpg')))==0:
    NEGDIR.mkdir(parents=True,exist_ok=True)
    bm=pd.read_csv(OUTPUTS/'experiments'/'stageD_bag'/'bag_meta_v2.csv')
    ibcsv=pd.read_csv('/root/autodl-tmp/A_Dissertation/data/raw/ibadan/sample_codes_parasite_diagnosis_crosscheck.csv')
    negids=list(bm[(bm.dataset=='ibadan')&(bm.label01==0)].sample_id)
    sd={r.sample_code:(IBADAN_PART1 if 'part1' in r.input_file else IBADAN_PART2)/r.sample_code for _,r in ibcsv.iterrows()}
    nfields=[]
    for sid in negids:
        d=sd.get(sid)
        if d and d.exists(): nfields+=[p for p in d.rglob('*') if p.suffix.lower() in EXT][:2]
        if len(nfields)>=40: break
    random.Random(1).shuffle(nfields); nc=0
    for k,fp in enumerate(nfields[:40]):
        try: im=Image.open(fp).convert('RGB')
        except Exception: continue
        arr=np.asarray(im); H,W=arr.shape[:2]; xs=list(range(0,max(1,W-CROP+1),CROP))+([W-CROP] if W>CROP else [0]); ys=list(range(0,max(1,H-CROP+1),CROP))+([H-CROP] if H>CROP else [0])
        for cxo in sorted(set(xs))[:4]:
            for cyo in sorted(set(ys))[:4]:
                sub=arr[cyo:cyo+CROP,cxo:cxo+CROP]
                if sub.shape[0]<CROP or sub.shape[1]<CROP:
                    pad=np.zeros((CROP,CROP,3),np.uint8); pad[:sub.shape[0],:sub.shape[1]]=sub; sub=pad
                if np.asarray(Image.fromarray(sub).convert('L')).std()<10: continue
                Image.fromarray(sub).save(NEGDIR/f'neg{k}_{cxo}_{cyo}.jpg',quality=92); nc+=1
        if nc>=300: break
    print(f'negative crops generated: {nc}',flush=True)
NEG_IMG=sorted(glob.glob(str(NEGDIR/'*.jpg'))); POS=sorted(glob.glob(str(POS_IMG/'*.jpg')))
print(f'pos crops {len(POS)} | neg crops {len(NEG_IMG)}',flush=True)
pos_items=[(p, os.path.join(str(POS_LAB),os.path.basename(p)[:-4]+'.txt'), 1.0) for p in POS]
neg_items=[(p, None, 0.0) for p in NEG_IMG]
class JDS(torch.utils.data.Dataset):
    def __init__(self,items): self.items=items
    def __len__(self): return len(self.items)
    def __getitem__(self,i):
        ip,lp,bag=self.items[i]; x=TF.to_tensor(Image.open(ip).convert('RGB')); bx=[]
        if lp and os.path.exists(lp):
            for ln in open(lp).read().splitlines():
                f=ln.split()
                if len(f)==5: _,cx,cy,w,h=f; cx,cy,w,h=float(cx)*CROP,float(cy)*CROP,float(w)*CROP,float(h)*CROP; bx.append([cx-w/2,cy-h/2,cx+w/2,cy+h/2])
        bx=torch.tensor(bx,dtype=torch.float32).reshape(-1,4)
        if len(bx): bx[:,[0,2]]=bx[:,[0,2]].clamp(0,CROP); bx[:,[1,3]]=bx[:,[1,3]].clamp(0,CROP); k=(bx[:,2]-bx[:,0]>=1)&(bx[:,3]-bx[:,1]>=1); bx=bx[k]
        return x,{'boxes':bx,'labels':torch.ones(len(bx),dtype=torch.int64)},torch.tensor(bag,dtype=torch.float32)
def collate(b): xs,ts,bg=zip(*b); return list(xs),list(ts),torch.stack(bg)
def build_ret(): return retinanet_resnet50_fpn(num_classes=2,weights=None,weights_backbone='IMAGENET1K_V1').to(dev)
def iou(a,b):
    ix1,iy1=max(a[0],b[0]),max(a[1],b[1]); ix2,iy2=min(a[2],b[2]),min(a[3],b[3]); iw,ih=max(0,ix2-ix1),max(0,iy2-iy1); it=iw*ih
    ua=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-it; return it/ua if ua>0 else 0.0
@torch.no_grad()
def evaluate(model):
    paths=sorted(glob.glob(str(VAL_IMG/'*.jpg'))); items=[(p,os.path.join(str(VAL_LAB),os.path.basename(p)[:-4]+'.txt'),1.0) for p in paths]
    ds=JDS(items); cropgt=[len(ds[i][1]['boxes']) for i in range(len(ds))]; med=np.median([c for c in cropgt if c>0]) if any(cropgt) else 0
    dl=torch.utils.data.DataLoader(ds,batch_size=BATCH,shuffle=False,collate_fn=collate,num_workers=6); model.eval()
    all_tp={0.3:[],0.5:[]}; sc=[]; ngt=0; btp={'low':[],'high':[]}; bsc={'low':[],'high':[]}; bgt={'low':0,'high':0}; idx=0
    for imgs,tgts,_ in dl:
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
    return dict(**{f'AP@{t}':round(ap(all_tp[t],sc,ngt),3) for t in (0.3,0.5)}, AP03_low=round(ap(btp['low'],bsc['low'],bgt['low']),3), AP03_high=round(ap(btp['high'],bsc['high'],bgt['high']),3))
def train(items, joint, seed):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    model=build_ret(); bag_head=nn.Linear(256,1).to(dev); cap={}
    h=model.backbone.register_forward_hook(lambda m,i,o: cap.__setitem__('f',o))
    params=list(model.parameters())+(list(bag_head.parameters()) if joint else [])
    opt=torch.optim.SGD([p for p in params if p.requires_grad],lr=BASE_LR,momentum=0.9,weight_decay=1e-4)
    sched=torch.optim.lr_scheduler.MultiStepLR(opt,milestones=[24,34],gamma=0.1)
    dl=torch.utils.data.DataLoader(JDS(items),batch_size=BATCH,shuffle=True,collate_fn=collate,num_workers=6); it=0
    for ep in range(EPOCHS):
        model.train()
        for imgs,tgts,bag in dl:
            if it<WARMUP:
                for g in opt.param_groups: g['lr']=BASE_LR*(0.01+0.99*it/WARMUP)
            imgs=[im.to(dev) for im in imgs]; tgts2=[{k:v.to(dev) for k,v in t.items()} for t in tgts]
            ld=model(imgs,tgts2); loss=sum(ld.values())
            if joint:
                feats=cap['f']; lvl=feats['pool'] if 'pool' in feats else list(feats.values())[-1]
                pooled=F.adaptive_avg_pool2d(lvl,1).flatten(1); blog=bag_head(pooled).squeeze(-1)
                loss=loss+LAM*F.binary_cross_entropy_with_logits(blog,bag.to(dev))
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(params,5.0); opt.step(); it+=1
        if it>=WARMUP: sched.step()
    h.remove(); return model
SEEDS=[0,1,2]; per=[]
for seed in SEEDS:
    r={}
    print(f'== seed {seed} ==',flush=True)
    r['baseline_pos']=evaluate(train(pos_items, False, seed)); print('  baseline_pos    ',r['baseline_pos'],flush=True)
    r['detonly_posneg']=evaluate(train(pos_items+neg_items, False, seed)); print('  detonly_posneg  ',r['detonly_posneg'],flush=True)
    r['joint_bagloss']=evaluate(train(pos_items+neg_items, True, seed)); print('  joint_bagloss   ',r['joint_bagloss'],flush=True)
    per.append(r)
agg={}
for arm in per[0]:
    agg[arm]={m:{'mean':round(float(np.mean([per[s][arm][m] for s in range(len(SEEDS))])),3),
                 'std':round(float(np.std([per[s][arm][m] for s in range(len(SEEDS))])),3)} for m in per[0][arm]}
json.dump({'seeds':SEEDS,'per_run':per,'agg':agg},open(RES/'joint_verify.json','w'),indent=2)
print('\n=== JOINT VERIFY agg (mean±std, 3 seeds) ===',flush=True)
print('arm               AP@0.3        AP@0.5        AP03_low      AP03_high',flush=True)
for arm in ['baseline_pos','detonly_posneg','joint_bagloss']:
    print(arm.ljust(17)+'  '.join(('%.3f±%.3f'%(agg[arm][m]['mean'],agg[arm][m]['std'])).ljust(12) for m in ['AP@0.3','AP@0.5','AP03_low','AP03_high']),flush=True)
print('JOINT_VERIFY_DONE',flush=True)
