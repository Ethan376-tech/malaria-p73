# -*- coding: utf-8 -*-
"""Phase 2 — MILCA bootstrap refinement + uncertainty at the bootstrap (the 2nd place MILCA's extension ② applies it).
D1 = RetinaNet trained on protected pseudo-labels. Apply D1 to the train crops; for each detection score it with the
§2.3 FT-ThickDINO MC-dropout (p̄, var). Two retain strategies -> refined labels: (a) MILCA conf>0.7; (b) ours =
faintness-protected (var>=b_protect OR p̄>=a_star). Retrain D2 on each; eval detection AP on FASTMAL (overall + crop
density band). Headline: does the faintness-protected retain preserve low-parasitaemia better than conf>0.7?
Output -> outputs/experiments/s7_bootstrap/."""
import os, sys, json, glob, shutil
os.environ['XFORMERS_DISABLED']='1'; os.environ['PYTORCH_CUDA_ALLOC_CONF']='expandable_segments:True'
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import torchvision.transforms.functional as TF
import torchvision.transforms as T
from PIL import Image; Image.MAX_IMAGE_PIXELS=None
from torchvision.models.detection import retinanet_resnet50_fpn
sys.path.insert(0,'/root/autodl-tmp/A_Dissertation/references/RedDino/train'); sys.path.append('/root/A_Dissertation')
from common.paths import OUTPUTS, CHECKPOINTS
from dinov2.models.vision_transformer import vit_small
dev='cuda'; torch.manual_seed(0); np.random.seed(0)
CROP=640; TILE=224; T_MC=20; EPOCHS=50; BASE_LR=0.005; WARMUP=500; BATCH=6
YO=OUTPUTS/'experiments'/'s5_yolo'; S6=OUTPUTS/'experiments'/'s6_retinanet'; RES=OUTPUTS/'experiments'/'s7_bootstrap'; RES.mkdir(parents=True,exist_ok=True)
FIN=json.load(open(OUTPUTS/'experiments'/'s3_filter'/'final.json'))['calib']; b_protect=FIN['b_protect']; a_star=FIN['a_star']
norm=T.Compose([T.ToTensor(),T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
# ---- D1 (RetinaNet trained on protected) ----
def build_ret(): return retinanet_resnet50_fpn(num_classes=2,weights=None,weights_backbone='IMAGENET1K_V1').to(dev)
D1=build_ret(); D1.load_state_dict(torch.load(S6/'retinanet_r50_protected.pth')); D1.eval()
# ---- FT-ThickDINO + head (MC uncertainty, the §2.3 filter signal) ----
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

# ---- apply D1 to train crops -> refined labels (milca conf>0.7 ; ours faintness-protected) ----
train_imgs=sorted(glob.glob(str(YO/'ds_a0'/'images'/'train'/'*.jpg')))
DM=RES/'ds_refmilca'; DO=RES/'ds_refours'
for d in (DM,DO):
    if d.exists(): shutil.rmtree(d)
    (d/'images'/'train').mkdir(parents=True); (d/'labels'/'train').mkdir(parents=True)
    (d/'images'/'val').mkdir(parents=True); (d/'labels'/'val').mkdir(parents=True)
nm=no=0
for k,ip in enumerate(train_imgs):
    im=Image.open(ip).convert('RGB'); arr=np.asarray(im); x=TF.to_tensor(im).to(dev)
    with torch.no_grad(): pred=D1([x])[0]
    boxes=pred['boxes'].cpu().numpy(); scores=pred['scores'].cpu().numpy()
    keep=np.where(scores>0.15)[0]; keep=keep[np.argsort(-scores[keep])][:60]
    if len(keep)==0:
        for d in (DM,DO): (d/'labels'/'train'/(os.path.basename(ip)[:-4]+'.txt')).write_text(""); shutil.copy(ip,d/'images'/'train'/os.path.basename(ip))
        continue
    tiles=[]
    for j in keep:
        cx=(boxes[j][0]+boxes[j][2])/2; cy=(boxes[j][1]+boxes[j][3])/2
        x0=int(min(max(cx-TILE/2,0),CROP-TILE)); y0=int(min(max(cy-TILE/2,0),CROP-TILE)); tiles.append(arr[y0:y0+TILE,x0:x0+TILE])
    pm,pv=mc_boxes(tiles)
    def yolo_line(b):
        cx=(b[0]+b[2])/2/CROP; cy=(b[1]+b[3])/2/CROP; w=(b[2]-b[0])/CROP; h=(b[3]-b[1])/CROP; return f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"
    milca=[yolo_line(boxes[j]) for ji,j in enumerate(keep) if scores[j]>0.7]
    ours=[yolo_line(boxes[j]) for ji,j in enumerate(keep) if (pv[ji]>=b_protect or pm[ji]>=a_star)]
    base=os.path.basename(ip)
    for d,labs in [(DM,milca),(DO,ours)]:
        shutil.copy(ip,d/'images'/'train'/base); (d/'labels'/'train'/(base[:-4]+'.txt')).write_text("\n".join(labs))
    nm+=len(milca); no+=len(ours)
    if (k+1)%150==0: print(f'{k+1}/{len(train_imgs)} crops | milca labels {nm} ours {no}',flush=True)
print(f'refined labels total: milca {nm} | ours {no}',flush=True)
# copy FASTMAL val into both
for vp in glob.glob(str(YO/'ds_a0'/'images'/'val'/'*.jpg')):
    base=os.path.basename(vp); lp=vp.replace('/images/','/labels/').replace('.jpg','.txt')
    for d in (DM,DO):
        shutil.copy(vp,d/'images'/'val'/base); shutil.copy(lp,d/'labels'/'val'/(base[:-4]+'.txt'))

# ---- train D2 + eval (reuse s6 recipe) ----
class DetDS(torch.utils.data.Dataset):
    def __init__(self,root,split): self.imgs=sorted(glob.glob(str(root/'images'/split/'*.jpg')))
    def __len__(self): return len(self.imgs)
    def __getitem__(self,i):
        ip=self.imgs[i]; x=TF.to_tensor(Image.open(ip).convert('RGB')); lp=ip.replace('/images/','/labels/').replace('.jpg','.txt'); bx=[]
        if os.path.exists(lp):
            for ln in open(lp).read().splitlines():
                f=ln.split()
                if len(f)==5: _,cx,cy,w,h=f; cx,cy,w,h=float(cx)*CROP,float(cy)*CROP,float(w)*CROP,float(h)*CROP; bx.append([cx-w/2,cy-h/2,cx+w/2,cy+h/2])
        bx=torch.tensor(bx,dtype=torch.float32).reshape(-1,4); return x,{'boxes':bx,'labels':torch.ones(len(bx),dtype=torch.int64)}
def collate(b): return tuple(zip(*b))
def train(model,root):
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
    return model
def iou(a,b):
    ix1,iy1=max(a[0],b[0]),max(a[1],b[1]); ix2,iy2=min(a[2],b[2]),min(a[3],b[3]); iw,ih=max(0,ix2-ix1),max(0,iy2-iy1); it=iw*ih
    ua=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-it; return it/ua if ua>0 else 0.0
@torch.no_grad()
def evaluate(model,root):
    ds=DetDS(root,'val'); model.eval()
    cropgt=[len(ds[i][1]['boxes']) for i in range(len(ds))]; med=np.median([c for c in cropgt if c>0]) if any(cropgt) else 0
    all_tp={0.3:[],0.5:[]}; sc=[]; ngt=0; band_tp={'low':[],'high':[]}; band_sc={'low':[],'high':[]}; band_gt={'low':0,'high':0}
    dl=torch.utils.data.DataLoader(ds,batch_size=BATCH,shuffle=False,collate_fn=collate,num_workers=6); idx=0
    for imgs,tgts in dl:
        preds=model([im.to(dev) for im in imgs])
        for pred,tgt in zip(preds,tgts):
            gt=tgt['boxes'].numpy().tolist(); ngt+=len(gt); bd='low' if cropgt[idx]<=med else 'high'; band_gt[bd]+=len(gt); idx+=1
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
                    if iouth==0.3: sc.append(float(scs[j])); band_tp[bd].append(tp); band_sc[bd].append(float(scs[j]))
    def ap(tps,scs,ng):
        if not tps: return 0.0
        o=np.argsort(-np.array(scs)); tps=np.array(tps)[o]; ctp=np.cumsum(tps); cfp=np.cumsum(1-tps); rec=ctp/max(1,ng); pre=ctp/np.maximum(1,ctp+cfp)
        mr=np.concatenate([[0],rec,[1]]); mp=np.concatenate([[0],pre,[0]])
        for i in range(len(mp)-1,0,-1): mp[i-1]=max(mp[i-1],mp[i])
        idx=np.where(mr[1:]!=mr[:-1])[0]; return float(np.sum((mr[idx+1]-mr[idx])*mp[idx+1]))
    return dict(**{f'AP@{t}':round(ap(all_tp[t],sc,ngt),3) for t in (0.3,0.5)},
                AP03_low=round(ap(band_tp['low'],band_sc['low'],band_gt['low']),3),
                AP03_high=round(ap(band_tp['high'],band_sc['high'],band_gt['high']),3), n_gt=ngt)
SEEDS=[0,1,2]                                    # 3 runs -> mean±std, aligned with MILCA's repeated-runs protocol
runs={'bootstrap_milca':[], 'bootstrap_ours':[]}
for seed in SEEDS:
    torch.manual_seed(seed); np.random.seed(seed)
    for nm_,root in [('bootstrap_milca',DM),('bootstrap_ours',DO)]:
        print(f'=== seed {seed} train D2: {nm_} ===',flush=True); m=train(build_ret(),root); r=evaluate(m,root); runs[nm_].append(r); print('  ->',r,flush=True)
res={'seeds':SEEDS,'runs':runs}
for nm_ in runs:
    ks=[k for k in runs[nm_][0] if k!='n_gt']
    res[nm_]={k:{'mean':round(float(np.mean([r[k] for r in runs[nm_]])),3),'std':round(float(np.std([r[k] for r in runs[nm_]])),3)} for k in ks}
# reference: D1 base (protected, no bootstrap)
res['D1_protected_base']=evaluate(D1,DM)
json.dump(res,open(RES/'bootstrap_results.json','w'),indent=2); print('\n=== bootstrap results (3-run mean±std) ==='); print(json.dumps(res,indent=2)); print('BOOTSTRAP_DONE',flush=True)
