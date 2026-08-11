# -*- coding: utf-8 -*-
"""ADAPTIVE multi-signal retain at the bootstrap stage. The §3.4 multi-signal thresholds over-filter DETECTIONS, so
we RE-CALIBRATE for the detector-refinement objective: keep the single-signal protect clause (which works: b_protect,
a_star from final.json) and ADD (a) a TTA-variance protector (OR, only adds faint protection) and (b) a LIGHT
typicality gate whose tau is calibrated on the DETECTION Mahalanobis distribution (not the pseudo-box one). Compare,
each retrain D2 (1-seed screen), against MILCA conf>0.7 and the single-signal retain. Output -> s7_adaptive/."""
import os, sys, json, glob, shutil
os.environ['XFORMERS_DISABLED']='1'; os.environ['PYTORCH_CUDA_ALLOC_CONF']='expandable_segments:True'
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import torchvision.transforms.functional as TF, torchvision.transforms as T
from PIL import Image; Image.MAX_IMAGE_PIXELS=None
from torchvision.models.detection import retinanet_resnet50_fpn
sys.path.insert(0,'/root/autodl-tmp/A_Dissertation/references/RedDino/train'); sys.path.append('/root/A_Dissertation')
from common.paths import OUTPUTS, CHECKPOINTS
from dinov2.models.vision_transformer import vit_small
from sklearn.decomposition import PCA; from sklearn.covariance import LedoitWolf
dev='cuda'; torch.manual_seed(0); np.random.seed(0)
CROP=640; TILE=224; T_MC=20; EPOCHS=50; BASE_LR=0.005; WARMUP=500; BATCH=6
YO=OUTPUTS/'experiments'/'s5_yolo'; S6=OUTPUTS/'experiments'/'s6_retinanet'; RES=OUTPUTS/'experiments'/'s7_adaptive'; RES.mkdir(parents=True,exist_ok=True)
S3=OUTPUTS/'experiments'/'s3_filter'
FIN=json.load(open(S3/'final.json'))['calib']; b_protect=FIN['b_protect']; a_star=FIN['a_star']   # single-signal (works)
pb=np.load(S3/'perbox_multi2.npz'); dv=pb['field_id'].astype(int)<95; tpm=pb['is_tp']==1
faint=dv&tpm&(pb['p_mean']<=np.quantile(pb['p_mean'][dv&tpm],1/3)); b_tta=float(np.quantile(pb['tta_var'][faint],0.10))
_E=pb['emb'][dv&tpm]; _mu=_E.mean(0); _sd=_E.std(0)+1e-6; _pca=PCA(40,random_state=0).fit((_E-_mu)/_sd); _lw=LedoitWolf().fit(_pca.transform((_E-_mu)/_sd))
def maha_of(emb): return _lw.mahalanobis(_pca.transform((emb-_mu)/_sd)) if len(emb) else np.array([])
print(f'single-signal floors: b_protect {b_protect:.5f} a_star {a_star:.4f} | added b_tta {b_tta:.5f}',flush=True)
norm=T.Compose([T.ToTensor(),T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
def build_ret(): return retinanet_resnet50_fpn(num_classes=2,weights=None,weights_backbone='IMAGENET1K_V1').to(dev)
D1=build_ret(); D1.load_state_dict(torch.load(S6/'retinanet_r50_protected.pth')); D1.eval()
ftm=vit_small(patch_size=16,img_size=224,block_chunks=0,init_values=1.0,drop_path_rate=0.1)
ftm.load_state_dict(torch.load(CHECKPOINTS/'ssl_vits_thickdino_ft_ibadan'/'encoder.pth',map_location='cpu'),strict=False); ftm=ftm.to(dev).eval()
fth=nn.Sequential(nn.Dropout(0.2),nn.Linear(384,1)).to(dev); fth.load_state_dict(torch.load(CHECKPOINTS/'ssl_vits_thickdino_ft_ibadan'/'head.pth',map_location='cpu')); fth.eval()
@torch.no_grad()
def mc(tiles,bs=48):
    for mod in list(ftm.modules())+list(fth.modules()):
        if isinstance(mod,nn.Dropout): mod.train()
    for mod in ftm.modules():
        if mod.__class__.__name__=='DropPath': mod.train()
    M=[];V=[]
    for i in range(0,len(tiles),bs):
        xb=torch.stack([norm(Image.fromarray(t)) for t in tiles[i:i+bs]]).to(dev); pr=[]
        for _ in range(T_MC): pr.append(torch.sigmoid(fth(ftm.forward_features(xb)['x_norm_clstoken']).squeeze(-1)).cpu().numpy())
        A=np.stack(pr); M.append(A.mean(0)); V.append(A.var(0))
    ftm.eval(); fth.eval(); return (np.concatenate(M),np.concatenate(V)) if M else (np.array([]),np.array([]))
@torch.no_grad()
def emb_tta(tiles,bs=48):
    ftm.eval(); fth.eval(); EM=[];TV=[]
    for i in range(0,len(tiles),bs):
        xb=torch.stack([norm(Image.fromarray(t)) for t in tiles[i:i+bs]]).to(dev)
        EM.append(ftm.forward_features(xb)['x_norm_clstoken'].cpu().numpy())
        views=[xb,torch.flip(xb,[3]),torch.flip(xb,[2]),torch.rot90(xb,1,[2,3]),torch.rot90(xb,2,[2,3]),torch.rot90(xb,3,[2,3])]
        pv=[torch.sigmoid(fth(ftm.forward_features(v)['x_norm_clstoken']).squeeze(-1)).cpu().numpy() for v in views]
        TV.append(np.stack(pv).var(0))
    return (np.concatenate(EM),np.concatenate(TV)) if EM else (np.zeros((0,384),np.float32),np.array([]))
def yolo_line(b):
    cx=(b[0]+b[2])/2/CROP; cy=(b[1]+b[3])/2/CROP; w=(b[2]-b[0])/CROP; h=(b[3]-b[1])/CROP; return f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"
# ---- apply D1, compute per-detection signals ----
train_imgs=sorted(glob.glob(str(YO/'ds_a0'/'images'/'train'/'*.jpg'))); DET={}
allmaha=[]
for k,ip in enumerate(train_imgs):
    im=Image.open(ip).convert('RGB'); arr=np.asarray(im); x=TF.to_tensor(im).to(dev)
    with torch.no_grad(): pred=D1([x])[0]
    boxes=pred['boxes'].cpu().numpy(); scores=pred['scores'].cpu().numpy()
    keep=np.where(scores>0.15)[0]; keep=keep[np.argsort(-scores[keep])][:60]
    base=os.path.basename(ip)
    if len(keep)==0: DET[base]=[]; continue
    tiles=[]
    for j in keep:
        cx=(boxes[j][0]+boxes[j][2])/2; cy=(boxes[j][1]+boxes[j][3])/2
        x0=int(min(max(cx-TILE/2,0),CROP-TILE)); y0=int(min(max(cy-TILE/2,0),CROP-TILE)); tiles.append(arr[y0:y0+TILE,x0:x0+TILE])
    pm,pv=mc(tiles); em,tta=emb_tta(tiles); mh=maha_of(em)
    DET[base]=[(yolo_line(boxes[j]),float(scores[j]),float(pm[i]),float(pv[i]),float(tta[i]),float(mh[i])) for i,j in enumerate(keep)]
    allmaha+=[d[5] for d in DET[base]]
    if (k+1)%150==0: print(f'D1 applied {k+1}/{len(train_imgs)} crops',flush=True)
allmaha=np.array(allmaha); tau99=float(np.quantile(allmaha,0.99)); tau95=float(np.quantile(allmaha,0.95))
print(f'detection maha: tau99 {tau99:.1f} tau95 {tau95:.1f} (n {len(allmaha)})',flush=True)
# ---- retain configs (recalibrated for the detection distribution) ----
def keep_milca(d): return d[1]>0.7
def keep_single(d): return (d[3]>=b_protect) or (d[2]>=a_star)
def keep_typ99(d): return ((d[3]>=b_protect) or (d[2]>=a_star)) and (d[5]<=tau99)
def keep_typ95(d): return ((d[3]>=b_protect) or (d[2]>=a_star)) and (d[5]<=tau95)
def keep_full(d): return ((d[3]>=b_protect) or (d[4]>=b_tta) or (d[2]>=a_star)) and (d[5]<=tau99)
CONFIGS={'milca':keep_milca,'single':keep_single,'typ99':keep_typ99,'typ95':keep_typ95,'full':keep_full}
counts={}
for name,fn in CONFIGS.items():
    D=RES/f'ds_{name}'
    if D.exists(): shutil.rmtree(D)
    (D/'images'/'train').mkdir(parents=True); (D/'labels'/'train').mkdir(parents=True); (D/'images'/'val').mkdir(parents=True); (D/'labels'/'val').mkdir(parents=True)
    n=0
    for base,dets in DET.items():
        ip=str(YO/'ds_a0'/'images'/'train'/base); labs=[d[0] for d in dets if fn(d)]
        shutil.copy(ip,D/'images'/'train'/base); (D/'labels'/'train'/(base[:-4]+'.txt')).write_text("\n".join(labs)); n+=len(labs)
    for vp in glob.glob(str(YO/'ds_a0'/'images'/'val'/'*.jpg')):
        b=os.path.basename(vp); shutil.copy(vp,D/'images'/'val'/b); shutil.copy(vp.replace('/images/','/labels/').replace('.jpg','.txt'),D/'labels'/'val'/(b[:-4]+'.txt'))
    counts[name]=n
print('retain label counts:',counts,flush=True)
# ---- train D2 per config (1-seed screen) + band eval ----
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
                for g in opt.param_groups: g['lr']=BASE_LR*(0.01+0.99*it/WARMUP)
            imgs=[im.to(dev) for im in imgs]; tgts=[{k:v.to(dev) for k,v in t.items()} for t in tgts]
            loss=sum(model(imgs,tgts).values()); opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(params,5.0); opt.step(); it+=1
        if it>=WARMUP: sched.step()
    return model
def iou(a,b):
    ix1,iy1=max(a[0],b[0]),max(a[1],b[1]); ix2,iy2=min(a[2],b[2]),min(a[3],b[3]); iw,ih=max(0,ix2-ix1),max(0,iy2-iy1); it=iw*ih
    ua=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-it; return it/ua if ua>0 else 0.0
@torch.no_grad()
def evaluate(model,root):
    ds=DetDS(root,'val'); model.eval(); cropgt=[len(ds[i][1]['boxes']) for i in range(len(ds))]; med=np.median([c for c in cropgt if c>0]) if any(cropgt) else 0
    at={0.3:[],0.5:[]}; sc=[]; ng=0; btp={'low':[],'high':[]}; bsc={'low':[],'high':[]}; bg={'low':0,'high':0}
    dl=torch.utils.data.DataLoader(ds,batch_size=BATCH,shuffle=False,collate_fn=collate,num_workers=6); idx=0
    for imgs,tgts in dl:
        for pred,tgt in zip(model([im.to(dev) for im in imgs]),tgts):
            gt=tgt['boxes'].numpy().tolist(); ng+=len(gt); bd='low' if cropgt[idx]<=med else 'high'; bg[bd]+=len(gt); idx+=1
            bxs=pred['boxes'].cpu().numpy(); scs=pred['scores'].cpu().numpy(); order=np.argsort(-scs)
            for th in (0.3,0.5):
                m=set()
                for j in order:
                    best=-1;bj=-1
                    for jg,g in enumerate(gt):
                        if jg in m: continue
                        v=iou(bxs[j],g)
                        if v>best: best,bj=v,jg
                    tp=1 if best>=th and bj>=0 else 0
                    if tp: m.add(bj)
                    at[th].append(tp)
                    if th==0.3: sc.append(float(scs[j])); btp[bd].append(tp); bsc[bd].append(float(scs[j]))
    def ap(t,s,n):
        if not t: return 0.0
        o=np.argsort(-np.array(s)); t=np.array(t)[o]; ct=np.cumsum(t); cf=np.cumsum(1-t); r=ct/max(1,n); p=ct/np.maximum(1,ct+cf)
        mr=np.r_[0,r,1]; mp=np.r_[0,p,0]
        for i in range(len(mp)-1,0,-1): mp[i-1]=max(mp[i-1],mp[i])
        j=np.where(mr[1:]!=mr[:-1])[0]; return float(np.sum((mr[j+1]-mr[j])*mp[j+1]))
    return dict(**{f'AP@{t}':round(ap(at[t],sc,ng),3) for t in (0.3,0.5)},low=round(ap(btp['low'],bsc['low'],bg['low']),3),high=round(ap(btp['high'],bsc['high'],bg['high']),3))
res={}
for name in ['milca','single','typ99','typ95','full']:
    torch.manual_seed(0); np.random.seed(0)
    print(f'=== train D2: {name} ({counts[name]} labels) ===',flush=True); m=train(build_ret(),RES/f'ds_{name}'); r=evaluate(m,RES/f'ds_{name}'); r['n_label']=counts[name]; res[name]=r; print('  ->',r,flush=True)
json.dump(dict(results=res,counts=counts,tau99=tau99,tau95=tau95),open(RES/'adaptive_results.json','w'),indent=2)
print('\n=== ADAPTIVE bootstrap retain (1-seed screen) — ref: single-signal AP@0.5 0.206, low 0.243; conf>0.7 0.134 ===')
for name in ['milca','single','typ99','typ95','full']: print(f'  {name:8s} {res[name]}')
print('S7_ADAPTIVE_DONE',flush=True)
