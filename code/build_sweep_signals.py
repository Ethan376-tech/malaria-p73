# -*- coding: utf-8 -*-
"""Compute per-box multi-signal signals (MC mean/var, TTA var, typicality Mahalanobis) for EVERY A0 pseudo-box in
ds_a0/train, so a filter can later be swept at ANY threshold cheaply (CPU) and the detector re-trained. This lets us
calibrate the filter for DOWNSTREAM detector AP (not pseudo-box precision). Output -> s6_retinanet/sweep_signals.json"""
import os, sys, json, glob
os.environ['XFORMERS_DISABLED']='1'
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image; Image.MAX_IMAGE_PIXELS=None
sys.path.insert(0,'/root/autodl-tmp/A_Dissertation/references/RedDino/train'); sys.path.append('/root/A_Dissertation')
from common.paths import OUTPUTS, CHECKPOINTS
from dinov2.models.vision_transformer import vit_small
from sklearn.decomposition import PCA; from sklearn.covariance import LedoitWolf
dev='cuda'; TILE=224; CROP=640; T_MC=20
torch.manual_seed(0); np.random.seed(0)
YO=OUTPUTS/'experiments'/'s5_yolo'; RES=OUTPUTS/'experiments'/'s6_retinanet'; S3=OUTPUTS/'experiments'/'s3_filter'
_pb=np.load(S3/'perbox_multi2.npz'); _dev=_pb['field_id'].astype(int)<95; _dtp=(_pb['is_tp']==1)&_dev; _E=_pb['emb'][_dtp]
_mu=_E.mean(0); _sd=_E.std(0)+1e-6; _pca=PCA(40,random_state=0).fit((_E-_mu)/_sd); _lw=LedoitWolf().fit(_pca.transform((_E-_mu)/_sd))
def maha_of(emb): return _lw.mahalanobis(_pca.transform((emb-_mu)/_sd)) if len(emb) else np.array([])
norm=T.Compose([T.ToTensor(),T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
enc=vit_small(patch_size=16,img_size=224,block_chunks=0,init_values=1.0,drop_path_rate=0.1)
enc.load_state_dict(torch.load(CHECKPOINTS/'ssl_vits_thickdino_ft_ibadan'/'encoder.pth',map_location='cpu'),strict=False); enc=enc.to(dev).eval()
head=nn.Sequential(nn.Dropout(0.2),nn.Linear(384,1)).to(dev); head.load_state_dict(torch.load(CHECKPOINTS/'ssl_vits_thickdino_ft_ibadan'/'head.pth',map_location='cpu')); head.eval()
@torch.no_grad()
def mc(tiles,bs=48):
    for mod in list(enc.modules())+list(head.modules()):
        if isinstance(mod,nn.Dropout): mod.train()
    for mod in enc.modules():
        if mod.__class__.__name__=='DropPath': mod.train()
    M=[];V=[]
    for i in range(0,len(tiles),bs):
        xb=torch.stack([norm(Image.fromarray(t)) for t in tiles[i:i+bs]]).to(dev); pr=[]
        for _ in range(T_MC): pr.append(torch.sigmoid(head(enc.forward_features(xb)['x_norm_clstoken']).squeeze(-1)).cpu().numpy())
        A=np.stack(pr); M.append(A.mean(0)); V.append(A.var(0))
    enc.eval(); head.eval(); return (np.concatenate(M),np.concatenate(V)) if M else (np.array([]),np.array([]))
@torch.no_grad()
def emb_tta(tiles,bs=48):
    enc.eval(); head.eval(); EM=[];TV=[]
    for i in range(0,len(tiles),bs):
        xb=torch.stack([norm(Image.fromarray(t)) for t in tiles[i:i+bs]]).to(dev)
        EM.append(enc.forward_features(xb)['x_norm_clstoken'].cpu().numpy())
        views=[xb,torch.flip(xb,[3]),torch.flip(xb,[2]),torch.rot90(xb,1,[2,3]),torch.rot90(xb,2,[2,3]),torch.rot90(xb,3,[2,3])]
        pv=[torch.sigmoid(head(enc.forward_features(v)['x_norm_clstoken']).squeeze(-1)).cpu().numpy() for v in views]
        TV.append(np.stack(pv).var(0))
    return (np.concatenate(EM),np.concatenate(TV)) if EM else (np.zeros((0,384),np.float32),np.array([]))
crops=sorted(glob.glob(str(YO/'ds_a0'/'images'/'train'/'*.jpg')))
data={}; nbox=0
for ci,ip in enumerate(crops):
    arr=np.asarray(Image.open(ip).convert('RGB')); base=os.path.basename(ip)
    lp=ip.replace('/images/','/labels/').replace('.jpg','.txt'); lines=[]
    if os.path.exists(lp): lines=[ln for ln in open(lp).read().splitlines() if ln.strip()]
    if not lines: data[base]=[]; continue
    tiles=[]; meta=[]
    for ln in lines:
        f=ln.split(); cx,cy,w,h=float(f[1])*CROP,float(f[2])*CROP,float(f[3])*CROP,float(f[4])*CROP
        x0=int(min(max(cx-TILE/2,0),CROP-TILE)); y0=int(min(max(cy-TILE/2,0),CROP-TILE)); tiles.append(arr[y0:y0+TILE,x0:x0+TILE]); meta.append(ln)
    pm,pv=mc(tiles); em,tta=emb_tta(tiles); mh=maha_of(em)
    data[base]=[[meta[j],round(float(pm[j]),4),round(float(pv[j]),6),round(float(tta[j]),6),round(float(mh[j]),2)] for j in range(len(meta))]
    nbox+=len(meta)
    if (ci+1)%100==0: print(f'{ci+1}/{len(crops)} crops | boxes {nbox}',flush=True)
json.dump(data,open(RES/'sweep_signals.json','w'))
print(f'total A0 boxes with signals: {nbox} over {len(crops)} crops',flush=True); print('SWEEP_SIGNALS_DONE',flush=True)
