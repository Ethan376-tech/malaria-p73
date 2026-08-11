# -*- coding: utf-8 -*-
"""MILCA-style hard-negative mining data: mine parasite-like artifacts from NEGATIVE Ibadan bags (parasite-free
fields), write 640px crops containing them with EMPTY labels (hard negatives), and combine with the ds_a0 positives
into ds_a0_hnm. Also records per-crop min-typicality (Mahalanobis to dev-TP) so a typicality-guided variant can be
built later. Output -> s5_yolo/ds_a0_hnm/ + data_a0_hnm.yaml + hnm_crop_typicality.json"""
import os, sys, json, random, shutil, glob
os.environ['XFORMERS_DISABLED']='1'
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image; Image.MAX_IMAGE_PIXELS=None
sys.path.insert(0,'/root/autodl-tmp/A_Dissertation/references/RedDino/train'); sys.path.append('/root/A_Dissertation')
from common.paths import RAW, CHECKPOINTS, OUTPUTS, IBADAN_PART1, IBADAN_PART2
from dinov2.models.vision_transformer import vit_small
from sklearn.decomposition import PCA; from sklearn.covariance import LedoitWolf
dev='cuda'; TILE=224; P=16; G=TILE//P; CROP=640
random.seed(0); np.random.seed(0); torch.manual_seed(0)
CAL=json.load(open(OUTPUTS/'experiments'/'s2_calib'/'calib_constants.json'))['A']
FIN=json.load(open(OUTPUTS/'experiments'/'s3_filter'/'final.json'))['calib']
gate_thr=FIN['gate_thr']; cam_thr=FIN['cam_thr']; BS=CAL['box_size_px']; MD=CAL['peak_min_dist_px']
OUT=OUTPUTS/'experiments'/'s5_yolo'
# typicality manifold (dev-TP ThickDINO embeddings)
_pb=np.load(OUTPUTS/'experiments'/'s3_filter'/'perbox_multi2.npz'); _dev=_pb['field_id'].astype(int)<95; _dtp=(_pb['is_tp']==1)&_dev; _E=_pb['emb'][_dtp]
_mu=_E.mean(0); _sd=_E.std(0)+1e-6; _pca=PCA(40,random_state=0).fit((_E-_mu)/_sd); _lw=LedoitWolf().fit(_pca.transform((_E-_mu)/_sd))
def maha_of(emb): return _lw.mahalanobis(_pca.transform((emb-_mu)/_sd)) if len(emb) else np.array([])
norm=T.Compose([T.ToTensor(),T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
def mk(ckpt):
    m=vit_small(patch_size=16,img_size=224,block_chunks=0,init_values=1.0,drop_path_rate=0.1)
    m.load_state_dict(torch.load(ckpt,map_location='cpu'),strict=False); return m.to(dev).eval()
genc=mk(CHECKPOINTS/'ssl_vits_thickdino_ft_ibadan'/'encoder.pth'); ghead=nn.Sequential(nn.Dropout(0.2),nn.Linear(384,1)).to(dev)
ghead.load_state_dict(torch.load(CHECKPOINTS/'ssl_vits_thickdino_ft_ibadan'/'head.pth',map_location='cpu')); ghead.eval()
acts={}
def hook(mod,i,o):
    if o.requires_grad: o.retain_grad()
    acts['o']=o
genc.blocks[-4].register_forward_hook(hook)
def toff(im):
    W,H=im.size; arr=np.asarray(im); xs=sorted(set(list(range(0,max(1,W-TILE+1),TILE))+[max(0,W-TILE)])); ys=sorted(set(list(range(0,max(1,H-TILE+1),TILE))+[max(0,H-TILE)]))
    return [(arr[ty:ty+TILE,tx:tx+TILE],(tx,ty)) for ty in ys for tx in xs],arr,W,H
def gen_logit_cam(crops,bs=24):
    lo=[];ca=[]
    for i in range(0,len(crops),bs):
        xb=torch.stack([norm(Image.fromarray(c)) for c in crops[i:i+bs]]).to(dev).requires_grad_(True)
        ff=genc.forward_features(xb); s=ghead(ff['x_norm_clstoken']).squeeze(-1); genc.zero_grad(); ghead.zero_grad(); s.sum().backward()
        o=acts['o']; al=o.grad[:,1:,:].mean(1,keepdim=True); cam=F.relu((o[:,1:,:].detach()*al).sum(-1))
        lo.append(s.detach().cpu().numpy()); ca.append(cam.detach().cpu().numpy())
    return np.concatenate(lo),np.concatenate(ca)
@torch.no_grad()
def emb_of(crops,bs=48):
    genc.eval(); E=[]
    for i in range(0,len(crops),bs):
        xb=torch.stack([norm(Image.fromarray(c)) for c in crops[i:i+bs]]).to(dev)
        E.append(genc.forward_features(xb)['x_norm_clstoken'].cpu().numpy())
    return np.concatenate(E) if E else np.zeros((0,384),np.float32)
def nms(c,md):
    c=sorted(c,key=lambda z:-z[2]); k=[]
    for cx,cy,s in c:
        if all((cx-kx)**2+(cy-ky)**2>=md*md for kx,ky,_ in k): k.append((cx,cy,s))
    return k
def field_peaks(im):
    tiles,arr,W,H=toff(im); crops=[c for c,_ in tiles]; offs=[o for _,o in tiles]; lo,ca=gen_logit_cam(crops); cand=[]
    for (tx,ty),lg,cm in zip(offs,lo,ca):
        if lg<gate_thr: continue
        sg=cm.reshape(G,G)
        for gy in range(G):
            for gx in range(G):
                if sg[gy,gx]>=cam_thr: cand.append((tx+gx*P+P/2,ty+gy*P+P/2,float(sg[gy,gx])))
    peaks=nms(cand,MD); peaks=sorted(peaks,key=lambda z:-z[2])[:300]
    return peaks,arr,W,H
def crop_grid(W,H):
    xs=list(range(0,max(1,W-CROP+1),CROP)) + ([W-CROP] if W>CROP else [0]); ys=list(range(0,max(1,H-CROP+1),CROP)) + ([H-CROP] if H>CROP else [0])
    return sorted(set(xs)),sorted(set(ys))
# ---- negative Ibadan bags (parasite-free) ----
bm=pd.read_csv(OUTPUTS/'experiments'/'stageD_bag'/'bag_meta_v2.csv')
ibcsv=pd.read_csv('/root/autodl-tmp/A_Dissertation/data/raw/ibadan/sample_codes_parasite_diagnosis_crosscheck.csv')
negids=set(bm[(bm.dataset=='ibadan')&(bm.label01==0)].sample_id)
sampdir={r.sample_code:(IBADAN_PART1 if 'part1' in r.input_file else IBADAN_PART2)/r.sample_code for _,r in ibcsv.iterrows()}
EXT={'.tif','.tiff','.png','.jpg','.jpeg'}; neg_fields=[]
for sid in sorted(negids):
    d=sampdir.get(sid)
    if d is None or not d.exists(): continue
    neg_fields+=[p for p in d.rglob('*') if p.suffix.lower() in EXT][:2]
    if len(neg_fields)>=80: break
random.Random(0).shuffle(neg_fields); neg_fields=neg_fields[:70]
print(f'negative Ibadan fields for HNM: {len(neg_fields)}',flush=True)
# ---- ds_a0_hnm = copy ds_a0 (positives + val), then add hard-neg crops ----
DST=OUT/'ds_a0_hnm'
if DST.exists(): shutil.rmtree(DST)
shutil.copytree(OUT/'ds_a0', DST)
npos=len(glob.glob(str(DST/'images'/'train'/'*.jpg')))
nhn=0; crop_typ={}
for k,ip in enumerate(neg_fields):
    try: im=Image.open(ip).convert('RGB')
    except Exception: continue
    peaks,arr,W,H=field_peaks(im)
    if not peaks: continue
    xs,ys=crop_grid(W,H)
    for cxo in xs:
        for cyo in ys:
            inpk=[(px,py) for (px,py,s) in peaks if cxo<=px<cxo+CROP and cyo<=py<cyo+CROP]
            if not inpk: continue
            sub=arr[cyo:cyo+CROP, cxo:cxo+CROP]
            if sub.shape[0]<CROP or sub.shape[1]<CROP:
                pad=np.zeros((CROP,CROP,3),np.uint8); pad[:sub.shape[0],:sub.shape[1]]=sub; sub=pad
            # typicality of the artifacts in this crop (min Mahalanobis = most parasite-like = hardest)
            tiles=[]
            for (px,py) in inpk:
                x0=int(min(max(px-TILE/2,0),W-TILE)); y0=int(min(max(py-TILE/2,0),H-TILE)); tiles.append(arr[y0:y0+TILE,x0:x0+TILE])
            mh=maha_of(emb_of(tiles)); min_maha=float(mh.min()) if len(mh) else 1e9
            name=f'hn{k}_{cxo}_{cyo}'; Image.fromarray(sub).save(DST/'images'/'train'/f'{name}.jpg',quality=92)
            (DST/'labels'/'train'/f'{name}.txt').write_text("")   # EMPTY label = hard negative image
            crop_typ[name]=round(min_maha,2); nhn+=1
    if (k+1)%20==0: print(f'  neg field {k+1}/{len(neg_fields)} | hard-neg crops {nhn}',flush=True)
print(f'ds_a0_hnm: {npos} positive crops + {nhn} hard-negative crops',flush=True)
(OUT/'data_a0_hnm.yaml').write_text(f"path: {DST}\ntrain: images/train\nval: images/val\nnc: 1\nnames: ['parasite']\n")
json.dump(crop_typ,open(OUT/'hnm_crop_typicality.json','w'))
print('HNM_DATA_DONE',flush=True)
