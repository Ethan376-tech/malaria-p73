# -*- coding: utf-8 -*-
"""Stage 3 data prep — build YOLO datasets from §2.3 pseudo-labels (domain A). 640px crops of Ibadan bag fields with
A0 vs faintness-PROTECTED pseudo-boxes (and a GT-supervised set from FASTMAL-dev real boxes); shared val = FASTMAL-test
crops with real GT. Output -> outputs/experiments/s5_yolo/ds_{a0,protected,gtsup}/ + data_*.yaml"""
import os, sys, json, random, shutil
os.environ['XFORMERS_DISABLED']='1'
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image; Image.MAX_IMAGE_PIXELS=None
sys.path.insert(0,'/root/autodl-tmp/A_Dissertation/references/RedDino/train'); sys.path.append('/root/A_Dissertation')
from common.paths import RAW, CHECKPOINTS, OUTPUTS, IBADAN_PART1, IBADAN_PART2
from dinov2.models.vision_transformer import vit_small
dev='cuda'; TILE=224; P=16; G=TILE//P; T_MC=20; CROP=640
random.seed(0); np.random.seed(0); torch.manual_seed(0)
CAL=json.load(open(OUTPUTS/'experiments'/'s2_calib'/'calib_constants.json'))['A']
FIN=json.load(open(OUTPUTS/'experiments'/'s3_filter'/'final.json'))['calib']
OUT=OUTPUTS/'experiments'/'s5_yolo'
gate_thr=FIN['gate_thr']; cam_thr=FIN['cam_thr']; BS=CAL['box_size_px']; MD=CAL['peak_min_dist_px']
from sklearn.decomposition import PCA; from sklearn.covariance import LedoitWolf
cj=json.load(open(OUTPUTS/'experiments'/'s3_filter'/'joint_calib.json'))['orig']['joint']['cal']; bMC=cj['bMC']; bTTA=cj['bTTA']; a_conf=cj['a']; tau=cj['tau']
_pb=np.load(OUTPUTS/'experiments'/'s3_filter'/'perbox_multi2.npz'); _dev=_pb['field_id'].astype(int)<95; _dtp=(_pb['is_tp']==1)&_dev; _E=_pb['emb'][_dtp]
_mu=_E.mean(0); _sd=_E.std(0)+1e-6; _pca=PCA(40,random_state=0).fit((_E-_mu)/_sd); _lw=LedoitWolf().fit(_pca.transform((_E-_mu)/_sd))
def maha_of(emb): return _lw.mahalanobis(_pca.transform((emb-_mu)/_sd)) if len(emb) else np.array([])
print(f'multi-signal filter thresholds: bMC {bMC:.5f} bTTA {bTTA:.5f} a {a_conf:.4f} tau {tau:.1f}',flush=True)
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
def gen_mc(crops,bs=48):
    for mod in list(genc.modules())+list(ghead.modules()):
        if isinstance(mod,nn.Dropout): mod.train()
    for mod in genc.modules():
        if mod.__class__.__name__=='DropPath': mod.train()
    means=[];vars=[]
    for i in range(0,len(crops),bs):
        xb=torch.stack([norm(Image.fromarray(c)) for c in crops[i:i+bs]]).to(dev); pr=[]
        for _ in range(T_MC): ff=genc.forward_features(xb); pr.append(torch.sigmoid(ghead(ff['x_norm_clstoken']).squeeze(-1)).cpu().numpy())
        A=np.stack(pr); means.append(A.mean(0)); vars.append(A.var(0))
    genc.eval(); ghead.eval(); return np.concatenate(means),np.concatenate(vars)
@torch.no_grad()
def embed_tta(crops,bs=48):
    genc.eval(); ghead.eval(); EMB=[]; TV=[]
    for i in range(0,len(crops),bs):
        xb=torch.stack([norm(Image.fromarray(c)) for c in crops[i:i+bs]]).to(dev)
        ff=genc.forward_features(xb); EMB.append(ff['x_norm_clstoken'].cpu().numpy())
        views=[xb,torch.flip(xb,[3]),torch.flip(xb,[2]),torch.rot90(xb,1,[2,3]),torch.rot90(xb,2,[2,3]),torch.rot90(xb,3,[2,3])]
        pv=[torch.sigmoid(ghead(genc.forward_features(v)['x_norm_clstoken']).squeeze(-1)).cpu().numpy() for v in views]
        TV.append(np.stack(pv).var(0))
    return (np.concatenate(EMB),np.concatenate(TV)) if EMB else (np.zeros((0,384),np.float32),np.array([]))
def nms(c,md):
    c=sorted(c,key=lambda z:-z[2]); k=[]
    for cx,cy,s in c:
        if all((cx-kx)**2+(cy-ky)**2>=md*md for kx,ky,_ in k): k.append((cx,cy,s))
    return k
def pseudo_boxes(im):
    tiles,arr,W,H=toff(im); crops=[c for c,_ in tiles]; offs=[o for _,o in tiles]; lo,ca=gen_logit_cam(crops); cand=[]
    for (tx,ty),lg,cm in zip(offs,lo,ca):
        if lg<gate_thr: continue
        sg=cm.reshape(G,G)
        for gy in range(G):
            for gx in range(G):
                if sg[gy,gx]>=cam_thr: cand.append((tx+gx*P+P/2,ty+gy*P+P/2,float(sg[gy,gx])))
    peaks=nms(cand,MD); peaks=sorted(peaks,key=lambda z:-z[2])[:300]
    if not peaks: return [],[]
    bc=[]
    for (cx,cy,s) in peaks:
        x0=int(min(max(cx-TILE/2,0),W-TILE)); y0=int(min(max(cy-TILE/2,0),H-TILE)); bc.append(arr[y0:y0+TILE,x0:x0+TILE])
    pm,pv=gen_mc(bc); emb,tta=embed_tta(bc); mh=maha_of(emb)
    A0=[(cx,cy) for (cx,cy,s) in peaks]
    prot=[(peaks[j][0],peaks[j][1]) for j in range(len(peaks)) if ((pv[j]>=bMC or tta[j]>=bTTA or pm[j]>=a_conf) and mh[j]<=tau)]
    return A0,prot

def crop_grid(W,H):
    xs=list(range(0,max(1,W-CROP+1),CROP)) + ([W-CROP] if W>CROP else [0]); ys=list(range(0,max(1,H-CROP+1),CROP)) + ([H-CROP] if H>CROP else [0])
    return sorted(set(xs)),sorted(set(ys))
def write_crops(im, boxes_xywh, root, split, tag):
    """boxes_xywh: list of (cx,cy,w,h) in field px. Write 640 crops + YOLO labels."""
    arr=np.asarray(im); H,W=arr.shape[:2]; xs,ys=crop_grid(W,H); n=0
    (root/'images'/split).mkdir(parents=True,exist_ok=True); (root/'labels'/split).mkdir(parents=True,exist_ok=True)
    for cxo in xs:
        for cyo in ys:
            sub=arr[cyo:cyo+CROP, cxo:cxo+CROP]
            if sub.shape[0]<CROP or sub.shape[1]<CROP:
                pad=np.zeros((CROP,CROP,3),np.uint8); pad[:sub.shape[0],:sub.shape[1]]=sub; sub=pad
            labs=[]
            for (cx,cy,w,h) in boxes_xywh:
                lx,ly=cx-cxo,cy-cyo
                if 0<=lx<CROP and 0<=ly<CROP:
                    labs.append(f"0 {lx/CROP:.6f} {ly/CROP:.6f} {w/CROP:.6f} {h/CROP:.6f}")
            if not labs and split=='train': continue   # skip empty train crops
            name=f"{tag}_{cxo}_{cyo}"; Image.fromarray(sub).save(root/'images'/split/f'{name}.jpg',quality=92)
            (root/'labels'/split/f'{name}.txt').write_text("\n".join(labs)); n+=1
    return n

# ---- fields ----
bm=pd.read_csv(OUTPUTS/'experiments'/'stageD_bag'/'bag_meta_v2.csv')
ibcsv=pd.read_csv('/root/autodl-tmp/A_Dissertation/data/raw/ibadan/sample_codes_parasite_diagnosis_crosscheck.csv')
posids=set(bm[(bm.dataset=='ibadan')&(bm.label01==1)].sample_id)
sampdir={r.sample_code:(IBADAN_PART1 if 'part1' in r.input_file else IBADAN_PART2)/r.sample_code for _,r in ibcsv.iterrows()}
EXT={'.tif','.tiff','.png','.jpg','.jpeg'}; train_fields=[]
for sid in sorted(posids):
    d=sampdir.get(sid)
    if d is None or not d.exists(): continue
    train_fields+=[p for p in d.rglob('*') if p.suffix.lower() in EXT][:2]
    if len(train_fields)>=80: break
random.Random(0).shuffle(train_fields); train_fields=train_fields[:70]
def fastmal_fields():
    fm=RAW/'ibadan'/'FASTMAL_THICK_FILMS'; o=[]
    for samp in sorted(p for p in fm.iterdir() if p.is_dir()):
        js=list(samp.glob('*.json'))
        if not js: continue
        for fld in json.loads(js[0].read_text())['rois']:
            ip=samp/fld['image_name']
            if not ip.exists(): continue
            b=[(float(r['x']),float(r['y']),float(r['width']),float(r['height'])) for r in fld['roi'] if 'PARASITE' in r.get('type','') and 'CROWD' not in r.get('type','')]
            if b: o.append((ip,b))
    return o
fields=sorted(fastmal_fields(),key=lambda z:str(z[0])); random.Random(0).shuffle(fields)
devF=fields[0::2][:20]; testF=fields[1::2][:40]
DPR=OUT/'ds_protected_multi'
if DPR.exists(): shutil.rmtree(DPR)
npr=0
for k,ip in enumerate(train_fields):
    try: im=Image.open(ip).convert('RGB')
    except Exception: continue
    a0,pr=pseudo_boxes(im); tag=f'tr{k}'
    npr+=write_crops(im,[(cx,cy,BS,BS) for (cx,cy) in pr],DPR,'train',tag)
    if (k+1)%20==0: print(f'train {k+1}/{len(train_fields)} crops prot_multi {npr}',flush=True)
# val (shared): copy FASTMAL-test GT crops from the existing ds_a0 val
import glob as _g
(DPR/'images'/'val').mkdir(parents=True,exist_ok=True); (DPR/'labels'/'val').mkdir(parents=True,exist_ok=True)
for vp in _g.glob(str(OUT/'ds_a0'/'images'/'val'/'*.jpg')):
    base=os.path.basename(vp); shutil.copy(vp,DPR/'images'/'val'/base); shutil.copy(vp.replace('/images/','/labels/').replace('.jpg','.txt'),DPR/'labels'/'val'/(base[:-4]+'.txt'))
nval=len(_g.glob(str(DPR/'images'/'val'/'*.jpg')))
print(f'train crops: protected_multi {npr} | val crops {nval}',flush=True)
(OUT/'data_protected_multi.yaml').write_text(f"path: {DPR}\ntrain: images/train\nval: images/val\nnc: 1\nnames: ['parasite']\n")
print('YOLO_DATA_MULTI_DONE',flush=True)
