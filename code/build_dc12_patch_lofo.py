# -*- coding: utf-8 -*-
"""LOFO GT-supervised Patch head (frozen SupViT tokens + logistic) for Table 4.7, custom AP (== build_s4). For each test
film f, train the logistic head on GT patches from all OTHER films, evaluate on film f's test fields; aggregate custom AP
over the full report test set. Output -> dc_lofo_gtsup/patch.json"""
import os, sys, json, random
os.environ['XFORMERS_DISABLED']='1'
import numpy as np, torch, torch.nn as nn
import torchvision.transforms as T
from PIL import Image; Image.MAX_IMAGE_PIXELS=None
sys.path.insert(0,'/root/autodl-tmp/A_Dissertation/references/RedDino/train'); sys.path.append('/root/A_Dissertation')
from common.paths import RAW, CHECKPOINTS, OUTPUTS
from dinov2.models.vision_transformer import vit_small
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
dev='cuda'; TILE=224; P=16; G=TILE//P
random.seed(0); np.random.seed(0); torch.manual_seed(0)
CAL=json.load(open(OUTPUTS/'experiments'/'s2_calib'/'calib_constants.json'))['A']
BS=CAL['box_size_px']; MD=CAL['peak_min_dist_px']
LOFO=OUTPUTS/'experiments'/'dc_lofo_gtsup'
norm=T.Compose([T.ToTensor(),T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
def mk(ckpt):
    m=vit_small(patch_size=16,img_size=224,block_chunks=0,init_values=1.0,drop_path_rate=0.0)
    m.load_state_dict(torch.load(ckpt,map_location='cpu'),strict=False); return m.to(dev).eval()
frozen=mk(CHECKPOINTS/'ssl_vits_supvit_dinoadapt'/'encoder_final.pth')
def toff(im):
    W,H=im.size; arr=np.asarray(im); xs=sorted(set(list(range(0,max(1,W-TILE+1),TILE))+[max(0,W-TILE)])); ys=sorted(set(list(range(0,max(1,H-TILE+1),TILE))+[max(0,H-TILE)]))
    return [(arr[ty:ty+TILE,tx:tx+TILE],(tx,ty)) for ty in ys for tx in xs],arr,W,H
@torch.no_grad()
def frozen_patches(crops,bs=48):
    out=[]
    for i in range(0,len(crops),bs):
        xb=torch.stack([norm(Image.fromarray(c)) for c in crops[i:i+bs]]).to(dev)
        out.append(frozen.forward_features(xb)['x_norm_patchtokens'].cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0,G*G,384),np.float32)
def nms(c,md):
    c=sorted(c,key=lambda z:-z[2]); k=[]
    for cx,cy,s in c:
        if all((cx-kx)**2+(cy-ky)**2>=md*md for kx,ky,_ in k): k.append((cx,cy,s))
    return k
def iou(a,b):
    ix1,iy1=max(a[0],b[0]),max(a[1],b[1]); ix2,iy2=min(a[2],b[2]),min(a[3],b[3]); iw,ih=max(0,ix2-ix1),max(0,iy2-iy1); it=iw*ih
    ua=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-it; return it/ua if ua>0 else 0.0
def fastmal_fields():
    fm=RAW/'ibadan'/'FASTMAL_THICK_FILMS'; o=[]
    for samp in sorted(p for p in fm.iterdir() if p.is_dir()):
        js=list(samp.glob('*.json'))
        if not js: continue
        for fld in json.loads(js[0].read_text())['rois']:
            ip=samp/fld['image_name']
            if not ip.exists(): continue
            b=[(float(r['x']),float(r['y']),float(r['width']),float(r['height'])) for r in fld['roi'] if 'PARASITE' in r.get('type','') and 'CROWD' not in r.get('type','')]
            if b: o.append((samp.name,ip,b))
    return o
allf=fastmal_fields()
keyed=sorted(allf,key=lambda z:(z[0],z[1].name)); random.Random(0).shuffle(keyed)
testF=keyed[1::2][:40]; test_films=sorted(set(f for f,_,_ in testF))
def collect_gt_patches(fset):
    X=[];Y=[]
    for film,ip,gtb in fset:
        try: im=Image.open(ip).convert('RGB')
        except Exception: continue
        cents=[(x+w/2,y+h/2) for (x,y,w,h) in gtb]; tiles,arr,W,H=toff(im); crops=[c for c,_ in tiles]; offs=[o for _,o in tiles]; toks=frozen_patches(crops)
        for (tx,ty),tk in zip(offs,toks):
            lab=np.zeros(G*G,bool); hb=False
            for (cx,cy) in cents:
                if tx<=cx<tx+TILE and ty<=cy<ty+TILE:
                    gx=int((cx-tx)//P); gy=int((cy-ty)//P)
                    for dy in(-1,0,1):
                        for dx in(-1,0,1):
                            nx,ny=gx+dx,gy+dy
                            if 0<=nx<G and 0<=ny<G: lab[ny*G+nx]=True; hb=True
            if hb or random.random()<0.15:
                pos=np.where(lab)[0]; neg=np.where(~lab)[0]; sel=list(pos)+list(np.random.choice(neg,min(len(neg),max(4,2*len(pos))),replace=False))
                for pidx in sel: X.append(tk[pidx]); Y.append(int(lab[pidx]))
    return np.array(X),np.array(Y)
# cache patches per FILM (train pool)
allfilms=sorted(set(f for f,_,_ in allf))
film_patches={f:collect_gt_patches([x for x in allf if x[0]==f]) for f in allfilms}
print('cached patches per film',flush=True)
def train_head(X,Y):
    sc=StandardScaler().fit(X); lr=LogisticRegression(max_iter=3000,class_weight='balanced').fit(sc.transform(X),Y); return sc,lr
def detect_field(im,sc,lr):
    tiles,arr,W,H=toff(im); crops=[c for c,_ in tiles]; offs=[o for _,o in tiles]; toks=frozen_patches(crops); cand=[]
    for (tx,ty),tk in zip(offs,toks):
        p=lr.predict_proba(sc.transform(tk))[:,1]
        for g in range(G*G):
            gx,gy=g%G,g//G; cand.append((tx+gx*P+P/2,ty+gy*P+P/2,float(p[g])))
    return nms(cand,MD)
# LOFO
fcnts=[len(gtb) for _,_,gtb in testF]; thr=np.median([c for c in fcnts if c>0]) if any(fcnts) else 0
all_tp={0.3:[],0.5:[]}; sc_all=[]; n_gt=0; tp03b={'low':[],'high':[]}; scb={'low':[],'high':[]}; gtb_={'low':0,'high':0}
for f in test_films:
    Xtr=np.concatenate([film_patches[g][0] for g in allfilms if g!=f and len(film_patches[g][0])])
    Ytr=np.concatenate([film_patches[g][1] for g in allfilms if g!=f and len(film_patches[g][1])])
    sc,lr=train_head(Xtr,Ytr)
    for film,ip,gtb in [x for x in testF if x[0]==f]:
        gt=[(x,y,x+w,y+h) for (x,y,w,h) in gtb]; n_gt+=len(gt); band='low' if len(gt)<=thr else 'high'; gtb_[band]+=len(gt)
        im=Image.open(ip).convert('RGB'); peaks=detect_field(im,sc,lr)
        boxes=[(cx-BS/2,cy-BS/2,cx+BS/2,cy+BS/2,s) for (cx,cy,s) in peaks]; boxes.sort(key=lambda z:-z[4])
        for th in (0.3,0.5):
            m=set()
            for b in boxes:
                bst=-1;bj=-1
                for jg,g in enumerate(gt):
                    if jg in m: continue
                    v=iou(b[:4],g)
                    if v>bst: bst,bj=v,jg
                tp=1 if bst>=th and bj>=0 else 0
                if tp:m.add(bj)
                all_tp[th].append(tp)
                if th==0.3: sc_all.append(b[4]); tp03b[band].append(tp); scb[band].append(b[4])
    print(f'  patch fold excl {f} done',flush=True)
def ap(t,s,n):
    if not t:return 0.0
    o=np.argsort(-np.array(s)); t=np.array(t)[o]; ct=np.cumsum(t);cf=np.cumsum(1-t);rc=ct/max(1,n);pr=ct/np.maximum(1,ct+cf)
    mr=np.r_[0,rc,1];mp=np.r_[0,pr,0]
    for i in range(len(mp)-1,0,-1):mp[i-1]=max(mp[i-1],mp[i])
    k=np.where(mr[1:]!=mr[:-1])[0];return round(float(np.sum((mr[k+1]-mr[k])*mp[k+1])),3)
res=dict(**{f'AP@{t}':ap(all_tp[t],sc_all,n_gt) for t in (0.3,0.5)}, AP03_low=ap(tp03b['low'],scb['low'],gtb_['low']), AP03_high=ap(tp03b['high'],scb['high'],gtb_['high']), n_gt=n_gt)
json.dump(res, open(LOFO/'patch.json','w'), indent=2)
print('PATCH LOFO clean GT-sup:',res,flush=True); print('DC12_DONE',flush=True)
