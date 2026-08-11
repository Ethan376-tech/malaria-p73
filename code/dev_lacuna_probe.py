# -*- coding: utf-8 -*-
"""BLOCK A: external validation on Lacuna (Ghana, P. falciparum thick smears).
Parasite-vs-background tile linear-probe AUC, mirroring Section 4.1 Table 4.1, extended to a THIRD independent
site. Directions: in-Lacuna 5-fold (representation quality on Ghana); B->Lacuna (Chittagong vivax -> Ghana Pf,
cross-species); A->Lacuna (Ibadan Pf -> Ghana Pf, SAME species -> isolates the stain/site vs species confound).
Encoders: ThickDINO vs RedDino-adapt (+ from-scratch baseline)."""
import os, sys, glob, json, random
os.environ['XFORMERS_DISABLED']='1'
sys.path.insert(0,'/root/autodl-tmp/A_Dissertation/references/RedDino/train'); sys.path.append('/root/A_Dissertation')
import numpy as np, pandas as pd, torch
import torchvision.transforms as T
from pathlib import Path
from PIL import Image; Image.MAX_IMAGE_PIXELS=None
from common.paths import RAW, SPLITS, CHECKPOINTS
from dinov2.models.vision_transformer import vit_small
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
random.seed(0); np.random.seed(0); dev='cuda'; HALF=112
norm=T.Compose([T.ToTensor(),T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
ENC={'ThickDINO':'ssl_vits_supvit_dinoadapt','RedDino-adapt':'ssl_vits_reddino_dinoadapt','from-scratch':'ssl_vits_fromscratch'}

def load_enc(name):
    sd=torch.load(CHECKPOINTS/ENC[name]/'encoder_final.pth',map_location='cpu')
    pw=sd.get('patch_embed.proj.weight'); p=int(pw.shape[-1]) if pw is not None else 16   # infer patch size
    m=vit_small(patch_size=p,img_size=224,block_chunks=0,init_values=1.0); m.load_state_dict(sd,strict=False)
    return m.eval().to(dev)

def crop(im,cx,cy):
    W,H=im.size; cx=int(min(max(cx,HALF),W-HALF)); cy=int(min(max(cy,HALF),H-HALF)); return im.crop((cx-HALF,cy-HALF,cx+HALF,cy+HALF))
def far(cx,cy,cents,r=HALF+45): return all(abs(bx-cx)>r or abs(by-cy)>r for bx,by in cents)  # tile (224) truly free of any box
def sample_img(im, cents, kp=14, kb=14):
    """kp parasite tiles (box centres) + kb background tiles (tile fully free of any parasite). -> list[(tile,label)]"""
    W,H=im.size; out=[]
    random.shuffle(cents)
    for cx,cy in cents[:kp]: out.append((np.asarray(crop(im,cx,cy).resize((224,224)),np.uint8),1))
    g=0
    for _ in range(kb*80):
        rx,ry=random.randint(HALF,W-HALF),random.randint(HALF,H-HALF)
        if not cents or far(rx,ry,cents): out.append((np.asarray(crop(im,rx,ry).resize((224,224)),np.uint8),0)); g+=1
        if g>=kb: break
    return out

# ---------- build tile sets per site ----------
def lacuna_tiles(nimg=150):
    D=Path('/root/autodl-tmp/A_Dissertation/data/raw/lacuna/Ghana/Thick')
    imgs=sorted(glob.glob(str(D/'images'/'*.jpg')), key=lambda p:int(Path(p).stem)); random.Random(1).shuffle(imgs)
    T_,L_=[],[]
    for ip in imgs[:nimg]:
        lp=D/'labels_yolo'/(Path(ip).stem+'.txt')
        if not lp.exists(): continue
        im=Image.open(ip).convert('RGB'); W,H=im.size; cents=[]
        for ln in lp.read_text().splitlines():
            f=ln.split()
            if len(f)==5 and f[0]=='0': cents.append((float(f[1])*W,float(f[2])*H))     # class 0 = parasite
        if not cents: continue
        for t,l in sample_img(im,cents): T_.append(t); L_.append(l)
    return T_,np.array(L_)

def chitt_tiles():
    s=pd.read_csv(SPLITS/'splits.csv'); c1=s[s.dataset=='chittagong-1']
    def boxes(txt):
        cs=[]
        for ln in Path(txt).read_text(errors='ignore').splitlines()[1:]:
            f=ln.split(',')
            if len(f)>=9 and 'Parasit' in f[1]:
                try: x1,y1,x2,y2=map(float,f[5:9]); cs.append(((x1+x2)/2,(y1+y2)/2))
                except ValueError: pass
        return cs
    T_,L_=[],[]
    for r in c1.itertuples():
        img_dir=RAW/r.rel_path; ann_dir=str(img_dir).replace('All_PvTk','All_annotations')
        imgs=sorted(glob.glob(str(img_dir)+'/*.jpg')); random.shuffle(imgs)
        for ip in imgs[:5]:
            stem=os.path.splitext(os.path.basename(ip))[0]; af=os.path.join(ann_dir,stem+'.txt')
            cents=boxes(af) if os.path.exists(af) else []
            if not cents: continue
            try: im=Image.open(ip).convert('RGB')
            except Exception: continue
            for t,l in sample_img(im,cents): T_.append(t); L_.append(l)
        if len(L_)>=3500: break
    return T_,np.array(L_)

def fastmal_tiles():
    fm=RAW/'ibadan'/'FASTMAL_THICK_FILMS'; T_,L_=[],[]
    for samp in sorted(p for p in fm.iterdir() if p.is_dir()):
        js=list(samp.glob('*.json'))
        if not js: continue
        for fld in json.loads(js[0].read_text())['rois']:
            ip=samp/fld['image_name']
            if not ip.exists(): continue
            cents=[(float(r['x'])+float(r['width'])/2,float(r['y'])+float(r['height'])/2)
                   for r in fld['roi'] if 'PARASITE' in r.get('type','') and 'CROWD' not in r.get('type','')]
            if not cents: continue
            try: im=Image.open(ip).convert('RGB')
            except Exception: continue
            for t,l in sample_img(im,cents): T_.append(t); L_.append(l)
    return T_,np.array(L_)

print('building tiles...',flush=True)
LT,LY=lacuna_tiles(); BT,BY=chitt_tiles(); AT,AY=fastmal_tiles()
print(f'Lacuna {len(LY)} (pos {int(LY.sum())}) | B-Chitt {len(BY)} (pos {int(BY.sum())}) | A-FASTMAL {len(AY)} (pos {int(AY.sum())})',flush=True)

@torch.no_grad()
def encode(model, tiles, bs=256):
    out=[]
    for i in range(0,len(tiles),bs):
        b=torch.stack([norm(Image.fromarray(t)) for t in tiles[i:i+bs]]).to(dev); out.append(model(b).float().cpu().numpy())
    return np.concatenate(out)

def probe(Xtr,ytr,Xte,yte):
    sc=StandardScaler().fit(Xtr); clf=LogisticRegression(max_iter=3000,class_weight='balanced').fit(sc.transform(Xtr),ytr)
    return roc_auc_score(yte, clf.predict_proba(sc.transform(Xte))[:,1])
def cv_auc(X,y,reps=5):
    v=[]
    for rr in range(reps):
        skf=StratifiedKFold(5,shuffle=True,random_state=rr); oof=np.zeros(len(y))
        for tr,va in skf.split(X,y):
            sc=StandardScaler().fit(X[tr]); clf=LogisticRegression(max_iter=3000,class_weight='balanced').fit(sc.transform(X[tr]),y[tr])
            oof[va]=clf.predict_proba(sc.transform(X[va]))[:,1]
        v.append(roc_auc_score(y,oof))
    return np.mean(v),np.std(v)

print(f'\n{"encoder":14s} {"in-B":>7s} {"in-A":>7s} {"in-Lac":>7s} | {"B->A":>6s} {"B->Lac":>7s} {"A->Lac":>7s}')
for name in ENC:
    m=load_enc(name)
    XL=encode(m,LT); XB=encode(m,BT); XA=encode(m,AT)
    inL=cv_auc(XL,LY)[0]; inB=cv_auc(XB,BY)[0]; inA=cv_auc(XA,AY)[0]     # in-domain sanity (same pipeline)
    b2a=probe(XB,BY,XA,AY); b2l=probe(XB,BY,XL,LY); a2l=probe(XA,AY,XL,LY)
    print(f'{name:14s} {inB:7.3f} {inA:7.3f} {inL:7.3f} | {b2a:6.3f} {b2l:7.3f} {a2l:7.3f}',flush=True)
    del m; torch.cuda.empty_cache()
print('\nDONE',flush=True)
