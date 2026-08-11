# -*- coding: utf-8 -*-
"""Bootstrap 95% CI for the cross-site (B->A) AUC, overall and low-parasitaemia, for both the pure-weak bag
decision (PW) and the parasite-aware top-k decision (TK). This is the review's #1 ask: put a CI on the n~=37
low-para ranking claim so 'broken/unrecoverable' can be softened to 'no usable ranking signal detected'.
Reuses the ws1a source-free pipeline. Output: printed CIs (+ 3-seed point range)."""
import os, sys, glob, random
os.environ['XFORMERS_DISABLED']='1'
sys.path.insert(0,'/root/autodl-tmp/A_Dissertation/references/RedDino/train'); sys.path.append('/root/A_Dissertation')
import numpy as np, pandas as pd, torch
import torchvision.transforms as T
from pathlib import Path
from PIL import Image; Image.MAX_IMAGE_PIXELS=None
from common.paths import RAW, SPLITS, CHECKPOINTS, OUTPUTS
from dinov2.models.vision_transformer import vit_small
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
random.seed(0); np.random.seed(0); dev='cuda'; HALF=112
norm=T.Compose([T.ToTensor(),T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
D=OUTPUTS/'experiments'/'stageD_bag'; meta=pd.read_csv(D/'bag_meta_v2.csv')
SEEDF={0:'emb_thickdino_v2.npy',1:'emb_thickdino_v2_s1.npy',2:'emb_thickdino_v2_s2.npy'}
EMB={s:np.load(D/f) for s,f in SEEDF.items()}
y=meta.label01.values; dom=meta.dataset.values; band=meta.para_band.values
A=(dom=='ibadan'); B=(dom=='chittagong-1'); lo=A&((band=='neg')|(band=='pos_lo')); loA=lo[A]
def dv(sd,patch):
    m=vit_small(patch_size=patch,img_size=224,block_chunks=0,init_values=1.0); m.load_state_dict(sd,strict=False); return m.eval().to(dev)
thick=dv(torch.load(CHECKPOINTS/'ssl_vits_supvit_dinoadapt'/'encoder_final.pth',map_location='cpu'),16)
s=pd.read_csv(SPLITS/'splits.csv'); c1=s[s.dataset=='chittagong-1']
def boxes(txt):
    cs=[]
    for ln in txt.read_text(errors='ignore').splitlines()[1:]:
        f=ln.split(',')
        if len(f)>=9 and 'Parasit' in f[1]:
            try: x1,y1,x2,y2=map(float,f[5:9]); cs.append(((x1+x2)/2,(y1+y2)/2))
            except ValueError: pass
    return cs
def cropc(im,cx,cy):
    W,H=im.size; cx=int(min(max(cx,HALF),W-HALF)); cy=int(min(max(cy,HALF),H-HALF)); return im.crop((cx-HALF,cy-HALF,cx+HALF,cy+HALF))
def frei(cx,cy,c): return all(abs(bx-cx)>HALF or abs(by-cy)>HALF for bx,by in c)
tiles=[]; labs=[]
for r in c1.itertuples():
    img_dir=RAW/r.rel_path; ann_dir=str(img_dir).replace('All_PvTk','All_annotations')
    imgs=sorted(glob.glob(str(img_dir)+'/*.jpg')); random.shuffle(imgs)
    for ip in imgs[:6]:
        stem=os.path.splitext(os.path.basename(ip))[0]; af=os.path.join(ann_dir,stem+'.txt')
        cents=boxes(Path(af)) if os.path.exists(af) else []
        try: im=Image.open(ip).convert('RGB')
        except Exception: continue
        W,H=im.size
        for cx,cy in cents[:20]: tiles.append(np.asarray(cropc(im,cx,cy),np.uint8)); labs.append(1)
        g=0
        for _ in range(40):
            rx,ry=random.randint(HALF,W-HALF),random.randint(HALF,H-HALF)
            if not cents or frei(rx,ry,cents): tiles.append(np.asarray(cropc(im,rx,ry),np.uint8)); labs.append(0); g+=1
            if g>=max(3,len(cents[:20])): break
    if sum(labs)>=4000: break
labs=np.array(labs)
@torch.no_grad()
def enc_tiles(model,bs=256):
    out=[]
    for i in range(0,len(tiles),bs):
        b=torch.stack([norm(Image.fromarray(t)) for t in tiles[i:i+bs]]).to(dev); out.append(model(b).float().cpu().numpy())
    return np.concatenate(out)
Xtile=enc_tiles(thick); tsc=StandardScaler().fit(Xtile); tlr=LogisticRegression(max_iter=3000,class_weight='balanced').fit(tsc.transform(Xtile),labs)
def run(feat, seed):
    fB=feat[B]; yB=y[B]; fA=feat[A]
    skf=StratifiedKFold(5,shuffle=True,random_state=seed); tgt=np.zeros(A.sum())
    for tr,va in skf.split(fB,yB):
        sc=StandardScaler().fit(fB[tr]); clf=LogisticRegression(max_iter=3000,C=0.3,class_weight='balanced').fit(sc.transform(fB[tr]),yB[tr])
        tgt+=clf.predict_proba(sc.transform(fA))[:,1]/5
    return tgt
def pw_feat(seed): return EMB[seed].max(1)
def tk_feat(seed,k=10):
    E=EMB[seed]; Pt=tlr.predict_proba(tsc.transform(E.reshape(-1,384)))[:,1].reshape(E.shape[0],256)
    return np.sort(Pt,1)[:,::-1][:,:k].mean(1,keepdims=True)
def auc_ci(yt,sc,reps=3000):
    yt=np.asarray(yt); sc=np.asarray(sc); n=len(yt); rng=np.random.RandomState(0); v=[]
    for _ in range(reps):
        idx=rng.randint(0,n,n)
        if len(set(yt[idx]))<2: continue
        v.append(roc_auc_score(yt[idx],sc[idx]))
    return float(np.mean(v)),float(np.percentile(v,2.5)),float(np.percentile(v,97.5))
print(f'n: A={A.sum()} A-pos={y[A].sum()} | low-para n={loA.sum()} low-para-pos={int(y[A][loA].sum())}',flush=True)
for name,ff in [('PW pure-weak bag',pw_feat),('TK top-k parasite',tk_feat)]:
    ov3=[]; lw3=[]
    for seed in SEEDF:
        sA=run(ff(seed),seed); ov3.append(roc_auc_score(y[A],sA)); lw3.append(roc_auc_score(y[A][loA],sA[loA]))
    sA0=run(ff(0),0)
    om,olo,ohi=auc_ci(y[A],sA0); lm,llo,lhi=auc_ci(y[A][loA],sA0[loA])
    print(f'\n{name}',flush=True)
    print(f'  overall  AUC 3-seed {np.mean(ov3):.3f} (range {min(ov3):.3f}-{max(ov3):.3f}) | boot95CI [{olo:.3f}, {ohi:.3f}]',flush=True)
    print(f'  low-para AUC 3-seed {np.mean(lw3):.3f} (range {min(lw3):.3f}-{max(lw3):.3f}) | boot95CI [{llo:.3f}, {lhi:.3f}]',flush=True)
print('\nDONE',flush=True)
