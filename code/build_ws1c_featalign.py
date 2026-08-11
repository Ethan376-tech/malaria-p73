# -*- coding: utf-8 -*-
"""WS1c (representation-level) — source-free feature-distribution alignment of the TARGET tile embeddings to the
SOURCE (domain-B) distribution, then re-score with the same B-trained parasite scorer. Unlike WS1b (decision/threshold
layer, which cannot change the ranking), aligning the 384-d features can change the RANKING itself. Decisive question:
does it raise the broken low-parasitaemia bag AUC (raw 0.47)?

Alignment (source-free, unsupervised): BN/diag (match per-feature mean/std) and CORAL (match mean + covariance) of
target tiles -> source tile distribution (the scorer's training set, C1 box tiles). Re-score -> top-k -> bag/image.
Report AUC (ranking) + BA@true-prior (achievable decision) for raw / BN / CORAL, on B->A overall + low-para and B->C.
Output -> outputs/experiments/ws1c_featalign/results.json
"""
import os, sys, json, glob, random
os.environ['XFORMERS_DISABLED']='1'
sys.path.insert(0,'/root/autodl-tmp/A_Dissertation/references/RedDino/train'); sys.path.append('/root/A_Dissertation')
import numpy as np, pandas as pd, torch, timm
import torchvision.transforms as T
from pathlib import Path
from PIL import Image; Image.MAX_IMAGE_PIXELS=None
from common.paths import RAW, SPLITS, CHECKPOINTS, OUTPUTS
from dinov2.models.vision_transformer import vit_small
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
random.seed(0); np.random.seed(0); dev='cuda'; HALF=112; K=10
norm=T.Compose([T.ToTensor(),T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
RES=OUTPUTS/'experiments'/'ws1c_featalign'; RES.mkdir(parents=True,exist_ok=True)
D=OUTPUTS/'experiments'/'stageD_bag'; meta=pd.read_csv(D/'bag_meta_v2.csv')
SEEDF={0:'emb_thickdino_v2.npy',1:'emb_thickdino_v2_s1.npy',2:'emb_thickdino_v2_s2.npy'}
EMB={s:np.load(D/f) for s,f in SEEDF.items()}
y=meta.label01.values; dom=meta.dataset.values; bnd=meta.para_band.values
A=(dom=='ibadan'); B=(dom=='chittagong-1'); loA=(bnd[A]=='neg')|(bnd[A]=='pos_lo')

def dv(sd,patch):
    m=vit_small(patch_size=patch,img_size=224,block_chunks=0,init_values=1.0); m.load_state_dict(sd,strict=False); return m.eval().to(dev)
thick=dv(torch.load(CHECKPOINTS/'ssl_vits_supvit_dinoadapt'/'encoder_final.pth',map_location='cpu'),16)

# ---- C1-box tile parasite scorer (source B) ----
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
def free(cx,cy,c): return all(abs(bx-cx)>HALF or abs(by-cy)>HALF for bx,by in c)
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
            if not cents or free(rx,ry,cents): tiles.append(np.asarray(cropc(im,rx,ry),np.uint8)); labs.append(0); g+=1
            if g>=max(3,len(cents[:20])): break
    if sum(labs)>=4000: break
labs=np.array(labs)
@torch.no_grad()
def enc(arr,bs=256):
    out=[]
    for i in range(0,len(arr),bs):
        b=torch.stack([norm(Image.fromarray(t)) for t in arr[i:i+bs]]).to(dev); out.append(thick(b).float().cpu().numpy())
    return np.concatenate(out)
Xs=enc(tiles).astype(np.float64)                 # SOURCE tile features (scorer training distribution)
tsc=StandardScaler().fit(Xs); tlr=LogisticRegression(max_iter=3000,class_weight='balanced').fit(tsc.transform(Xs),labs)
print('tile scorer trained on',len(tiles),'C1 tiles',flush=True)

# ---- source alignment stats ----
muS=Xs.mean(0); sdS=Xs.std(0)+1e-6
def mat_sqrt(C,inv=False,eps=1e-4):
    w,V=np.linalg.eigh(C); w=np.clip(w,eps,None); d=1.0/np.sqrt(w) if inv else np.sqrt(w); return (V*d)@V.T
SigS=np.cov(Xs,rowvar=False); S_sqrt=mat_sqrt(SigS)
def align(Xt,mode):
    Xt=Xt.astype(np.float64)
    if mode=='raw': return Xt
    if mode=='BN': return (Xt-Xt.mean(0))/(Xt.std(0)+1e-6)*sdS+muS
    if mode=='CORAL':
        muT=Xt.mean(0); Tinv=mat_sqrt(np.cov(Xt,rowvar=False),inv=True); return (Xt-muT)@Tinv@S_sqrt+muS
def score_tiles(Xt): return tlr.predict_proba(tsc.transform(Xt))[:,1]

# ---- Tanzania target ----
def content_tiles(img,n,thr=18):
    W,H=img.size; ts=[]
    for _ in range(n*40):
        cx=random.randint(HALF,max(HALF,W-HALF)); cy=random.randint(HALF,max(HALF,H-HALF)); t=img.crop((cx-HALF,cy-HALF,cx+HALF,cy+HALF))
        if np.asarray(t.convert('L')).std()>thr:
            ts.append(np.asarray(t.resize((224,224))).astype(np.uint8))
            if len(ts)>=n: break
    while len(ts)<n: ts.append(np.zeros((224,224,3),np.uint8))
    return ts
random.seed(0); np.random.seed(0); NPER=16; NIMG=500
inf=sorted(glob.glob(str(RAW/'tanzania'/'Thick_Infected'/'*.jpg')))[:NIMG]; unin=sorted(glob.glob(str(RAW/'tanzania'/'Thick_Uninfected'/'*.jpg')))[:NIMG]
its=[(p,1) for p in inf]+[(p,0) for p in unin]; random.shuffle(its)
tzt=[]; tzid=[]; lm={}
for k,(p,lab) in enumerate(its):
    try: im=Image.open(p).convert('RGB')
    except Exception: continue
    for t in content_tiles(im,NPER): tzt.append(t); tzid.append(k)
    lm[k]=lab
tzt=np.stack(tzt); tzid=np.array(tzid); uniqC=sorted(lm); yC=np.array([lm[k] for k in uniqC])
Etz=enc(tzt).astype(np.float64)
print(f'Tanzania {len(uniqC)} imgs prior={yC.mean():.3f}',flush=True)

def topk(P, ids=None):
    if ids is None:  # P is (n_bags,256)
        return np.sort(P,1)[:,::-1][:,:K].mean(1)
    return np.array([np.sort(P[ids==i])[::-1][:min(K,int((ids==i).sum()))].mean() for i in uniqC])
def ba_trueprior(yt,sc): thr=np.quantile(sc,1-yt.mean()); return balanced_accuracy_score(yt,(sc>=thr).astype(int))

report={'setup':dict(note='AUC=ranking; BA@true-prior=achievable decision; align target tiles -> source(C1-box) dist')}
for mode in ['raw','BN','CORAL']:
    # B->A (3 seeds)
    ov_auc=[]; lo_auc=[]; ov_ba=[]; lo_ba=[]
    for seed in SEEDF:
        E=EMB[seed][A]                                   # (nA,256,384)
        Pa=score_tiles(align(E.reshape(-1,384),mode)).reshape(E.shape[0],256)
        sc=topk(Pa)
        ov_auc.append(roc_auc_score(y[A],sc)); ov_ba.append(ba_trueprior(y[A],sc))
        lo_auc.append(roc_auc_score(y[A][loA],sc[loA])); lo_ba.append(ba_trueprior(y[A][loA],sc[loA]))
    # B->C
    Pc=score_tiles(align(Etz,mode)); scC=topk(Pc,tzid)
    report[mode]=dict(
        A_overall_AUC=round(float(np.mean(ov_auc)),3), A_overall_BA=round(float(np.mean(ov_ba)),3),
        A_lowpara_AUC=round(float(np.mean(lo_auc)),3), A_lowpara_BA=round(float(np.mean(lo_ba)),3),
        C_AUC=round(float(roc_auc_score(yC,scC)),3), C_BA=round(float(ba_trueprior(yC,scC)),3))
    print(mode, report[mode], flush=True)
json.dump(report,open(RES/'results.json','w'),indent=2); print('WROTE_DONE',RES/'results.json',flush=True)
