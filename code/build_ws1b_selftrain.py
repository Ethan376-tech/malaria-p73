# -*- coding: utf-8 -*-
"""WS1b (step 1) — source-free decision adaptation by self-training on the UNLABELLED target, comparing
pseudo-label selection rules. Builds on WS1a: the cross-site ranking partly survives (B->A overall AUC 0.63,
B->C 0.78) but the naive decision is at chance because the source threshold collapses under covariate shift.
Can source-free self-training recover it toward the WS1a true-prior ceiling (0.62 / 0.77) WITHOUT target labels,
and does faintness-protected pseudo-label selection help vs naive-confidence?

Per bag/image feature (from its tile parasite-probs P): [top-k mean, MC bootstrap uncertainty, mean, max, frac>0.5].
Source head = balanced logistic on B bags. Methods on target: M0 source-only; M1 feature-align (diag-CORAL);
M2 self-train naive-confidence; M3 self-train faintness-protected. Eval balanced-acc overall + low-para (A) and C.
Encoder ThickDINO. Output -> outputs/experiments/ws1b_selftrain/results.json
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
random.seed(0); np.random.seed(0); dev='cuda'; HALF=112; K=10; NBOOT=20
norm=T.Compose([T.ToTensor(),T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
RES=OUTPUTS/'experiments'/'ws1b_selftrain'; RES.mkdir(parents=True,exist_ok=True)
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
Xt=enc(tiles); tsc=StandardScaler().fit(Xt); tlr=LogisticRegression(max_iter=3000,class_weight='balanced').fit(tsc.transform(Xt),labs)
print('tile scorer trained on',len(tiles),'C1 tiles',flush=True)

# ---- Tanzania (target C), image-level ----
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
Ptz=tlr.predict_proba(tsc.transform(enc(tzt)))[:,1]
print(f'Tanzania {len(uniqC)} imgs prior={yC.mean():.3f}',flush=True)

# ---- per-bag/image feature vector (5-d) from tile parasite probs ----
rng=np.random.RandomState(0)
def feats_from_P(Plist):
    """Plist: list of 1-D arrays of tile parasite-probs. Returns (n,5): topk, uncertainty, mean, max, frac>.5."""
    out=[]
    for P in Plist:
        P=np.sort(P)[::-1]; kk=min(K,len(P)); tk=P[:kk].mean()
        boots=[np.sort(P[rng.randint(0,len(P),len(P))])[::-1][:kk].mean() for _ in range(NBOOT)]
        out.append([tk, float(np.std(boots)), float(P.mean()), float(P.max()), float((P>0.5).mean())])
    return np.array(out)
def bag_feats(seed):
    E=EMB[seed]; Pt=tlr.predict_proba(tsc.transform(E.reshape(-1,384)))[:,1].reshape(E.shape[0],256)
    return feats_from_P([Pt[i] for i in range(Pt.shape[0])])
def c_feats():
    return feats_from_P([Ptz[tzid==i] for i in uniqC])

# ---- adaptation methods (source-free) — decision score = raw top-k (col 0, ranking transfers); col 1 = uncertainty
def platt(sB,yB):
    lr=LogisticRegression(max_iter=3000,class_weight='balanced'); lr.fit(sB.reshape(-1,1),yB); return lr
def psrc(lr,s): return lr.predict_proba(s.reshape(-1,1))[:,1]

def m1_align(sB,yB,sT):                        # match target score dist to source, then source threshold
    sa=(sT-sT.mean())/(sT.std()+1e-6)*(sB.std()+1e-6)+sB.mean(); return psrc(platt(sB,yB),sa)
def m_selftrain(sB,yB,sT,uT,protected,rounds=3,frac=0.5):
    lr=platt(sB,yB)                            # init threshold from source
    for _ in range(rounds):
        p=psrc(lr,sT)
        if not protected:
            conf=np.abs(p-0.5); idx=np.argsort(-conf)[:int(frac*len(p))]; pl=(p[idx]>=0.5).astype(int); ss=sT[idx]
        else:                                  # protect faint pos: uncertain-mid -> pos, not neg
            umed=np.median(uT); neg=(p<0.4)&(uT<=umed); pos=(p>0.6)|((p>0.5)&(uT>umed))
            idx=np.where(neg|pos)[0]; pl=pos[idx].astype(int); ss=sT[idx]
        if len(set(pl))<2: break
        lr=platt(ss,pl)                        # refit threshold on pseudo-labelled TARGET
    return psrc(lr,sT)

def BA(yt,p): return balanced_accuracy_score(yt,(p>=0.5).astype(int))
def ba_trueprior(yt,s):                        # ceiling for any threshold method (WS1a reference), source knows prior
    thr=np.quantile(s,1-yt.mean()); return balanced_accuracy_score(yt,(s>=thr).astype(int))
from sklearn.mixture import GaussianMixture
def m5_gmm(sT):                                # source-free, prior-free: 2-component GMM valley on target scores
    g=GaussianMixture(2,random_state=0).fit(sT.reshape(-1,1)); comp=g.predict(sT.reshape(-1,1))
    poscomp=int(np.argmax(g.means_.ravel())); return (comp==poscomp).astype(float)
def evaluate(FB,yB,FT,yT,masks):
    sB=FB[:,0]; sT=FT[:,0]; uT=FT[:,1]         # top-k score + uncertainty
    p0=psrc(platt(sB,yB),sT)
    pM1=m1_align(sB,yB,sT); pM2=m_selftrain(sB,yB,sT,uT,False); pM3=m_selftrain(sB,yB,sT,uT,True)
    sTa=(sT-sT.mean())/(sT.std()+1e-6)*(sB.std()+1e-6)+sB.mean()                 # aligned scores
    pM4=m_selftrain(sB,yB,sTa,uT,True)         # align THEN self-train-protected (fair init)
    pM5=m5_gmm(sT)                             # GMM valley threshold
    out={}
    for mname,mask in masks.items():
        yt=yT[mask]
        out[mname]=dict(
            AUC=round(float(roc_auc_score(yt,sT[mask])),3),
            M0_source=round(float(BA(yt,p0[mask])),3),
            M1_align=round(float(BA(yt,pM1[mask])),3),
            M2_selftrain_naive=round(float(BA(yt,pM2[mask])),3),
            M3_selftrain_protected=round(float(BA(yt,pM3[mask])),3),
            M4_align_then_st=round(float(BA(yt,pM4[mask])),3),
            M5_gmm_valley=round(float(balanced_accuracy_score(yt,pM5[mask])),3),
            ceiling_trueprior=round(float(ba_trueprior(yt,sT[mask])),3))
    return out

report={'setup':dict(note='balanced-acc; source B=chittagong-1; ceilings from WS1a true-prior: A-overall 0.62, A-lowpara ~0.52(chance), C 0.77')}
# B->A (3 seeds)
accum={'overall':[], 'lowpara':[]}
for seed in SEEDF:
    FA=bag_feats(seed); FB=FA[B]; FT=FA[A]
    r=evaluate(FB,y[B],FT,y[A],{'overall':np.ones(A.sum(),bool),'lowpara':loA})
    for kk in accum: accum[kk].append(r[kk])
avg=lambda rows:{k:round(float(np.mean([r[k] for r in rows])),3) for k in rows[0]}
report['B2A']={kk:avg(accum[kk]) for kk in accum}
# B->C
FAll=bag_feats(0); FB=FAll[B]; FC=c_feats()
report['B2C']=evaluate(FB,y[B],FC,yC,{'overall':np.ones(len(yC),bool)})['overall']
print('B2A',json.dumps(report['B2A'],indent=2)); print('B2C',json.dumps(report['B2C'],indent=2))
json.dump(report,open(RES/'results.json','w'),indent=2); print('WROTE_DONE',RES/'results.json',flush=True)
