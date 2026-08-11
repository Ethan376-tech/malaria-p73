# -*- coding: utf-8 -*-
"""T7 — Post-hoc calibration recovery under cross-site shift. Operationalises the decision-transfer diagnosis
(Table 4.6) in standard calibration vocabulary: does STANDARD post-hoc calibration (temperature / vector scaling),
fit on a few labelled target samples, recover the cross-site decision? Uses the SAME source-trained TK scores as
ws1a (B->A overall + low-parasitaemia, and B->C Tanzania). Reports ECE / Brier / acc@0.5 / AUC before vs after
calibration. Key point: temperature and vector scaling are MONOTONIC in the logit -> AUC-invariant, so they can fix
a calibration failure (overall / Tanzania: ranking preserved) but PROVABLY cannot fix a ranking failure
(low-parasitaemia), independent of sample size. Output -> outputs/experiments/t7_calib/results.json"""
import os, sys, json, glob, random, time
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
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
random.seed(0); np.random.seed(0); dev='cuda'; HALF=112
norm=T.Compose([T.ToTensor(),T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
RES=OUTPUTS/'experiments'/'t7_calib'; RES.mkdir(parents=True,exist_ok=True)
STAT=OUTPUTS/'experiments'/'t7_status.log'
def hb(m):
    with open(STAT,'a') as f: f.write(f'{int(time.time())} {m}\n')
D=OUTPUTS/'experiments'/'stageD_bag'; meta=pd.read_csv(D/'bag_meta_v2.csv')
SEEDF={0:'emb_thickdino_v2.npy',1:'emb_thickdino_v2_s1.npy',2:'emb_thickdino_v2_s2.npy'}
EMB={s:np.load(D/f) for s,f in SEEDF.items()}
y=meta.label01.values; dom=meta.dataset.values; band=meta.para_band.values
A=(dom=='ibadan'); B=(dom=='chittagong-1')
lo=A&((band=='neg')|(band=='pos_lo')); loA=lo[A]
PI_S=float(y[B].mean())
hb('T7 START (calibration recovery)')

# ---------- encoder (ThickDINO), same checkpoint as ws1a ----------
def dv(sd,patch):
    m=vit_small(patch_size=patch,img_size=224,block_chunks=0,init_values=1.0); m.load_state_dict(sd,strict=False); return m.eval().to(dev)
thick=dv(torch.load(CHECKPOINTS/'ssl_vits_supvit_dinoadapt'/'encoder_final.pth',map_location='cpu'),16)

# ---------- TK tile scorer on Chittagong-1 boxes (== ws1a) ----------
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
def enc_tiles(model,arr,bs=256):
    out=[]
    for i in range(0,len(arr),bs):
        b=torch.stack([norm(Image.fromarray(t)) for t in arr[i:i+bs]]).to(dev); out.append(model(b).float().cpu().numpy())
    return np.concatenate(out)
Xtile=enc_tiles(thick,tiles); tsc=StandardScaler().fit(Xtile); tlr=LogisticRegression(max_iter=3000,class_weight='balanced').fit(tsc.transform(Xtile),labs)
hb(f'TK scorer trained on {len(tiles)} C1 box tiles')

# ---------- source-trained TK scores on target A (== ws1a run/tk_feat) ----------
def run(feat, seed):
    fB=feat[B]; yB=y[B]; fA=feat[A]
    skf=StratifiedKFold(5,shuffle=True,random_state=seed); tgt=np.zeros(A.sum())
    for tr,va in skf.split(fB,yB):
        sc=StandardScaler().fit(fB[tr]); clf=LogisticRegression(max_iter=3000,C=0.3,class_weight='balanced').fit(sc.transform(fB[tr]),yB[tr])
        tgt+=clf.predict_proba(sc.transform(fA))[:,1]/5
    return tgt
def tk_feat(seed,k=10):
    E=EMB[seed]; Pt=tlr.predict_proba(tsc.transform(E.reshape(-1,384)))[:,1].reshape(E.shape[0],256)
    return np.sort(Pt,1)[:,::-1][:,:k].mean(1,keepdims=True)
scoreA_seeds=[run(tk_feat(sd),sd) for sd in SEEDF]        # list of (nA,) source-trained probs on A
hb('B->A TK target scores computed (3 seeds)')

# ================= B -> C Tanzania (== ws1a) : naive = source-Platt prob on C =================
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
itemsz=[(p,1) for p in inf]+[(p,0) for p in unin]; random.shuffle(itemsz)
tz_tiles=[]; tz_imgid=[]; labmap={}
for k,(p,lab) in enumerate(itemsz):
    try: im=Image.open(p).convert('RGB')
    except Exception: continue
    for t in content_tiles(im,NPER): tz_tiles.append(t); tz_imgid.append(k)
    labmap[k]=lab
tz_tiles=np.stack(tz_tiles); tz_imgid=np.array(tz_imgid); uniq=sorted(labmap); yC=np.array([labmap[k] for k in uniq])
Etz=enc_tiles(thick,tz_tiles); Ptz=tlr.predict_proba(tsc.transform(Etz))[:,1]
def topk_img(k=10): return np.array([np.sort(Ptz[tz_imgid==i])[::-1][:k].mean() for i in uniq])
scC=topk_img()
def c1_bag_topk(seed,k=10):
    E=EMB[seed]; Pt=tlr.predict_proba(tsc.transform(E.reshape(-1,384)))[:,1].reshape(E.shape[0],256)
    return np.sort(Pt,1)[:,::-1][:,:k].mean(1)
pC_seeds=[]
for sd in SEEDF:
    sB=c1_bag_topk(sd); plat=LogisticRegression(max_iter=3000,class_weight='balanced').fit(sB[B].reshape(-1,1),y[B])
    pC_seeds.append(plat.predict_proba(scC.reshape(-1,1))[:,1])
hb('B->C Tanzania naive (source-Platt) probs computed (3 seeds)')

# ================= calibration machinery =================
def logit(p): p=np.clip(p,1e-6,1-1e-6); return np.log(p/(1-p))
def sigmoid(x): return 1.0/(1.0+np.exp(-x))
def ece(yt,p,nbins=10):
    yt=np.asarray(yt,float); p=np.asarray(p,float); edges=np.linspace(0,1,nbins+1); e=0.0; n=len(yt)
    for i in range(nbins):
        m=(p>=edges[i])&(p<edges[i+1]) if i<nbins-1 else (p>=edges[i])&(p<=edges[i+1])
        if m.sum(): e+=abs(p[m].mean()-yt[m].mean())*m.sum()/n
    return e
def _nll(p,yt): p=np.clip(p,1e-6,1-1e-6); return -np.mean(yt*np.log(p)+(1-yt)*np.log(1-p))
def temp_scale(lg_cal,y_cal):
    # scale only: p=sigmoid(z/T), T>0 -> a=1/T>0, symmetric about 0 (does NOT move the 0.5-threshold operating point)
    best,bn=1.0,1e18
    for Tt in np.logspace(-1.0,1.3,141):
        n=_nll(sigmoid(lg_cal/Tt),y_cal)
        if n<bn: bn,best=n,Tt
    return best
def platt_fit(lg_cal,y_cal):
    # Platt scaling p=sigmoid(a*z+b) with a>0 ENFORCED (grid) -> monotone -> AUC-invariant; b recovers the operating point
    best=(1.0,0.0); bn=1e18
    for a in np.logspace(-1.5,1.0,26):
        za=a*lg_cal
        for b in np.linspace(-8,8,81):
            n=_nll(sigmoid(za+b),y_cal)
            if n<bn: bn=n; best=(a,b)
    return best
def mets(yt,p):
    yt=np.asarray(yt); p=np.asarray(p)
    return dict(ECE=ece(yt,p), Brier=float(np.mean((p-yt)**2)), acc=float(np.mean((p>=0.5)==yt)),
                BA=float(balanced_accuracy_score(yt,(p>=0.5).astype(int))),
                AUC=float(roc_auc_score(yt,p)) if len(set(yt))>1 else float('nan'))

def calib_eval(score_list, ytrue, band_mask, nsplit=30, calib_frac=0.4):
    """Fit temp/vector on an overall target calib split; evaluate naive/temp/vector on the held-out test (test∩band).
    Averaged over nsplit random splits × the score seeds. band_mask: bool over the target subset."""
    acc={nm:[] for nm in ('naive','temp','vector')}
    for score in score_list:
        lg=logit(score); sss=StratifiedShuffleSplit(n_splits=nsplit,test_size=1-calib_frac,random_state=0)
        for cal,te in sss.split(score,ytrue):
            if len(set(ytrue[cal]))<2: continue
            Tt=temp_scale(lg[cal],ytrue[cal]); a,b=platt_fit(lg[cal],ytrue[cal])
            tem=te[band_mask[te]]
            if len(set(ytrue[tem]))<2: continue
            yt=ytrue[tem]; l=lg[tem]
            acc['naive'].append(mets(yt,score[tem])); acc['temp'].append(mets(yt,sigmoid(l/Tt))); acc['vector'].append(mets(yt,sigmoid(a*l+b)))
    out={}
    for nm in acc:
        if acc[nm]: out[nm]={k:round(float(np.nanmean([r[k] for r in acc[nm]])),3) for k in acc[nm][0]}
    return out

nA=A.sum(); allA=np.ones(nA,bool)
report={'setup':dict(src='chittagong-1',tgt='ibadan',src_prior=round(PI_S,3),
                     tgt_prior_overall=round(float(y[A].mean()),3),tgt_prior_lowpara=round(float(y[lo].mean()),3),
                     methods='naive=source op-point; temp=temperature scaling (scalar); vector=affine a*logit+b, both fit on a labelled target calib split',
                     note='temp & vector are monotonic in the logit -> AUC-invariant by construction')}
report['B2A_overall']=calib_eval(scoreA_seeds, y[A], allA)
report['B2A_lowpara']=calib_eval(scoreA_seeds, y[A], loA)
report['B2C_Tanzania']=calib_eval(pC_seeds, yC, np.ones(len(yC),bool))
for k in ('B2A_overall','B2A_lowpara','B2C_Tanzania'):
    hb(f'{k}: naive={report[k].get("naive")} vector={report[k].get("vector")}')
json.dump(report,open(RES/'results.json','w'),indent=2)

# ---------------- reliability diagram (naive vs OOF-Platt), B->A overall + Tanzania ----------------
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
FIGD=Path('/root/autodl-tmp/A_Dissertation/outputs/experiments/figures'); FIGD.mkdir(parents=True,exist_ok=True)
def oof_platt(score,yv,nsplit=5,seeds=(0,1,2)):
    lg2=logit(score); oof=np.zeros(len(score)); cnt=np.zeros(len(score))
    for sd in seeds:
        for tr,te in StratifiedKFold(nsplit,shuffle=True,random_state=sd).split(score,yv):
            a,b=platt_fit(lg2[tr],yv[tr]); oof[te]+=sigmoid(a*lg2[te]+b); cnt[te]+=1
    return oof/np.maximum(cnt,1)
def rel_curve(yv,p,nbins=10,minn=5):
    e=np.linspace(0,1,nbins+1); xs=[];ys=[]
    for i in range(nbins):
        m=(p>=e[i])&(p<e[i+1]) if i<nbins-1 else (p>=e[i])&(p<=e[i+1])
        if m.sum()>=minn: xs.append(float(p[m].mean())); ys.append(float(yv[m].mean()))
    return np.array(xs),np.array(ys)
scA=np.mean(scoreA_seeds,0); pCm=np.mean(pC_seeds,0)
np.savez(RES/'scores.npz', scoreA=scA, yA=y[A], loA=loA, pC=pCm, yC=yC)
panels=[('B→A (overall)', scA, y[A]), ('B→C Tanzania', pCm, yC)]
fig,axs=plt.subplots(1,2,figsize=(8.0,3.9))
for ax,(ttl,sc,yv) in zip(axs,panels):
    pc=oof_platt(sc,yv); ax.plot([0,1],[0,1],'--',color='0.6',lw=1)
    xn,yn=rel_curve(yv,sc); xk,yk=rel_curve(yv,pc)
    ax.plot(xn,yn,'o-',color='#c0392b',label='naive (source op-point)',ms=5)
    ax.plot(xk,yk,'s-',color='#2471a3',label='after Platt scaling',ms=5)
    ax.set_title(ttl,fontsize=11); ax.set_xlabel('predicted probability'); ax.set_xlim(0,1); ax.set_ylim(0,1)
    ax.legend(fontsize=8,loc='upper left'); ax.grid(alpha=0.25)
axs[0].set_ylabel('empirical frequency')
plt.tight_layout(); plt.savefig(FIGD/'fig_calib_reliability.png',dpi=150); plt.close()
hb('reliability fig written')
hb('T7_DONE')
print('T7_DONE', json.dumps(report,indent=2), flush=True)
