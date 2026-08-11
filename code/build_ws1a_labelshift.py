# -*- coding: utf-8 -*-
"""WS1a — cheapest source-free diagnostic: is the cross-site (B->A) decision failure a LABEL/PRIOR shift
(fixable by re-estimating the target class prior, source-free via EM) or a genuine decision-direction failure?
Big real prior shift: source B pos-rate 0.75 -> target A pos-rate 0.39.

Two source-trained decisions on ThickDINO features:
  PW  = pure-weak bag logistic (max-pooled bag embedding, trained on B, applied to A)
  TK  = parasite-aware top-k scorer (tile LogReg on C1 boxes -> top-k over A's cached tiles -> Platt on B bags)
For each: EM label-shift correction (Saerens); metrics {AUC (invariant sanity), balanced-acc @ naive / EM /
oracle threshold}; overall vs low-parasitaemia band; estimated vs true target prior; bootstrap CI (low-para).
Averaged over the 3 tile-sampling seeds. Output -> outputs/experiments/ws1a_labelshift/results.json
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
random.seed(0); np.random.seed(0); dev='cuda'; HALF=112
norm=T.Compose([T.ToTensor(),T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
RES=OUTPUTS/'experiments'/'ws1a_labelshift'; RES.mkdir(parents=True,exist_ok=True)
D=OUTPUTS/'experiments'/'stageD_bag'; meta=pd.read_csv(D/'bag_meta_v2.csv')
SEEDF={0:'emb_thickdino_v2.npy',1:'emb_thickdino_v2_s1.npy',2:'emb_thickdino_v2_s2.npy'}
EMB={s:np.load(D/f) for s,f in SEEDF.items()}
y=meta.label01.values; dom=meta.dataset.values; band=meta.para_band.values
A=(dom=='ibadan'); B=(dom=='chittagong-1')
lo=A&((band=='neg')|(band=='pos_lo'))                 # low-parasitaemia band of the target
print(f'target A: n={A.sum()} pos-rate={y[A].mean():.3f} | low-para band n={lo.sum()} pos-rate={y[lo].mean():.3f}',flush=True)
print(f'source B: n={B.sum()} pos-rate={y[B].mean():.3f}',flush=True)

# ---------- encoder (ThickDINO) ----------
def dv(sd,patch):
    m=vit_small(patch_size=patch,img_size=224,block_chunks=0,init_values=1.0); m.load_state_dict(sd,strict=False); return m.eval().to(dev)
thick=dv(torch.load(CHECKPOINTS/'ssl_vits_supvit_dinoadapt'/'encoder_final.pth',map_location='cpu'),16)

# ---------- TK: train tile parasite scorer on Chittagong-1 boxes (lifted from build_xsite_v2) ----------
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
def enc_tiles(model,bs=256):
    out=[]
    for i in range(0,len(tiles),bs):
        b=torch.stack([norm(Image.fromarray(t)) for t in tiles[i:i+bs]]).to(dev); out.append(model(b).float().cpu().numpy())
    return np.concatenate(out)
Xtile=enc_tiles(thick); tsc=StandardScaler().fit(Xtile); tlr=LogisticRegression(max_iter=3000,class_weight='balanced').fit(tsc.transform(Xtile),labs)
print('TK tile scorer trained on',len(tiles),'C1 box tiles',flush=True)

# ---------- BBSE (Black-Box Shift Estimation, Lipton 2018): calibration-robust target-prior estimate ----------
def bbse_prior(yhat_val, y_val, yhat_tgt, pi_s):
    """Ĉ[i,j]=P_s(yhat=i,y=j) on source val; μ̂[i]=P_t(yhat=i); solve Ĉ w = μ̂; π_t[j]=w[j]*π_s[j]."""
    C=np.zeros((2,2))
    for i in (0,1):
        for j in (0,1): C[i,j]=np.mean((yhat_val==i)&(y_val==j))
    mu=np.array([np.mean(yhat_tgt==0),np.mean(yhat_tgt==1)])
    try: w=np.linalg.solve(C,mu)
    except np.linalg.LinAlgError: w=np.array([1.0,1.0])
    pri=np.array([(1-pi_s),pi_s])*np.clip(w,0,None); pri=pri/max(pri.sum(),1e-9)
    return float(pri[1])                                # estimated target positive prior

def ba_at_posrate(yt, score, posrate):
    """Predict the top `posrate` fraction of target as positive (quantile threshold); return balanced accuracy."""
    posrate=min(max(posrate,1e-3),1-1e-3); thr=np.quantile(score,1-posrate)
    return balanced_accuracy_score(yt,(score>=thr).astype(int))

def ba_oracle(yt, score):
    return max(balanced_accuracy_score(yt,(score>=t).astype(int)) for t in np.unique(score))

# ---------- per-seed source-free pipeline (5-fold on source for OOF confusion + fold-averaged target scores) ----------
from sklearn.model_selection import StratifiedKFold
def run(feat, seed):
    """feat: (441,d) per-bag feature. Returns target(A) score, source OOF hard preds + labels (for BBSE)."""
    fB=feat[B]; yB=y[B]; fA=feat[A]
    skf=StratifiedKFold(5,shuffle=True,random_state=seed); tgt=np.zeros(A.sum()); oof=np.zeros(B.sum())
    for tr,va in skf.split(fB,yB):
        sc=StandardScaler().fit(fB[tr]); clf=LogisticRegression(max_iter=3000,C=0.3,class_weight='balanced').fit(sc.transform(fB[tr]),yB[tr])
        tgt+=clf.predict_proba(sc.transform(fA))[:,1]/5
        oof[va]=clf.predict_proba(sc.transform(fB[va]))[:,1]
    return tgt, (oof>=0.5).astype(int), yB

def pw_feat(seed): return EMB[seed].max(1)                                   # max-pool bag embedding
def tk_feat(seed,k=10):
    E=EMB[seed]; Pt=tlr.predict_proba(tsc.transform(E.reshape(-1,384)))[:,1].reshape(E.shape[0],256)
    return np.sort(Pt,1)[:,::-1][:,:k].mean(1,keepdims=True)                 # top-k mean parasite prob (1-D)

PI_S=float(y[B].mean())
def evaluate(scoreA, oof_pred, yB, mask_within_A):
    """mask_within_A: boolean over the A-subset (overall=all True, lowpara=neg|pos_lo)."""
    yt=y[A][mask_within_A]; sc=scoreA[mask_within_A]
    auc=roc_auc_score(yt,sc) if len(set(yt))>1 else float('nan')
    yhat_tgt=(sc>=0.5).astype(int)
    pi_hat=bbse_prior(oof_pred, yB, yhat_tgt, PI_S)
    return dict(AUC=round(float(auc),3),
                tgt_pred_posrate=round(float(yhat_tgt.mean()),3),   # why BBSE fails: covariate shift skews predictions
                BA_naive=round(float(balanced_accuracy_score(yt,(sc>=0.5).astype(int))),3),
                BA_priorcorr=round(float(ba_at_posrate(yt,sc,pi_hat)),3),       # source-free BBSE prior
                BA_trueprior=round(float(ba_at_posrate(yt,sc,yt.mean())),3),    # if we KNEW the target prior
                BA_oracle=round(float(ba_oracle(yt,sc)),3),                     # best possible threshold
                prior_true=round(float(yt.mean()),3), prior_BBSE=round(float(pi_hat),3))

def boot_ci(yt,sc,pi_hat,reps=1000):
    yt=np.asarray(yt); sc=np.asarray(sc); n=len(yt); rng=np.random.RandomState(0); v=[]
    for _ in range(reps):
        idx=rng.randint(0,n,n)
        if len(set(yt[idx]))<2: continue
        v.append(ba_at_posrate(yt[idx],sc[idx],pi_hat))
    return [round(float(np.percentile(v,2.5)),3),round(float(np.percentile(v,97.5)),3)]

report={'setup':dict(src='chittagong-1',tgt='ibadan',src_prior=round(PI_S,3),
                     tgt_prior_overall=round(float(y[A].mean()),3),tgt_prior_lowpara=round(float(y[lo].mean()),3),
                     note='BA_priorcorr = balanced-acc at source-free BBSE-estimated positive rate (quantile threshold)')}
loA=lo[A]                                              # low-para mask within the A subset
for name,featfn in [('PW_pure_weak_bag_logistic',pw_feat),('TK_topk_parasite_scorer',tk_feat)]:
    ov=[]; lw=[]
    for seed in SEEDF:
        scoreA,oof,yB=run(featfn(seed),seed)
        ov.append(evaluate(scoreA,oof,yB,np.ones(A.sum(),bool)))
        lw.append(evaluate(scoreA,oof,yB,loA))
    avg=lambda rows:{k:round(float(np.mean([r[k] for r in rows])),3) for k in rows[0]}
    sc0,oof0,yB0=run(featfn(0),0)
    ci_ov=boot_ci(y[A],sc0,avg(ov)['prior_BBSE']); ci_lw=boot_ci(y[lo],sc0[loA],avg(lw)['prior_BBSE'])
    report[name]={'overall':{**avg(ov),'BA_priorcorr_95CI':ci_ov},'lowpara':{**avg(lw),'BA_priorcorr_95CI':ci_lw}}
    print(name,'\n  overall',report[name]['overall'],'\n  lowpara',report[name]['lowpara'],flush=True)

# ================= B -> C (Tanzania), same C1-box source scorer; image-level only (no low-para band) =================
from common.paths import CHITTAGONG2  # noqa
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
items=[(p,1) for p in inf]+[(p,0) for p in unin]; random.shuffle(items)
tz_tiles=[]; tz_imgid=[]; labmap={}
for k,(p,lab) in enumerate(items):
    try: im=Image.open(p).convert('RGB')
    except Exception: continue
    for t in content_tiles(im,NPER): tz_tiles.append(t); tz_imgid.append(k)
    labmap[k]=lab
tz_tiles=np.stack(tz_tiles); tz_imgid=np.array(tz_imgid); uniq=sorted(labmap); yC=np.array([labmap[k] for k in uniq])
@torch.no_grad()
def enc_batch(arr,bs=256):
    out=[]
    for i in range(0,len(arr),bs):
        b=torch.stack([norm(Image.fromarray(t)) for t in arr[i:i+bs]]).to(dev); out.append(thick(b).float().cpu().numpy())
    return np.concatenate(out)
Etz=enc_batch(tz_tiles); Ptz=tlr.predict_proba(tsc.transform(Etz))[:,1]
def topk_img(k=10): return np.array([np.sort(Ptz[tz_imgid==i])[::-1][:k].mean() for i in uniq])
scC=topk_img()
print(f'Tanzania: {len(uniq)} imgs, C pos-rate={yC.mean():.3f}, B2C top-k AUC={roc_auc_score(yC,scC):.3f}',flush=True)
# source(C1 bag) top-k scores for Platt + BBSE confusion
def c1_bag_topk(seed,k=10):
    E=EMB[seed]; Pt=tlr.predict_proba(tsc.transform(E.reshape(-1,384)))[:,1].reshape(E.shape[0],256)
    return np.sort(Pt,1)[:,::-1][:,:k].mean(1)
def eval_b2c(seed):
    sB=c1_bag_topk(seed); plat=LogisticRegression(max_iter=3000,class_weight='balanced').fit(sB[B].reshape(-1,1),y[B])
    pC=plat.predict_proba(scC.reshape(-1,1))[:,1]; oofB=(plat.predict_proba(sB[B].reshape(-1,1))[:,1]>=0.5).astype(int)
    yhatC=(pC>=0.5).astype(int); pi_hat=bbse_prior(oofB,y[B],yhatC,PI_S)
    return dict(AUC=round(float(roc_auc_score(yC,pC)),3), tgt_pred_posrate=round(float(yhatC.mean()),3),
                BA_naive=round(float(balanced_accuracy_score(yC,(pC>=0.5).astype(int))),3),
                BA_priorcorr=round(float(ba_at_posrate(yC,pC,pi_hat)),3),
                BA_trueprior=round(float(ba_at_posrate(yC,pC,yC.mean())),3),
                BA_oracle=round(float(ba_oracle(yC,pC)),3),
                prior_true=round(float(yC.mean()),3), prior_BBSE=round(float(pi_hat),3))
b2c=[eval_b2c(s) for s in SEEDF]; avg=lambda rows:{k:round(float(np.mean([r[k] for r in rows])),3) for k in rows[0]}
report['B2C_Tanzania_topk']={'overall':avg(b2c)}
print('B2C_Tanzania_topk\n  overall',report['B2C_Tanzania_topk']['overall'],flush=True)
json.dump(report,open(RES/'results.json','w'),indent=2)
print('\nWROTE_DONE',RES/'results.json',flush=True)
