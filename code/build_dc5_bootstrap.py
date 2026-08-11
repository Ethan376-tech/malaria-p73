# -*- coding: utf-8 -*-
"""Part-2 (2) rigorous analysis: detection-count vs bag decision with 1000x stratified bootstrap 95% CIs, on the
per-FIELD-clean detection signals (persample_clean.csv). Bag = mean-pool logistic MIL on emb_reddino_v2, OOF pos-prob
(one per sample). Report AUC + 95% CI for detection (n@0.7 count, max-conf presence) and bag, per band, AND the paired
bootstrap CI of the DIFFERENCE (detection - bag) to test significance. Bands: overall / high-para / low-para."""
import sys, json
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
sys.path.append('/root/A_Dissertation'); from common.paths import OUTPUTS
RES=OUTPUTS/'experiments'/'dc_detcount'
df=pd.read_csv(RES/'persample_clean.csv').reset_index(drop=True)
print(f'clean samples: {len(df)}  bands: {df.band.value_counts().to_dict()}',flush=True)
# ---- bag OOF pos-prob (mean-pool logistic, one per sample) ----
bm=pd.read_csv(OUTPUTS/'experiments'/'stageD_bag'/'bag_meta_v2.csv'); Am=bm[bm.domain=='A_Nigeria'].reset_index(drop=True)
emb=np.load(OUTPUTS/'experiments'/'stageD_bag'/'emb_reddino_v2.npy'); embA=emb[(bm.domain=='A_Nigeria').values]; bagfeat=embA.mean(1)
idx={sid:i for i,sid in enumerate(Am.sample_id)}
df['bag']=np.nan; X=np.array([bagfeat[idx[s]] for s in df.sample_id]); folds=df.fold.values; ylab=(df.label==1).astype(int).values
for f in np.unique(folds):
    tr=folds!=f; te=folds==f
    if len(np.unique(ylab[tr]))<2: continue
    df.loc[te,'bag']=LogisticRegression(max_iter=2000).fit(X[tr],ylab[tr]).predict_proba(X[te])[:,1]
def band_mask(band): return (df.band=='neg')|(df.band==band) if band in ('pos_lo','pos_hi') else np.ones(len(df),bool)
def yof(band, m):
    return (df.band[m]==band).astype(int).values if band in ('pos_lo','pos_hi') else (df.label[m]==1).astype(int).values
rng=np.random.RandomState(0); B=1000
def boot_auc_ci(sig, y):
    pt=roc_auc_score(y,sig); n=len(y); pos=np.where(y==1)[0]; neg=np.where(y==0)[0]; vals=[]
    for _ in range(B):
        ii=np.concatenate([rng.choice(pos,len(pos),True), rng.choice(neg,len(neg),True)])
        try: vals.append(roc_auc_score(y[ii],sig[ii]))
        except Exception: pass
    lo,hi=np.percentile(vals,[2.5,97.5]); return round(float(pt),3),round(float(lo),3),round(float(hi),3)
def boot_diff_ci(sig1, sig2, y):
    # paired: same resample indices for both signals
    d0=roc_auc_score(y,sig1)-roc_auc_score(y,sig2); pos=np.where(y==1)[0]; neg=np.where(y==0)[0]; vals=[]
    for _ in range(B):
        ii=np.concatenate([rng.choice(pos,len(pos),True), rng.choice(neg,len(neg),True)])
        try: vals.append(roc_auc_score(y[ii],sig1[ii])-roc_auc_score(y[ii],sig2[ii]))
        except Exception: pass
    lo,hi=np.percentile(vals,[2.5,97.5]); return round(float(d0),3),round(float(lo),3),round(float(hi),3)
DET={'n07':'detection count (conf≥0.7)','maxconf':'detection max-conf'}
out={}
for band,name in [('overall','overall (pos vs neg)'),('pos_hi','high-parasitaemia'),('pos_lo','low-parasitaemia')]:
    m=band_mask(band); y=yof(band,m); sub=df[m]
    npos=int((y==1).sum()); nneg=int((y==0).sum())
    r={'n_pos':npos,'n_neg':nneg,'bag':boot_auc_ci(sub['bag'].values,y)}
    for sig in DET: r[sig]=boot_auc_ci(sub[sig].values.astype(float),y); r[sig+'_vs_bag']=boot_diff_ci(sub[sig].values.astype(float),sub['bag'].values,y)
    out[band]=r
json.dump(out, open(RES/'bootstrap_results.json','w'), indent=2)
print('\n=== detection vs bag, AUC [95% CI] + difference (per-field clean, 1000x bootstrap) ===')
for band,name in [('overall','overall'),('pos_hi','high-para'),('pos_lo','low-para')]:
    r=out[band]; print(f'-- {name} (n_pos={r["n_pos"]}, n_neg={r["n_neg"]}) --')
    print(f'   bag            AUC {r["bag"][0]} [{r["bag"][1]}, {r["bag"][2]}]')
    for sig in DET:
        a=r[sig]; d=r[sig+'_vs_bag']; sig_flag='SIGNIF' if (d[1]>0 or d[2]<0) else 'n.s.'
        print(f'   {sig:8s}       AUC {a[0]} [{a[1]}, {a[2]}]   diff vs bag {d[0]} [{d[1]}, {d[2]}] {sig_flag}')
print('DC5_DONE',flush=True)
