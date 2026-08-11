# -*- coding: utf-8 -*-
"""Part-2 (2) analysis: detection-count decision vs bag aggregation, overall + by parasitaemia band.
Detection signals from dc_detcount/persample.csv; bag anchor = mean-pool logistic MIL on emb_reddino_v2 (OOF by fold),
on the SAME samples. Metrics: ROC-AUC (threshold-free) and OOF balanced accuracy (fold-picked threshold). Bands:
overall (pos vs neg), low-parasitaemia (pos_lo vs neg) = the thesis-critical decision, high (pos_hi vs neg).
Compares to the report's MIL bag decision (overall BA 0.651/AUC 0.68; low BA 0.58/AUC ~0.45)."""
import sys, json
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from sklearn.linear_model import LogisticRegression
sys.path.append('/root/A_Dissertation'); from common.paths import OUTPUTS
RES=OUTPUTS/'experiments'/'dc_detcount'
df=pd.read_csv(RES/'persample.csv')
print(f'samples: {len(df)}  bands: {df.band.value_counts().to_dict()}',flush=True)
def subset(bandpos):
    m=df.band.isin(['neg',bandpos]); s=df[m].copy(); y=(s.band==bandpos).astype(int).values; return s,y
def auc_of(sig, y):
    try: return round(float(roc_auc_score(y, sig)),3)
    except Exception: return float('nan')
def oof_ba(sig, y, folds):
    # per-fold: pick threshold maximizing BA on the OTHER folds, apply to held-out fold; aggregate
    pred=np.zeros(len(y)); uf=np.unique(folds)
    for f in uf:
        tr=folds!=f; te=folds==f
        if tr.sum()==0 or len(np.unique(y[tr]))<2:
            thr=np.median(sig); pred[te]=(sig[te]>=thr).astype(int); continue
        cand=np.unique(sig[tr]); best=(-1,cand[0])
        for t in cand:
            ba=balanced_accuracy_score(y[tr], (sig[tr]>=t).astype(int))
            if ba>best[0]: best=(ba,t)
        pred[te]=(sig[te]>=best[1]).astype(int)
    return round(float(balanced_accuracy_score(y, pred)),3)
SIGS=['maxconf','n03','n05','n07','top5sum','ndet']
out={}
for band in ['pos_lo','pos_hi']:
    s,y=subset(band); folds=s.fold.values
    out[band]={sig:{'AUC':auc_of(s[sig].values,y),'BA':oof_ba(s[sig].values.astype(float),y,folds)} for sig in SIGS}
# overall pos vs neg
s=df.copy(); y=(s.label==1).astype(int).values; folds=s.fold.values
out['overall']={sig:{'AUC':auc_of(s[sig].values,y),'BA':oof_ba(s[sig].values.astype(float),y,folds)} for sig in SIGS}
# ---- bag anchor: mean-pool logistic MIL on emb_reddino_v2, OOF ----
try:
    bm=pd.read_csv(OUTPUTS/'experiments'/'stageD_bag'/'bag_meta_v2.csv'); A=bm[bm.domain=='A_Nigeria'].reset_index(drop=True)
    emb=np.load(OUTPUTS/'experiments'/'stageD_bag'/'emb_reddino_v2.npy')[:len(bm)]  # (N,256,384)
    embA=emb[(bm.domain=='A_Nigeria').values]; bagfeat=embA.mean(1)  # mean-pool
    idx={sid:i for i,sid in enumerate(A.sample_id)}
    keep=[i for i,sid in enumerate(df.sample_id) if sid in idx]
    dfb=df.iloc[keep].reset_index(drop=True); X=np.array([bagfeat[idx[s]] for s in dfb.sample_id]); folds=dfb.fold.values
    def bag_oof_score(y):
        sc=np.zeros(len(y))
        for f in np.unique(folds):
            tr=folds!=f; te=folds==f
            if len(np.unique(y[tr]))<2: continue
            lr=LogisticRegression(max_iter=2000,C=1.0).fit(X[tr],y[tr]); sc[te]=lr.predict_proba(X[te])[:,1]
        return sc
    bag={}
    for band in ['pos_lo','pos_hi']:
        m=dfb.band.isin(['neg',band]); y=(dfb.band[m]==band).astype(int).values
        Xb=X[m.values]; fb=folds[m.values]
        sc=np.zeros(len(y))
        for f in np.unique(fb):
            tr=fb!=f; te=fb==f
            if len(np.unique(y[tr]))<2: continue
            lr=LogisticRegression(max_iter=2000).fit(Xb[tr],y[tr]); sc[te]=lr.predict_proba(Xb[te])[:,1]
        bag[band]={'AUC':auc_of(sc,y),'BA':oof_ba(sc,y,fb)}
    yov=(dfb.label==1).astype(int).values; scov=bag_oof_score(yov)
    bag['overall']={'AUC':auc_of(scov,yov),'BA':oof_ba(scov,yov,folds)}
    out['bag_meanpool_reddino']=bag
except Exception as e:
    out['bag_meanpool_reddino']={'error':str(e)[:120]}
json.dump(out, open(RES/'decision_results.json','w'), indent=2)
print('\n=== DETECTION-COUNT decision vs bag (AUC / OOF balanced-acc) ===')
for band in ['overall','pos_lo','pos_hi']:
    print(f'-- {band} --')
    for sig in SIGS: print(f'   {sig:8s} AUC={out[band][sig]["AUC"]}  BA={out[band][sig]["BA"]}')
print('-- bag mean-pool (reddino, OOF, same samples) --')
for band in ['overall','pos_lo','pos_hi']:
    b=out['bag_meanpool_reddino'].get(band,{}); print(f'   {band:8s} AUC={b.get("AUC")}  BA={b.get("BA")}')
print('-- report MIL bag (parasite-aware): overall BA 0.651/AUC 0.68 ; low BA 0.58/AUC ~0.45 --')
print('DC2_DONE',flush=True)
