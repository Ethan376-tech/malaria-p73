# -*- coding: utf-8 -*-
"""DEV probe: is SPATIAL CONTEXT (local neighbour density of a pseudo-box) a discriminative TP-vs-FP signal that
would EXTEND the multi-signal filter? Tests whether adding spatial features to the existing per-box signals raises
the TP/FP AUC (dev-trained logistic, test AUC). If yes -> worth retraining the detector with a spatial filter signal.
npz-based, no encoder. Output -> prints + outputs/experiments/cluster_probe/spatial_signal.json."""
import sys, json
import numpy as np
from sklearn.decomposition import PCA
from sklearn.covariance import LedoitWolf
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
sys.path.append('/root/A_Dissertation'); from common.paths import OUTPUTS
RES=OUTPUTS/'experiments'/'s3_filter'; OUT=OUTPUTS/'experiments'/'cluster_probe'
z=np.load(RES/'perbox_multi2.npz')
CX=z['cx'].astype(float); CY=z['cy'].astype(float); PM=z['p_mean']; PV=z['p_var']; TV=z['tta_var']; TP=z['is_tp'].astype(int); EMB=z['emb']; FID=z['field_id'].astype(int)
dev=FID<95; test=~dev
# typicality (existing multi-signal feature)
dtp=np.where(dev&(TP==1))[0]; mu=EMB[dtp].mean(0); sd=EMB[dtp].std(0)+1e-6
pca=PCA(40,random_state=0).fit((EMB[dtp]-mu)/sd); lw=LedoitWolf().fit(pca.transform((EMB[dtp]-mu)/sd))
MAHA=lw.mahalanobis(pca.transform((EMB-mu)/sd))
# spatial features per box within its field
nd100=np.zeros(len(TP)); nd200=np.zeros(len(TP)); dknn=np.zeros(len(TP)); k=3
for f in np.unique(FID):
    idx=np.where(FID==f)[0]
    if len(idx)<2:
        dknn[idx]=1e4; continue
    P=np.column_stack([CX[idx],CY[idx]]); D=np.sqrt(((P[:,None,:]-P[None,:,:])**2).sum(-1)); np.fill_diagonal(D,np.inf)
    nd100[idx]=(D<=100).sum(1); nd200[idx]=(D<=200).sum(1)
    Ds=np.sort(D,1); dknn[idx]=Ds[:,min(k-1,len(idx)-2)]
def auc(feats):
    X=np.column_stack(feats); sc=StandardScaler().fit(X[dev])
    m=LogisticRegression(max_iter=2000).fit(sc.transform(X[dev]),TP[dev])
    return roc_auc_score(TP[test], m.predict_proba(sc.transform(X[test]))[:,1])
base=[PM,PV,TV,MAHA]; spat=[nd100,nd200,dknn]
res={'n_boxes':int(len(TP)),'n_dev':int(dev.sum()),'n_test':int(test.sum()),'tp_frac_test':float(TP[test].mean())}
res['auc_base']=float(auc(base))
res['auc_spatial_only']=float(auc(spat))
res['auc_base_plus_spatial']=float(auc(base+spat))
# univariate AUC of each spatial feature (sign-agnostic)
for nm,fe in [('nd100',nd100),('nd200',nd200),('dknn',dknn)]:
    a=roc_auc_score(TP[test],fe[test]); res['uni_'+nm]=float(max(a,1-a))
# also: do TPs have more neighbours than FPs? (means)
res['nd200_mean_TP']=float(nd200[test&(TP==1)].mean()); res['nd200_mean_FP']=float(nd200[test&(TP==0)].mean())
json.dump(res,open(OUT/'spatial_signal.json','w'),indent=2)
print(json.dumps(res,indent=2),flush=True)
print(f"\nAUC base {res['auc_base']:.3f} -> base+spatial {res['auc_base_plus_spatial']:.3f}  (Δ {res['auc_base_plus_spatial']-res['auc_base']:+.3f}); spatial-only {res['auc_spatial_only']:.3f}",flush=True)
print('SPATIAL_SIGNAL_DONE',flush=True)
