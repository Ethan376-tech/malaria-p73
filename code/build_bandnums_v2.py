# -*- coding: utf-8 -*-
"""Print every text-referenced number for the §3.1 by-band / confusion narrative (seed-averaged v2).
Reads emb_*_v2*.npy + bag_meta_v2.csv. Output: spec(neg), recall(pos_lo), recall(pos_hi), FP/FN counts per encoder."""
import sys
import numpy as np, pandas as pd
sys.path.append('/root/A_Dissertation'); from common.paths import OUTPUTS
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, confusion_matrix, roc_auc_score
S=OUTPUTS/'experiments'/'stageD_bag'; bm=pd.read_csv(S/'bag_meta_v2.csv'); ibm=(bm.dataset=='ibadan').values
y=bm.label01.values[ibm]; band=bm.para_band.values[ibm]
SEEDF={0:'emb_{e}_v2.npy',1:'emb_{e}_v2_s1.npy',2:'emb_{e}_v2_s2.npy'}
XI={s:{e:np.load(S/SEEDF[s].format(e=e))[ibm].max(1) for e in ['supvit','reddino','thickdino']} for s in SEEDF}
def oof_seed(Xe,split=0):
    p=np.zeros(len(y)); skf=StratifiedKFold(3,shuffle=True,random_state=split)
    for tr,te in skf.split(Xe,y):
        sc=StandardScaler().fit(Xe[tr]); lr=LogisticRegression(max_iter=3000).fit(sc.transform(Xe[tr]),y[tr]); p[te]=lr.predict_proba(sc.transform(Xe[te]))[:,1]
    return p
def oof_avg(e): return np.mean([oof_seed(XI[s][e]) for s in SEEDF],axis=0)
for e in ['supvit','reddino','thickdino']:
    p=oof_avg(e); fpr,tpr,thr=roc_curve(y,p); j=np.argmax(tpr-fpr); t=thr[j]; pred=(p>=t).astype(int)
    cm=confusion_matrix(y,pred); sens=cm[1,1]/cm[1].sum(); spec=cm[0,0]/cm[0].sum()
    rec_lo=(pred[band=='pos_lo']==1).mean(); rec_hi=(pred[band=='pos_hi']==1).mean()
    print(f'{e}: AUC {roc_auc_score(y,p):.3f} | spec {spec:.3f} (FP {cm[0,1]}) | sens {sens:.3f} (FN {cm[1,0]}) | recall pos_lo {rec_lo:.3f} | recall pos_hi {rec_hi:.3f}')
print('counts: neg',int((band=="neg").sum()),'pos_lo',int((band=="pos_lo").sum()),'pos_hi',int((band=="pos_hi").sum()))
