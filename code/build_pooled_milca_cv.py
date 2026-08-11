# -*- coding: utf-8 -*-
"""Reproduces the 'In-domain pooled (MILCA-matched)' row of report Table 3.2, plus the Ibadan-only and
Chittagong-1-only decomposition. Pooled 3-fold CV (repeated 5x) of max-pooled frozen bag embeddings + logistic.
(Was originally an inline command; saved here so every report number is reproducible from code.)
Output -> outputs/experiments/stageD_bag/pooled_milca_cv.csv"""
import sys, numpy as np, pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
sys.path.append('/root/A_Dissertation'); from common.paths import OUTPUTS
S = OUTPUTS/'experiments'/'stageD_bag'
bm = pd.read_csv(S/'bag_meta.csv'); y = bm.label01.values

def cv(X, y, reps=5):
    rk = RepeatedStratifiedKFold(n_splits=3, n_repeats=reps, random_state=0); a = []
    for tr, te in rk.split(X, y):
        sc = StandardScaler().fit(X[tr]); lr = LogisticRegression(max_iter=2000).fit(sc.transform(X[tr]), y[tr])
        a.append(roc_auc_score(y[te], lr.predict_proba(sc.transform(X[te]))[:, 1]))
    return np.mean(a), np.std(a)

rows = []
for e in ['supvit', 'reddino', 'thickdino']:
    X = np.load(S/f'emb_{e}.npy').max(1)                 # (409,384) max-pool over 256 tiles
    pm, ps = cv(X, y); ibm = (bm.dataset == 'ibadan').values
    im, _ = cv(X[ibm], y[ibm]); cm, _ = cv(X[~ibm], y[~ibm])
    rows.append(dict(encoder=e, pooled_mean=round(pm, 3), pooled_std=round(ps, 3),
                     ibadan_only=round(im, 3), chitt1_only=round(cm, 3)))
    print(rows[-1])
pd.DataFrame(rows).to_csv(S/'pooled_milca_cv.csv', index=False)
print('MILCA reference (pooled in-domain 3-fold): 0.969 +/- 0.013')
print('WROTE', S/'pooled_milca_cv.csv')
