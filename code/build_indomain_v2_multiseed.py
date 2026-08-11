# -*- coding: utf-8 -*-
"""Seed-averaged (tile-sampling seeds 0,1,2) re-run of the in-domain analyses on the 441-bag v2 set.
Headline AUC = mean over 3 seeds x 15 folds (45 fold-AUCs); seed_std = std of the 3 per-seed means (tile-sampling
stability); CI = bootstrap over the 45. Figures use OOF probability averaged over the 3 embedding seeds (fixed CV split).
Final figures OVERWRITE the *_v2 ones (now seed-averaged). Outputs -> stageD_bag/*_v2*.csv|json + classviz/figures *_v2.png"""
import sys, json
import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
sys.path.append('/root/A_Dissertation'); from common.paths import OUTPUTS
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
plt.rcParams.update({'font.family':'serif','font.size':11})
S=OUTPUTS/'experiments'/'stageD_bag'; CV=OUTPUTS/'experiments'/'classviz'; FIG=OUTPUTS/'experiments'/'figures'
bm=pd.read_csv(S/'bag_meta_v2.csv'); ibm=(bm.dataset=='ibadan').values
yall=bm.label01.values; y=yall[ibm]; band=bm.para_band.values[ibm]
ENCS=['supvit','reddino','thickdino','supcon']
SEEDFILES={0:'emb_{e}_v2.npy',1:'emb_{e}_v2_s1.npy',2:'emb_{e}_v2_s2.npy'}
# load per seed: max-pool ibadan + pooled-all
XI={s:{e:np.load(S/SEEDFILES[s].format(e=e))[ibm].max(1) for e in ENCS} for s in SEEDFILES}
XP={s:{e:np.load(S/SEEDFILES[s].format(e=e)).max(1) for e in ['supvit','reddino','thickdino']} for s in SEEDFILES}
print('seeds loaded:',list(SEEDFILES),'| ibadan',ibm.sum(),'pos',int(y.sum()),flush=True)

def boot_ci(v,n=2000):
    v=np.array(v); idx=np.random.RandomState(0).randint(0,len(v),(n,len(v))); m=v[idx].mean(1); return np.percentile(m,[2.5,97.5])

# ===== (1) in-domain repeated 5x3, per seed then pooled over seeds =====
rskf=RepeatedStratifiedKFold(n_splits=3,n_repeats=5,random_state=0); splits=list(rskf.split(XI[0]['thickdino'],y))
allfold={e:[] for e in ENCS}; permean={e:{} for e in ENCS}
for s in SEEDFILES:
    perseed={e:[] for e in ENCS}
    for tr,te in splits:
        for e in ENCS:
            sc=StandardScaler().fit(XI[s][e][tr]); lr=LogisticRegression(max_iter=3000).fit(sc.transform(XI[s][e][tr]),y[tr])
            a=roc_auc_score(y[te],lr.predict_proba(sc.transform(XI[s][e][te]))[:,1]); perseed[e].append(a); allfold[e].append(a)
    for e in ENCS: permean[e][s]=np.mean(perseed[e])
rows=[]
for e in ENCS:
    seedmeans=[permean[e][s] for s in SEEDFILES]; ci=boot_ci(allfold[e])
    rows.append(dict(encoder=e,indomain_mean=round(np.mean(allfold[e]),3),seed_std=round(np.std(seedmeans),3),
                     fold_std=round(np.std(allfold[e]),3),ci95_lo=round(ci[0],3),ci95_hi=round(ci[1],3),
                     seed_means=[round(x,3) for x in seedmeans]))
diff=np.array(allfold['thickdino'])-np.array(allfold['reddino'])
paired=dict(mean_diff=round(diff.mean(),3),pct_thickdino_better=round(100*(diff>0).mean(),1),dci=[round(x,3) for x in boot_ci(diff)],n=len(diff))
pd.DataFrame(rows).set_index('encoder').to_csv(S/'stats_indomain_repeatedCV_v2.csv')
json.dump(paired,open(S/'paired_thickdino_vs_reddino_v2.json','w'),indent=2)
print('\n=== (1) in-domain seed-averaged (45 fold-AUCs) ==='); print(pd.DataFrame(rows).to_string(index=False)); print('paired TD-RD:',paired,flush=True)

# ===== (2) pooled-MILCA, per seed then averaged =====
def cv(X,yy,reps=5):
    rk=RepeatedStratifiedKFold(n_splits=3,n_repeats=reps,random_state=0); a=[]
    for tr,te in rk.split(X,yy):
        sc=StandardScaler().fit(X[tr]); lr=LogisticRegression(max_iter=3000).fit(sc.transform(X[tr]),yy[tr])
        a.append(roc_auc_score(yy[te],lr.predict_proba(sc.transform(X[te]))[:,1]))
    return np.mean(a)
prows=[]
for e in ['supvit','reddino','thickdino']:
    po=[cv(XP[s][e],yall) for s in SEEDFILES]; io=[cv(XP[s][e][ibm],yall[ibm]) for s in SEEDFILES]; co=[cv(XP[s][e][~ibm],yall[~ibm]) for s in SEEDFILES]
    prows.append(dict(encoder=e,pooled_mean=round(np.mean(po),3),pooled_seed_std=round(np.std(po),3),ibadan_only=round(np.mean(io),3),chitt1_only=round(np.mean(co),3)))
pd.DataFrame(prows).to_csv(S/'pooled_milca_cv_v2.csv',index=False)
print('\n=== (2) pooled-MILCA seed-averaged ==='); print(pd.DataFrame(prows).to_string(index=False),flush=True)

# ===== OOF prob averaged over seeds (fixed split) for figures =====
COL={'supvit':'#999999','reddino':'#4477aa','thickdino':'#228833'}; NAME={'supvit':'SupViT','reddino':'RedDino','thickdino':'ThickDINO'}
def oof_seed(Xe,seed_split=0):
    p=np.zeros(len(y)); skf=StratifiedKFold(3,shuffle=True,random_state=seed_split)
    for tr,te in skf.split(Xe,y):
        sc=StandardScaler().fit(Xe[tr]); lr=LogisticRegression(max_iter=3000).fit(sc.transform(Xe[tr]),y[tr]); p[te]=lr.predict_proba(sc.transform(Xe[te]))[:,1]
    return p
def oof_avg(e,seed_split=0): return np.mean([oof_seed(XI[s][e],seed_split) for s in SEEDFILES],axis=0)

# (3) ROC: mean over 5 CV-splits of the seed-averaged prob
def mean_roc(e,reps=5):
    grid=np.linspace(0,1,101); tprs=[]; aucs=[]
    for cs in range(reps):
        p=oof_avg(e,cs); fpr,tpr,_=roc_curve(y,p); tprs.append(np.interp(grid,fpr,tpr)); aucs.append(roc_auc_score(y,p))
    tprs=np.array(tprs); tprs[:,0]=0; return grid,tprs.mean(0),tprs.std(0),np.mean(aucs),np.std(aucs)
fig,ax=plt.subplots(figsize=(5.6,5.0))
tblnum={r['encoder']:r for r in rows}   # legend AUC = repeated-CV mean (matches Table 3.2)
for e in ['supvit','reddino','thickdino']:
    g,mn,sd,a,asd=mean_roc(e); t=tblnum[e]
    ax.plot(g,mn,color=COL[e],lw=2,label=f"{NAME[e]}  AUC {t['indomain_mean']:.3f}±{t['fold_std']:.3f}")
    if e=='thickdino': ax.fill_between(g,mn-sd,mn+sd,color=COL[e],alpha=0.15)
ax.plot([0,1],[0,1],ls=':',color='gray'); ax.set_xlabel('1 − specificity (FPR)'); ax.set_ylabel('sensitivity (TPR)')
ax.set_title('Ibadan in-domain sample classification (OOF, seed-averaged)',fontsize=10); ax.legend(loc='lower right',fontsize=9); ax.grid(alpha=0.3)
plt.tight_layout(); plt.savefig(CV/'fig_roc_indomain_v2.png',dpi=150,bbox_inches='tight'); plt.close(); print('(3) ROC done')

# (4) confusion triptych on seed-averaged prob
fig,axes=plt.subplots(1,3,figsize=(10.5,3.8)); cmstats=[]
for ax,e in zip(axes,['supvit','reddino','thickdino']):
    p=oof_avg(e); fpr,tpr,thr=roc_curve(y,p); j=np.argmax(tpr-fpr); t=thr[j]; cm=confusion_matrix(y,(p>=t).astype(int))
    sens=cm[1,1]/cm[1].sum(); spec=cm[0,0]/cm[0].sum(); cmstats.append(dict(encoder=e,sens=round(sens,3),spec=round(spec,3)))
    ax.imshow(cm,cmap='Blues',vmin=0,vmax=cm.max())
    for i in range(2):
        for k in range(2): ax.text(k,i,str(cm[i,k]),ha='center',va='center',fontsize=15,color='white' if cm[i,k]>cm.max()/2 else 'black')
    ax.set_xticks([0,1]); ax.set_xticklabels(['pred neg','pred pos'],fontsize=9); ax.set_yticks([0,1]); ax.set_yticklabels(['true neg','true pos'],fontsize=9)
    ax.set_title(f'{NAME[e]}\nsens {sens:.2f} · spec {spec:.2f}',fontsize=11)
plt.tight_layout(); plt.savefig(CV/'fig_confusion_triptych_v2.png',dpi=150,bbox_inches='tight'); plt.close(); print('(4) confusion:',cmstats)

# (5) sensitivity by band on seed-averaged prob
fig,ax=plt.subplots(figsize=(6.4,4.0)); xx=np.arange(3); w=0.26
groups=[('neg',band=='neg'),('pos_lo',band=='pos_lo'),('pos_hi',band=='pos_hi')]
for ei,e in enumerate(['supvit','reddino','thickdino']):
    p=oof_avg(e); fpr,tpr,thr=roc_curve(y,p); jj=np.argmax(tpr-fpr); to=thr[jj]; pred=(p>=to).astype(int)
    vals=[(pred[m]==0).mean() if g=='neg' else (pred[m]==1).mean() for g,m in groups]
    ax.bar(xx+(ei-1)*w,vals,w,color=COL[e],label=NAME[e])
ax.axhline(0.5,ls=':',color='gray'); ax.set_xticks(xx); ax.set_xticklabels(['specificity\n(neg)','recall\npos_lo','recall\npos_hi'])
ax.set_ylim(0,1.05); ax.set_ylabel('rate at Youden threshold'); ax.set_title('Ibadan: detection by parasitaemia band (seed-averaged)',fontsize=10)
ax.legend(fontsize=9); ax.grid(axis='y',alpha=0.3)
plt.tight_layout(); plt.savefig(CV/'fig_sens_by_band_v2.png',dpi=150,bbox_inches='tight'); plt.close(); print('(5) band fig done')
json.dump(cmstats,open(CV/'confusion_v2_stats.json','w'),indent=2)
print('\nMULTISEED_DONE',flush=True)
