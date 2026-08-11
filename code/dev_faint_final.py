# -*- coding: utf-8 -*-
"""Consolidated FINAL numbers for the multi-signal faintness filter, ONE consistent pipeline:
model-based faint (p_mean bottom third), progression A0 -> MC-var -> MC or TTA -> multi-signal(JOINT calib).
Single-signal/OR stages use independent per-signal floors (their natural calibration); the multi-signal endpoint
uses joint constrained calibration (max dev precision s.t. faint>=0.90 & AP>=A0). Reports orig-split point + FP,
3-split mean+/-std, discrimination AUC, and the b_MC tightening (independent vs joint). Rewrites fig data json.
Uses perbox_multi2.npz + ngt_field.npz.  Output -> outputs/experiments/s3_filter/final_numbers.json + console."""
import sys, json, numpy as np
sys.path.append('/root/A_Dissertation'); from common.paths import OUTPUTS
from sklearn.decomposition import PCA; from sklearn.covariance import LedoitWolf; from sklearn.metrics import roc_auc_score
RES=OUTPUTS/'experiments'/'s3_filter'
z=np.load(RES/'perbox_multi2.npz'); NG=np.load(RES/'ngt_field.npz')['ngt']
PM=z['p_mean']; PV=z['p_var']; TV=z['tta_var']; SC=z['score']; TP=z['is_tp']; EMB=z['emb']; FID=z['field_id'].astype(int)
allf=np.unique(FID)
def faint_mask(mask):
    fm=np.zeros(len(TP),bool); m=mask&(TP==1)
    if m.sum()>0: fm=m&(PM<=np.quantile(PM[m],1/3))
    return fm
def maha_fit(train_idx):
    dtp=train_idx[TP[train_idx]==1]; mu=EMB[dtp].mean(0); sd=EMB[dtp].std(0)+1e-6
    pca=PCA(40,random_state=0).fit((EMB[dtp]-mu)/sd); lw=LedoitWolf().fit(pca.transform((EMB[dtp]-mu)/sd))
    return lw.mahalanobis(pca.transform((EMB-mu)/sd)), dtp
def metrics(idx,keep,n_gt):
    tpv=TP[idx][keep]; sc=SC[idx][keep]; TPn=int(tpv.sum()); FPn=len(tpv)-TPn
    Pp=TPn/max(1,TPn+FPn); Rr=TPn/max(1,n_gt)
    o=np.argsort(-sc); t=tpv[o]; ctp=np.cumsum(t); cfp=np.cumsum(1-t); rec=ctp/max(1,n_gt); pre=ctp/np.maximum(1,ctp+cfp)
    mr=np.r_[0,rec,1]; mp=np.r_[0,pre,0]
    for i in range(len(mp)-1,0,-1): mp[i-1]=max(mp[i-1],mp[i])
    j=np.where(mr[1:]!=mr[:-1])[0]; ap=float(np.sum((mr[j+1]-mr[j])*mp[j+1]))
    return Pp,Rr,ap,TPn,FPn
def a_for(dev_idx,protect_fn,ngt_d):
    best=(-1,0.)
    for a in np.quantile(PM[dev_idx],np.linspace(0.2,0.95,16)):
        keep=protect_fn(dev_idx,a); P,R,ap,_,_=metrics(dev_idx,keep,ngt_d); f1=2*P*R/max(1e-9,P+R)
        if f1>best[0]: best=(f1,float(a))
    return best[1]
def joint_calib(dev_idx,maha,R=0.90):
    dmask=np.isin(np.arange(len(TP)),dev_idx); Df=faint_mask(dmask); fv=PV[Df]; ft=TV[Df]; Dfl=Df[dev_idx]
    ngt_d=int(NG[np.unique(FID[dev_idx])].sum()); apA0=metrics(dev_idx,np.ones(len(dev_idx),bool),ngt_d)[2]
    gMC=[np.quantile(fv,q) for q in [0,.1,.2,.3,.4,.5]]; gTTA=[np.quantile(ft,q) for q in [0,.1,.2,.3,.4,.5]]
    gA=list(np.quantile(PM[dev_idx],np.linspace(0.2,0.95,8))); gT=[np.quantile(maha[dev_idx[TP[dev_idx]==1]],q) for q in [.75,.8,.85,.9,.95,1.]]
    best=(-1,None)
    for bMC in gMC:
        for bTTA in gTTA:
            for a in gA:
                base=(PV[dev_idx]>=bMC)|(TV[dev_idx]>=bTTA)|(PM[dev_idx]>=a)
                for tau in gT:
                    keep=base&(maha[dev_idx]<=tau); fr=keep[Dfl].mean() if Dfl.sum()>0 else 0
                    if fr<R: continue
                    P,R2,ap,_,_=metrics(dev_idx,keep,ngt_d)
                    if ap<apA0-0.003: continue
                    if P>best[0]: best=(P,dict(bMC=float(bMC),bTTA=float(bTTA),a=float(a),tau=float(tau)))
    return best[1]
def progression(dev_fids,test_fids):
    di=np.where(np.isin(FID,dev_fids))[0]; ti=np.where(np.isin(FID,test_fids))[0]
    maha,dtp=maha_fit(di); ngt_d=int(NG[dev_fids].sum()); ngt_t=int(NG[test_fids].sum())
    Tf=faint_mask(np.isin(np.arange(len(TP)),ti)); Tfl=Tf[ti]; Df=faint_mask(np.isin(np.arange(len(TP)),di))
    bMC=np.quantile(PV[Df],0.10); bTTA=np.quantile(TV[Df],0.10)
    aMC=a_for(di,lambda idx,a:(PV[idx]>=bMC)|(PM[idx]>=a),ngt_d)
    aOR=a_for(di,lambda idx,a:(PV[idx]>=bMC)|(TV[idx]>=bTTA)|(PM[idx]>=a),ngt_d)
    cj=joint_calib(di,maha)
    def row(keep):
        P,R,ap,TPn,FPn=metrics(ti,keep,ngt_t); fr=float(keep[Tfl].mean()) if Tfl.sum()>0 else None
        return dict(P=round(P,3),R=round(R,3),AP=round(ap,3),faint=round(fr,3),FP=FPn)
    R={}
    R['A0']=row(np.ones(len(ti),bool))
    R['MC-var']=row((PV[ti]>=bMC)|(PM[ti]>=aMC))
    R['MC or TTA']=row((PV[ti]>=bMC)|(TV[ti]>=bTTA)|(PM[ti]>=aOR))
    R['multi-signal (joint)']=row(((PV[ti]>=cj['bMC'])|(TV[ti]>=cj['bTTA'])|(PM[ti]>=cj['a']))&(maha[ti]<=cj['tau']))
    return R, dict(indep_bMC=float(bMC), joint_bMC=float(cj['bMC']), joint=cj)
# discrimination AUC (orig split)
dev0=allf[allf<95]; test0=allf[allf>=95]; ti0=np.where(np.isin(FID,test0))[0]
maha0,_=maha_fit(np.where(np.isin(FID,dev0))[0]); Tf0=faint_mask(np.isin(np.arange(len(TP)),ti0)); fp0=(TP==0)&np.isin(np.arange(len(TP)),ti0)
def auc(sig,pos,neg,hi=True):
    s=sig if hi else -sig; y=np.r_[np.ones(pos.sum()),np.zeros(neg.sum())]; return round(float(roc_auc_score(y,np.r_[s[pos],s[neg]])),3)
disc={'MC var':auc(PV,Tf0,fp0),'TTA var':auc(TV,Tf0,fp0),'typicality':auc(maha0,Tf0,fp0,hi=False)}
prog0,cal=progression(dev0,test0)
print('=== FINAL progression (orig split, test; model-based faint; multi-signal JOINT-calibrated) ===')
for k,v in prog0.items(): print(f'  {k:22s} P {v["P"]:.3f}  R {v["R"]:.3f}  AP {v["AP"]:.3f}  faint {v["faint"]:.3f}  FP {v["FP"]}')
print(f'  b_MC tightened by joint calib: {cal["indep_bMC"]:.5f} -> {cal["joint_bMC"]:.5f} ({cal["joint_bMC"]/cal["indep_bMC"]:.1f}x)')
print(f'  discrimination AUC (faint-TP vs FP): MC {disc["MC var"]}  TTA {disc["TTA var"]}  typicality {disc["typicality"]}')
# 3-split mean+/-std
agg={k:{'P':[],'AP':[],'faint':[],'FP':[]} for k in prog0}
for s in range(3):
    ff=allf.copy(); np.random.default_rng(300+s).shuffle(ff); r,_=progression(ff[:len(ff)//2],ff[len(ff)//2:])
    for k in r:
        for m in ['P','AP','faint','FP']: agg[k][m].append(r[k][m])
print('\n=== 3-split mean+/-std ===')
prog3={}
for k in prog0:
    prog3[k]={m:[float(np.mean(agg[k][m])),float(np.std(agg[k][m]))] for m in ['P','AP','faint','FP']}
    print(f'  {k:22s} P {np.mean(agg[k]["P"]):.3f}+/-{np.std(agg[k]["P"]):.3f}  AP {np.mean(agg[k]["AP"]):.3f}  faint {np.mean(agg[k]["faint"]):.3f}  FP {int(np.mean(agg[k]["FP"]))}')
json.dump(dict(orig=prog0,orig_calib=cal,disc=disc,prog3=prog3),open(RES/'final_numbers.json','w'),indent=2)
print('\nFINAL_NUMBERS_DONE',flush=True)
