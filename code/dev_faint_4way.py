# -*- coding: utf-8 -*-
"""Unified 4-way comparison on ONE protocol (perbox_multi2): A0 / naive-variance / MC-protected / multi-signal(joint).
Model-based faint. Dev-calibrated, test-evaluated; orig split + 3-split mean+/-std. -> four_way.json + console."""
import sys, json, numpy as np
sys.path.append('/root/A_Dissertation'); from common.paths import OUTPUTS
from sklearn.decomposition import PCA; from sklearn.covariance import LedoitWolf
RES=OUTPUTS/'experiments'/'s3_filter'
z=np.load(RES/'perbox_multi2.npz'); NG=np.load(RES/'ngt_field.npz')['ngt']
PM=z['p_mean']; PV=z['p_var']; TV=z['tta_var']; SC=z['score']; TP=z['is_tp']; EMB=z['emb']; FID=z['field_id'].astype(int)
allf=np.unique(FID); cj=json.load(open(RES/'joint_calib.json'))['orig']['joint']['cal']
def faint_mask(mask):
    fm=np.zeros(len(TP),bool); m=mask&(TP==1)
    if m.sum()>0: fm=m&(PM<=np.quantile(PM[m],1/3))
    return fm
def maha_fit(tr):
    dtp=tr[TP[tr]==1]; mu=EMB[dtp].mean(0); sd=EMB[dtp].std(0)+1e-6
    pca=PCA(40,random_state=0).fit((EMB[dtp]-mu)/sd); lw=LedoitWolf().fit(pca.transform((EMB[dtp]-mu)/sd))
    return lw.mahalanobis(pca.transform((EMB-mu)/sd))
def metrics(idx,keep,n_gt):
    tpv=TP[idx][keep]; sc=SC[idx][keep]; TPn=int(tpv.sum()); FPn=len(tpv)-TPn
    Pp=TPn/max(1,TPn+FPn); Rr=TPn/max(1,n_gt)
    o=np.argsort(-sc); t=tpv[o]; ctp=np.cumsum(t); cfp=np.cumsum(1-t); rec=ctp/max(1,n_gt); pre=ctp/np.maximum(1,ctp+cfp)
    mr=np.r_[0,rec,1]; mp=np.r_[0,pre,0]
    for i in range(len(mp)-1,0,-1): mp[i-1]=max(mp[i-1],mp[i])
    j=np.where(mr[1:]!=mr[:-1])[0]; ap=float(np.sum((mr[j+1]-mr[j])*mp[j+1]))
    return Pp,Rr,ap,TPn,FPn
def run(devf,testf):
    di=np.where(np.isin(FID,devf))[0]; ti=np.where(np.isin(FID,testf))[0]; maha=maha_fit(di)
    ngt_d=int(NG[devf].sum()); ngt_t=int(NG[testf].sum())
    Df=faint_mask(np.isin(np.arange(len(TP)),di)); Tf=faint_mask(np.isin(np.arange(len(TP)),ti)); Tfl=Tf[ti]
    bMC=np.quantile(PV[Df],0.10)
    aMC=max(np.quantile(PM[di],np.linspace(0.2,0.95,16)),key=lambda a:(lambda P,R,ap,_,__:2*P*R/max(1e-9,P+R))(*metrics(di,(PV[di]>=bMC)|(PM[di]>=a),ngt_d)))
    t_naive=np.quantile(PV[di],0.5)      # naive: drop the high-variance half (keeps low-variance -> drops faint)
    rules={'A0':np.ones(len(ti),bool),
           'naive variance':PV[ti]<=t_naive,
           'MC-protected':(PV[ti]>=bMC)|(PM[ti]>=aMC),
           'multi-signal (joint)':((PV[ti]>=cj['bMC'])|(TV[ti]>=cj['bTTA'])|(PM[ti]>=cj['a']))&(maha[ti]<=cj['tau'])}
    out={}
    for nm,keep in rules.items():
        P,R,ap,TPn,FPn=metrics(ti,keep,ngt_t); fr=float(keep[Tfl].mean()) if Tfl.sum() else None
        out[nm]=dict(P=round(P,3),R=round(R,3),AP=round(ap,3),faint=round(fr,3),FP=FPn)
    return out
dev0=allf[allf<95]; test0=allf[allf>=95]; o=run(dev0,test0)
print('=== 4-way, ORIGINAL split (test) ===')
print(f'  {"filter":22s} {"P":>6s} {"R":>6s} {"AP":>6s} {"faint":>6s} {"FP":>6s}')
for k,v in o.items(): print(f'  {k:22s} {v["P"]:.3f} {v["R"]:.3f} {v["AP"]:.3f} {v["faint"]:.3f} {v["FP"]:6d}')
agg={k:{'P':[],'R':[],'AP':[],'faint':[],'FP':[]} for k in o}
for s in range(3):
    ff=allf.copy(); np.random.default_rng(400+s).shuffle(ff); r=run(ff[:len(ff)//2],ff[len(ff)//2:])
    for k in r:
        for m in ['P','R','AP','faint','FP']: agg[k][m].append(r[k][m])
print('\n=== 4-way, 3-split mean+/-std ===')
prog3={}
for k in o:
    prog3[k]={m:[float(np.mean(agg[k][m])),float(np.std(agg[k][m]))] for m in ['P','R','AP','faint','FP']}
    print(f'  {k:22s} P {np.mean(agg[k]["P"]):.3f}±{np.std(agg[k]["P"]):.3f}  R {np.mean(agg[k]["R"]):.3f}  AP {np.mean(agg[k]["AP"]):.3f}  faint {np.mean(agg[k]["faint"]):.3f}  FP {int(np.mean(agg[k]["FP"]))}')
json.dump(dict(orig=o,prog3=prog3),open(RES/'four_way.json','w'),indent=2); print('\n4WAY_DONE',flush=True)
