# -*- coding: utf-8 -*-
"""Joint constrained threshold calibration for the multi-signal faintness filter.
Rule: keep <=> [pvar>=bMC OR ttavar>=bTTA OR pmean>=a] AND maha<=tau.
Objective: maximize DEV precision s.t. dev faint-TP retention >= 0.90 AND dev AP >= A0 AP.
All four thresholds chosen JOINTLY by percentile grid (bounded/interpretable). Compares to the old INDEPENDENT
calibration; nested dev-train/dev-val overfit check; 3-split robustness. Uses perbox_multi2.npz + ngt_field.npz.
Output -> outputs/experiments/s3_filter/joint_calib.json + console."""
import sys, json, numpy as np
sys.path.append('/root/A_Dissertation'); from common.paths import OUTPUTS
from sklearn.decomposition import PCA
from sklearn.covariance import LedoitWolf
RES=OUTPUTS/'experiments'/'s3_filter'
z=np.load(RES/'perbox_multi2.npz'); NG=np.load(RES/'ngt_field.npz')['ngt']
PM=z['p_mean']; PV=z['p_var']; TV=z['tta_var']; SC=z['score']; TP=z['is_tp']; EMB=z['emb']; FID=z['field_id'].astype(int)
allf=np.unique(FID)

def faint_mask(mask):
    fm=np.zeros(len(TP),bool); m=mask&(TP==1)
    if m.sum()>0: fm=m&(PM<=np.quantile(PM[m],1/3))     # model-based faint (p_mean bottom third of TPs)
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
def keepmask(idx,maha,bMC,bTTA,a,tau):
    return ((PV[idx]>=bMC)|(TV[idx]>=bTTA)|(PM[idx]>=a))&(maha[idx]<=tau)

def independent_calib(dev_idx,maha,dtp):
    """OLD scheme: each floor set in isolation."""
    Df=faint_mask(np.isin(np.arange(len(TP)),dev_idx))
    bMC=np.quantile(PV[Df],0.10); bTTA=np.quantile(TV[Df],0.10); tau=np.quantile(maha[dtp],0.90)
    ngt_d=int(NG[np.unique(FID[dev_idx])].sum())
    ag=np.quantile(PM[dev_idx],np.linspace(0.2,0.95,16)); best=(-1,ag[0])
    for a in ag:  # a swept on the var-only rule (as in the old prototype)
        keep=(PV[dev_idx]>=bMC)|(PM[dev_idx]>=a); P,R,ap,_,_=metrics(dev_idx,keep,ngt_d); f1=2*P*R/max(1e-9,P+R)
        if f1>best[0]: best=(f1,float(a))
    return dict(bMC=float(bMC),bTTA=float(bTTA),a=float(best[1]),tau=float(tau))

def joint_calib(dev_idx,maha,R_target=0.90):
    """JOINT: max dev precision s.t. faint retention>=R_target AND AP>=A0 AP. Percentile-parameterised grid."""
    dmask=np.isin(np.arange(len(TP)),dev_idx); Df=faint_mask(dmask); fv=PV[Df]; ft=TV[Df]
    ngt_d=int(NG[np.unique(FID[dev_idx])].sum())
    Df_local=Df[dev_idx]
    apA0=metrics(dev_idx,np.ones(len(dev_idx),bool),ngt_d)[2]
    gMC=[np.quantile(fv,q) for q in [0,.1,.2,.3,.4,.5]]           # tighten each protect floor beyond the 10th pct
    gTTA=[np.quantile(ft,q) for q in [0,.1,.2,.3,.4,.5]]
    gA=list(np.quantile(PM[dev_idx],np.linspace(0.2,0.95,8)))
    gTAU=[np.quantile(maha[dev_idx[TP[dev_idx]==1]],q) for q in [0.75,0.8,0.85,0.9,0.95,1.0]]
    best=(-1,None)
    for bMC in gMC:
        for bTTA in gTTA:
            for a in gA:
                base=(PV[dev_idx]>=bMC)|(TV[dev_idx]>=bTTA)|(PM[dev_idx]>=a)
                for tau in gTAU:
                    keep=base&(maha[dev_idx]<=tau)
                    fr=keep[Df_local].mean() if Df_local.sum()>0 else 0
                    if fr<R_target: continue
                    P,R,ap,_,_=metrics(dev_idx,keep,ngt_d)
                    if ap<apA0-0.003: continue                    # guard: no AP loss vs A0
                    if P>best[0]: best=(P,dict(bMC=float(bMC),bTTA=float(bTTA),a=float(a),tau=float(tau)))
    return best[1] if best[1] else dict(bMC=float(gMC[1]),bTTA=float(gTTA[1]),a=float(gA[0]),tau=float(gTAU[3]))

def evalsplit(dev_fids,test_fids,report=False):
    dev_idx=np.where(np.isin(FID,dev_fids))[0]; test_idx=np.where(np.isin(FID,test_fids))[0]
    maha,dtp=maha_fit(dev_idx); ngt_t=int(NG[test_fids].sum())
    Tf=faint_mask(np.isin(np.arange(len(TP)),test_idx)); Tfl=Tf[test_idx]
    out={}
    for name,cal in [('independent',independent_calib(dev_idx,maha,dtp)),('joint',joint_calib(dev_idx,maha))]:
        keep=keepmask(test_idx,maha,**cal); P,R,ap,TPn,FPn=metrics(test_idx,keep,ngt_t)
        fr=float(keep[Tfl].mean()) if Tfl.sum()>0 else None
        out[name]=dict(cal={k:round(v,5) for k,v in cal.items()},P=round(P,3),R=round(R,3),AP=round(ap,3),faint=round(fr,3),FP=FPn,kept=int(keep.sum()))
        if report: print(f'  {name:12s} cal {out[name]["cal"]}\n               P {P:.3f} R {R:.3f} AP {ap:.3f} faint {fr:.3f} FP {FPn}')
    # A0
    P,R,ap,_,FPn=metrics(test_idx,np.ones(len(test_idx),bool),ngt_t); out['A0']=dict(P=round(P,3),AP=round(ap,3),FP=FPn)
    return out

# 1) original split, independent vs joint
dev0=allf[allf<95]; test0=allf[allf>=95]
print('=== ORIGINAL split: independent vs joint constrained calibration (test) ===')
print(f'  A0 baseline: (all kept)')
o=evalsplit(dev0,test0,report=True)
print(f'  A0           P {o["A0"]["P"]:.3f} AP {o["A0"]["AP"]:.3f} FP {o["A0"]["FP"]}')

# 2) overfit check: select joint thresholds on dev-train, confirm on dev-val (both inside dev)
dtr=dev0[::2]; dva=dev0[1::2]
mtr,_=maha_fit(np.where(np.isin(FID,dtr))[0]); cj=joint_calib(np.where(np.isin(FID,dtr))[0],mtr)
def frP(fids,maha,cal):
    idx=np.where(np.isin(FID,fids))[0]; ng=int(NG[fids].sum()); keep=keepmask(idx,maha,**cal)
    Tf=faint_mask(np.isin(np.arange(len(TP)),idx)); P,R,ap,_,FPn=metrics(idx,keep,ng)
    return round(P,3),round(float(keep[Tf[idx]].mean()),3),round(ap,3)
print('\n=== overfit check: joint thresholds selected on dev-train ===')
print(f'  chosen {"; ".join(f"{k}={v:.4f}" for k,v in cj.items())}')
print(f'  dev-train  P/faint/AP {frP(dtr,mtr,cj)}   dev-val P/faint/AP {frP(dva,mtr,cj)}')

# 3) robustness: 3 random 50/50 field splits, joint calibration each
print('\n=== 3-split robustness (joint calibration), test P/AP/faint mean+/-std ===')
Ps=[];APs=[];Fs=[]
for s in range(3):
    ff=allf.copy(); np.random.default_rng(200+s).shuffle(ff); dv=ff[:len(ff)//2]; te=ff[len(ff)//2:]
    r=evalsplit(dv,te)['joint']; Ps.append(r['P']); APs.append(r['AP']); Fs.append(r['faint'])
print(f'  joint: P {np.mean(Ps):.3f}+/-{np.std(Ps):.3f}  AP {np.mean(APs):.3f}+/-{np.std(APs):.3f}  faint {np.mean(Fs):.3f}+/-{np.std(Fs):.3f}')
json.dump(dict(orig=o,overfit=dict(chosen=cj,dev_train=frP(dtr,mtr,cj),dev_val=frP(dva,mtr,cj)),
              robust=dict(P=[float(np.mean(Ps)),float(np.std(Ps))],AP=[float(np.mean(APs)),float(np.std(APs))],faint=[float(np.mean(Fs)),float(np.std(Fs))])),
          open(RES/'joint_calib.json','w'),indent=2)
print('\nJOINT_CALIB_DONE',flush=True)
