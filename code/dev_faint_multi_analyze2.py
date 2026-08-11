# -*- coding: utf-8 -*-
"""Faintness multi-signal PROTOTYPE v2 analysis: pick best PHYSICAL faint proxy, field-level bootstrap CI, 3-split robustness.
Loads perbox_multi2.npz + ngt_field.npz. Output -> outputs/experiments/s3_filter/multi_analysis2.json + console."""
import sys, json, numpy as np
sys.path.append('/root/A_Dissertation'); from common.paths import OUTPUTS
from sklearn.metrics import roc_auc_score
from sklearn.decomposition import PCA
from sklearn.covariance import LedoitWolf
RES=OUTPUTS/'experiments'/'s3_filter'; rng=np.random.default_rng(0)
z=np.load(RES/'perbox_multi2.npz'); NG=np.load(RES/'ngt_field.npz')['ngt']
A={k:z[k] for k in ['p_mean','p_var','tta_var','score','is_tp','weber','od_peak','od_integ','blob_area','purple','cx','cy','field_id','split']}
EMB=z['emb']; TP=A['is_tp']; FID=A['field_id'].astype(int)
PHYS=['weber','od_peak','od_integ','blob_area','purple']       # all: LOW value = faint (weak stain)
print(f'boxes {len(TP)} (TP {int(TP.sum())})  fields {len(np.unique(FID))}  ngt total {NG.sum()}')

# ---- physical proxy selection: which physical measure best tracks MODEL faintness? ----
tp=TP==1; fp=TP==0
print('\n[physical faint proxy vs model faintness]  corr with p_mean|TP (want +)  corr p_var|TP (want -)  AUC(TP vs FP, hi=stain)')
sel={}
for m in PHYS:
    v=A[m]; cpm=float(np.corrcoef(v[tp],A['p_mean'][tp])[0,1]); cpv=float(np.corrcoef(v[tp],A['p_var'][tp])[0,1])
    auc=float(roc_auc_score(np.r_[np.ones(tp.sum()),np.zeros(fp.sum())],np.r_[v[tp],v[fp]]))
    sel[m]=dict(corr_pmean=round(cpm,3),corr_pvar=round(cpv,3),auc_tp_fp=round(auc,3))
    print(f'  {m:10s} {cpm:+.3f}          {cpv:+.3f}          {auc:.3f}')
best_phys=max(PHYS,key=lambda m:sel[m]['corr_pmean'])           # best = low value most aligned with low model confidence
print(f'  -> best physical faint proxy: {best_phys}')

def faint_by(measure_or_pm,S_mask):
    """faint-TP mask (global index space) on boxes in S_mask, faintest third of TPs by the measure (or p_mean)."""
    fm=np.zeros(len(TP),bool); m=S_mask&tp
    if m.sum()>0:
        vals=A['p_mean'] if measure_or_pm=='pmean' else A[measure_or_pm]
        thr=np.quantile(vals[m],1/3); fm=m&(vals<=thr)
    return fm

# ---- metrics on a set of box indices ----
def metrics(idx,keep,n_gt):
    tpv=TP[idx][keep]; sc=A['score'][idx][keep]; TPn=int(tpv.sum()); FPn=len(tpv)-TPn
    Pp=TPn/max(1,TPn+FPn); Rr=TPn/max(1,n_gt)
    o=np.argsort(-sc); t=tpv[o]; ctp=np.cumsum(t); cfp=np.cumsum(1-t); rec=ctp/max(1,n_gt); pre=ctp/np.maximum(1,ctp+cfp)
    mr=np.r_[0,rec,1]; mp=np.r_[0,pre,0]
    for i in range(len(mp)-1,0,-1): mp[i-1]=max(mp[i-1],mp[i])
    j=np.where(mr[1:]!=mr[:-1])[0]; ap=float(np.sum((mr[j+1]-mr[j])*mp[j+1]))
    return Pp,Rr,ap,TPn,FPn

def run_split(dev_fids,test_fids,faint_measure,boot=0):
    dmask=np.isin(FID,dev_fids); tmask=np.isin(FID,test_fids)
    didx=np.where(dmask)[0]; tidx=np.where(tmask)[0]
    ngt_d=int(NG[dev_fids].sum()); ngt_t=int(NG[test_fids].sum())
    # typicality: Mahalanobis to dev-TP emb (PCA40 + LedoitWolf)
    dtp=didx[TP[didx]==1]; mu=EMB[dtp].mean(0); sd=EMB[dtp].std(0)+1e-6
    pca=PCA(40,random_state=0).fit((EMB[dtp]-mu)/sd); lw=LedoitWolf().fit(pca.transform((EMB[dtp]-mu)/sd))
    maha=lw.mahalanobis(pca.transform((EMB-mu)/sd))
    # dev-calibrate floors from dev faint-TP (faint defined by chosen measure)
    Dfaint=faint_by(faint_measure,dmask); Tfaint=faint_by(faint_measure,tmask)
    bvar=np.quantile(A['p_var'][Dfaint],0.10); btta=np.quantile(A['tta_var'][Dfaint],0.10)
    bphys=np.quantile(A[best_phys][Dfaint],0.90); tmaha=np.quantile(maha[dtp],0.90)
    def kmask(idx,protects,typ):
        kp=A['p_mean'][idx]>=a_star
        for p in protects:
            if p=='var': kp=kp|(A['p_var'][idx]>=bvar)
            elif p=='tta': kp=kp|(A['tta_var'][idx]>=btta)
            elif p=='phys': kp=kp|(A[best_phys][idx]<=bphys)
        if typ: kp=kp&(maha[idx]<=tmaha)
        return kp
    # a* on dev (max F1 for var rule)
    a_star=0.
    ag=np.quantile(A['p_mean'][didx],np.linspace(0.2,0.95,16)); bestf=-1
    for a in ag:
        a_star=a; keep=kmask(didx,['var'],False); P,R,ap,_,_=metrics(didx,keep,ngt_d); f1=2*P*R/max(1e-9,P+R)
        if f1>bestf: bestf,best_a=f1,float(a)
    a_star=best_a
    RULES=[('A0',[],False),('var',['var'],False),('tta',['tta'],False),(f'phys({best_phys})',['phys'],False),
           ('var+tta',['var','tta'],False),('tta +TYP',['tta'],True),('var+tta +TYP',['var','tta'],True),('TYP only',[],True)]
    rows={}
    tfaint_local=Tfaint[tidx]
    for nm,pr,tg in RULES:
        keep=np.ones(len(tidx),bool) if (nm=='A0') else kmask(tidx,pr,tg)
        P,R,ap,TPn,FPn=metrics(tidx,keep,ngt_t); fr=float(keep[tfaint_local].mean()) if tfaint_local.sum()>0 else None
        row=dict(P=round(P,3),R=round(R,3),AP=round(ap,3),faint=round(fr,3) if fr is not None else None,TP=TPn,FP=FPn)
        if boot>0:  # field-level bootstrap CI on P and AP
            Ps=[];APs=[]
            for _ in range(boot):
                bf=rng.choice(test_fids,len(test_fids),replace=True)
                bidx=np.concatenate([np.where(FID==f)[0] for f in bf]); bng=int(NG[bf].sum())
                bkeep=np.ones(len(bidx),bool) if nm=='A0' else kmask(bidx,pr,tg)
                Pp,_,ap2,_,_=metrics(bidx,bkeep,bng); Ps.append(Pp); APs.append(ap2)
            row['P_CI']=[round(float(np.percentile(Ps,2.5)),3),round(float(np.percentile(Ps,97.5)),3)]
            row['AP_CI']=[round(float(np.percentile(APs,2.5)),3),round(float(np.percentile(APs,97.5)),3)]
        rows[nm]=row
    return rows,dict(bvar=float(bvar),btta=float(btta),bphys=float(bphys),tmaha=float(tmaha),a_star=float(a_star),
                     dev_faint=int(Dfaint.sum()),test_faint=int(Tfaint.sum()))

# discrimination AUC (fixed original split)
allf=np.unique(FID); dev0=allf[allf<95]; test0=allf[allf>=95]
tmask0=np.isin(FID,test0); Tf0=faint_by(best_phys,tmask0); Tfp=faint_by('pmean',tmask0); fpm=(TP==0)&tmask0
# typicality on original split for AUC
dtp0=np.where(np.isin(FID,dev0)&(TP==1))[0]; mu=EMB[dtp0].mean(0); sd=EMB[dtp0].std(0)+1e-6
pca=PCA(40,random_state=0).fit((EMB[dtp0]-mu)/sd); lw=LedoitWolf().fit(pca.transform((EMB[dtp0]-mu)/sd)); MAHA=lw.mahalanobis(pca.transform((EMB-mu)/sd))
def auc(sig,pos,neg,hi=True):
    s=sig.copy() if hi else -sig; y=np.r_[np.ones(pos.sum()),np.zeros(neg.sum())]; return round(float(roc_auc_score(y,np.r_[s[pos],s[neg]])),3)
disc={}
for nm,sig,hi in [('MC var',A['p_var'],True),('TTA var',A['tta_var'],True),(f'phys(-{best_phys})',A[best_phys],False),('typicality(-maha)',MAHA,False),('p_mean',A['p_mean'],True)]:
    disc[nm]=dict(faintP=auc(sig,Tfp,fpm,hi),faintPHYS=auc(sig,Tf0,fpm,hi),allTP=auc(sig,tmask0&(TP==1),fpm,hi))
print('\n[discrimination AUC, orig split]  faintTP(pmean)vsFP | faintTP(phys)vsFP | allTP vsFP')
for nm in disc: print(f'  {nm:18s} {disc[nm]["faintP"]:.3f}              {disc[nm]["faintPHYS"]:.3f}             {disc[nm]["allTP"]:.3f}')

# main table: original split, field-bootstrap CI, faint defined by BEST PHYSICAL proxy
print(f'\n[MAIN: orig split, faint=phys({best_phys}), field-bootstrap 300]')
rows,cal=run_split(dev0,test0,best_phys,boot=300)
print(f'  calib bvar {cal["bvar"]:.5f} btta {cal["btta"]:.5f} bphys {cal["bphys"]:.3f} tmaha {cal["tmaha"]:.1f} a* {cal["a_star"]:.4f} | faint dev {cal["dev_faint"]} test {cal["test_faint"]}')
print(f'  {"rule":16s} {"P":>6s} {"P 95%CI":>14s} {"R":>6s} {"AP":>6s} {"AP 95%CI":>14s} {"faint":>6s} {"FP":>6s}')
for nm,r in rows.items():
    print(f'  {nm:16s} {r["P"]:.3f} [{r["P_CI"][0]:.3f},{r["P_CI"][1]:.3f}] {r["R"]:.3f} {r["AP"]:.3f} [{r["AP_CI"][0]:.3f},{r["AP_CI"][1]:.3f}] {str(r["faint"]):>6s} {r["FP"]:6d}')

# 3-split robustness (mean+-std) for key rules, faint=physical
print('\n[3-split robustness, mean +/- std over 3 random 50/50 field splits]')
key=['A0','var','tta','tta +TYP','var+tta +TYP','TYP only']; agg={k:{'P':[],'AP':[],'faint':[]} for k in key}
for s in range(3):
    ff=allf.copy(); np.random.default_rng(100+s).shuffle(ff); dv=ff[:len(ff)//2]; te=ff[len(ff)//2:]
    rw,_=run_split(dv,te,best_phys,boot=0)
    for k in key:
        agg[k]['P'].append(rw[k]['P']); agg[k]['AP'].append(rw[k]['AP'])
        if rw[k]['faint'] is not None: agg[k]['faint'].append(rw[k]['faint'])
print(f'  {"rule":16s} {"P":>13s} {"AP":>13s} {"faint":>13s}')
for k in key:
    P=agg[k]['P']; AP=agg[k]['AP']; F=agg[k]['faint']
    print(f'  {k:16s} {np.mean(P):.3f}+/-{np.std(P):.3f}  {np.mean(AP):.3f}+/-{np.std(AP):.3f}  {np.mean(F):.3f}+/-{np.std(F):.3f}')
json.dump(dict(phys_selection=sel,best_phys=best_phys,discrimination=disc,main=dict(calib=cal,rows=rows),
              robustness={k:{m:[float(np.mean(agg[k][m])),float(np.std(agg[k][m]))] for m in ['P','AP','faint']} for k in key}),
          open(RES/'multi_analysis2.json','w'),indent=2)
print('\nMULTI_ANALYZE2_DONE',flush=True)
