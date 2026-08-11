# -*- coding: utf-8 -*-
"""A3: sensitivity analysis of the faintness-protected filter's two thresholds (a = confidence floor, b = variance
floor). Sweeps both on the TEST perbox set and shows the operating REGION (not a single dev-chosen point): where
faint-TP retention stays >= 0.90 AND AP stays within 0.005 of the unfiltered A0. Reuses the exact S3 metrics/faint
definitions. Output -> figures/fig_filter_sensitivity.png + s3_filter/sensitivity.json"""
import sys, json
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.append('/root/A_Dissertation'); from common.paths import OUTPUTS
S3=OUTPUTS/'experiments'/'s3_filter'; FIG=OUTPUTS/'experiments'/'figures'; FIG.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.family':'serif','font.size':10})

z=np.load(S3/'perbox.npz'); PM,PV,SC,TP,n_gt=z['p_mean'],z['p_var'],z['score'],z['is_tp'],int(z['n_gt'])
final=json.load(open(S3/'final.json')); b_star=final['calib']['b_protect']; a_star=final['calib']['a_star']
A0_AP=final['A0']['AP']
faint=(TP==1)&(PM<=np.quantile(PM[TP==1],1/3))   # faintest third of TPs by predictive mean (S3 definition)

def metrics(keep):
    tp=TP[keep]; sc=SC[keep]; TPn=int(tp.sum()); FP=len(tp)-TPn; Pp=TPn/max(1,TPn+FP); Rr=TPn/max(1,n_gt)
    o=np.argsort(-sc); tps=tp[o]; ctp=np.cumsum(tps); cfp=np.cumsum(1-tps); rec=ctp/max(1,n_gt); pre=ctp/np.maximum(1,ctp+cfp)
    mr=np.concatenate([[0],rec,[1]]); mp=np.concatenate([[0],pre,[0]])
    for i in range(len(mp)-1,0,-1): mp[i-1]=max(mp[i-1],mp[i])
    idx=np.where(mr[1:]!=mr[:-1])[0]; ap=float(np.sum((mr[idx+1]-mr[idx])*mp[idx+1]))
    fr=float(keep[faint].mean()) if faint.sum()>0 else np.nan
    fp_removed=(final['A0']['FP']-FP)/max(1,final['A0']['FP'])
    return ap,Rr,fr,fp_removed

# grid: a over predictive-mean quantiles; b over a log range bracketing b_star
A=np.quantile(PM,np.linspace(0.05,0.95,28))
B=np.concatenate([[0.0],np.geomspace(1e-5,5e-3,27)])
AP=np.zeros((len(B),len(A))); FR=np.zeros_like(AP); FPR=np.zeros_like(AP)
for i,b in enumerate(B):
    for j,a in enumerate(A):
        keep=(PV>=b)|(PM>=a); AP[i,j],_,FR[i,j],FPR[i,j]=metrics(keep)
safe=(FR>=0.90)&(AP>=A0_AP-0.005)
# b-headroom AT the operating a* (how far b can rise from b* before leaving the safe region)
ja=int(np.argmin(np.abs(A-a_star)))
safe_b_at_astar=B[safe[:,ja]]
b_upper_at_astar=float(safe_b_at_astar.max()) if safe_b_at_astar.size else None
report=dict(a_star=a_star,b_star=b_star,A0_AP=A0_AP,
            safe_frac=round(float(safe.mean()),3),
            b_upper_at_astar=b_upper_at_astar,
            b_headroom_x_at_astar=round(b_upper_at_astar/b_star,2) if b_upper_at_astar else None,
            a_is_flat=bool(np.allclose(AP[1],AP[1,0],atol=0.01)))
json.dump(report,open(S3/'sensitivity.json','w'),indent=2); print(json.dumps(report,indent=2))

# ---- figure: two heatmaps (FP removed = benefit; faint retention = cost guard) with safe region + operating point
fig,ax=plt.subplots(1,2,figsize=(11,4.4))
yext=[0,len(B)-1]; xext=[A[0],A[-1]]
def bticks(axx):
    idx=[1,7,14,20,26]; axx.set_yticks(idx); axx.set_yticklabels([f'{B[k]:.0e}' for k in idx])
    axx.set_yticks([0]+idx); axx.set_yticklabels(['0']+[f'{B[k]:.0e}' for k in idx])
im0=ax[0].imshow(FPR*100,origin='lower',aspect='auto',cmap='Blues',extent=[0,len(A)-1,0,len(B)-1])
ax[0].set_title('False positives removed (%)'); fig.colorbar(im0,ax=ax[0],fraction=0.046)
im1=ax[1].imshow(FR,origin='lower',aspect='auto',cmap='RdYlGn',vmin=0.5,vmax=1.0,extent=[0,len(A)-1,0,len(B)-1])
ax[1].set_title('Faint (low-parasitaemia) TP retention'); fig.colorbar(im1,ax=ax[1],fraction=0.046)
# overlay safe-region contour + operating point on both
ai=np.interp(a_star,A,np.arange(len(A))); bi=np.interp(b_star,B,np.arange(len(B)))
for k,axx in enumerate(ax):
    axx.contour(np.arange(len(A)),np.arange(len(B)),safe.astype(float),levels=[0.5],colors='k',linewidths=1.6,linestyles='--')
    axx.plot([ai],[bi],marker='*',ms=18,c='yellow',mec='k',mew=1.2,zorder=5)
    axx.set_xlabel('confidence floor  a  (predictive-mean percentile)')
    xt=[0,9,18,27]; axx.set_xticks(xt); axx.set_xticklabels([f'{A[t]:.2f}' for t in xt])
    bt=[0,7,14,20,26]; axx.set_yticks(bt); axx.set_yticklabels(['0']+[f'{B[t]:.0e}' for t in bt[1:]])
    axx.set_ylabel('variance floor  b')
ax[1].text(0.5,0.97,'dashed = safe region\n(faint>=0.90 & AP>=A0-0.005)\nyellow star = dev-chosen (a*, b*)',
           transform=ax[1].transAxes,va='top',ha='left',fontsize=8,bbox=dict(fc='white',alpha=0.8,ec='gray'))
plt.tight_layout(); out=FIG/'fig_filter_sensitivity.png'; plt.savefig(out,dpi=160,bbox_inches='tight'); print('WROTE',out)
