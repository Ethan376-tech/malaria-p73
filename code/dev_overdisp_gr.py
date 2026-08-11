# -*- coding: utf-8 -*-
"""DEV (local, NOT report): characterise intra-field parasite aggregation.
1(c) pair-correlation g(r) on GT centroids (population) + Thomas cluster-process minimum-contrast fit -> scale sigma, mean cluster size.
1(b) over-dispersion: quadrat variance-to-mean ratio (VMR) at several scales + negative-binomial k + Fisher dispersion test;
      field-to-field VMR (grouped by sample); illustrative Poisson-vs-NB detection probability vs #fields (LoD link).
GT boxes from FASTMAL annotations only (no encoder). Output -> outputs/experiments/cluster_probe/aggreg_*.{png,json}."""
import sys, json, random
import numpy as np
from scipy.optimize import minimize
from scipy.stats import chi2
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
sys.path.append('/root/A_Dissertation'); from common.paths import RAW, OUTPUTS
OUT=OUTPUTS/'experiments'/'cluster_probe'; plt.rcParams.update({'font.family':'serif'})
W,H=2560,2160; A=W*H
def fastmal_fields():
    fm=RAW/'ibadan'/'FASTMAL_THICK_FILMS'; o=[]
    for samp in sorted(p for p in fm.iterdir() if p.is_dir()):
        js=list(samp.glob('*.json'))
        if not js: continue
        for fld in json.loads(js[0].read_text())['rois']:
            ip=samp/fld['image_name']
            if not ip.exists(): continue
            par=[(float(r['x'])+float(r['width'])/2,float(r['y'])+float(r['height'])/2) for r in fld['roi'] if 'PARASITE' in r.get('type','') and 'CROWD' not in r.get('type','')]
            if par: o.append((samp.name,ip,np.array(par,float)))
    return o
F=fastmal_fields()
print(f'{len(F)} positive fields, {sum(len(p) for _,_,p in F)} GT parasites',flush=True)

# ---------- 1(c) pair-correlation g(r) + Thomas fit ----------
edges=np.arange(0,801,25.0); ctr=(edges[:-1]+edges[1:])/2
num=np.zeros(len(ctr)); Kn=np.zeros(len(edges)-1)  # for g(r)
# pooled K via per-field translation-corrected, weighted by n(n-1)
Kacc=np.zeros(len(ctr)); wsum=0.0; lam_pts=0; lam_area=0.0
for _,_,P in F:
    n=len(P)
    if n<2: continue
    dx=np.abs(P[:,0:1]-P[:,0:1].T); dy=np.abs(P[:,1:2]-P[:,1:2].T); d=np.sqrt(dx*dx+dy*dy)
    e=A/((W-dx)*(H-dy)); iu=np.triu_indices(n,1); dd=d[iu]; ee=e[iu]
    lam=n/A; lam_pts+=n; lam_area+=A
    # g(r): weighted pair counts per annulus / expected
    hist,_=np.histogram(dd,bins=edges,weights=ee)
    num+=2*hist
    Kn+= (n*n/A)   # per-field expected density factor, per bin later
    # K_f(r) at ctr (cumulative)
    o=np.argsort(dd); dds=dd[o]; cw=np.cumsum(2*ee[o]); idx=np.searchsorted(dds,ctr,side='right')
    Kf=np.where(idx>0, cw[np.clip(idx-1,0,len(cw)-1)],0.0)/(n*(n-1)) * A
    Kacc+=Kf*(n*(n-1)); wsum+=n*(n-1)
Kobs=Kacc/wsum
lam_bar=lam_pts/lam_area
# g(r): denominator = sum_f (n^2/A) * 2*pi*r*dr ; approximate with pooled n^2/A
den=np.zeros(len(ctr))
for _,_,P in F:
    n=len(P)
    if n<2: continue
    den+= (n*n/A)*2*np.pi*ctr*(edges[1]-edges[0])
g=np.divide(num,den,out=np.full(len(ctr),np.nan),where=den>0)
# characteristic scale: peak of g beyond the hard core (r>=100)
mask=ctr>=100; gpk_r=float(ctr[mask][np.nanargmax(g[mask])]); gpk=float(np.nanmax(g[mask]))
# Thomas minimum contrast on K over r in [50,600]
fitm=(ctr>=50)&(ctr<=600)
def Kthomas(r,logsig,logkappa):
    sig=np.exp(logsig); kap=np.exp(logkappa)
    return np.pi*r*r + (1-np.exp(-r*r/(4*sig*sig)))/kap
def contrast(theta):
    Kt=Kthomas(ctr[fitm],*theta); q=0.25
    return np.sum((Kobs[fitm]**q - np.clip(Kt,1e-9,None)**q)**2)
best=None
for s0 in (4.0,5.0,6.0):
    for k0 in (-15,-13,-11):
        r=minimize(contrast,[s0,k0],method='Nelder-Mead')
        if best is None or r.fun<best.fun: best=r
sig=float(np.exp(best.x[0])); kappa=float(np.exp(best.x[1])); mean_cluster=float(lam_bar/kappa)
print(f'g(r): peak {gpk:.2f} at r={gpk_r:.0f}px | Thomas: sigma={sig:.0f}px, parent kappa={kappa:.2e}/px^2, mean cluster size={mean_cluster:.1f}',flush=True)

# ---------- 1(b) over-dispersion ----------
def quadrat_vmr(s):
    counts=[]
    nx=W//s; ny=H//s
    for _,_,P in F:
        gx=np.clip((P[:,0]//s).astype(int),0,nx-1); gy=np.clip((P[:,1]//s).astype(int),0,ny-1)
        cell=gy*nx+gx; c=np.bincount(cell,minlength=nx*ny); counts.append(c)
    c=np.concatenate(counts); m=c.mean(); v=c.var(ddof=1); Nq=len(c)
    D=(Nq-1)*v/m if m>0 else np.nan; p=1-chi2.cdf(D,Nq-1) if m>0 else np.nan
    k=m*m/(v-m) if v>m else np.inf
    return dict(s=int(s),nquad=int(Nq),mean=float(m),var=float(v),vmr=float(v/m),nb_k=float(k),fisher_p=float(p))
qv=[quadrat_vmr(s) for s in (128,256,512)]
for q in qv: print(f"quadrat {q['s']}px: mean {q['mean']:.3f} VMR {q['vmr']:.2f} NB_k {q['nb_k']:.2f} Fisher p {q['fisher_p']:.1e}",flush=True)
# field-to-field WITHIN sample (removes between-sample parasitaemia confound = the LoD-relevant over-dispersion)
from collections import defaultdict
samp=defaultdict(list)
for s,_,P in F: samp[s].append(len(P))
wvmr=[]; wmean=[]
for s,counts in samp.items():
    c=np.array(counts,float)
    if len(c)>=5 and c.mean()>0: wvmr.append(c.var(ddof=1)/c.mean()); wmean.append(c.mean())
wvmr=np.array(wvmr); wmean=np.array(wmean)
vmr_w=float(np.median(wvmr)); mean_w=float(np.median(wmean)); kf=mean_w/(vmr_w-1) if vmr_w>1 else np.inf
# pooled between-sample (for contrast, confounded by parasitaemia)
fc=np.array([n for v in samp.values() for n in v],float); vmr_between=float(fc.var(ddof=1)/fc.mean())
print(f"within-sample field VMR: median {vmr_w:.2f} (n={len(wvmr)} samples, med mean/field {mean_w:.1f}) NB_k {kf:.2f} | (pooled between-sample VMR {vmr_between:.1f} = parasitaemia-confounded)",flush=True)
res=dict(gr=dict(r=ctr.tolist(),g=[None if np.isnan(x) else float(x) for x in g],peak_g=gpk,peak_r=gpk_r),
         thomas=dict(sigma_px=sig,mean_cluster_size=mean_cluster,note='minimum-contrast Thomas fit UNSTABLE (mean cluster size<1) -> clustering real but too diffuse/weak for a clean cluster-process model; rely on g(r)+VMR'),
         quadrat=qv, field=dict(nsamples=int(len(wvmr)),within_vmr_median=vmr_w,mean_per_field=mean_w,nb_k=float(kf),between_vmr_confounded=vmr_between))
json.dump(res,open(OUT/'aggreg_stats.json','w'),indent=2)

# ---------- figures: g(r) + VMR-vs-scale ----------
scales=[q['s'] for q in qv]+[2560]; vmrs=[q['vmr'] for q in qv]+[vmr_w]; labs=[f'{s}px\nquadrat' for s in scales[:-1]]+['field\n(2560px)']
fig,ax=plt.subplots(1,2,figsize=(12.6,5.0))
ax[0].axhline(1,color='k',lw=0.7,ls=':')
ax[0].plot(ctr,g,color='#1f77b4',lw=2.0,label='pair correlation g(r)')
ax[0].axvline(gpk_r,color='#d62728',ls='--',lw=1.0,label=f'peak ≈ {gpk_r:.0f} px, g={gpk:.2f}')
ax[0].set_xlabel('r (px)'); ax[0].set_ylabel('g(r)   (>1 = clustered)')
ax[0].set_title('GT parasites: intra-field aggregation is significant but moderate',fontsize=10); ax[0].legend(fontsize=8)
x=np.arange(len(scales))
ax[1].axhline(1,color='k',lw=0.8,ls=':',label='Poisson (VMR=1)')
ax[1].bar(x,vmrs,0.6,color=['#d62728','#d62728','#d62728','#999999'])
ax[1].set_xticks(x); ax[1].set_xticklabels(labs,fontsize=8); ax[1].set_ylabel('variance-to-mean ratio')
ax[1].set_title('Over-dispersion is SUB-FIELD; it washes out at the field scale',fontsize=10); ax[1].legend(fontsize=8)
for xi,v in zip(x,vmrs): ax[1].text(xi,v+0.02,f'{v:.2f}',ha='center',fontsize=8)
plt.tight_layout(); plt.savefig(OUT/'aggreg_summary.png',dpi=150,bbox_inches='tight'); plt.close()
print('AGGREG_DONE',flush=True)
