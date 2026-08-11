# -*- coding: utf-8 -*-
"""Generate detector-oriented filtered label sets at several aggressiveness levels (CPU), from sweep_signals.json.
Filter: keep(b) = (pv>=bMC0 OR tta>=bTTA0 OR pm>=aMC)  AND  maha<=tau  — LOOSE faint-protecting floors (recall-first),
typicality gate tau swept from precision-optimal (aggressive) to off (=A0). Writes ds_sweep_t{tag}/ + reports box counts."""
import os, sys, json, glob, shutil
import numpy as np
sys.path.append('/root/A_Dissertation'); from common.paths import OUTPUTS
YO=OUTPUTS/'experiments'/'s5_yolo'; RES=OUTPUTS/'experiments'/'s6_retinanet'; S3=OUTPUTS/'experiments'/'s3_filter'
sig=json.load(open(RES/'sweep_signals.json'))
# loose faint-protecting floors from dev faint-TP (10th pct = retain >=90% faint via each clause)
pb=np.load(S3/'perbox_multi2.npz'); dev=pb['field_id'].astype(int)<95; tp=pb['is_tp']==1
faint=dev&tp&(pb['p_mean']<=np.quantile(pb['p_mean'][dev&tp],1/3))
bMC0=float(np.quantile(pb['p_var'][faint],0.10)); bTTA0=float(np.quantile(pb['tta_var'][faint],0.10))
aMC=float(np.quantile(pb['p_mean'][dev],0.5))                       # moderate confidence floor
dtp=dev&tp; devmaha=None
# typicality thresholds: use A0-box maha percentiles so tau directly controls keep fraction
allmaha=np.array([b[4] for c in sig for b in sig[c]])
TAUS=[('t99',float(np.quantile(allmaha,0.99))),('t95',float(np.quantile(allmaha,0.95))),
      ('t90',float(np.quantile(allmaha,0.90))),('t80',float(np.quantile(allmaha,0.80))),
      ('t65',float(np.quantile(allmaha,0.65)))]
print(f'floors: bMC0 {bMC0:.5f} bTTA0 {bTTA0:.5f} aMC {aMC:.4f} | A0 boxes {len(allmaha)}',flush=True)
print('tau candidates (A0-maha pct):',[(t,round(v,1)) for t,v in TAUS],flush=True)
def keep(b,tau):
    _,pm,pv,tta,mh=b
    return (pv>=bMC0 or tta>=bTTA0 or pm>=aMC) and mh<=tau
for tag,tau in TAUS:
    DST=YO/f'ds_sweep_{tag}'
    if DST.exists(): shutil.rmtree(DST)
    shutil.copytree(YO/'ds_a0', DST)                                # copy positives+val, then overwrite train labels
    nb=0
    for c in sig:
        lp=DST/'labels'/'train'/(c[:-4]+'.txt')
        if not lp.exists(): continue
        kept=[b[0] for b in sig[c] if keep(b,tau)]
        lp.write_text("\n".join(kept)); nb+=len(kept)
    (YO/f'data_sweep_{tag}.yaml').write_text(f"path: {DST}\ntrain: images/train\nval: images/val\nnc: 1\nnames: ['parasite']\n")
    print(f'  {tag} (tau {tau:.1f}): {nb} boxes kept',flush=True)
print('SWEEP_DATASETS_DONE',flush=True)
