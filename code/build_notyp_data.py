# -*- coding: utf-8 -*-
"""Ablation: filter WITHOUT the typicality gate. keep(b) = pv>=bMC0 OR tta>=bTTA0 OR pm>=aMC (protect clause only,
no maha<=tau). Isolates the typicality gate's contribution vs the uncertainty-protection. -> ds_sweep_notyp/"""
import os, sys, json, glob, shutil
import numpy as np
sys.path.append('/root/A_Dissertation'); from common.paths import OUTPUTS
YO=OUTPUTS/'experiments'/'s5_yolo'; RES=OUTPUTS/'experiments'/'s6_retinanet'; S3=OUTPUTS/'experiments'/'s3_filter'
sig=json.load(open(RES/'sweep_signals.json'))
pb=np.load(S3/'perbox_multi2.npz'); dev=pb['field_id'].astype(int)<95; tp=pb['is_tp']==1
faint=dev&tp&(pb['p_mean']<=np.quantile(pb['p_mean'][dev&tp],1/3))
bMC0=float(np.quantile(pb['p_var'][faint],0.10)); bTTA0=float(np.quantile(pb['tta_var'][faint],0.10)); aMC=float(np.quantile(pb['p_mean'][dev],0.5))
def keep(b): _,pm,pv,tta,mh=b; return (pv>=bMC0 or tta>=bTTA0 or pm>=aMC)   # NO typicality gate
DST=YO/'ds_sweep_notyp'
if DST.exists(): shutil.rmtree(DST)
shutil.copytree(YO/'ds_a0', DST); nb=0
for c in sig:
    lp=DST/'labels'/'train'/(c[:-4]+'.txt')
    if not lp.exists(): continue
    kept=[b[0] for b in sig[c] if keep(b)]; lp.write_text("\n".join(kept)); nb+=len(kept)
(YO/'data_sweep_notyp.yaml').write_text(f"path: {DST}\ntrain: images/train\nval: images/val\nnc: 1\nnames: ['parasite']\n")
print(f'floors: bMC0 {bMC0:.5f} bTTA0 {bTTA0:.5f} aMC {aMC:.4f}')
print(f'ds_sweep_notyp (protect clause only, NO typicality gate): {nb} boxes  [cf A0 20544, t99 15747]')
print('NOTYP_DATA_DONE',flush=True)
