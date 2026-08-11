# -*- coding: utf-8 -*-
"""Robust frozen baseline for Table 3.5 (v2): the B2 frozen baseline was a single 3-fold OOF (64-tile) and
noisy (RedDino lucky 0.686, ThickDINO unlucky 0.569). Here we encode the 64-tile v2 cache once (frozen, max-pool)
and run repeated 5x3 CV on Ibadan -> stable frozen 64-tile AUC, comparable to the 2-seed FT numbers.
Output -> outputs/experiments/b2_finetune_v2b/frozen_robust_v2.csv"""
import os, sys
os.environ['XFORMERS_DISABLED']='1'
import numpy as np, pandas as pd, torch, timm
sys.path.insert(0,'/root/autodl-tmp/A_Dissertation/references/RedDino/train'); sys.path.append('/root/A_Dissertation')
from common.paths import OUTPUTS, CHECKPOINTS
from dinov2.models.vision_transformer import vit_small
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
dev='cuda'
CACHE=OUTPUTS/'experiments'/'b2_tilecache_v2'; RES=OUTPUTS/'experiments'/'b2_finetune_v2b'
man=pd.read_csv(CACHE/'manifest_v2.csv'); ib=man[man.dataset=='ibadan'].reset_index(drop=True)
y=ib.label01.values; low=((ib.label01==0)|(ib.para_band=='pos_lo')).values
MEAN=torch.tensor([0.485,0.456,0.406]).view(1,3,1,1).to(dev); STD=torch.tensor([0.229,0.224,0.225]).view(1,3,1,1).to(dev)
def make_encoder(init):
    if init=='SupViT': return timm.create_model('vit_small_patch16_224.augreg_in21k_ft_in1k',pretrained=True,num_classes=0).eval().to(dev)
    sd=(timm.create_model('hf_hub:Snarcy/RedDino-small',pretrained=True).state_dict() if init=='RedDino'
        else torch.load(CHECKPOINTS/'ssl_vits_supvit_dinoadapt'/'encoder_final.pth',map_location='cpu'))
    m=vit_small(patch_size=14 if init=='RedDino' else 16,img_size=224,block_chunks=0,init_values=1.0); m.load_state_dict(sd,strict=False); return m.eval().to(dev)
@torch.no_grad()
def feats(init):
    enc=make_encoder(init); out=[]
    for s in ib.sample_id:
        a=np.load(CACHE/f'{s}.npy')                       # (64,224,224,3) uint8
        x=torch.from_numpy(a).to(dev).float().div(255).permute(0,3,1,2); x=(x-MEAN)/STD
        with torch.autocast('cuda',dtype=torch.bfloat16): h=enc(x)
        out.append(h.amax(0).float().cpu().numpy())
    return np.stack(out)
rk=RepeatedStratifiedKFold(n_splits=3,n_repeats=5,random_state=0)
rows=[]
for init in ['SupViT','RedDino','ThickDINO']:
    X=feats(init); aucs=[]; laucs=[]
    for tr,te in rk.split(X,y):
        sc=StandardScaler().fit(X[tr]); lr=LogisticRegression(max_iter=3000).fit(sc.transform(X[tr]),y[tr])
        p=lr.predict_proba(sc.transform(X[te]))[:,1]; aucs.append(roc_auc_score(y[te],p))
        lm=low[te]
        if len(set(y[te][lm]))>1: laucs.append(roc_auc_score(y[te][lm],p[lm]))
    rows.append(dict(init=init,frozen_auc=round(np.mean(aucs),3),frozen_sd=round(np.std(aucs),3),frozen_low=round(np.mean(laucs),3)))
    print(rows[-1],flush=True)
pd.DataFrame(rows).to_csv(RES/'frozen_robust_v2.csv',index=False)
print('FROZEN_ROBUST_DONE',flush=True)
