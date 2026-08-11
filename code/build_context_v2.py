# -*- coding: utf-8 -*-
"""Cheap probe: does context-aware aggregation help malaria MIL, or dilute sparse positives?
Compares, on the SAME cached bag embeddings (256 tiles x 384) and SAME splits:
  max-pool+LR | ABMIL (per-instance attention, no context) | TransMIL (inter-instance self-attention = context)
Reports in-domain (Ibadan repeated CV) and cross-site (B->A) AUC, each overall AND on the low-parasitaemia stratum.
Output -> outputs/experiments/context_probe/"""
import sys, numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
sys.path.append('/root/A_Dissertation'); from common.paths import OUTPUTS
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
dev='cuda' if torch.cuda.is_available() else 'cpu'
S=OUTPUTS/'experiments'/'stageD_bag'; RES=OUTPUTS/'experiments'/'context_probe'; RES.mkdir(parents=True,exist_ok=True)
bm=pd.read_csv(S/'bag_meta_v2.csv')
ENC=['supvit','reddino','thickdino']
NAME={'supvit':'SupViT','reddino':'RedDino','thickdino':'ThickDINO'}

ib = (bm.dataset=='ibadan').values
B  = (bm.dataset!='ibadan').values   # chittagong-1 = domain B
y  = bm.label01.values.astype(np.float32)
lowmask = ((bm.label01==0) | (bm.para_band=='pos_lo')).values   # neg + low-para positives

def torch_seed(s): torch.manual_seed(s); np.random.seed(s)

class ABMIL(nn.Module):
    def __init__(self,d=384,dh=128,drop=0.3):
        super().__init__()
        self.proj=nn.Sequential(nn.Linear(d,dh),nn.ReLU(),nn.Dropout(drop))
        self.V=nn.Linear(dh,dh); self.U=nn.Linear(dh,dh); self.w=nn.Linear(dh,1); self.head=nn.Linear(dh,1)
    def forward(self,x):
        h=self.proj(x)                                   # (B,N,dh)
        a=self.w(torch.tanh(self.V(h))*torch.sigmoid(self.U(h)))  # gated attention (per-instance)
        a=torch.softmax(a,dim=1)
        z=(a*h).sum(1)
        return self.head(z).squeeze(-1)

class TransMIL(nn.Module):
    def __init__(self,d=384,dh=128,heads=4,layers=1,drop=0.3):
        super().__init__()
        self.proj=nn.Sequential(nn.Linear(d,dh),nn.ReLU(),nn.Dropout(drop))
        self.cls=nn.Parameter(torch.randn(1,1,dh)*0.02)
        el=nn.TransformerEncoderLayer(dh,heads,dim_feedforward=dh*2,dropout=drop,batch_first=True,activation='gelu')
        self.enc=nn.TransformerEncoder(el,layers); self.head=nn.Linear(dh,1)
    def forward(self,x):
        h=self.proj(x)
        cls=self.cls.expand(x.size(0),-1,-1)
        h=self.enc(torch.cat([cls,h],1))
        return self.head(h[:,0]).squeeze(-1)             # CLS attends to all tiles -> context

def train_eval(model_cls, Xtr,ytr, Xte, n_sub=64, epochs=60, lr=1e-3, wd=1e-2, seed=0):
    torch_seed(seed); m=model_cls().to(dev)
    opt=torch.optim.AdamW(m.parameters(),lr=lr,weight_decay=wd)
    Xtr_t=torch.tensor(Xtr,dtype=torch.float32); ytr_t=torch.tensor(ytr,dtype=torch.float32).to(dev)
    N=Xtr.shape[1]
    for ep in range(epochs):
        m.train(); perm=torch.randperm(len(Xtr))
        # tile subsample per epoch (bag augmentation)
        idx=np.random.choice(N, min(n_sub,N), replace=False)
        xb=Xtr_t[perm][:,idx,:].to(dev); yb=ytr_t[perm]
        opt.zero_grad(); loss=F.binary_cross_entropy_with_logits(m(xb),yb); loss.backward(); opt.step()
    m.eval()
    with torch.no_grad():
        out=torch.sigmoid(m(torch.tensor(Xte,dtype=torch.float32).to(dev))).cpu().numpy()
    return out

def maxpool_lr(Xtr,ytr,Xte):
    sc=StandardScaler().fit(Xtr.max(1)); lr=LogisticRegression(max_iter=2000).fit(sc.transform(Xtr.max(1)),ytr)
    return lr.predict_proba(sc.transform(Xte.max(1)))[:,1]

rows=[]
for e in ENC:
    X=np.load(S/f'emb_{e}_v2.npy')              # (409,256,384)
    Xi, yi, lowi = X[ib], y[ib], lowmask[ib]
    XB, yB        = X[B], y[B]
    # ---- in-domain Ibadan: repeated 3-fold OOF ----
    res={'max':{'all':[],'low':[]}, 'ABMIL':{'all':[],'low':[]}, 'TransMIL':{'all':[],'low':[]}}
    for rep in range(5):
        skf=StratifiedKFold(5 if False else 3, shuffle=True, random_state=rep)
        oof={k:np.zeros(len(yi)) for k in res}
        for tr,te in skf.split(Xi,yi):
            oof['max'][te]=maxpool_lr(Xi[tr],yi[tr],Xi[te])
            oof['ABMIL'][te]=train_eval(ABMIL,Xi[tr],yi[tr],Xi[te],seed=rep)
            oof['TransMIL'][te]=train_eval(TransMIL,Xi[tr],yi[tr],Xi[te],seed=rep)
        for k in res:
            res[k]['all'].append(roc_auc_score(yi,oof[k]))
            res[k]['low'].append(roc_auc_score(yi[lowi],oof[k][lowi]))
    # ---- cross-site B->A: train all B, test all Ibadan, 5 seeds ----
    xs={'max':[], 'ABMIL':[], 'TransMIL':[]}; xslow={'max':[], 'ABMIL':[], 'TransMIL':[]}
    for sd in range(5):
        pm=maxpool_lr(XB,yB,Xi); pa=train_eval(ABMIL,XB,yB,Xi,seed=sd); pt=train_eval(TransMIL,XB,yB,Xi,seed=sd)
        for k,p in [('max',pm),('ABMIL',pa),('TransMIL',pt)]:
            xs[k].append(roc_auc_score(yi,p)); xslow[k].append(roc_auc_score(yi[lowi],p[lowi]))
    for k in res:
        rows.append(dict(encoder=NAME[e], method=k,
            indom=round(np.mean(res[k]['all']),3), indom_sd=round(np.std(res[k]['all']),3),
            indom_low=round(np.mean(res[k]['low']),3),
            xsite=round(np.mean(xs[k]),3), xsite_sd=round(np.std(xs[k]),3),
            xsite_low=round(np.mean(xslow[k]),3), xsite_low_sd=round(np.std(xslow[k]),3)))
        print(rows[-1], flush=True)

df=pd.DataFrame(rows); df.to_csv(RES/'context_probe_v2.csv',index=False)
print('\n', df.to_string(index=False))
# verdict
for e in [NAME[x] for x in ENC]:
    d=df[df.encoder==e].set_index('method')
    print(f'\n{e}: TransMIL vs ABMIL vs max  | xsite {d.loc["TransMIL","xsite"]}/{d.loc["ABMIL","xsite"]}/{d.loc["max","xsite"]}'
          f' | xsite_low {d.loc["TransMIL","xsite_low"]}/{d.loc["ABMIL","xsite_low"]}/{d.loc["max","xsite_low"]}')
print('\nWROTE ->',RES, flush=True)
