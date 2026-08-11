# -*- coding: utf-8 -*-
"""B2 v2 (rigorous): end-to-end MIL fine-tuning with validation-based early stopping, an LR sweep,
multiple seeds, cosine schedule, backbone-drift logging, and a PROTOCOL-MATCHED frozen baseline.
Encoders: SupViT/RedDino/ThickDINO inits. Sample-level weak labels only (no boxes).
Output -> outputs/experiments/b2_finetune_v2/ (results.csv, val_curves.json)."""
import os, sys, json, time, copy
os.environ['XFORMERS_DISABLED']='1'
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F, timm
sys.path.insert(0,'/root/autodl-tmp/A_Dissertation/references/RedDino/train'); sys.path.append('/root/A_Dissertation')
from common.paths import OUTPUTS, CHECKPOINTS
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from dinov2.models.vision_transformer import vit_small
dev='cuda'
SMOKE=os.environ.get('SMOKE','0')=='1'
CACHE=OUTPUTS/'experiments'/'b2_tilecache_v2'; RES=OUTPUTS/'experiments'/'b2_finetune_v2b'; RES.mkdir(parents=True,exist_ok=True)
man=pd.read_csv(CACHE/'manifest_v2.csv'); M=man.set_index('sample_id')
MEAN=torch.tensor([0.485,0.456,0.406]).view(1,3,1,1).to(dev); STD=torch.tensor([0.229,0.224,0.225]).view(1,3,1,1).to(dev)
K_TR=32; K_TE=64; BBAG=2; MAXEP=3 if SMOKE else 30; PATIENCE=5; LRS=[5e-5] if SMOKE else [1e-5,5e-5]; SEEDS=[0] if SMOKE else [0,1]

def make_encoder(init):
    if init=='SupViT': return timm.create_model('vit_small_patch16_224.augreg_in21k_ft_in1k',pretrained=True,num_classes=0)
    sd=(timm.create_model('hf_hub:Snarcy/RedDino-small',pretrained=True).state_dict() if init=='RedDino'
        else torch.load(CHECKPOINTS/'ssl_vits_supvit_dinoadapt'/'encoder_final.pth',map_location='cpu'))
    m=vit_small(patch_size=14 if init=='RedDino' else 16,img_size=224,block_chunks=0,init_values=1.0,drop_path_rate=0.1)
    m.load_state_dict(sd,strict=False); return m
class MILNet(nn.Module):
    def __init__(self,enc,d=384,drop=0.2):
        super().__init__(); self.enc=enc; self.head=nn.Sequential(nn.Dropout(drop),nn.Linear(d,1))
    def forward(self,x):
        B,K=x.shape[:2]; h=self.enc(x.flatten(0,1)).view(B,K,-1).amax(1); return self.head(h).squeeze(-1)
def load_bag(sid,k,train):
    a=np.load(CACHE/f'{sid}.npy'); idx=np.random.choice(len(a),k,replace=len(a)<k) if train else np.arange(min(k,len(a))); return a[idx]
def to_dev(arrs):
    x=torch.from_numpy(np.stack(arrs)).to(dev).float().div(255).permute(0,1,4,2,3); return (x-MEAN.unsqueeze(1))/STD.unsqueeze(1)
def labels(ids): return M.loc[ids,'label01'].values.astype(float)
def lowpara(ids): return [s for s in ids if (M.loc[s,'label01']==0 or M.loc[s,'para_band']=='pos_lo')]

@torch.no_grad()
def predict(net,ids):
    net.eval(); out={}
    for s in ids:
        x=to_dev([load_bag(s,K_TE,False)])
        with torch.autocast('cuda',dtype=torch.bfloat16): out[s]=torch.sigmoid(net(x)).item()
    return out
def auc(ids,sc):
    y=labels(ids); p=np.array([sc[s] for s in ids]); return roc_auc_score(y,p) if len(set(y))>1 else float('nan')

def param_vec(enc): return torch.cat([p.detach().flatten() for p in enc.parameters()])
def train_ft(init,tr,va,te,lr_bb,seed):
    torch.manual_seed(seed); np.random.seed(seed)
    net=MILNet(make_encoder(init)).to(dev); th0=param_vec(net.enc); n0=th0.norm().item()
    opt=torch.optim.AdamW([{'params':net.enc.parameters(),'lr':lr_bb},{'params':net.head.parameters(),'lr':1e-3}],weight_decay=0.05)
    sched=torch.optim.lr_scheduler.OneCycleLR(opt,max_lr=[lr_bb,1e-3],total_steps=MAXEP*max(1,len(tr)//BBAG),pct_start=0.1)
    ytr=labels(tr); pw=torch.tensor([(ytr==0).sum()/max(1,(ytr==1).sum())],device=dev)
    best=-1; best_state=None; bad=0; curve=[]
    for ep in range(MAXEP):
        net.train(); order=np.random.permutation(len(tr))
        for i in range(0,len(tr),BBAG):
            ids=[tr[j] for j in order[i:i+BBAG]]
            x=to_dev([load_bag(s,K_TR,True) for s in ids]); y=torch.tensor(labels(ids),dtype=torch.float32,device=dev)
            with torch.autocast('cuda',dtype=torch.bfloat16): loss=F.binary_cross_entropy_with_logits(net(x),y,pos_weight=pw)
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(),3.0); opt.step(); sched.step()
        va_auc=auc(va,predict(net,va)); curve.append(round(va_auc,3))
        if va_auc>best: best=va_auc; best_state=copy.deepcopy(net.state_dict()); bad=0
        else:
            bad+=1
            if bad>=PATIENCE: break
    net.load_state_dict(best_state)
    drift=((param_vec(net.enc)-th0).norm().item())/n0
    return predict(net,te), best, curve, round(drift,4), net

def frozen_feats(init,ids):
    enc=make_encoder(init).eval().to(dev); F_=[]
    with torch.no_grad():
        for s in ids:
            x=to_dev([load_bag(s,K_TE,False)])[0]
            with torch.autocast('cuda',dtype=torch.bfloat16): h=enc(x)
            F_.append(h.amax(0).float().cpu().numpy())
    return np.stack(F_)

ib=man[man.dataset=='ibadan']; B=man[man.dataset!='ibadan']
inits=['ThickDINO'] if SMOKE else ['SupViT','RedDino','ThickDINO']
rows=[]; curves={}
def split_trainval(ids,seed,frac=0.2):
    rng=np.random.RandomState(seed); ids=list(ids); rng.shuffle(ids); n=int(len(ids)*frac); return ids[n:], ids[:n]

for init in inits:
    t0=time.time()
    # ---- protocol-matched FROZEN baseline (max-pool + logistic, same e1 folds) ----
    foldlist=[0] if SMOKE else sorted(ib.e1_fold.unique())
    Xib=frozen_feats(init, ib.sample_id.tolist()); idmap={s:i for i,s in enumerate(ib.sample_id)}
    oof_fr=np.zeros(len(ib))
    for f in foldlist:
        tr=ib[ib.e1_fold!=f].sample_id.tolist(); te=ib[ib.e1_fold==f].sample_id.tolist()
        sc=StandardScaler().fit(Xib[[idmap[s] for s in tr]]); lr=LogisticRegression(max_iter=2000).fit(sc.transform(Xib[[idmap[s] for s in tr]]),labels(tr))
        oof_fr[[idmap[s] for s in te]]=lr.predict_proba(sc.transform(Xib[[idmap[s] for s in te]]))[:,1]
    fr_sc={s:oof_fr[idmap[s]] for s in ib.sample_id}
    rows.append(dict(init=init,regime='frozen',setting='in-domain Ibadan',auc=round(auc(list(fr_sc),fr_sc),3),
                     low_auc=round(auc(lowpara(list(fr_sc)),fr_sc),3)))
    print(rows[-1],flush=True)
    # ---- LR sweep (fold0, seed0) ----
    f0=foldlist[0]; tv=ib[ib.e1_fold!=f0].sample_id.tolist(); te0=ib[ib.e1_fold==f0].sample_id.tolist()
    tr0,va0=split_trainval(tv,0)
    best_lr,best_vl=LRS[0],-1
    for lr in LRS:
        _,vl,cv,dr,_=train_ft(init,tr0,va0,te0,lr,0); curves[f'{init}_lr{lr}']=cv
        print(f'  [{init}] lr {lr}: val {vl:.3f} drift {dr}',flush=True)
        if vl>best_vl: best_vl,best_lr=vl,lr
    print(f'  [{init}] selected lr={best_lr}',flush=True)
    if SMOKE:
        rows.append(dict(init=init,regime='FT',setting='in-domain Ibadan(smoke)',auc=round(vl,3),lr=best_lr)); break
    # ---- FT main: chosen lr x 3 folds x 3 seeds (OOF per seed) ----
    seed_aucs=[]; seed_low=[]; drifts=[]
    for sd in SEEDS:
        oof={}
        for f in foldlist:
            tv=ib[ib.e1_fold!=f].sample_id.tolist(); te=ib[ib.e1_fold==f].sample_id.tolist()
            tr,va=split_trainval(tv,sd); scs,_,cv,dr,_=train_ft(init,tr,va,te,best_lr,sd); oof.update(scs); drifts.append(dr)
        seed_aucs.append(auc(list(oof),oof)); seed_low.append(auc(lowpara(list(oof)),oof))
        print(f'  [{init}] FT seed {sd} in-domain {seed_aucs[-1]:.3f}',flush=True)
    rows.append(dict(init=init,regime='FT',setting='in-domain Ibadan',auc=round(np.mean(seed_aucs),3),
                     auc_sd=round(np.std(seed_aucs),3),low_auc=round(np.mean(seed_low),3),
                     lr=best_lr,backbone_drift=round(np.mean(drifts),4)))
    print(rows[-1],flush=True)
    # ---- cross-site B->A (train B w/ val, test A), chosen lr, 3 seeds ----
    xs=[]; xl=[]
    teA=ib.sample_id.tolist()
    for sd in SEEDS:
        trB,vaB=split_trainval(B.sample_id.tolist(),sd); scs,_,_,_,net=train_ft(init,trB,vaB,teA,best_lr,sd)
        xs.append(auc(teA,scs)); xl.append(auc(lowpara(teA),scs))
        if init=='ThickDINO' and sd==0:
            out=CHECKPOINTS/'ssl_vits_thickdino_ft_v2b'; out.mkdir(parents=True,exist_ok=True); torch.save(net.enc.state_dict(),out/'encoder_final.pth')
    rows.append(dict(init=init,regime='FT',setting='cross-site B->A',auc=round(np.mean(xs),3),auc_sd=round(np.std(xs),3),
                     low_auc=round(np.mean(xl),3),lr=best_lr))
    print(rows[-1],'| %.1f min'%((time.time()-t0)/60),flush=True)

pd.DataFrame(rows).to_csv(RES/('smoke.csv' if SMOKE else 'results.csv'),index=False)
json.dump(curves,open(RES/'val_curves.json','w'),indent=2)
print('\n',pd.DataFrame(rows).to_string(index=False)); print('\nWROTE_DONE ->',RES,flush=True)
