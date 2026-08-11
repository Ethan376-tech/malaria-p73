# -*- coding: utf-8 -*-
"""WS2 — selective prediction / abstention with a distribution-free conformal guarantee. Turns the WS1 finding
(cross-site low-parasitaemia is undecidable at the bag level) into a clinical triage tool: predict pos/neg only
when confident, otherwise ABSTAIN (refer to a human microscopist), with a guaranteed error rate on what it predicts.

Binary split-conformal prediction sets: nonconformity s(x,y)=1-p_y; calibrate q̂ at level 1-alpha; set
C(x)={y: p_y >= 1-q̂}. |C|=1 -> confident single-class prediction; |C|!=1 -> abstain. Marginal coverage
P(y in C) >= 1-alpha (exchangeability). Evaluated IN-DOMAIN (domain A, valid) and CROSS-SITE (source B-trained
scorer on A, split-conformal on a small labelled target subset). Abstention reported BY PARASITAEMIA BAND.
Output -> outputs/experiments/ws2_selective/{results.json, fig_ws2.png}
"""
import os, sys, json, glob, random
os.environ['XFORMERS_DISABLED']='1'
sys.path.insert(0,'/root/autodl-tmp/A_Dissertation/references/RedDino/train'); sys.path.append('/root/A_Dissertation')
import numpy as np, pandas as pd, torch, timm
import torchvision.transforms as T
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image; Image.MAX_IMAGE_PIXELS=None
from common.paths import RAW, SPLITS, CHECKPOINTS, OUTPUTS
from dinov2.models.vision_transformer import vit_small
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
random.seed(0); np.random.seed(0); dev='cuda'; HALF=112; K=10
norm=T.Compose([T.ToTensor(),T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
RES=OUTPUTS/'experiments'/'ws2_selective'; RES.mkdir(parents=True,exist_ok=True)
D=OUTPUTS/'experiments'/'stageD_bag'; meta=pd.read_csv(D/'bag_meta_v2.csv')
EMB=np.load(D/'emb_thickdino_v2.npy')
y=meta.label01.values; dom=meta.dataset.values; bnd=meta.para_band.values
A=(dom=='ibadan'); B=(dom=='chittagong-1')
yA=y[A]; bandA=bnd[A]

def dv(sd,patch):
    m=vit_small(patch_size=patch,img_size=224,block_chunks=0,init_values=1.0); m.load_state_dict(sd,strict=False); return m.eval().to(dev)
thick=dv(torch.load(CHECKPOINTS/'ssl_vits_supvit_dinoadapt'/'encoder_final.pth',map_location='cpu'),16)

# ---- in-domain A posteriors: 5-fold CV logistic on max-pool bag vectors ----
bagvec=EMB.max(1)                                        # (441,384)
XA=bagvec[A]; skf=StratifiedKFold(5,shuffle=True,random_state=0); p_ind=np.zeros(len(yA))
for tr,te in skf.split(XA,yA):
    sc=StandardScaler().fit(XA[tr]); lr=LogisticRegression(max_iter=3000,class_weight='balanced').fit(sc.transform(XA[tr]),yA[tr])
    p_ind[te]=lr.predict_proba(sc.transform(XA[te]))[:,1]
print('in-domain A AUC',round(roc_auc_score(yA,p_ind),3),flush=True)

# ---- cross-site B->A posteriors: C1-box tile scorer -> top-k -> Platt on B ----
s=pd.read_csv(SPLITS/'splits.csv'); c1=s[s.dataset=='chittagong-1']
def boxes(txt):
    cs=[]
    for ln in txt.read_text(errors='ignore').splitlines()[1:]:
        f=ln.split(',')
        if len(f)>=9 and 'Parasit' in f[1]:
            try: x1,y1,x2,y2=map(float,f[5:9]); cs.append(((x1+x2)/2,(y1+y2)/2))
            except ValueError: pass
    return cs
def cropc(im,cx,cy):
    W,H=im.size; cx=int(min(max(cx,HALF),W-HALF)); cy=int(min(max(cy,HALF),H-HALF)); return im.crop((cx-HALF,cy-HALF,cx+HALF,cy+HALF))
def free(cx,cy,c): return all(abs(bx-cx)>HALF or abs(by-cy)>HALF for bx,by in c)
tiles=[]; labs=[]
for r in c1.itertuples():
    img_dir=RAW/r.rel_path; ann_dir=str(img_dir).replace('All_PvTk','All_annotations')
    imgs=sorted(glob.glob(str(img_dir)+'/*.jpg')); random.shuffle(imgs)
    for ip in imgs[:6]:
        stem=os.path.splitext(os.path.basename(ip))[0]; af=os.path.join(ann_dir,stem+'.txt')
        cents=boxes(Path(af)) if os.path.exists(af) else []
        try: im=Image.open(ip).convert('RGB')
        except Exception: continue
        W,H=im.size
        for cx,cy in cents[:20]: tiles.append(np.asarray(cropc(im,cx,cy),np.uint8)); labs.append(1)
        g=0
        for _ in range(40):
            rx,ry=random.randint(HALF,W-HALF),random.randint(HALF,H-HALF)
            if not cents or free(rx,ry,cents): tiles.append(np.asarray(cropc(im,rx,ry),np.uint8)); labs.append(0); g+=1
            if g>=max(3,len(cents[:20])): break
    if sum(labs)>=4000: break
labs=np.array(labs)
@torch.no_grad()
def enc(arr,bs=256):
    out=[]
    for i in range(0,len(arr),bs):
        b=torch.stack([norm(Image.fromarray(t)) for t in arr[i:i+bs]]).to(dev); out.append(thick(b).float().cpu().numpy())
    return np.concatenate(out)
Xt=enc(tiles); tsc=StandardScaler().fit(Xt); tlr=LogisticRegression(max_iter=3000,class_weight='balanced').fit(tsc.transform(Xt),labs)
Ptile=tlr.predict_proba(tsc.transform(EMB.reshape(-1,384)))[:,1].reshape(EMB.shape[0],256)
topk=np.sort(Ptile,1)[:,::-1][:,:K].mean(1)
plat=LogisticRegression(max_iter=3000,class_weight='balanced').fit(topk[B].reshape(-1,1),y[B])
p_xs=plat.predict_proba(topk[A].reshape(-1,1))[:,1]
print('cross-site B->A AUC',round(roc_auc_score(yA,p_xs),3),flush=True)

# ---- conformal binary selective prediction ----
def conformal(p_cal,yc,p_te,alpha):
    s=np.where(yc==1,1-p_cal,p_cal)                     # nonconformity of true class
    n=len(s); q=min(1.0,np.ceil((n+1)*(1-alpha))/n); qhat=np.quantile(s,q,method='higher'); thr=1-qhat
    inc_pos=p_te>=thr; inc_neg=(1-p_te)>=thr
    pred=np.where(inc_pos&~inc_neg,1,np.where(inc_neg&~inc_pos,0,-1))   # -1 = abstain (|set|!=1)
    return pred, inc_pos, inc_neg, thr

def eval_conformal(p, yt, bands, alpha, nsplit=200):
    rng=np.random.RandomState(0); idx=np.arange(len(yt))
    agg={'coverage':[], 'abstain':[], 'sel_err':[],
         'abstain_neg':[], 'abstain_pos_lo':[], 'abstain_pos_hi':[]}
    for _ in range(nsplit):
        rng.shuffle(idx); h=len(idx)//2; cal,te=idx[:h],idx[h:]
        pred,ip,ineg,thr=conformal(p[cal],yt[cal],p[te],alpha)
        yc_te=yt[te]
        inset=np.where(yc_te==1,ip,ineg)               # is true label in the set?
        agg['coverage'].append(float(inset.mean()))
        pmask=pred!=-1
        agg['abstain'].append(float((~pmask).mean()))
        agg['sel_err'].append(float((pred[pmask]!=yc_te[pmask]).mean()) if pmask.sum()>0 else np.nan)
        bte=bands[te]
        for bn in ['neg','pos_lo','pos_hi']:
            m=bte==bn; agg['abstain_'+bn].append(float((pred[m]==-1).mean()) if m.sum()>0 else np.nan)
    return {k:round(float(np.nanmean(v)),3) for k,v in agg.items()}

# ---- risk-coverage via confidence threshold (for the figure) ----
def risk_cov(p,yt):
    conf=np.maximum(p,1-p); pred=(p>=0.5).astype(int); order=np.argsort(-conf)
    cov=[]; risk=[]
    for f in np.linspace(0.05,1.0,40):
        k=max(1,int(f*len(yt))); sel=order[:k]; cov.append(k/len(yt)); risk.append(float((pred[sel]!=yt[sel]).mean()))
    return np.array(cov),np.array(risk)

report={'setup':dict(nA=int(A.sum()), bands={k:int(v) for k,v in pd.Series(bandA).value_counts().items()},
                     in_domain_AUC=round(float(roc_auc_score(yA,p_ind)),3), cross_site_AUC=round(float(roc_auc_score(yA,p_xs)),3))}
for setting,p in [('in_domain',p_ind),('cross_site',p_xs)]:
    report[setting]={}
    for a in [0.05,0.10,0.20]:
        report[setting][f'alpha={a}']=eval_conformal(p,yA,bandA,a)
    print(setting,json.dumps(report[setting],indent=2),flush=True)
json.dump(report,open(RES/'results.json','w'),indent=2)

# ---- figure: risk-coverage + by-band abstention ----
fig,ax=plt.subplots(1,2,figsize=(11,4.3)); plt.rcParams.update({'font.family':'serif'})
for setting,p,c in [('in-domain',p_ind,'#228833'),('cross-site B→A',p_xs,'#ee6677')]:
    cov,risk=risk_cov(p,yA); ax[0].plot(cov,risk,'-o',ms=3,color=c,label=setting)
ax[0].set_xlabel('coverage (fraction auto-predicted)'); ax[0].set_ylabel('selective error (on covered)')
ax[0].set_title('Risk–coverage: abstain on the least confident'); ax[0].legend(); ax[0].grid(alpha=0.3)
bands=['neg','pos_lo','pos_hi']; xb=np.arange(3); w=0.35
for i,(setting,p,c) in enumerate([('in-domain',p_ind,'#228833'),('cross-site',p_xs,'#ee6677')]):
    r=eval_conformal(p,yA,bandA,0.10); vals=[r['abstain_'+b] for b in bands]
    ax[1].bar(xb+(i-0.5)*w,vals,w,color=c,label=setting)
ax[1].set_xticks(xb); ax[1].set_xticklabels(['neg','low-para\n(pos_lo)','high-para\n(pos_hi)'])
ax[1].set_ylabel('abstention rate'); ax[1].set_ylim(0,1.05)
ax[1].set_title('Abstention by parasitaemia band (90% guarantee, α=0.1)'); ax[1].legend(); ax[1].grid(axis='y',alpha=0.3)
plt.tight_layout(); plt.savefig(RES/'fig_ws2.png',dpi=160,bbox_inches='tight')
print('WROTE_DONE',RES/'results.json',flush=True)
