# -*- coding: utf-8 -*-
"""DEV (local, NOT report): formal Ripley's L test of the grouped-pseudo-box conjecture, over ALL positive fields.
Two point patterns per FASTMAL field:
  (1) GT parasite centroids     -> null = CSR (uniform n points in the field window)
  (2) gated-tile centroids      -> null = choose n_gated of the field's tile centres at random (label permutation;
                                    respects the 224-grid, tests clustering BEYOND random tile selection)
Estimator: Ripley's K with TRANSLATION edge correction on the rectangular window; L(r)=sqrt(K/pi); signal L(r)-r.
Population test: mean of L(r)-r across fields; matched null (one simulated pattern per field per iteration, averaged);
global MAD envelope test (Myllymaki 2017 style) -> single simultaneous p-value + envelope; plus pointwise envelope.
Only forward LOGITS are needed for the gate (no Grad-CAM). Output -> outputs/experiments/cluster_probe/ripley_*."""
import os, sys, json, time
os.environ['XFORMERS_DISABLED']='1'; os.environ['PYTORCH_CUDA_ALLOC_CONF']='expandable_segments:True'
import numpy as np, torch, torch.nn as nn
import torchvision.transforms as T
from PIL import Image; Image.MAX_IMAGE_PIXELS=None
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
sys.path.insert(0,'/root/autodl-tmp/A_Dissertation/references/RedDino/train'); sys.path.append('/root/A_Dissertation')
from common.paths import RAW, CHECKPOINTS, OUTPUTS
from dinov2.models.vision_transformer import vit_small
dev='cuda'; TILE=224
np.random.seed(0); torch.manual_seed(0)
FIN=json.load(open(OUTPUTS/'experiments'/'s3_filter'/'final.json'))['calib']; gate_thr=FIN['gate_thr']
OUT=OUTPUTS/'experiments'/'cluster_probe'; OUT.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.family':'serif'})
norm=T.Compose([T.ToTensor(),T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
def mk(ckpt):
    m=vit_small(patch_size=16,img_size=224,block_chunks=0,init_values=1.0,drop_path_rate=0.0)
    m.load_state_dict(torch.load(ckpt,map_location='cpu'),strict=False); return m.to(dev).eval()
enc=mk(CHECKPOINTS/'ssl_vits_thickdino_ft_ibadan'/'encoder.pth')
head=nn.Sequential(nn.Dropout(0.2),nn.Linear(384,1)).to(dev)
head.load_state_dict(torch.load(CHECKPOINTS/'ssl_vits_thickdino_ft_ibadan'/'head.pth',map_location='cpu')); head.eval()
@torch.no_grad()
def logits(crops,bs=64):
    out=[]
    for i in range(0,len(crops),bs):
        xb=torch.stack([norm(Image.fromarray(c)) for c in crops[i:i+bs]]).to(dev)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            ff=enc.forward_features(xb); s=head(ff['x_norm_clstoken']).squeeze(-1)
        out.append(s.float().cpu().numpy())
    return np.concatenate(out)
def fastmal_fields():
    fm=RAW/'ibadan'/'FASTMAL_THICK_FILMS'; o=[]
    for samp in sorted(p for p in fm.iterdir() if p.is_dir()):
        js=list(samp.glob('*.json'))
        if not js: continue
        for fld in json.loads(js[0].read_text())['rois']:
            ip=samp/fld['image_name']
            if not ip.exists(): continue
            b=[(float(r['x']),float(r['y']),float(r['width']),float(r['height'])) for r in fld['roi'] if 'PARASITE' in r.get('type','') and 'CROWD' not in r.get('type','')]
            if b: o.append((ip,b))
    return o
def toff(im):
    W,H=im.size; arr=np.asarray(im)
    xs=sorted(set(list(range(0,max(1,W-TILE+1),TILE))+[max(0,W-TILE)]))
    ys=sorted(set(list(range(0,max(1,H-TILE+1),TILE))+[max(0,H-TILE)]))
    return [(arr[ty:ty+TILE,tx:tx+TILE],(tx,ty)) for ty in ys for tx in xs],arr,W,H

# ---------- extract point patterns (encoder forward for gate) ----------
RMAX=1000.0; RG=np.linspace(0,RMAX,41); RG[0]=1e-6
def Kfun(P,W,H):
    n=len(P)
    if n<2: return np.full(len(RG),np.nan)
    dx=np.abs(P[:,0:1]-P[:,0:1].T); dy=np.abs(P[:,1:2]-P[:,1:2].T); d=np.sqrt(dx*dx+dy*dy)
    gamma=(W-dx)*(H-dy); iu=np.triu_indices(n,1)
    dd=d[iu]; gg=gamma[iu]; w=(W*H)**2/(n*(n-1)*np.clip(gg,1e-6,None))*2.0   # *2: triu counts each pair once
    order=np.argsort(dd); dd=dd[order]; w=w[order]; cw=np.cumsum(w)
    idx=np.searchsorted(dd,RG,side='right'); K=np.where(idx>0, cw[np.clip(idx-1,0,len(cw)-1)], 0.0)
    return K
def Lminusr(P,W,H):
    K=Kfun(P,W,H); return np.sqrt(np.clip(K,0,None)/np.pi)-RG

fields=fastmal_fields()
print(f'{len(fields)} positive fields',flush=True)
FD=[]  # per field: dict(gt, gated, centers, W,H)
t0=time.time()
for k,(ip,gtb) in enumerate(fields):
    im=Image.open(ip).convert('RGB'); tiles,arr,W,H=toff(im)
    crops=[c for c,_ in tiles]; offs=np.array([o for _,o in tiles],float)
    lg=logits(crops); gated=lg>=gate_thr
    centers=offs+TILE/2.0
    gt=np.array([(x+w/2,y+h/2) for (x,y,w,h) in gtb],float)
    FD.append(dict(name=ip.name,W=W,H=H,gt=gt,gated=centers[gated],centers=centers,n_gt=len(gt),n_gated=int(gated.sum()),n_tiles=len(tiles)))
    if (k+1)%20==0: print(f'  {k+1}/{len(fields)} fields ({time.time()-t0:.0f}s)',flush=True)
np.save(OUT/'ripley_patterns.npy', np.array(FD,dtype=object), allow_pickle=True)
print('patterns extracted',flush=True)

# ---------- Ripley L population test ----------
M=999; rng=np.random.default_rng(0)
def population_test(kind, minpts):
    obs=[]; simstack=[]  # per included field: observed curve; and M null curves
    inc=[]
    for f in FD:
        if kind=='gt':
            P=f['gt']; n=len(P); W,H=f['W'],f['H']
            if n<minpts: continue
            oc=Lminusr(P,W,H)
            sims=np.empty((M,len(RG)))
            for m in range(M):
                Q=np.column_stack([rng.random(n)*W, rng.random(n)*H]); sims[m]=Lminusr(Q,W,H)
        else:  # gated tiles: permute which tiles are gated
            C=f['centers']; n=f['n_gated']; W,H=f['W'],f['H']; Nt=len(C)
            if n<minpts or n>Nt-1: continue
            oc=Lminusr(f['gated'],W,H)
            sims=np.empty((M,len(RG)))
            for m in range(M):
                sel=rng.choice(Nt,n,replace=False); sims[m]=Lminusr(C[sel],W,H)
        obs.append(oc); simstack.append(sims); inc.append(f['name'])
    obs=np.array(obs); simstack=np.array(simstack)   # (F,R) and (F,M,R)
    O=np.nanmean(obs,0)                               # population observed mean curve
    S=np.nanmean(simstack,0)                          # (M,R) population null curves
    mbar=S.mean(0)
    u_sim=np.nanmax(np.abs(S-mbar),1); u_obs=np.nanmax(np.abs(O-mbar))
    p=(1+np.sum(u_sim>=u_obs))/(M+1)
    c95=np.quantile(u_sim,0.95)
    lo_pt=np.quantile(S,0.025,0); hi_pt=np.quantile(S,0.975,0)
    return dict(kind=kind,nfields=len(obs),O=O.tolist(),mbar=mbar.tolist(),glob_lo=(mbar-c95).tolist(),
                glob_hi=(mbar+c95).tolist(),pt_lo=lo_pt.tolist(),pt_hi=hi_pt.tolist(),p_global=float(p),RG=RG.tolist())
res={}
for kind,mp in [('gt',5),('gated',5)]:
    res[kind]=population_test(kind,mp); print(f"{kind}: nfields={res[kind]['nfields']} global MAD p={res[kind]['p_global']:.4f}",flush=True)
# stratify GT by density
def strat(kind,minpts,lo,hi,label):
    global FD
    keep=[f for f in FD if lo<=f['n_gt']<hi]
    save=FD; FD=keep; r=population_test(kind,minpts); FD=save; r['label']=label; return r
res['gt_lowdens']=strat('gt',5,5,15,'GT n_gt in [5,15)')
res['gt_highdens']=strat('gt',10,15,10**9,'GT n_gt>=15')
json.dump(res,open(OUT/'ripley_results.json','w'),indent=2)

# ---------- figure ----------
fig,ax=plt.subplots(1,2,figsize=(12.5,5.0))
for a,kind,ttl in [(ax[0],'gt','GT parasite centroids  (null: CSR)'),(ax[1],'gated','gated-tile centroids  (null: random tile subset)')]:
    R=res[kind]; rg=np.array(R['RG'])
    a.axhline(0,color='k',lw=0.8)
    a.fill_between(rg,R['pt_lo'],R['pt_hi'],color='#bbbbbb',alpha=0.5,label='pointwise 95% null')
    a.plot(rg,R['glob_lo'],'--',color='#d62728',lw=1.0); a.plot(rg,R['glob_hi'],'--',color='#d62728',lw=1.0,label='global 95% envelope')
    a.plot(rg,R['O'],color='#1f77b4',lw=2.0,label='observed mean')
    a.set_title(f"{ttl}\nfields={R['nfields']}  global MAD p={R['p_global']:.3f}",fontsize=10)
    a.set_xlabel('r (px)'); a.set_ylabel('L(r) - r'); a.legend(fontsize=8)
plt.tight_layout(); plt.savefig(OUT/'ripley_summary.png',dpi=150,bbox_inches='tight'); plt.close()
print('RIPLEY_DONE',flush=True)
