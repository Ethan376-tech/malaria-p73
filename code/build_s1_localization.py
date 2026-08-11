# -*- coding: utf-8 -*-
"""S1 — compare ViT parasite-localisation methods on FROZEN ThickDINO (weak / training-free; GT boxes only for eval).
Methods: (a) patch-token CAM (weak MIL direction · patch); (b) CLS-similarity (attention proxy); (c) Grad-CAM.
Eval on FASTMAL (Ibadan, domain A): per-tile patch-level ROC-AUC + pointing-game hit-rate (argmax / top-3 patch
inside a parasite box). Output -> outputs/experiments/s1_localization/"""
import os, sys, json, glob, random
os.environ['XFORMERS_DISABLED']='1'
import numpy as np, pandas as pd, torch, torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image; Image.MAX_IMAGE_PIXELS=None
sys.path.insert(0,'/root/autodl-tmp/A_Dissertation/references/RedDino/train'); sys.path.append('/root/A_Dissertation')
from common.paths import RAW, CHECKPOINTS, OUTPUTS
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from dinov2.models.vision_transformer import vit_small
dev='cuda'; TILE=224; P=16; G=TILE//P   # 14x14 grid
random.seed(0); np.random.seed(0)
RES=OUTPUTS/'experiments'/'s1_localization'; RES.mkdir(parents=True,exist_ok=True)
norm=T.Compose([T.ToTensor(),T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])

# ---- frozen ThickDINO ----
m=vit_small(patch_size=P,img_size=224,block_chunks=0,init_values=1.0)
m.load_state_dict(torch.load(CHECKPOINTS/'ssl_vits_supvit_dinoadapt'/'encoder_final.pth',map_location='cpu'),strict=False)
m.eval().to(dev)
for p in m.parameters(): p.requires_grad_(False)

# ---- weak MIL parasite direction w (logistic on max-pooled bag emb, SAMPLE labels) ----
S=OUTPUTS/'experiments'/'stageD_bag'; bm=pd.read_csv(S/'bag_meta.csv')
Xb=np.load(S/'emb_thickdino.npy').max(1); yb=bm.label01.values
sc=StandardScaler().fit(Xb); lr=LogisticRegression(max_iter=2000).fit(sc.transform(Xb),yb)
w=torch.tensor((lr.coef_[0]/sc.scale_),dtype=torch.float32,device=dev)   # fold standardiser into w
print('weak parasite direction ready', flush=True)

# ---- FASTMAL parasite-centred eval tiles ----
def parasite_centres(field_rois):
    c=[]
    for r in field_rois:
        t=r.get('type','')
        if 'PARASITE' in t and 'CROWD' not in t:
            c.append((float(r['x'])+float(r['width'])/2, float(r['y'])+float(r['height'])/2))
    return c
tiles=[]; labels=[]   # tiles: uint8 (224,224,3); labels: bool (196,)
fm=RAW/'ibadan'/'FASTMAL_THICK_FILMS'; MAXT=500
for samp in sorted(p for p in fm.iterdir() if p.is_dir()):
    js=list(samp.glob('*.json'))
    if not js: continue
    for fld in json.loads(js[0].read_text())['rois']:
        ip=samp/fld['image_name']
        if not ip.exists(): continue
        cents=parasite_centres(fld['roi'])
        if not cents: continue
        im=Image.open(ip).convert('RGB'); W,H=im.size
        for (cx,cy) in random.sample(cents,min(2,len(cents))):       # ≤2 tiles/field
            x0=int(min(max(cx-TILE/2,0),W-TILE)); y0=int(min(max(cy-TILE/2,0),H-TILE))
            crop=im.crop((x0,y0,x0+TILE,y0+TILE))
            lab=np.zeros(G*G,bool)
            for (bx,by) in cents:
                px,py=bx-x0,by-y0
                if 0<=px<TILE and 0<=py<TILE:
                    gx,gy=int(px//P),int(py//P)
                    for dy in (-1,0,1):                  # 3x3 neighbourhood (parasite size + grid coarseness)
                        for dx in (-1,0,1):
                            nx,ny=gx+dx,gy+dy
                            if 0<=nx<G and 0<=ny<G: lab[ny*G+nx]=True
            if lab.any():
                tiles.append(np.asarray(crop,np.uint8)); labels.append(lab)
        if len(tiles)>=MAXT: break
    if len(tiles)>=MAXT: break
print(f'eval tiles: {len(tiles)} | mean parasite-patches/tile: {np.mean([l.sum() for l in labels]):.1f}', flush=True)

# ---- saliencies ----
def sal_all(tiles_u8, bs=32):
    A=[]; Bs=[]; C=[]
    acts={}
    def hook(mod,i,o):
        if o.requires_grad: o.retain_grad()        # only under enable_grad
        acts['o']=o
    h=m.blocks[-4].register_forward_hook(hook)      # earlier block (last-block patch grads ~0 for a CLS head)
    for i in range(0,len(tiles_u8),bs):
        xb=torch.stack([norm(Image.fromarray(t)) for t in tiles_u8[i:i+bs]]).to(dev)
        # (a) patch-CAM + (b) CLS-sim (no grad)
        with torch.no_grad():
            ff=m.forward_features(xb); cls=ff['x_norm_clstoken']; patch=ff['x_norm_patchtokens']
        A.append((patch@w).cpu().numpy())                                   # (B,196)
        Bs.append(F.cosine_similarity(patch, cls.unsqueeze(1), dim=-1).cpu().numpy())
        # (c) Grad-CAM: grad of the parasite score w.r.t. last-block patch activations
        with torch.enable_grad():
            xg=xb.detach().clone().requires_grad_(True)
            ff2=m.forward_features(xg); score=(ff2['x_norm_clstoken']@w).sum(); m.zero_grad(); score.backward()
            o=acts['o']; gp=o.grad[:,1:,:]; ap=o[:,1:,:].detach()            # drop CLS token -> 196 patches
            alpha=gp.mean(1,keepdim=True); cam=F.relu((ap*alpha).sum(-1))    # (B,196)
        C.append(cam.cpu().numpy())
    h.remove()
    return np.concatenate(A),np.concatenate(Bs),np.concatenate(C)
SA,SB,SC=sal_all(tiles)

# ---- metrics: per-tile patch AUC + pointing game ----
def evaluate(S):
    aucs=[]; hit1=[]; hit3=[]
    for s,lab in zip(S,labels):
        if lab.all() or not lab.any(): continue
        aucs.append(roc_auc_score(lab,s))
        order=np.argsort(-s); hit1.append(bool(lab[order[0]])); hit3.append(bool(lab[order[:3]].any()))
    return np.mean(aucs),np.std(aucs),np.mean(hit1),np.mean(hit3),len(aucs)
rows=[]
for name,S_ in [('(a) patch-CAM',SA),('(b) CLS-sim (attn proxy)',SB),('(c) Grad-CAM',SC)]:
    au,sd,h1,h3,n=evaluate(S_)
    rows.append(dict(method=name,patch_AUC=round(au,3),AUC_std=round(sd,3),point_top1=round(h1,3),point_top3=round(h3,3),n_tiles=n))
    print(rows[-1],flush=True)
df=pd.DataFrame(rows); df.to_csv(RES/'s1_localization_fastmal.csv',index=False)
print('\n',df.to_string(index=False)); print('best by patch_AUC:', df.loc[df.patch_AUC.idxmax(),'method'])

# ---- example overlay figure for the best method (4 tiles) ----
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
best=[SA,SB,SC][int(df.patch_AUC.idxmax())]
fig,axes=plt.subplots(2,4,figsize=(12,6))
for j in range(4):
    t=tiles[j]; s=best[j].reshape(G,G); lab=labels[j].reshape(G,G)
    axes[0,j].imshow(t); axes[0,j].set_title('tile + GT patches'); axes[0,j].axis('off')
    yy,xx=np.where(lab); axes[0,j].scatter(xx*P+P/2,yy*P+P/2,s=80,facecolors='none',edgecolors='lime',linewidths=2)
    axes[1,j].imshow(t); axes[1,j].imshow(np.kron(s,np.ones((P,P))),cmap='jet',alpha=0.45); axes[1,j].set_title('saliency'); axes[1,j].axis('off')
plt.tight_layout(); plt.savefig(RES/'s1_overlay_best.png',dpi=130,bbox_inches='tight')
print('WROTE ->',RES,flush=True)
