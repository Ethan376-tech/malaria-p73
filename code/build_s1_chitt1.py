# -*- coding: utf-8 -*-
"""S1 cross-domain check on Chittagong-1 (domain B, P. vivax, smartphone). Same as build_s1_localization.py but
parses Chittagong-1 boxes. Confirms whether the localiser ranking (Grad-CAM best for pointing) and frozen
localisation hold cross-domain — INDEPENDENT of the S2 ICAM-iteration question.
Output -> outputs/experiments/s1_localization/s1_localization_chitt1.csv"""
import os, sys, random
os.environ['XFORMERS_DISABLED']='1'
import numpy as np, pandas as pd, torch, torch.nn.functional as F
import torchvision.transforms as T
import scipy.ndimage as ndi
from PIL import Image; Image.MAX_IMAGE_PIXELS=None
sys.path.insert(0,'/root/autodl-tmp/A_Dissertation/references/RedDino/train'); sys.path.append('/root/A_Dissertation')
from common.paths import CHITTAGONG1, CHECKPOINTS, OUTPUTS
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from dinov2.models.vision_transformer import vit_small
dev='cuda'; TILE=224; P=16; G=TILE//P
random.seed(0); np.random.seed(0)
RES=OUTPUTS/'experiments'/'s1_localization'; RES.mkdir(parents=True,exist_ok=True)
norm=T.Compose([T.ToTensor(),T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])

m=vit_small(patch_size=P,img_size=224,block_chunks=0,init_values=1.0)
m.load_state_dict(torch.load(CHECKPOINTS/'ssl_vits_supvit_dinoadapt'/'encoder_final.pth',map_location='cpu'),strict=False)
m.eval().to(dev)
for p in m.parameters(): p.requires_grad_(False)
acts={}
def _hook(mod,i,o):
    if o.requires_grad: o.retain_grad()
    acts['o']=o
m.blocks[-4].register_forward_hook(_hook)

S_=OUTPUTS/'experiments'/'stageD_bag'; bm=pd.read_csv(S_/'bag_meta.csv')
Xb=np.load(S_/'emb_thickdino.npy').max(1); yb=bm.label01.values
sc=StandardScaler().fit(Xb); lr=LogisticRegression(max_iter=2000).fit(sc.transform(Xb),yb)
w=torch.tensor(lr.coef_[0]/sc.scale_,dtype=torch.float32,device=dev)

# ---- Chittagong-1 parasite-centred tiles + GT ----
ANN=CHITTAGONG1/'NIH-NLM-ThickBloodSmearsPV'/'All_annotations'
IMG=CHITTAGONG1/'NIH-NLM-ThickBloodSmearsPV'/'All_PvTk'
def boxes(txt):
    c=[]
    for ln in txt.read_text(errors='ignore').splitlines()[1:]:
        f=ln.split(',')
        if len(f)>=9 and f[1].strip()=='Parasitized':
            try: x1,y1,x2,y2=map(float,f[5:9]); c.append(((x1+x2)/2,(y1+y2)/2))
            except ValueError: pass
    return c
tiles=[]; labels=[]; MAXT=400
pvs=sorted([d for d in ANN.iterdir() if d.is_dir()]); random.shuffle(pvs)
for pv in pvs:
    for txt in sorted(pv.glob('*.txt')):
        cs=boxes(txt)
        if not cs: continue
        ip=IMG/pv.name/(txt.stem+'.jpg')
        if not ip.exists(): continue
        im=Image.open(ip).convert('RGB'); W,H=im.size
        if min(W,H)<TILE: continue
        for (cx,cy) in random.sample(cs,min(2,len(cs))):
            x0=int(min(max(cx-TILE/2,0),W-TILE)); y0=int(min(max(cy-TILE/2,0),H-TILE))
            lab=np.zeros(G*G,bool)
            for (bx,by) in cs:
                px,py=bx-x0,by-y0
                if 0<=px<TILE and 0<=py<TILE:
                    gx,gy=int(px//P),int(py//P)
                    for dy in (-1,0,1):
                        for dx in (-1,0,1):
                            nx,ny=gx+dx,gy+dy
                            if 0<=nx<G and 0<=ny<G: lab[ny*G+nx]=True
            if lab.any(): tiles.append(np.asarray(im.crop((x0,y0,x0+TILE,y0+TILE)),np.uint8)); labels.append(lab)
        if len(tiles)>=MAXT: break
    if len(tiles)>=MAXT: break
print(f'Chittagong-1 eval tiles: {len(tiles)} | mean parasite-patches/tile: {np.mean([l.sum() for l in labels]):.1f}', flush=True)

def sal_all(tiles_u8, bs=32):
    A=[]; Bs=[]; C=[]
    def hook2(mod,i,o):
        if o.requires_grad: o.retain_grad()
        acts['o']=o
    for i in range(0,len(tiles_u8),bs):
        xb=torch.stack([norm(Image.fromarray(t)) for t in tiles_u8[i:i+bs]]).to(dev)
        with torch.no_grad():
            ff=m.forward_features(xb); cls=ff['x_norm_clstoken']; patch=ff['x_norm_patchtokens']
        A.append((patch@w).cpu().numpy()); Bs.append(F.cosine_similarity(patch,cls.unsqueeze(1),dim=-1).cpu().numpy())
        with torch.enable_grad():
            xg=xb.detach().clone().requires_grad_(True); ff2=m.forward_features(xg); score=(ff2['x_norm_clstoken']@w).sum(); m.zero_grad(); score.backward()
            o=acts['o']; alpha=o.grad[:,1:,:].mean(1,keepdim=True); cam=F.relu((o[:,1:,:].detach()*alpha).sum(-1))
        C.append(cam.cpu().numpy())
    return np.concatenate(A),np.concatenate(Bs),np.concatenate(C)
SA,SB,SC=sal_all(tiles)
def evaluate(Sm):
    au=[]; h1=[]; h3=[]
    for s,lab in zip(Sm,labels):
        if lab.all() or not lab.any(): continue
        au.append(roc_auc_score(lab,s)); order=np.argsort(-s); h1.append(bool(lab[order[0]])); h3.append(bool(lab[order[:3]].any()))
    return round(np.mean(au),3),round(np.std(au),3),round(np.mean(h1),3),round(np.mean(h3),3),len(au)
rows=[]
for name,Sm in [('(a) patch-CAM',SA),('(b) CLS-sim (attn proxy)',SB),('(c) Grad-CAM',SC)]:
    a,sd,h1,h3,n=evaluate(Sm); rows.append(dict(method=name,patch_AUC=a,AUC_std=sd,point_top1=h1,point_top3=h3,n_tiles=n)); print(rows[-1],flush=True)
pd.DataFrame(rows).to_csv(RES/'s1_localization_chitt1.csv',index=False)
print('\n',pd.DataFrame(rows).to_string(index=False)); print('best by point_top1:', rows[int(np.argmax([r['point_top1'] for r in rows]))]['method'])
