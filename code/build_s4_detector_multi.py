# -*- coding: utf-8 -*-
"""(ii) §2.4 downstream patch-level detector — EXTRINSIC validation of the §2.3 pseudo-labels (domain A).
Train a patch parasite detector (FROZEN ThickDINO patch tokens + logistic head) on pseudo-labels generated on Ibadan
BAG fields (disjoint from FASTMAL), under two label sets — A0 (all pseudo-boxes) vs faintness-PROTECTED — plus a
GT-supervised upper bound (trained on FASTMAL-dev real boxes). Test = detection AP on FASTMAL GT (overall + by
parasitaemia band). Headline: do protected pseudo-labels train a better detector than A0?  Output -> s4_detector/."""
import os, sys, json, random
os.environ['XFORMERS_DISABLED']='1'
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image; Image.MAX_IMAGE_PIXELS=None
sys.path.insert(0,'/root/autodl-tmp/A_Dissertation/references/RedDino/train'); sys.path.append('/root/A_Dissertation')
from common.paths import RAW, CHECKPOINTS, OUTPUTS, IBADAN_PART1, IBADAN_PART2
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve
from dinov2.models.vision_transformer import vit_small
dev='cuda'; TILE=224; P=16; G=TILE//P; T_MC=20
random.seed(0); np.random.seed(0); torch.manual_seed(0)
CAL=json.load(open(OUTPUTS/'experiments'/'s2_calib'/'calib_constants.json'))['A']
FIN=json.load(open(OUTPUTS/'experiments'/'s3_filter'/'final.json'))['calib']
RES=OUTPUTS/'experiments'/'s4_detector_multi'; RES.mkdir(parents=True,exist_ok=True)
norm=T.Compose([T.ToTensor(),T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
def mk(ckpt,ft=False):
    m=vit_small(patch_size=16,img_size=224,block_chunks=0,init_values=1.0,drop_path_rate=0.1 if ft else 0.0)
    m.load_state_dict(torch.load(ckpt,map_location='cpu'),strict=False); return m.to(dev).eval()
frozen=mk(CHECKPOINTS/'ssl_vits_supvit_dinoadapt'/'encoder_final.pth')                 # detector backbone (frozen)
genc=mk(CHECKPOINTS/'ssl_vits_thickdino_ft_ibadan'/'encoder.pth',ft=True)               # generator (FT) for pseudo-labels
ghead=nn.Sequential(nn.Dropout(0.2),nn.Linear(384,1)).to(dev); ghead.load_state_dict(torch.load(CHECKPOINTS/'ssl_vits_thickdino_ft_ibadan'/'head.pth',map_location='cpu')); ghead.eval()
acts={}
def hook(mod,i,o):
    if o.requires_grad: o.retain_grad()
    acts['o']=o
genc.blocks[-4].register_forward_hook(hook)
gate_thr=FIN['gate_thr']; cam_thr=FIN['cam_thr']; b_protect=FIN['b_protect']; a_star=FIN['a_star']; BS=CAL['box_size_px']; MD=CAL['peak_min_dist_px']
# multi-signal joint thresholds + typicality manifold (dev-TP genc embeddings, from perbox_multi2)
from sklearn.decomposition import PCA; from sklearn.covariance import LedoitWolf
cj=json.load(open(OUTPUTS/'experiments'/'s3_filter'/'joint_calib.json'))['orig']['joint']['cal']; bMC=cj['bMC']; bTTA=cj['bTTA']; a_conf=cj['a']; tauT=cj['tau']
_pb=np.load(OUTPUTS/'experiments'/'s3_filter'/'perbox_multi2.npz'); _dev=_pb['field_id'].astype(int)<95; _dtp=(_pb['is_tp']==1)&_dev; _E=_pb['emb'][_dtp]
_mu=_E.mean(0); _sd=_E.std(0)+1e-6; _pca=PCA(40,random_state=0).fit((_E-_mu)/_sd); _lw=LedoitWolf().fit(_pca.transform((_E-_mu)/_sd))
def maha_of(emb): return _lw.mahalanobis(_pca.transform((emb-_mu)/_sd)) if len(emb) else np.array([])
@torch.no_grad()
def gen_embed_tta(crops,bs=48):
    genc.eval(); ghead.eval(); EMB=[]; TV=[]
    for i in range(0,len(crops),bs):
        xb=torch.stack([norm(Image.fromarray(c)) for c in crops[i:i+bs]]).to(dev)
        EMB.append(genc.forward_features(xb)['x_norm_clstoken'].cpu().numpy())
        views=[xb,torch.flip(xb,[3]),torch.flip(xb,[2]),torch.rot90(xb,1,[2,3]),torch.rot90(xb,2,[2,3]),torch.rot90(xb,3,[2,3])]
        pv=[torch.sigmoid(ghead(genc.forward_features(v)['x_norm_clstoken']).squeeze(-1)).cpu().numpy() for v in views]
        TV.append(np.stack(pv).var(0))
    return (np.concatenate(EMB),np.concatenate(TV)) if EMB else (np.zeros((0,384),np.float32),np.array([]))

@torch.no_grad()
def frozen_patches(crops,bs=48):
    out=[]
    for i in range(0,len(crops),bs):
        xb=torch.stack([norm(Image.fromarray(c)) for c in crops[i:i+bs]]).to(dev)
        out.append(frozen.forward_features(xb)['x_norm_patchtokens'].cpu().numpy())  # (b,196,384)
    return np.concatenate(out) if out else np.zeros((0,G*G,384),np.float32)
def gen_logit_cam(crops,bs=24):
    lo=[];ca=[]
    for i in range(0,len(crops),bs):
        xb=torch.stack([norm(Image.fromarray(c)) for c in crops[i:i+bs]]).to(dev).requires_grad_(True)
        ff=genc.forward_features(xb); s=ghead(ff['x_norm_clstoken']).squeeze(-1); genc.zero_grad(); ghead.zero_grad(); s.sum().backward()
        o=acts['o']; al=o.grad[:,1:,:].mean(1,keepdim=True); cam=F.relu((o[:,1:,:].detach()*al).sum(-1))
        lo.append(s.detach().cpu().numpy()); ca.append(cam.detach().cpu().numpy())
    return np.concatenate(lo),np.concatenate(ca)
@torch.no_grad()
def gen_mc(crops,bs=48):
    for mod in list(genc.modules())+list(ghead.modules()):
        if isinstance(mod,nn.Dropout): mod.train()
    for mod in genc.modules():
        if mod.__class__.__name__=='DropPath': mod.train()
    means=[]; vars=[]
    for i in range(0,len(crops),bs):
        xb=torch.stack([norm(Image.fromarray(c)) for c in crops[i:i+bs]]).to(dev); pr=[]
        for _ in range(T_MC): ff=genc.forward_features(xb); pr.append(torch.sigmoid(ghead(ff['x_norm_clstoken']).squeeze(-1)).cpu().numpy())
        A=np.stack(pr); means.append(A.mean(0)); vars.append(A.var(0))
    genc.eval(); ghead.eval(); return np.concatenate(means),np.concatenate(vars)
def toff(im):
    W,H=im.size; arr=np.asarray(im); xs=sorted(set(list(range(0,max(1,W-TILE+1),TILE))+[max(0,W-TILE)])); ys=sorted(set(list(range(0,max(1,H-TILE+1),TILE))+[max(0,H-TILE)]))
    return [(arr[ty:ty+TILE,tx:tx+TILE],(tx,ty)) for ty in ys for tx in xs],arr,W,H
def nms(c,md):
    c=sorted(c,key=lambda z:-z[2]); k=[]
    for cx,cy,s in c:
        if all((cx-kx)**2+(cy-ky)**2>=md*md for kx,ky,_ in k): k.append((cx,cy,s))
    return k
def iou(a,b):
    ix1,iy1=max(a[0],b[0]),max(a[1],b[1]); ix2,iy2=min(a[2],b[2]),min(a[3],b[3]); iw,ih=max(0,ix2-ix1),max(0,iy2-iy1); it=iw*ih
    ua=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-it; return it/ua if ua>0 else 0.0

# ---- generate pseudo-boxes on a field; return A0 boxes and protected boxes (field coords) ----
def pseudo_boxes(im):
    tiles,arr,W,H=toff(im); crops=[c for c,_ in tiles]; offs=[o for _,o in tiles]; lo,ca=gen_logit_cam(crops); cand=[]
    for (tx,ty),lg,cm in zip(offs,lo,ca):
        if lg<gate_thr: continue
        sg=cm.reshape(G,G)
        for gy in range(G):
            for gx in range(G):
                if sg[gy,gx]>=cam_thr: cand.append((tx+gx*P+P/2,ty+gy*P+P/2,float(sg[gy,gx])))
    peaks=nms(cand,MD); peaks=sorted(peaks,key=lambda z:-z[2])[:300]   # cap top-300 by CAM score (bounds MC compute)
    if not peaks: return [],[],[]
    bc=[]
    for (cx,cy,s) in peaks:
        x0=int(min(max(cx-TILE/2,0),W-TILE)); y0=int(min(max(cy-TILE/2,0),H-TILE)); bc.append(arr[y0:y0+TILE,x0:x0+TILE])
    pm,pv=gen_mc(bc); emb,tta=gen_embed_tta(bc); mh=maha_of(emb)
    A0=[(cx,cy) for (cx,cy,s) in peaks]
    prot=[(peaks[j][0],peaks[j][1]) for j in range(len(peaks)) if (pv[j]>=b_protect or pm[j]>=a_star)]
    prot_multi=[(peaks[j][0],peaks[j][1]) for j in range(len(peaks)) if ((pv[j]>=bMC or tta[j]>=bTTA or pm[j]>=a_conf) and mh[j]<=tauT)]
    return A0,prot,prot_multi

# ---- TRAIN fields: Ibadan positive-sample fields (disjoint from FASTMAL) ----
bm=pd.read_csv(OUTPUTS/'experiments'/'stageD_bag'/'bag_meta_v2.csv')
ibcsv=pd.read_csv('/root/autodl-tmp/A_Dissertation/data/raw/ibadan/sample_codes_parasite_diagnosis_crosscheck.csv')
posids=set(bm[(bm.dataset=='ibadan')&(bm.label01==1)].sample_id)
sampdir={r.sample_code:(IBADAN_PART1 if 'part1' in r.input_file else IBADAN_PART2)/r.sample_code for _,r in ibcsv.iterrows()}
EXT={'.tif','.tiff','.png','.jpg','.jpeg'}
train_fields=[]
for sid in sorted(posids):
    d=sampdir.get(sid)
    if d is None or not d.exists(): continue
    fs=[p for p in d.rglob('*') if p.suffix.lower() in EXT][:2]   # 2 fields/sample
    train_fields+=fs
    if len(train_fields)>=80: break
random.Random(0).shuffle(train_fields); train_fields=train_fields[:70]
print(f'train fields {len(train_fields)} (Ibadan pos, disjoint from FASTMAL)',flush=True)

def collect_patches(boxes_per_field, fields):
    """frozen patch tokens + label (in any pseudo-box of this set)."""
    X=[]; Y=[]
    for ip,boxes in zip(fields,boxes_per_field):
        if not boxes: continue
        try: im=Image.open(ip).convert('RGB')
        except Exception: continue
        tiles,arr,W,H=toff(im); crops=[c for c,_ in tiles]; offs=[o for _,o in tiles]
        # only sample tiles that contain a pseudo-box (positives) + their patches; plus some background tiles
        toks=frozen_patches(crops)
        for ti,((tx,ty),tk) in enumerate(zip(offs,toks)):
            # label patches by box membership
            lab=np.zeros(G*G,bool); hasbox=False
            for (cx,cy) in boxes:
                if tx-BS<=cx<tx+TILE+BS and ty-BS<=cy<ty+TILE+BS:
                    gx=int((cx-tx)//P); gy=int((cy-ty)//P)
                    for dy in(-1,0,1):
                        for dx in(-1,0,1):
                            nx,ny=gx+dx,gy+dy
                            if 0<=nx<G and 0<=ny<G: lab[ny*G+nx]=True; hasbox=True
            if hasbox or random.random()<0.15:   # all box-tiles + 15% background tiles
                pos=np.where(lab)[0]; neg=np.where(~lab)[0]
                sel=list(pos)+list(np.random.choice(neg,min(len(neg),max(4,2*len(pos))),replace=False))
                for pidx in sel: X.append(tk[pidx]); Y.append(int(lab[pidx]))
    return np.array(X),np.array(Y)

# generate pseudo-labels on train fields
A0_tr=[]; PR_tr=[]; PM_tr=[]
for k,ip in enumerate(train_fields):
    try: im=Image.open(ip).convert('RGB')
    except Exception: A0_tr.append([]); PR_tr.append([]); PM_tr.append([]); continue
    a0,pr,pmul=pseudo_boxes(im); A0_tr.append(a0); PR_tr.append(pr); PM_tr.append(pmul)
    if (k+1)%20==0: print(f'pseudo-labelled {k+1}/{len(train_fields)} fields',flush=True)
XA,YA=collect_patches(A0_tr,train_fields); XP,YP=collect_patches(PR_tr,train_fields); XM,YM=collect_patches(PM_tr,train_fields)
nb_a0=sum(len(b) for b in A0_tr); nb_pr=sum(len(b) for b in PR_tr); nb_mu=sum(len(b) for b in PM_tr)
print(f'boxes: A0 {nb_a0} | single {nb_pr} | multi {nb_mu}',flush=True)
print(f'patches: A0 {len(YA)} (pos {int(YA.sum())}) | single {len(YP)} (pos {int(YP.sum())}) | multi {len(YM)} (pos {int(YM.sum())})',flush=True)

# ---- FASTMAL fields for GT-supervised training (dev) + test ----
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
fields=sorted(fastmal_fields(),key=lambda z:str(z[0])); random.Random(0).shuffle(fields)
devF=fields[0::2][:20]; testF=fields[1::2][:40]
# GT-supervised patches from dev (real boxes)
def collect_gt_patches(fset):
    X=[];Y=[]
    for ip,gtb in fset:
        try: im=Image.open(ip).convert('RGB')
        except Exception: continue
        cents=[(x+w/2,y+h/2) for (x,y,w,h) in gtb]; tiles,arr,W,H=toff(im); crops=[c for c,_ in tiles]; offs=[o for _,o in tiles]; toks=frozen_patches(crops)
        for (tx,ty),tk in zip(offs,toks):
            lab=np.zeros(G*G,bool); hb=False
            for (cx,cy) in cents:
                if tx<=cx<tx+TILE and ty<=cy<ty+TILE:
                    gx=int((cx-tx)//P); gy=int((cy-ty)//P)
                    for dy in(-1,0,1):
                        for dx in(-1,0,1):
                            nx,ny=gx+dx,gy+dy
                            if 0<=nx<G and 0<=ny<G: lab[ny*G+nx]=True; hb=True
            if hb or random.random()<0.15:
                pos=np.where(lab)[0]; neg=np.where(~lab)[0]; sel=list(pos)+list(np.random.choice(neg,min(len(neg),max(4,2*len(pos))),replace=False))
                for pidx in sel: X.append(tk[pidx]); Y.append(int(lab[pidx]))
    return np.array(X),np.array(Y)
XG,YG=collect_gt_patches(devF); print(f'GT-sup train patches {len(YG)} (pos {int(YG.sum())})',flush=True)

def train_head(X,Y):
    sc=StandardScaler().fit(X); lr=LogisticRegression(max_iter=3000,class_weight='balanced').fit(sc.transform(X),Y); return sc,lr
heads={'A0':train_head(XA,YA),'single_signal':train_head(XP,YP),'multi_signal':train_head(XM,YM),'GT_sup':train_head(XG,YG)}

# ---- TEST: detection AP on FASTMAL test, per detector, by band ----
def detect_field(im, sc, lr):
    tiles,arr,W,H=toff(im); crops=[c for c,_ in tiles]; offs=[o for _,o in tiles]; toks=frozen_patches(crops); cand=[]
    for (tx,ty),tk in zip(offs,toks):
        p=lr.predict_proba(sc.transform(tk))[:,1]
        for g in range(G*G):
            gx,gy=g%G,g//G; cand.append((tx+gx*P+P/2,ty+gy*P+P/2,float(p[g])))
    return nms(cand,MD)
def evaluate(sc,lr):
    all_tp={0.3:[],0.5:[]}; sc_all=[]; n_gt=0; fcounts=[]; tp03_band={'low':[],'high':[]}; sc_band={'low':[],'high':[]}; gtband={'low':0,'high':0}
    rows=[]
    for ip,gtb in testF:
        try: im=Image.open(ip).convert('RGB')
        except Exception: continue
        gt=[(x,y,x+w,y+h) for (x,y,w,h) in gtb]; n_gt+=len(gt); fcounts.append(len(gt)); rows.append((ip,gt,len(gt)))
    thr=np.median([r[2] for r in rows]) if rows else 0
    for ip,gt,fcnt in rows:
        band='low' if fcnt<=thr else 'high'; gtband[band]+=len(gt)
        im=Image.open(ip).convert('RGB'); peaks=detect_field(im,sc,lr)
        boxes=[(cx-BS/2,cy-BS/2,cx+BS/2,cy+BS/2,s) for (cx,cy,s) in peaks]; boxes.sort(key=lambda z:-z[4])
        for iouth in (0.3,0.5):
            matched=set()
            for b in boxes:
                best=-1;bj=-1
                for jg,g in enumerate(gt):
                    if jg in matched: continue
                    v=iou(b[:4],g)
                    if v>best: best,bj=v,jg
                tp=1 if best>=iouth and bj>=0 else 0
                if tp: matched.add(bj)
                all_tp[iouth].append(tp)
                if iouth==0.3: sc_all.append(b[4]); tp03_band[band].append(tp); sc_band[band].append(b[4])
    def ap(tps,scs,ng):
        if not tps: return 0.0
        o=np.argsort(-np.array(scs)); tps=np.array(tps)[o]; ctp=np.cumsum(tps); cfp=np.cumsum(1-tps); rec=ctp/max(1,ng); pre=ctp/np.maximum(1,ctp+cfp)
        mr=np.concatenate([[0],rec,[1]]); mp=np.concatenate([[0],pre,[0]])
        for i in range(len(mp)-1,0,-1): mp[i-1]=max(mp[i-1],mp[i])
        idx=np.where(mr[1:]!=mr[:-1])[0]; return float(np.sum((mr[idx+1]-mr[idx])*mp[idx+1]))
    out={'AP@0.3':round(ap(all_tp[0.3],sc_all,n_gt),3),'AP@0.5':round(ap(all_tp[0.5],sc_all,n_gt),3),'n_gt':n_gt,
         'AP@0.3_low':round(ap(tp03_band['low'],sc_band['low'],gtband['low']),3),
         'AP@0.3_high':round(ap(tp03_band['high'],sc_band['high'],gtband['high']),3)}
    return out
results={k:evaluate(sc,lr) for k,(sc,lr) in heads.items()}
print('\n=== PATCH-HEAD detection AP on FASTMAL (test): A0 vs single vs multi vs GT ===')
for k in ['A0','single_signal','multi_signal','GT_sup']: print(f'  {k:14s} {results[k]}')
json.dump(dict(results=results,boxes=dict(A0=nb_a0,single=nb_pr,multi=nb_mu),
               patches=dict(A0=len(YA),single=len(YP),multi=len(YM)),pos=dict(A0=int(YA.sum()),single=int(YP.sum()),multi=int(YM.sum()))),
          open(RES/'s4_detector_multi.json','w'),indent=2)
print('S4_MULTI_DONE',flush=True)
