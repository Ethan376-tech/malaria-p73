# -*- coding: utf-8 -*-
"""S2.2 v3 — class-discriminative pseudo-box generation with the IN-DOMAIN fine-tuned ThickDINO (domain A / Ibadan).
v1/v2 showed frozen CLS-sim is foreground-saliency (floods) and the weak gate doesn't transfer. Fix: fine-tune
ThickDINO+MIL head end-to-end on Ibadan sample labels (purely weak), then use (i) the per-tile logit as a
class-discriminative GATE and (ii) Grad-CAM (gradient of the parasite logit w.r.t. patch tokens) as the localiser.
ICAM runs in-domain, so the A->B Grad-CAM collapse (S1) is irrelevant. Constants: S2.1 (A: box 42.5, peak-dist 21);
gate + Grad-CAM thresholds calibrated on dev FASTMAL fields (Youden). Eval vs FASTMAL GT on test fields.
Output -> outputs/experiments/s2_generate/s2_v3_A.json (+ figure)."""
import os, sys, json, random, time
os.environ['XFORMERS_DISABLED']='1'
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image; Image.MAX_IMAGE_PIXELS=None
sys.path.insert(0,'/root/autodl-tmp/A_Dissertation/references/RedDino/train'); sys.path.append('/root/A_Dissertation')
from common.paths import RAW, CHECKPOINTS, OUTPUTS
from sklearn.metrics import roc_curve, roc_auc_score
from scipy.stats import pearsonr
from dinov2.models.vision_transformer import vit_small
dev='cuda'; TILE=224; P=16; G=TILE//P
random.seed(0); np.random.seed(0); torch.manual_seed(0)
CAL=json.load(open(OUTPUTS/'experiments'/'s2_calib'/'calib_constants.json'))['A']
RES=OUTPUTS/'experiments'/'s2_generate'; CACHE=OUTPUTS/'experiments'/'b2_tilecache_v2_128'
norm=T.Compose([T.ToTensor(),T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
MEAN=torch.tensor([0.485,0.456,0.406]).view(1,3,1,1).to(dev); STD=torch.tensor([0.229,0.224,0.225]).view(1,3,1,1).to(dev)

def thickdino():
    m=vit_small(patch_size=16,img_size=224,block_chunks=0,init_values=1.0,drop_path_rate=0.1)
    m.load_state_dict(torch.load(CHECKPOINTS/'ssl_vits_supvit_dinoadapt'/'encoder_final.pth',map_location='cpu'),strict=False); return m
class MILNet(nn.Module):
    def __init__(self): super().__init__(); self.enc=thickdino(); self.head=nn.Sequential(nn.Dropout(0.2),nn.Linear(384,1))
    def forward(self,x): B,K=x.shape[:2]; h=self.enc(x.flatten(0,1)).view(B,K,-1).amax(1); return self.head(h).squeeze(-1)

# ---- (1) in-domain FT on Ibadan tiles (purely weak; FASTMAL fields are a disjoint detection-GT subset) ----
man=pd.read_csv(CACHE/'manifest_v2_128.csv'); ib=man[man.dataset=='ibadan'].reset_index(drop=True)
def load_bag(sid,k):
    a=np.load(CACHE/f'{sid}.npy'); idx=np.random.choice(len(a),k,replace=len(a)<k); return a[idx]
def to_dev(arrs): x=torch.from_numpy(np.stack(arrs)).to(dev).float().div(255).permute(0,1,4,2,3); return (x-MEAN.unsqueeze(1))/STD.unsqueeze(1)
net=MILNet().to(dev); ids=ib.sample_id.tolist(); y=ib.set_index('sample_id').label01
opt=torch.optim.AdamW([{'params':net.enc.parameters(),'lr':5e-5},{'params':net.head.parameters(),'lr':1e-3}],weight_decay=0.05)
yarr=np.array([y[s] for s in ids]); pw=torch.tensor([(yarr==0).sum()/max(1,(yarr==1).sum())],device=dev,dtype=torch.float32)
EP=6; BBAG=2; t0=time.time()
for ep in range(EP):
    net.train(); order=np.random.permutation(len(ids))
    for i in range(0,len(ids),BBAG):
        bids=[ids[j] for j in order[i:i+BBAG]]
        x=to_dev([load_bag(s,32) for s in bids]); yb=torch.tensor([y[s] for s in bids],dtype=torch.float32,device=dev)
        with torch.autocast('cuda',dtype=torch.bfloat16): loss=F.binary_cross_entropy_with_logits(net(x),yb,pos_weight=pw)
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(),3.0); opt.step()
    print(f'ep{ep} loss {loss.item():.3f} {(time.time()-t0)/60:.1f}min',flush=True)
net.eval()
out=CHECKPOINTS/'ssl_vits_thickdino_ft_ibadan'; out.mkdir(parents=True,exist_ok=True)
torch.save(net.enc.state_dict(),out/'encoder.pth'); torch.save(net.head.state_dict(),out/'head.pth')
print('in-domain FT (A) trained + saved',flush=True)

# ---- Grad-CAM on the FT model ----
acts={}
def hook(mod,i,o):
    if o.requires_grad: o.retain_grad()
    acts['o']=o
net.enc.blocks[-4].register_forward_hook(hook)
def tile_logit_and_cam(crops, bs=24):
    logits=[]; cams=[]
    for i in range(0,len(crops),bs):
        xb=torch.stack([norm(Image.fromarray(c)) for c in crops[i:i+bs]]).to(dev).requires_grad_(True)
        ff=net.enc.forward_features(xb); cls=ff['x_norm_clstoken']; score=net.head(cls).squeeze(-1)
        net.zero_grad(); score.sum().backward()
        o=acts['o']; alpha=o.grad[:,1:,:].mean(1,keepdim=True); cam=F.relu((o[:,1:,:].detach()*alpha).sum(-1))  # (b,196)
        logits.append(score.detach().cpu().numpy()); cams.append(cam.detach().cpu().numpy())
    return np.concatenate(logits), np.concatenate(cams)

# ---- FASTMAL fields (domain A); dev=even, test=odd ----
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
def field_tiles_offsets(im):
    W,H=im.size; arr=np.asarray(im)
    xs=sorted(set(list(range(0,max(1,W-TILE+1),TILE))+[max(0,W-TILE)])); ys=sorted(set(list(range(0,max(1,H-TILE+1),TILE))+[max(0,H-TILE)]))
    offs=[(tx,ty) for ty in ys for tx in xs]; return [arr[ty:ty+TILE,tx:tx+TILE] for (tx,ty) in offs], offs
def iou(a,b):
    ix1,iy1=max(a[0],b[0]),max(a[1],b[1]); ix2,iy2=min(a[2],b[2]),min(a[3],b[3]); iw,ih=max(0,ix2-ix1),max(0,iy2-iy1); inter=iw*ih
    ua=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter; return inter/ua if ua>0 else 0.0
def nms(c,md):
    c=sorted(c,key=lambda z:-z[2]); k=[]
    for cx,cy,s in c:
        if all((cx-kx)**2+(cy-ky)**2>=md*md for kx,ky,_ in k): k.append((cx,cy,s))
    return k

fields=sorted(fastmal_fields(),key=lambda z:str(z[0])); random.Random(0).shuffle(fields)
devF=fields[0::2][:20]; testF=fields[1::2][:40]
# calibrate gate (tile logit) + CAM threshold on dev
gl=[]; ghas=[]; camvals=[]; camlab=[]
for ip,gtb in devF:
    try: im=Image.open(ip).convert('RGB')
    except Exception: continue
    cents=[(x+w/2,y+h/2) for (x,y,w,h) in gtb]; crops,offs=field_tiles_offsets(im); logit,cam=tile_logit_and_cam(crops)
    for (tx,ty),lg,cm in zip(offs,logit,cam):
        has=any(tx<=cx<tx+TILE and ty<=cy<ty+TILE for (cx,cy) in cents); gl.append(lg); ghas.append(1 if has else 0)
        if has:
            for gy in range(G):
                for gx in range(G):
                    px,py=tx+gx*P+P/2, ty+gy*P+P/2
                    near=any((px-cx)**2+(py-cy)**2<=(P*1.5)**2 for (cx,cy) in cents); camvals.append(cm[gy*G+gx]); camlab.append(1 if near else 0)
gl=np.array(gl); ghas=np.array(ghas); camvals=np.array(camvals); camlab=np.array(camlab)
gauc=roc_auc_score(ghas,gl); f,t,th=roc_curve(ghas,gl); gate_thr=float(th[np.argmax(t-f)])
cauc=roc_auc_score(camlab,camvals); f2,t2,th2=roc_curve(camlab,camvals); cam_thr=float(th2[np.argmax(t2-f2)])
print(f'gate dev-AUC {gauc:.3f} thr {gate_thr:.3f} | CAM dev-AUC {cauc:.3f} thr {cam_thr:.3f}',flush=True)

bs=CAL['box_size_px']; md=CAL['peak_min_dist_px']
all_tp={0.3:[],0.5:[]}; all_sc=[]; n_gt=0; pc=[]; gc=[]; ov=None
for ip,gtb in testF:
    try: im=Image.open(ip).convert('RGB')
    except Exception: continue
    gt=[(x,y,x+w,y+h) for (x,y,w,h) in gtb]; n_gt+=len(gt); gc.append(len(gt))
    crops,offs=field_tiles_offsets(im); logit,cam=tile_logit_and_cam(crops); cand=[]
    for (tx,ty),lg,cm in zip(offs,logit,cam):
        if lg<gate_thr: continue
        sg=cm.reshape(G,G)
        for gy in range(G):
            for gx in range(G):
                if sg[gy,gx]>=cam_thr: cand.append((tx+gx*P+P/2,ty+gy*P+P/2,float(sg[gy,gx])))
    peaks=nms(cand,md); pred=[(cx-bs/2,cy-bs/2,cx+bs/2,cy+bs/2,s) for (cx,cy,s) in peaks]; pc.append(len(pred)); pred.sort(key=lambda z:-z[4])
    for iouth in (0.3,0.5):
        matched=set()
        for pb in pred:
            best=-1;bj=-1
            for jg,g in enumerate(gt):
                if jg in matched: continue
                v=iou(pb[:4],g)
                if v>best: best,bj=v,jg
            tp=1 if best>=iouth and bj>=0 else 0
            if tp: matched.add(bj)
            all_tp[iouth].append(tp)
            if iouth==0.3: all_sc.append(pb[4])
    if ov is None and len(gt)>=3 and len(pred)>=3: ov=(np.asarray(im),gt,pred)
def ap(tps,sc,ng):
    if not tps: return 0.0
    o=np.argsort(-np.array(sc)); tps=np.array(tps)[o]; ctp=np.cumsum(tps); cfp=np.cumsum(1-tps); rec=ctp/max(1,ng); pre=ctp/np.maximum(1,ctp+cfp)
    mr=np.concatenate([[0],rec,[1]]); mp=np.concatenate([[0],pre,[0]])
    for i in range(len(mp)-1,0,-1): mp[i-1]=max(mp[i-1],mp[i])
    idx=np.where(mr[1:]!=mr[:-1])[0]; return float(np.sum((mr[idx+1]-mr[idx])*mp[idx+1]))
res={'domain':'A_ibadan_fastmal_FT','gate_dev_auc':round(gauc,3),'cam_dev_auc':round(cauc,3),'gate_thr':round(gate_thr,3),'cam_thr':round(cam_thr,3),
     'n_test_fields':len(testF),'n_gt':n_gt,'n_pred':int(sum(pc)),'box_size':bs,'peak_min_dist':md}
for iouth in (0.3,0.5):
    tps=all_tp[iouth]; TP=int(np.sum(tps)); FP=len(tps)-TP
    res[f'P@{iouth}']=round(TP/max(1,TP+FP),3); res[f'R@{iouth}']=round(TP/max(1,n_gt),3); res[f'AP@{iouth}']=round(ap(tps,all_sc,n_gt),3)
if len(pc)>2 and np.std(pc)>0 and np.std(gc)>0: res['count_pearson']=round(float(pearsonr(pc,gc)[0]),3)
json.dump(res,open(RES/'s2_v3_A.json','w'),indent=2); print(res,flush=True)
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt; import matplotlib.patches as mpat
if ov is not None:
    arr,gt,pred=ov; fig,ax=plt.subplots(figsize=(8,7)); ax.imshow(arr); ax.axis('off')
    ax.set_title(f"A in-domain FT Grad-CAM: P@0.3 {res['P@0.3']} R@0.3 {res['R@0.3']} AP@0.3 {res['AP@0.3']}",fontsize=10)
    matched=set()
    for pb in sorted(pred,key=lambda z:-z[4]):
        best=-1;bj=-1
        for jg,g in enumerate(gt):
            if jg in matched: continue
            v=iou(pb[:4],g)
            if v>best: best,bj=v,jg
        col='lime' if best>=0.3 else 'orange'
        if best>=0.3: matched.add(bj)
        ax.add_patch(mpat.Rectangle((pb[0],pb[1]),pb[2]-pb[0],pb[3]-pb[1],fill=False,edgecolor=col,lw=1.3))
    for jg,g in enumerate(gt):
        if jg not in matched: ax.add_patch(mpat.Rectangle((g[0],g[1]),g[2]-g[0],g[3]-g[1],fill=False,edgecolor='red',lw=1.0,ls='--'))
    plt.tight_layout(); plt.savefig(RES/'fig_s2_v3_A.png',dpi=130,bbox_inches='tight')
print('S2_V3_DONE',flush=True)
