# -*- coding: utf-8 -*-
"""S3 (i) — formalised faintness-protected filter, fully DEV-calibrated (removes the test-var-median borrowing).
Filter: DROP a box iff (p̄ < a AND var < b_protect) — i.e. drop confident-background, PROTECT high-variance (faint).
b_protect = 10th-percentile of DEV faint-TP variance (protects ≥90% of dev faint TPs); a = single swept knob, operating
point a* chosen on DEV (max F1@IoU0.3). Test (loaded from S3.2 perbox.npz) is used only to report the curve + a*.
Output -> outputs/experiments/s3_filter/ (final.json, fig_s3_final.png)."""
import os, sys, json, random
os.environ['XFORMERS_DISABLED']='1'
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image; Image.MAX_IMAGE_PIXELS=None
sys.path.insert(0,'/root/autodl-tmp/A_Dissertation/references/RedDino/train'); sys.path.append('/root/A_Dissertation')
from common.paths import RAW, CHECKPOINTS, OUTPUTS
from sklearn.metrics import roc_curve
dev='cuda'; TILE=224; P=16; G=TILE//P; T_MC=30
random.seed(0); np.random.seed(0); torch.manual_seed(0)
CAL=json.load(open(OUTPUTS/'experiments'/'s2_calib'/'calib_constants.json'))['A']
RES=OUTPUTS/'experiments'/'s3_filter'
norm=T.Compose([T.ToTensor(),T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
from dinov2.models.vision_transformer import vit_small
def td():
    m=vit_small(patch_size=16,img_size=224,block_chunks=0,init_values=1.0,drop_path_rate=0.1)
    m.load_state_dict(torch.load(CHECKPOINTS/'ssl_vits_thickdino_ft_ibadan'/'encoder.pth',map_location='cpu'),strict=False); return m
enc=td().to(dev); head=nn.Sequential(nn.Dropout(0.2),nn.Linear(384,1)).to(dev)
head.load_state_dict(torch.load(CHECKPOINTS/'ssl_vits_thickdino_ft_ibadan'/'head.pth',map_location='cpu')); enc.eval(); head.eval()
acts={}
def hook(mod,i,o):
    if o.requires_grad: o.retain_grad()
    acts['o']=o
enc.blocks[-4].register_forward_hook(hook)
def logit_cam(crops,bs=24):
    lo=[];ca=[]
    for i in range(0,len(crops),bs):
        xb=torch.stack([norm(Image.fromarray(c)) for c in crops[i:i+bs]]).to(dev).requires_grad_(True)
        ff=enc.forward_features(xb); s=head(ff['x_norm_clstoken']).squeeze(-1); enc.zero_grad(); head.zero_grad(); s.sum().backward()
        o=acts['o']; al=o.grad[:,1:,:].mean(1,keepdim=True); cam=F.relu((o[:,1:,:].detach()*al).sum(-1))
        lo.append(s.detach().cpu().numpy()); ca.append(cam.detach().cpu().numpy())
    return np.concatenate(lo),np.concatenate(ca)
@torch.no_grad()
def mc(crops):
    for mod in list(enc.modules())+list(head.modules()):
        if isinstance(mod,nn.Dropout): mod.train()
    for mod in enc.modules():
        if mod.__class__.__name__=='DropPath': mod.train()
    xb=torch.stack([norm(Image.fromarray(c)) for c in crops]).to(dev); pr=[]
    for _ in range(T_MC): ff=enc.forward_features(xb); pr.append(torch.sigmoid(head(ff['x_norm_clstoken']).squeeze(-1)).cpu().numpy())
    enc.eval(); head.eval(); A=np.stack(pr); return A.mean(0),A.var(0)
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
    W,H=im.size; arr=np.asarray(im); xs=sorted(set(list(range(0,max(1,W-TILE+1),TILE))+[max(0,W-TILE)])); ys=sorted(set(list(range(0,max(1,H-TILE+1),TILE))+[max(0,H-TILE)]))
    offs=[(tx,ty) for ty in ys for tx in xs]; return [arr[ty:ty+TILE,tx:tx+TILE] for (tx,ty) in offs],offs,arr
def iou(a,b):
    ix1,iy1=max(a[0],b[0]),max(a[1],b[1]); ix2,iy2=min(a[2],b[2]),min(a[3],b[3]); iw,ih=max(0,ix2-ix1),max(0,iy2-iy1); it=iw*ih
    ua=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-it; return it/ua if ua>0 else 0.0
def nms(c,md):
    c=sorted(c,key=lambda z:-z[2]); k=[]
    for cx,cy,s in c:
        if all((cx-kx)**2+(cy-ky)**2>=md*md for kx,ky,_ in k): k.append((cx,cy,s))
    return k
fields=sorted(fastmal_fields(),key=lambda z:str(z[0])); random.Random(0).shuffle(fields)
devF=fields[0::2][:20]; bs=CAL['box_size_px']; md=CAL['peak_min_dist_px']
# gate/cam thresholds on dev (same as v3)
gl=[];gh=[];cv=[];cl=[]
for ip,gtb in devF:
    try: im=Image.open(ip).convert('RGB')
    except Exception: continue
    cents=[(x+w/2,y+h/2) for (x,y,w,h) in gtb]; crops,offs,_=toff(im); lo,ca=logit_cam(crops)
    for (tx,ty),lg,cm in zip(offs,lo,ca):
        has=any(tx<=cx<tx+TILE and ty<=cy<ty+TILE for (cx,cy) in cents); gl.append(lg); gh.append(1 if has else 0)
        if has:
            for gy in range(G):
                for gx in range(G):
                    px,py=tx+gx*P+P/2,ty+gy*P+P/2; cv.append(cm[gy*G+gx]); cl.append(1 if any((px-cx)**2+(py-cy)**2<=(P*1.5)**2 for (cx,cy) in cents) else 0)
f,t,th=roc_curve(gh,gl); gate_thr=float(th[np.argmax(t-f)]); f2,t2,th2=roc_curve(cl,cv); cam_thr=float(th2[np.argmax(t2-f2)])
# generate A0 boxes + MC on DEV fields
def gen_mc(fset):
    PM=[];PV=[];SC=[];TP=[]; n_gt=0
    for ip,gtb in fset:
        try: im=Image.open(ip).convert('RGB')
        except Exception: continue
        gt=[(x,y,x+w,y+h) for (x,y,w,h) in gtb]; n_gt+=len(gt); crops,offs,arr=toff(im); lo,ca=logit_cam(crops); cand=[]
        for (tx,ty),lg,cm in zip(offs,lo,ca):
            if lg<gate_thr: continue
            sg=cm.reshape(G,G)
            for gy in range(G):
                for gx in range(G):
                    if sg[gy,gx]>=cam_thr: cand.append((tx+gx*P+P/2,ty+gy*P+P/2,float(sg[gy,gx])))
        peaks=nms(cand,md)
        if not peaks: continue
        H,W=arr.shape[:2]; bc=[]
        for (cx,cy,s) in peaks:
            x0=int(min(max(cx-TILE/2,0),W-TILE)); y0=int(min(max(cy-TILE/2,0),H-TILE)); bc.append(arr[y0:y0+TILE,x0:x0+TILE])
        pm,pv=mc(bc); boxes=[(cx-bs/2,cy-bs/2,cx+bs/2,cy+bs/2) for (cx,cy,s) in peaks]
        order=sorted(range(len(peaks)),key=lambda j:-peaks[j][2]); matched=set()
        for j in order:
            b=boxes[j]; best=-1;bj=-1
            for jg,g in enumerate(gt):
                if jg in matched: continue
                v=iou(b,g)
                if v>best: best,bj=v,jg
            tp=1 if best>=0.3 and bj>=0 else 0
            if tp: matched.add(bj)
            PM.append(pm[j]); PV.append(pv[j]); SC.append(peaks[j][2]); TP.append(tp)
    return np.array(PM),np.array(PV),np.array(SC),np.array(TP),n_gt
dPM,dPV,dSC,dTP,d_ngt=gen_mc(devF)
np.savez(RES/'perbox_dev.npz',p_mean=dPM,p_var=dPV,score=dSC,is_tp=dTP,n_gt=d_ngt)
# ---- DEV calibration ----
dfaint=(dTP==1)&(dPM<=np.quantile(dPM[dTP==1],1/3))
b_protect=float(np.quantile(dPV[dfaint],0.10))    # protect >=90% of dev faint TPs via var clause
def metrics(PM,PV,SC,TP,n_gt,keep):
    tp=TP[keep]; sc=SC[keep]; TPn=int(tp.sum()); FP=len(tp)-TPn; Pp=TPn/max(1,TPn+FP); Rr=TPn/max(1,n_gt)
    o=np.argsort(-sc); tps=tp[o]; ctp=np.cumsum(tps); cfp=np.cumsum(1-tps); rec=ctp/max(1,n_gt); pre=ctp/np.maximum(1,ctp+cfp)
    mr=np.concatenate([[0],rec,[1]]); mp=np.concatenate([[0],pre,[0]])
    for i in range(len(mp)-1,0,-1): mp[i-1]=max(mp[i-1],mp[i])
    idx=np.where(mr[1:]!=mr[:-1])[0]; ap=float(np.sum((mr[idx+1]-mr[idx])*mp[idx+1])); f1=2*Pp*Rr/max(1e-9,Pp+Rr)
    return Pp,Rr,ap,f1,TPn,FP
# choose a* on dev: sweep a over dev p̄ quantiles, max dev F1
a_grid=np.quantile(dPM,np.linspace(0.2,0.95,16)); best=(-1,None)
for a in a_grid:
    keep=(dPV>=b_protect)|(dPM>=a); Pp,Rr,ap,f1,_,_=metrics(dPM,dPV,dSC,dTP,d_ngt,keep)
    if f1>best[0]: best=(f1,float(a))
a_star=best[1]
print(f'gate_thr {gate_thr:.3f} cam_thr {cam_thr:.3f} | b_protect {b_protect:.5f} | a* {a_star:.4f} (dev F1 {best[0]:.3f})',flush=True)
# ---- TEST eval (load S3.2 perbox.npz) ----
z=np.load(RES/'perbox.npz'); tPM,tPV,tSC,tTP,t_ngt=z['p_mean'],z['p_var'],z['score'],z['is_tp'],int(z['n_gt'])
tfaint=(tTP==1)&(tPM<=np.quantile(tPM[tTP==1],1/3))
def row(keep):
    Pp,Rr,ap,f1,TPn,FP=metrics(tPM,tPV,tSC,tTP,t_ngt,keep); fr=round(float(keep[tfaint].mean()),3) if tfaint.sum()>0 else None
    return dict(P=round(Pp,3),R=round(Rr,3),AP=round(ap,3),F1=round(f1,3),faint_TP_retained=fr,TP=TPn,FP=FP,kept=int(keep.sum()))
A0=row(np.ones(len(tTP),bool))
protected_star=row((tPV>=b_protect)|(tPM>=a_star))
# protected sweep (a) for the curve
psweep=[]
for a in np.quantile(tPM,np.linspace(0.2,0.95,16)):
    psweep.append({**row((tPV>=b_protect)|(tPM>=a)),'a':round(float(a),4)})
# naive sweep for comparison
nsweep=[]
for q in [0.9,0.8,0.7,0.6,0.5,0.4,0.3]:
    nsweep.append({**row(tPV<=np.quantile(tPV,q)),'keep':q})
out=dict(calib=dict(b_protect=round(b_protect,5),a_star=round(a_star,4),dev_F1=round(best[0],3),
                    dev_faint_n=int(dfaint.sum()),gate_thr=round(gate_thr,3),cam_thr=round(cam_thr,3)),
         A0=A0, protected_at_dev_astar=protected_star, protected_sweep=psweep, naive_sweep=nsweep)
json.dump(out,open(RES/'final.json','w'),indent=2); print(json.dumps({k:out[k] for k in ['calib','A0','protected_at_dev_astar']},indent=2),flush=True)
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({'font.family':'serif'})
fig,ax=plt.subplots(1,2,figsize=(12,4.6))
ax[0].plot([m['R'] for m in nsweep],[m['P'] for m in nsweep],'-s',color='#ee6677',label='naive variance')
ax[0].plot([m['R'] for m in psweep],[m['P'] for m in psweep],'-o',color='#228833',label='faintness-protected')
ax[0].scatter([A0['R']],[A0['P']],c='black',zorder=5,label='A0 (no filter)')
ax[0].scatter([protected_star['R']],[protected_star['P']],marker='*',s=220,c='#117733',zorder=6,edgecolors='k',label='protected @ dev-chosen a*')
ax[0].set_xlabel('recall'); ax[0].set_ylabel('precision'); ax[0].set_title('Pseudo-box PR (test): A0 vs naive vs protected (dev-calibrated)'); ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3)
ax[1].plot([m['R'] for m in nsweep],[m['faint_TP_retained'] for m in nsweep],'-s',color='#ee6677',label='naive')
ax[1].plot([m['R'] for m in psweep],[m['faint_TP_retained'] for m in psweep],'-o',color='#228833',label='protected')
ax[1].axhline(0.9,ls=':',c='gray'); ax[1].set_xlabel('overall recall'); ax[1].set_ylabel('faint (low-parasitaemia) TP retained'); ax[1].set_title('Low-parasitaemia protection (test)'); ax[1].legend(); ax[1].grid(alpha=0.3); ax[1].set_ylim(0,1.05)
plt.tight_layout(); plt.savefig(RES/'fig_s3_final.png',dpi=130,bbox_inches='tight')
print('S3_FINAL_DONE',flush=True)
