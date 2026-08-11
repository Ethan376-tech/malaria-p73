# -*- coding: utf-8 -*-
"""(iii) B-domain generation + BY-PARASITAEMIA-BAND reporting. For A (Ibadan/FASTMAL, clean) and B (Chittagong-1,
caveats: 19px parasites < 16px patch grid -> resolution-limited; annotated fields overlap bag samples -> in-domain
overlap), run the in-domain-FT-ThickDINO generator (gate + Grad-CAM) + MC-dropout + dev-calibrated faintness-protected
filter, tracking each box's field GT-count. Stratify test fields into LOW vs HIGH parasite-density bands (median GT
count) and report A0 vs protected recall/precision per band — the key low-parasitaemia check. Output -> s3_filter/byband.json (+ fig)."""
import os, sys, json, random
os.environ['XFORMERS_DISABLED']='1'
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image; Image.MAX_IMAGE_PIXELS=None
sys.path.insert(0,'/root/autodl-tmp/A_Dissertation/references/RedDino/train'); sys.path.append('/root/A_Dissertation')
from common.paths import RAW, CHITTAGONG1, CHECKPOINTS, OUTPUTS
from sklearn.metrics import roc_curve
dev='cuda'; TILE=224; P=16; G=TILE//P; T_MC=30
random.seed(0); np.random.seed(0); torch.manual_seed(0)
CAL=json.load(open(OUTPUTS/'experiments'/'s2_calib'/'calib_constants.json'))
RES=OUTPUTS/'experiments'/'s3_filter'
norm=T.Compose([T.ToTensor(),T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
from dinov2.models.vision_transformer import vit_small
def load_ft(ckpt):
    m=vit_small(patch_size=16,img_size=224,block_chunks=0,init_values=1.0,drop_path_rate=0.1)
    m.load_state_dict(torch.load(CHECKPOINTS/ckpt/'encoder.pth',map_location='cpu'),strict=False); m=m.to(dev).eval()
    h=nn.Sequential(nn.Dropout(0.2),nn.Linear(384,1)).to(dev); h.load_state_dict(torch.load(CHECKPOINTS/ckpt/'head.pth',map_location='cpu')); h.eval()
    return m,h
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
def chitt_fields():
    ANN=CHITTAGONG1/'NIH-NLM-ThickBloodSmearsPV'/'All_annotations'; IMG=CHITTAGONG1/'NIH-NLM-ThickBloodSmearsPV'/'All_PvTk'; o=[]
    for pv in sorted(d for d in ANN.iterdir() if d.is_dir()):
        for txt in sorted(pv.glob('*.txt')):
            boxes=[]
            for ln in txt.read_text(errors='ignore').splitlines()[1:]:
                f=ln.split(',')
                if len(f)>=9 and f[1].strip()=='Parasitized':
                    try: x1,y1,x2,y2=map(float,f[5:9]); boxes.append((x1,y1,x2-x1,y2-y1))
                    except ValueError: pass
            ip=IMG/pv.name/(txt.stem+'.jpg')
            if boxes and ip.exists(): o.append((ip,boxes))
    return o
def toff(im):
    W,H=im.size; arr=np.asarray(im); xs=sorted(set(list(range(0,max(1,W-TILE+1),TILE))+[max(0,W-TILE)])); ys=sorted(set(list(range(0,max(1,H-TILE+1),TILE))+[max(0,H-TILE)]))
    return [(arr[ty:ty+TILE,tx:tx+TILE],(tx,ty)) for ty in ys for tx in xs],arr
def iou(a,b):
    ix1,iy1=max(a[0],b[0]),max(a[1],b[1]); ix2,iy2=min(a[2],b[2]),min(a[3],b[3]); iw,ih=max(0,ix2-ix1),max(0,iy2-iy1); it=iw*ih
    ua=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-it; return it/ua if ua>0 else 0.0
def nms(c,md):
    c=sorted(c,key=lambda z:-z[2]); k=[]
    for cx,cy,s in c:
        if all((cx-kx)**2+(cy-ky)**2>=md*md for kx,ky,_ in k): k.append((cx,cy,s))
    return k

def run_domain(name, ckpt, fields_fn, cal, n_test):
    enc,head=load_ft(ckpt); acts={}
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
    bs=cal['box_size_px']; md=cal['peak_min_dist_px']
    fields=sorted(fields_fn(),key=lambda z:str(z[0])); random.Random(0).shuffle(fields)
    devF=fields[0::2][:20]; testF=fields[1::2][:n_test]
    # gate/cam dev calibration
    gl=[];gh=[];cv=[];cl=[]
    for ip,gtb in devF:
        try: im=Image.open(ip).convert('RGB')
        except Exception: continue
        cents=[(x+w/2,y+h/2) for (x,y,w,h) in gtb]; tiles,_=toff(im); crops=[c for c,_ in tiles]; offs=[o for _,o in tiles]; lo,ca=logit_cam(crops)
        for (tx,ty),lg,cm in zip(offs,lo,ca):
            has=any(tx<=cx<tx+TILE and ty<=cy<ty+TILE for (cx,cy) in cents); gl.append(lg); gh.append(1 if has else 0)
            if has:
                for gy in range(G):
                    for gx in range(G):
                        px,py=tx+gx*P+P/2,ty+gy*P+P/2; cv.append(cm[gy*G+gx]); cl.append(1 if any((px-cx)**2+(py-cy)**2<=(P*1.5)**2 for (cx,cy) in cents) else 0)
    f,t,th=roc_curve(gh,gl); gate_thr=float(th[np.argmax(t-f)]); f2,t2,th2=roc_curve(cl,cv); cam_thr=float(th2[np.argmax(t2-f2)])
    # generate + MC on a field set; track field gt count
    def gen(fset):
        PM=[];PV=[];SC=[];TP=[];FC=[]; ngt=0; field_counts=[]
        for ip,gtb in fset:
            try: im=Image.open(ip).convert('RGB')
            except Exception: continue
            gt=[(x,y,x+w,y+h) for (x,y,w,h) in gtb]; ngt+=len(gt); fcount=len(gt); field_counts.append(fcount)
            tiles,arr=toff(im); crops=[c for c,_ in tiles]; offs=[o for _,o in tiles]; lo,ca=logit_cam(crops); cand=[]
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
                PM.append(pm[j]); PV.append(pv[j]); SC.append(peaks[j][2]); TP.append(tp); FC.append(fcount)
        return dict(pm=np.array(PM),pv=np.array(PV),sc=np.array(SC),tp=np.array(TP),fc=np.array(FC),ngt=ngt,field_counts=np.array(field_counts))
    devset=gen(devF); test=gen(testF)
    # dev-calibrate b_protect, a*
    dfa=(devset['tp']==1)&(devset['pm']<=np.quantile(devset['pm'][devset['tp']==1],1/3)) if (devset['tp']==1).sum()>3 else (devset['tp']==1)
    b_protect=float(np.quantile(devset['pv'][dfa],0.10)) if dfa.sum()>0 else float(np.median(devset['pv']))
    def m(pm,pv,sc,tp,ngt,keep):
        t=tp[keep]; TPn=int(t.sum()); FP=len(t)-TPn; return TPn/max(1,TPn+FP), TPn/max(1,ngt), TPn, FP
    a_grid=np.quantile(devset['pm'],np.linspace(0.2,0.95,16)); best=(-1,None)
    for a in a_grid:
        keep=(devset['pv']>=b_protect)|(devset['pm']>=a); Pp,Rr,_,_=m(devset['pm'],devset['pv'],devset['sc'],devset['tp'],devset['ngt'],keep); f1=2*Pp*Rr/max(1e-9,Pp+Rr)
        if f1>best[0]: best=(f1,float(a))
    a_star=best[1]
    # TEST: A0 and protected; BY BAND (field gt-count median split). Band recall uses band GT total.
    fc=test['fc']; fcounts=test['field_counts']; thrband=float(np.median(fcounts)) if len(fcounts)>0 else 0.0
    GT_low=float(fcounts[fcounts<=thrband].sum()); GT_high=float(fcounts[fcounts>thrband].sum()); GT_all=float(fcounts.sum())
    def band_metrics(keep):
        res={}
        for bn,mask,gtb in [('low_density',fc<=thrband,GT_low),('high_density',fc>thrband,GT_high),('all',np.ones(len(fc),bool),GT_all)]:
            mm=mask&keep; TPn=int(test['tp'][mm].sum()); FP=int((1-test['tp'][mm]).sum())
            res[bn]=dict(TP=TPn,FP=FP,GT=int(gtb),P=round(TPn/max(1,TPn+FP),3),R=round(TPn/max(1,gtb),3))
        return res
    keepA0=np.ones(len(test['tp']),bool); keepP=(test['pv']>=b_protect)|(test['pm']>=a_star)
    # band GT counts (distinct fields): approximate from unique field gt — track via fc on TP+FP boxes is per-box; instead compute band recall using A0 TP as denom proxy
    out=dict(domain=name,gate_thr=round(gate_thr,3),cam_thr=round(cam_thr,3),b_protect=round(b_protect,6),a_star=round(a_star,4),
             n_test_fields=len(testF),density_split=float(thrband),
             A0=band_metrics(keepA0), protected=band_metrics(keepP))
    # faint retention per band (faint = low-pm tertile of test TP)
    tpm=test['tp']==1
    if tpm.sum()>3:
        ft=tpm&(test['pm']<=np.quantile(test['pm'][tpm],1/3))
        for bn,mask in [('low_density',fc<=thrband),('high_density',fc>thrband)]:
            sel=ft&mask
            if sel.sum()>0:
                out.setdefault('faint_retained_protected',{})[bn]=round(float(keepP[sel].mean()),3)
    print(json.dumps(out,indent=2),flush=True); return out

import torch as _t
RA=run_domain('A_ibadan_fastmal','ssl_vits_thickdino_ft_ibadan',fastmal_fields,CAL['A'],40)
RB=None
if (CHECKPOINTS/'ssl_vits_thickdino_ft_chitt'/'encoder.pth').exists():
    RB=run_domain('B_chittagong1','ssl_vits_thickdino_ft_chitt',chitt_fields,CAL['B'],25)
json.dump({'A':RA,'B':RB},open(RES/'byband.json','w'),indent=2)
print('BYBAND_DONE',flush=True)
