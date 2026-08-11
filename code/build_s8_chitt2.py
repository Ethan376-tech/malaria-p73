# -*- coding: utf-8 -*-
"""Stage 2 — the OTHER train/test design (domain B): generate pseudo-labels on Chittagong-1 with the chitt FT
generator, train RetinaNet, test detection AP on the held-out Chittagong-2 (GT_updated). + GT-supervised reference
(Chittagong-1 real boxes). Expected resolution-limited (19px parasites < 16px patch grid). Output -> s8_chitt2/."""
import os, sys, json, glob, shutil, random
os.environ['XFORMERS_DISABLED']='1'; os.environ['PYTORCH_CUDA_ALLOC_CONF']='expandable_segments:True'
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import torchvision.transforms.functional as TF
import torchvision.transforms as T
from PIL import Image; Image.MAX_IMAGE_PIXELS=None
from torchvision.models.detection import retinanet_resnet50_fpn
sys.path.insert(0,'/root/autodl-tmp/A_Dissertation/references/RedDino/train'); sys.path.append('/root/A_Dissertation')
from common.paths import CHITTAGONG1, CHITTAGONG2, CHECKPOINTS, OUTPUTS
from dinov2.models.vision_transformer import vit_small
from sklearn.metrics import roc_curve
dev='cuda'; TILE=224; P=16; G=TILE//P; CROP=640; EPOCHS=40; BASE_LR=0.005; WARMUP=400; BATCH=6
random.seed(0); np.random.seed(0); torch.manual_seed(0)
CAL=json.load(open(OUTPUTS/'experiments'/'s2_calib'/'calib_constants.json'))['B']; BS=CAL['box_size_px']; MD=CAL['peak_min_dist_px']
RES=OUTPUTS/'experiments'/'s8_chitt2'; RES.mkdir(parents=True,exist_ok=True)
norm=T.Compose([T.ToTensor(),T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
genc=vit_small(patch_size=16,img_size=224,block_chunks=0,init_values=1.0,drop_path_rate=0.1)
genc.load_state_dict(torch.load(CHECKPOINTS/'ssl_vits_thickdino_ft_chitt'/'encoder.pth',map_location='cpu'),strict=False); genc=genc.to(dev).eval()
ghead=nn.Sequential(nn.Dropout(0.2),nn.Linear(384,1)).to(dev); ghead.load_state_dict(torch.load(CHECKPOINTS/'ssl_vits_thickdino_ft_chitt'/'head.pth',map_location='cpu')); ghead.eval()
acts={}
def hook(mod,i,o):
    if o.requires_grad: o.retain_grad()
    acts['o']=o
genc.blocks[-4].register_forward_hook(hook)
def toff(im):
    W,H=im.size; arr=np.asarray(im); xs=sorted(set(list(range(0,max(1,W-TILE+1),TILE))+[max(0,W-TILE)])); ys=sorted(set(list(range(0,max(1,H-TILE+1),TILE))+[max(0,H-TILE)]))
    return [(arr[ty:ty+TILE,tx:tx+TILE],(tx,ty)) for ty in ys for tx in xs],arr,W,H
def gen_logit_cam(crops,bs=24):
    lo=[];ca=[]
    for i in range(0,len(crops),bs):
        xb=torch.stack([norm(Image.fromarray(c)) for c in crops[i:i+bs]]).to(dev).requires_grad_(True)
        ff=genc.forward_features(xb); s=ghead(ff['x_norm_clstoken']).squeeze(-1); genc.zero_grad(); ghead.zero_grad(); s.sum().backward()
        o=acts['o']; al=o.grad[:,1:,:].mean(1,keepdim=True); cam=F.relu((o[:,1:,:].detach()*al).sum(-1))
        lo.append(s.detach().cpu().numpy()); ca.append(cam.detach().cpu().numpy())
    return np.concatenate(lo),np.concatenate(ca)
def nms(c,md):
    c=sorted(c,key=lambda z:-z[2]); k=[]
    for cx,cy,s in c:
        if all((cx-kx)**2+(cy-ky)**2>=md*md for kx,ky,_ in k): k.append((cx,cy,s))
    return k
# ---- Chittagong-1 fields (+GT) and Chittagong-2 fields (+GT) ----
def chitt1_fields():
    ANN=CHITTAGONG1/'NIH-NLM-ThickBloodSmearsPV'/'All_annotations'; IMG=CHITTAGONG1/'NIH-NLM-ThickBloodSmearsPV'/'All_PvTk'; o=[]
    for pv in sorted(d for d in ANN.iterdir() if d.is_dir()):
        for txt in sorted(pv.glob('*.txt')):
            bx=[]
            for ln in txt.read_text(errors='ignore').splitlines()[1:]:
                f=ln.split(',')
                if len(f)>=9 and f[1].strip()=='Parasitized':
                    try: x1,y1,x2,y2=map(float,f[5:9]); bx.append((x1,y1,x2-x1,y2-y1))
                    except ValueError: pass
            ip=IMG/pv.name/(txt.stem+'.jpg')
            if bx and ip.exists(): o.append((ip,bx))
    return o
def chitt2_fields():
    GT=CHITTAGONG2/'GT_updated'; o=[]
    for tf in sorted(d for d in CHITTAGONG2.iterdir() if d.is_dir() and d.name!='GT_updated'):
        for ip in sorted(tf.glob('*.jpg')):
            bf=GT/tf.name/(ip.stem+'.txt')
            if not bf.exists(): continue
            bx=[]
            for ln in bf.read_text(errors='ignore').splitlines()[1:]:
                f=ln.split(',')
                if len(f)>=9 and 'Parasite' in f[1]:
                    try: x1,y1,x2,y2=map(float,f[5:9]); bx.append((x1,y1,x2-x1,y2-y1))
                    except ValueError: pass
            if bx: o.append((ip,bx))
    return o
c1=sorted(chitt1_fields(),key=lambda z:str(z[0])); random.Random(0).shuffle(c1)
c2=sorted(chitt2_fields(),key=lambda z:str(z[0])); random.Random(0).shuffle(c2)
devC1=c1[:15]; trainC1=c1[15:55]; testC2=c2[:40]
print(f'chitt1 fields {len(c1)} | chitt2 fields {len(c2)} | train {len(trainC1)} test {len(testC2)}',flush=True)
# calibrate gate/cam thresholds on Chittagong-1 dev
gl=[];gh=[];cv=[];cl=[]
for ip,gtb in devC1:
    try: im=Image.open(ip).convert('RGB')
    except Exception: continue
    cents=[(x+w/2,y+h/2) for (x,y,w,h) in gtb]; tiles,_,_,_=toff(im); crops=[c for c,_ in tiles]; offs=[o for _,o in tiles]; lo,ca=gen_logit_cam(crops)
    for (tx,ty),lg,cm in zip(offs,lo,ca):
        has=any(tx<=cx<tx+TILE and ty<=cy<ty+TILE for (cx,cy) in cents); gl.append(lg); gh.append(1 if has else 0)
        if has:
            for gy in range(G):
                for gx in range(G):
                    px,py=tx+gx*P+P/2,ty+gy*P+P/2; cv.append(cm[gy*G+gx]); cl.append(1 if any((px-cx)**2+(py-cy)**2<=(P*1.5)**2 for (cx,cy) in cents) else 0)
gh=np.array(gh); gl=np.array(gl); cl=np.array(cl); cv=np.array(cv)
gate_thr=float(roc_curve(gh,gl)[2][np.argmax(roc_curve(gh,gl)[1]-roc_curve(gh,gl)[0])]) if len(set(gh))>1 else -1e9
f2,t2,th2=roc_curve(cl,cv); cam_thr=float(th2[np.argmax(t2-f2)]) if len(set(cl))>1 else 0.0
print(f'B gate_thr {gate_thr:.3f} cam_thr {cam_thr:.3f} | box {BS} dist {MD}',flush=True)
def pseudo_boxes(im):
    tiles,arr,W,H=toff(im); crops=[c for c,_ in tiles]; offs=[o for _,o in tiles]; lo,ca=gen_logit_cam(crops); cand=[]
    for (tx,ty),lg,cm in zip(offs,lo,ca):
        if lg<gate_thr: continue
        sg=cm.reshape(G,G)
        for gy in range(G):
            for gx in range(G):
                if sg[gy,gx]>=cam_thr: cand.append((tx+gx*P+P/2,ty+gy*P+P/2,float(sg[gy,gx])))
    return [(cx,cy) for (cx,cy,s) in sorted(nms(cand,MD),key=lambda z:-z[2])[:400]]
# ---- write YOLO datasets ----
def crop_grid(W,H):
    xs=list(range(0,max(1,W-CROP+1),CROP))+([W-CROP] if W>CROP else [0]); ys=list(range(0,max(1,H-CROP+1),CROP))+([H-CROP] if H>CROP else [0]); return sorted(set(xs)),sorted(set(ys))
def write(im,boxes_xywh,root,split,tag):
    arr=np.asarray(im); H,W=arr.shape[:2]; xs,ys=crop_grid(W,H); (root/'images'/split).mkdir(parents=True,exist_ok=True); (root/'labels'/split).mkdir(parents=True,exist_ok=True); n=0
    for cxo in xs:
        for cyo in ys:
            sub=arr[cyo:cyo+CROP,cxo:cxo+CROP]
            if sub.shape[0]<CROP or sub.shape[1]<CROP:
                pad=np.zeros((CROP,CROP,3),np.uint8); pad[:sub.shape[0],:sub.shape[1]]=sub; sub=pad
            labs=[]
            for (cx,cy,w,h) in boxes_xywh:
                lx,ly=cx-cxo,cy-cyo
                if 0<=lx<CROP and 0<=ly<CROP: labs.append(f"0 {lx/CROP:.6f} {ly/CROP:.6f} {w/CROP:.6f} {h/CROP:.6f}")
            if not labs and split=='train': continue
            Image.fromarray(sub).save(root/'images'/split/f'{tag}_{cxo}_{cyo}.jpg',quality=92); (root/'labels'/split/f'{tag}_{cxo}_{cyo}.txt').write_text("\n".join(labs)); n+=1
    return n
DP=RES/'ds_pseudo'; DG=RES/'ds_gtsup'
if len(glob.glob(str(DP/'images'/'train'/'*.jpg')))==0:
    for d in (DP,DG):
        if d.exists(): shutil.rmtree(d)
    np_=ng=0
    for k,(ip,gtb) in enumerate(trainC1):
        try: im=Image.open(ip).convert('RGB')
        except Exception: continue
        pb=pseudo_boxes(im); np_+=write(im,[(cx,cy,BS,BS) for (cx,cy) in pb],DP,'train',f'c1_{k}')
        ng+=write(im,[(x+w/2,y+h/2,w,h) for (x,y,w,h) in gtb],DG,'train',f'c1_{k}')
    nt=0
    for k,(ip,gtb) in enumerate(testC2):
        try: im=Image.open(ip).convert('RGB')
        except Exception: continue
        boxes=[(x+w/2,y+h/2,w,h) for (x,y,w,h) in gtb]
        for d in (DP,DG): write(im,boxes,d,'val',f'c2_{k}')
        nt+=1
    print(f'train crops: pseudo {np_} | gtsup {ng} | test C2 fields {nt}',flush=True)
else:
    print('reusing existing crops',flush=True)
# ---- RetinaNet train/eval ----
class DetDS(torch.utils.data.Dataset):
    def __init__(self,root,split): self.imgs=sorted(glob.glob(str(root/'images'/split/'*.jpg')))
    def __len__(self): return len(self.imgs)
    def __getitem__(self,i):
        ip=self.imgs[i]; x=TF.to_tensor(Image.open(ip).convert('RGB')); lp=ip.replace('/images/','/labels/').replace('.jpg','.txt'); bx=[]
        if os.path.exists(lp):
            for ln in open(lp).read().splitlines():
                f=ln.split()
                if len(f)==5: _,cx,cy,w,h=f; cx,cy,w,h=float(cx)*CROP,float(cy)*CROP,float(w)*CROP,float(h)*CROP; bx.append([cx-w/2,cy-h/2,cx+w/2,cy+h/2])
        bx=torch.tensor(bx,dtype=torch.float32).reshape(-1,4)
        if len(bx):
            bx[:,[0,2]]=bx[:,[0,2]].clamp(0,CROP); bx[:,[1,3]]=bx[:,[1,3]].clamp(0,CROP)
            keep=(bx[:,2]-bx[:,0]>=1)&(bx[:,3]-bx[:,1]>=1); bx=bx[keep]
        return x,{'boxes':bx,'labels':torch.ones(len(bx),dtype=torch.int64)}
def collate(b): return tuple(zip(*b))
def build(): return retinanet_resnet50_fpn(num_classes=2,weights=None,weights_backbone='IMAGENET1K_V1').to(dev)
def train(model,root):
    dl=torch.utils.data.DataLoader(DetDS(root,'train'),batch_size=BATCH,shuffle=True,collate_fn=collate,num_workers=6)
    if len(dl)==0: return model
    params=[p for p in model.parameters() if p.requires_grad]; opt=torch.optim.SGD(params,lr=BASE_LR,momentum=0.9,weight_decay=1e-4)
    sched=torch.optim.lr_scheduler.MultiStepLR(opt,milestones=[25,35],gamma=0.1); it=0
    for ep in range(EPOCHS):
        model.train()
        for imgs,tgts in dl:
            if it<WARMUP:
                f=it/WARMUP
                for g in opt.param_groups: g['lr']=BASE_LR*(0.01+0.99*f)
            imgs=[im.to(dev) for im in imgs]; tgts=[{k:v.to(dev) for k,v in t.items()} for t in tgts]
            loss=sum(model(imgs,tgts).values()); opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(params,5.0); opt.step(); it+=1
        if it>=WARMUP: sched.step()
    return model
def iou(a,b):
    ix1,iy1=max(a[0],b[0]),max(a[1],b[1]); ix2,iy2=min(a[2],b[2]),min(a[3],b[3]); iw,ih=max(0,ix2-ix1),max(0,iy2-iy1); it=iw*ih
    ua=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-it; return it/ua if ua>0 else 0.0
@torch.no_grad()
def evaluate(model,root):
    dl=torch.utils.data.DataLoader(DetDS(root,'val'),batch_size=BATCH,shuffle=False,collate_fn=collate,num_workers=6); model.eval()
    all_tp={0.3:[],0.5:[]}; sc=[]; ngt=0
    for imgs,tgts in dl:
        preds=model([im.to(dev) for im in imgs])
        for pred,tgt in zip(preds,tgts):
            gt=tgt['boxes'].numpy().tolist(); ngt+=len(gt); bxs=pred['boxes'].cpu().numpy(); scs=pred['scores'].cpu().numpy(); order=np.argsort(-scs)
            for iouth in (0.3,0.5):
                matched=set()
                for j in order:
                    b=bxs[j]; best=-1;bj=-1
                    for jg,g in enumerate(gt):
                        if jg in matched: continue
                        v=iou(b,g)
                        if v>best: best,bj=v,jg
                    tp=1 if best>=iouth and bj>=0 else 0
                    if tp: matched.add(bj)
                    all_tp[iouth].append(tp)
                    if iouth==0.3: sc.append(float(scs[j]))
    def ap(tps,scs,ng):
        if not tps: return 0.0
        o=np.argsort(-np.array(scs)); tps=np.array(tps)[o]; ctp=np.cumsum(tps); cfp=np.cumsum(1-tps); rec=ctp/max(1,ng); pre=ctp/np.maximum(1,ctp+cfp)
        mr=np.concatenate([[0],rec,[1]]); mp=np.concatenate([[0],pre,[0]])
        for i in range(len(mp)-1,0,-1): mp[i-1]=max(mp[i-1],mp[i])
        idx=np.where(mr[1:]!=mr[:-1])[0]; return float(np.sum((mr[idx+1]-mr[idx])*mp[idx+1]))
    return dict(**{f'AP@{t}':round(ap(all_tp[t],sc,ngt),3) for t in (0.3,0.5)}, n_gt=ngt)
res={}
for nm,root in [('pseudo_chitt1',DP),('gtsup_chitt1',DG)]:
    ntr=len(glob.glob(str(root/'images'/'train'/'*.jpg'))); print(f'=== {nm} (train crops {ntr}) ===',flush=True)
    if ntr==0: res[nm]={'AP@0.3':0.0,'AP@0.5':0.0,'note':'no train crops (pseudo-labels empty)'}; print('  -> empty'); continue
    m=train(build(),root); r=evaluate(m,root); res[nm]=r; print('  ->',r,flush=True)
json.dump(res,open(RES/'chitt2_results.json','w'),indent=2); print('\n=== Stage2 Chittagong-2 detection AP ==='); print(json.dumps(res,indent=2)); print('CHITT2_DONE',flush=True)
