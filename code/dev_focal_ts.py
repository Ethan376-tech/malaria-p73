# -*- coding: utf-8 -*-
"""Step 1 (§2.4, Pillar 3) — Teacher–student semi-supervised detection, extending MILCA's one-shot bootstrap to a
continuous EMA-teacher refinement, with the §2.3 faintness-protected retain integrated (Pillar 2 ⊕ Pillar 3).
Student starts from the protected-pseudo RetinaNet; teacher = EMA(student). Periodically the teacher regenerates
refined pseudo-labels on the train crops; the student trains on them. Two retain rules compared: naive conf>0.7 vs
ours (FT-ThickDINO MC-dropout, faintness-protected). Eval the final teacher on FASTMAL (AP@0.3/0.5 + low-density band).
Output -> outputs/experiments/s9_teacher_student/."""
import os, sys, json, glob, shutil, copy
os.environ['XFORMERS_DISABLED']='1'; os.environ['PYTORCH_CUDA_ALLOC_CONF']='expandable_segments:True'
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import torchvision.transforms.functional as TF
import torchvision.transforms as T
from PIL import Image; Image.MAX_IMAGE_PIXELS=None
from torchvision.models.detection import retinanet_resnet50_fpn
import torchvision.models.detection.retinanet as _R
_orig_sfl=_R.sigmoid_focal_loss
import sys as _sys
GAMMA=float(_sys.argv[1]) if len(_sys.argv)>1 else 2.0
def _sfl(inputs,targets,alpha=0.25,gamma=2,reduction='none'): return _orig_sfl(inputs,targets,alpha=alpha,gamma=GAMMA,reduction=reduction)
_R.sigmoid_focal_loss=_sfl
sys.path.insert(0,'/root/autodl-tmp/A_Dissertation/references/RedDino/train'); sys.path.append('/root/A_Dissertation')
from common.paths import OUTPUTS, CHECKPOINTS
from dinov2.models.vision_transformer import vit_small
dev='cuda'; torch.manual_seed(0); np.random.seed(0)
CROP=640; TILE=224; T_MC=15; EPOCHS=10; REFRESH=[0,5]; EMA=0.95; LR=0.002; BATCH=6; MC_CAP=40
YO=OUTPUTS/'experiments'/'s5_yolo'; S6=OUTPUTS/'experiments'/'s6_retinanet'; RES=OUTPUTS/'experiments'/'s9_teacher_student'; RES.mkdir(parents=True,exist_ok=True)
FIN=json.load(open(OUTPUTS/'experiments'/'s3_filter'/'final.json'))['calib']; b_protect=FIN['b_protect']; a_star=FIN['a_star']
norm=T.Compose([T.ToTensor(),T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
def build_ret(): return retinanet_resnet50_fpn(num_classes=2,weights=None,weights_backbone='IMAGENET1K_V1').to(dev)
# FT-ThickDINO MC (faintness signal)
ftm=vit_small(patch_size=16,img_size=224,block_chunks=0,init_values=1.0,drop_path_rate=0.1)
ftm.load_state_dict(torch.load(CHECKPOINTS/'ssl_vits_thickdino_ft_ibadan'/'encoder.pth',map_location='cpu'),strict=False); ftm=ftm.to(dev).eval()
fth=nn.Sequential(nn.Dropout(0.2),nn.Linear(384,1)).to(dev); fth.load_state_dict(torch.load(CHECKPOINTS/'ssl_vits_thickdino_ft_ibadan'/'head.pth',map_location='cpu')); fth.eval()
@torch.no_grad()
def mc_boxes(tiles,bs=48):
    for mod in list(ftm.modules())+list(fth.modules()):
        if isinstance(mod,nn.Dropout): mod.train()
    for mod in ftm.modules():
        if mod.__class__.__name__=='DropPath': mod.train()
    M=[];V=[]
    for i in range(0,len(tiles),bs):
        xb=torch.stack([norm(Image.fromarray(t)) for t in tiles[i:i+bs]]).to(dev); pr=[]
        for _ in range(T_MC): ff=ftm.forward_features(xb); pr.append(torch.sigmoid(fth(ff['x_norm_clstoken']).squeeze(-1)).cpu().numpy())
        A=np.stack(pr); M.append(A.mean(0)); V.append(A.var(0))
    ftm.eval(); fth.eval(); return (np.concatenate(M),np.concatenate(V)) if M else (np.array([]),np.array([]))
train_imgs=sorted(glob.glob(str(YO/'ds_a0'/'images'/'train'/'*.jpg')))
# ---- teacher generates refined pseudo-labels into a dataset dir ----
def yolo_line(b): cx=(b[0]+b[2])/2/CROP; cy=(b[1]+b[3])/2/CROP; w=(b[2]-b[0])/CROP; h=(b[3]-b[1])/CROP; return f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"
@torch.no_grad()
def teacher_generate(teacher, dsroot, mode):
    teacher.eval(); (dsroot/'labels'/'train').mkdir(parents=True,exist_ok=True); nlab=0
    for ip in train_imgs:
        im=Image.open(ip).convert('RGB'); arr=np.asarray(im); x=TF.to_tensor(im).to(dev)
        pred=teacher([x])[0]; boxes=pred['boxes'].cpu().numpy(); scores=pred['scores'].cpu().numpy()
        keep=np.where(scores>0.15)[0]; keep=keep[np.argsort(-scores[keep])][:MC_CAP]
        base=os.path.basename(ip)[:-4]
        if len(keep)==0: (dsroot/'labels'/'train'/(base+'.txt')).write_text(""); continue
        if mode=='naive':
            sel=[j for j in keep if scores[j]>0.7]
        else:
            tiles=[]
            for j in keep:
                cx=(boxes[j][0]+boxes[j][2])/2; cy=(boxes[j][1]+boxes[j][3])/2
                x0=int(min(max(cx-TILE/2,0),CROP-TILE)); y0=int(min(max(cy-TILE/2,0),CROP-TILE)); tiles.append(arr[y0:y0+TILE,x0:x0+TILE])
            pm,pv=mc_boxes(tiles); sel=[keep[ji] for ji in range(len(keep)) if (pv[ji]>=b_protect or pm[ji]>=a_star)]
        (dsroot/'labels'/'train'/(base+'.txt')).write_text("\n".join(yolo_line(boxes[j]) for j in sel)); nlab+=len(sel)
    return nlab
# ---- dataset / training ----
class DetDS(torch.utils.data.Dataset):
    def __init__(self,imgdir,labdir): self.imgs=sorted(glob.glob(str(imgdir/'*.jpg'))); self.labdir=labdir
    def __len__(self): return len(self.imgs)
    def __getitem__(self,i):
        ip=self.imgs[i]; x=TF.to_tensor(Image.open(ip).convert('RGB')); lp=self.labdir/(os.path.basename(ip)[:-4]+'.txt'); bx=[]
        if lp.exists():
            for ln in lp.read_text().splitlines():
                f=ln.split()
                if len(f)==5: _,cx,cy,w,h=f; cx,cy,w,h=float(cx)*CROP,float(cy)*CROP,float(w)*CROP,float(h)*CROP; bx.append([cx-w/2,cy-h/2,cx+w/2,cy+h/2])
        bx=torch.tensor(bx,dtype=torch.float32).reshape(-1,4)
        if len(bx): bx[:,[0,2]]=bx[:,[0,2]].clamp(0,CROP); bx[:,[1,3]]=bx[:,[1,3]].clamp(0,CROP); k=(bx[:,2]-bx[:,0]>=1)&(bx[:,3]-bx[:,1]>=1); bx=bx[k]
        return x,{'boxes':bx,'labels':torch.ones(len(bx),dtype=torch.int64)}
def collate(b): return tuple(zip(*b))
def train_epoch(model,imgdir,labdir,opt):
    dl=torch.utils.data.DataLoader(DetDS(imgdir,labdir),batch_size=BATCH,shuffle=True,collate_fn=collate,num_workers=6)
    params=[p for p in model.parameters() if p.requires_grad]; model.train()
    for imgs,tgts in dl:
        imgs=[im.to(dev) for im in imgs]; tgts=[{k:v.to(dev) for k,v in t.items()} for t in tgts]
        if sum(len(t['boxes']) for t in tgts)==0: continue
        loss=sum(model(imgs,tgts).values()); opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(params,5.0); opt.step()
def ema(teacher,student,a):
    with torch.no_grad():
        for tp,sp in zip(teacher.parameters(),student.parameters()): tp.mul_(a).add_(sp,alpha=1-a)
        for tb,sb in zip(teacher.buffers(),student.buffers()): tb.copy_(sb)
def iou(a,b):
    ix1,iy1=max(a[0],b[0]),max(a[1],b[1]); ix2,iy2=min(a[2],b[2]),min(a[3],b[3]); iw,ih=max(0,ix2-ix1),max(0,iy2-iy1); it=iw*ih
    ua=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-it; return it/ua if ua>0 else 0.0
TR_IMG=YO/'ds_a0'/'images'/'train'; VAL_IMG=YO/'ds_a0'/'images'/'val'; VAL_LAB=YO/'ds_a0'/'labels'/'val'
@torch.no_grad()
def evaluate(model):
    ds=DetDS(VAL_IMG,VAL_LAB); model.eval(); cropgt=[len(ds[i][1]['boxes']) for i in range(len(ds))]; med=np.median([c for c in cropgt if c>0]) if any(cropgt) else 0
    dl=torch.utils.data.DataLoader(ds,batch_size=BATCH,shuffle=False,collate_fn=collate,num_workers=6)
    all_tp={0.3:[],0.5:[]}; sc=[]; ngt=0; btp={'low':[],'high':[]}; bsc={'low':[],'high':[]}; bgt={'low':0,'high':0}; idx=0
    for imgs,tgts in dl:
        preds=model([im.to(dev) for im in imgs])
        for pred,tgt in zip(preds,tgts):
            gt=tgt['boxes'].numpy().tolist(); ngt+=len(gt); bd='low' if cropgt[idx]<=med else 'high'; bgt[bd]+=len(gt); idx+=1
            bxs=pred['boxes'].cpu().numpy(); scs=pred['scores'].cpu().numpy(); order=np.argsort(-scs)
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
                    if iouth==0.3: sc.append(float(scs[j])); btp[bd].append(tp); bsc[bd].append(float(scs[j]))
    def ap(tps,scs,ng):
        if not tps: return 0.0
        o=np.argsort(-np.array(scs)); tps=np.array(tps)[o]; ctp=np.cumsum(tps); cfp=np.cumsum(1-tps); rec=ctp/max(1,ng); pre=ctp/np.maximum(1,ctp+cfp)
        mr=np.concatenate([[0],rec,[1]]); mp=np.concatenate([[0],pre,[0]])
        for i in range(len(mp)-1,0,-1): mp[i-1]=max(mp[i-1],mp[i])
        idx=np.where(mr[1:]!=mr[:-1])[0]; return float(np.sum((mr[idx+1]-mr[idx])*mp[idx+1]))
    return dict(**{f'AP@{t}':round(ap(all_tp[t],sc,ngt),3) for t in (0.3,0.5)}, AP03_low=round(ap(btp['low'],bsc['low'],bgt['low']),3), AP03_high=round(ap(btp['high'],bsc['high'],bgt['high']),3))

def run(mode):
    student=build_ret(); student.load_state_dict(torch.load(S6/'retinanet_r50_protected.pth')); teacher=copy.deepcopy(student)
    opt=torch.optim.SGD([p for p in student.parameters() if p.requires_grad],lr=LR,momentum=0.9,weight_decay=1e-4)
    dsroot=RES/f'ds_{mode}'; (dsroot).mkdir(parents=True,exist_ok=True)
    for ep in range(EPOCHS):
        if ep in REFRESH:
            n=teacher_generate(teacher,dsroot,mode); print(f'  [{mode}] ep{ep} refreshed pseudo-labels: {n}',flush=True)
        train_epoch(student,TR_IMG,dsroot/'labels'/'train',opt)
        ema(teacher,student,EMA)
    r=evaluate(teacher); print(f'=== teacher-student [{mode}] -> {r}',flush=True); return r
SEEDS=[0,1,2]                                    # 3 runs -> mean±std (MILCA repeated-runs protocol)
runs={'ts_naive':[], 'ts_ours':[]}
for seed in SEEDS:
    torch.manual_seed(seed); np.random.seed(seed)
    for mode in ['naive','ours']:
        print(f'=== seed {seed} teacher-student: {mode} ===',flush=True); runs[f'ts_{mode}'].append(run(mode))
res={'seeds':SEEDS,'runs':runs}
for nm_ in runs:
    ks=list(runs[nm_][0].keys())
    res[nm_]={k:{'mean':round(float(np.mean([r[k] for r in runs[nm_]])),3),'std':round(float(np.std([r[k] for r in runs[nm_]])),3)} for k in ks}
# references
res['D1_protected_base']={'AP@0.3':0.342,'AP@0.5':0.125,'AP03_low':0.267,'note':'no refinement (Phase1)'}
res['oneshot_bootstrap_ours']={'AP@0.3':0.336,'AP@0.5':0.208,'AP03_low':0.234,'note':'Phase2 one-shot'}
json.dump(res,open(RES/('ts_gamma%g.json'%GAMMA),'w'),indent=2); print('\n=== teacher-student results (3-run mean±std) ==='); print(json.dumps(res,indent=2)); print('TS_DONE',flush=True)
