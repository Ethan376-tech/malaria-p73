# -*- coding: utf-8 -*-
"""RetinaNet teacher-student SPACE retain, IMPROVED: density computed among the SINGLE-SIGNAL-kept candidates only
(not all raw detections), and swept over 3 operating points (NDMAX on single-kept density = aggressive->mild). Maps the
AP@0.5 <-> low-density trade-off to pick a principled operating point for the report. 3-seed each. Same protocol as
dev_s9_space.py / build_s9_teacher_student.py. Output -> s9_teacher_student/space_sweep_results.json"""
import os, sys, json, glob, shutil, copy
os.environ['XFORMERS_DISABLED']='1'; os.environ['PYTORCH_CUDA_ALLOC_CONF']='expandable_segments:True'
import numpy as np, torch, torch.nn as nn
import torchvision.transforms.functional as TF
import torchvision.transforms as T
from PIL import Image; Image.MAX_IMAGE_PIXELS=None
from torchvision.models.detection import retinanet_resnet50_fpn
sys.path.insert(0,'/root/autodl-tmp/A_Dissertation/references/RedDino/train'); sys.path.append('/root/A_Dissertation')
from common.paths import OUTPUTS, CHECKPOINTS
from dinov2.models.vision_transformer import vit_small
dev='cuda'; CROP=640; TILE=224; T_MC=15; EPOCHS=10; REFRESH=[0,5]; EMA=0.95; LR=0.002; BATCH=6; MC_CAP=40; RSPACE=200.0
OPLEVELS=[('agg',1),('mid',2),('mild',5)]   # NDMAX on single-kept density: keep a box iff <= NDMAX single-neighbours within RSPACE
YO=OUTPUTS/'experiments'/'s5_yolo'; S6=OUTPUTS/'experiments'/'s6_retinanet'; RES=OUTPUTS/'experiments'/'s9_teacher_student'; RES.mkdir(parents=True,exist_ok=True)
FIN=json.load(open(OUTPUTS/'experiments'/'s3_filter'/'final.json'))['calib']; b_protect=FIN['b_protect']; a_star=FIN['a_star']
norm=T.Compose([T.ToTensor(),T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
def build_ret(): return retinanet_resnet50_fpn(num_classes=2,weights=None,weights_backbone='IMAGENET1K_V1').to(dev)
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
def yolo_line(b): cx=(b[0]+b[2])/2/CROP; cy=(b[1]+b[3])/2/CROP; w=(b[2]-b[0])/CROP; h=(b[3]-b[1])/CROP; return f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"
@torch.no_grad()
def teacher_generate(teacher, dsroot, ndmax):
    teacher.eval(); (dsroot/'labels'/'train').mkdir(parents=True,exist_ok=True); nlab=0; nsingle=0
    for ip in train_imgs:
        im=Image.open(ip).convert('RGB'); arr=np.asarray(im); x=TF.to_tensor(im).to(dev)
        pred=teacher([x])[0]; boxes=pred['boxes'].cpu().numpy(); scores=pred['scores'].cpu().numpy()
        keep=np.where(scores>0.15)[0]; keep=keep[np.argsort(-scores[keep])][:MC_CAP]
        base=os.path.basename(ip)[:-4]
        if len(keep)==0: (dsroot/'labels'/'train'/(base+'.txt')).write_text(""); continue
        tiles=[]
        for j in keep:
            cx=(boxes[j][0]+boxes[j][2])/2; cy=(boxes[j][1]+boxes[j][3])/2
            x0=int(min(max(cx-TILE/2,0),CROP-TILE)); y0=int(min(max(cy-TILE/2,0),CROP-TILE)); tiles.append(arr[y0:y0+TILE,x0:x0+TILE])
        pm,pv=mc_boxes(tiles)
        single=[ji for ji in range(len(keep)) if (pv[ji]>=b_protect or pm[ji]>=a_star)]; nsingle+=len(single)
        # density AMONG SINGLE-KEPT candidates only
        if len(single)>=2:
            Cs=np.array([[(boxes[keep[ji]][0]+boxes[keep[ji]][2])/2,(boxes[keep[ji]][1]+boxes[keep[ji]][3])/2] for ji in single],float)
            D=np.sqrt(((Cs[:,None,:]-Cs[None,:,:])**2).sum(-1)); nds=(D<=RSPACE).sum(1)-1
            selji=[single[m] for m in range(len(single)) if nds[m]<=ndmax]
        else: selji=single
        sel=[keep[ji] for ji in selji]
        (dsroot/'labels'/'train'/(base+'.txt')).write_text("\n".join(yolo_line(boxes[j]) for j in sel)); nlab+=len(sel)
    print(f'   [nd{ndmax}] single {nsingle} -> space {nlab} ({100*nlab/max(1,nsingle):.0f}%)',flush=True); return nlab
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
    return dict(**{f'AP@{t}':round(ap(all_tp[t],sc,ngt),3) for t in (0.3,0.5)}, AP03_low=round(ap(btp['low'],bsc['low'],bgt['low']),3))
def run(ndmax):
    student=build_ret(); student.load_state_dict(torch.load(S6/'retinanet_r50_protected.pth')); teacher=copy.deepcopy(student)
    opt=torch.optim.SGD([p for p in student.parameters() if p.requires_grad],lr=LR,momentum=0.9,weight_decay=1e-4)
    dsroot=RES/f'ds_sweep_nd{ndmax}'; (dsroot).mkdir(parents=True,exist_ok=True)
    for ep in range(EPOCHS):
        if ep in REFRESH: teacher_generate(teacher,dsroot,ndmax)
        train_epoch(student,TR_IMG,dsroot/'labels'/'train',opt); ema(teacher,student,EMA)
    return evaluate(teacher)
out={'RSPACE':RSPACE,'levels':{},'report_ref':{'single':{'AP@0.3':0.374,'AP@0.5':0.158,'AP03_low':0.292},'multi':{'AP@0.3':0.372,'AP@0.5':0.155,'AP03_low':0.293}}}
for tag,nd in OPLEVELS:
    rr=[]
    for seed in [0,1,2]:
        torch.manual_seed(seed); np.random.seed(seed); print(f'=== {tag} (nd{nd}) seed {seed} ===',flush=True); rr.append(run(nd))
    ks=list(rr[0].keys()); out['levels'][tag]={'ndmax':nd,'runs':rr,**{m:{'mean':round(float(np.mean([r[m] for r in rr])),3),'std':round(float(np.std([r[m] for r in rr])),3)} for m in ks}}
    print(f'--- {tag}: '+' '.join(f'{m} {out["levels"][tag][m]["mean"]:.3f}' for m in ks),flush=True)
    json.dump(out,open(RES/'space_sweep_results.json','w'),indent=2)
print('\n=== RetinaNet TS space sweep (3-seed) ==='); print(json.dumps({t:{k:out['levels'][t][k] for k in ('ndmax','AP@0.3','AP@0.5','AP03_low')} for t in out['levels']},indent=2)); print('S9_SWEEP_DONE',flush=True)
