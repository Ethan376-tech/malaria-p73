# -*- coding: utf-8 -*-
"""T2 report-grade: uncertainty-WEIGHTED pseudo-box loss under FULL augmentation (mosaic on). Weight alignment through
mosaic/perspective is solved by piggybacking the per-box weight as a 2nd column of `cls` (which ultralytics concatenates
on mosaic and filters on out-of-view removal, in sync with boxes). Custom loss splits cls -> (class, weight) and scales
target_scores by the assigned box's weight. Compare a0 / protected / soft, 3 seeds, standard MicroYOLO recipe (mosaic on,
box=9, 100ep). Custom AP + bands. Output -> outputs/experiments/t2_fullaug/. Heartbeat -> t2fa_status.log.
T2FA_TEST=1 -> soft arm 3-epoch smoke test."""
import os, sys, json, glob, time
os.environ['XFORMERS_DISABLED']='1'; os.environ['PYTORCH_CUDA_ALLOC_CONF']='expandable_segments:True'
import numpy as np, torch
from PIL import Image; Image.MAX_IMAGE_PIXELS=None
from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.tasks import DetectionModel
from ultralytics.data.dataset import YOLODataset
from ultralytics.utils.loss import v8DetectionLoss
from ultralytics.utils.tal import make_anchors
sys.path.append('/root/A_Dissertation'); from common.paths import OUTPUTS
dev='cuda'; CROP=640; TEST=bool(os.environ.get('T2FA_TEST')); EPOCHS=3 if TEST else 100
YO=OUTPUTS/'experiments'/'s5_yolo'; RES=OUTPUTS/'experiments'/'t2_fullaug'; RES.mkdir(parents=True,exist_ok=True)
STAT=OUTPUTS/'experiments'/'t2fa_status.log'
def hb(m):
    with open(STAT,'a') as f: f.write(f'{int(time.time())} {m}\n')
WMAP=json.load(open(YO/'ds_soft'/'weights.json')); _WPRINTED=False
# ---- dataset: attach per-box weight as cls col-2 BEFORE transforms so mosaic/perspective carry it in sync ----
class WeightedDatasetFA(YOLODataset):
    def get_image_and_label(self, index):
        label=super().get_image_and_label(index)
        if self.augment:   # TRAIN only: attach weight col; VAL keeps cls (n,1) so the built-in validator works
            stem=os.path.splitext(os.path.basename(label['im_file']))[0]; n=label['cls'].shape[0]; w=WMAP.get(stem)
            wcol=np.array(w,dtype=np.float32).reshape(-1,1) if (w is not None and len(w)==n) else np.ones((n,1),np.float32)
            label['cls']=np.concatenate([label['cls'].astype(np.float32), wcol], axis=1)  # (n,2)=[class, weight]
        return label
    @staticmethod
    def collate_fn(batch):
        W=1
        for b in batch:
            if b['cls'].dim()>1 and b['cls'].shape[1]>W: W=b['cls'].shape[1]
        for b in batch:   # pad all cls to batch-max width (train W=2, val W=1); empty imgs padded with weight 1
            c=b['cls']
            if c.dim()==1: c=c.view(-1,1)
            if c.shape[1]<W: c=torch.cat([c, torch.ones((c.shape[0],W-c.shape[1]),dtype=c.dtype)],1)
            b['cls']=c
        return YOLODataset.collate_fn(batch)
# ---- loss: split cls -> (class, weight); scale target_scores by assigned-box weight ----
class WeightedLossFA(v8DetectionLoss):
    def get_assigned_targets_and_loss(self, preds, batch):
        loss=torch.zeros(3, device=self.device)
        pred_distri, pred_scores = preds["boxes"].permute(0,2,1).contiguous(), preds["scores"].permute(0,2,1).contiguous()
        anchor_points, stride_tensor = make_anchors(preds["feats"], self.stride, 0.5)
        dtype=pred_scores.dtype; batch_size=pred_scores.shape[0]
        imgsz=torch.tensor(preds["feats"][0].shape[2:], device=self.device, dtype=dtype)*self.stride[0]
        clsw=batch["cls"].to(self.device)
        if clsw.dim()==1: clsw=clsw.view(-1,1)
        cls=clsw[:,0:1]; bw=clsw[:,1:2] if clsw.shape[1]>=2 else torch.ones_like(clsw[:,0:1])
        global _WPRINTED
        if not _WPRINTED and bw.numel(): _WPRINTED=True; hb(f'LOSS weight check: n={bw.numel()} min={float(bw.min()):.3f} mean={float(bw.mean()):.3f} frac<1={float((bw<0.999).float().mean()):.3f}')
        targets=torch.cat((batch["batch_idx"].view(-1,1).to(self.device), cls, batch["bboxes"].to(self.device), bw), 1)  # (n,7)
        targets=self.preprocess(targets, batch_size, scale_tensor=imgsz[[1,0,1,0]])   # (bs,nmax,6)
        gt_labels, gt_bboxes, gt_w = targets.split((1,4,1), 2)
        mask_gt=gt_bboxes.sum(2, keepdim=True).gt_(0.0)
        pred_bboxes=self.bbox_decode(anchor_points, pred_distri)
        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores.detach().sigmoid(), (pred_bboxes.detach()*stride_tensor).type(gt_bboxes.dtype),
            anchor_points*stride_tensor, gt_labels, gt_bboxes, mask_gt)
        per_anchor_w = gt_w.squeeze(-1).gather(1, target_gt_idx)
        target_scores = target_scores * per_anchor_w.unsqueeze(-1)
        target_scores_sum=max(target_scores.sum(), 1)
        loss[1]=self.bce(pred_scores, target_scores.to(dtype)).sum()/target_scores_sum
        if fg_mask.sum():
            loss[0], loss[2]=self.bbox_loss(pred_distri, pred_bboxes, anchor_points, target_bboxes/stride_tensor,
                                            target_scores, target_scores_sum, fg_mask, imgsz, stride_tensor)
        loss[0]*=self.hyp.box; loss[1]*=self.hyp.cls; loss[2]*=self.hyp.dfl
        return (fg_mask, target_gt_idx, target_bboxes, anchor_points, stride_tensor), loss, loss.detach()
class WeightedModelFA(DetectionModel):
    def init_criterion(self): return WeightedLossFA(self)
class WeightedTrainerFA(DetectionTrainer):
    def get_model(self, cfg=None, weights=None, verbose=True):
        m=WeightedModelFA(cfg or 'yolov8n-p2.yaml', nc=self.data['nc'], verbose=verbose)
        if weights: m.load(weights)
        return m
    def build_dataset(self, img_path, mode='train', batch=None):
        ds=super().build_dataset(img_path, mode, batch); ds.__class__=WeightedDatasetFA; return ds
# ---- eval (custom AP) ----
val_imgs=sorted(glob.glob(str(YO/'ds_gtsup'/'images'/'val'/'*.jpg'))); VLAB=YO/'ds_gtsup'/'labels'/'val'
def load_gt(ip):
    lp=VLAB/(os.path.basename(ip)[:-4]+'.txt'); bx=[]
    if lp.exists():
        for ln in lp.read_text().splitlines():
            f=ln.split()
            if len(f)==5: _,cx,cy,w,h=[float(z) for z in f]; cx,cy,w,h=cx*CROP,cy*CROP,w*CROP,h*CROP; bx.append([cx-w/2,cy-h/2,cx+w/2,cy+h/2])
    return bx
gts=[load_gt(ip) for ip in val_imgs]; cropgt=[len(g) for g in gts]; med=np.median([c for c in cropgt if c>0]) if any(cropgt) else 0
def iou(a,b):
    ix1,iy1=max(a[0],b[0]),max(a[1],b[1]); ix2,iy2=min(a[2],b[2]),min(a[3],b[3]); iw,ih=max(0,ix2-ix1),max(0,iy2-iy1); it=iw*ih
    ua=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-it; return it/ua if ua>0 else 0
def evaluate(best):
    ck=torch.load(str(best), weights_only=False); ye=YOLO('yolov8n-p2.yaml'); ye.model=ck['model'].float().to(dev).eval()
    at={0.3:[],0.5:[]}; sc=[]; ng=0; btp={'low':[],'high':[]}; bsc={'low':[],'high':[]}; bg={'low':0,'high':0}
    for idx,ip in enumerate(val_imgs):
        gt=gts[idx]; ng+=len(gt); bd='low' if cropgt[idx]<=med else 'high'; bg[bd]+=len(gt)
        r=ye.predict(ip,imgsz=CROP,conf=0.05,verbose=False,device=0)[0]; bxs=r.boxes.xyxy.cpu().numpy(); scs=r.boxes.conf.cpu().numpy(); o=np.argsort(-scs)
        for th in (0.3,0.5):
            mm=set()
            for j in o:
                bst=-1;bj=-1
                for jg,g in enumerate(gt):
                    if jg in mm: continue
                    v=iou(bxs[j],g)
                    if v>bst: bst,bj=v,jg
                tp=1 if bst>=th and bj>=0 else 0
                if tp:mm.add(bj)
                at[th].append(tp)
                if th==0.3: sc.append(float(scs[j])); btp[bd].append(tp); bsc[bd].append(float(scs[j]))
    def ap(t,s,n):
        if not t:return 0.0
        oo=np.argsort(-np.array(s)); t=np.array(t)[oo]; ct=np.cumsum(t);cf=np.cumsum(1-t);rc=ct/max(1,n);pr=ct/np.maximum(1,ct+cf)
        mr=np.r_[0,rc,1];mp=np.r_[0,pr,0]
        for i in range(len(mp)-1,0,-1):mp[i-1]=max(mp[i-1],mp[i])
        k=np.where(mr[1:]!=mr[:-1])[0];return float(np.sum((mr[k+1]-mr[k])*mp[k+1]))
    return dict(**{f'AP@{t}':round(ap(at[t],sc,ng),3) for t in (0.3,0.5)}, AP03_low=round(ap(btp['low'],bsc['low'],bg['low']),3), AP03_high=round(ap(btp['high'],bsc['high'],bg['high']),3))
AUG=dict(mosaic=1.0, close_mosaic=10, mixup=0.0, copy_paste=0.0, fliplr=0.5)   # FULL aug (report recipe)
def run(arm, yaml, seed):
    y=YOLO('yolov8n-p2.yaml')
    kw=dict(data=yaml, epochs=EPOCHS, imgsz=CROP, batch=16, device=0, workers=8, box=9.0, pretrained='yolov8n.pt',
            project=str(RES/'runs'), name=f'{arm}_s{seed}', exist_ok=True, verbose=False, plots=False, seed=seed, **AUG)
    if arm=='soft': y.train(trainer=WeightedTrainerFA, **kw)
    else: y.train(**kw)
    return evaluate(RES/'runs'/f'{arm}_s{seed}'/'weights'/'best.pt')
if TEST:
    hb('T2FA TEST: soft 3ep mosaic-on smoke test'); r=run('soft', str(YO/'data_soft.yaml'), 0); hb(f'T2FA_TEST_DONE -> {r}'); sys.exit(0)
ARMS=[('a0', str(YO/'data_a0.yaml')), ('protected', str(YO/'data_protected.yaml')), ('soft', str(YO/'data_soft.yaml'))]
hb('T2FA START (FULL aug, mosaic on)')
runs={a[0]:[] for a in ARMS}
for seed in [0,1,2]:
    for arm,yaml in ARMS:
        try: r=run(arm,yaml,seed); runs[arm].append(r); hb(f'{arm} s{seed} -> {r}')
        except Exception as e: hb(f'{arm} s{seed} FAILED: {type(e).__name__}: {str(e)[:150]}')
    json.dump(runs,open(RES/'results.json','w'),indent=2)
agg={}
for k in runs:
    if runs[k]:
        ks=list(runs[k][0].keys()); agg[k]={m:{'mean':round(float(np.mean([r[m] for r in runs[k]])),3),'std':round(float(np.std([r[m] for r in runs[k]])),3)} for m in ks}
json.dump(dict(agg=agg,runs=runs),open(RES/'results.json','w'),indent=2)
hb('T2FA_DONE ' + ' | '.join(f'{k}={agg[k]}' for k in agg))
