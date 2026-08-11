# -*- coding: utf-8 -*-
"""T2 flagship: SOFT filtering via uncertainty-WEIGHTED pseudo-box loss. Instead of hard keep/drop, train MicroYOLO on
ALL A0 boxes but scale each box's loss contribution by its per-box weight (from ds_soft/weights.json). Custom dataset
carries box_weights; custom loss scales target_scores by the assigned box's weight. Box-preserving aug only (mosaic/scale/
translate/perspective OFF) so weights stay aligned. Compare 3 arms x 3 seeds under the SAME reduced aug: A0 (all boxes,
weight 1), protected (hard filter), soft (weighted). Custom AP + low/high bands. Output -> outputs/experiments/t2_softloss/.
Heartbeat -> t2_status.log"""
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
from ultralytics.utils.ops import xywh2xyxy
sys.path.append('/root/A_Dissertation'); from common.paths import OUTPUTS
dev='cuda'; CROP=640; TEST=bool(os.environ.get('T2_TEST')); EPOCHS=3 if TEST else 100
SWEEP=os.environ.get('T2_SWEEP','')   # e.g. 'w00' -> soft-only on ds_soft{w00}, output t2_softloss{w00}
YO=OUTPUTS/'experiments'/'s5_yolo'; RES=OUTPUTS/'experiments'/f't2_softloss{SWEEP}'; RES.mkdir(parents=True,exist_ok=True)
STAT=OUTPUTS/'experiments'/f't2_status{SWEEP}.log'
def hb(m):
    with open(STAT,'a') as f: f.write(f'{int(time.time())} {m}\n')
WMAP=json.load(open(YO/f'ds_soft{SWEEP}'/'weights.json'))  # cropstem -> [weights in label order]
# ---- custom dataset: attach per-box weights aligned with bboxes ----
class WeightedDataset(YOLODataset):
    def __getitem__(self, index):
        label=super().__getitem__(index)
        stem=os.path.splitext(os.path.basename(label['im_file']))[0]
        n=label['bboxes'].shape[0]; w=WMAP.get(stem, None)
        if w is not None and len(w)==n: bw=torch.tensor(w, dtype=torch.float32)
        else: bw=torch.ones(n, dtype=torch.float32)   # val / mismatch -> weight 1
        label['box_weights']=bw
        return label
    @staticmethod
    def collate_fn(batch):
        new=YOLODataset.collate_fn(batch)
        new['box_weights']=torch.cat([b['box_weights'] for b in batch], 0)
        return new
# ---- custom loss: scale target_scores by assigned box weight ----
class WeightedLoss(v8DetectionLoss):
    def get_assigned_targets_and_loss(self, preds, batch):
        loss=torch.zeros(3, device=self.device)
        pred_distri, pred_scores = preds["boxes"].permute(0,2,1).contiguous(), preds["scores"].permute(0,2,1).contiguous()
        anchor_points, stride_tensor = make_anchors(preds["feats"], self.stride, 0.5)
        dtype=pred_scores.dtype; batch_size=pred_scores.shape[0]
        imgsz=torch.tensor(preds["feats"][0].shape[2:], device=self.device, dtype=dtype)*self.stride[0]
        bw = batch["box_weights"].view(-1,1).to(self.device) if "box_weights" in batch else torch.ones_like(batch["cls"].view(-1,1)).to(self.device)
        targets=torch.cat((batch["batch_idx"].view(-1,1), batch["cls"].view(-1,1), batch["bboxes"], bw), 1)  # (n,7)
        targets=self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1,0,1,0]])           # (bs,nmax,6)
        gt_labels, gt_bboxes, gt_w = targets.split((1,4,1), 2)
        mask_gt=gt_bboxes.sum(2, keepdim=True).gt_(0.0)
        pred_bboxes=self.bbox_decode(anchor_points, pred_distri)
        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores.detach().sigmoid(), (pred_bboxes.detach()*stride_tensor).type(gt_bboxes.dtype),
            anchor_points*stride_tensor, gt_labels, gt_bboxes, mask_gt)
        # SOFT WEIGHT: scale target_scores by the weight of the box each anchor is assigned to
        per_anchor_w = gt_w.squeeze(-1).gather(1, target_gt_idx)   # (bs, num_anchors)
        target_scores = target_scores * per_anchor_w.unsqueeze(-1)
        target_scores_sum=max(target_scores.sum(), 1)
        loss[1]=self.bce(pred_scores, target_scores.to(dtype)).sum()/target_scores_sum
        if fg_mask.sum():
            loss[0], loss[2]=self.bbox_loss(pred_distri, pred_bboxes, anchor_points, target_bboxes/stride_tensor,
                                            target_scores, target_scores_sum, fg_mask, imgsz, stride_tensor)
        loss[0]*=self.hyp.box; loss[1]*=self.hyp.cls; loss[2]*=self.hyp.dfl
        return (fg_mask, target_gt_idx, target_bboxes, anchor_points, stride_tensor), loss, loss.detach()
class WeightedModel(DetectionModel):
    def init_criterion(self): return WeightedLoss(self)
class WeightedTrainer(DetectionTrainer):
    def get_model(self, cfg=None, weights=None, verbose=True):
        m=WeightedModel(cfg or 'yolov8n-p2.yaml', nc=self.data['nc'], verbose=verbose)
        if weights: m.load(weights)
        return m
    def build_dataset(self, img_path, mode='train', batch=None):
        ds=super().build_dataset(img_path, mode, batch)
        ds.__class__=WeightedDataset   # promote to weighted variant (keeps loaded labels/transforms)
        return ds
# ---- eval (custom AP, == Table 4.9) ----
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
            m=set()
            for j in o:
                bst=-1;bj=-1
                for jg,g in enumerate(gt):
                    if jg in m: continue
                    v=iou(bxs[j],g)
                    if v>bst: bst,bj=v,jg
                tp=1 if bst>=th and bj>=0 else 0
                if tp:m.add(bj)
                at[th].append(tp)
                if th==0.3: sc.append(float(scs[j])); btp[bd].append(tp); bsc[bd].append(float(scs[j]))
    def ap(t,s,n):
        if not t:return 0.0
        oo=np.argsort(-np.array(s)); t=np.array(t)[oo]; ct=np.cumsum(t);cf=np.cumsum(1-t);rc=ct/max(1,n);pr=ct/np.maximum(1,ct+cf)
        mr=np.r_[0,rc,1];mp=np.r_[0,pr,0]
        for i in range(len(mp)-1,0,-1):mp[i-1]=max(mp[i-1],mp[i])
        k=np.where(mr[1:]!=mr[:-1])[0];return float(np.sum((mr[k+1]-mr[k])*mp[k+1]))
    return dict(**{f'AP@{t}':round(ap(at[t],sc,ng),3) for t in (0.3,0.5)}, AP03_low=round(ap(btp['low'],bsc['low'],bg['low']),3), AP03_high=round(ap(btp['high'],bsc['high'],bg['high']),3))
AUG=dict(mosaic=0.0, close_mosaic=0, mixup=0.0, copy_paste=0.0, scale=0.0, translate=0.0, degrees=0.0, shear=0.0, perspective=0.0, fliplr=0.5, flipud=0.0)
def run(arm, yaml, seed):
    y=YOLO('yolov8n-p2.yaml')
    kw=dict(data=yaml, epochs=EPOCHS, imgsz=CROP, batch=16, device=0, workers=8, box=9.0, pretrained='yolov8n.pt',
            project=str(RES/'runs'), name=f'{arm}_s{seed}', exist_ok=True, verbose=False, plots=False, seed=seed, **AUG)
    if arm=='soft': y.train(trainer=WeightedTrainer, **kw)
    else: y.train(**kw)
    return evaluate(RES/'runs'/f'{arm}_s{seed}'/'weights'/'best.pt')
ARMS=[('a0', str(YO/'data_a0.yaml')), ('protected', str(YO/'data_protected.yaml')), ('soft', str(YO/f'data_soft{SWEEP}.yaml'))]
if TEST:
    hb('T2 TEST: soft arm, 3 epochs (custom-loss smoke test)')
    r=run('soft', str(YO/'data_soft.yaml'), 0); hb(f'T2_TEST_DONE soft(3ep) -> {r}'); sys.exit(0)
if SWEEP:
    ARMS=[('soft', str(YO/f'data_soft{SWEEP}.yaml'))]; hb(f'T2 SWEEP {SWEEP}: soft arm only, 3 seeds')
hb('T2 START (reduced aug: mosaic/scale/translate off, flip+hsv on)')
runs={a[0]:[] for a in ARMS}
for seed in [0,1,2]:
    for arm,yaml in ARMS:
        r=run(arm,yaml,seed); runs[arm].append(r); hb(f'{arm} s{seed} -> {r}')
    json.dump(runs,open(RES/'results.json','w'),indent=2)
agg={}
for k in runs:
    ks=list(runs[k][0].keys()); agg[k]={m:{'mean':round(float(np.mean([r[m] for r in runs[k]])),3),'std':round(float(np.std([r[m] for r in runs[k]])),3)} for m in ks}
json.dump(dict(agg=agg,runs=runs),open(RES/'results.json','w'),indent=2)
hb('T2_DONE ' + ' | '.join(f'{k}={agg[k]}' for k in agg))
