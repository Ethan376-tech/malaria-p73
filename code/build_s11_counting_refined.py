# -*- coding: utf-8 -*-
"""Table 4.15 with the REFINED MicroYOLO (teacher-student, MC-dropout retain) as the weakly-supervised detector — the
pipeline's final detector, which recovers low-density recall. Same counting procedure as build_s11. Weakly-supervised
is run for all 3 teacher-student seeds (y13 protected_s{0,1,2}_r1) -> mean±std; GT-supervised = MicroYOLO-gtsup (y11).
Output -> outputs/experiments/s11_counting_refined/."""
import os, sys, json
os.environ['XFORMERS_DISABLED']='1'; os.environ['PYTORCH_CUDA_ALLOC_CONF']='expandable_segments:True'
import numpy as np
from PIL import Image; Image.MAX_IMAGE_PIXELS=None
from ultralytics import YOLO
sys.path.append('/root/A_Dissertation'); from common.paths import RAW, OUTPUTS
from scipy.stats import pearsonr
TILE=640; NMS_D=21; PRED_CONF=0.01
Y11=OUTPUTS/'experiments'/'y11_microyolo_table47'/'runs'; Y13=OUTPUTS/'experiments'/'y13_microyolo_ts'/'runs'
RES=OUTPUTS/'experiments'/'s11_counting_refined'; RES.mkdir(parents=True,exist_ok=True)
import random; random.seed(0)
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
def nms_pts(pts, md):
    pts=sorted(pts,key=lambda z:-z[2]); k=[]
    for x,y,s in pts:
        if all((x-kx)**2+(y-ky)**2>=md*md for kx,ky,_ in k): k.append((x,y,s))
    return k
def field_dets(model, im):
    W,H=im.size; arr=np.asarray(im); xs=sorted(set(list(range(0,max(1,W-TILE+1),TILE))+[max(0,W-TILE)])); ys=sorted(set(list(range(0,max(1,H-TILE+1),TILE))+[max(0,H-TILE)]))
    pts=[]
    for ty in ys:
        for tx in xs:
            sub=arr[ty:ty+TILE,tx:tx+TILE]
            if sub.shape[0]<TILE or sub.shape[1]<TILE:
                pad=np.zeros((TILE,TILE,3),np.uint8); pad[:sub.shape[0],:sub.shape[1]]=sub; sub=pad
            p=model.predict(Image.fromarray(sub), imgsz=TILE, conf=PRED_CONF, verbose=False, device=0)[0]
            b=p.boxes.xyxy.cpu().numpy(); s=p.boxes.conf.cpu().numpy()
            for j in range(len(s)): pts.append((tx+(b[j][0]+b[j][2])/2, ty+(b[j][1]+b[j][3])/2, float(s[j])))
    return nms_pts(pts, NMS_D)
def counts_at(detsets, tau): return np.array([sum(1 for (_,_,s) in d if s>=tau) for d in detsets])
def metrics(pred,gt):
    pred=np.asarray(pred,float); gt=np.asarray(gt,float)
    ss_res=np.sum((gt-pred)**2); ss_tot=np.sum((gt-gt.mean())**2); r2=1-ss_res/ss_tot if ss_tot>0 else float('nan')
    r=pearsonr(pred,gt)[0] if np.std(pred)>0 and np.std(gt)>0 else float('nan'); mae=np.mean(np.abs(pred-gt))
    return dict(R2=round(float(r2),3), pearson=round(float(r),3), MAE=round(float(mae),2))
def eval_ckpt(ckpt):
    m=YOLO(str(ckpt))
    dev_d=[field_dets(m,Image.open(ip).convert('RGB')) for ip,_ in devF]; dev_gt=np.array([len(b) for _,b in devF])
    best=(1e9,0.1)
    for t in np.linspace(0.02,0.9,45):
        mae=np.mean(np.abs(counts_at(dev_d,t)-dev_gt))
        if mae<best[0]: best=(mae,float(t))
    tau=best[1]
    test_d=[field_dets(m,Image.open(ip).convert('RGB')) for ip,_ in testF]; test_gt=np.array([len(b) for _,b in testF])
    pred=counts_at(test_d,tau); med=np.median(test_gt); low=test_gt<=med; high=test_gt>med
    return dict(tau=round(tau,3), overall=metrics(pred,test_gt), low_density=metrics(pred[low],test_gt[low]),
                high_density=metrics(pred[high],test_gt[high]), mean_gt=round(float(test_gt.mean()),1), mean_pred=round(float(pred.mean()),1)), (test_gt.tolist(),pred.tolist())
weak_runs=[]; scatter={}
for s in [0,1,2]:
    r,sc=eval_ckpt(Y13/f'protected_s{s}_r1'/'weights'/'best.pt'); weak_runs.append(r)
    if s==0: scatter['weakly_supervised']=sc
    print(f'weakly_supervised(refined) s{s}',r,flush=True)
gt_r,gt_sc=eval_ckpt(Y11/'gtsup'/'weights'/'best.pt'); scatter['GT_supervised']=gt_sc
print('GT_supervised',gt_r,flush=True)
def agg(runs,path):
    out={}
    for band in ['overall','low_density','high_density']:
        for k in ['R2','pearson','MAE']:
            v=[x[band][k] for x in runs]; out[f'{band}.{k}']={'mean':round(float(np.mean(v)),3),'std':round(float(np.std(v)),3)}
    return out
res={'weakly_supervised_refined':{'per_seed':weak_runs,'agg':agg(weak_runs,None),'tau_mean':round(float(np.mean([x['tau'] for x in weak_runs])),3)},
     'GT_supervised':gt_r,
     'ref_retinanet_Table4.15':{'weak':{'r':0.47,'low_r':0.70,'MAE':5.95},'GT':{'r':0.80}}}
json.dump(res, open(RES/'counting_results.json','w'), indent=2)
a=res['weakly_supervised_refined']['agg']
print('\n=== REFINED MicroYOLO counting (3-seed) vs RetinaNet Table 4.15 (weak r0.47/low0.70, GT r0.80) ===')
print(f"  weakly-sup refined  overall r={a['overall.pearson']['mean']}±{a['overall.pearson']['std']}  MAE={a['overall.MAE']['mean']}  low-density r={a['low_density.pearson']['mean']}±{a['low_density.pearson']['std']}")
print(f"  GT-sup              overall r={gt_r['overall']['pearson']}  low-density r={gt_r['low_density']['pearson']}")
# --- regenerate Figure 4.12 (predicted vs GT scatter) with MicroYOLO, into the report figures dir ---
json.dump({'weakly_supervised_s0':scatter['weakly_supervised'],'GT_supervised':scatter['GT_supervised']}, open(RES/'scatter.json','w'))
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({'font.family':'serif'})
FIG=OUTPUTS/'experiments'/'figures'; am=a; taum=res['weakly_supervised_refined']['tau_mean']
fig,ax=plt.subplots(1,2,figsize=(11,4.6))
panels=[('weakly_supervised','#228833','weakly-supervised (MicroYOLO, refined)',
         f"r={am['overall.pearson']['mean']}  R²={am['overall.R2']['mean']}  MAE={am['overall.MAE']['mean']}  (3-run mean; τ≈{taum})"),
        ('GT_supervised','#4477aa','GT-supervised (MicroYOLO)',
         f"r={gt_r['overall']['pearson']}  R²={gt_r['overall']['R2']}  MAE={gt_r['overall']['MAE']}  (τ={gt_r['tau']})")]
for k,(name,col,title,sub) in enumerate(panels):
    gt,pred=scatter[name]; mx=max(max(gt),max(pred))+1
    ax[k].scatter(gt,pred,c=col,alpha=0.7,edgecolors='k',linewidths=0.4); ax[k].plot([0,mx],[0,mx],ls='--',color='gray')
    ax[k].set_title(f"{title}\n{sub}",fontsize=10); ax[k].set_xlabel('GT parasite count / field'); ax[k].set_ylabel('predicted count'); ax[k].grid(alpha=0.3)
plt.tight_layout(); plt.savefig(FIG/'fig_counting.png',dpi=140,bbox_inches='tight'); plt.savefig(RES/'fig_counting.png',dpi=140,bbox_inches='tight')
print('fig saved ->',str(FIG/'fig_counting.png'),flush=True)
print('COUNTING_REFINED_DONE',flush=True)
