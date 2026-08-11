# -*- coding: utf-8 -*-
"""Leakage robustness for Part-2 (2): the detector trained on pseudo-labels from up to 70 Ibadan POSITIVE fields.
Identify which eval samples contributed those training fields, and re-evaluate the low-parasitaemia (pos_lo vs neg)
detection decision EXCLUDING train-exposed positives, to confirm the detection>bag result is not leakage."""
import sys, json, random
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
sys.path.append('/root/A_Dissertation'); from common.paths import OUTPUTS, IBADAN_PART1, IBADAN_PART2
RES=OUTPUTS/'experiments'/'dc_detcount'; df=pd.read_csv(RES/'persample.csv')
# reconstruct detector's training fields exactly as build_s5_yolo_data.py
bm=pd.read_csv(OUTPUTS/'experiments'/'stageD_bag'/'bag_meta_v2.csv')
ib=pd.read_csv('/root/autodl-tmp/A_Dissertation/data/raw/ibadan/sample_codes_parasite_diagnosis_crosscheck.csv')
sampdir={r.sample_code:(IBADAN_PART1 if 'part1' in r.input_file else IBADAN_PART2)/r.sample_code for _,r in ib.iterrows()}
posids=set(bm[(bm.dataset=='ibadan')&(bm.label01==1)].sample_id)
EXT={'.tif','.tiff','.png','.jpg','.jpeg'}; train_fields=[]; field_sample=[]
for sid in sorted(posids):
    d=sampdir.get(sid)
    if d is None or not d.exists(): continue
    fs=[p for p in d.rglob('*') if p.suffix.lower() in EXT][:2]
    train_fields+=fs; field_sample+=[sid]*len(fs)
    if len(train_fields)>=80: break
order=list(range(len(train_fields))); random.Random(0).shuffle(order); order=order[:70]
exposed=set(field_sample[i] for i in order)
print(f'train-exposed positive samples: {len(exposed)}',flush=True)
df['exposed']=df.sample_id.isin(exposed)
print('exposed by band:', df[df.exposed].band.value_counts().to_dict(),flush=True)
def evalset(d, band, sig):
    m=d.band.isin(['neg',band]); s=d[m]; y=(s.band==band).astype(int).values; x=s[sig].values.astype(float)
    if len(np.unique(y))<2: return None
    auc=round(float(roc_auc_score(y,x)),3)
    folds=s.fold.values; pred=np.zeros(len(y))
    for f in np.unique(folds):
        tr=folds!=f; te=folds==f
        if tr.sum()==0 or len(np.unique(y[tr]))<2: pred[te]=(x[te]>=np.median(x)).astype(int); continue
        best=(-1,0)
        for t in np.unique(x[tr]):
            ba=balanced_accuracy_score(y[tr],(x[tr]>=t).astype(int))
            if ba>best[0]: best=(ba,t)
        pred[te]=(x[te]>=best[1]).astype(int)
    return auc, round(float(balanced_accuracy_score(y,pred)),3), int((y==1).sum())
print('\n=== pos_lo vs neg, detection signals: ALL vs train-exposed-EXCLUDED ===')
clean=df[~(df.exposed & (df.band!='neg'))]  # drop exposed positives (keep all negs)
for sig in ['maxconf','top5sum','n05']:
    a=evalset(df,'pos_lo',sig); c=evalset(clean,'pos_lo',sig)
    print(f'  {sig:8s}  ALL: AUC={a[0]} BA={a[1]} (n_pos={a[2]})   CLEAN: AUC={c[0]} BA={c[1]} (n_pos={c[2]})')
json.dump({'exposed_n':len(exposed),'exposed_band':df[df.exposed].band.value_counts().to_dict()}, open(RES/'leakcheck.json','w'),indent=2)
print('DC3_DONE',flush=True)
