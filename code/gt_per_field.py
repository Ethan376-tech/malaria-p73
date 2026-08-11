# -*- coding: utf-8 -*-
"""Per-field GT parasite count aligned to field_id used in dev_faint_multi_extract2 (dev fields 0..94, test 95..).
No GPU. Saves outputs/experiments/s3_filter/ngt_field.npz -> ngt[field_id]."""
import sys, json, random
import numpy as np
sys.path.append('/root/A_Dissertation'); from common.paths import RAW, OUTPUTS
RES=OUTPUTS/'experiments'/'s3_filter'
def fastmal_fields():
    fm=RAW/'ibadan'/'FASTMAL_THICK_FILMS'; o=[]
    for samp in sorted(p for p in fm.iterdir() if p.is_dir()):
        js=list(samp.glob('*.json'))
        if not js: continue
        for fld in json.loads(js[0].read_text())['rois']:
            ip=samp/fld['image_name']
            if not ip.exists(): continue
            b=[r for r in fld['roi'] if 'PARASITE' in r.get('type','') and 'CROWD' not in r.get('type','')]
            if b: o.append((ip,len(b)))
    return o
fields=sorted(fastmal_fields(),key=lambda z:str(z[0])); random.Random(0).shuffle(fields)
devF=fields[0::2]; testF=fields[1::2]
ngt=[n for (_,n) in devF]+[n for (_,n) in testF]     # index = field_id
np.savez(RES/'ngt_field.npz', ngt=np.array(ngt), n_dev=len(devF), n_test=len(testF))
print(f'saved ngt_field: {len(ngt)} fields, dev {len(devF)} (gt {sum(n for _,n in devF)}) test {len(testF)} (gt {sum(n for _,n in testF)})')
