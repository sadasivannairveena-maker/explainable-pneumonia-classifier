from __future__ import annotations
import argparse, hashlib, json, random, re
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from PIL import Image, ImageOps
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

EXT={'.jpeg','.jpg','.png'}

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()

def dhash(path):
    with Image.open(path) as im:
        a=np.asarray(ImageOps.exif_transpose(im).convert('L').resize((9,8),Image.Resampling.LANCZOS),dtype=np.int16)
    out=0
    for bit in (a[:,1:]>a[:,:-1]).flat: out=(out<<1)|int(bit)
    return out

def inventory(raw):
    rows=[]
    for split in ('train','val','test'):
        for name,label in (('NORMAL',0),('PNEUMONIA',1)):
            folder=raw/split/name
            if not folder.is_dir(): raise FileNotFoundError(folder)
            for p in sorted(folder.iterdir()):
                if p.is_file() and p.suffix.lower() in EXT:
                    m=re.match(r'person(\d+)_',p.name.lower()) if label else None
                    rows.append(dict(path=str(p.resolve()),relative_path=str(p.relative_to(raw)),filename=p.name,original_split=split,label=label,class_name=name,sha256=digest(p),patient_id=f'person_{m.group(1)}' if m else None))
    return pd.DataFrame(rows)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--config',default='configs/config.yaml'); a=ap.parse_args()
    cfg=yaml.safe_load(Path(a.config).read_text()); seed=cfg['seed']; random.seed(seed); np.random.seed(seed)
    raw=Path(cfg['raw_dir']); out=Path(cfg['manifest_dir']); reports=Path(cfg['report_dir']); out.mkdir(parents=True,exist_ok=True); (reports/'figures').mkdir(parents=True,exist_ok=True)
    original=inventory(raw)
    if len(original)!=5856: raise RuntimeError(f'Expected 5856 canonical images, got {len(original)}. Check raw_dir.')
    exact=original[original.duplicated('sha256',keep=False)].copy()
    exact.to_csv(reports/'exact_duplicates.csv',index=False)
    cross_label=original.groupby('sha256')['label'].nunique(); ambiguous=set(cross_label[cross_label>1].index)
    if ambiguous: raise RuntimeError(f'{len(ambiguous)} exact-hash groups have conflicting labels; review before proceeding.')
    unique=original.drop_duplicates('sha256',keep='first').copy().reset_index(drop=True)
    # A patient identifier is only a grouping proxy, not verified patient identity.
    # For NORMAL, individual images have no reliable patient identifier.
    unique['group_key']=[f'pneumonia_{pid}' if pd.notna(pid) else f'normal_{sha}' for pid,sha in zip(unique.patient_id,unique.sha256)]
    # If exact matches connect multiple patient IDs, merge those groups using union-find.
    all_keys=[f'pneumonia_{pid}' if pd.notna(pid) else f'normal_{sha}' for pid,sha in zip(original.patient_id,original.sha256)]
    parent={g:g for g in all_keys}
    def find(g):
        while parent[g]!=g:
            parent[g]=parent[parent[g]]; g=parent[g]
        return g
    def union(x,y):
        x,y=find(x),find(y)
        if x!=y: parent[max(x,y)]=min(x,y)
    for _,sub in original.groupby('sha256'):
        keys=[]
        for r in sub.itertuples(): keys.append(f'pneumonia_{r.patient_id}' if pd.notna(r.patient_id) else f'normal_{r.sha256}')
        for k in keys[1:]: union(keys[0],k)
    unique['group_key']=unique.group_key.map(find)
    groups=unique.groupby('group_key').agg(label=('label','first'),n=('label','size')).reset_index()
    if (unique.groupby('group_key')['label'].nunique()>1).any(): raise RuntimeError('Conflicting labels in a group')
    # Split groups, not images; evaluate actual image-level percentages after splitting.
    g_train,g_hold=train_test_split(groups,test_size=.30,stratify=groups.label,random_state=seed)
    g_val,g_test=train_test_split(g_hold,test_size=.50,stratify=g_hold.label,random_state=seed+1)
    lookup={**{k:'train' for k in g_train.group_key},**{k:'val' for k in g_val.group_key},**{k:'test' for k in g_test.group_key}}
    unique['split']=unique.group_key.map(lookup)
    assert unique.split.notna().all()
    assert unique.groupby('group_key')['split'].nunique().max()==1
    assert unique.sha256.nunique()==len(unique)
    for s in ('train','val','test'):
        subset=unique[unique.split==s]
        assert set(subset.label)=={0,1},f'Missing class in {s}'
        subset.to_csv(out/f'{s}.csv',index=False)
    unique.to_csv(out/'all_images.csv',index=False)
    counts=pd.crosstab(unique.split,unique.class_name).reindex(['train','val','test'])
    counts.plot.bar(rot=0,figsize=(8,5),title='Deduplicated images by split')
    plt.ylabel('Images'); plt.tight_layout(); plt.savefig(reports/'figures'/'class_distribution.png',dpi=160); plt.close()
    fig,axes=plt.subplots(2,4,figsize=(12,6))
    for row,(name,label) in enumerate((('NORMAL',0),('PNEUMONIA',1))):
        samples=unique[unique.label==label].sample(n=4,random_state=seed)
        for col,path in enumerate(samples.path):
            with Image.open(path) as im: axes[row,col].imshow(im.convert('L'),cmap='gray')
            axes[row,col].set_title(name); axes[row,col].axis('off')
    plt.tight_layout(); plt.savefig(reports/'figures'/'sample_images.png',dpi=160); plt.close()
    ratios=unique.split.value_counts(normalize=True)
    summary={'original_images':len(original),'deduplicated_images':len(unique),'removed_exact_duplicate_rows':len(original)-len(unique),'groups':len(groups),'counts':counts.to_dict(orient='index'),'split_proportions':{s:round(float(ratios[s]),4) for s in ('train','val','test')},'exact_duplicate_group_overlap':0,'limitations':['Pneumonia filename IDs are unverified patient proxies','NORMAL images have no reliable patient identifiers','Near-duplicate candidate pairs have not been adjudicated','Group-stratified split may deviate from target image proportions','Original benchmark test partition was not preserved']}
    (reports/'data_audit.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2)); print('\nDONE: Split manifests and figures saved. Review proportions before training.')
if __name__=='__main__': main()
