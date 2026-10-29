"""Verify published S4 hashes, unique pairs and frozen isolation axes."""
from pathlib import Path
import hashlib
import json
import pandas as pd


def audit(root=None):
    root=Path(root) if root else Path(__file__).parent/'scaffold_cluster_cold_protein'
    manifest=json.loads((root/'release_audit.json').read_text())
    sc=pd.read_csv(root/'audit/drug_scaffold_groups.csv',dtype=str).set_index('drug_id')
    cl=pd.read_csv(root/'audit/protein_cluster_groups.csv',dtype=str).set_index('prot_id')
    sets={}
    for name,expected in manifest['files'].items():
        path=root/name
        if hashlib.sha256(path.read_bytes()).hexdigest()!=expected['sha256']:
            raise ValueError(f'{name}: checksum differs from the audited release')
        df=pd.read_csv(path,dtype={'drug_id':str,'prot_id':str})
        if len(df)!=expected['rows'] or df.duplicated(['drug_id','prot_id']).any():
            raise ValueError(f'{name}: unexpected row count or nonunique pairs')
        wanted='human' if name.startswith('human') else 'virus'
        if not df.prot_domain.eq(wanted).all():raise ValueError(f'{name}: unexpected domain')
        if name in ['virus_val.csv','virus_test.csv'] and not df.label.isin([0,1]).all():
            raise ValueError(f'{name}: nondefinitive evaluation labels')
        sets[name]={'pairs':set(zip(df.drug_id,df.prot_id)),
                    'scaffolds':set(sc.loc[df.drug_id,'scaffold_group']),
                    'clusters':set(cl.loc[df.prot_id,'protein_cluster'])}
        if name in ['human_pretrain.csv','virus_finetune.csv']:
            if sc.loc[df.drug_id,'frozen_role'].isin(['1','2']).any():
                raise ValueError(f'{name}: reserved scaffold entered training')
            if cl.loc[df.prot_id,'frozen_role'].isin(['1','2']).any():
                raise ValueError(f'{name}: reserved cluster entered training')
    names=list(sets)
    for i,left in enumerate(names):
        for right in names[i+1:]:
            if {left,right}=={'human_pretrain.csv','virus_finetune.csv'}:
                continue  # Training domains may share compounds/scaffolds.
            for axis in ['pairs','scaffolds','clusters']:
                if sets[left][axis]&sets[right][axis]:
                    raise ValueError(f'{left}/{right}: overlapping {axis}')
    return {'status':'passed','source_version':manifest['source_version'],
            'rows':{k:v['rows'] for k,v in manifest['files'].items()},
            'precision_coverage':manifest['precision_coverage']}


if __name__=='__main__':print(json.dumps(audit(),indent=2))
