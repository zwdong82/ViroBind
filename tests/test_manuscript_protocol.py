"""Protocol and data regressions using anonymous synthetic inputs."""
import importlib.util
import sys
import tempfile
import unittest
import types
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

AVAILABLE=all(importlib.util.find_spec(n) for n in ['torch','pandas','numpy','sklearn'])
if AVAILABLE:
 import pandas as pd
 import numpy as np
 import torch
 sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'source'))
 from virobind import base,model
 from virobind.data_integrity import clean_pair_records

@unittest.skipUnless(AVAILABLE,'Model dependencies unavailable')
class ManuscriptProtocolTests(unittest.TestCase):
 def frame(self,rows):
  return pd.DataFrame(rows,columns=['drug_id','prot_id','label','prot_domain'])

 def test_duplicate_and_conflicting_pairs(self):
  frame=self.frame([[1,1,1,'human'],[1,1,1,'human'],[2,1,0,'human'],[2,1,1,'human'],[2,1,-1,'human'],[3,2,-1,'virus']])
  clean,stats=clean_pair_records(frame)
  self.assertEqual(stats['duplicate_rows_removed'],1)
  self.assertEqual(stats['conflicting_pairs_removed'],1)
  self.assertEqual(stats['conflicting_rows_removed'],3)
  self.assertEqual(clean.drug_id.tolist(),['1','3'])
  again,stats2=clean_pair_records(clean)
  pd.testing.assert_frame_equal(clean,again)
  self.assertEqual(stats2['duplicate_rows_removed'],0)

 def test_ambiguous_gray_hard_evidence_is_not_guessed(self):
  with self.assertRaisesRegex(ValueError,'Unresolved gray/hard'):
   clean_pair_records(self.frame([[1,1,1,'virus'],[1,1,-1,'virus']]))

 def test_pretraining_excludes_viral_rows_and_ranking_excludes_gray(self):
  class Bank:
   x=torch.zeros(4,3)
   def lookup(self,keys):return int(keys[0])-1
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'pairs.csv'
   self.frame([[1,1,1,'human'],[2,1,0,'human'],[3,2,1,'virus'],[4,2,-1,'virus']]).to_csv(p,index=False)
   human=base.ViroBindPairDataset(str(p),Bank(),None,Bank(),'ecfp_esmc','human',False,'gray',domain_filter=0)
   rank=base.ViroBindPairDataset(str(p),Bank(),None,Bank(),'ecfp_esmc','rank',True,'binary',domain_filter=1)
   self.assertEqual(human.y.tolist(),[1,0]);self.assertEqual(human.domain.tolist(),[0,0])
   self.assertEqual(rank.y.tolist(),[1]);self.assertEqual(rank.domain.tolist(),[1])

 def test_defaults_match_manuscript(self):
  with patch.object(sys,'argv',['train']):args=base.parse_args()
  v5,_=model.parse_v5_args([])
  self.assertEqual(args.mode,'scaffold_cluster_cold_protein')
  self.assertEqual(args.use_domain_adapter,1)
  self.assertEqual(args.train_lr,3e-5)
  self.assertEqual(args.gray_per_protein,2)
  self.assertEqual((args.pretrain_label_mode,args.train_label_mode),('gray','gray'))
  self.assertEqual(v5.rank_train_lr,1e-4)
  self.assertEqual(v5.cls_checkpoint_policy,'finetune')
  self.assertEqual(v5.rank_train_scope,'rank_head')

 def test_gray_weighted_classification_matches_equation(self):
  logits=torch.tensor([0.2,-1.0,0.5]);y=torch.tensor([1.0,0.0,-1.0])
  args=SimpleNamespace(gray_target=.2,gray_cls_weight=None,gray_bce_weight=.03)
  weights=torch.tensor([1.,1.,.03])
  expected=(torch.nn.functional.binary_cross_entropy_with_logits(logits,torch.tensor([1.,0.,.2]),pos_weight=torch.tensor([2.]),reduction='none')*weights).sum()/weights.sum()
  torch.testing.assert_close(model.cls_bce_loss(logits,y,2,args),expected)

 def test_descriptor_scaling_excludes_heldout_outlier(self):
  p=Path(__file__).resolve().parents[1]/'Feature_generation/generate_rdkit2d.py'
  if importlib.util.find_spec('rdkit') is None:self.skipTest('RDKit unavailable')
  spec=importlib.util.spec_from_file_location('rdkit_generator_test',p)
  module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
  raw=np.array([[1.,np.nan],[3.,4.],[10000.,99999.]],dtype=np.float32)
  stats=module.fit_standardization(raw,np.array([True,True,False]))
  self.assertEqual(stats['mean'],[2.,4.]);self.assertEqual(stats['median'],[2.,4.])
  self.assertEqual(stats['std'],[1.,1.])

 def test_global_generator_normalizes_full_sequences_and_excludes_boundary_tokens(self):
  calls=[]
  class FakeESMC:
   @classmethod
   def from_pretrained(cls,name):return cls()
   def to(self,device):return self
   def eval(self):return self
   def encode(self,protein):calls.append(protein.sequence);return protein.sequence
   def logits(self,seq,cfg):
    values=torch.tensor([9999.]+list(range(1,len(seq)+1))+[9999.])
    return SimpleNamespace(embeddings=values.reshape(1,-1,1))
  esmc=types.ModuleType('esm.models.esmc');esmc.ESMC=FakeESMC
  api=types.ModuleType('esm.sdk.api');api.ESMProtein=lambda sequence:SimpleNamespace(sequence=sequence)
  api.LogitsConfig=lambda **kwargs:kwargs
  stubs={'esm':types.ModuleType('esm'),'esm.models':types.ModuleType('esm.models'),
         'esm.models.esmc':esmc,'esm.sdk':types.ModuleType('esm.sdk'),'esm.sdk.api':api}
  path=Path(__file__).resolve().parents[1]/'Feature_generation/generate_esmc_global.py'
  with tempfile.TemporaryDirectory() as d,patch.dict(sys.modules,stubs):
   src=Path(d)/'proteins.csv'
   pd.DataFrame({'prot_id':[1,2],'Sequence':['>example\na b','a'*1100]}).to_csv(src,index=False)
   spec=importlib.util.spec_from_file_location('global_generator_test',path)
   module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
   with patch.object(sys,'argv',['generate','--input',str(src),'--outdir',d,'--dtype','fp32']):module.main()
   bank=torch.load(Path(d)/'prot_esmc_fullseq.pt',weights_only=False)
   self.assertEqual(calls,['AB','A'*1100])
   self.assertEqual(bank['embeddings'].flatten().tolist(),[1.5,550.5])
   self.assertEqual(bank['sequence_mode'],'full')

if __name__=='__main__':unittest.main()
