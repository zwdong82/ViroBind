# Runnable anonymized examples

`pairs.csv` contains ten ID-only examples from the released
scaffold-plus-cluster-cold-protein test split:

- protein 262: three positive and three negative pairs;
- protein 217: one positive and three negative pairs.

`proteins.csv` contains the two matching anonymous protein targets for the
screening input schema. `TARGET_217` and `TARGET_262` are example labels, not
biological identifiers or PDB accessions.

No SMILES, protein sequence, UniProt accession or original identity metadata is
included. `library.csv` is the matching anonymous screening-library schema.

`create_mock_assets.py` generates deterministic random tensors with the exact
drug, global-protein and residue-token dimensions expected by the checkpoints.
They are only for a software smoke test and must not be interpreted as molecular
or protein representations.

After installing ViroBind and placing the released checkpoint files under
`Pretrained_models/ViroBind/`, run:

```bash
python examples/create_mock_assets.py

virobind-predict \
  --external-csv examples/pairs.csv \
  --drug-feat examples/generated_assets/drug_features.pt \
  --prot-feat examples/generated_assets/protein_features.pt \
  --token-root examples/generated_assets/residue_tokens \
  --ckpts Pretrained_models/ViroBind/virobind_classification.pt \
  --out-dir outputs/example_prediction \
  --device cpu
```

The generated assets are ignored by Git and can be recreated at any time.

Continuous integration also creates small random checkpoints so the complete
prediction and screening commands can be tested without downloading scientific
release weights:

```bash
python examples/create_mock_checkpoints.py
```

These checkpoints are software fixtures only; their predictions have no
scientific meaning and must never be used for analysis.

The same assets can validate the training data/feature plumbing without running
optimization:

```bash
virobind-train \
  --token_root examples/generated_assets/residue_tokens \
  --pretrain_csv examples/pairs.csv \
  --train_csv examples/pairs.csv \
  --val_csv examples/pairs.csv \
  --test_csv examples/pairs.csv \
  --ecfp_path examples/generated_assets/drug_features.pt \
  --prot_path examples/generated_assets/protein_features.pt \
  --out_root outputs/example_training \
  --dry_run 1
```
