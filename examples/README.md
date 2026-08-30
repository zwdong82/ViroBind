# Runnable anonymized examples

`pairs.csv` contains ten ID-only examples for checking the public input schema:

- protein 262: three positive and three negative pairs;
- protein 217: one positive and three negative pairs.

`proteins.csv` contains the two matching anonymous protein targets for the
screening input schema. `TARGET_217` and `TARGET_262` are example labels, not
biological identifiers or PDB accessions.

No SMILES, protein sequence, UniProt accession or original identity metadata is
included. `library.csv` is the matching anonymous screening-library schema.

`create_mock_assets.py` generates deterministic random tensors with the exact
drug, global-protein and residue-token feature schemas expected by the checkpoints.
They are only for a software smoke test and must not be interpreted as molecular
or protein representations.

After installing ViroBind, generate the synthetic assets and mock checkpoints:

```bash
python examples/create_mock_assets.py
python examples/create_mock_checkpoints.py

virobind-predict \
  --external-csv examples/pairs.csv \
  --drug-feat examples/generated_assets/drug_features.pt \
  --prot-feat examples/generated_assets/protein_features.pt \
  --token-root examples/generated_assets/residue_tokens \
  --ckpts examples/generated_assets/mock_checkpoints/virobind_classification.pt \
  --out-dir outputs/example_prediction \
  --device cpu
```

The generated assets are ignored by Git and can be recreated at any time.

Continuous integration also creates these small random checkpoints so the
prediction and screening commands can be tested:

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
