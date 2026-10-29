# ID-only training and evaluation inputs

Pair CSV schema: `drug_id,prot_id,label,prot_domain`.

## Current S4 release

`scaffold_cluster_cold_protein/` contains the selected fourth version
`04_moderate_test_811_20261011`, audited on 2026-10-29. Frozen source inputs
are preserved. The release excludes molecular structures, protein sequences
and biological identity mappings.

| File | Records | Positive | Negative | Gray |
|---|---:|---:|---:|---:|
| human_pretrain.csv | 1,383,761 | 488,083 | 307,395 | 588,283 |
| virus_finetune.csv | 46,199 | 14,736 | 20,079 | 11,384 |
| virus_val.csv | 4,352 | 1,813 | 2,539 | 0 |
| virus_test.csv | 4,352 | 2,793 | 1,559 | 0 |

Human pretraining contains human proteins only. Classification training retains
gray label -1; ranking and evaluation use definitive labels 0 and 1 only.
Viral hard-label train/validation/test sizes are 34,815/4,352/4,352.

All four source inputs already contained zero duplicates and zero conflicting
hard-label pairs. The export independently verifies this. The loader also
removes duplicate records, excludes every row of a pair with conflicting 0/1
labels and rejects unresolved gray/hard mixtures.

The test pool contains 20 proteins; Precision@10/@50 cover 20/14 proteins.
It uses disclosed frozen label/coverage-conditioned sampling, without model
predictions. It is neither a natural-prevalence sample nor independent external
validation. Old manuscript scores are not validated on this new pool by simply
replacing the files.

`release_audit.json` records source/output hashes, counts and isolation checks.
`audit/` contains anonymous scaffold/cluster group IDs with frozen roles:
0=train, 1=validation, 2=test, -1=not assigned by the viral role map. Reserved
axes remain excluded even for held-out candidates outside the selected pool.

Verify the complete published release from the repository root:

```bash
python Datasets/audit_s4.py
```

## Preserved legacy inputs

`random/` and `scaffold/` retain their existing binary CSVs unchanged. Their
`human_pretrain.csv` files contain mixed rows; the current loader explicitly
selects human rows for auxiliary pretraining. Their historical duplicate counts
remain in `split_manifest.json`. `remove_conflicting_pairs.py` is the legacy
cleanup utility for those two releases; it does not modify the current S4.

## Feature correspondence

Anonymous IDs require matching feature banks and cannot reconstruct original
structures or sequences. Generate features from authorized private identity
tables. Fit RDKit2D normalization on current human and viral training IDs only,
as described in `Feature_generation/README.md`.
