# ID-only model-input datasets

This directory contains the final ID-only inputs used by the released software.
To avoid publishing molecular and protein identity metadata, every CSV is
reduced to exactly four columns:

```text
drug_id,prot_id,label,prot_domain
```

The four released split strategies are:

```text
random/
scaffold/
cluster_cold_protein/
scaffold_cluster_cold_protein/
```

Each split contains:

```text
human_pretrain.csv
virus_finetune.csv
virus_val.csv
virus_test.csv
```

Despite its concise release name, `human_pretrain.csv` is the split-specific
mixed binary pretraining input: it contains cleaned human auxiliary rows plus
the corresponding virus fine-tuning rows. The other three files contain
virus-only fine-tuning, validation and testing inputs. Drug keys, protein keys,
UniProt IDs, SMILES and protein sequences are excluded.

The files still contain the complete anonymized ID-to-label relationships.

The two root files, `human_dti.csv` and `virus_dti.csv`, are unsplit provenance
snapshots. They include gray/uncertain label `-1`; they are not expected to equal
a concatenation of the released binary train, validation and test tables. The
four split directories are the frozen inputs for training/evaluation with a
selected split mode.

`split_manifest.json` records the split seeds, label semantics, row counts,
duplicate/conflicting-pair counts and train/validation/test overlap audit. Some
frozen non-random inputs contain exact duplicate rows and a small number of
drug-protein pairs with conflicting labels. They are documented rather than
silently changed so that the release remains identical to the model inputs.

## Reproducibility boundary

These ID-only files cannot regenerate molecular or protein features. The public
release supports software verification and training with matching locally held
features, but it cannot independently reconstruct the manuscript datasets from
anonymous IDs. Users must generate features from molecular structures and
protein sequences they are authorized to use. Data-source citations and any
controlled-access procedure will be added with the finalized manuscript.
