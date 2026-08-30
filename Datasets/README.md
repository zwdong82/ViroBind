# ID-only model-input datasets

This directory contains the final ID-only inputs used by the released software.
To avoid publishing molecular and protein identity metadata, every CSV is
reduced to exactly four columns:

```text
drug_id,prot_id,label,prot_domain
```

The two released split strategies are:

```text
random/
scaffold/
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

The public files contain only the frozen binary split inputs. Unsplit provenance
tables and uncertain-label training data are intentionally excluded from the
public repository.

`split_manifest.json` records the binary label semantics, row counts,
duplicate/conflicting-pair counts and train/validation/test overlap audit. Some
frozen non-random inputs contain exact duplicate rows and a small number of
drug-protein pairs with conflicting labels. They are documented rather than
silently changed so that the release remains identical to the model inputs.

## Reproducibility boundary

These ID-only files cannot regenerate molecular or protein features. Matching
feature assets are distributed separately through the project Zenodo record;
see `Feature_generation/ZENODO.md`. The public release cannot independently
reconstruct the original source datasets from anonymous IDs. Data-source
citations and any controlled-access procedure will be added with the finalized
manuscript.
