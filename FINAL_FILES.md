# ViroBind final-file manifest

This project deliberately separates software inputs from experiment outputs.

## Final model

The frozen production release contains two V5H global-plus-residue-token
checkpoints with explicit task semantics. They are named in this release as:

```text
Pretrained_models/ViroBind/virobind_classification.pt
Pretrained_models/ViroBind/virobind_ranking.pt
```

The seed-3407 classification checkpoint was selected by validation
classification AUPR. The seed-42 ranking checkpoint has the strongest
validation pooled ranking AUC/AUPR among the three final seeds. The manuscript's
target-centered per-protein metrics remain three-seed summary results.

## Software files included in Git

- `source/virobind/model.py`: final V5H architecture and complete training CLI.
- `source/virobind/base.py`: feature-bank, dataset and training utilities.
- `source/virobind/predict.py`: paired prediction and seed ensemble.
- `source/virobind/screen.py`: chunked large-library virtual screening.
- `pyproject.toml`, `requirements.txt`: installation and dependencies.
- `examples/`: minimal input schemas.
- `Datasets/split_manifest.json`: de-identified split statistics and overlap audit.
- `scripts/download_models.py`: checksum-verifying release-asset downloader.
- `LICENSE`: noncommercial software terms.
- `DATA_AND_MODEL_USE.md`: data, checkpoint and third-party-rights boundary.
- `Pretrained_models/ViroBind/MODEL_CARD.md`: intended use and limitations.
- `CONTRIBUTING.md`: privacy-safe contribution checks.

The implementation contains the optional virus/domain adapter. However, the
released `paper/model/results/main` checkpoints record `use_domain_adapter=0`;
their active architecture is residue-token attention without the domain
adapter. This distinction is preserved in the release documentation.

## Runtime feature files

The final model expects precomputed features with the exact training schema:

- ECFP4 + MACCS + RDKit2D drug features (`drug_combo_ecfp4_maccs_rdkit2d.pt`).
- ESMC global protein features (`prot_esmc_fullseq.pt`).
- ESMC residue-token files and `manifest.csv` (`prot_esmc_residue_tokens/`).

The binary feature banks are several gigabytes, contain private molecular or
protein metadata, and are not committed or published. The repository provides
their generation code and schema documentation so users can generate features
from their own inputs.

## ID-only dataset inputs

`Datasets/` contains the complete final inputs for four split strategies:
random, scaffold, cluster-cold-protein and scaffold+cluster-cold-protein. Every
CSV is reduced to `drug_id,prot_id,label,prot_domain`; SMILES, sequences,
UniProt IDs and other identity metadata are excluded. Each split provides the
human pretraining, virus fine-tuning, validation and test files.

## Explicitly excluded from the software repository

The following are experiment artifacts, not runtime requirements:

- `results/` metrics, predictions, logs, ablation outputs and intermediate runs;
- cached baseline features and third-party model outputs;
- temporary files, Python caches and cluster job logs;
- manuscript figures, tables and word-processing files.

These files remain in the original research directory and are not deleted.
