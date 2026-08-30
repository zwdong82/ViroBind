# ViroBind final-file manifest

This project deliberately separates software inputs from experiment outputs.

## Final model

The frozen production release contains two task-specific checkpoints. They are
named in this release as:

```text
Pretrained_models/ViroBind/virobind_classification.pt
Pretrained_models/ViroBind/virobind_ranking.pt
```

The classification checkpoint is intended for CPI prediction, and the ranking
checkpoint is intended for compound prioritization. Quantitative evaluation
and model-selection details belong to the manuscript and its associated
experiment record.

## Software files included in Git

- `source/virobind/model.py`: model runtime used by the released checkpoints.
- `source/virobind/base.py`: feature-bank, dataset and training utilities.
- `source/virobind/predict.py`: paired prediction and seed ensemble.
- `source/virobind/screen.py`: chunked large-library virtual screening.
- `pyproject.toml`, `requirements.txt`: installation and dependencies.
- `examples/`: minimal input schemas.
- `Datasets/split_manifest.json`: de-identified binary split statistics and overlap audit.
- `scripts/download_models.py`: checksum-verifying release-asset downloader.
- `LICENSE`: noncommercial software terms.
- `DATA_AND_MODEL_USE.md`: data, checkpoint and third-party-rights boundary.
- `Pretrained_models/ViroBind/MODEL_CARD.md`: intended use and limitations.
- `CONTRIBUTING.md`: privacy-safe contribution checks.

## Runtime feature files

The final model expects precomputed features with the exact training schema:

- ECFP4 + MACCS + RDKit2D drug features (`drug_combo_ecfp4_maccs_rdkit2d.pt`).
- ESMC global protein features (`prot_esmc_fullseq.pt`).
- ESMC residue-token files and `manifest.csv` (`prot_esmc_residue_tokens/`).

The binary feature banks are several gigabytes and are not committed to Git.
They are published as versioned Zenodo assets with checksums; see
`Feature_generation/ZENODO.md`. The repository also provides generation code so
users can build compatible features from inputs they are authorized to use.

## ID-only dataset inputs

`Datasets/` contains the complete final inputs for two split strategies:
random and scaffold. Every
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
