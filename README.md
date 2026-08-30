# ViroBind

ViroBind is a deep-learning model for drug–virus interaction prediction and
candidate prioritization. This repository contains feature-generation,
training, prediction and memory-safe virtual-screening workflows.

The software is released for noncommercial use, including academic research,
education and public-interest research, under the PolyForm Noncommercial
License 1.0.0. Commercial use is not licensed.

The [model card](MODEL_CARD.md) describes intended
use and scientific limitations. [Data and model use](DATA_AND_MODEL_USE.md)
clarifies anonymization and third-party rights.

**Official web server:** [https://lmmd.ecust.edu.cn/ViroBind/](https://lmmd.ecust.edu.cn/ViroBind/)

## Repository layout

```text
Feature_generation/ feature-bank generation scripts
Predictions/        external drug-library preprocessing guidance
Datasets/           final binary human-pretrain/virus-finetune split CSVs
source/virobind/    model runtime, prediction and screening implementation
examples/           runnable anonymized ID-only examples
outputs/            generated predictions (ignored by Git)
```

## Installation

The released environment was tested with Python 3.10.19, PyTorch 2.5.1 and
CUDA 12.1. Install the locked environment before installing the package:

```bash
conda create -n virobind python=3.10.19
conda activate virobind
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
```

`requirements.txt` locks the tested CUDA 12.1 build. CPU-only users must replace
the `torch` and `torchvision` CUDA wheels as described at the top of that file.

## Anonymous smoke test

The repository includes ID-only example pairs and a deterministic generator for
small synthetic feature banks. These tensors contain no molecular structure,
protein sequence or biological identity and are intended only to verify that
the prediction pipeline runs; their scores have no scientific meaning.

Generate small synthetic features and a mock checkpoint, then run:

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

## Feature assets

The released feature banks are distributed separately through Zenodo because
they are too large for normal Git history. Download links, checksums and the
expected local directory layout are documented in
[`Feature_generation/ZENODO.md`](Feature_generation/ZENODO.md). Replace the
record placeholder in that file after the Zenodo deposit has been published.

## Training

After downloading and arranging the Zenodo feature files, train a model with:

```bash
virobind-train \
  --mode scaffold \
  --out_root outputs/training
```

Use `--mode random` for the random split. Use `--dry_run 1` to validate the
dataset and feature mappings before training.

## CPI prediction

Input pairs and precomputed feature banks must follow the schemas used during
training. Run:

```bash
virobind-predict \
  --external-csv examples/pairs.csv \
  --drug-feat Feature_generation/features/drug/drug_combo_ecfp4_maccs_rdkit2d.pt \
  --prot-feat Feature_generation/features/protein/prot_esmc_fullseq.pt \
  --token-root Feature_generation/features/protein/prot_esmc_residue_tokens \
  --ckpts <TRAINED_CLASSIFICATION_CHECKPOINT.pt> \
  --out_dir outputs/predictions \
  --device cuda
```

## Large-library screening

```bash
virobind-screen \
  --library-csv examples/library.csv \
  --drug-feat examples/generated_assets/drug_features.pt \
  --protein-feat examples/generated_assets/protein_features.pt \
  --protein-csv examples/proteins.csv \
  --token-root examples/generated_assets/residue_tokens \
  --ckpts <TRAINED_RANKING_CHECKPOINT.pt> \
  --out-dir outputs/example_screening \
  --top-k 5 \
  --device cpu
```

The screening implementation memory-maps the drug feature tensor and processes
it in chunks. Use `--help` on either command for all options.

## Reproducibility boundary

The released binary split CSVs contain anonymous IDs and labels. Molecular
structures, protein sequences and identity mappings are not distributed in Git;
feature banks are released separately through Zenodo. Consequently, this
repository supports software verification and prediction with compatible
assets, but it cannot reconstruct the original source datasets from anonymous
IDs alone. See `Datasets/README.md` and
`Datasets/split_manifest.json` for the frozen split audit.

## Publishing checklist

1. Review data/model redistribution permissions.
2. Create a GitHub repository and push this source tree.
3. Publish the Zenodo feature deposit and replace every `RECORD_ID` placeholder.
4. Verify the uploaded feature files against the Zenodo `SHA256SUMS` manifest.
5. Add the finalized manuscript citation, DOI and `CITATION.cff`.

## Citation

The manuscript citation and DOI will be added when they become publicly
available. A `CITATION.cff` will be added at the same time; no provisional
citation is asserted by this release candidate.

## License

ViroBind is licensed under the
[PolyForm Noncommercial License 1.0.0](LICENSE). Academic research, education
and other noncommercial uses are permitted under its terms. Commercial use is
not licensed. This is a source-available noncommercial license, not an
OSI-approved open-source license.
