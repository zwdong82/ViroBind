# ViroBind

ViroBind is a deep-learning model for drug–virus interaction prediction and
candidate prioritization. This repository contains feature-generation,
training, prediction and chunked virtual-screening workflows.

The software is released for noncommercial use, including academic research,
education and public-interest research, under the PolyForm Noncommercial
License 1.0.0. Commercial use is not licensed.

**Official web server:** [https://lmmd.ecust.edu.cn/ViroBind/](https://lmmd.ecust.edu.cn/ViroBind/)

## Repository layout

```text
Feature_generation/ feature-bank generation scripts
Predictions/        compound-library preprocessing scripts and guidance
Datasets/           random, scaffold and scaffold-cluster cold-protein splits
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

## Quick start

The quick start demonstrates prediction using anonymous example pairs,
synthetic feature banks and randomly initialized checkpoints. For scientific
predictions, use molecular and protein features with trained checkpoints.

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

## Data and feature inputs

Precomputed drug features are available through
[Zenodo](https://zenodo.org/records/22169453). Download details are provided in
[`Feature_generation/ZENODO.md`](Feature_generation/ZENODO.md).

Training requires a combined ECFP4 + MACCS + RDKit2D drug feature bank,
an ESMC global protein feature bank and residue-token files. See
[`Feature_generation/README.md`](Feature_generation/README.md) for input
schemas, feature-generation commands and the expected directory layout.

The split CSVs in `Datasets/` use the columns
`drug_id,prot_id,label,prot_domain`. Labels are 1 (positive), 0 (negative)
and -1 (gray, in S4 training inputs). The CSVs contain anonymous IDs rather
than molecular structures or protein sequences; IDs must match the feature
banks used for training and prediction.

## Training

After preparing the matching drug, protein and residue-token feature files,
train a model with:

```bash
virobind-train \
  --mode scaffold_cluster_cold_protein \
  --use_domain_adapter 1 \
  --out_root outputs/training
```

The default split is `scaffold_cluster_cold_protein` (S4). Use `--mode random`
or `--mode scaffold` for the other split strategies. Use `--dry_run 1` to
validate dataset and feature mappings before training.

Training proceeds through human-only auxiliary pretraining, viral
classification fine-tuning and ranking-head optimization, with learning rates
1e-3, 3e-5 and 1e-4, respectively. Classification training uses gray-label weak
supervision. Ranking uses definitive positive and negative pairs and starts
from the selected viral classification fine-tuning checkpoint.

### Model architecture and domain adaptation

With `--use_domain_adapter 1`, ViroBind constructs and layer-normalizes the
joint compound-protein interaction representation before selecting the human
or virus residual adapter.

The final ranking stage (`--rank_train_scope rank_head`) freezes the upstream
model. The `rank_head_adapter` and `rank_head_pair` ablations additionally
update the virus-domain adapter, or that adapter and the interaction layers,
respectively.

### Fixed-K ranking evaluation

Evaluation follows Supplementary Note S1: only definitive labels 0 and 1 enter
the candidate pools. Precision@10 and Precision@50 include proteins with at
least 10 or 50 eligible candidates, respectively. For each eligible protein,
the positive count among the top K candidates is divided by K, then the
per-protein values are macro-averaged with equal weight.

Single-class proteins remain eligible for Precision@K when their pool is
large enough. Per-protein AUROC and AUPRC instead require both label classes.
The dual-head model ranks by the ranking-head logit rather than the
classification probability or a potentially saturated sigmoid score.

Metric outputs include `n_proteins_p_at_10` and `n_proteins_p_at_50`
(also under the `rank_` prefix) to report the actual coverage of each metric.
If no protein qualifies for a given K, its metric is NaN and its count is zero.

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

## Citation

When referencing the software, include the
[ViroBind repository](https://github.com/zwdong82/ViroBind) and the commit
identifier used for the analysis. Cite the
[Zenodo feature record](https://doi.org/10.5281/zenodo.22169453) when using
its precomputed drug features.

## License

ViroBind is licensed under the
[PolyForm Noncommercial License 1.0.0](LICENSE). Academic research, education
and other noncommercial uses are permitted under its terms. Commercial use is
not licensed. This is a source-available noncommercial license, not an
OSI-approved open-source license.
