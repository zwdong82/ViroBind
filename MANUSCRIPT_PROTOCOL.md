# Manuscript protocol and current release

The current S4 inputs use the selected fourth version,
`04_moderate_test_811_20261011`. Historical manuscript metrics are not regenerated
or validated by this code/data update.

| Manuscript location | Implementation |
|---|---|
| 2.3.1 | 2,048 ECFP4 + 167 MACCS + 208 RDKit2D; residual multiview encoder |
| 2.3.2, Note S2.2 | Frozen ESMC global/residue features; masked drug-conditioned attention; 1,024-residue central crop; bounded gate |
| 2.3.3–2.3.5 | Product/difference/projected bilinear interactions; normalization; human/virus adapters on the joint representation; separate heads |
| 2.3.6, S15/S16 | Gray target 0.20 and weight 0.03; BCE normalized by sum of sample weights |
| 2.3.7, Note S2.5 | Human-only pretraining, viral classification fine-tuning, ranking-head-only optimization |
| 2.4, Table S1 | AdamW; learning rates 1e-3/3e-5/1e-4; epochs 20/80/50; weight decay 1e-4; dropout 0.2; clip norm 5; patience 10 |
| Note S2.5 | Validation AP selects classification checkpoints; ranking starts from viral fine-tuning, rather than selecting an earlier pretraining checkpoint by default |
| Note S5.4 | 16 proteins, 4 positive and 4 negative pairs per protein, plus up to 2 gray pairs during classification |
| Eq 4/S17 | Protein-grouped BPR, equal protein weights, at most 512 positive/negative pairs per protein; no gray ranking labels |
| Note S1 | Ranking-head continuous scores; macro per-protein AUROC/AP need both classes; fixed-K precision includes N>=K pools and gives each protein equal weight |

After generating matching features, run:

```bash
virobind-train --mode scaffold_cluster_cold_protein --seed 42
```

Repeat with the intended independent seeds and retain configs, code revisions,
predictions and checkpoint hashes. Classification thresholds are selected on
validation and fixed for test evaluation. Do not select runs using test results.
Alternative architecture/loss flags remain available for documented ablations.

For retraining, fit RDKit2D normalization only on current human/viral training
compound IDs. Global ESMC generation defaults to full-sequence encoding with
boundary tokens excluded from mean pooling; sliding windows are an explicit
visualization/legacy option. Residue tokens are aligned to amino acids and
centrally cropped by the training loader. Feature-generation changes do not
retroactively update existing feature banks or historical checkpoints.

The legacy random/scaffold inputs are preserved and remain binary-only.
Their original files alone cannot reproduce gray-supervised experiments.

CPU regression tests verify implementation and data rules. This revision did
not retrain the full model or recalculate manuscript tables, baseline results,
endpoint sensitivity, bootstrap intervals or external validation. Those require
matched code, data, features and actual experimental runs.
