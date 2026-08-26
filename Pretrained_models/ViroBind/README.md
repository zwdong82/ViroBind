# Model weights

The local release bundle contains two task-specific checkpoints:

- `virobind_classification.pt`: use its `cls` output for CPI binary prediction.
- `virobind_ranking.pt`: use its `rank` output for compound prioritization.

The two files are intended for their named tasks. Quantitative evaluation and
model-selection details are reported in the manuscript and its associated
experiment record.

They are intentionally ignored by Git. Upload them to a GitHub Release and keep
`SHA256SUMS` in both the repository and the release assets. Verify downloads with:

```bash
cd Pretrained_models/ViroBind
sha256sum -c SHA256SUMS
```

After the GitHub repository and release tag exist, replace the placeholder URL
below and run the checksum-verifying downloader from the repository root:

```bash
python scripts/download_models.py \
  --base-url https://github.com/OWNER/REPOSITORY/releases/download/v0.1.0
```

Do not commit the `.pt` files to normal Git history.

Both checkpoints use ESMC global and residue-token protein representations.
Inference therefore requires matching drug, protein and residue-token feature
files; the feature schema must match the checkpoint training configuration.

See `MODEL_CARD.md` for intended use, out-of-scope use and scientific
limitations.
