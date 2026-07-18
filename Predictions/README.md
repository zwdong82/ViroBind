# Prediction and screening

The authoritative prediction and screening implementations are installed from
`source/virobind/`:

```text
virobind-predict   CPI pair classification
virobind-screen    memory-safe compound ranking
```

`preprocess_drug_library.py` prepares an external molecule table before the
three drug feature generators are run.

Example:

```bash
python Predictions/preprocess_drug_library.py \
  --input private/chembl.csv \
  --outdir outputs/preprocessed_library
```

The input is the semicolon-separated ChEMBL compound-table export described by
the script's `--help`. The main output is `chembl_drug2id.csv`, which can be
passed to the ECFP4, MACCS and RDKit2D generators. Molecular structures and
metadata produced by this command are private runtime inputs and must be
reviewed before publication.

The commands require feature banks whose row order and dimensions exactly match
the checkpoint metadata. See the root README and `FINAL_FILES.md`.
