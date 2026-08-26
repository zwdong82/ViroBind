# ViroBind model card

## Model summary

ViroBind is a PyTorch drug-virus interaction model with separate binary
classification and compound-ranking heads. It combines molecular descriptors
with global and residue-level protein representations.

The release uses two task-specific checkpoints:

- `virobind_classification.pt`: CPI classification;
- `virobind_ranking.pt`: compound ranking.

## Intended use

- noncommercial academic and public-interest research;
- prioritizing compounds for subsequent experimental review;
- software-method comparison using compatible inputs;
- education and reproducibility testing.

## Out-of-scope use

- commercial screening or product development;
- clinical diagnosis, treatment selection or patient-level decision-making;
- claims of physical binding, antiviral activity, safety or efficacy without
  independent experimental evidence;
- interpreting scores from the anonymous synthetic example fixtures as
  scientific predictions.

## Inputs and outputs

The prediction command consumes paired anonymous IDs and matching precomputed
feature banks. The screening command scores a drug library against one or more
protein targets in chunks. Classification probabilities describe the model's
binary head; ranking scores are intended primarily for ordering compounds for a
given target and are not calibrated binding affinities.

## Limitations

- Performance may degrade for viral families, protein conformations, compound
  chemotypes or assay conditions unlike the training distribution.
- Anonymous public split files cannot reconstruct the original structures,
  sequences or feature banks.
- A high score is a hypothesis-generation signal, not mechanistic evidence.
- Dataset duplicates and a small number of conflicting input pairs are retained
  in the frozen release and disclosed in `Datasets/split_manifest.json`.
- Final quantitative evaluation tables and the definitive citation remain
  pending until the manuscript is finalized.

## Integrity and versioning

Always pair a checkpoint with the software release that published it and verify
the file using `SHA256SUMS`. Random checkpoints produced by
`examples/create_mock_checkpoints.py` are CI fixtures and must never be used for
scientific analysis.
