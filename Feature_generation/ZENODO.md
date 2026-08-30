# Zenodo feature assets

The versioned ViroBind feature banks are distributed separately from GitHub.

> Publication placeholder: replace every `RECORD_ID` below with the finalized
> Zenodo record number before making a release.

- Record: `https://zenodo.org/records/RECORD_ID`
- DOI: `https://doi.org/10.5281/zenodo.RECORD_ID`

Expected assets:

```text
drug_combo_ecfp4_maccs_rdkit2d.pt
prot_esmc_fullseq.pt
prot_esmc_residue_tokens.tar.zst
SHA256SUMS
```

After download, arrange them as follows:

```text
Feature_generation/features/
├── drug/
│   └── drug_combo_ecfp4_maccs_rdkit2d.pt
└── protein/
    ├── prot_esmc_fullseq.pt
    └── prot_esmc_residue_tokens/
```

Verify every asset against the `SHA256SUMS` file supplied in the same Zenodo
record. The Zenodo deposit should also state the associated Git commit/tag,
software version, feature-generation environment, license, source-data rights
and whether any embedded identifiers or metadata are present.
