# Formal feature generation

The final ViroBind model uses exactly five feature types and this directory
contains exactly five generation scripts.

## Input tables

Drug scripts require a private drug table with, at minimum:

```csv
drug_id,drug_key,SMILES_ChargeAware
0,drug_0,CCO
```

Protein scripts require a private protein table with, at minimum:

```csv
prot_id,protein_key,UniProt_ID,Protein_Sequence
0,protein_0,P00000,MKT...
```

The real SMILES and sequences are not distributed in this repository.

## Five feature generators

### 1. ECFP4

```bash
python Feature_generation/generate_ecfp4.py \
  --input private/drug2id.csv \
  --output features/drug_ecfp4.pt
```

Output: `drug_ecfp4.pt`, shape `[n_drugs, 2048]`, Morgan radius 2.

### 2. MACCS

```bash
python Feature_generation/generate_maccs.py \
  --input private/drug2id.csv \
  --output features/drug_maccs.pt
```

Output: `drug_maccs.pt`, shape `[n_drugs, 167]`.

### 3. RDKit2D

```bash
python Feature_generation/generate_rdkit2d.py \
  --input private/drug2id.csv \
  --output features/drug_rdkit2d.pt \
  --ecfp4 features/drug_ecfp4.pt \
  --maccs features/drug_maccs.pt \
  --combo-output features/drug_combo_ecfp4_maccs_rdkit2d.pt
```

Output: `drug_rdkit2d.pt`, shape `[n_drugs, 208]`. The bundled
`rdkit2d_descriptor_meta.json` fixes descriptor names and training-set
normalization. With the three optional combination arguments, this script also
outputs the final `[n_drugs, 2423]` drug bank in the required order ECFP4 +
MACCS + RDKit2D.

### 4. ESMC global protein representation

```bash
python Feature_generation/generate_esmc_global.py \
  --input private/protein2id.csv \
  --seq_col Protein_Sequence \
  --outdir features/protein_global
```

Output: `features/protein_global/prot_esmc_fullseq.pt`, shape
`[n_proteins, 1152]`. Long sequences use 1024-residue windows with stride 512,
followed by mean pooling.

### 5. ESMC residue representation

```bash
python Feature_generation/generate_esmc_residue.py \
  --input private/protein2id.csv \
  --seq-col Protein_Sequence \
  --outdir features/protein_residue_tokens
```

Outputs:

```text
features/protein_residue_tokens/manifest.csv
features/protein_residue_tokens/tokens/<prot_id>.pt
```

Each token file contains an `[protein_length, 1152]` residue-aligned tensor.

## Final model inputs

When all generators are run, the local ignored `features/` directory has the
layout below. A working checkout may retain only the final concatenated drug
bank because the three standalone component files are not loaded at runtime.
Generated tensors are ignored by Git because the serialized objects can contain
private molecular or protein metadata.

The five feature types are stored as follows:

```text
features/
├── drug/
│   ├── drug_ecfp4.pt                         # ECFP4, 2048 dimensions
│   ├── drug_maccs.pt                         # MACCS, 167 dimensions
│   ├── drug_rdkit2d.pt                       # RDKit2D, 208 dimensions
│   └── drug_combo_ecfp4_maccs_rdkit2d.pt    # concatenated final drug input, 2423 dimensions
└── protein/
    ├── prot_esmc_fullseq.pt                   # ESMC global representation, 1152 dimensions
    └── prot_esmc_residue_tokens/              # ESMC residue representations
        ├── manifest.csv
        └── tokens/*.pt
```

Although five feature types are generated, the three drug fingerprints are
concatenated before training. Therefore `virobind-train` directly reads only
these three paths:

```text
Feature_generation/features/drug/drug_combo_ecfp4_maccs_rdkit2d.pt
Feature_generation/features/protein/prot_esmc_fullseq.pt
Feature_generation/features/protein/prot_esmc_residue_tokens/
```

The three standalone drug files are intermediate outputs used to construct and
audit the final combined bank. They are not separately loaded by the final
training run and need not be retained after their ID order and combination have
been verified. Large generated tensors are runtime assets and are not committed
to Git.
