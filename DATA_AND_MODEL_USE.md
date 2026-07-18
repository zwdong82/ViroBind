# Data and model use

## Noncommercial scope

The original ViroBind software and release materials are made available under
the PolyForm Noncommercial License 1.0.0. Commercial use is not licensed.

## Third-party rights

The repository license does not create or transfer rights in third-party source
databases, molecular structures, protein sequences, identifiers or other
upstream materials. Users are responsible for complying with the terms of every
data source they access. Inclusion of an anonymous relationship table or a
derived feature-generation procedure must not be interpreted as permission to
redistribute an upstream dataset.

## Released CSV tables

The CSV tables under `Datasets/` contain anonymous integer identifiers, labels
and broad protein-domain tags. They do not include the private ID mappings,
molecular structures, protein sequences or biological accessions required to
reverse the anonymization or regenerate feature banks. Their statistics and
known duplicate/conflict counts are recorded in `Datasets/split_manifest.json`.

## Model weights

Scientific checkpoints are distributed separately as versioned release assets,
not as Git blobs. Their use is noncommercial and their integrity must be checked
against `Pretrained_models/ViroBind/SHA256SUMS`. Publication of a checkpoint
does not grant rights to any third-party training datum.

## Scientific limitation

ViroBind outputs are computational research scores. They are not experimental
evidence of binding, antiviral efficacy, safety or clinical benefit, and must
not be used as a substitute for laboratory validation or medical judgment.
