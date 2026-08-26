# Pretrained models

`ViroBind/` contains two task-specific ViroBind checkpoints and their SHA-256
checksums: one for CPI classification and one for compound ranking. Large `.pt`
files remain local and should be uploaded as GitHub Release assets rather than
ordinary Git blobs.

The root `scripts/download_models.py` command downloads release assets and
verifies them against the committed checksum manifest once the final GitHub
Release URL is available.

The upstream ESMC-600M parameters are downloaded by the `esm` package and are
not redistributed here.
