# ViroBind Python package

This directory is the single authoritative implementation of the released
model:

- `model.py`: ViroBind architecture, checkpoint configuration and training CLI.
- `base.py`: feature-bank loading, ID mapping, datasets, metrics and loaders.
- `predict.py`: CPI pair classification with the classification checkpoint.
- `screen.py`: chunked target-centered compound ranking with the ranking checkpoint.
- `__init__.py`: package metadata.

The commands registered by `pyproject.toml` are:

```text
virobind-train
virobind-predict
virobind-screen
```

All default paths are resolved from the repository root. Generated feature
banks and model weights remain local runtime assets and are ignored by Git.
