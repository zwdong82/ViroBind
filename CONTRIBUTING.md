# Contributing

Contributions that improve correctness, documentation, reproducibility or
noncommercial research use are welcome after the public repository is created.

Before proposing a change:

1. Do not add molecular structures, protein sequences, private identifiers,
   credentials, model weights or generated feature banks to Git.
2. Keep the public CSV schema limited to the documented anonymous columns.
3. Run `python -m unittest discover -s tests -v`.
4. Run the anonymous prediction, screening and training dry-run workflow in
   `.github/workflows/repository-checks.yml` for runtime changes.
5. Update checksums and the model card when publishing a new checkpoint.

By contributing, you agree that your contribution may be distributed under the
repository's PolyForm Noncommercial License 1.0.0. Do not submit code or data
that you do not have permission to contribute.
