#!/usr/bin/env python3
"""Create anonymous synthetic feature banks for a ViroBind smoke test.

The generated tensors have the production dimensions but contain deterministic
random values. They do not encode molecules, proteins or biological identities,
and predictions made from them have no scientific meaning.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch


DRUG_DIM = 2048 + 167 + 208
PROTEIN_DIM = 1152


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", default="examples/pairs.csv")
    parser.add_argument("--proteins", default="examples/proteins.csv")
    parser.add_argument("--outdir", default="examples/generated_assets")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--token-length", type=int, default=32)
    args = parser.parse_args()

    pairs = pd.read_csv(args.pairs)
    proteins = pd.read_csv(args.proteins)
    required_pairs = {"drug_id", "prot_id", "label", "prot_domain"}
    required_proteins = {"prot_id", "protein_key", "pdb_id"}
    if not required_pairs.issubset(pairs.columns):
        raise ValueError(f"pairs CSV must contain {sorted(required_pairs)}")
    if not required_proteins.issubset(proteins.columns):
        raise ValueError(f"proteins CSV must contain {sorted(required_proteins)}")
    if args.token_length < 1:
        raise ValueError("--token-length must be positive")

    drug_ids = sorted(pairs["drug_id"].astype(int).unique().tolist())
    protein_ids = sorted(pairs["prot_id"].astype(int).unique().tolist())
    protein_rows = proteins.set_index("prot_id")
    missing = sorted(set(protein_ids) - set(protein_rows.index.astype(int)))
    if missing:
        raise ValueError(f"proteins CSV is missing prot_id values: {missing}")

    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    # Mimic the final concatenation: sparse binary ECFP4/MACCS followed by
    # standardized continuous RDKit2D descriptors.
    ecfp = (torch.rand(len(drug_ids), 2048, generator=generator) < 0.04).float()
    maccs = (torch.rand(len(drug_ids), 167, generator=generator) < 0.12).float()
    rdkit2d = torch.randn(len(drug_ids), 208, generator=generator)
    drug_features = torch.cat([ecfp, maccs, rdkit2d], dim=1)
    protein_features = torch.randn(len(protein_ids), PROTEIN_DIM, generator=generator)

    outdir = Path(args.outdir)
    token_root = outdir / "residue_tokens"
    token_dir = token_root / "tokens"
    token_dir.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "embeddings": drug_features,
            "drug_ids": drug_ids,
            "feature_order": ["ECFP4", "MACCS", "RDKit2D"],
            "synthetic": True,
        },
        outdir / "drug_features.pt",
    )

    protein_keys = [str(protein_rows.loc[pid, "protein_key"]) for pid in protein_ids]
    torch.save(
        {
            "embeddings": protein_features,
            "prot_ids": protein_ids,
            "protein_keys": protein_keys,
            "synthetic": True,
        },
        outdir / "protein_features.pt",
    )

    manifest = []
    for pid, protein_key in zip(protein_ids, protein_keys):
        relative = Path("tokens") / f"{pid}.pt"
        tokens = torch.randn(
            args.token_length, PROTEIN_DIM, generator=generator, dtype=torch.float32
        ).half()
        torch.save(
            {"prot_id": pid, "embeddings": tokens, "synthetic": True},
            token_root / relative,
        )
        manifest.append(
            {
                "prot_id": pid,
                "protein_key": protein_key,
                "length": args.token_length,
                "token_file": str(relative),
                "embedding_dim": PROTEIN_DIM,
            }
        )
    pd.DataFrame(manifest).to_csv(token_root / "manifest.csv", index=False)

    print(f"Created synthetic drug bank: {tuple(drug_features.shape)}")
    print(f"Created synthetic protein bank: {tuple(protein_features.shape)}")
    print(f"Created {len(manifest)} synthetic residue-token files under {token_root}")


if __name__ == "__main__":
    main()
