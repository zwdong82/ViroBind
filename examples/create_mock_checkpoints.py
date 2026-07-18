#!/usr/bin/env python3
"""Create small random ViroBind checkpoints for software testing only."""

from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

import torch

from virobind import model as v5


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", default="examples/generated_assets/mock_checkpoints")
    parser.add_argument("--token-root", default="examples/generated_assets/residue_tokens")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--bilinear-dim", type=int, default=8)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    checkpoint_args = SimpleNamespace(
        seed=args.seed,
        feature_preset="ecfp_esmc",
        hidden_dim=args.hidden_dim,
        bilinear_dim=args.bilinear_dim,
        dropout=0.0,
        use_domain_adapter=0,
        token_root=args.token_root,
        token_max_tokens=1024,
        drug_ecfp_dim=2048,
        drug_maccs_dim=167,
        drug_rdkit_dim=208,
        drug_fusion_mode="residual_multiview",
        drug_view_mask="1,1,1",
        drug_aux_gate_init=-5.0,
        drug_aux_gate_max=0.10,
        protein_esmc_dim=1152,
        protein_desc_dim=574,
        protein_fusion_mode="esmc_only",
        pair_feature_mode="full_bilinear",
        residue_rank_only=False,
    )
    v5.configure_from_namespace(checkpoint_args)
    model = v5.ViroBindV5Decoupled(
        "ecfp_esmc",
        2048 + 167 + 208,
        1,
        1152,
        args.hidden_dim,
        args.bilinear_dim,
        0.0,
        False,
    )

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    payload = {
        "args": vars(checkpoint_args),
        "model": model.state_dict(),
        "synthetic": True,
        "warning": "Random software-test checkpoint; outputs have no scientific meaning.",
    }
    for name in ["virobind_classification.pt", "virobind_ranking.pt"]:
        torch.save(payload, outdir / name)
        print(f"Created synthetic checkpoint: {outdir / name}")


if __name__ == "__main__":
    main()
