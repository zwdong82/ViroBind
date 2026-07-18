#!/usr/bin/env python3
"""Run frozen V5H checkpoints on an external DTI CSV and ensemble seed outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from . import base as data_utils
from . import model as v5

PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parents[1]
DEFAULT_DRUG_FEATURE = PROJECT_ROOT / "Feature_generation/features/drug/drug_combo_ecfp4_maccs_rdkit2d.pt"
DEFAULT_PROTEIN_FEATURE = PROJECT_ROOT / "Feature_generation/features/protein/prot_esmc_fullseq.pt"
DEFAULT_TOKEN_ROOT = PROJECT_ROOT / "Feature_generation/features/protein/prot_esmc_residue_tokens"
DEFAULT_CHECKPOINT = PROJECT_ROOT / "Pretrained_models/ViroBind/virobind_classification.pt"


def predict_one(
    ckpt_path: Path,
    external_csv: Path,
    drug_feat: Path,
    prot_feat: Path,
    token_root: Path,
    out_csv: Path,
    device: torch.device,
) -> pd.DataFrame:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    args = SimpleNamespace(**ckpt.get("args", {}))
    v5.configure_from_namespace(args, token_root_override=str(token_root))

    ecfp = data_utils.FeatureBank(str(drug_feat), "external_drug_combo")
    prot = data_utils.FeatureBank(str(prot_feat), "external_prot_global")
    dataset = data_utils.ViroBindPairDataset(
        str(external_csv),
        ecfp,
        None,
        prot,
        getattr(args, "feature_preset", "ecfp_esmc"),
        "external_activity_labeled_virus_dti_set",
        True,
        "binary",
    )
    loader = DataLoader(dataset, batch_size=int(getattr(args, "batch_size", 512)), shuffle=False, num_workers=0)
    features = {"ecfp": ecfp.x, "prot": prot.x}

    model = v5.ViroBindV5Decoupled(
        getattr(args, "feature_preset", "ecfp_esmc"),
        ecfp.dim,
        1,
        prot.dim,
        int(getattr(args, "hidden_dim", 512)),
        int(getattr(args, "bilinear_dim", 128)),
        float(getattr(args, "dropout", 0.2)),
        bool(int(getattr(args, "use_domain_adapter", 0))),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    if model.use_residue_tokens:
        manifest = token_root / "manifest.csv"
        if not manifest.is_file():
            raise FileNotFoundError(f"Residue-token checkpoint requires {manifest}")
        v5.TOKEN_STORE = v5.ResidueTokenStore(str(token_root), prot)
    pred = v5.predict(model, dataset, loader, features, device, getattr(args, "feature_preset", "ecfp_esmc"))
    pred.insert(0, "checkpoint", str(ckpt_path))
    pred.insert(1, "seed", int(getattr(args, "seed", -1)))
    pred.to_csv(out_csv, index=False)
    return pred


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--external-csv", "--external_csv", dest="external_csv", required=True)
    parser.add_argument("--drug-feat", "--drug_feat", dest="drug_feat", default=str(DEFAULT_DRUG_FEATURE))
    parser.add_argument("--prot-feat", "--prot_feat", dest="prot_feat", default=str(DEFAULT_PROTEIN_FEATURE))
    parser.add_argument("--token-root", default=str(DEFAULT_TOKEN_ROOT))
    parser.add_argument("--ckpts", default=str(DEFAULT_CHECKPOINT), help="Comma-separated checkpoint paths.")
    parser.add_argument("--out-dir", "--out_dir", dest="out_dir", default=str(PROJECT_ROOT / "outputs/predictions"))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    preds = []
    for ckpt in [Path(x) for x in args.ckpts.split(",") if x.strip()]:
        out_csv = out_dir / f"external_predictions_{ckpt.stem}.csv"
        preds.append(
            predict_one(
                ckpt,
                Path(args.external_csv),
                Path(args.drug_feat),
                Path(args.prot_feat),
                Path(args.token_root),
                out_csv,
                device,
            )
        )
    if not preds:
        raise ValueError("No checkpoint was supplied")

    base = preds[0][["drug", "protein", "label"]].copy()
    cls = np.stack([p["cls_logit"].to_numpy(dtype=float) for p in preds], axis=1)
    rank = np.stack([p["rank_logit"].to_numpy(dtype=float) for p in preds], axis=1)
    base["cls_logit_mean"] = cls.mean(axis=1)
    base["cls_prob_mean"] = 1.0 / (1.0 + np.exp(-base["cls_logit_mean"]))
    base["rank_logit_mean"] = rank.mean(axis=1)
    base["rank_prob_mean"] = 1.0 / (1.0 + np.exp(-base["rank_logit_mean"]))
    base["cls_logit_std"] = cls.std(axis=1)
    base["rank_logit_std"] = rank.std(axis=1)
    ensemble_csv = out_dir / "external_predictions_ensemble.csv"
    base.to_csv(ensemble_csv, index=False)

    manifest = {
        "external_csv": args.external_csv,
        "drug_feat": args.drug_feat,
        "prot_feat": args.prot_feat,
        "ckpts": [str(x) for x in args.ckpts.split(",") if x.strip()],
        "prediction_files": [str(out_dir / f"external_predictions_{Path(x).stem}.csv") for x in args.ckpts.split(",") if x.strip()],
        "ensemble_csv": str(ensemble_csv),
    }
    (out_dir / "prediction_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
