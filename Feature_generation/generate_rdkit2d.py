#!/usr/bin/env python3
"""Generate the final 208-dimensional standardized RDKit2D drug view.

Input CSV columns: drug_id, drug_key, SMILES_ChargeAware (configurable), plus
the bundled training-time descriptor metadata. Output is [n_drugs, 208].
Optionally pass ECFP4 and MACCS files to also create the final 2423-D drug bank.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import Descriptors
from tqdm import tqdm


def fit_standardization(raw, train_mask):
    """Fit imputation and z-score statistics on training compounds only."""
    train = np.asarray(raw, dtype=np.float32)[train_mask].copy()
    if len(train) == 0:
        raise ValueError("No training compounds available to fit RDKit2D statistics")
    train[~np.isfinite(train)] = np.nan
    median = np.nanmedian(train, axis=0)
    median = np.where(np.isfinite(median), median, 0.0)
    train = np.where(np.isfinite(train), train, median)
    mean, std = train.mean(axis=0), train.std(axis=0)
    std = np.where(np.isfinite(std) & (std > 0), std, 1.0)
    return {k: v.tolist() for k, v in zip(("median", "mean", "std"), (median, mean, std))}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True, help="Drug CSV containing IDs and SMILES.")
    ap.add_argument("--output", required=True, help="RDKit2D output .pt file.")
    ap.add_argument("--meta", default=str(Path(__file__).with_name("rdkit2d_descriptor_meta.json")))
    ap.add_argument("--drug-id-col", default="drug_id")
    ap.add_argument("--drug-key-col", default="drug_key")
    ap.add_argument("--smiles-col", default="SMILES_ChargeAware")
    ap.add_argument("--ecfp4", default="", help="Optional ECFP4 .pt file for final concatenation.")
    ap.add_argument("--maccs", default="", help="Optional MACCS .pt file for final concatenation.")
    ap.add_argument("--combo-output", default="", help="Output for ECFP4+MACCS+RDKit2D.")
    ap.add_argument("--fit-training-csv", action="append", default=[],
                    help="Training pair CSV with drug_id; repeat for human and viral training. Never pass held-out data.")
    ap.add_argument("--fitted-meta-output", default="", help="Save split-specific fitted normalization metadata.")
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    meta = json.loads(Path(args.meta).read_text(encoding="utf-8"))
    desc_map = dict(Descriptors.descList)
    fns = [(name, desc_map.get(name)) for name in meta["descriptor_names"]]
    unavailable = [name for name, fn in fns if fn is None]
    if unavailable:
        raise ValueError(f"RDKit descriptor implementation unavailable: {unavailable}")
    ids = pd.to_numeric(df[args.drug_id_col], errors="raise").astype(int).tolist()
    keys = df[args.drug_key_col].astype(str).tolist() if args.drug_key_col in df else list(map(str, ids))
    rows, invalid = [], 0
    for value in tqdm(df[args.smiles_col], desc="RDKit2D"):
        mol = Chem.MolFromSmiles(str(value)) if pd.notna(value) else None
        if mol is None:
            invalid += 1
        vals = []
        for _, fn in fns:
            try:
                vals.append(float(fn(mol)) if mol is not None and fn is not None else np.nan)
            except Exception:
                vals.append(np.nan)
        rows.append(np.asarray(vals, dtype=np.float32))
    raw = np.stack(rows)
    if args.fit_training_csv:
        if not args.fitted_meta_output:
            raise ValueError("--fit-training-csv requires --fitted-meta-output")
        training_ids = set()
        for path in args.fit_training_csv:
            training_ids.update(pd.read_csv(path, usecols=["drug_id"])["drug_id"].astype(int))
        missing = training_ids - set(ids)
        if missing:
            raise ValueError(f"{len(missing)} training compounds are missing from the structure table")
        meta["standardization"] = fit_standardization(raw, np.isin(ids, list(training_ids)))
        meta["fit_scope"] = {"training_only": True, "unique_training_compounds": len(training_ids),
                             "training_files": [Path(p).name for p in args.fit_training_csv]}
        Path(args.fitted_meta_output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.fitted_meta_output).write_text(json.dumps(meta, indent=2), encoding="utf-8")
    stats = meta["standardization"]
    med, mean, std = (np.asarray(stats[k], dtype=np.float32) for k in ("median", "mean", "std"))
    if not np.isfinite(np.concatenate([med, mean, std])).all() or np.any(std <= 0):
        raise ValueError("Invalid fitted normalization statistics")
    raw = np.where(np.isfinite(raw), raw, med)
    x = torch.from_numpy(np.clip((raw - mean) / std, -10.0, 10.0).astype(np.float32))
    obj = {"drug_ids": ids, "drug_keys": keys, "embeddings": x,
           "feature_name": "rdkit2d_zscore", "dim": len(fns),
           "descriptor_names": meta["descriptor_names"], "invalid_smiles": invalid,
           "standardization": meta}
    torch.save(obj, args.output)
    print(f"[SAVED] {args.output} shape={tuple(x.shape)} invalid_smiles={invalid}")

    if any((args.ecfp4, args.maccs, args.combo_output)):
        if not all((args.ecfp4, args.maccs, args.combo_output)):
            raise ValueError("--ecfp4, --maccs and --combo-output must be provided together")
        e = torch.load(args.ecfp4, map_location="cpu", weights_only=False)
        m = torch.load(args.maccs, map_location="cpu", weights_only=False)
        if e["drug_ids"] != ids or m["drug_ids"] != ids:
            raise ValueError("Drug ID order differs across feature files")
        combo = torch.cat([e["embeddings"].float(), m["embeddings"].float(), x.float()], dim=1)
        torch.save({"drug_ids": ids, "drug_keys": keys, "embeddings": combo,
                    "feature_name": "combo_ecfp4_maccs_rdkit2d", "dim": 2423,
                    "meta": {"components": ["ecfp4_r2_2048", "maccs_167", "rdkit2d_zscore"],
                             "rdkit2d_standardization": meta}},
                   args.combo_output)
        print(f"[SAVED] {args.combo_output} shape={tuple(combo.shape)}")


if __name__ == "__main__":
    main()
