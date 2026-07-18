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
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    meta = json.loads(Path(args.meta).read_text(encoding="utf-8"))
    desc_map = dict(Descriptors.descList)
    fns = [(name, desc_map.get(name)) for name in meta["descriptor_names"]]
    stats = meta["standardization"]
    med, mean, std = (np.asarray(stats[k], dtype=np.float32) for k in ("median", "mean", "std"))
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
        raw = np.asarray(vals, dtype=np.float32)
        raw = np.where(np.isfinite(raw), raw, med)
        rows.append(np.clip((raw - mean) / std, -10.0, 10.0).astype(np.float32))
    x = torch.from_numpy(np.stack(rows))
    obj = {"drug_ids": ids, "drug_keys": keys, "embeddings": x,
           "feature_name": "rdkit2d_zscore", "dim": len(fns),
           "descriptor_names": meta["descriptor_names"], "invalid_smiles": invalid}
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
                    "meta": {"components": ["ecfp4_r2_2048", "maccs_167", "rdkit2d_zscore"]}},
                   args.combo_output)
        print(f"[SAVED] {args.combo_output} shape={tuple(combo.shape)}")


if __name__ == "__main__":
    main()
