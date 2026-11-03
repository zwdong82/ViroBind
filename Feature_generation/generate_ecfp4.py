#!/usr/bin/env python3
"""Generate the final 2048-dimensional ECFP4 drug view.

Input CSV columns: drug_id, drug_key, SMILES_ChargeAware (configurable).
Output: a PyTorch dictionary containing drug_ids, drug_keys and embeddings
with shape [n_drugs, 2048].
"""
import argparse

import numpy as np
import pandas as pd
import torch
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from tqdm import tqdm


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True, help="Drug CSV containing IDs and SMILES.")
    ap.add_argument("--output", required=True, help="Output .pt file.")
    ap.add_argument("--drug-id-col", default="drug_id")
    ap.add_argument("--drug-key-col", default="drug_key")
    ap.add_argument("--smiles-col", default="SMILES_ChargeAware")
    ap.add_argument("--radius", type=int, default=2)
    ap.add_argument("--n-bits", type=int, default=2048)
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    for col in (args.drug_id_col, args.smiles_col):
        if col not in df:
            raise ValueError(f"Missing {col}; columns={list(df.columns)}")
    ids = pd.to_numeric(df[args.drug_id_col], errors="raise").astype(int).tolist()
    keys = df[args.drug_key_col].astype(str).tolist() if args.drug_key_col in df else list(map(str, ids))
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=args.radius, fpSize=args.n_bits)
    rows, invalid = [], 0
    for value in tqdm(df[args.smiles_col], desc="ECFP4"):
        mol = Chem.MolFromSmiles(str(value)) if pd.notna(value) else None
        arr = np.zeros(args.n_bits, dtype=np.float32)
        if mol is None:
            invalid += 1
        else:
            DataStructs.ConvertToNumpyArray(gen.GetFingerprint(mol), arr)
        rows.append(arr)
    x = torch.from_numpy(np.stack(rows))
    torch.save({"drug_ids": ids, "drug_keys": keys, "embeddings": x,
                "feature_name": "ecfp4_r2_2048", "dim": args.n_bits,
                "invalid_smiles": invalid}, args.output)
    print(f"[SAVED] {args.output} shape={tuple(x.shape)} invalid_smiles={invalid}")


if __name__ == "__main__":
    main()
