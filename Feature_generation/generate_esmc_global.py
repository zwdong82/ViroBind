#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ESMC feature extraction from protein2id.csv
Expected columns:
prot_id, prot_domain, protein_key, UniProt_ID, Protein_ID, Sequence
"""

import os
import argparse
import pandas as pd
import torch
from tqdm import tqdm

from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, LogitsConfig


def generate_windows(seq: str, max_len: int = 1024, stride: int = 512):
    L = len(seq)
    if L <= max_len:
        return [seq]

    windows = []
    start = 0
    while start < L:
        end = start + max_len
        windows.append(seq[start:end])
        if end >= L:
            break
        start += stride
    return windows


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="protein2id.csv")
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--id_col", default="prot_id", choices=["prot_id"])
    parser.add_argument("--seq_col", default="Sequence")
    parser.add_argument("--protein_key_col", default="protein_key")
    parser.add_argument("--uniprot_col", default="UniProt_ID")

    parser.add_argument("--model_name", default="esmc_600m")
    parser.add_argument("--max_len", type=int, default=1024)
    parser.add_argument("--stride", type=int, default=512)
    parser.add_argument("--pool", default="mean", choices=["mean", "cls"])
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    df = pd.read_csv(args.input)
    need_cols = [args.id_col, args.seq_col]
    for c in need_cols:
        if c not in df.columns:
            raise ValueError(f"Column '{c}' not found. Available columns: {list(df.columns)}")

    df = df.copy()
    df[args.id_col] = pd.to_numeric(df[args.id_col], errors="coerce")
    df[args.seq_col] = df[args.seq_col].astype(str).str.strip()
    df = df.dropna(subset=[args.id_col, args.seq_col]).copy()
    df = df[df[args.seq_col].str.len() > 0].copy()
    df = df[df[args.seq_col].str.lower() != "nan"].copy()
    df[args.id_col] = df[args.id_col].astype(int)
    df = df.sort_values(args.id_col).drop_duplicates(subset=[args.id_col], keep="first")

    protein_ids = df[args.id_col].tolist()
    sequences = df[args.seq_col].tolist()
    protein_keys = df[args.protein_key_col].astype(str).tolist() if args.protein_key_col in df.columns else [str(x) for x in protein_ids]
    uniprot_ids = df[args.uniprot_col].astype(str).tolist() if args.uniprot_col in df.columns else protein_keys

    print("Total proteins:", len(protein_ids))

    print("Loading model:", args.model_name)
    model = ESMC.from_pretrained(args.model_name).to(device)
    model.eval()

    if args.dtype == "bf16":
        amp_dtype = torch.bfloat16
        use_amp = True
    elif args.dtype == "fp16":
        amp_dtype = torch.float16
        use_amp = True
    else:
        amp_dtype = None
        use_amp = False

    def autocast_ctx():
        if use_amp and device.type == "cuda":
            return torch.amp.autocast("cuda", dtype=amp_dtype)
        return torch.no_grad()

    cfg = LogitsConfig(return_embeddings=True)

    embeddings = []
    long_protein_count = 0
    total_windows = 0

    print("Start encoding with sliding window...")
    for _, seq in tqdm(list(zip(protein_ids, sequences)), total=len(sequences), desc="ESMC Encoding"):
        windows = generate_windows(seq, max_len=args.max_len, stride=args.stride)
        if len(windows) > 1:
            long_protein_count += 1

        window_vecs = []
        for wseq in windows:
            protein = ESMProtein(sequence=wseq)
            with autocast_ctx():
                encoded = model.encode(protein)
                out = model.logits(encoded, cfg)

            emb = out.embeddings.squeeze(0)
            vec = emb.mean(dim=0) if args.pool == "mean" else emb[0]
            window_vecs.append(vec)

        prot_vec = torch.stack(window_vecs, dim=0).mean(dim=0)
        embeddings.append(prot_vec.detach().cpu())
        total_windows += len(windows)

    emb_all = torch.stack(embeddings, dim=0) if len(embeddings) > 0 else torch.zeros((0, 1152), dtype=torch.float32)

    print("Encoding finished")
    print("Embedding shape:", tuple(emb_all.shape))
    print("Long proteins:", long_protein_count)
    print("Total windows processed:", total_windows)

    save_path = os.path.join(args.outdir, "prot_esmc_fullseq.pt")
    torch.save(
        {
            "prot_ids": protein_ids,
            "protein_keys": protein_keys,
            "uniprot_ids": uniprot_ids,
            "embeddings": emb_all,
            "model": args.model_name,
            "pool": args.pool,
            "window": args.max_len,
            "stride": args.stride,
            "source_csv": args.input,
        },
        save_path,
    )
    print("Saved to:", save_path)


if __name__ == "__main__":
    main()
