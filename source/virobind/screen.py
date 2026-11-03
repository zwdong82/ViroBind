#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import heapq
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from . import model as v5

PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parents[1]
DEFAULT_CKPTS = [
    PROJECT_ROOT / "Pretrained_models/ViroBind/virobind_ranking.pt"
]


def load_models(
    ckpts: list[Path],
    device: torch.device,
    drug_dim: int = 2423,
    prot_dim: int = 1152,
    token_root: Path | None = None,
):
    models = []
    for path in ckpts:
        ckpt = torch.load(path, map_location=device, weights_only=False)
        args = SimpleNamespace(**ckpt["args"])
        override = str(token_root) if token_root is not None else None
        v5.configure_from_namespace(args, token_root_override=override)
        model = v5.ViroBindV5Decoupled(
            getattr(args, "feature_preset", "ecfp_esmc"), drug_dim, 1, prot_dim,
            int(getattr(args, "hidden_dim", 512)), int(getattr(args, "bilinear_dim", 128)),
            float(getattr(args, "dropout", 0.2)), bool(int(getattr(args, "use_domain_adapter", 0))),
        ).to(device)
        model.load_state_dict(ckpt["model"])
        model.eval()
        models.append((int(getattr(args, "seed", -1)), model))
    return models


def load_validation_tokens(token_root: Path, protein_row, device: torch.device, max_tokens: int):
    """Load one residue-aligned ESMC sequence with the training crop rule."""
    manifest = pd.read_csv(token_root / "manifest.csv")
    key = str(protein_row.protein_key)
    pid = str(protein_row.prot_id)
    hit = manifest[
        (manifest["protein_key"].astype(str) == key)
        | (manifest["prot_id"].astype(str) == pid)
    ]
    if len(hit) != 1:
        raise KeyError(f"Expected one residue-token record for {key} / prot_id={pid}, found {len(hit)}")
    token_file = token_root / str(hit.iloc[0].token_file)
    x = torch.load(token_file, map_location="cpu", weights_only=False)["embeddings"].float()
    if x.ndim != 2 or x.shape[1] != 1152:
        raise ValueError(f"Invalid residue-token tensor {token_file}: {tuple(x.shape)}")
    if x.shape[0] > max_tokens:
        start = (x.shape[0] - max_tokens) // 2
        x = x[start:start + max_tokens]
    tokens = x.unsqueeze(0).to(device=device, dtype=torch.float32, non_blocking=True)
    mask = torch.ones((1, x.shape[0]), dtype=torch.bool, device=device)
    return tokens, mask


@torch.inference_mode()
def predict_chunk(models, drug_x, protein_x, device, residue_tokens=None, residue_mask=None):
    x = drug_x.to(device=device, dtype=torch.float32, non_blocking=True)
    p = protein_x.expand(x.shape[0], -1)
    d = torch.ones(x.shape[0], dtype=torch.long, device=device)
    cls, rank = [], []
    for _, model in models:
        if model.use_residue_tokens:
            if residue_tokens is None or residue_mask is None:
                raise ValueError("Token checkpoint requires --token-root validation residue features")
            pair_index = torch.zeros(x.shape[0], dtype=torch.long, device=device)
            out = model(x, None, p, d, residue_tokens, residue_mask, pair_index)
        else:
            out = model(x, None, p, d)
        cls.append(out["cls"].float().cpu().numpy())
        rank.append(out["rank"].float().cpu().numpy())
    return np.stack(cls, axis=1), np.stack(rank, axis=1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--library-csv", required=True)
    ap.add_argument("--drug-feat", required=True)
    ap.add_argument("--protein-feat", required=True)
    ap.add_argument("--protein-csv", required=True)
    ap.add_argument("--token-root", default="", help="Validation residue-token directory for residue-ranking checkpoints.")
    ap.add_argument("--pdb-ids", default="", help="Optional comma-separated PDB ids to screen, e.g. 4HLA,7SI9.")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--ckpts", default=",".join(map(str, DEFAULT_CKPTS)))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--chunk-size", type=int, default=4096)
    ap.add_argument("--top-k", type=int, default=1000)
    ap.add_argument("--bottom-k", type=int, default=0)
    ap.add_argument("--uncertainty-beta", type=float, default=0.0)
    ap.add_argument("--cls-min-prob", type=float, default=0.0, help="Apply this cls probability gate to top hits only.")
    ap.add_argument("--eligibility-csv", default="", help="Optional metadata CSV used to restrict top-hit eligibility.")
    ap.add_argument("--eligibility-key-col", default="drug_key")
    ap.add_argument("--library-key-col", default="drug_key")
    ap.add_argument("--eligibility-query", default="", help="Pandas query applied to eligibility-csv before mapping keys to drug_id.")
    ap.add_argument("--write-all", action="store_true", help="Also write every pair's scores (large for ChEMBL).")
    args = ap.parse_args()

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    ckpts = [Path(x) for x in args.ckpts.split(",") if x]
    if not all(p.is_file() for p in ckpts):
        raise FileNotFoundError([str(p) for p in ckpts if not p.is_file()])
    # mmap avoids materializing a large drug embedding tensor twice in RAM.
    drug_bank = torch.load(args.drug_feat, map_location="cpu", weights_only=False, mmap=True)
    prot_bank = torch.load(args.protein_feat, map_location="cpu", weights_only=False, mmap=True)
    drug_x = drug_bank["embeddings"]
    drug_ids = np.asarray(drug_bank["drug_ids"], dtype=np.int64)
    protein_x = prot_bank["embeddings"]
    token_root_arg = Path(args.token_root) if args.token_root else None
    models = load_models(
        ckpts,
        device,
        drug_dim=int(drug_x.shape[1]),
        prot_dim=int(protein_x.shape[1]),
        token_root=token_root_arg,
    )
    checkpoint_args = SimpleNamespace(**torch.load(ckpts[0], map_location="cpu", weights_only=False)["args"])
    token_max_tokens = int(getattr(checkpoint_args, "token_max_tokens", 1024))
    token_models = any(model.use_residue_tokens for _, model in models)
    if token_models and not args.token_root:
        raise ValueError("The supplied checkpoint uses residue tokens; pass --token-root.")
    if token_models and not all(model.use_residue_tokens for _, model in models):
        raise ValueError("Cannot ensemble residue-token and non-token checkpoints.")
    token_root = Path(args.token_root) if token_models else None
    if token_root is not None and not (token_root / "manifest.csv").is_file():
        raise FileNotFoundError(token_root / "manifest.csv")
    proteins = pd.read_csv(args.protein_csv)
    if args.pdb_ids.strip():
        keep = {x.strip() for x in args.pdb_ids.split(",") if x.strip()}
        proteins = proteins[proteins["pdb_id"].astype(str).isin(keep)].reset_index(drop=True)
        if proteins.empty:
            raise ValueError(f"No proteins matched --pdb-ids={args.pdb_ids!r}")
    library_ids = pd.read_csv(args.library_csv, usecols=["drug_id"])["drug_id"].to_numpy(dtype=np.int64)
    if len(drug_x) != len(library_ids) or not np.array_equal(drug_ids, library_ids):
        raise ValueError("Drug feature rows do not exactly match library-csv drug_id order.")
    allowed_top_ids = None
    eligibility_count = None
    if args.eligibility_csv:
        eligibility = pd.read_csv(args.eligibility_csv)
        if args.eligibility_query:
            eligibility = eligibility.query(args.eligibility_query).copy()
        eligible_keys = set(eligibility[args.eligibility_key_col].astype(str))
        id_key = pd.read_csv(args.library_csv, usecols=["drug_id", args.library_key_col])
        allowed_top_ids = set(
            id_key.loc[id_key[args.library_key_col].astype(str).isin(eligible_keys), "drug_id"].astype(int)
        )
        eligibility_count = len(allowed_top_ids)
        if not allowed_top_ids:
            raise ValueError("Eligibility filtering removed every library molecule.")
    prot_keys = {str(k): i for i, k in enumerate(prot_bank["protein_keys"])}
    for i, prot_id in enumerate(prot_bank.get("prot_ids", [])):
        prot_keys.setdefault(str(prot_id), i)
    fields = [
        "drug_id", "protein_key", "pdb_id",
        "cls_logit_mean", "cls_prob_mean",
        "rank_logit_mean", "rank_prob_mean",
        "cls_logit_std", "rank_logit_std",
        "final_score",
    ]
    all_path = out_dir / "ensemble_scores.csv" if args.write_all else None
    with (all_path.open("w", newline="", encoding="utf-8") if all_path else open("/dev/null", "w", encoding="utf-8")) as handle:
        writer = csv.writer(handle)
        if all_path:
            writer.writerow(fields)
        top = {str(row.protein_key): [] for row in proteins.itertuples(index=False)}
        bottom = {str(row.protein_key): [] for row in proteins.itertuples(index=False)}
        n = len(drug_x)
        for p_row in proteins.itertuples(index=False):
            protein_key = str(p_row.protein_key)
            protein_lookup = protein_key if protein_key in prot_keys else str(p_row.prot_id)
            if protein_lookup not in prot_keys:
                raise KeyError(f"Protein feature missing: protein_key={protein_key}, prot_id={p_row.prot_id}")
            px = protein_x[prot_keys[protein_lookup]].to(device).view(1, -1)
            if token_models:
                residue_tokens, residue_mask = load_validation_tokens(
                    token_root, p_row, device, token_max_tokens
                )
            else:
                residue_tokens = residue_mask = None
            heap = top[protein_key]
            bottom_heap = bottom[protein_key]
            for start in range(0, n, args.chunk_size):
                end = min(start + args.chunk_size, n)
                cls, rank = predict_chunk(models, drug_x[start:end], px, device, residue_tokens, residue_mask)
                cm, rm = cls.mean(1), rank.mean(1)
                cp, rp = 1 / (1 + np.exp(-cm)), 1 / (1 + np.exp(-rm))
                for i in range(end - start):
                    rank_std = float(rank[i].std())
                    final_score = float(rm[i] - args.uncertainty_beta * rank_std)
                    row = [
                        int(drug_ids[start + i]), protein_key, p_row.pdb_id,
                        float(cm[i]), float(cp[i]),
                        float(rm[i]), float(rp[i]),
                        float(cls[i].std()), rank_std,
                        final_score,
                    ]
                    if all_path:
                        writer.writerow(row)
                    if (allowed_top_ids is None or int(drug_ids[start + i]) in allowed_top_ids) and float(cp[i]) >= args.cls_min_prob:
                        item = (final_score, int(drug_ids[start + i]), row)
                        if len(heap) < args.top_k:
                            heapq.heappush(heap, item)
                        elif item[:2] > heap[0][:2]:
                            heapq.heapreplace(heap, item)
                    if args.bottom_k:
                        # A min-heap of negated scores retains the lowest original scores.
                        bottom_item = (-final_score, -int(drug_ids[start + i]), row)
                        if len(bottom_heap) < args.bottom_k:
                            heapq.heappush(bottom_heap, bottom_item)
                        elif bottom_item[:2] > bottom_heap[0][:2]:
                            heapq.heapreplace(bottom_heap, bottom_item)
                if start % (args.chunk_size * 50) == 0:
                    print(f"{protein_key}: {end}/{n}", flush=True)
        selected_ids = {item[2][0] for heap in top.values() for item in heap}
        selected_ids.update(item[2][0] for heap in bottom.values() for item in heap)
        metadata = pd.concat(
            [chunk[chunk["drug_id"].isin(selected_ids)] for chunk in pd.read_csv(args.library_csv, chunksize=100_000)],
            ignore_index=True,
        )
        for protein_key, heap in top.items():
            score_df = pd.DataFrame([x[2] for x in sorted(heap, reverse=True)], columns=fields)
            score_df = score_df.merge(metadata, on="drug_id", how="left", validate="one_to_one")
            score_df.to_csv(out_dir / f"top_{args.top_k}_{protein_key.replace('::', '_')}.csv", index=False)
            if args.bottom_k:
                bottom_df = pd.DataFrame([x[2] for x in sorted(bottom[protein_key], reverse=True)], columns=fields)
                bottom_df = bottom_df.sort_values(["final_score", "drug_id"], ascending=[True, True]).merge(metadata, on="drug_id", how="left", validate="one_to_one")
                bottom_df.to_csv(out_dir / f"bottom_{args.bottom_k}_{protein_key.replace('::', '_')}.csv", index=False)
    manifest = {
        "device": str(device),
        "library_csv": args.library_csv,
        "drug_feat": args.drug_feat,
        "protein_feat": args.protein_feat,
        "protein_csv": args.protein_csv,
        "token_root": args.token_root or None,
        "pdb_ids": args.pdb_ids,
        "checkpoints": [str(x) for x in ckpts],
        "rows": int(len(drug_x) * len(proteins)),
        "all_scores": str(all_path) if all_path else None,
        "top_k": args.top_k,
        "bottom_k": args.bottom_k,
        "uncertainty_beta": args.uncertainty_beta,
        "cls_min_prob": args.cls_min_prob,
        "eligibility_csv": args.eligibility_csv,
        "eligibility_key_col": args.eligibility_key_col,
        "library_key_col": args.library_key_col,
        "eligibility_query": args.eligibility_query,
        "eligible_top_drug_count": eligibility_count,
        "score_formula": "rank_logit_mean - uncertainty_beta * rank_logit_std",
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
