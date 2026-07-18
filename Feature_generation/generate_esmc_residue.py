#!/usr/bin/env python3
"""Extract frozen, residue-aligned ESMC embeddings for token-level ViroBind.

Each output ``tokens/<prot_id>.pt`` contains an [L, 1152] tensor whose row i maps
to residue i (0-based) in the input sequence. Long sequences are stitched from
overlapping windows at their original residue coordinates; they are never pooled.
"""
import argparse
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm
from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, LogitsConfig


def windows(sequence, width, stride):
    if len(sequence) <= width:
        return [(0, sequence)]
    starts = list(range(0, len(sequence) - width + 1, stride))
    final = len(sequence) - width
    if starts[-1] != final:
        starts.append(final)
    return [(s, sequence[s:s + width]) for s in starts]


def residue_tokens(emb, n_residues):
    """Remove only confirmed BOS/EOS tokens; fail rather than shift residue indices."""
    if emb.ndim != 2:
        raise ValueError(f"Expected [tokens, dim], got {tuple(emb.shape)}")
    if emb.shape[0] == n_residues:
        return emb
    if emb.shape[0] == n_residues + 2:
        return emb[1:-1]
    raise ValueError(f"ESMC output has {emb.shape[0]} tokens for {n_residues} residues; inspect tokenizer before use.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True); ap.add_argument('--outdir', required=True)
    ap.add_argument('--id-col', default='prot_id'); ap.add_argument('--seq-col', default='Sequence')
    ap.add_argument('--protein-key-col', default='protein_key'); ap.add_argument('--model-name', default='esmc_600m')
    ap.add_argument('--max-len', type=int, default=1024); ap.add_argument('--stride', type=int, default=512)
    ap.add_argument('--dtype', choices=['bf16', 'fp16', 'fp32'], default='bf16'); ap.add_argument('--overwrite', action='store_true')
    args = ap.parse_args(); outdir = Path(args.outdir); token_dir = outdir / 'tokens'; token_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.input)
    for c in [args.id_col, args.seq_col]:
        if c not in df: raise ValueError(f'Missing {c}; columns={list(df.columns)}')
    df = df.dropna(subset=[args.id_col, args.seq_col]).drop_duplicates(args.id_col).copy()
    df[args.id_col] = pd.to_numeric(df[args.id_col], errors='raise').astype(int); df[args.seq_col] = df[args.seq_col].astype(str).str.strip()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'); print(f'[INFO] device={device}; proteins={len(df)}')
    model = ESMC.from_pretrained(args.model_name).to(device).eval(); cfg = LogitsConfig(return_embeddings=True)
    amp = {'bf16': torch.bfloat16, 'fp16': torch.float16}.get(args.dtype)
    manifest = []
    for row in tqdm(df.to_dict('records'), desc='ESMC token extraction'):
        pid, seq = int(row[args.id_col]), row[args.seq_col]; path = token_dir / f'{pid}.pt'
        if path.exists() and not args.overwrite:
            x = torch.load(path, map_location='cpu', weights_only=False)['embeddings']; manifest.append({'prot_id':pid, 'protein_key':str(row.get(args.protein_key_col, pid)), 'length':len(seq), 'token_file':str(path.relative_to(outdir)), 'embedding_dim':x.shape[1]}); continue
        accum, count = None, None
        for start, piece in windows(seq, args.max_len, args.stride):
            with torch.no_grad(), torch.autocast('cuda', dtype=amp, enabled=(device.type == 'cuda' and amp is not None)):
                o = model.logits(model.encode(ESMProtein(sequence=piece)), cfg)
            h = residue_tokens(o.embeddings.squeeze(0), len(piece)).float().cpu()
            if accum is None: accum, count = torch.zeros((len(seq), h.shape[1])), torch.zeros((len(seq), 1))
            end = start + len(piece); accum[start:end] += h; count[start:end] += 1
        if (count == 0).any(): raise RuntimeError(f'Uncovered residue(s) in protein {pid}')
        x = (accum / count).half(); torch.save({'prot_id':pid, 'sequence':seq, 'embeddings':x, 'model':args.model_name, 'residue_index_base':0}, path)
        manifest.append({'prot_id':pid, 'protein_key':str(row.get(args.protein_key_col, pid)), 'length':len(seq), 'token_file':str(path.relative_to(outdir)), 'embedding_dim':x.shape[1]})
    pd.DataFrame(manifest).sort_values('prot_id').to_csv(outdir / 'manifest.csv', index=False)
    print(f'[SAVED] {outdir / "manifest.csv"}\n[SAVED] {token_dir}')

if __name__ == '__main__': main()
