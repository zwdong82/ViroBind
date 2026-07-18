#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import json
import math
import time
import random
import argparse
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader, Sampler
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
)


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


# -------------------------
# Basic utils
# -------------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def is_missing(x):
    try:
        return pd.isna(x)
    except Exception:
        return x is None


def norm_key(x) -> List[str]:
    if x is None or is_missing(x):
        return []
    s = str(x).strip()
    if s == "" or s.lower() == "nan":
        return []

    keys = [s]

    try:
        f = float(s)
        if math.isfinite(f) and abs(f - int(f)) < 1e-8:
            keys.append(str(int(f)))
    except Exception:
        pass

    out = []
    seen = set()
    for k in keys:
        if k not in seen:
            out.append(k)
            seen.add(k)
    return out


def add_key(d: Dict[str, int], key, idx: int):
    for k in norm_key(key):
        if k not in d:
            d[k] = idx


def json_dump(obj, path: str):
    def conv(x):
        if isinstance(x, np.integer):
            return int(x)
        if isinstance(x, np.floating):
            return float(x)
        if isinstance(x, np.ndarray):
            return x.tolist()
        if torch.is_tensor(x):
            return x.detach().cpu().tolist()
        return str(x)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=conv)


def safe_auc(y, p):
    y = np.asarray(y)
    p = np.asarray(p)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, p))


def safe_aupr(y, p):
    y = np.asarray(y)
    p = np.asarray(p)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(average_precision_score(y, p))


def best_f1_threshold(y, prob):
    y = np.asarray(y).astype(int)
    prob = np.asarray(prob).astype(float)

    best_t = 0.5
    best_f1 = -1.0

    for t in np.linspace(0.0, 1.0, 501):
        pred = (prob >= t).astype(int)
        f1 = f1_score(y, pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_t = float(t)

    return best_t, best_f1


# -------------------------
# FeatureBank
# -------------------------

class FeatureBank:
    def __init__(self, path: str, name: str):
        self.path = path
        self.name = name
        try:
            obj = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
        except (TypeError, RuntimeError):
            obj = torch.load(path, map_location="cpu", weights_only=False)
        self.x, self.key_to_idx, self.meta = self._parse(obj)
        self.x = self.x.float().contiguous()
        self.dim = int(self.x.shape[1])

        print(f"[FEATURE] {name}: {path}")
        print(f"[FEATURE] {name}: shape={tuple(self.x.shape)} dim={self.dim} n_keys={len(self.key_to_idx)}")
        print(f"[FEATURE] {name}: meta_keys={self.meta.get('meta_keys', [])}")

    def _parse(self, obj: Any):
        if not isinstance(obj, dict):
            raise ValueError(f"{self.name}: unsupported pt type: {type(obj)}")

        tensor_candidates = []
        for k, v in obj.items():
            if torch.is_tensor(v) and v.ndim == 2:
                tensor_candidates.append((k, v))
            elif isinstance(v, np.ndarray) and v.ndim == 2:
                tensor_candidates.append((k, torch.as_tensor(v)))

        preferred = [
            "embeddings", "embedding", "features", "feature", "feat",
            "x", "X", "fp", "fps", "ecfp", "unimol"
        ]

        if tensor_candidates:
            tensor_candidates = sorted(
                tensor_candidates,
                key=lambda kv: preferred.index(kv[0]) if kv[0] in preferred else 999
            )
            tensor_key, x = tensor_candidates[0]
            n = x.shape[0]

            key_to_idx = {}
            meta_keys = []

            for k, v in obj.items():
                if k == tensor_key:
                    continue
                if isinstance(v, (list, tuple, np.ndarray)) and len(v) == n:
                    ok = True
                    sample = list(v[: min(5, n)])
                    for a in sample:
                        if isinstance(a, (list, tuple, dict)):
                            ok = False
                    if ok:
                        meta_keys.append(k)
                        for i, val in enumerate(v):
                            add_key(key_to_idx, val, i)

            if not key_to_idx:
                for i in range(n):
                    add_key(key_to_idx, i, i)
                meta_keys.append("auto_range_index")

            return x, key_to_idx, {"tensor_key": tensor_key, "meta_keys": meta_keys}

        keys = []
        vals = []

        for k, v in obj.items():
            try:
                if torch.is_tensor(v):
                    t = v.detach().cpu()
                elif isinstance(v, np.ndarray):
                    t = torch.as_tensor(v)
                elif isinstance(v, (list, tuple)):
                    t = torch.as_tensor(v)
                else:
                    continue

                if t.ndim == 1 and t.numel() > 0:
                    keys.append(k)
                    vals.append(t.float())
            except Exception:
                continue

        if vals:
            x = torch.stack(vals, dim=0)
            key_to_idx = {}
            for i, k in enumerate(keys):
                add_key(key_to_idx, k, i)
            return x, key_to_idx, {"tensor_key": "dict_values", "meta_keys": ["dict_keys"]}

        raise ValueError(f"{self.name}: cannot parse feature file: {self.path}")

    def lookup(self, candidates: List[Any]) -> Optional[int]:
        for c in candidates:
            for k in norm_key(c):
                if k in self.key_to_idx:
                    return self.key_to_idx[k]
        return None


# -------------------------
# Dataset
# -------------------------

DRUG_COLS = [
    "drug_id", "drug_key", "Drug_ID", "drug", "compound_id",
    "SMILES_ChargeAware", "SMILES_Std", "SMILES", "smiles"
]

PROT_COLS = [
    "prot_id", "protein_key", "Protein_ID", "UniProt_ID",
    "uniprot_id", "protein_id", "target_id", "prot"
]


def row_candidates(row: Dict[str, Any], cols: List[str]) -> List[Any]:
    out = []
    for c in cols:
        if c in row and not is_missing(row[c]):
            out.append(row[c])
    return out


def infer_domain_id(row: Dict[str, Any]) -> int:
    vals = []
    for c in ["prot_domain", "domain", "protein_domain", "protein_key"]:
        if c in row and not is_missing(row[c]):
            vals.append(str(row[c]).lower())

    text = " ".join(vals)

    if "virus" in text or "viral" in text:
        return 1
    if "human" in text or "homo" in text:
        return 0

    return 0


def infer_protein_name(row: Dict[str, Any]) -> str:
    for c in ["protein_key", "prot_id", "UniProt_ID", "Protein_ID", "uniprot_id", "protein_id"]:
        if c in row and not is_missing(row[c]):
            return str(row[c])
    return "unknown_protein"


def infer_drug_name(row: Dict[str, Any]) -> str:
    for c in ["drug_key", "drug_id", "Drug_ID", "SMILES_ChargeAware", "SMILES_Std", "SMILES"]:
        if c in row and not is_missing(row[c]):
            return str(row[c])
    return "unknown_drug"


class ViroBindPairDataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        ecfp: Optional[FeatureBank],
        unimol: Optional[FeatureBank],
        prot: FeatureBank,
        feature_preset: str,
        name: str,
        labeled_only: bool = True,
        label_mode: str = "binary",
    ):
        self.csv_path = csv_path
        self.name = name
        self.feature_preset = feature_preset

        self.use_ecfp = feature_preset in ["ecfp_esmc", "ecfp_unimol_esmc"]
        self.use_unimol = feature_preset in ["unimol_esmc", "ecfp_unimol_esmc"]

        df = pd.read_csv(csv_path)
        raw_n = len(df)

        if "label" not in df.columns:
            raise ValueError(f"{csv_path} has no label column")

        df["label"] = pd.to_numeric(df["label"], errors="coerce")

        if labeled_only or label_mode == "binary":
            df = df[df["label"].isin([0, 1])].copy()
        elif label_mode == "gray":
            df = df[df["label"].isin([-1, 0, 1])].copy()
        else:
            raise ValueError(f"Unsupported label_mode={label_mode}")

        labeled_n = len(df)

        ecfp_idx = []
        unimol_idx = []
        prot_idx = []
        y = []
        domain = []
        group_id = []
        group_names = []
        drug_names = []

        protein_to_gid = {}

        missing = {
            "ecfp": 0,
            "unimol": 0,
            "prot": 0,
            "any": 0,
        }

        for row in df.to_dict("records"):
            d_cands = row_candidates(row, DRUG_COLS)
            p_cands = row_candidates(row, PROT_COLS)

            ie = -1
            iu = -1

            if self.use_ecfp:
                ie = ecfp.lookup(d_cands)
            if self.use_unimol:
                iu = unimol.lookup(d_cands)

            ip = prot.lookup(p_cands)

            bad = False

            if self.use_ecfp and ie is None:
                missing["ecfp"] += 1
                bad = True
            if self.use_unimol and iu is None:
                missing["unimol"] += 1
                bad = True
            if ip is None:
                missing["prot"] += 1
                bad = True

            if bad:
                missing["any"] += 1
                continue

            protein_name = infer_protein_name(row)
            if protein_name not in protein_to_gid:
                protein_to_gid[protein_name] = len(protein_to_gid)

            ecfp_idx.append(int(ie) if ie is not None else -1)
            unimol_idx.append(int(iu) if iu is not None else -1)
            prot_idx.append(int(ip))
            y.append(float(row["label"]))
            domain.append(infer_domain_id(row))
            gid = protein_to_gid[protein_name]
            group_id.append(gid)
            group_names.append(protein_name)
            drug_names.append(infer_drug_name(row))

        self.ecfp_idx = torch.tensor(ecfp_idx, dtype=torch.long)
        self.unimol_idx = torch.tensor(unimol_idx, dtype=torch.long)
        self.prot_idx = torch.tensor(prot_idx, dtype=torch.long)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.domain = torch.tensor(domain, dtype=torch.long)
        self.group = torch.tensor(group_id, dtype=torch.long)

        self.group_names_per_item = group_names
        self.drug_names_per_item = drug_names

        n = len(self.y)
        pos = int((self.y == 1).sum().item()) if n else 0
        neg = int((self.y == 0).sum().item()) if n else 0
        gray = int((self.y == -1).sum().item()) if n else 0

        print("=" * 90)
        print(f"[DATASET] {name}")
        print(f"[DATASET] path={csv_path}")
        print(f"[DATASET] raw={raw_n} selected_by_label_mode={labeled_n} final_with_features={n}")
        print(f"[DATASET] label_mode={label_mode} pos={pos} neg={neg} gray={gray} pos_rate_01={pos / max(pos + neg, 1):.6f}")
        print(f"[DATASET] n_proteins={len(protein_to_gid)}")
        print(f"[DATASET] missing={missing}")
        print("=" * 90)

        if n <= 0:
            raise RuntimeError(f"{name}: no usable samples after feature matching")

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return {
            "ecfp_idx": self.ecfp_idx[idx],
            "unimol_idx": self.unimol_idx[idx],
            "prot_idx": self.prot_idx[idx],
            "y": self.y[idx],
            "domain": self.domain[idx],
            "group": self.group[idx],
            "idx": torch.tensor(idx, dtype=torch.long),
        }

    def pos_weight(self):
        pos = float((self.y == 1).sum().item())
        neg = float((self.y == 0).sum().item())
        if pos <= 0:
            return 1.0
        return neg / pos


# -------------------------
# Protein-balanced sampler
# -------------------------

class ProteinBalancedBatchSampler(Sampler[List[int]]):
    """
    每个 batch:
        采样 proteins_per_batch 个 protein
        每个 protein 采样 pos_per_protein 个真实正样本 label=1
        每个 protein 采样 neg_per_protein 个真实负样本 label=0

    注意：
        这里的 negative 是数据里真实 label=0 的样本，不是伪负样本。
    """

    def __init__(
        self,
        dataset: ViroBindPairDataset,
        proteins_per_batch: int = 16,
        pos_per_protein: int = 4,
        neg_per_protein: int = 4,
        gray_per_protein: int = 0,
        batches_per_epoch: Optional[int] = None,
        seed: int = 42,
    ):
        self.dataset = dataset
        self.proteins_per_batch = proteins_per_batch
        self.pos_per_protein = pos_per_protein
        self.neg_per_protein = neg_per_protein
        self.gray_per_protein = gray_per_protein
        self.seed = seed
        self.epoch = 0

        y = dataset.y.cpu().numpy()
        g = dataset.group.cpu().numpy()

        self.group_to_pos = {}
        self.group_to_neg = {}
        self.group_to_gray = {}

        for i, (yy, gg) in enumerate(zip(y, g)):
            gg = int(gg)
            if yy == 1:
                self.group_to_pos.setdefault(gg, []).append(i)
            elif yy == 0:
                self.group_to_neg.setdefault(gg, []).append(i)
            elif yy == -1:
                self.group_to_gray.setdefault(gg, []).append(i)

        self.valid_groups = sorted(
            set(self.group_to_pos.keys()) & set(self.group_to_neg.keys())
        )

        if len(self.valid_groups) == 0:
            raise RuntimeError("ProteinBalancedBatchSampler: no protein has both positive and negative samples.")

        batch_size = proteins_per_batch * (pos_per_protein + neg_per_protein + gray_per_protein)

        if batches_per_epoch is None:
            batches_per_epoch = max(1, math.ceil(len(dataset) / batch_size))

        self.batches_per_epoch = int(batches_per_epoch)

        print("[SAMPLER] protein_balanced")
        print(f"[SAMPLER] valid_proteins={len(self.valid_groups)}")
        print(f"[SAMPLER] proteins_per_batch={proteins_per_batch}")
        print(f"[SAMPLER] pos_per_protein={pos_per_protein}")
        print(f"[SAMPLER] neg_per_protein={neg_per_protein}")
        print(f"[SAMPLER] gray_per_protein={gray_per_protein}")
        print(f"[SAMPLER] batch_size={batch_size}")
        print(f"[SAMPLER] batches_per_epoch={self.batches_per_epoch}")

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        self.epoch += 1

        valid_groups = np.array(self.valid_groups)

        for _ in range(self.batches_per_epoch):
            replace_group = len(valid_groups) < self.proteins_per_batch
            groups = rng.choice(
                valid_groups,
                size=self.proteins_per_batch,
                replace=replace_group,
            )

            batch = []

            for gg in groups:
                gg = int(gg)
                pos_pool = self.group_to_pos[gg]
                neg_pool = self.group_to_neg[gg]

                pos_replace = len(pos_pool) < self.pos_per_protein
                neg_replace = len(neg_pool) < self.neg_per_protein

                pos_idx = rng.choice(
                    pos_pool,
                    size=self.pos_per_protein,
                    replace=pos_replace,
                ).tolist()

                neg_idx = rng.choice(
                    neg_pool,
                    size=self.neg_per_protein,
                    replace=neg_replace,
                ).tolist()

                batch.extend(pos_idx)
                batch.extend(neg_idx)

                if self.gray_per_protein > 0:
                    gray_pool = self.group_to_gray.get(gg, [])
                    if len(gray_pool) > 0:
                        gray_replace = len(gray_pool) < self.gray_per_protein
                        gray_idx = rng.choice(
                            gray_pool,
                            size=self.gray_per_protein,
                            replace=gray_replace,
                        ).tolist()
                        batch.extend(gray_idx)

            rng.shuffle(batch)
            yield batch

    def __len__(self):
        return self.batches_per_epoch


# -------------------------
# Model
# -------------------------

class MLPEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

    def forward(self, x):
        return self.net(x)


class DomainAdapter(nn.Module):
    def __init__(self, dim: int, bottleneck: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, bottleneck),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(bottleneck, dim),
        )

    def forward(self, x):
        return x + self.net(x)


class ViroBindV2(nn.Module):
    def __init__(
        self,
        feature_preset: str,
        ecfp_dim: int,
        unimol_dim: int,
        prot_dim: int,
        hidden_dim: int = 512,
        bilinear_dim: int = 128,
        dropout: float = 0.2,
        use_domain_adapter: bool = False,
    ):
        super().__init__()

        self.feature_preset = feature_preset
        self.use_ecfp = feature_preset in ["ecfp_esmc", "ecfp_unimol_esmc"]
        self.use_unimol = feature_preset in ["unimol_esmc", "ecfp_unimol_esmc"]
        self.use_domain_adapter = use_domain_adapter

        if self.use_ecfp:
            self.ecfp_encoder = MLPEncoder(ecfp_dim, hidden_dim, dropout)

        if self.use_unimol:
            self.unimol_encoder = MLPEncoder(unimol_dim, hidden_dim, dropout)

        if self.use_ecfp and self.use_unimol:
            self.drug_gate = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.Sigmoid(),
            )

        self.prot_encoder = MLPEncoder(prot_dim, hidden_dim, dropout)

        if use_domain_adapter:
            bottleneck = max(32, hidden_dim // 4)
            self.human_adapter = DomainAdapter(hidden_dim, bottleneck, dropout)
            self.virus_adapter = DomainAdapter(hidden_dim, bottleneck, dropout)

        self.drug_bilinear = nn.Linear(hidden_dim, bilinear_dim, bias=False)
        self.prot_bilinear = nn.Linear(hidden_dim, bilinear_dim, bias=False)

        pair_dim = hidden_dim * 4 + bilinear_dim

        self.head = nn.Sequential(
            nn.Linear(pair_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        x_ecfp: Optional[torch.Tensor],
        x_unimol: Optional[torch.Tensor],
        x_prot: torch.Tensor,
        domain: torch.Tensor,
    ):
        drug_reprs = []

        if self.use_ecfp:
            drug_reprs.append(self.ecfp_encoder(x_ecfp))

        if self.use_unimol:
            drug_reprs.append(self.unimol_encoder(x_unimol))

        if len(drug_reprs) == 1:
            h_drug = drug_reprs[0]
        elif len(drug_reprs) == 2:
            h1, h2 = drug_reprs
            gate = self.drug_gate(torch.cat([h1, h2], dim=-1))
            h_drug = gate * h1 + (1.0 - gate) * h2
        else:
            raise RuntimeError("No drug representation is enabled.")

        h_prot = self.prot_encoder(x_prot)

        if self.use_domain_adapter:
            h_human = self.human_adapter(h_prot)
            h_virus = self.virus_adapter(h_prot)
            mask = domain.float().view(-1, 1)
            h_prot = h_human * (1.0 - mask) + h_virus * mask

        bilinear = self.drug_bilinear(h_drug) * self.prot_bilinear(h_prot)

        pair = torch.cat(
            [
                h_drug,
                h_prot,
                h_drug * h_prot,
                torch.abs(h_drug - h_prot),
                bilinear,
            ],
            dim=-1,
        )

        logits = self.head(pair).squeeze(-1)
        return logits


# -------------------------
# Train / eval
# -------------------------

def bpr_rank_loss(logits, y, groups, max_pairs_per_group: int = 512):
    losses = []
    unique_groups = torch.unique(groups)

    for g in unique_groups:
        mask = groups == g
        pos = logits[mask & (y == 1)]
        neg = logits[mask & (y == 0)]

        if pos.numel() == 0 or neg.numel() == 0:
            continue

        n_pairs = pos.numel() * neg.numel()

        if n_pairs <= max_pairs_per_group:
            diff = pos.view(-1, 1) - neg.view(1, -1)
        else:
            pi = torch.randint(0, pos.numel(), (max_pairs_per_group,), device=logits.device)
            ni = torch.randint(0, neg.numel(), (max_pairs_per_group,), device=logits.device)
            diff = pos[pi] - neg[ni]

        losses.append(F.softplus(-diff).mean())

    if not losses:
        return logits.new_tensor(0.0)

    return torch.stack(losses).mean()


def make_loader(
    dataset: ViroBindPairDataset,
    args,
    train: bool,
):
    if train and args.sampler == "protein_balanced":
        batch_sampler = ProteinBalancedBatchSampler(
            dataset=dataset,
            proteins_per_batch=args.proteins_per_batch,
            pos_per_protein=args.pos_per_protein,
            neg_per_protein=args.neg_per_protein,
            gray_per_protein=args.gray_per_protein,
            batches_per_epoch=args.batches_per_epoch if args.batches_per_epoch > 0 else None,
            seed=args.seed,
        )

        return DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            num_workers=args.num_workers,
            pin_memory=True,
        )

    return DataLoader(
        dataset,
        batch_size=args.batch_size if train else args.eval_batch_size,
        shuffle=train,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )


def build_features_on_device(ecfp, unimol, prot, feature_preset, device):
    features = {
        "prot": prot.x.to(device),
    }

    if feature_preset in ["ecfp_esmc", "ecfp_unimol_esmc"]:
        features["ecfp"] = ecfp.x.to(device)

    if feature_preset in ["unimol_esmc", "ecfp_unimol_esmc"]:
        features["unimol"] = unimol.x.to(device)

    return features


def batch_forward(model, batch, features, feature_preset, device):
    prot_idx = batch["prot_idx"].to(device, non_blocking=True)
    domain = batch["domain"].to(device, non_blocking=True)

    x_prot = features["prot"][prot_idx]

    x_ecfp = None
    x_unimol = None

    if feature_preset in ["ecfp_esmc", "ecfp_unimol_esmc"]:
        ecfp_idx = batch["ecfp_idx"].to(device, non_blocking=True)
        x_ecfp = features["ecfp"][ecfp_idx]

    if feature_preset in ["unimol_esmc", "ecfp_unimol_esmc"]:
        unimol_idx = batch["unimol_idx"].to(device, non_blocking=True)
        x_unimol = features["unimol"][unimol_idx]

    return model(x_ecfp, x_unimol, x_prot, domain)




def graylite_bce_loss(logits, y, pos_weight: float, args):
    """
    Binary BCE for label 0/1 plus optional low-weight soft supervision for label=-1.
    Ranking loss is still computed only on confirmed 0/1 labels.
    """
    target = torch.where(y == -1, torch.full_like(y, float(args.gray_target)), y)
    weight = torch.where(y == -1, torch.full_like(y, float(args.gray_bce_weight)), torch.ones_like(y))

    pw = torch.tensor([float(pos_weight)], dtype=torch.float32, device=logits.device)
    loss = F.binary_cross_entropy_with_logits(logits, target, pos_weight=pw, reduction="none")
    return (loss * weight).sum() / weight.sum().clamp_min(1.0)

def train_one_epoch(model, loader, features, device, args, optimizer, pos_weight):
    model.train()

    total_loss = 0.0
    total_bce = 0.0
    total_rank = 0.0
    total_n = 0

    for batch in loader:
        y = batch["y"].to(device, non_blocking=True)
        groups = batch["group"].to(device, non_blocking=True)

        logits = batch_forward(model, batch, features, args.feature_preset, device)

        loss_bce = graylite_bce_loss(logits, y, pos_weight, args)

        if args.loss_mode == "bce_rank":
            loss_rank = bpr_rank_loss(
                logits,
                y,
                groups,
                max_pairs_per_group=args.max_rank_pairs_per_group,
            )
        else:
            loss_rank = logits.new_tensor(0.0)

        loss = loss_bce + args.rank_lambda * loss_rank

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        bs = y.numel()
        total_n += bs
        total_loss += float(loss.item()) * bs
        total_bce += float(loss_bce.item()) * bs
        total_rank += float(loss_rank.item()) * bs

    return {
        "loss": total_loss / max(total_n, 1),
        "bce": total_bce / max(total_n, 1),
        "rank": total_rank / max(total_n, 1),
    }


@torch.no_grad()
def predict(model, dataset, loader, features, device, feature_preset):
    model.eval()

    all_y = []
    all_prob = []
    all_logit = []
    all_idx = []

    for batch in loader:
        logits = batch_forward(model, batch, features, feature_preset, device)
        prob = torch.sigmoid(logits)

        all_y.append(batch["y"].cpu().numpy())
        all_prob.append(prob.detach().cpu().numpy())
        all_logit.append(logits.detach().cpu().numpy())
        all_idx.append(batch["idx"].cpu().numpy())

    y = np.concatenate(all_y)
    prob = np.concatenate(all_prob)
    logit = np.concatenate(all_logit)
    idx = np.concatenate(all_idx)

    proteins = [dataset.group_names_per_item[int(i)] for i in idx]
    drugs = [dataset.drug_names_per_item[int(i)] for i in idx]

    return pd.DataFrame(
        {
            "drug": drugs,
            "protein": proteins,
            "label": y.astype(int),
            "logit": logit,
            "prob": prob,
        }
    )


def compute_metrics(pred_df: pd.DataFrame, threshold: Optional[float] = None):
    y = pred_df["label"].values.astype(int)
    prob = pred_df["prob"].values.astype(float)

    if threshold is None:
        threshold, _ = best_f1_threshold(y, prob)

    pred = (prob >= threshold).astype(int)

    metrics = {
        "auc": safe_auc(y, prob),
        "aupr": safe_aupr(y, prob),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "acc": float(accuracy_score(y, pred)),
        "balanced_acc": float(balanced_accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "threshold": float(threshold),
        "pp_mean": float(np.mean(prob)),
        "n": int(len(y)),
        "pos": int(np.sum(y == 1)),
        "neg": int(np.sum(y == 0)),
    }

    per_auc = []
    per_aupr = []
    p10 = []
    p50 = []
    hit10 = []
    hit50 = []

    for protein, sub in pred_df.groupby("protein"):
        sub = sub.sort_values("prob", ascending=False)
        yy = sub["label"].values.astype(int)
        pp = sub["prob"].values.astype(float)

        if len(np.unique(yy)) < 2:
            continue
        per_auc.append(roc_auc_score(yy, pp))
        per_aupr.append(average_precision_score(yy, pp))

        top10 = sub.head(10)
        top50 = sub.head(50)

        if len(top10) > 0:
            lab10 = top10["label"].values.astype(int)
            p10.append(float(np.mean(lab10)))
            hit10.append(float(np.sum(lab10) > 0))

        if len(top50) > 0:
            lab50 = top50["label"].values.astype(int)
            p50.append(float(np.mean(lab50)))
            hit50.append(float(np.sum(lab50) > 0))

    metrics.update(
        {
            "perprotein_auc_mean": float(np.mean(per_auc)) if per_auc else float("nan"),
            "perprotein_aupr_mean": float(np.mean(per_aupr)) if per_aupr else float("nan"),
            "p_at_10_mean": float(np.mean(p10)) if p10 else float("nan"),
            "p_at_50_mean": float(np.mean(p50)) if p50 else float("nan"),
            "hit_at_10_mean": float(np.mean(hit10)) if hit10 else float("nan"),
            "hit_at_50_mean": float(np.mean(hit50)) if hit50 else float("nan"),
            "n_proteins": int(pred_df["protein"].nunique()),
            "n_proteins_with_both_labels": int(len(per_aupr)),
        }
    )

    return metrics


def monitor_value(metrics: Dict[str, float], key: str):
    v = metrics.get(key, float("nan"))
    if v is None or math.isnan(float(v)):
        return -1e18
    return float(v)


def train_stage(
    stage_name,
    model,
    train_ds,
    val_ds,
    features,
    device,
    args,
    out_dir,
    lr,
    epochs,
):
    print(f"\n[STAGE] {stage_name}")
    print(f"[STAGE] lr={lr} epochs={epochs} loss_mode={args.loss_mode} sampler={args.sampler}")

    train_loader = make_loader(train_ds, args, train=True)
    val_loader = make_loader(val_ds, args, train=False)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=args.weight_decay,
    )

    pos_weight = train_ds.pos_weight()
    print(f"[STAGE] pos_weight={pos_weight:.6f}")

    monitors = [
        "aupr",
        "auc",
        "perprotein_aupr_mean",
        "p_at_10_mean",
    ]

    best = {
        m: {
            "score": -1e18,
            "epoch": -1,
            "path": os.path.join(out_dir, f"{stage_name}_best_by_{m}.pt"),
            "metrics": None,
        }
        for m in monitors
    }

    history = []
    bad_epochs = 0
    main_monitor = args.monitor

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        train_log = train_one_epoch(
            model=model,
            loader=train_loader,
            features=features,
            device=device,
            args=args,
            optimizer=optimizer,
            pos_weight=pos_weight,
        )

        val_pred = predict(
            model=model,
            dataset=val_ds,
            loader=val_loader,
            features=features,
            device=device,
            feature_preset=args.feature_preset,
        )

        val_metrics = compute_metrics(val_pred, threshold=None)

        row = {
            "stage": stage_name,
            "epoch": epoch,
            "time_sec": time.time() - t0,
            **{f"train_{k}": v for k, v in train_log.items()},
            **{f"val_{k}": v for k, v in val_metrics.items()},
        }
        history.append(row)

        print(
            f"[{stage_name}] epoch={epoch:03d} "
            f"loss={train_log['loss']:.6f} "
            f"bce={train_log['bce']:.6f} "
            f"rank={train_log['rank']:.6f} "
            f"val_auc={val_metrics['auc']:.6f} "
            f"val_aupr={val_metrics['aupr']:.6f} "
            f"val_pp_aupr={val_metrics['perprotein_aupr_mean']:.6f} "
            f"val_p10={val_metrics['p_at_10_mean']:.6f} "
            f"thr={val_metrics['threshold']:.3f}"
        )

        improved_main = False

        for m in monitors:
            score = monitor_value(val_metrics, m)
            if score > best[m]["score"]:
                best[m]["score"] = score
                best[m]["epoch"] = epoch
                best[m]["metrics"] = val_metrics

                torch.save(
                    {
                        "model": model.state_dict(),
                        "stage": stage_name,
                        "epoch": epoch,
                        "monitor": m,
                        "val_metrics": val_metrics,
                        "args": vars(args),
                    },
                    best[m]["path"],
                )

                val_pred.to_csv(
                    os.path.join(out_dir, f"{stage_name}_best_by_{m}_val_predictions.csv"),
                    index=False,
                )

                if m == main_monitor:
                    improved_main = True

        pd.DataFrame(history).to_csv(
            os.path.join(out_dir, f"{stage_name}_history.csv"),
            index=False,
        )

        if improved_main:
            bad_epochs = 0
        else:
            bad_epochs += 1

        if bad_epochs >= args.patience:
            print(
                f"[EARLY_STOP] {stage_name}: "
                f"best_{main_monitor}_epoch={best[main_monitor]['epoch']} "
                f"best_{main_monitor}={best[main_monitor]['score']:.6f}"
            )
            break

    # load main monitor best checkpoint
    ckpt = torch.load(best[main_monitor]["path"], map_location=device)
    model.load_state_dict(ckpt["model"])

    return best


# -------------------------
# Data resolver
# -------------------------

def first_existing(paths: List[str]) -> Optional[str]:
    for p in paths:
        if p and os.path.exists(p):
            return p
    return None


def first_glob(patterns: List[str]) -> Optional[str]:
    hits = []
    for pat in patterns:
        hits += glob.glob(pat)
    hits = sorted(set(hits))
    return hits[0] if hits else None


def resolve_csvs(args):
    mode = args.mode
    release_dir = os.path.join(args.dataset_root, mode)
    allhuman_dir = os.path.join(args.allhuman_root, mode) if args.allhuman_root else ""
    virus_dir = os.path.join(args.virusonly_root, mode) if args.virusonly_root else ""

    pretrain_csv = args.pretrain_csv or first_existing([
        os.path.join(release_dir, "human_pretrain.csv"),
        os.path.join(allhuman_dir, f"train_mixed_binary_{mode}.csv"),
        os.path.join(allhuman_dir, f"train_mixed_{mode}.csv"),
    ]) or first_glob([
        os.path.join(allhuman_dir, "train*mixed*.csv"),
        os.path.join(allhuman_dir, "train*.csv"),
    ])

    train_csv = args.train_csv or first_existing([
        os.path.join(release_dir, "virus_finetune.csv"),
        os.path.join(virus_dir, f"train_virus_only_{mode}.csv"),
        os.path.join(virus_dir, f"train_mixed_binary_{mode}.csv"),
        os.path.join(virus_dir, f"train_{mode}.csv"),
        os.path.join(allhuman_dir, f"train_virus_only_{mode}.csv"),
    ]) or first_glob([
        os.path.join(virus_dir, "train*virus*only*.csv"),
        os.path.join(virus_dir, "train*.csv"),
    ])

    val_csv = args.val_csv or first_existing([
        os.path.join(release_dir, "virus_val.csv"),
        os.path.join(virus_dir, f"val_{mode}.csv"),
        os.path.join(virus_dir, "val.csv"),
        os.path.join(allhuman_dir, f"val_{mode}.csv"),
    ]) or first_glob([
        os.path.join(virus_dir, "val*.csv"),
        os.path.join(allhuman_dir, "val*.csv"),
    ])

    test_csv = args.test_csv or first_existing([
        os.path.join(release_dir, "virus_test.csv"),
        os.path.join(virus_dir, f"test_{mode}.csv"),
        os.path.join(virus_dir, "test.csv"),
        os.path.join(allhuman_dir, f"test_{mode}.csv"),
    ]) or first_glob([
        os.path.join(virus_dir, "test*.csv"),
        os.path.join(allhuman_dir, "test*.csv"),
    ])

    if args.protocol == "allhuman_pt" and not pretrain_csv:
        raise FileNotFoundError(f"Cannot find pretrain csv under {allhuman_dir}")

    if not train_csv:
        raise FileNotFoundError(f"Cannot find virusonly train csv under {virus_dir}")
    if not val_csv:
        raise FileNotFoundError(f"Cannot find val csv under {virus_dir}")
    if not test_csv:
        raise FileNotFoundError(f"Cannot find test csv under {virus_dir}")

    return {
        "pretrain_csv": pretrain_csv,
        "train_csv": train_csv,
        "val_csv": val_csv,
        "test_csv": test_csv,
    }


# -------------------------
# Main
# -------------------------

def parse_args():
    ap = argparse.ArgumentParser()

    ap.add_argument("--protocol", default="allhuman_pt", choices=["virusonly", "allhuman_pt"])
    ap.add_argument(
        "--mode",
        default="scaffold_cluster_cold_protein",
        choices=["random", "scaffold", "cluster_cold_protein", "scaffold_cluster_cold_protein"],
    )
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--dataset_root", default=os.path.join(PROJECT_ROOT, "Datasets"))
    ap.add_argument("--allhuman_root", default="", help="Optional legacy all-human split root.")
    ap.add_argument("--virusonly_root", default="", help="Optional legacy virus-only split root.")

    ap.add_argument("--pretrain_csv", default="")
    ap.add_argument("--train_csv", default="")
    ap.add_argument("--val_csv", default="")
    ap.add_argument("--test_csv", default="")

    ap.add_argument(
        "--ecfp_path",
        default=os.path.join(PROJECT_ROOT, "Feature_generation/features/drug/drug_combo_ecfp4_maccs_rdkit2d.pt"),
    )
    ap.add_argument("--unimol_path", default=os.path.join(PROJECT_ROOT, "features/drug_unimol_embedding.pt"))
    ap.add_argument(
        "--prot_path",
        default=os.path.join(PROJECT_ROOT, "Feature_generation/features/protein/prot_esmc_fullseq.pt"),
    )

    ap.add_argument(
        "--feature_preset",
        default="ecfp_esmc",
        choices=["ecfp_esmc", "unimol_esmc", "ecfp_unimol_esmc"],
    )

    ap.add_argument("--loss_mode", default="bce_rank", choices=["bce", "bce_rank"])
    ap.add_argument("--sampler", default="protein_balanced", choices=["random", "protein_balanced"])

    ap.add_argument("--out_root", default=os.path.join(PROJECT_ROOT, "outputs/training"))

    ap.add_argument("--hidden_dim", type=int, default=512)
    ap.add_argument("--bilinear_dim", type=int, default=128)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--use_domain_adapter", type=int, default=0)

    ap.add_argument("--pretrain_epochs", type=int, default=20)
    ap.add_argument("--train_epochs", type=int, default=80)
    ap.add_argument("--pretrain_lr", type=float, default=1e-3)
    ap.add_argument("--train_lr", type=float, default=1e-4)

    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--eval_batch_size", type=int, default=2048)
    ap.add_argument("--num_workers", type=int, default=0)

    ap.add_argument("--proteins_per_batch", type=int, default=16)
    ap.add_argument("--pos_per_protein", type=int, default=4)
    ap.add_argument("--neg_per_protein", type=int, default=4)
    ap.add_argument("--batches_per_epoch", type=int, default=0)

    ap.add_argument("--rank_lambda", type=float, default=0.1)
    ap.add_argument("--max_rank_pairs_per_group", type=int, default=512)

    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--grad_clip", type=float, default=5.0)
    ap.add_argument("--patience", type=int, default=10)

    ap.add_argument(
        "--monitor",
        default="aupr",
        choices=["aupr", "auc", "perprotein_aupr_mean", "p_at_10_mean"],
    )

    ap.add_argument("--pretrain_label_mode", default="binary", choices=["binary", "gray"])
    ap.add_argument("--train_label_mode", default="binary", choices=["binary", "gray"])
    ap.add_argument("--gray_target", type=float, default=0.20)
    ap.add_argument("--gray_bce_weight", type=float, default=0.03)
    ap.add_argument("--gray_per_protein", type=int, default=0)

    ap.add_argument("--dry_run", type=int, default=0)

    return ap.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    exp_name = (
        f"{args.protocol}"
        f"__{args.feature_preset}"
        f"__{args.loss_mode}"
        f"__{args.sampler}"
        f"__adapter{args.use_domain_adapter}"
    )

    out_dir = os.path.join(
        args.out_root,
        args.mode,
        exp_name,
        f"seed_{args.seed}",
    )

    ensure_dir(out_dir)

    print(f"[INFO] device={device}")
    print(f"[INFO] out_dir={out_dir}")
    print(f"[INFO] exp_name={exp_name}")

    csvs = resolve_csvs(args)
    print("[INFO] Resolved CSVs:")
    for k, v in csvs.items():
        print(f"  {k}: {v}")

    ecfp = None
    unimol = None

    if args.feature_preset in ["ecfp_esmc", "ecfp_unimol_esmc"]:
        ecfp = FeatureBank(args.ecfp_path, "drug_ecfp")

    if args.feature_preset in ["unimol_esmc", "ecfp_unimol_esmc"]:
        unimol = FeatureBank(args.unimol_path, "drug_unimol")

    prot = FeatureBank(args.prot_path, "prot_esmc_global")

    pretrain_ds = None

    if args.protocol == "allhuman_pt":
        pretrain_ds = ViroBindPairDataset(
            csv_path=csvs["pretrain_csv"],
            ecfp=ecfp,
            unimol=unimol,
            prot=prot,
            feature_preset=args.feature_preset,
            name=f"pretrain_allhuman_label_mode_{args.pretrain_label_mode}",
            labeled_only=(args.pretrain_label_mode == "binary"),
            label_mode=args.pretrain_label_mode,
        )

    train_ds = ViroBindPairDataset(
        csv_path=csvs["train_csv"],
        ecfp=ecfp,
        unimol=unimol,
        prot=prot,
        feature_preset=args.feature_preset,
        name=f"train_virusonly_label_mode_{args.train_label_mode}",
        labeled_only=(args.train_label_mode == "binary"),
        label_mode=args.train_label_mode,
    )

    val_ds = ViroBindPairDataset(
        csv_path=csvs["val_csv"],
        ecfp=ecfp,
        unimol=unimol,
        prot=prot,
        feature_preset=args.feature_preset,
        name="val_virusonly_labeled_0_1",
        labeled_only=True,
        label_mode="binary",
    )

    test_ds = ViroBindPairDataset(
        csv_path=csvs["test_csv"],
        ecfp=ecfp,
        unimol=unimol,
        prot=prot,
        feature_preset=args.feature_preset,
        name="test_virusonly_labeled_0_1",
        labeled_only=True,
        label_mode="binary",
    )

    config = {
        "args": vars(args),
        "csvs": csvs,
        "exp_name": exp_name,
        "out_dir": out_dir,
    }
    json_dump(config, os.path.join(out_dir, "config.json"))

    if args.dry_run:
        print("[DRY_RUN] finished dataset and feature checking.")
        return

    ecfp_dim = ecfp.dim if ecfp is not None else 1
    unimol_dim = unimol.dim if unimol is not None else 1

    model = ViroBindV2(
        feature_preset=args.feature_preset,
        ecfp_dim=ecfp_dim,
        unimol_dim=unimol_dim,
        prot_dim=prot.dim,
        hidden_dim=args.hidden_dim,
        bilinear_dim=args.bilinear_dim,
        dropout=args.dropout,
        use_domain_adapter=bool(args.use_domain_adapter),
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[MODEL] trainable_params={n_params:,}")

    features = build_features_on_device(
        ecfp=ecfp,
        unimol=unimol,
        prot=prot,
        feature_preset=args.feature_preset,
        device=device,
    )

    final_summary = {
        "protocol": args.protocol,
        "mode": args.mode,
        "seed": args.seed,
        "exp_name": exp_name,
        "feature_preset": args.feature_preset,
        "loss_mode": args.loss_mode,
        "sampler": args.sampler,
        "use_domain_adapter": int(args.use_domain_adapter),
        "hidden_dim": args.hidden_dim,
        "bilinear_dim": args.bilinear_dim,
        "dropout": args.dropout,
        "n_params": n_params,
    }

    if args.protocol == "allhuman_pt":
        pretrain_best = train_stage(
            stage_name="pretrain",
            model=model,
            train_ds=pretrain_ds,
            val_ds=val_ds,
            features=features,
            device=device,
            args=args,
            out_dir=out_dir,
            lr=args.pretrain_lr,
            epochs=args.pretrain_epochs,
        )

        for m, obj in pretrain_best.items():
            final_summary[f"pretrain_best_{m}_epoch"] = obj["epoch"]
            final_summary[f"pretrain_best_{m}_score"] = obj["score"]

    train_best = train_stage(
        stage_name="train",
        model=model,
        train_ds=train_ds,
        val_ds=val_ds,
        features=features,
        device=device,
        args=args,
        out_dir=out_dir,
        lr=args.train_lr,
        epochs=args.train_epochs,
    )

    test_loader = make_loader(test_ds, args, train=False)
    val_loader = make_loader(val_ds, args, train=False)

    # Evaluate all saved training-stage best checkpoints
    for monitor, obj in train_best.items():
        ckpt_path = obj["path"]
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model"])

        val_pred = predict(
            model=model,
            dataset=val_ds,
            loader=val_loader,
            features=features,
            device=device,
            feature_preset=args.feature_preset,
        )
        val_metrics = compute_metrics(val_pred, threshold=None)
        threshold = val_metrics["threshold"]

        test_pred = predict(
            model=model,
            dataset=test_ds,
            loader=test_loader,
            features=features,
            device=device,
            feature_preset=args.feature_preset,
        )
        test_metrics = compute_metrics(test_pred, threshold=threshold)

        val_pred.to_csv(
            os.path.join(out_dir, f"final_best_by_{monitor}_val_predictions.csv"),
            index=False,
        )
        test_pred.to_csv(
            os.path.join(out_dir, f"final_best_by_{monitor}_test_predictions.csv"),
            index=False,
        )

        for k, v in val_metrics.items():
            final_summary[f"best_by_{monitor}_val_{k}"] = v
        for k, v in test_metrics.items():
            final_summary[f"best_by_{monitor}_test_{k}"] = v

        final_summary[f"best_by_{monitor}_epoch"] = obj["epoch"]
        final_summary[f"best_by_{monitor}_ckpt"] = ckpt_path

    json_dump(final_summary, os.path.join(out_dir, "metrics.json"))
    pd.DataFrame([final_summary]).to_csv(os.path.join(out_dir, "metrics.csv"), index=False)

    print("\n[FINAL SUMMARY]")
    for monitor in ["aupr", "auc", "perprotein_aupr_mean", "p_at_10_mean"]:
        print(f"----- best_by_{monitor} -----")
        for k in [
            f"best_by_{monitor}_val_auc",
            f"best_by_{monitor}_val_aupr",
            f"best_by_{monitor}_val_perprotein_aupr_mean",
            f"best_by_{monitor}_val_p_at_10_mean",
            f"best_by_{monitor}_test_auc",
            f"best_by_{monitor}_test_aupr",
            f"best_by_{monitor}_test_perprotein_aupr_mean",
            f"best_by_{monitor}_test_p_at_10_mean",
            f"best_by_{monitor}_test_hit_at_10_mean",
        ]:
            print(f"{k}: {final_summary.get(k)}")

    print(f"[SAVED] {out_dir}")


if __name__ == "__main__":
    main()
