#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ViroBind v5: multi-view fusion with decoupled classification and ranking heads."""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score, f1_score, matthews_corrcoef
from torch.utils.data import Sampler

from . import base


PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parents[1]

# Import-safe defaults. Training CLI arguments or checkpoint metadata replace
# these values through configure_from_namespace() before model construction.
V5_MONITOR = "rank_perprotein_aupr_mean"
RANK_LOSS_MODE = "bpr"
RANK_MARGIN = 0.5
CLS_LOSS_MODE = "bce"
FOCAL_GAMMA = 2.0
FOCAL_ALPHA = 0.5
CLS_THRESHOLD_METRIC = "f1"
MIN_CLS_THRESHOLD = 0.0
FIXED_CLS_THRESHOLD = 0.5
GRAY_CLS_WEIGHT = None
PRETRAIN_GRAY_PER_PROTEIN = None
CLS_TRAIN_GRAY_PER_PROTEIN = None
RANK_TRAIN_GRAY_PER_PROTEIN = 0
RANK_TRAIN_CSV = ""
RANK_TRAIN_EPOCHS = None
RANK_TRAIN_LR = None
RANK_TRAIN_SCOPE = "rank_head"
CLS_CHECKPOINT_POLICY = "best_stage"
DRUG_FUSION_MODE = "residual_multiview"
DRUG_ECFP_DIM = 2048
DRUG_MACCS_DIM = 167
DRUG_RDKIT_DIM = 208
DRUG_VIEW_MASK = "1,1,1"
DRUG_VIEW_ACTIVE = [1.0, 1.0, 1.0]
DRUG_VIEW_MASK_VECTOR = [1.0] * (DRUG_ECFP_DIM + DRUG_MACCS_DIM + DRUG_RDKIT_DIM)
DRUG_AUX_GATE_INIT = -5.0
DRUG_AUX_GATE_MAX = 0.10
PROTEIN_FUSION_MODE = "esmc_only"
PROTEIN_DESC_INPUT_SCALE = 1.0
PROTEIN_ESMC_DIM = 1152
PROTEIN_DESC_DIM = 574
PROTEIN_DESC_GATE_INIT = -5.0
PROTEIN_DESC_GATE_MAX = 1.0
PROTEIN_DESC_GATE_MODE = "learned"
PROTEIN_DESC_GATE_VALUE = 1.0
TOKEN_ROOT = ""
TOKEN_MAX_TOKENS = 1024
TOKEN_GATE_INIT = -4.0
TOKEN_GATE_MAX = 0.10
RESIDUE_RANK_ONLY = False
INIT_CHECKPOINT = ""
TOKEN_STORE = None
PAIR_FEATURE_MODE = "full_bilinear"
ORIG_PARSE_ARGS = base.parse_args
ORIG_COMPUTE_METRICS = base.compute_metrics


def configure_from_namespace(args, token_root_override=None):
    """Configure the architecture from CLI arguments or checkpoint metadata."""
    global V5_MONITOR, RANK_LOSS_MODE, RANK_MARGIN
    global CLS_LOSS_MODE, FOCAL_GAMMA, FOCAL_ALPHA
    global CLS_THRESHOLD_METRIC, MIN_CLS_THRESHOLD, FIXED_CLS_THRESHOLD
    global GRAY_CLS_WEIGHT, PRETRAIN_GRAY_PER_PROTEIN
    global CLS_TRAIN_GRAY_PER_PROTEIN, RANK_TRAIN_GRAY_PER_PROTEIN
    global RANK_TRAIN_CSV, RANK_TRAIN_EPOCHS, RANK_TRAIN_LR
    global RANK_TRAIN_SCOPE, CLS_CHECKPOINT_POLICY
    global DRUG_FUSION_MODE, DRUG_ECFP_DIM, DRUG_MACCS_DIM, DRUG_RDKIT_DIM
    global DRUG_VIEW_MASK, DRUG_VIEW_ACTIVE, DRUG_VIEW_MASK_VECTOR
    global DRUG_AUX_GATE_INIT, DRUG_AUX_GATE_MAX
    global PROTEIN_FUSION_MODE, PROTEIN_DESC_INPUT_SCALE, PROTEIN_ESMC_DIM
    global PROTEIN_DESC_DIM, PROTEIN_DESC_GATE_INIT, PROTEIN_DESC_GATE_MAX
    global PROTEIN_DESC_GATE_MODE, PROTEIN_DESC_GATE_VALUE
    global TOKEN_ROOT, TOKEN_MAX_TOKENS, TOKEN_GATE_INIT, TOKEN_GATE_MAX
    global RESIDUE_RANK_ONLY, INIT_CHECKPOINT, TOKEN_STORE, PAIR_FEATURE_MODE

    V5_MONITOR = getattr(args, "v5_monitor", V5_MONITOR)
    RANK_LOSS_MODE = getattr(args, "rank_loss_mode", RANK_LOSS_MODE)
    RANK_MARGIN = float(getattr(args, "rank_margin", RANK_MARGIN))
    CLS_LOSS_MODE = getattr(args, "cls_loss_mode", CLS_LOSS_MODE)
    FOCAL_GAMMA = float(getattr(args, "focal_gamma", FOCAL_GAMMA))
    FOCAL_ALPHA = float(getattr(args, "focal_alpha", FOCAL_ALPHA))
    CLS_THRESHOLD_METRIC = getattr(args, "cls_threshold_metric", CLS_THRESHOLD_METRIC)
    MIN_CLS_THRESHOLD = float(getattr(args, "min_cls_threshold", MIN_CLS_THRESHOLD))
    FIXED_CLS_THRESHOLD = float(getattr(args, "fixed_cls_threshold", FIXED_CLS_THRESHOLD))
    GRAY_CLS_WEIGHT = getattr(args, "gray_cls_weight", GRAY_CLS_WEIGHT)
    PRETRAIN_GRAY_PER_PROTEIN = getattr(args, "pretrain_gray_per_protein", PRETRAIN_GRAY_PER_PROTEIN)
    CLS_TRAIN_GRAY_PER_PROTEIN = getattr(args, "cls_train_gray_per_protein", CLS_TRAIN_GRAY_PER_PROTEIN)
    RANK_TRAIN_GRAY_PER_PROTEIN = int(getattr(args, "rank_train_gray_per_protein", RANK_TRAIN_GRAY_PER_PROTEIN))
    RANK_TRAIN_CSV = getattr(args, "rank_train_csv", RANK_TRAIN_CSV)
    RANK_TRAIN_EPOCHS = getattr(args, "rank_train_epochs", RANK_TRAIN_EPOCHS)
    RANK_TRAIN_LR = getattr(args, "rank_train_lr", RANK_TRAIN_LR)
    RANK_TRAIN_SCOPE = getattr(args, "rank_train_scope", RANK_TRAIN_SCOPE)
    CLS_CHECKPOINT_POLICY = getattr(args, "cls_checkpoint_policy", CLS_CHECKPOINT_POLICY)
    DRUG_FUSION_MODE = getattr(args, "drug_fusion_mode", DRUG_FUSION_MODE)
    DRUG_ECFP_DIM = int(getattr(args, "drug_ecfp_dim", DRUG_ECFP_DIM))
    DRUG_MACCS_DIM = int(getattr(args, "drug_maccs_dim", DRUG_MACCS_DIM))
    DRUG_RDKIT_DIM = int(getattr(args, "drug_rdkit_dim", DRUG_RDKIT_DIM))
    DRUG_VIEW_MASK = str(getattr(args, "drug_view_mask", DRUG_VIEW_MASK))
    DRUG_VIEW_ACTIVE = [float(x) for x in DRUG_VIEW_MASK.split(",")]
    if len(DRUG_VIEW_ACTIVE) != 3:
        raise ValueError("drug_view_mask must contain three comma-separated values")
    DRUG_VIEW_MASK_VECTOR = (
        [DRUG_VIEW_ACTIVE[0]] * DRUG_ECFP_DIM
        + [DRUG_VIEW_ACTIVE[1]] * DRUG_MACCS_DIM
        + [DRUG_VIEW_ACTIVE[2]] * DRUG_RDKIT_DIM
    )
    DRUG_AUX_GATE_INIT = float(getattr(args, "drug_aux_gate_init", DRUG_AUX_GATE_INIT))
    DRUG_AUX_GATE_MAX = float(getattr(args, "drug_aux_gate_max", DRUG_AUX_GATE_MAX))
    PROTEIN_FUSION_MODE = getattr(args, "protein_fusion_mode", PROTEIN_FUSION_MODE)
    PROTEIN_DESC_INPUT_SCALE = float(getattr(args, "protein_desc_input_scale", PROTEIN_DESC_INPUT_SCALE))
    PROTEIN_ESMC_DIM = int(getattr(args, "protein_esmc_dim", PROTEIN_ESMC_DIM))
    PROTEIN_DESC_DIM = int(getattr(args, "protein_desc_dim", PROTEIN_DESC_DIM))
    PROTEIN_DESC_GATE_INIT = float(getattr(args, "protein_desc_gate_init", PROTEIN_DESC_GATE_INIT))
    PROTEIN_DESC_GATE_MAX = float(getattr(args, "protein_desc_gate_max", PROTEIN_DESC_GATE_MAX))
    PROTEIN_DESC_GATE_MODE = getattr(args, "protein_desc_gate_mode", PROTEIN_DESC_GATE_MODE)
    PROTEIN_DESC_GATE_VALUE = float(getattr(args, "protein_desc_gate_value", PROTEIN_DESC_GATE_VALUE))
    checkpoint_token_root = getattr(args, "token_root", TOKEN_ROOT)
    TOKEN_ROOT = str(checkpoint_token_root if token_root_override is None else token_root_override)
    TOKEN_MAX_TOKENS = int(getattr(args, "token_max_tokens", TOKEN_MAX_TOKENS))
    TOKEN_GATE_INIT = float(getattr(args, "token_gate_init", TOKEN_GATE_INIT))
    TOKEN_GATE_MAX = float(getattr(args, "token_gate_max", TOKEN_GATE_MAX))
    RESIDUE_RANK_ONLY = bool(getattr(args, "residue_rank_only", RESIDUE_RANK_ONLY))
    INIT_CHECKPOINT = getattr(args, "init_checkpoint", INIT_CHECKPOINT)
    PAIR_FEATURE_MODE = getattr(args, "pair_feature_mode", PAIR_FEATURE_MODE)
    TOKEN_STORE = None


def parse_v5_args(argv):
    ap = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    ap.add_argument(
        "--v5_monitor",
        choices=[
            "cls_aupr",
            "cls_f1",
            "cls_balanced_acc",
            "rank_perprotein_aupr_mean",
            "combo_cls_rank",
            "combo_cls_f1_rank",
        ],
        default="rank_perprotein_aupr_mean",
    )
    ap.add_argument("--rank_loss_mode", choices=["bpr", "margin", "bce"], default="bpr")
    ap.add_argument("--rank_margin", type=float, default=0.5)
    ap.add_argument("--cls_loss_mode", choices=["bce", "focal"], default="bce")
    ap.add_argument("--focal_gamma", type=float, default=2.0)
    ap.add_argument("--focal_alpha", type=float, default=0.5)
    ap.add_argument("--cls_threshold_metric", choices=["f1", "balanced_acc", "youden", "mcc", "fixed"], default="f1")
    ap.add_argument("--min_cls_threshold", type=float, default=0.0)
    ap.add_argument("--fixed_cls_threshold", type=float, default=0.5)
    ap.add_argument("--gray_cls_weight", type=float, default=None)
    ap.add_argument("--pretrain_gray_per_protein", type=int, default=None)
    ap.add_argument("--cls_train_gray_per_protein", type=int, default=None)
    ap.add_argument("--rank_train_gray_per_protein", type=int, default=0)
    ap.add_argument("--rank_train_csv", default="")
    ap.add_argument("--rank_train_epochs", type=int, default=50)
    ap.add_argument("--rank_train_lr", type=float, default=3e-4)
    ap.add_argument("--rank_train_scope", choices=["rank_head", "rank_head_adapter", "rank_head_pair"], default="rank_head")
    ap.add_argument("--cls_checkpoint_policy", choices=["finetune", "best_stage"], default="best_stage")
    ap.add_argument("--drug_ecfp_dim", type=int, default=2048)
    ap.add_argument("--drug_maccs_dim", type=int, default=167)
    ap.add_argument("--drug_rdkit_dim", type=int, default=208)
    ap.add_argument("--drug_fusion_mode", choices=["direct_mlp", "tri_gate", "residual_multiview", "independent_residual"], default="residual_multiview")
    ap.add_argument("--drug_view_mask", default="1,1,1")
    ap.add_argument("--drug_aux_gate_init", type=float, default=-5.0)
    ap.add_argument("--drug_aux_gate_max", type=float, default=0.10)
    ap.add_argument("--protein_esmc_dim", type=int, default=1152)
    ap.add_argument("--protein_desc_dim", type=int, default=574)
    ap.add_argument("--protein_desc_gate_init", type=float, default=-4.0)
    ap.add_argument("--protein_desc_gate_max", type=float, default=1.0)
    ap.add_argument("--protein_desc_gate_mode", choices=["learned", "fixed"], default="learned")
    ap.add_argument("--protein_desc_gate_value", type=float, default=1.0)
    ap.add_argument("--protein_fusion_mode", choices=["esmc_only", "residual_gate"], default="esmc_only")
    ap.add_argument("--protein_desc_input_scale", type=float, default=1.0)
    # Optional residue-token branch.  It is deliberately opt-in so existing V5H
    # checkpoints and commands remain unchanged.
    ap.add_argument(
        "--token_root",
        default=str(PROJECT_ROOT / "Feature_generation/features/protein/prot_esmc_residue_tokens"),
        help="Directory containing manifest.csv and ESMC residue token files.",
    )
    ap.add_argument("--token_max_tokens", type=int, default=1024)
    ap.add_argument("--token_gate_init", type=float, default=-4.0)
    ap.add_argument("--token_gate_max", type=float, default=0.10)
    ap.add_argument("--residue_rank_only", action="store_true", help="Keep the V5H classification path frozen; train residue tokens only for ranking.")
    ap.add_argument("--init_checkpoint", default="", help="Baseline V5H checkpoint required by --residue_rank_only.")
    ap.add_argument(
        "--pair_feature_mode",
        choices=["concat_only", "concat_product", "concat_product_absdiff", "full_bilinear"],
        default="full_bilinear",
    )
    return ap.parse_known_args(argv)


class V5TemperatureProteinBalancedBatchSampler(Sampler[List[int]]):
    def __init__(
        self,
        dataset,
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

        self.valid_groups = sorted(set(self.group_to_pos) & set(self.group_to_neg))
        if not self.valid_groups:
            raise RuntimeError("V5 sampler: no protein has both positive and negative samples.")

        batch_size = proteins_per_batch * (pos_per_protein + neg_per_protein + gray_per_protein)
        if batches_per_epoch is None:
            batches_per_epoch = max(1, math.ceil(len(dataset) / batch_size))
        self.batches_per_epoch = int(batches_per_epoch)
        print("[SAMPLER] v5_protein_balanced")
        print(f"[SAMPLER] valid_proteins={len(self.valid_groups)}")
        print(f"[SAMPLER] gray_per_protein={gray_per_protein}")
        print(f"[SAMPLER] batch_size={batch_size}")
        print(f"[SAMPLER] batches_per_epoch={self.batches_per_epoch}")

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        self.epoch += 1
        groups_arr = np.asarray(self.valid_groups)
        replace_group = len(groups_arr) < self.proteins_per_batch
        for _ in range(self.batches_per_epoch):
            groups = rng.choice(groups_arr, size=self.proteins_per_batch, replace=replace_group)
            batch = []
            for gg in groups:
                gg = int(gg)
                pos_pool = self.group_to_pos[gg]
                neg_pool = self.group_to_neg[gg]
                batch.extend(rng.choice(pos_pool, size=self.pos_per_protein, replace=len(pos_pool) < self.pos_per_protein).tolist())
                batch.extend(rng.choice(neg_pool, size=self.neg_per_protein, replace=len(neg_pool) < self.neg_per_protein).tolist())
                if self.gray_per_protein > 0:
                    gray_pool = self.group_to_gray.get(gg, [])
                    if gray_pool:
                        batch.extend(rng.choice(gray_pool, size=self.gray_per_protein, replace=len(gray_pool) < self.gray_per_protein).tolist())
            rng.shuffle(batch)
            yield batch

    def __len__(self):
        return self.batches_per_epoch


class DrugTriViewGateEncoder(nn.Module):
    def __init__(self, in_dim: int, ecfp_dim: int, maccs_dim: int, rdkit_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        if in_dim != ecfp_dim + maccs_dim + rdkit_dim:
            raise ValueError(f"drug dim mismatch: {in_dim} != {ecfp_dim}+{maccs_dim}+{rdkit_dim}")
        self.ecfp_dim = ecfp_dim
        self.maccs_dim = maccs_dim
        self.rdkit_dim = rdkit_dim
        self.ecfp_encoder = base.MLPEncoder(ecfp_dim, hidden_dim, dropout)
        self.maccs_encoder = base.MLPEncoder(maccs_dim, hidden_dim, dropout)
        self.rdkit_encoder = base.MLPEncoder(rdkit_dim, hidden_dim, dropout)
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 3),
        )

    def forward(self, x):
        i = 0
        x_ecfp = x[:, i : i + self.ecfp_dim]
        i += self.ecfp_dim
        x_maccs = x[:, i : i + self.maccs_dim]
        i += self.maccs_dim
        x_rdkit = x[:, i : i + self.rdkit_dim]
        h1 = self.ecfp_encoder(x_ecfp)
        h2 = self.maccs_encoder(x_maccs)
        h3 = self.rdkit_encoder(x_rdkit)
        view_mask = h1.new_tensor(DRUG_VIEW_ACTIVE)
        h1 = h1 * view_mask[0]
        h2 = h2 * view_mask[1]
        h3 = h3 * view_mask[2]
        h = torch.stack([h1, h2, h3], dim=1)
        w = torch.softmax(self.gate(torch.cat([h1, h2, h3], dim=-1)), dim=-1).unsqueeze(-1)
        return (w * h).sum(dim=1), w.squeeze(-1)


class DrugBaselineResidualMultiViewEncoder(nn.Module):
    def __init__(
        self,
        in_dim: int,
        ecfp_dim: int,
        maccs_dim: int,
        rdkit_dim: int,
        hidden_dim: int,
        dropout: float,
        gate_init: float,
    ):
        super().__init__()
        self.base_encoder = base.MLPEncoder(in_dim, hidden_dim, dropout)
        self.aux_encoder = DrugTriViewGateEncoder(
            in_dim, ecfp_dim, maccs_dim, rdkit_dim, hidden_dim, dropout
        )
        self.aux_gate_logit = nn.Parameter(torch.tensor(float(gate_init)))

    def forward(self, x):
        h_base = self.base_encoder(x)
        h_aux, view_weights = self.aux_encoder(x)
        gate = torch.sigmoid(self.aux_gate_logit)
        return h_base + gate * h_aux, view_weights, gate


class DrugIndependentResidualEncoder(nn.Module):
    def __init__(
        self,
        in_dim: int,
        ecfp_dim: int,
        maccs_dim: int,
        rdkit_dim: int,
        hidden_dim: int,
        dropout: float,
        gate_init: float,
        gate_max: float,
    ):
        super().__init__()
        if in_dim != ecfp_dim + maccs_dim + rdkit_dim:
            raise ValueError(f"drug dim mismatch: {in_dim} != {ecfp_dim}+{maccs_dim}+{rdkit_dim}")
        self.ecfp_dim = ecfp_dim
        self.maccs_dim = maccs_dim
        self.rdkit_dim = rdkit_dim
        self.gate_max = float(gate_max)
        self.base_encoder = base.MLPEncoder(in_dim, hidden_dim, dropout)
        self.ecfp_encoder = base.MLPEncoder(ecfp_dim, hidden_dim, dropout)
        self.maccs_encoder = base.MLPEncoder(maccs_dim, hidden_dim, dropout)
        self.rdkit_encoder = base.MLPEncoder(rdkit_dim, hidden_dim, dropout)
        self.gate_logits = nn.Parameter(torch.full((3,), float(gate_init)))

    def forward(self, x):
        i = 0
        x_ecfp = x[:, i : i + self.ecfp_dim]
        i += self.ecfp_dim
        x_maccs = x[:, i : i + self.maccs_dim]
        i += self.maccs_dim
        x_rdkit = x[:, i : i + self.rdkit_dim]
        h_base = self.base_encoder(x)
        h_views = torch.stack(
            [
                self.ecfp_encoder(x_ecfp),
                self.maccs_encoder(x_maccs),
                self.rdkit_encoder(x_rdkit),
            ],
            dim=1,
        )
        h_views = h_views * h_views.new_tensor(DRUG_VIEW_ACTIVE).view(1, 3, 1)
        gates = self.gate_max * torch.sigmoid(self.gate_logits)
        residual = (gates.view(1, 3, 1) * h_views).sum(dim=1)
        return h_base + residual, gates.expand(x.shape[0], -1), gates.mean()


class ProteinDualViewResidualEncoder(nn.Module):
    def __init__(self, in_dim: int, esmc_dim: int, desc_dim: int, hidden_dim: int, dropout: float, gate_init: float):
        super().__init__()
        if in_dim != esmc_dim + desc_dim:
            raise ValueError(f"protein dim mismatch: {in_dim} != {esmc_dim}+{desc_dim}")
        self.esmc_dim = esmc_dim
        self.desc_dim = desc_dim
        self.esmc_encoder = base.MLPEncoder(esmc_dim, hidden_dim, dropout)
        self.desc_encoder = nn.Sequential(
            nn.Linear(desc_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.desc_gate_logit = nn.Parameter(torch.tensor(float(gate_init)))

    def forward(self, x):
        x_esmc = x[:, : self.esmc_dim]
        x_desc = x[:, self.esmc_dim : self.esmc_dim + self.desc_dim]
        h_esmc = self.esmc_encoder(x_esmc)
        h_desc = self.desc_encoder(x_desc) * PROTEIN_DESC_INPUT_SCALE
        if PROTEIN_DESC_GATE_MODE == "fixed":
            gate = h_desc.new_tensor(float(PROTEIN_DESC_GATE_VALUE))
        else:
            gate = PROTEIN_DESC_GATE_MAX * torch.sigmoid(self.desc_gate_logit)
        return h_esmc + gate * h_desc


class ProteinEsmcOnlyEncoder(nn.Module):
    def __init__(self, in_dim: int, esmc_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        if in_dim not in [esmc_dim, esmc_dim + PROTEIN_DESC_DIM]:
            raise ValueError(f"protein_fusion_mode=esmc_only expects dim {esmc_dim} or {esmc_dim + PROTEIN_DESC_DIM}, got {in_dim}")
        self.esmc_dim = esmc_dim
        self.encoder = base.MLPEncoder(esmc_dim, hidden_dim, dropout)

    def forward(self, x):
        return self.encoder(x[:, : self.esmc_dim])


class ResidueTokenStore:
    """Map the global protein feature index to its frozen residue-aligned ESMC tokens."""
    def __init__(self, root, prot_bank, max_cache=128):
        from collections import OrderedDict
        from pathlib import Path
        self.root = Path(root)
        manifest = pd.read_csv(self.root / "manifest.csv")
        self.paths, self.cache, self.max_cache = {}, OrderedDict(), int(max_cache)
        for row in manifest.itertuples(index=False):
            idx = prot_bank.lookup([getattr(row, "prot_id"), getattr(row, "protein_key")])
            if idx is not None:
                self.paths[int(idx)] = str(getattr(row, "token_file"))
        if len(self.paths) != prot_bank.x.shape[0]:
            raise RuntimeError(f"Token/global protein mismatch: mapped {len(self.paths)} of {prot_bank.x.shape[0]} proteins")

    def _get(self, idx):
        idx = int(idx)
        if idx not in self.paths:
            raise KeyError(f"No residue tokens for global protein feature index {idx}")
        path = self.paths[idx]
        if path not in self.cache:
            self.cache[path] = torch.load(self.root / path, map_location="cpu", weights_only=False)["embeddings"].float()
        self.cache.move_to_end(path)
        if len(self.cache) > self.max_cache:
            self.cache.popitem(last=False)
        return self.cache[path]

    def batch(self, indices, device):
        # The protein-balanced sampler repeats each protein for several drugs.
        # Encode each unique residue sequence once, then map pair queries back to it.
        unique, inverse = torch.unique(indices.detach().cpu(), sorted=True, return_inverse=True)
        values = []
        for idx in unique.tolist():
            x = self._get(idx)
            if x.shape[0] > TOKEN_MAX_TOKENS:
                # Deterministic centre crop keeps validation/test reproducible.
                start = (x.shape[0] - TOKEN_MAX_TOKENS) // 2
                x = x[start:start + TOKEN_MAX_TOKENS]
            values.append(x)
        length = max(x.shape[0] for x in values)
        out = torch.zeros(len(values), length, values[0].shape[1], dtype=torch.float32)
        mask = torch.zeros(len(values), length, dtype=torch.bool)
        for i, x in enumerate(values):
            out[i, :x.shape[0]] = x
            mask[i, :x.shape[0]] = True
        return (
            out.to(device, non_blocking=True),
            mask.to(device, non_blocking=True),
            inverse.to(device, non_blocking=True),
        )


class ViroBindV5Decoupled(nn.Module):
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
        if feature_preset != "ecfp_esmc":
            raise ValueError("V5 expects feature_preset=ecfp_esmc with drug_combo as ecfp_path.")
        self.feature_preset = feature_preset
        self.use_domain_adapter = use_domain_adapter
        self.rank_train_scope = RANK_TRAIN_SCOPE
        self.drug_fusion_mode = DRUG_FUSION_MODE
        self.protein_fusion_mode = PROTEIN_FUSION_MODE
        self.pair_feature_mode = PAIR_FEATURE_MODE
        if DRUG_FUSION_MODE == "tri_gate":
            self.drug_encoder = DrugTriViewGateEncoder(
                ecfp_dim,
                DRUG_ECFP_DIM,
                DRUG_MACCS_DIM,
                DRUG_RDKIT_DIM,
                hidden_dim,
                dropout,
            )
        elif DRUG_FUSION_MODE == "residual_multiview":
            self.drug_encoder = DrugBaselineResidualMultiViewEncoder(
                ecfp_dim,
                DRUG_ECFP_DIM,
                DRUG_MACCS_DIM,
                DRUG_RDKIT_DIM,
                hidden_dim,
                dropout,
                DRUG_AUX_GATE_INIT,
            )
        elif DRUG_FUSION_MODE == "independent_residual":
            self.drug_encoder = DrugIndependentResidualEncoder(
                ecfp_dim,
                DRUG_ECFP_DIM,
                DRUG_MACCS_DIM,
                DRUG_RDKIT_DIM,
                hidden_dim,
                dropout,
                DRUG_AUX_GATE_INIT,
                DRUG_AUX_GATE_MAX,
            )
        else:
            self.drug_encoder = base.MLPEncoder(ecfp_dim, hidden_dim, dropout)

        if PROTEIN_FUSION_MODE == "residual_gate":
            self.prot_encoder = ProteinDualViewResidualEncoder(
                prot_dim,
                PROTEIN_ESMC_DIM,
                PROTEIN_DESC_DIM,
                hidden_dim,
                dropout,
                PROTEIN_DESC_GATE_INIT,
            )
        else:
            self.prot_encoder = ProteinEsmcOnlyEncoder(prot_dim, PROTEIN_ESMC_DIM, hidden_dim, dropout)
        self.use_residue_tokens = bool(TOKEN_ROOT)
        self.residue_rank_only = bool(RESIDUE_RANK_ONLY)
        if self.use_residue_tokens:
            # Drug-conditioned single-query attention over frozen residue tokens.
            # The local summary is added as a small residual to the existing V5H
            # global protein representation; zero initial influence is avoided but
            # the gate remains bounded during early training.
            self.token_encoder = nn.Sequential(
                nn.Linear(PROTEIN_ESMC_DIM, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
            )
            self.token_key = nn.Linear(hidden_dim, hidden_dim, bias=False)
            self.token_value = nn.Linear(hidden_dim, hidden_dim, bias=False)
            self.token_query = nn.Linear(hidden_dim, hidden_dim, bias=False)
            self.token_gate_logit = nn.Parameter(torch.tensor(float(TOKEN_GATE_INIT)))
        if use_domain_adapter:
            bottleneck = max(32, hidden_dim // 4)
            self.human_adapter = base.DomainAdapter(hidden_dim, bottleneck, dropout)
            self.virus_adapter = base.DomainAdapter(hidden_dim, bottleneck, dropout)

        self.drug_bilinear = nn.Linear(hidden_dim, bilinear_dim, bias=False)
        self.prot_bilinear = nn.Linear(hidden_dim, bilinear_dim, bias=False)
        pair_dim = self._pair_dim(hidden_dim, bilinear_dim)
        self.pair_norm = nn.LayerNorm(pair_dim)
        self.rank_adapter = nn.Sequential(
            nn.LayerNorm(pair_dim),
            nn.Linear(pair_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, pair_dim),
        )
        self.cls_head = self._make_head(pair_dim, hidden_dim, dropout)
        self.rank_head = self._make_head(pair_dim, hidden_dim, dropout)

    @staticmethod
    def _make_head(pair_dim: int, hidden_dim: int, dropout: float):
        return nn.Sequential(
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

    @staticmethod
    def _pair_dim(hidden_dim: int, bilinear_dim: int):
        if PAIR_FEATURE_MODE == "concat_only":
            return hidden_dim * 2
        if PAIR_FEATURE_MODE == "concat_product":
            return hidden_dim * 3
        if PAIR_FEATURE_MODE == "concat_product_absdiff":
            return hidden_dim * 4
        if PAIR_FEATURE_MODE == "full_bilinear":
            return hidden_dim * 4 + bilinear_dim
        raise ValueError(f"Unsupported pair_feature_mode: {PAIR_FEATURE_MODE}")

    def _pair_from_embeddings(self, h_drug, h_prot):
        parts = [h_drug, h_prot]
        if self.pair_feature_mode in ["concat_product", "concat_product_absdiff", "full_bilinear"]:
            parts.append(h_drug * h_prot)
        if self.pair_feature_mode in ["concat_product_absdiff", "full_bilinear"]:
            parts.append(torch.abs(h_drug - h_prot))
        if self.pair_feature_mode == "full_bilinear":
            parts.append(self.drug_bilinear(h_drug) * self.prot_bilinear(h_prot))
        return self.pair_norm(torch.cat(parts, dim=-1))

    def encode_pair(self, x_ecfp, x_unimol, x_prot, domain, tokens=None, token_mask=None, token_pair_index=None):
        x_ecfp = x_ecfp * x_ecfp.new_tensor(DRUG_VIEW_MASK_VECTOR).view(1, -1)
        if self.drug_fusion_mode == "tri_gate":
            h_drug, drug_weights = self.drug_encoder(x_ecfp)
            drug_aux_gate = x_ecfp.new_tensor(1.0)
        elif self.drug_fusion_mode in ["residual_multiview", "independent_residual"]:
            h_drug, drug_weights, drug_aux_gate = self.drug_encoder(x_ecfp)
        else:
            h_drug = self.drug_encoder(x_ecfp)
            drug_weights = x_ecfp.new_full((x_ecfp.shape[0], 3), float("nan"))
            drug_aux_gate = x_ecfp.new_tensor(0.0)
        h_prot = self.prot_encoder(x_prot)
        if self.use_domain_adapter:
            h_human = self.human_adapter(h_prot)
            h_virus = self.virus_adapter(h_prot)
            mask = domain.float().view(-1, 1)
            h_prot = h_human * (1.0 - mask) + h_virus * mask
        h_prot_global = h_prot
        if self.use_residue_tokens:
            if tokens is None or token_mask is None or token_pair_index is None:
                raise ValueError("Residue-token V5H requires tokens, token_mask and token_pair_index")
            h_token = self.token_encoder(tokens)
            q = self.token_query(h_drug)
            h_local = torch.empty_like(h_drug)
            # Only 16 unique proteins per normal V5H batch, versus 128 pairs.
            # This retains drug-specific attention without repeating token MLP work.
            for token_i in range(h_token.shape[0]):
                pair_i = torch.nonzero(token_pair_index == token_i, as_tuple=False).squeeze(1)
                key = self.token_key(h_token[token_i])
                score = q[pair_i] @ key.transpose(0, 1) / math.sqrt(h_token.shape[-1])
                score = score.masked_fill(~token_mask[token_i].unsqueeze(0), -torch.inf)
                attention = torch.softmax(score, dim=1)
                h_local[pair_i] = attention @ self.token_value(h_token[token_i])
            token_gate = TOKEN_GATE_MAX * torch.sigmoid(self.token_gate_logit)
            h_prot = h_prot + token_gate * h_local
        # In rank-only mode, classification always sees the original global V5H
        # representation. Residue evidence is exposed exclusively to the rank head.
        pair_cls = self._pair_from_embeddings(h_drug, h_prot_global)
        pair_rank = self._pair_from_embeddings(h_drug, h_prot)
        if not self.residue_rank_only:
            pair_cls = pair_rank
        return pair_cls, pair_rank, drug_weights, drug_aux_gate

    def forward(self, x_ecfp, x_unimol, x_prot, domain, tokens=None, token_mask=None, token_pair_index=None):
        pair, rank_pair, drug_weights, drug_aux_gate = self.encode_pair(x_ecfp, x_unimol, x_prot, domain, tokens, token_mask, token_pair_index)
        if self.rank_train_scope in ["rank_head_adapter", "rank_head_pair"]:
            rank_pair = pair + self.rank_adapter(pair)
        return {
            "cls": self.cls_head(pair).squeeze(-1),
            "rank": self.rank_head(rank_pair).squeeze(-1),
            "drug_weights": drug_weights,
            "drug_aux_gate": drug_aux_gate,
            "protein_desc_gate": (
                (
                    pair.new_tensor(float(PROTEIN_DESC_GATE_VALUE))
                    if PROTEIN_DESC_GATE_MODE == "fixed"
                    else PROTEIN_DESC_GATE_MAX * torch.sigmoid(self.prot_encoder.desc_gate_logit)
                )
                if hasattr(self.prot_encoder, "desc_gate_logit")
                else pair.new_tensor(float("nan"))
            ),
            "token_gate": (
                TOKEN_GATE_MAX * torch.sigmoid(self.token_gate_logit)
                if self.use_residue_tokens else pair.new_tensor(float("nan"))
            ),
        }


def batch_forward(model, batch, features, feature_preset, device):
    prot_idx_cpu = batch["prot_idx"]
    ecfp_idx_cpu = batch["ecfp_idx"]
    domain = batch["domain"].to(device, non_blocking=True)
    tokens = token_mask = token_pair_index = None
    if TOKEN_STORE is not None:
        tokens, token_mask, token_pair_index = TOKEN_STORE.batch(prot_idx_cpu, device)
    x_ecfp = features["ecfp"][ecfp_idx_cpu].to(device, non_blocking=True)
    x_prot = features["prot"][prot_idx_cpu].to(device, non_blocking=True)
    return model(x_ecfp, None, x_prot, domain, tokens, token_mask, token_pair_index)


def cls_bce_loss(cls_logits, y, pos_weight, args):
    target = torch.where(y == -1, torch.full_like(y, float(args.gray_target)), y)
    gray_weight = args.gray_cls_weight if args.gray_cls_weight is not None else args.gray_bce_weight
    weight = torch.where(y == -1, torch.full_like(y, float(gray_weight)), torch.ones_like(y))
    pw = torch.tensor([float(pos_weight)], dtype=torch.float32, device=cls_logits.device)
    loss = F.binary_cross_entropy_with_logits(cls_logits, target, pos_weight=pw, reduction="none")
    return (loss * weight).sum() / weight.sum().clamp_min(1.0)


def cls_focal_loss(cls_logits, y, pos_weight, args):
    target = torch.where(y == -1, torch.full_like(y, float(args.gray_target)), y)
    gray_weight = args.gray_cls_weight if args.gray_cls_weight is not None else args.gray_bce_weight
    sample_weight = torch.where(y == -1, torch.full_like(y, float(gray_weight)), torch.ones_like(y))
    pw = torch.tensor([float(pos_weight)], dtype=torch.float32, device=cls_logits.device)
    bce = F.binary_cross_entropy_with_logits(cls_logits, target, pos_weight=pw, reduction="none")
    prob = torch.sigmoid(cls_logits)
    pt = prob * target + (1.0 - prob) * (1.0 - target)
    alpha = float(args.focal_alpha)
    alpha_t = alpha * target + (1.0 - alpha) * (1.0 - target)
    focal = alpha_t * torch.pow((1.0 - pt).clamp_min(1e-6), float(args.focal_gamma)) * bce
    return (focal * sample_weight).sum() / sample_weight.sum().clamp_min(1.0)


def cls_loss(cls_logits, y, pos_weight, args):
    if args.cls_loss_mode == "focal":
        return cls_focal_loss(cls_logits, y, pos_weight, args)
    return cls_bce_loss(cls_logits, y, pos_weight, args)


def margin_rank_loss(logits, y, groups, margin, max_pairs_per_group):
    losses = []
    for g in torch.unique(groups):
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
        losses.append(F.relu(float(margin) - diff).mean())
    return torch.stack(losses).mean() if losses else logits.new_tensor(0.0)


def rank_loss(rank_logits, y, groups, args):
    if args.rank_loss_mode == "bce":
        return F.binary_cross_entropy_with_logits(rank_logits, y.float())
    if args.rank_loss_mode == "margin":
        return margin_rank_loss(rank_logits, y, groups, args.rank_margin, args.max_rank_pairs_per_group)
    return base.bpr_rank_loss(rank_logits, y, groups, max_pairs_per_group=args.max_rank_pairs_per_group)


def set_rank_trainable(model, scope):
    for p in model.parameters():
        p.requires_grad = False
    for p in model.rank_head.parameters():
        p.requires_grad = True
    if getattr(model, "residue_rank_only", False):
        for module in [model.token_encoder, model.token_key, model.token_value, model.token_query]:
            for p in module.parameters():
                p.requires_grad = True
        model.token_gate_logit.requires_grad = True
    if scope == "rank_head_adapter":
        for p in model.rank_adapter.parameters():
            p.requires_grad = True
    if scope == "rank_head_pair":
        for p in model.rank_adapter.parameters():
            p.requires_grad = True
        for p in model.pair_norm.parameters():
            p.requires_grad = True
        for p in model.drug_bilinear.parameters():
            p.requires_grad = True
        for p in model.prot_bilinear.parameters():
            p.requires_grad = True


def train_one_epoch(model, loader, features, device, args, optimizer, pos_weight, task):
    if task == "rank":
        model.eval()
        model.rank_head.train()
        if args.rank_train_scope in ["rank_head_adapter", "rank_head_pair"]:
            model.rank_adapter.train()
        if args.rank_train_scope == "rank_head_pair":
            model.pair_norm.train()
    else:
        model.train()
    total = {"loss": 0.0, "cls_bce": 0.0, "rank": 0.0}
    total_n = 0
    for batch in loader:
        y = batch["y"].to(device, non_blocking=True)
        groups = batch["group"].to(device, non_blocking=True)
        out = batch_forward(model, batch, features, args.feature_preset, device)
        if task == "cls":
            loss_cls = cls_loss(out["cls"], y, pos_weight, args)
            loss_rank = out["cls"].new_tensor(0.0)
            loss = loss_cls
        elif task == "rank":
            loss_cls = out["rank"].new_tensor(0.0)
            loss_rank = rank_loss(out["rank"], y, groups, args)
            loss = loss_rank
        else:
            raise ValueError(task)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], args.grad_clip)
        optimizer.step()
        bs = y.numel()
        total_n += bs
        total["loss"] += float(loss.item()) * bs
        total["cls_bce"] += float(loss_cls.item()) * bs
        total["rank"] += float(loss_rank.item()) * bs
    out_log = {k: v / max(total_n, 1) for k, v in total.items()}
    out_log["bce"] = out_log["cls_bce"]
    return out_log


@torch.no_grad()
def predict(model, dataset, loader, features, device, feature_preset):
    model.eval()
    all_y, all_idx, all_cls, all_rank, all_drug_weights = [], [], [], [], []
    drug_aux_gate = None
    protein_desc_gate = None
    for batch in loader:
        out = batch_forward(model, batch, features, feature_preset, device)
        all_y.append(batch["y"].cpu().numpy())
        all_idx.append(batch["idx"].cpu().numpy())
        all_cls.append(out["cls"].detach().cpu().numpy())
        all_rank.append(out["rank"].detach().cpu().numpy())
        all_drug_weights.append(out["drug_weights"].detach().cpu().numpy())
        drug_aux_gate = float(out["drug_aux_gate"].detach().cpu().item())
        protein_desc_gate = float(out["protein_desc_gate"].detach().cpu().item())
    y = np.concatenate(all_y)
    idx = np.concatenate(all_idx)
    cls_logit = np.concatenate(all_cls)
    rank_logit = np.concatenate(all_rank)
    drug_weights = np.concatenate(all_drug_weights)
    proteins = [dataset.group_names_per_item[int(i)] for i in idx]
    drugs = [dataset.drug_names_per_item[int(i)] for i in idx]
    return pd.DataFrame(
        {
            "drug": drugs,
            "protein": proteins,
            "label": y.astype(int),
            "logit": cls_logit,
            "prob": 1.0 / (1.0 + np.exp(-cls_logit)),
            "cls_logit": cls_logit,
            "cls_prob": 1.0 / (1.0 + np.exp(-cls_logit)),
            "rank_logit": rank_logit,
            "rank_prob": 1.0 / (1.0 + np.exp(-rank_logit)),
            "drug_weight_ecfp": drug_weights[:, 0],
            "drug_weight_maccs": drug_weights[:, 1],
            "drug_weight_rdkit2d": drug_weights[:, 2],
            "drug_aux_gate": drug_aux_gate,
            "protein_desc_gate": protein_desc_gate,
        }
    )


def select_cls_threshold(y, prob):
    metric = CLS_THRESHOLD_METRIC
    if metric == "fixed":
        return float(FIXED_CLS_THRESHOLD)

    y = np.asarray(y).astype(int)
    prob = np.asarray(prob).astype(float)
    lo = max(0.0, min(1.0, float(MIN_CLS_THRESHOLD)))
    n_steps = max(2, int(round((1.0 - lo) * 500)) + 1)
    best_t = float(lo)
    best_score = -1e18
    for t in np.linspace(lo, 1.0, n_steps):
        pred = (prob >= t).astype(int)
        if metric == "balanced_acc":
            score = balanced_accuracy_score(y, pred)
        elif metric == "youden":
            tn = np.sum((y == 0) & (pred == 0))
            fp = np.sum((y == 0) & (pred == 1))
            fn = np.sum((y == 1) & (pred == 0))
            tp = np.sum((y == 1) & (pred == 1))
            score = (tp / max(tp + fn, 1)) - (fp / max(fp + tn, 1))
        elif metric == "mcc":
            score = matthews_corrcoef(y, pred)
        else:
            score = f1_score(y, pred, zero_division=0)
        if score > best_score:
            best_score = float(score)
            best_t = float(t)
    return best_t


def compute_metrics(pred_df, threshold=None):
    cls_df = pred_df[["drug", "protein", "label"]].copy()
    cls_df["logit"] = pred_df["cls_logit"].values
    cls_df["prob"] = pred_df["cls_prob"].values
    if threshold is None:
        threshold = select_cls_threshold(cls_df["label"].values, cls_df["prob"].values)
    cls_metrics = ORIG_COMPUTE_METRICS(cls_df, threshold=threshold)

    rank_df = pred_df[["drug", "protein", "label"]].copy()
    rank_df["logit"] = pred_df["rank_logit"].values
    rank_df["prob"] = pred_df["rank_prob"].values
    rank_metrics = ORIG_COMPUTE_METRICS(rank_df, threshold=None)

    metrics = {}
    for k, v in cls_metrics.items():
        metrics[f"cls_{k}"] = v
    for k, v in rank_metrics.items():
        metrics[f"rank_{k}"] = v

    # Top-level metrics intentionally map to the user's two-task reporting contract:
    # classification head for global AUC/AUPR, ranking head for per-protein ranking.
    for k in ["auc", "aupr", "f1", "acc", "balanced_acc", "precision", "recall", "threshold", "pp_mean", "n", "pos", "neg"]:
        metrics[k] = cls_metrics.get(k)
    for k in ["perprotein_auc_mean", "perprotein_aupr_mean", "p_at_10_mean", "p_at_50_mean", "hit_at_10_mean", "hit_at_50_mean", "n_proteins", "n_proteins_with_both_labels"]:
        metrics[k] = rank_metrics.get(k)
    metrics["combo_cls_rank"] = 0.5 * metrics["cls_aupr"] + 0.5 * metrics["rank_perprotein_aupr_mean"]
    metrics["combo_cls_f1_rank"] = 0.5 * metrics["cls_f1"] + 0.5 * metrics["rank_perprotein_aupr_mean"]
    metrics["combo_aupr_ppaupr"] = metrics["combo_cls_rank"]
    metrics["cls_threshold_metric"] = CLS_THRESHOLD_METRIC
    for column in [
        "drug_weight_ecfp",
        "drug_weight_maccs",
        "drug_weight_rdkit2d",
        "drug_aux_gate",
        "protein_desc_gate",
    ]:
        values = pd.to_numeric(pred_df[column], errors="coerce").values.astype(float)
        finite = values[np.isfinite(values)]
        metrics[f"fusion_{column}_mean"] = float(np.mean(finite)) if finite.size else float("nan")
    return metrics


def monitor_value(metrics, key):
    if key in ["cls_aupr", "cls_f1", "cls_balanced_acc", "rank_perprotein_aupr_mean", "combo_cls_rank", "combo_cls_f1_rank"]:
        key = key
    v = metrics.get(key, float("nan"))
    if v is None or math.isnan(float(v)):
        return -1e18
    return float(v)


def make_stage_loader(dataset, args, stage_name, train):
    original = getattr(args, "gray_per_protein", 0)
    stage_gray = original
    if stage_name == "pretrain" and args.pretrain_gray_per_protein is not None:
        stage_gray = args.pretrain_gray_per_protein
    elif stage_name == "cls_train" and args.cls_train_gray_per_protein is not None:
        stage_gray = args.cls_train_gray_per_protein
    elif stage_name == "rank_train" and args.rank_train_gray_per_protein is not None:
        stage_gray = args.rank_train_gray_per_protein
    args.gray_per_protein = stage_gray
    print(f"[STAGE] gray_per_protein={stage_gray}")
    loader = base.make_loader(dataset, args, train=train)
    args.gray_per_protein = original
    return loader


def train_stage(stage_name, task, model, train_ds, val_ds, features, device, args, out_dir, lr, epochs, monitor):
    print(f"\n[STAGE] {stage_name} task={task}")
    print(f"[STAGE] lr={lr} epochs={epochs} monitor={monitor}")
    train_loader = make_stage_loader(train_ds, args, stage_name, train=True)
    val_loader = base.make_loader(val_ds, args, train=False)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=args.weight_decay)
    pos_weight = train_ds.pos_weight()
    monitors = ["cls_aupr", "cls_f1", "cls_balanced_acc", "rank_perprotein_aupr_mean", "combo_cls_rank", "combo_cls_f1_rank", "combo_aupr_ppaupr"]
    best = {
        m: {"score": -1e18, "epoch": -1, "path": os.path.join(out_dir, f"{stage_name}_best_by_{m}.pt"), "metrics": None}
        for m in monitors
    }
    history = []
    bad_epochs = 0
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_log = train_one_epoch(model, train_loader, features, device, args, optimizer, pos_weight, task)
        val_pred = predict(model, val_ds, val_loader, features, device, args.feature_preset)
        val_metrics = compute_metrics(val_pred, threshold=None)
        row = {
            "stage": stage_name,
            "task": task,
            "epoch": epoch,
            "time_sec": time.time() - t0,
            **{f"train_{k}": v for k, v in train_log.items()},
            **{f"val_{k}": v for k, v in val_metrics.items()},
        }
        history.append(row)
        print(
            f"[{stage_name}] epoch={epoch:03d} loss={train_log['loss']:.6f} "
            f"cls_bce={train_log['cls_bce']:.6f} rank={train_log['rank']:.6f} "
            f"val_cls_auc={val_metrics['cls_auc']:.6f} val_cls_aupr={val_metrics['cls_aupr']:.6f} "
            f"val_rank_pp_aupr={val_metrics['rank_perprotein_aupr_mean']:.6f} "
            f"val_rank_p10={val_metrics['rank_p_at_10_mean']:.6f} "
            f"val_combo={val_metrics['combo_cls_rank']:.6f}"
        )
        improved_main = False
        for m in monitors:
            score = monitor_value(val_metrics, m)
            if score > best[m]["score"]:
                best[m].update({"score": score, "epoch": epoch, "metrics": val_metrics})
                torch.save({"model": model.state_dict(), "stage": stage_name, "task": task, "epoch": epoch, "monitor": m, "val_metrics": val_metrics, "args": vars(args)}, best[m]["path"])
                val_pred.to_csv(os.path.join(out_dir, f"{stage_name}_best_by_{m}_val_predictions.csv"), index=False)
                if m == monitor:
                    improved_main = True
        pd.DataFrame(history).to_csv(os.path.join(out_dir, f"{stage_name}_history.csv"), index=False)
        bad_epochs = 0 if improved_main else bad_epochs + 1
        if bad_epochs >= args.patience:
            print(f"[EARLY_STOP] {stage_name}: best_{monitor}_epoch={best[monitor]['epoch']} best_{monitor}={best[monitor]['score']:.6f}")
            break
    ckpt = torch.load(best[monitor]["path"], map_location=device)
    model.load_state_dict(ckpt["model"])
    return best


def parse_args_with_v5_fields():
    args = ORIG_PARSE_ARGS()
    args.v5_monitor = V5_MONITOR
    args.rank_loss_mode = RANK_LOSS_MODE
    args.rank_margin = RANK_MARGIN
    args.cls_loss_mode = CLS_LOSS_MODE
    args.focal_gamma = FOCAL_GAMMA
    args.focal_alpha = FOCAL_ALPHA
    args.cls_threshold_metric = CLS_THRESHOLD_METRIC
    args.min_cls_threshold = MIN_CLS_THRESHOLD
    args.fixed_cls_threshold = FIXED_CLS_THRESHOLD
    args.gray_cls_weight = GRAY_CLS_WEIGHT
    args.pretrain_gray_per_protein = PRETRAIN_GRAY_PER_PROTEIN
    args.cls_train_gray_per_protein = CLS_TRAIN_GRAY_PER_PROTEIN
    args.rank_train_gray_per_protein = RANK_TRAIN_GRAY_PER_PROTEIN
    args.rank_train_csv = RANK_TRAIN_CSV
    args.rank_train_epochs = RANK_TRAIN_EPOCHS if RANK_TRAIN_EPOCHS is not None else args.train_epochs
    args.rank_train_lr = RANK_TRAIN_LR if RANK_TRAIN_LR is not None else args.train_lr
    args.rank_train_scope = RANK_TRAIN_SCOPE
    args.cls_checkpoint_policy = CLS_CHECKPOINT_POLICY
    args.drug_fusion_mode = DRUG_FUSION_MODE
    args.drug_view_mask = DRUG_VIEW_MASK
    args.drug_aux_gate_init = DRUG_AUX_GATE_INIT
    args.drug_aux_gate_max = DRUG_AUX_GATE_MAX
    args.protein_fusion_mode = PROTEIN_FUSION_MODE
    args.protein_desc_input_scale = PROTEIN_DESC_INPUT_SCALE
    args.protein_desc_gate_max = PROTEIN_DESC_GATE_MAX
    args.protein_desc_gate_mode = PROTEIN_DESC_GATE_MODE
    args.protein_desc_gate_value = PROTEIN_DESC_GATE_VALUE
    args.token_root = TOKEN_ROOT
    args.token_max_tokens = TOKEN_MAX_TOKENS
    args.token_gate_init = TOKEN_GATE_INIT
    args.token_gate_max = TOKEN_GATE_MAX
    args.residue_rank_only = RESIDUE_RANK_ONLY
    args.init_checkpoint = INIT_CHECKPOINT
    args.pair_feature_mode = PAIR_FEATURE_MODE
    return args


def main():
    args = parse_args_with_v5_fields()
    base.set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    suffix = "__residue_token" if args.token_root else ""
    exp_name = f"{args.protocol}__v5_decoupled_multiview__{args.sampler}__adapter{args.use_domain_adapter}{suffix}"
    out_dir = os.path.join(args.out_root, args.mode, exp_name, f"seed_{args.seed}")
    base.ensure_dir(out_dir)
    print(f"[INFO] device={device}")
    print(f"[INFO] out_dir={out_dir}")

    csvs = base.resolve_csvs(args)
    if args.rank_train_csv:
        csvs["rank_train_csv"] = args.rank_train_csv
    else:
        csvs["rank_train_csv"] = csvs["train_csv"]
    print("[INFO] Resolved CSVs:")
    for k, v in csvs.items():
        print(f"  {k}: {v}")

    ecfp = base.FeatureBank(args.ecfp_path, "drug_ecfp")
    prot = base.FeatureBank(args.prot_path, "prot_esmc_global")
    global TOKEN_STORE
    TOKEN_STORE = ResidueTokenStore(args.token_root, prot) if args.token_root else None
    if TOKEN_STORE is not None:
        print(f"[TOKEN] root={args.token_root} max_tokens={args.token_max_tokens} mapped_proteins={len(TOKEN_STORE.paths)}")
    pretrain_ds = None
    if args.protocol == "allhuman_pt":
        pretrain_ds = base.ViroBindPairDataset(csvs["pretrain_csv"], ecfp, None, prot, args.feature_preset, f"pretrain_allhuman_{args.pretrain_label_mode}", args.pretrain_label_mode == "binary", args.pretrain_label_mode)
    cls_train_ds = base.ViroBindPairDataset(csvs["train_csv"], ecfp, None, prot, args.feature_preset, f"cls_train_virus_{args.train_label_mode}", args.train_label_mode == "binary", args.train_label_mode)
    rank_train_ds = base.ViroBindPairDataset(csvs["rank_train_csv"], ecfp, None, prot, args.feature_preset, "rank_train_virus_binary", True, "binary")
    val_ds = base.ViroBindPairDataset(csvs["val_csv"], ecfp, None, prot, args.feature_preset, "val_virusonly_labeled_0_1", True, "binary")
    test_ds = base.ViroBindPairDataset(csvs["test_csv"], ecfp, None, prot, args.feature_preset, "test_virusonly_labeled_0_1", True, "binary")

    config = {"args": vars(args), "csvs": csvs, "exp_name": exp_name, "out_dir": out_dir}
    base.json_dump(config, os.path.join(out_dir, "config.json"))
    if args.dry_run:
        print("[DRY_RUN] finished dataset and feature checking.")
        return

    model = ViroBindV5Decoupled(args.feature_preset, ecfp.dim, 1, prot.dim, args.hidden_dim, args.bilinear_dim, args.dropout, bool(args.use_domain_adapter)).to(device)
    if args.residue_rank_only:
        if not args.token_root:
            raise ValueError("--residue_rank_only requires --token_root")
        if not args.init_checkpoint:
            raise ValueError("--residue_rank_only requires --init_checkpoint from the matching baseline V5H run")
        state = torch.load(args.init_checkpoint, map_location=device, weights_only=False)
        missing, unexpected = model.load_state_dict(state["model"], strict=False)
        allowed_missing = {"token_encoder.0.weight", "token_encoder.0.bias", "token_encoder.1.weight", "token_encoder.1.bias", "token_key.weight", "token_value.weight", "token_query.weight", "token_gate_logit"}
        if set(missing) != allowed_missing or unexpected:
            raise RuntimeError(f"Checkpoint/model mismatch; missing={missing}, unexpected={unexpected}")
        print(f"[RANK_TOKEN_ONLY] initialized baseline weights from {args.init_checkpoint}")
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[MODEL] trainable_params={n_params:,}")
    # Keep full feature banks on CPU.  The combined drug fingerprint matrix is
    # too large to pin on smaller GPUs; batch_forward moves only indexed slices.
    features = {"ecfp": ecfp.x, "prot": prot.x}

    final_summary = {
        "protocol": args.protocol,
        "mode": args.mode,
        "seed": args.seed,
        "exp_name": exp_name,
        "feature_preset": args.feature_preset,
        "n_params": n_params,
        "v5_monitor": args.v5_monitor,
    }

    if args.residue_rank_only:
        final_summary["init_checkpoint"] = args.init_checkpoint
        final_summary["training_mode"] = "frozen_global_v5h_rank_only_residue_adapter"
        set_rank_trainable(model, args.rank_train_scope)
        rank_best = train_stage("rank_token_only", "rank", model, rank_train_ds, val_ds, features, device, args, out_dir, args.rank_train_lr, args.rank_train_epochs, args.v5_monitor)
        final_summary["rank_train_best_monitor_epoch"] = rank_best[args.v5_monitor]["epoch"]
        val_loader = base.make_loader(val_ds, args, train=False)
        test_loader = base.make_loader(test_ds, args, train=False)
        for monitor, obj in rank_best.items():
            model.load_state_dict(torch.load(obj["path"], map_location=device, weights_only=False)["model"])
            val_pred = predict(model, val_ds, val_loader, features, device, args.feature_preset)
            val_metrics = compute_metrics(val_pred, threshold=None)
            test_pred = predict(model, test_ds, test_loader, features, device, args.feature_preset)
            test_metrics = compute_metrics(test_pred, threshold=val_metrics["cls_threshold"])
            val_pred.to_csv(os.path.join(out_dir, f"final_best_by_{monitor}_val_predictions.csv"), index=False)
            test_pred.to_csv(os.path.join(out_dir, f"final_best_by_{monitor}_test_predictions.csv"), index=False)
            out_monitor = "combo_aupr_ppaupr" if monitor == "combo_cls_rank" else monitor
            for k, v in val_metrics.items(): final_summary[f"best_by_{out_monitor}_val_{k}"] = v
            for k, v in test_metrics.items(): final_summary[f"best_by_{out_monitor}_test_{k}"] = v
            final_summary[f"best_by_{out_monitor}_epoch"] = obj["epoch"]
            final_summary[f"best_by_{out_monitor}_ckpt"] = obj["path"]
        base.json_dump(final_summary, os.path.join(out_dir, "metrics.json"))
        pd.DataFrame([final_summary]).to_csv(os.path.join(out_dir, "metrics.csv"), index=False)
        print("[DONE] rank-only residue adapter")
        return

    pretrain_best = None
    if args.protocol == "allhuman_pt":
        for p in model.parameters():
            p.requires_grad = True
        pretrain_best = train_stage("pretrain", "cls", model, pretrain_ds, val_ds, features, device, args, out_dir, args.pretrain_lr, args.pretrain_epochs, "cls_aupr")
        final_summary["pretrain_best_cls_aupr_epoch"] = pretrain_best["cls_aupr"]["epoch"]

    for p in model.parameters():
        p.requires_grad = True
    cls_best = train_stage("cls_train", "cls", model, cls_train_ds, val_ds, features, device, args, out_dir, args.train_lr, args.train_epochs, "cls_aupr")
    final_summary["cls_train_best_cls_aupr_epoch"] = cls_best["cls_aupr"]["epoch"]
    selected_cls = cls_best["cls_aupr"]
    selected_cls_stage = "cls_train"
    if args.cls_checkpoint_policy == "best_stage" and pretrain_best is not None:
        if pretrain_best["cls_aupr"]["score"] > selected_cls["score"]:
            selected_cls = pretrain_best["cls_aupr"]
            selected_cls_stage = "pretrain"
    final_summary["selected_cls_stage"] = selected_cls_stage
    final_summary["selected_cls_val_cls_aupr"] = selected_cls["score"]
    final_summary["selected_cls_ckpt"] = selected_cls["path"]
    print(
        f"[CLS_SELECT] policy={args.cls_checkpoint_policy} "
        f"stage={selected_cls_stage} val_cls_aupr={selected_cls['score']:.6f}"
    )
    cls_ckpt = torch.load(selected_cls["path"], map_location=device)
    model.load_state_dict(cls_ckpt["model"])

    set_rank_trainable(model, args.rank_train_scope)
    rank_best = train_stage("rank_train", "rank", model, rank_train_ds, val_ds, features, device, args, out_dir, args.rank_train_lr, args.rank_train_epochs, args.v5_monitor)
    final_summary["rank_train_best_monitor_epoch"] = rank_best[args.v5_monitor]["epoch"]

    val_loader = base.make_loader(val_ds, args, train=False)
    test_loader = base.make_loader(test_ds, args, train=False)
    for monitor, obj in rank_best.items():
        ckpt = torch.load(obj["path"], map_location=device)
        model.load_state_dict(ckpt["model"])
        val_pred = predict(model, val_ds, val_loader, features, device, args.feature_preset)
        val_metrics = compute_metrics(val_pred, threshold=None)
        threshold = val_metrics["cls_threshold"]
        test_pred = predict(model, test_ds, test_loader, features, device, args.feature_preset)
        test_metrics = compute_metrics(test_pred, threshold=threshold)
        val_pred.to_csv(os.path.join(out_dir, f"final_best_by_{monitor}_val_predictions.csv"), index=False)
        test_pred.to_csv(os.path.join(out_dir, f"final_best_by_{monitor}_test_predictions.csv"), index=False)
        out_monitor = "combo_aupr_ppaupr" if monitor == "combo_cls_rank" else monitor
        for k, v in val_metrics.items():
            final_summary[f"best_by_{out_monitor}_val_{k}"] = v
        for k, v in test_metrics.items():
            final_summary[f"best_by_{out_monitor}_test_{k}"] = v
        final_summary[f"best_by_{out_monitor}_epoch"] = obj["epoch"]
        final_summary[f"best_by_{out_monitor}_ckpt"] = obj["path"]

    base.json_dump(final_summary, os.path.join(out_dir, "metrics.json"))
    pd.DataFrame([final_summary]).to_csv(os.path.join(out_dir, "metrics.csv"), index=False)
    print("[FINAL SUMMARY]")
    for k in [
        "best_by_combo_aupr_ppaupr_val_cls_aupr",
        "best_by_combo_aupr_ppaupr_val_rank_perprotein_aupr_mean",
        "best_by_combo_aupr_ppaupr_test_cls_aupr",
        "best_by_combo_aupr_ppaupr_test_rank_perprotein_aupr_mean",
        "best_by_combo_aupr_ppaupr_test_rank_p_at_10_mean",
    ]:
        print(f"{k}: {final_summary.get(k)}")


def cli_main(argv=None):
    """Console entry point for final ViroBind training."""
    argv = list(sys.argv[1:] if argv is None else argv)
    v5_args, remaining = parse_v5_args(argv)
    configure_from_namespace(v5_args)
    base.ProteinBalancedBatchSampler = V5TemperatureProteinBalancedBatchSampler
    old_argv = sys.argv
    try:
        sys.argv = [old_argv[0]] + remaining
        main()
    finally:
        sys.argv = old_argv


if __name__ == "__main__":
    cli_main()
