"""Unique compound-protein records for the manuscript training protocol."""
from __future__ import annotations

import pandas as pd


def clean_pair_records(frame):
    """Deduplicate pairs and exclude all evidence for conflicting hard labels.

    Gray/hard mixtures without conflicting hard labels must be resolved from
    upstream assay evidence; this function does not guess their final label.
    """
    required = ["drug_id", "prot_id", "label", "prot_domain"]
    missing = set(required) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing pair columns: {sorted(missing)}")
    result = frame.copy()
    for col in ["drug_id", "prot_id"]:
        if result[col].isna().any():
            raise ValueError(f"Missing {col}")
        result[col] = result[col].astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
        if result[col].eq("").any():
            raise ValueError(f"Empty {col}")
    result["label"] = pd.to_numeric(result["label"], errors="raise")
    if not result["label"].isin([-1, 0, 1]).all():
        raise ValueError("Labels must be -1, 0 or 1")
    result["label"] = result["label"].astype(int)
    result["prot_domain"] = result["prot_domain"].astype(str).str.strip().str.lower()
    if not result["prot_domain"].isin(["human", "virus"]).all():
        raise ValueError("prot_domain must be human or virus")
    keys = ["drug_id", "prot_id"]
    domains = result.groupby(keys)["prot_domain"].nunique()
    if domains.gt(1).any():
        raise ValueError("A compound-protein pair has inconsistent domains")
    hard = result[result["label"].isin([0, 1])]
    conflicts = hard.groupby(keys)["label"].nunique()
    bad = conflicts[conflicts.gt(1)].index
    conflict_mask = pd.MultiIndex.from_frame(result[keys]).isin(bad)
    stats = {"input_rows": len(result), "conflicting_pairs_removed": len(bad),
             "conflicting_rows_removed": int(conflict_mask.sum())}
    result = result.loc[~conflict_mask].copy()
    before = len(result)
    result = result.drop_duplicates(required, keep="first")
    stats["duplicate_rows_removed"] = before - len(result)
    if result.duplicated(keys).any():
        raise ValueError("Unresolved gray/hard pair evidence; resolve upstream before training")
    stats["output_rows"] = len(result)
    return result.reset_index(drop=True), stats
