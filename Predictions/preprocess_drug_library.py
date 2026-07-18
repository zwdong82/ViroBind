#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Prepare a ChEMBL compound export for ViroBind drug-only screening.

Input is the semicolon-separated ChEMBL compound table exported as ``chembl.csv``.
The main output, ``chembl_drug2id.csv``, matches the project drug table schema:
``drug_id, drug_key, SMILES, SMILES_Std, SMILES_ChargeAware``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

from rdkit import Chem, RDLogger
from rdkit.Chem import Crippen, Descriptors, Lipinski, rdMolDescriptors
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams
from rdkit.Chem.MolStandardize import rdMolStandardize


RDLogger.DisableLog("rdApp.error")
RDLogger.DisableLog("rdApp.warning")

DEFAULT_ALLOWED_ELEMENTS = {
    "H",
    "C",
    "N",
    "O",
    "F",
    "P",
    "S",
    "Cl",
    "Br",
    "I",
    "B",
    "Si",
    "Se",
}
METAL_ELEMENTS = {
    "Li",
    "Na",
    "K",
    "Mg",
    "Ca",
    "Mn",
    "Fe",
    "Co",
    "Ni",
    "Cu",
    "Zn",
    "Pt",
    "Pd",
    "Ag",
    "Au",
}


def build_filter_catalog() -> FilterCatalog:
    params = FilterCatalogParams()
    params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS_A)
    params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS_B)
    params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS_C)
    params.AddCatalog(FilterCatalogParams.FilterCatalogs.BRENK)
    return FilterCatalog(params)


NORMALIZER = rdMolStandardize.Normalizer()
REMOVER = rdMolStandardize.FragmentRemover()
LFC = rdMolStandardize.LargestFragmentChooser()
UNCHARGER = rdMolStandardize.Uncharger()
TAUTOMER_ENUMERATOR = rdMolStandardize.TautomerEnumerator()
FILTER_CATALOG = build_filter_catalog()


def filter_params(mode: str) -> dict[str, float] | None:
    if mode == "standard":
        return {
            "mw_min": 120,
            "mw_max": 850,
            "heavy_min": 8,
            "heavy_max": 70,
            "logp_min": -2,
            "logp_max": 8,
            "hbd_max": 8,
            "hba_max": 15,
            "tpsa_max": 220,
            "rotb_max": 20,
        }
    if mode == "relaxed":
        return {
            "mw_min": 80,
            "mw_max": 1200,
            "heavy_min": 4,
            "heavy_max": 100,
            "logp_min": -3,
            "logp_max": 10,
            "hbd_max": 12,
            "hba_max": 20,
            "tpsa_max": 300,
            "rotb_max": 30,
        }
    if mode == "none":
        return None
    raise ValueError("chem_filter_mode must be one of: standard, relaxed, none")


def clean_text(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip()
    if text.lower() in {"nan", "none", "null"}:
        return ""
    return text


def to_float(value: Any, default: float = math.nan) -> float:
    text = clean_text(value)
    if not text:
        return default
    try:
        return float(text)
    except ValueError:
        return default


def to_bool_false_ok(value: Any) -> bool:
    return clean_text(value).lower() in {"false", "0", "no", "n", ""}


def alert_tags(mol: Chem.Mol) -> dict[str, Any]:
    hit_names = [hit.GetDescription() for hit in FILTER_CATALOG.GetMatches(mol)]
    pains_hits = [x for x in hit_names if "pains" in x.lower()]
    brenk_hits = [x for x in hit_names if "brenk" in x.lower()]
    return {
        "PAINS_Hit": int(bool(pains_hits)),
        "Brenk_Hit": int(bool(brenk_hits)),
        "PAINS_Detail": "; ".join(pains_hits),
        "Brenk_Detail": "; ".join(brenk_hits),
    }


def empty_result(status: str) -> dict[str, Any]:
    return {
        "SMILES_Std": "",
        "SMILES_ChargeAware": "",
        "SMILES_Status": status,
        "PAINS_Hit": 0,
        "Brenk_Hit": 0,
        "PAINS_Detail": "",
        "Brenk_Detail": "",
        "MolWt": math.nan,
        "HeavyAtoms": math.nan,
        "LogP": math.nan,
        "HBD": math.nan,
        "HBA": math.nan,
        "TPSA": math.nan,
        "RotB": math.nan,
        "QED": math.nan,
    }


def finalize_variant(base_mol: Chem.Mol, do_uncharge: bool, canonical_tautomer: bool) -> tuple[Chem.Mol | None, str]:
    try:
        mol = Chem.Mol(base_mol)
        if do_uncharge:
            mol = UNCHARGER.uncharge(mol)
        if canonical_tautomer:
            mol = TAUTOMER_ENUMERATOR.Canonicalize(mol)
        Chem.SanitizeMol(mol)
    except Exception:
        return None, "variant_finalize_fail"

    smi = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    if "." in smi:
        return None, "multi_component"
    return mol, "ok"


def process_smiles(payload: tuple[str, str, bool, bool, bool]) -> tuple[str, dict[str, Any]]:
    smiles, chem_filter_mode, allow_metals, drop_alerts, canonical_tautomer = payload
    smiles = clean_text(smiles)
    if not smiles:
        return smiles, empty_result("empty")

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return smiles, empty_result("parse_fail")

    try:
        mol = NORMALIZER.normalize(mol)
        mol = REMOVER.remove(mol)
        mol = LFC.choose(mol)
        Chem.SanitizeMol(mol)
    except Exception:
        return smiles, empty_result("standardize_fail")

    mol_charge, status = finalize_variant(mol, do_uncharge=False, canonical_tautomer=canonical_tautomer)
    if mol_charge is None:
        return smiles, empty_result(status)

    elems = {atom.GetSymbol() for atom in mol_charge.GetAtoms()}
    allowed = DEFAULT_ALLOWED_ELEMENTS | (METAL_ELEMENTS if allow_metals else set())
    if not elems.issubset(allowed):
        out = empty_result("bad_element")
        out["SMILES_ChargeAware"] = Chem.MolToSmiles(mol_charge, canonical=True, isomericSmiles=True)
        return smiles, out

    props = {
        "MolWt": Descriptors.MolWt(mol_charge),
        "HeavyAtoms": mol_charge.GetNumHeavyAtoms(),
        "LogP": Crippen.MolLogP(mol_charge),
        "HBD": Lipinski.NumHDonors(mol_charge),
        "HBA": Lipinski.NumHAcceptors(mol_charge),
        "TPSA": rdMolDescriptors.CalcTPSA(mol_charge),
        "RotB": Lipinski.NumRotatableBonds(mol_charge),
        "QED": Chem.QED.qed(mol_charge),
    }

    params = filter_params(chem_filter_mode)
    if params is not None:
        checks = [
            (props["MolWt"] < params["mw_min"] or props["MolWt"] > params["mw_max"], "mw_out_of_range"),
            (props["HeavyAtoms"] < params["heavy_min"] or props["HeavyAtoms"] > params["heavy_max"], "heavy_atom_out_of_range"),
            (props["LogP"] < params["logp_min"] or props["LogP"] > params["logp_max"], "logp_out_of_range"),
            (props["HBD"] > params["hbd_max"], "hbd_out_of_range"),
            (props["HBA"] > params["hba_max"], "hba_out_of_range"),
            (props["TPSA"] > params["tpsa_max"], "tpsa_out_of_range"),
            (props["RotB"] > params["rotb_max"], "rotb_out_of_range"),
        ]
        for failed, reason in checks:
            if failed:
                out = empty_result(reason)
                out.update(props)
                out["SMILES_ChargeAware"] = Chem.MolToSmiles(mol_charge, canonical=True, isomericSmiles=True)
                return smiles, out

    alerts = alert_tags(mol_charge)
    if drop_alerts and (alerts["PAINS_Hit"] or alerts["Brenk_Hit"]):
        out = empty_result("structural_alert")
        out.update(props)
        out.update(alerts)
        out["SMILES_ChargeAware"] = Chem.MolToSmiles(mol_charge, canonical=True, isomericSmiles=True)
        return smiles, out

    mol_std, _ = finalize_variant(mol, do_uncharge=True, canonical_tautomer=canonical_tautomer)
    smiles_std = Chem.MolToSmiles(mol_std, canonical=True, isomericSmiles=True) if mol_std is not None else ""
    smiles_charge = Chem.MolToSmiles(mol_charge, canonical=True, isomericSmiles=True)

    out = {
        "SMILES_Std": smiles_std,
        "SMILES_ChargeAware": smiles_charge,
        "SMILES_Status": "ok" if smiles_std else "std_variant_fail",
    }
    out.update(alerts)
    out.update(props)
    return smiles, out


def read_candidate_rows(input_path: Path, chunk_size: int) -> tuple[pd.DataFrame, dict[str, int]]:
    usecols = [
        "Compound ChEMBL ID",
        "Name",
        "Type",
        "Max Phase",
        "Molecular Weight",
        "Targets",
        "Bioactivities",
        "AlogP",
        "Polar Surface Area",
        "HBA",
        "HBD",
        "#RO5 Violations",
        "#Rotatable Bonds",
        "Structure Type",
        "Inorganic Flag",
        "Heavy Atoms",
        "Smiles",
        "Inchi Key",
        "Withdrawn Flag",
    ]
    counters: Counter[str] = Counter()
    rows: list[pd.DataFrame] = []
    reader = pd.read_csv(
        input_path,
        sep=";",
        quotechar='"',
        usecols=usecols,
        dtype=str,
        chunksize=chunk_size,
        low_memory=False,
        on_bad_lines="skip",
    )
    for chunk in tqdm(reader, desc="Read/filter ChEMBL", unit="chunk"):
        counters["raw_rows"] += len(chunk)
        chunk = chunk.rename(
            columns={
                "Compound ChEMBL ID": "chembl_id",
                "Name": "name",
                "Type": "type",
                "Max Phase": "max_phase",
                "Molecular Weight": "chembl_mw",
                "Targets": "targets",
                "Bioactivities": "bioactivities",
                "AlogP": "chembl_alogp",
                "Polar Surface Area": "chembl_tpsa",
                "HBA": "chembl_hba",
                "HBD": "chembl_hbd",
                "#RO5 Violations": "chembl_ro5_violations",
                "#Rotatable Bonds": "chembl_rotb",
                "Structure Type": "structure_type",
                "Inorganic Flag": "inorganic_flag",
                "Heavy Atoms": "chembl_heavy_atoms",
                "Smiles": "SMILES",
                "Inchi Key": "inchi_key",
                "Withdrawn Flag": "withdrawn_flag",
            }
        )
        for col in chunk.columns:
            chunk[col] = chunk[col].map(clean_text)

        mask_smiles = chunk["SMILES"] != ""
        mask_type = chunk["type"].str.lower().eq("small molecule")
        mask_structure = chunk["structure_type"].str.upper().eq("MOL")
        mask_not_withdrawn = chunk["withdrawn_flag"].map(to_bool_false_ok)
        mask_not_inorganic = ~chunk["inorganic_flag"].isin(["1", "True", "true"])
        keep = mask_smiles & mask_type & mask_structure & mask_not_withdrawn & mask_not_inorganic

        counters["missing_smiles"] += int((~mask_smiles).sum())
        counters["non_small_molecule"] += int((mask_smiles & ~mask_type).sum())
        counters["non_mol_structure"] += int((mask_smiles & mask_type & ~mask_structure).sum())
        counters["withdrawn"] += int((mask_smiles & mask_type & mask_structure & ~mask_not_withdrawn).sum())
        counters["inorganic"] += int((mask_smiles & mask_type & mask_structure & mask_not_withdrawn & ~mask_not_inorganic).sum())

        if keep.any():
            rows.append(chunk.loc[keep].copy())
            counters["prefilter_kept_rows"] += int(keep.sum())

    if not rows:
        return pd.DataFrame(columns=["chembl_id", "SMILES"]), dict(counters)
    return pd.concat(rows, ignore_index=True), dict(counters)


def metadata_druglike_prefilter(df: pd.DataFrame, mode: str) -> tuple[pd.DataFrame, dict[str, int]]:
    """Use ChEMBL-exported physicochemical properties as a fast prefilter."""
    params = filter_params(mode)
    if params is None or df.empty:
        return df.copy(), {}

    checks = [
        ("chembl_mw", "mw_prefilter_out_of_range", lambda s: (s >= params["mw_min"]) & (s <= params["mw_max"])),
        ("chembl_heavy_atoms", "heavy_atom_prefilter_out_of_range", lambda s: (s >= params["heavy_min"]) & (s <= params["heavy_max"])),
        ("chembl_alogp", "logp_prefilter_out_of_range", lambda s: (s >= params["logp_min"]) & (s <= params["logp_max"])),
        ("chembl_hbd", "hbd_prefilter_out_of_range", lambda s: s <= params["hbd_max"]),
        ("chembl_hba", "hba_prefilter_out_of_range", lambda s: s <= params["hba_max"]),
        ("chembl_tpsa", "tpsa_prefilter_out_of_range", lambda s: s <= params["tpsa_max"]),
        ("chembl_rotb", "rotb_prefilter_out_of_range", lambda s: s <= params["rotb_max"]),
    ]

    keep = pd.Series(True, index=df.index)
    counters: Counter[str] = Counter()
    for col, reason, predicate in checks:
        values = pd.to_numeric(df[col], errors="coerce")
        known = values.notna()
        ok = (~known) | predicate(values)
        newly_failed = keep & ~ok
        counters[reason] = int(newly_failed.sum())
        keep &= ok

    counters["metadata_prefilter_removed_rows"] = int((~keep).sum())
    counters["metadata_prefilter_kept_rows"] = int(keep.sum())
    return df.loc[keep].copy(), dict(counters)


def build_smiles_cache(
    smiles: list[str],
    chem_filter_mode: str,
    allow_metals: bool,
    drop_alerts: bool,
    canonical_tautomer: bool,
    workers: int,
    mp_chunksize: int,
) -> dict[str, dict[str, Any]]:
    payload = [(s, chem_filter_mode, allow_metals, drop_alerts, canonical_tautomer) for s in smiles]
    if workers <= 1:
        return {
            raw: result
            for raw, result in tqdm(
                map(process_smiles, payload),
                total=len(payload),
                desc="Standardize/filter SMILES",
                unit="smi",
            )
        }

    out: dict[str, dict[str, Any]] = {}
    with ProcessPoolExecutor(max_workers=workers) as executor:
        iterator = executor.map(process_smiles, payload, chunksize=mp_chunksize)
        for raw, result in tqdm(iterator, total=len(payload), desc="Standardize/filter SMILES", unit="smi"):
            out[raw] = result
    return out


def choose_representatives(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    work["max_phase_num"] = work["max_phase"].map(to_float).fillna(-1)
    work["targets_num"] = work["targets"].map(to_float).fillna(-1)
    work["bioactivities_num"] = work["bioactivities"].map(to_float).fillna(-1)
    work["name_present"] = (work["name"] != "").astype(int)
    work = work.sort_values(
        ["SMILES_Std", "max_phase_num", "bioactivities_num", "targets_num", "name_present", "chembl_id"],
        ascending=[True, False, False, False, False, True],
    )
    return work.drop_duplicates("SMILES_Std", keep="first").reset_index(drop=True)


def write_summary(path: Path, summary: dict[str, Any]) -> None:
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    txt = path.with_suffix(".txt")
    lines = []
    for key, value in summary.items():
        if isinstance(value, dict):
            lines.append(f"{key}:")
            lines.extend(f"  {k}: {v}" for k, v in value.items())
        else:
            lines.append(f"{key}: {value}")
    txt.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input semicolon-separated ChEMBL export.")
    parser.add_argument("--outdir", default="outputs/preprocessed_library")
    parser.add_argument("--chunk-size", type=int, default=200000)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--mp-chunksize", type=int, default=256)
    parser.add_argument("--chem-filter-mode", choices=["standard", "relaxed", "none"], default="standard")
    parser.add_argument("--allow-metals", action="store_true")
    parser.add_argument("--keep-alerts", action="store_true", help="Annotate but do not remove PAINS/Brenk hits.")
    parser.add_argument(
        "--skip-tautomer-canonicalization",
        action="store_true",
        help="Faster mode: keep normalized largest-fragment tautomer instead of RDKit canonical tautomer.",
    )
    parser.add_argument("--limit-unique-smiles", type=int, default=0, help="Debug only.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    rows, prefilter_counts = read_candidate_rows(input_path, args.chunk_size)
    rows, metadata_prefilter_counts = metadata_druglike_prefilter(rows, args.chem_filter_mode)
    unique_smiles = sorted(rows["SMILES"].dropna().astype(str).unique().tolist())
    if args.limit_unique_smiles:
        unique_smiles = unique_smiles[: args.limit_unique_smiles]
        rows = rows[rows["SMILES"].isin(unique_smiles)].copy()

    cache = build_smiles_cache(
        unique_smiles,
        args.chem_filter_mode,
        args.allow_metals,
        not args.keep_alerts,
        not args.skip_tautomer_canonicalization,
        args.workers,
        args.mp_chunksize,
    )

    proc = pd.DataFrame.from_dict(cache, orient="index").reset_index(names="SMILES")
    merged = rows.merge(proc, on="SMILES", how="left")
    merged["keep_final"] = (merged["SMILES_Status"] == "ok") & (merged["SMILES_Std"] != "")
    qc_path = outdir / "chembl_drug_qc.csv"
    merged.to_csv(qc_path, index=False, quoting=csv.QUOTE_MINIMAL)

    kept = merged[merged["keep_final"]].copy()
    reps = choose_representatives(kept)
    reps = reps.sort_values("SMILES_Std").reset_index(drop=True)
    reps.insert(0, "drug_id", np.arange(len(reps), dtype=np.int64))
    reps["drug_key"] = reps["SMILES_Std"]
    reps["source_db"] = "ChEMBL"

    drug_cols = ["drug_id", "drug_key", "SMILES", "SMILES_Std", "SMILES_ChargeAware"]
    meta_cols = [
        "drug_id",
        "chembl_id",
        "drug_key",
        "SMILES",
        "SMILES_Std",
        "SMILES_ChargeAware",
        "inchi_key",
        "name",
        "max_phase",
        "targets",
        "bioactivities",
        "MolWt",
        "HeavyAtoms",
        "LogP",
        "HBD",
        "HBA",
        "TPSA",
        "RotB",
        "QED",
        "PAINS_Hit",
        "Brenk_Hit",
        "PAINS_Detail",
        "Brenk_Detail",
        "source_db",
    ]
    reps[drug_cols].to_csv(outdir / "chembl_drug2id.csv", index=False)
    reps[[c for c in meta_cols if c in reps.columns]].to_csv(outdir / "chembl_drug_metadata.csv", index=False)

    status_counts = merged["SMILES_Status"].fillna("missing_process_result").value_counts().to_dict()
    summary = {
        "input": str(input_path),
        "outdir": str(outdir),
        "chem_filter_mode": args.chem_filter_mode,
        "allow_metals": bool(args.allow_metals),
        "drop_pains_brenk_alerts": not bool(args.keep_alerts),
        "canonical_tautomer": not bool(args.skip_tautomer_canonicalization),
        "prefilter_counts": prefilter_counts,
        "metadata_druglike_prefilter_counts": metadata_prefilter_counts,
        "candidate_rows_after_prefilter": int(len(rows)),
        "unique_raw_smiles_processed": int(len(unique_smiles)),
        "rows_after_rdkit_filter": int(len(kept)),
        "unique_drugs_after_dedup": int(len(reps)),
        "duplicate_rows_removed_by_smiles_std": int(len(kept) - len(reps)),
        "smiles_status_counts": {str(k): int(v) for k, v in status_counts.items()},
        "outputs": {
            "drug2id": str(outdir / "chembl_drug2id.csv"),
            "metadata": str(outdir / "chembl_drug_metadata.csv"),
            "qc": str(qc_path),
        },
    }
    write_summary(outdir / "chembl_processing_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
