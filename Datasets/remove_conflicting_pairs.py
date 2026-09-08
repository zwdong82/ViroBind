"""Remove every row belonging to a drug-protein pair with conflicting labels.

Conflicts are detected independently in each released CSV.  If the same
``(drug_id, prot_id)`` pair has both label 0 and label 1, all rows for that
pair are removed; the script never guesses which label is correct.

After cleaning, ``split_manifest.json`` is rebuilt from the CSV files so its
row counts, duplicate counts, label counts, and overlap audit stay accurate.
"""

from __future__ import annotations

import csv
import json
import os
from collections import Counter, defaultdict
from pathlib import Path


DATASETS = Path(__file__).resolve().parent
SPLITS = ("random", "scaffold")
FILES = (
    "human_pretrain.csv",
    "virus_finetune.csv",
    "virus_val.csv",
    "virus_test.csv",
)
KEY_COLUMNS = ("drug_id", "prot_id")
EXPECTED_COLUMNS = ["drug_id", "prot_id", "label", "prot_domain"]


def find_conflicts(path: Path) -> set[tuple[str, str]]:
    labels_by_pair: dict[tuple[str, str], set[str]] = defaultdict(set)
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != EXPECTED_COLUMNS:
            raise ValueError(f"Unexpected columns in {path}: {reader.fieldnames}")
        for row in reader:
            pair = tuple(row[column] for column in KEY_COLUMNS)
            labels_by_pair[pair].add(row["label"])
    return {pair for pair, labels in labels_by_pair.items() if len(labels) > 1}


def remove_conflicts(path: Path, conflicts: set[tuple[str, str]]) -> int:
    if not conflicts:
        return 0

    temporary = path.with_suffix(path.suffix + ".tmp")
    removed = 0
    with path.open(newline="", encoding="utf-8") as source, temporary.open(
        "w", newline="", encoding="utf-8"
    ) as destination:
        reader = csv.DictReader(source)
        writer = csv.DictWriter(destination, fieldnames=EXPECTED_COLUMNS)
        writer.writeheader()
        for row in reader:
            pair = tuple(row[column] for column in KEY_COLUMNS)
            if pair in conflicts:
                removed += 1
            else:
                writer.writerow(row)
    os.replace(temporary, path)
    return removed


def audit_file(path: Path) -> tuple[dict[str, object], set[tuple[str, str]], set[str], set[str]]:
    rows = 0
    row_counts: Counter[tuple[str, ...]] = Counter()
    labels_by_pair: dict[tuple[str, str], set[str]] = defaultdict(set)
    label_counts: Counter[str] = Counter()
    drugs: set[str] = set()
    proteins: set[str] = set()

    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows += 1
            values = tuple(row[column] for column in EXPECTED_COLUMNS)
            pair = (row["drug_id"], row["prot_id"])
            row_counts[values] += 1
            labels_by_pair[pair].add(row["label"])
            label_counts[row["label"]] += 1
            drugs.add(row["drug_id"])
            proteins.add(row["prot_id"])

    pairs = set(labels_by_pair)
    audit = {
        "rows": rows,
        "exact_duplicate_rows": sum(count - 1 for count in row_counts.values()),
        "conflicting_drug_protein_pairs": sum(
            len(labels) > 1 for labels in labels_by_pair.values()
        ),
        "label_counts": {label: label_counts[label] for label in sorted(label_counts)},
        "unique_drugs": len(drugs),
        "unique_proteins": len(proteins),
    }
    return audit, pairs, drugs, proteins


def overlap(
    left: tuple[dict[str, object], set[tuple[str, str]], set[str], set[str]],
    right: tuple[dict[str, object], set[tuple[str, str]], set[str], set[str]],
) -> dict[str, int]:
    return {
        "pairs": len(left[1] & right[1]),
        "drugs": len(left[2] & right[2]),
        "proteins": len(left[3] & right[3]),
    }


def rebuild_manifest() -> None:
    manifest_path = DATASETS / "split_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for split in SPLITS:
        audits = {name: audit_file(DATASETS / split / name) for name in FILES}
        manifest["splits"][split]["files"] = {
            name: audits[name][0] for name in FILES
        }
        manifest["splits"][split]["virus_partition_overlap"] = {
            "finetune_val": overlap(
                audits["virus_finetune.csv"], audits["virus_val.csv"]
            ),
            "finetune_test": overlap(
                audits["virus_finetune.csv"], audits["virus_test.csv"]
            ),
            "val_test": overlap(audits["virus_val.csv"], audits["virus_test.csv"]),
        }

    manifest["interpretation_notes"] = [
        "Every row belonging to a drug-protein pair with conflicting binary labels was removed; no label was selected heuristically.",
        "Exact duplicate rows that do not have conflicting labels are retained and reported.",
        "The anonymous IDs cannot be converted back to SMILES, UniProt accessions or protein sequences from this public release.",
    ]
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def main() -> None:
    report: dict[str, dict[str, dict[str, int]]] = {}
    for split in SPLITS:
        report[split] = {}
        for name in FILES:
            path = DATASETS / split / name
            conflicts = find_conflicts(path)
            removed = remove_conflicts(path, conflicts)
            report[split][name] = {
                "conflicting_pairs": len(conflicts),
                "removed_rows": removed,
            }
    rebuild_manifest()
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
