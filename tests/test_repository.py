from __future__ import annotations

import ast
import csv
import json
import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPLITS = {
    "random",
    "scaffold",
}
SPLIT_FILES = {
    "human_pretrain.csv",
    "virus_finetune.csv",
    "virus_val.csv",
    "virus_test.csv",
}
DTI_COLUMNS = ["drug_id", "prot_id", "label", "prot_domain"]


class RepositoryIntegrityTests(unittest.TestCase):
    def test_required_release_files_exist(self) -> None:
        required = [
            "README.md",
            "LICENSE",
            "DATA_AND_MODEL_USE.md",
            "CONTRIBUTING.md",
            "FINAL_FILES.md",
            "requirements.txt",
            "pyproject.toml",
            "source/virobind/__init__.py",
            "source/virobind/base.py",
            "source/virobind/model.py",
            "source/virobind/predict.py",
            "source/virobind/screen.py",
            "Pretrained_models/ViroBind/SHA256SUMS",
            "Pretrained_models/ViroBind/MODEL_CARD.md",
            "Datasets/split_manifest.json",
            "examples/create_mock_assets.py",
            "examples/create_mock_checkpoints.py",
            "examples/pairs.csv",
            "examples/proteins.csv",
            "examples/library.csv",
            "scripts/download_models.py",
        ]
        missing = [path for path in required if not (ROOT / path).is_file()]
        self.assertEqual(missing, [])

    def test_python_sources_parse(self) -> None:
        for path in ROOT.rglob("*.py"):
            if ".git" in path.parts:
                continue
            with self.subTest(path=path.relative_to(ROOT)):
                ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

    def test_split_layout_and_headers(self) -> None:
        present = {path.name for path in (ROOT / "Datasets").iterdir() if path.is_dir()}
        self.assertEqual(present, SPLITS)
        for split in SPLITS:
            split_dir = ROOT / "Datasets" / split
            files = {path.name for path in split_dir.glob("*.csv")}
            self.assertEqual(files, SPLIT_FILES)
            for path in split_dir.glob("*.csv"):
                with path.open(newline="", encoding="utf-8") as handle:
                    self.assertEqual(next(csv.reader(handle)), DTI_COLUMNS)

    def test_example_ids_are_internally_consistent(self) -> None:
        with (ROOT / "examples/pairs.csv").open(newline="", encoding="utf-8") as handle:
            pairs = list(csv.DictReader(handle))
        with (ROOT / "examples/proteins.csv").open(newline="", encoding="utf-8") as handle:
            proteins = list(csv.DictReader(handle))
        with (ROOT / "examples/library.csv").open(newline="", encoding="utf-8") as handle:
            library = list(csv.DictReader(handle))
        self.assertEqual({row["prot_id"] for row in pairs}, {row["prot_id"] for row in proteins})
        self.assertEqual({row["drug_id"] for row in pairs}, {row["drug_id"] for row in library})

    def test_requirements_are_exactly_pinned(self) -> None:
        pins = []
        for raw in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith(("#", "--")):
                continue
            self.assertRegex(line, r"^[A-Za-z0-9_.-]+==[^=\s]+$")
            pins.append(line.split("==", 1)[0].lower().replace("_", "-"))
        self.assertEqual(len(pins), len(set(pins)))
        for direct in ["numpy", "pandas", "rdkit", "scikit-learn", "torch", "tqdm", "esm"]:
            self.assertIn(direct, pins)
        self.assertNotIn("torchtext", pins)

    def test_noncommercial_license_is_declared(self) -> None:
        license_text = (ROOT / "LICENSE").read_text(encoding="utf-8")
        pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        marker = "PolyForm Noncommercial License 1.0.0"
        self.assertIn(marker, license_text)
        self.assertIn('license = {file = "LICENSE"}', pyproject)
        self.assertIn(marker, readme)

    def test_split_manifest_matches_csv_row_counts(self) -> None:
        manifest = json.loads(
            (ROOT / "Datasets/split_manifest.json").read_text(encoding="utf-8")
        )
        for split, split_audit in manifest["splits"].items():
            for name, audit in split_audit["files"].items():
                with (ROOT / "Datasets" / split / name).open(encoding="utf-8") as handle:
                    self.assertEqual(sum(1 for _ in handle) - 1, audit["rows"])

    def test_weight_manifest(self) -> None:
        lines = [
            line.strip()
            for line in (ROOT / "Pretrained_models/ViroBind/SHA256SUMS")
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        ]
        expected = {"virobind_classification.pt", "virobind_ranking.pt"}
        names = set()
        for line in lines:
            match = re.fullmatch(r"([0-9a-f]{64})\s+(.+)", line)
            self.assertIsNotNone(match)
            names.add(match.group(2))
        self.assertEqual(names, expected)

    def test_public_text_has_no_private_absolute_path(self) -> None:
        suffixes = {".md", ".py", ".toml", ".txt", ".yml", ".yaml"}
        private_prefix = "/" + "data/zwdong/"
        offenders = []
        for path in ROOT.rglob("*"):
            if not path.is_file() or ".git" in path.parts or path.suffix not in suffixes:
                continue
            if private_prefix in path.read_text(encoding="utf-8"):
                offenders.append(str(path.relative_to(ROOT)))
        self.assertEqual(offenders, [])


if __name__ == "__main__":
    unittest.main()
