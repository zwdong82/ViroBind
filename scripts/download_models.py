#!/usr/bin/env python3
"""Download ViroBind release checkpoints and verify their SHA-256 digests."""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
from urllib.parse import quote
from urllib.request import urlopen


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DESTINATION = PROJECT_ROOT / "Pretrained_models" / "ViroBind"


def read_checksums(path: Path) -> dict[str, str]:
    checksums: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        digest, name = line.split(maxsplit=1)
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError(f"Invalid SHA-256 line in {path}: {raw!r}")
        checksums[name] = digest
    return checksums


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, destination: Path) -> None:
    temporary = destination.with_suffix(destination.suffix + ".part")
    try:
        with urlopen(url) as response, temporary.open("wb") as handle:
            while chunk := response.read(1024 * 1024):
                handle.write(chunk)
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-url",
        default=os.environ.get("VIROBIND_MODEL_BASE_URL", ""),
        help="Release-asset directory URL; may also be set with VIROBIND_MODEL_BASE_URL.",
    )
    parser.add_argument("--destination", default=str(DEFAULT_DESTINATION))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if not args.base_url:
        parser.error("pass --base-url or set VIROBIND_MODEL_BASE_URL")

    destination = Path(args.destination)
    destination.mkdir(parents=True, exist_ok=True)
    manifest = destination / "SHA256SUMS"
    if not manifest.is_file():
        raise FileNotFoundError(f"Checksum manifest not found: {manifest}")

    for name, expected in read_checksums(manifest).items():
        output = destination / name
        if output.is_file() and sha256(output) == expected and not args.overwrite:
            print(f"[OK] {output} already matches {expected}")
            continue
        url = args.base_url.rstrip("/") + "/" + quote(name)
        print(f"[DOWNLOAD] {url}")
        download(url, output)
        actual = sha256(output)
        if actual != expected:
            output.unlink()
            raise ValueError(f"SHA-256 mismatch for {name}: expected {expected}, got {actual}")
        print(f"[OK] {output} matches {expected}")


if __name__ == "__main__":
    main()
