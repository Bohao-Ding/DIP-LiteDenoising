from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
EVIDENCE_ROOT = REPOSITORY_ROOT / "evidence"
MANIFEST_PATH = EVIDENCE_ROOT / "distribution_manifest.json"
STUDIES = {
    "capacity_study_v1": {
        "trained_runs": 405,
        "scope": "3 recordings x 3 windows x 9 trained models x 5 seeds",
    },
    "paper_evidence_v1": {
        "trained_runs": 252,
        "scope": "225 locked core runs plus 27 locked mechanism runs",
    },
}
PUBLIC_SUFFIXES = {".csv", ".json", ".pdf", ".png", ".svg"}
FORBIDDEN_SUFFIXES = {
    ".ckpt",
    ".dat",
    ".dat2",
    ".h5",
    ".hdf5",
    ".npy",
    ".npz",
    ".onnx",
    ".pt",
    ".pth",
    ".tif",
    ".tiff",
}
TEXT_SUFFIXES = {".csv", ".json", ".svg"}
PRIVATE_PATH = re.compile(r"(?:[A-Za-z]:\\|/home/|/Users/)")
MAX_FILE_BYTES = 100 * 1024 * 1024


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def evidence_files() -> list[Path]:
    files: list[Path] = []
    for study in STUDIES:
        root = EVIDENCE_ROOT / study
        if not root.is_dir():
            raise FileNotFoundError(root)
        files.extend(path for path in root.rglob("*") if path.is_file())
    return sorted(files, key=lambda path: path.relative_to(REPOSITORY_ROOT).as_posix())


def validate_public_file(path: Path) -> None:
    if path.suffix.lower() in FORBIDDEN_SUFFIXES:
        raise ValueError(f"Forbidden evidence type: {path}")
    if path.suffix.lower() not in PUBLIC_SUFFIXES:
        raise ValueError(f"Unexpected evidence type: {path}")
    if path.stat().st_size > MAX_FILE_BYTES:
        raise ValueError(f"Evidence file exceeds 100 MiB: {path}")
    if path.suffix.lower() in TEXT_SUFFIXES:
        text = path.read_text(encoding="utf-8", errors="strict")
        if PRIVATE_PATH.search(text):
            raise ValueError(f"Private absolute path found in: {path}")


def build_manifest() -> dict[str, object]:
    records: list[dict[str, object]] = []
    for path in evidence_files():
        validate_public_file(path)
        relative = path.relative_to(REPOSITORY_ROOT).as_posix()
        records.append(
            {
                "path": relative,
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
        )
    return {
        "schema_version": 1,
        "source_protocol": "paper_evidence_v1",
        "source_freeze_hash": (
            "a723648a6ed1e510d65701b44a647b8fa6018dc0ad36474c3ebcec0fd9a71e74"
        ),
        "origin_evidence_manifest_sha256": (
            "478fa2ec5a50e9ed649480bcd9eebfd510d686384a3cc2394c537a0a936b742f"
        ),
        "studies": STUDIES,
        "files": records,
        "redactions": [
            {
                "source_path": "artifacts/paper_evidence_v1/tables/source_mapping.csv",
                "source_sha256": (
                    "e7ac3946e84577be2c8a914914e9a5964b39f5ff87816511e8559cd6f3d89dc3"
                ),
                "public_path": (
                    "evidence/paper_evidence_v1/tables/source_mapping.public.csv"
                ),
                "rule": (
                    "Renamed to distinguish the public derivative and replaced "
                    "machine-local dataset and split paths with repository-relative paths."
                ),
            },
            {
                "source_path": "artifacts/paper_evidence_v1/evidence_manifest.json",
                "source_sha256": (
                    "478fa2ec5a50e9ed649480bcd9eebfd510d686384a3cc2394c537a0a936b742f"
                ),
                "public_path": None,
                "rule": (
                    "Not redistributed because it names the private source mapping; its "
                    "SHA-256 is retained in origin_evidence_manifest_sha256."
                ),
            },
        ],
        "excluded_content": [
            "model checkpoints and state dictionaries",
            "Neurofinder pixels and recoverable image arrays",
            "predicted frames and full experiment work directories",
            "machine-local absolute paths",
        ],
        "claim_boundary": (
            "These files support audit and reproduction of reported local measurements; "
            "they do not redistribute Neurofinder recordings or establish performance "
            "outside the frozen study conditions."
        ),
    }


def serialise_manifest(payload: dict[str, object]) -> str:
    return json.dumps(payload, indent=2, ensure_ascii=False) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Build or verify the public evidence manifest")
    parser.add_argument("--check", action="store_true", help="verify without writing")
    args = parser.parse_args()
    expected = serialise_manifest(build_manifest())
    if args.check:
        if not MANIFEST_PATH.is_file():
            raise FileNotFoundError(MANIFEST_PATH)
        if MANIFEST_PATH.read_text(encoding="utf-8") != expected:
            raise RuntimeError("distribution manifest is stale")
        print(f"verified {MANIFEST_PATH.relative_to(REPOSITORY_ROOT)}")
        return 0
    MANIFEST_PATH.write_text(expected, encoding="utf-8")
    print(f"wrote {MANIFEST_PATH.relative_to(REPOSITORY_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
