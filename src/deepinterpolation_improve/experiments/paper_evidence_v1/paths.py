from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any


PATHS_ENV = "DIP_EVIDENCE_PATHS"


def activate_path_overrides(path: str | Path | None) -> None:
    """Select a machine-local path map without changing the frozen study config."""
    if path is None:
        return
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    os.environ[PATHS_ENV] = str(resolved)


def load_path_overrides() -> tuple[dict[str, Any], Path | None, str | None]:
    selected = os.environ.get(PATHS_ENV)
    if not selected:
        return {}, None, None
    source = Path(selected).resolve()
    with source.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if int(payload.get("schema_version", 0)) != 1:
        raise ValueError(f"Unsupported path-override schema in {source}")
    roots = payload.get("recording_roots", {})
    if not isinstance(roots, dict):
        raise ValueError("recording_roots must be an object")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    return payload, source, digest


def resolve_override(source: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path.resolve()
    return (source.parent / path).resolve()
