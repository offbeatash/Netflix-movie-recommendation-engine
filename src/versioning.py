"""Lightweight dataset and model version identifiers."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

from src.config import PROJECT_VERSION


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def _hash_files(paths: list[Path]) -> str:
    digest = hashlib.sha256()

    for path in sorted(paths, key=str):
        digest.update(path.name.encode("utf-8"))
        digest.update(sha256_file(path).encode("ascii"))

    return digest.hexdigest()


def dataset_version(paths: list[Path]) -> str:
    return _hash_files(paths)[:12]


def model_version(
    params: dict[str, Any],
    data_version: str,
    source_paths: list[Path] | None = None,
) -> str:
    payload = {
        "project": PROJECT_VERSION,
        "params": params,
        "data": data_version,
    }

    if source_paths is not None:
        payload["source"] = _hash_files(source_paths)

    payload_json = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )

    return hashlib.sha256(payload_json.encode("utf-8")).hexdigest()[:12]


def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None
    