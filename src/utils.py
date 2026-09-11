import hashlib
import json
import os
import time
from functools import wraps
from pathlib import Path
from typing import Any, Callable, TypeVar, cast

import psutil

from src.config import PROJECT_VERSION
from src.versioning import dataset_version, git_commit, model_version

F = TypeVar("F", bound=Callable[..., Any])


def _metadata_path(artifact_path: Path | str) -> Path:
    return Path(f"{artifact_path}.metadata.json")


def _hash_params(params: dict[str, Any]) -> str:
    payload = json.dumps(params, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _hash_file(filepath: str | Path, block_size: int = 65536) -> str:
    sha256 = hashlib.sha256()
    with Path(filepath).open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            sha256.update(block)
    return sha256.hexdigest()


def _data_hash(data_paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(data_paths, key=str):
        digest.update(_hash_file(path).encode("ascii"))
    return digest.hexdigest()


def check_artifact_freshness(
    artifact_path: Path | str,
    current_params: dict[str, Any],
    data_path: Path | str | list[Path | str] | None = None,
) -> bool:
    """Check that an artifact was created with the current code inputs."""
    metadata_path = _metadata_path(artifact_path)
    if not Path(artifact_path).exists() or not metadata_path.exists():
        return False

    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("params_hash") != _hash_params(current_params):
            return False

        if data_path is not None:
            paths = (
                [data_path] if isinstance(data_path, (str, Path)) else list(data_path)
            )
            if metadata.get("data_hash") != _data_hash([Path(p) for p in paths]):
                return False
    except (OSError, json.JSONDecodeError, TypeError):
        return False

    return True


def save_artifact_metadata(
    artifact_path: Path | str,
    current_params: dict[str, Any],
    data_path: Path | str | list[Path | str] | None = None,
) -> None:
    """Persist reproducibility metadata beside an artifact."""
    artifact_path = Path(artifact_path)
    metadata_path = _metadata_path(artifact_path)
    paths: list[Path] = []
    if data_path is not None:
        raw_paths = (
            [data_path] if isinstance(data_path, (str, Path)) else list(data_path)
        )
        paths = [Path(path) for path in raw_paths]

    data_hash = _data_hash(paths) if paths else None
    data_version = dataset_version(paths) if paths else None
    metadata = {
        "project_version": PROJECT_VERSION,
        "params": current_params,
        "params_hash": _hash_params(current_params),
        "data_hash": data_hash,
        "dataset_version": data_version,
        "model_version": model_version(current_params, data_version or "none"),
        "git_commit": git_commit(),
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    temp_path = metadata_path.with_name(f".{metadata_path.name}.tmp")
    temp_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temp_path.replace(metadata_path)


def log_memory(stage_name: str = "") -> None:
    process = psutil.Process(os.getpid())
    mem_gb = process.memory_info().rss / (1024**3)
    print(f"[Memory] {stage_name} | RAM in use: {mem_gb:.2f} GB")


def memory_tracker(func: F) -> F:
    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        log_memory(f"Starting {func.__name__}")
        result = func(*args, **kwargs)
        log_memory(f"Finished {func.__name__}")
        return result

    return cast(F, wrapper)
