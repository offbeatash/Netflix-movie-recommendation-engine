import hashlib
import json
import os
from pathlib import Path
import psutil
from functools import wraps


def _metadata_path(artifact_path):
    return Path(f"{artifact_path}.metadata.json")


def _hash_params(params: dict) -> str:
    """Compute a deterministic hash of the parameters dictionary."""
    # Sort keys to ensure deterministic ordering
    params_str = json.dumps(params, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(params_str.encode('utf-8')).hexdigest()


def _hash_file(filepath: str, block_size=65536) -> str:
    """Compute SHA256 hash of a file, reading in chunks to avoid memory issues."""
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for block in iter(lambda: f.read(block_size), b""):
            sha256.update(block)
    return sha256.hexdigest()


def check_artifact_freshness(artifact_path, current_params: dict, data_path=None):
    """Return whether an artifact has metadata matching the current parameters and data.

    Args:
        artifact_path: Path to the artifact file.
        current_params: Dictionary of current parameters.
        data_path: Optional path or list of paths to the data file(s). If provided, their hash(es) will be
                   stored and checked for freshness.

    Returns:
        True if artifact is fresh (matches params and data), False otherwise.
    """
    metadata_path = _metadata_path(artifact_path)
    if not metadata_path.exists():
        return False

    try:
        with metadata_path.open("r", encoding="utf-8") as metadata_file:
            metadata = json.load(metadata_file)
    except (OSError, json.JSONDecodeError):
        return False

    # Check params hash if available (new format)
    if "params_hash" in metadata:
        current_params_hash = _hash_params(current_params)
        if metadata["params_hash"] != current_params_hash:
            return False
    else:
        # Fallback for old format: compare entire params dict
        if metadata.get("params") != current_params:
            return False

    # Check data hash if applicable
    if data_path is not None:
        # Handle both single string/path and list of strings/paths for backward compatibility
        if isinstance(data_path, (str, Path)):
            data_paths = [data_path]
        else:
            data_paths = list(data_path)

        if "data_hash" in metadata:
            try:
                # Compute combined hash of all data files
                combined_hash = hashlib.sha256()
                for path in sorted(data_paths, key=lambda p: str(p)):  # sort for deterministic order
                    file_hash = _hash_file(str(path))
                    combined_hash.update(file_hash.encode('utf-8'))
                current_data_hash = combined_hash.hexdigest()
            except (OSError, IOError):
                # If we cannot read any data file, consider it not fresh
                return False
            if metadata["data_hash"] != current_data_hash:
                return False
        else:
            # Old format without data hash: we cannot verify data freshness,
            # so we consider it not fresh to be safe.
            return False

    return True


def save_artifact_metadata(artifact_path, current_params: dict, data_path=None):
    """Save metadata used to create an artifact.

    Args:
        artifact_path: Path to the artifact file.
        current_params: Dictionary of parameters used to create the artifact.
        data_path: Optional path or list of paths to the data file(s). If provided, their hash(es)
                   will be stored for future freshness checks.
    """
    metadata_path = _metadata_path(artifact_path)
    temp_metadata_path = metadata_path.with_suffix(".tmp")

    metadata = {
        "params": current_params,
        "params_hash": _hash_params(current_params),
        "timestamp": __import__('time').time(),
    }
    if data_path is not None:
        # Handle both single string/path and list of strings/paths for backward compatibility
        if isinstance(data_path, (str, Path)):
            data_paths = [data_path]
        else:
            data_paths = list(data_path)

        try:
            # Compute combined hash of all data files
            combined_hash = hashlib.sha256()
            for path in sorted(data_paths, key=lambda p: str(p)):  # sort for deterministic order
                file_hash = _hash_file(str(path))
                combined_hash.update(file_hash.encode('utf-8'))
            metadata["data_hash"] = combined_hash.hexdigest()
        except (OSError, IOError):
            # If we cannot hash any data file, we skip storing data hash.
            # This will cause future freshness checks to fall back to param-only
            # comparison, which is safe but less strict.
            pass

    # Write to temp file first, then atomically replace
    with temp_metadata_path.open("w", encoding="utf-8") as metadata_file:
        json.dump(metadata, metadata_file, indent=2, sort_keys=True)
        metadata_file.write("\n")

    # Atomic replace
    temp_metadata_path.replace(metadata_path)


def log_memory(stage_name=""):
    """Logs current RAM usage of the Python process in GB."""
    process = psutil.Process(os.getpid())
    mem_gb = process.memory_info().rss / (1024**3)
    print(f"[Memory] {stage_name} | RAM in use: {mem_gb:.2f} GB")


def memory_tracker(func):
    """Decorator to track memory before and after a pipeline function."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        log_memory(f"Starting {func.__name__}")
        result = func(*args, **kwargs)
        log_memory(f"Finished {func.__name__}")
        return result

    return wrapper