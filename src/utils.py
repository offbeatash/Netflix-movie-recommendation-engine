import os
import json
from pathlib import Path
import psutil
from functools import wraps


def _metadata_path(artifact_path):
    return Path(f"{artifact_path}.metadata.json")


def check_artifact_freshness(artifact_path, current_params: dict):
    """Return whether an artifact has metadata matching the current parameters."""
    metadata_path = _metadata_path(artifact_path)
    if not metadata_path.exists():
        return False

    try:
        with metadata_path.open("r", encoding="utf-8") as metadata_file:
            return json.load(metadata_file) == current_params
    except (OSError, json.JSONDecodeError):
        return False


def save_artifact_metadata(artifact_path, current_params: dict):
    """Save the parameters used to create an artifact beside that artifact."""
    metadata_path = _metadata_path(artifact_path)
    with metadata_path.open("w", encoding="utf-8") as metadata_file:
        json.dump(current_params, metadata_file, indent=2, sort_keys=True)
        metadata_file.write("\n")


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
