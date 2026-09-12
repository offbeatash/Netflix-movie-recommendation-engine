import json
import tempfile
import os
from pathlib import Path

from src.utils import check_artifact_freshness, save_artifact_metadata
from src.versioning import model_version, _hash_files


def test_artifact_metadata_is_reproducible(tmp_path):
    data = tmp_path / "train.csv"
    artifact = tmp_path / "model.pkl"
    data.write_text("a,b\n1,2\n", encoding="utf-8")
    artifact.write_bytes(b"model")

    params = {"alpha": 0.5, "seed": 42}
    save_artifact_metadata(artifact, params, data)
    assert check_artifact_freshness(artifact, params, data)
    metadata = json.loads((tmp_path / "model.pkl.metadata.json").read_text())
    assert metadata["dataset_version"]
    assert metadata["model_version"]
    assert metadata["project_version"] == "2.2.0"


def test_model_version_includes_source(tmp_path):
    """Test that model_version function includes source file hashes when provided."""
    # Create temporary source files
    source1 = tmp_path / "source1.py"
    source2 = tmp_path / "source2.py"
    source1.write_text("print('hello')", encoding="utf-8")
    source2.write_text("print('world')", encoding="utf-8")

    params = {"alpha": 0.5, "seed": 42}
    data_version = "abc123"

    # Test without source paths (backward compatibility)
    version_without_source = model_version(params, data_version)

    # Test with source paths
    version_with_source = model_version(params, data_version, [source1, source2])

    # Versions should be different when source paths are provided
    assert version_without_source != version_with_source

    # Same source paths should produce same version
    version_with_source_2 = model_version(params, data_version, [source1, source2])
    assert version_with_source == version_with_source_2

    # Changing source files should change the version
    source1.write_text("print('hello changed')", encoding="utf-8")
    version_with_modified_source = model_version(params, data_version, [source1, source2])
    assert version_with_source != version_with_modified_source


def test_artifact_freshness_respects_source_changes(tmp_path):
    """Test that artifact freshness detection respects source code changes."""
    data = tmp_path / "train.csv"
    artifact = tmp_path / "model.pkl"
    data.write_text("a,b\n1,2\n", encoding="utf-8")

    # Create a temporary source file to hash
    source_file = tmp_path / "model_impl.py"
    source_file.write_text("# Original implementation\nPARAM = 0.5", encoding="utf-8")

    params = {"alpha": 0.5, "seed": 42}

    # Create the artifact file (can be empty for this test)
    artifact.write_bytes(b"")

    # Save artifact with source paths
    save_artifact_metadata(artifact, params, data, source_paths=[source_file])
    assert check_artifact_freshness(artifact, params, data, source_paths=[source_file])

    # Modify the source file
    source_file.write_text("# Modified implementation\nPARAM = 0.6", encoding="utf-8")

    # Now the artifact should appear stale due to source change
    assert not check_artifact_freshness(artifact, params, data, source_paths=[source_file])

    # But it should still be fresh if we don't check source paths
    assert check_artifact_freshness(artifact, params, data)


def test_artifact_freshness_ignore_source_when_not_provided(tmp_path):
    """Test that backward compatibility is maintained when source paths not provided."""
    data = tmp_path / "train.csv"
    artifact = tmp_path / "model.pkl"
    data.write_text("a,b\n1,2\n", encoding="utf-8")
    artifact.write_bytes(b"model")

    params = {"alpha": 0.5, "seed": 42}

    # Save artifact WITH source paths
    source_file = tmp_path / "source.py"
    source_file.write_text("original code", encoding="utf-8")
    save_artifact_metadata(artifact, params, data, source_paths=[source_file])

    # Modify the source file
    source_file.write_text("modified code", encoding="utf-8")

    # When checking freshness WITHOUT source paths, it should still appear fresh
    # (backward compatibility - ignores source changes if not requested)
    assert check_artifact_freshness(artifact, params, data)

    # When checking freshness WITH source paths, it should appear stale
    assert not check_artifact_freshness(artifact, params, data, source_paths=[source_file])
