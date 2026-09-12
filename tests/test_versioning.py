import json

from src.utils import check_artifact_freshness, save_artifact_metadata
from src.versioning import model_version


def test_artifact_metadata_is_reproducible(tmp_path):
    data = tmp_path / "train.csv"
    artifact = tmp_path / "model.pkl"

    data.write_text("a,b\n1,2\n", encoding="utf-8")
    artifact.write_bytes(b"model")

    params = {"alpha": 0.5, "seed": 42}

    save_artifact_metadata(artifact, params, data)

    assert check_artifact_freshness(artifact, params, data)

    metadata = json.loads(
        (tmp_path / "model.pkl.metadata.json").read_text()
    )

    assert metadata["dataset_version"]
    assert metadata["model_version"]
    assert metadata["project_version"] == "2.2.0"


def test_model_version_includes_source(tmp_path):
    source1 = tmp_path / "source1.py"
    source2 = tmp_path / "source2.py"

    source1.write_text("print('hello')", encoding="utf-8")
    source2.write_text("print('world')", encoding="utf-8")

    params = {"alpha": 0.5, "seed": 42}
    data_version = "abc123"

    version_without_source = model_version(
        params,
        data_version,
    )

    version_with_source = model_version(
        params,
        data_version,
        [source1, source2],
    )

    assert version_without_source != version_with_source

    version_with_source_2 = model_version(
        params,
        data_version,
        [source1, source2],
    )

    assert version_with_source == version_with_source_2

    source1.write_text(
        "print('hello changed')",
        encoding="utf-8",
    )

    version_with_modified_source = model_version(
        params,
        data_version,
        [source1, source2],
    )

    assert version_with_source != version_with_modified_source


def test_artifact_freshness_respects_source_changes(tmp_path):
    data = tmp_path / "train.csv"
    artifact = tmp_path / "model.pkl"
    source_file = tmp_path / "model_impl.py"

    data.write_text("a,b\n1,2\n", encoding="utf-8")
    source_file.write_text(
        "# Original implementation\nPARAM = 0.5",
        encoding="utf-8",
    )
    artifact.write_bytes(b"")

    params = {"alpha": 0.5, "seed": 42}

    save_artifact_metadata(
        artifact,
        params,
        data,
        source_paths=[source_file],
    )

    assert check_artifact_freshness(
        artifact,
        params,
        data,
        source_paths=[source_file],
    )

    source_file.write_text(
        "# Modified implementation\nPARAM = 0.6",
        encoding="utf-8",
    )

    assert not check_artifact_freshness(
        artifact,
        params,
        data,
        source_paths=[source_file],
    )

    assert check_artifact_freshness(
        artifact,
        params,
        data,
    )


def test_artifact_freshness_ignore_source_when_not_provided(tmp_path):
    data = tmp_path / "train.csv"
    artifact = tmp_path / "model.pkl"
    source_file = tmp_path / "source.py"

    data.write_text("a,b\n1,2\n", encoding="utf-8")
    artifact.write_bytes(b"model")
    source_file.write_text("original code", encoding="utf-8")

    params = {"alpha": 0.5, "seed": 42}

    save_artifact_metadata(
        artifact,
        params,
        data,
        source_paths=[source_file],
    )

    source_file.write_text("modified code", encoding="utf-8")

    assert check_artifact_freshness(
        artifact,
        params,
        data,
    )

    assert not check_artifact_freshness(
        artifact,
        params,
        data,
        source_paths=[source_file],
    )


def test_svd_source_dependencies():
    from src.models.svd_model import SVD_SOURCE_PATHS

    source_names = {path.as_posix() for path in SVD_SOURCE_PATHS}

    assert source_names == {
        "src/models/svd_model.py",
        "src/utils.py",
        "src/config.py",
    }


def test_popularity_source_dependencies():
    from src.models.popularity import POPULARITY_SOURCE_PATHS

    source_names = {path.as_posix() for path in POPULARITY_SOURCE_PATHS}

    assert source_names == {
        "src/models/popularity.py",
        "src/utils.py",
        "src/config.py",
    }


def test_ensemble_source_dependencies():
    from src.models.ensemble import ENSEMBLE_SOURCE_PATHS

    source_names = {path.as_posix() for path in ENSEMBLE_SOURCE_PATHS}

    assert source_names == {
        "src/models/ensemble.py",
        "src/models/popularity.py",
        "src/models/svd_model.py",
        "src/utils.py",
        "src/config.py",
    }


def test_als_source_dependencies():
    from src.models.als_model import ALS_SOURCE_PATHS

    source_names = {path.as_posix() for path in ALS_SOURCE_PATHS}

    assert source_names == {
        "src/models/als_model.py",
        "src/utils.py",
        "src/config.py",
    }
    