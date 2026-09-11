import json

from src.utils import check_artifact_freshness, save_artifact_metadata


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
