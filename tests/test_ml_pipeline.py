"""
Tests for the core Netflix ML pipeline.

These tests focus on correctness contracts rather than expensive full-model
training. The real fixture data is used where practical, while small
synthetic datasets and mocks are used to keep CI fast and deterministic.
"""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"


# INGESTION


def test_process_raw_data_parses_netflix_files(tmp_path, monkeypatch):
    from src.data import ingest

    data_dir = tmp_path / "raw"
    data_dir.mkdir()

    (data_dir / "combined_data_1.txt").write_text(
        "1:\n" "100,4,2005-01-01\n" "101,5,2005-01-02\n" "2:\n" "100,3,2005-01-03\n",
        encoding="utf-8",
    )

    processed_path = tmp_path / "processed.parquet"

    monkeypatch.setattr(ingest, "DATA_DIR", data_dir)
    monkeypatch.setattr(ingest, "PROCESSED_DATA_PATH", processed_path)

    ingest.process_raw_data()

    result = pd.read_parquet(processed_path)

    assert list(result.columns) == [
        "Movie_ID",
        "CustomerID",
        "Rating",
        "Date",
    ]

    assert len(result) == 3
    assert result["Movie_ID"].tolist() == [1, 1, 2]
    assert result["CustomerID"].tolist() == [100, 101, 100]
    assert result["Rating"].tolist() == [4, 5, 3]

    assert str(result["Movie_ID"].dtype) == "int32"
    assert str(result["CustomerID"].dtype) == "int32"
    assert str(result["Rating"].dtype) == "int8"
    assert pd.api.types.is_datetime64_any_dtype(result["Date"])


def test_process_raw_data_raises_when_no_raw_files_exist(
    tmp_path,
    monkeypatch,
):
    from src.data import ingest

    data_dir = tmp_path / "raw"
    data_dir.mkdir()

    processed_path = tmp_path / "processed.parquet"

    monkeypatch.setattr(ingest, "DATA_DIR", data_dir)
    monkeypatch.setattr(ingest, "PROCESSED_DATA_PATH", processed_path)

    with pytest.raises(FileNotFoundError, match="No raw Netflix"):
        ingest.process_raw_data()


def test_process_raw_data_skips_existing_output(tmp_path, monkeypatch):
    from src.data import ingest

    data_dir = tmp_path / "raw"
    data_dir.mkdir()

    processed_path = tmp_path / "processed.parquet"

    existing = pd.DataFrame(
        {
            "Movie_ID": [1],
            "CustomerID": [100],
            "Rating": [5],
            "Date": [pd.Timestamp("2005-01-01")],
        }
    )
    existing.to_parquet(processed_path, index=False)

    monkeypatch.setattr(ingest, "DATA_DIR", data_dir)
    monkeypatch.setattr(ingest, "PROCESSED_DATA_PATH", processed_path)

    ingest.process_raw_data()

    result = pd.read_parquet(processed_path)

    pd.testing.assert_frame_equal(result, existing)


# FEATURE ENGINEERING


def test_fixture_train_contains_required_model_columns():
    """
    The repository training fixture contains the columns consumed by the
    ALS, SVD, ensemble, and evaluation code.
    """
    train_path = FIXTURES_DIR / "train.parquet"

    assert train_path.exists()

    train = pd.read_parquet(train_path)

    required_columns = {
        "CustomerID",
        "Movie_ID",
        "Rating",
        "user_idx",
        "movie_idx",
    }

    assert required_columns.issubset(train.columns)
    assert len(train) > 0


def test_temporal_split_prevents_future_only_entities_from_entering_model_data(
    tmp_path,
    monkeypatch,
):
    """
    Regression test for temporal leakage.

    User/movie activity thresholds must be calculated using only the
    training period.
    """
    from src.data import build_features

    processed_path = tmp_path / "processed.parquet"
    movies_path = tmp_path / "movies.csv"
    train_path = tmp_path / "train.parquet"
    val_path = tmp_path / "val.parquet"
    test_path = tmp_path / "test.parquet"

    rows = []

    # Active user/movie with sufficient training history.
    for i in range(100):
        rows.append(
            {
                "MovieID": 1,
                "CustomerID": 1,
                "Rating": 4,
                "Date": pd.Timestamp("2000-01-01") + pd.Timedelta(days=i),
            }
        )

    # Second training-period entity helps establish a broader timeline.
    for i in range(100):
        rows.append(
            {
                "MovieID": 2,
                "CustomerID": 2,
                "Rating": 3,
                "Date": pd.Timestamp("2000-04-01") + pd.Timedelta(days=i),
            }
        )

    # Movie 3 has 50 ratings, but they are entirely in the future.
    # It must NOT pass the movie activity threshold.
    for i in range(50):
        rows.append(
            {
                "MovieID": 3,
                "CustomerID": 999 if i < 10 else 3,
                "Rating": 5,
                "Date": pd.Timestamp("2001-01-01") + pd.Timedelta(days=i),
            }
        )

    # Future ratings.
    for i in range(50):
        rows.append(
            {
                "MovieID": 2,
                "CustomerID": 2,
                "Rating": 4,
                "Date": pd.Timestamp("2001-03-01") + pd.Timedelta(days=i),
            }
        )

    pd.DataFrame(rows).to_parquet(processed_path, index=False)

    pd.DataFrame(
        {
            "Movie_ID": [1, 2, 3],
            "Title": [
                "Active Movie",
                "Second Movie",
                "Future Only Movie",
            ],
            "Genre": [
                "Drama",
                "Comedy",
                "Science Fiction",
            ],
        }
    ).to_csv(movies_path, index=False)

    monkeypatch.setattr(
        build_features,
        "PROCESSED_DATA_PATH",
        processed_path,
    )
    monkeypatch.setattr(
        build_features,
        "ENRICHED_MOVIES_PATH",
        movies_path,
    )
    monkeypatch.setattr(
        build_features,
        "TRAIN_DATA_PATH",
        train_path,
    )
    monkeypatch.setattr(
        build_features,
        "VAL_DATA_PATH",
        val_path,
    )
    monkeypatch.setattr(
        build_features,
        "TEST_DATA_PATH",
        test_path,
    )

    build_features.create_splits()

    train = pd.read_parquet(train_path)
    val = pd.read_parquet(val_path)
    test = pd.read_parquet(test_path)

    assert train["Date"].max() < val["Date"].min()
    assert val["Date"].max() < test["Date"].min()

    # Future-only user must never become active.
    assert 999 not in train["CustomerID"].unique()
    assert 999 not in val["CustomerID"].unique()
    assert 999 not in test["CustomerID"].unique()

    # Future-only movie must never become active.
    assert 3 not in train["Movie_ID"].unique()
    assert 3 not in val["Movie_ID"].unique()
    assert 3 not in test["Movie_ID"].unique()

    # Genuine active entities remain.
    assert 1 in train["CustomerID"].unique()
    assert 1 in train["Movie_ID"].unique()


# POPULARITY


def test_popularity_uses_training_data(tmp_path, monkeypatch):
    from src.models import popularity

    train_path = tmp_path / "train.parquet"
    artifact_path = tmp_path / "popularity.pkl"

    train = pd.DataFrame(
        {
            "Movie_ID": [1, 1, 1, 2, 2],
            "Rating": [5, 4, 3, 2, 2],
        }
    )
    train.to_parquet(train_path, index=False)

    monkeypatch.setattr(
        popularity,
        "TRAIN_DATA_PATH",
        train_path,
    )
    monkeypatch.setattr(
        popularity,
        "BASELINE_MODEL_PATH",
        artifact_path,
    )
    monkeypatch.setattr(
        popularity,
        "MIN_RATINGS_COUNT",
        2,
    )

    artifact = popularity.get_or_train_popularity(force_retrain=True)

    assert artifact["global_mean"] == pytest.approx(3.2)
    assert artifact["movie_avgs"][1] == pytest.approx(4.0)
    assert artifact["movie_avgs"][2] == pytest.approx(2.0)
    assert artifact_path.exists()


# ALS


def test_als_builds_training_matrix_from_train_ratings(
    tmp_path,
    monkeypatch,
):
    """
    ALS must build its fitted matrix from train ratings only.

    Validation/test files may determine matrix dimensions, but their ratings
    must never enter the sparse training matrix.
    """
    from src.models import als_model

    train_path = tmp_path / "train.parquet"
    val_path = tmp_path / "val.parquet"
    test_path = tmp_path / "test.parquet"
    model_path = tmp_path / "als_model.npz"

    pd.DataFrame(
        {
            "user_idx": [0, 0, 1],
            "movie_idx": [0, 1, 1],
            "Rating": [5, 4, 3],
        }
    ).to_parquet(train_path, index=False)

    pd.DataFrame(
        {
            "user_idx": [1],
            "movie_idx": [0],
        }
    ).to_parquet(val_path, index=False)

    pd.DataFrame(
        {
            "user_idx": [2],
            "movie_idx": [2],
        }
    ).to_parquet(test_path, index=False)

    monkeypatch.setattr(
        als_model,
        "TRAIN_DATA_PATH",
        train_path,
    )
    monkeypatch.setattr(
        als_model,
        "VAL_DATA_PATH",
        val_path,
    )
    monkeypatch.setattr(
        als_model,
        "TEST_DATA_PATH",
        test_path,
    )
    monkeypatch.setattr(
        als_model,
        "ALS_MODEL_PATH",
        model_path,
    )

    monkeypatch.setattr(
        als_model,
        "check_artifact_freshness",
        lambda *args, **kwargs: False,
    )

    monkeypatch.setattr(
        als_model,
        "save_artifact_metadata",
        lambda *args, **kwargs: None,
    )

    captured = {}

    class FakeALS:
        def __init__(self, *args, **kwargs):
            captured["args"] = args
            captured["params"] = kwargs

            self.user_factors = None
            self.item_factors = None

        def fit(self, matrix):
            captured["matrix"] = matrix.copy()

            self.user_factors = np.ones((matrix.shape[0], 2))
            self.item_factors = np.ones((matrix.shape[1], 2))

        def save(self, path):
            Path(path).touch()

    monkeypatch.setattr(
        als_model.implicit.cpu.als,
        "AlternatingLeastSquares",
        FakeALS,
    )

    model = als_model.get_or_train_als(force_retrain=True)

    matrix = captured["matrix"]

    # Matrix dimensions include indices appearing in validation/test.
    assert matrix.shape == (3, 3)

    # Only TRAIN ratings enter the matrix.
    assert matrix[0, 0] == 5
    assert matrix[0, 1] == 4
    assert matrix[1, 1] == 3

    # Validation/test-only entries remain zero.
    assert matrix[1, 0] == 0
    assert matrix[2, 2] == 0

    assert model.user_factors.shape == (3, 2)
    assert model.item_factors.shape == (3, 2)

    assert model_path.exists()


# SVD


def test_svd_prediction_equation_and_unknown_movie_handling():
    from src.models.svd_model import predict_batch

    class FakeTrainset:
        global_mean = 3.0

        def to_inner_uid(self, user_id):
            assert user_id == "10"
            return 0

        def to_inner_iid(self, movie_id):
            mapping = {
                "100": 0,
                "200": 1,
            }

            if movie_id not in mapping:
                raise ValueError(movie_id)

            return mapping[movie_id]

    model = SimpleNamespace(
        trainset=FakeTrainset(),
        pu=np.array(
            [
                [0.5, 1.0],
            ]
        ),
        bu=np.array([0.2]),
        qi=np.array(
            [
                [1.0, 0.0],
                [0.0, 2.0],
            ]
        ),
        bi=np.array([0.1, -0.2]),
    )

    predictions = predict_batch(
        model,
        user_id=10,
        movie_ids=[100, 200, 999],
    )

    expected = np.array(
        [
            3.0 + 0.2 + 0.1 + 0.5,
            3.0 + 0.2 - 0.2 + 2.0,
            3.0 + 0.2,
        ]
    )

    np.testing.assert_allclose(
        predictions,
        expected,
    )


def test_svd_prediction_output_length_matches_movie_ids():
    from src.models.svd_model import predict_batch

    class FakeTrainset:
        global_mean = 3.5

        def to_inner_uid(self, user_id):
            return 0

        def to_inner_iid(self, movie_id):
            if movie_id == "1":
                return 0
            raise ValueError(movie_id)

    model = SimpleNamespace(
        trainset=FakeTrainset(),
        pu=np.array([[1.0, 0.0]]),
        bu=np.array([0.0]),
        qi=np.array([[1.0, 0.0]]),
        bi=np.array([0.0]),
    )

    movie_ids = [1, 999, 1]

    predictions = predict_batch(
        model,
        user_id=123,
        movie_ids=movie_ids,
    )

    assert len(predictions) == len(movie_ids)
    assert np.isfinite(predictions).all()


# ENSEMBLE


def test_ensemble_optimizes_svd_vs_popularity_weight(
    tmp_path,
    monkeypatch,
):
    """
    The ensemble optimizer should choose SVD when SVD perfectly predicts
    the validation set and popularity does not.
    """
    from src.models import ensemble

    val_path = tmp_path / "val.parquet"
    ensemble_path = tmp_path / "ensemble.json"

    val = pd.DataFrame(
        {
            "CustomerID": [1, 2, 3],
            "Movie_ID": [10, 20, 30],
            "user_idx": [0, 1, 2],
            "movie_idx": [0, 1, 2],
            "Rating": [5, 4, 3],
        }
    )
    val.to_parquet(val_path, index=False)

    monkeypatch.setattr(
        ensemble,
        "VAL_DATA_PATH",
        val_path,
    )
    monkeypatch.setattr(
        ensemble,
        "ENSEMBLE_MODEL_PATH",
        ensemble_path,
    )

    monkeypatch.setattr(
        ensemble,
        "get_or_train_popularity",
        lambda: {
            "global_mean": 1.0,
            "movie_avgs": {
                10: 1.0,
                20: 1.0,
                30: 1.0,
            },
        },
    )

    class FakePrediction:
        def __init__(self, estimate):
            self.est = estimate

    class FakeSVD:
        def test(self, testset):
            return [
                FakePrediction(5.0),
                FakePrediction(4.0),
                FakePrediction(3.0),
            ]

    monkeypatch.setattr(
        ensemble,
        "get_or_train_svd",
        lambda: FakeSVD(),
    )

    artifact = ensemble.get_or_train_ensemble(force_retrain=True)

    assert artifact["svd_alpha"] == pytest.approx(1.0)
    assert artifact["popularity_alpha"] == pytest.approx(0.0)
    assert artifact["val_rmse"] == pytest.approx(0.0)

    assert ensemble_path.exists()


# METRICS / EVALUATION


def test_evaluation_als_prediction_clipping():
    raw_predictions = np.array([-5.0, 1.0, 3.5, 5.0, 10.0])

    clipped = np.clip(
        raw_predictions,
        1.0,
        5.0,
    )

    np.testing.assert_array_equal(
        clipped,
        np.array([1.0, 1.0, 3.5, 5.0, 5.0]),
    )


def test_evaluation_ensemble_prediction_clipping():
    pred_svd = np.array([0.0, 3.0, 6.0])
    pred_pop = np.array([0.0, 4.0, 6.0])

    alpha = 0.5

    predictions = np.clip(
        alpha * pred_svd + (1.0 - alpha) * pred_pop,
        1.0,
        5.0,
    )

    assert predictions.min() >= 1.0
    assert predictions.max() <= 5.0


def test_rmse_and_mae_are_zero_for_perfect_predictions():
    from sklearn.metrics import mean_absolute_error

    actual = np.array([1.0, 3.0, 5.0])
    predicted = np.array([1.0, 3.0, 5.0])

    rmse = np.sqrt(((actual - predicted) ** 2).mean())
    mae = mean_absolute_error(
        actual,
        predicted,
    )

    assert rmse == pytest.approx(0.0)
    assert mae == pytest.approx(0.0)


def test_rmse_and_mae_known_values():
    from sklearn.metrics import mean_absolute_error

    actual = np.array([1.0, 3.0, 5.0])
    predicted = np.array([2.0, 2.0, 4.0])

    rmse = np.sqrt(((actual - predicted) ** 2).mean())
    mae = mean_absolute_error(
        actual,
        predicted,
    )

    assert rmse == pytest.approx(1.0)
    assert mae == pytest.approx(1.0)
