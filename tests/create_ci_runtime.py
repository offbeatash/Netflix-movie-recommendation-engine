import json
import pickle
import shutil
from pathlib import Path

import pandas as pd
from surprise import Dataset, Reader, SVD

ROOT = Path(__file__).resolve().parent.parent
FIXTURES_DIR = ROOT / "tests" / "fixtures"
RUNTIME_DIR = ROOT / ".ci-runtime"

DATA_DIR = RUNTIME_DIR / "data"
ARTIFACTS_DIR = RUNTIME_DIR / "artifacts"

TRAIN_FIXTURE = FIXTURES_DIR / "train.parquet"
MOVIES_FIXTURE = FIXTURES_DIR / "movies_with_genres.csv"

# Must match src/config.py
RANDOM_STATE = 42
SVD_N_FACTORS = 50
SVD_N_EPOCHS = 20
SVD_LR_ALL = 0.005
SVD_REG_ALL = 0.04


def create_runtime():
    """Create a tiny self-contained runtime for Docker CI smoke testing."""

    if not TRAIN_FIXTURE.exists():
        raise FileNotFoundError(f"Missing CI fixture: {TRAIN_FIXTURE}")

    if not MOVIES_FIXTURE.exists():
        raise FileNotFoundError(f"Missing CI fixture: {MOVIES_FIXTURE}")

    if RUNTIME_DIR.exists():
        shutil.rmtree(RUNTIME_DIR)

    DATA_DIR.mkdir(parents=True)
    ARTIFACTS_DIR.mkdir(parents=True)

    train = pd.read_parquet(TRAIN_FIXTURE)

    required_columns = {
        "CustomerID",
        "Movie_ID",
        "Rating",
    }

    missing = required_columns - set(train.columns)

    if missing:
        raise ValueError(
            f"train.parquet is missing required columns: {sorted(missing)}"
        )

    # Keep the CI runtime small and deterministic.
    train = train[["CustomerID", "Movie_ID", "Rating"]].copy()

    train["CustomerID"] = train["CustomerID"].astype(str)
    train["Movie_ID"] = train["Movie_ID"].astype(str)
    train["Rating"] = train["Rating"].astype(float)

    smoke_user = train["CustomerID"].iloc[0]

    train.to_parquet(
        DATA_DIR / "train.parquet",
        index=False,
    )

    shutil.copy2(
        MOVIES_FIXTURE,
        DATA_DIR / "movies_with_genres.csv",
    )

    # ------------------------------------------------------------------
    # Popularity artifact
    # ------------------------------------------------------------------

    global_mean = float(train["Rating"].mean())

    popularity = train.groupby("Movie_ID").agg(
        avg_rating=("Rating", "mean"),
        count=("Rating", "count"),
    )

    movie_avgs = popularity["avg_rating"].to_dict()

    popularity_artifact = {
        "global_mean": global_mean,
        "movie_avgs": movie_avgs,
    }

    with open(
        ARTIFACTS_DIR / "popularity_model.pkl",
        "wb",
    ) as f:
        pickle.dump(
            popularity_artifact,
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    # ------------------------------------------------------------------
    # SVD artifact
    #
    # IMPORTANT:
    # These parameters intentionally match src/config.py.
    # This allows production get_or_train_svd() to recognize the fixture
    # artifact as fresh and load it without retraining.
    # ------------------------------------------------------------------

    reader = Reader(rating_scale=(1, 5))

    dataset = Dataset.load_from_df(
        train[["CustomerID", "Movie_ID", "Rating"]],
        reader,
    )

    trainset = dataset.build_full_trainset()

    svd = SVD(
        n_factors=SVD_N_FACTORS,
        n_epochs=SVD_N_EPOCHS,
        lr_all=SVD_LR_ALL,
        reg_all=SVD_REG_ALL,
        random_state=RANDOM_STATE,
    )

    svd.fit(trainset)

    svd_model_path = ARTIFACTS_DIR / "svd_model.pkl"

    with open(
        svd_model_path,
        "wb",
    ) as f:
        pickle.dump(
            svd,
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    # Match src.models.svd_model.SVD_PARAMS exactly.
    svd_params = {
        "n_factors": SVD_N_FACTORS,
        "n_epochs": SVD_N_EPOCHS,
        "lr_all": SVD_LR_ALL,
        "reg_all": SVD_REG_ALL,
        "random_state": RANDOM_STATE,
    }

    with open(
        ARTIFACTS_DIR / "svd_model.pkl.metadata.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            svd_params,
            f,
            indent=2,
            sort_keys=True,
        )
        f.write("\n")

    # ------------------------------------------------------------------
    # Ensemble artifact
    # ------------------------------------------------------------------

    ensemble = {
        "svd_alpha": 0.5,
        "popularity_alpha": 0.5,
        "val_rmse": 0.0,
    }

    with open(
        ARTIFACTS_DIR / "ensemble_weights.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(ensemble, f)

    print("CI runtime created successfully.")
    print(f"Runtime directory: {RUNTIME_DIR}")
    print(f"Smoke-test user: {smoke_user}")


if __name__ == "__main__":
    create_runtime()
