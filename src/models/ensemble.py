import gc
import json

import numpy as np
import pandas as pd

from src.config import BASELINE_MODEL_PATH, ENSEMBLE_MODEL_PATH, SVD_MODEL_PATH, VAL_DATA_PATH
from src.models.popularity import get_or_train_popularity
from src.models.svd_model import get_or_train_svd
from src.utils import check_artifact_freshness, save_artifact_metadata
from src.versioning import _hash_files
from pathlib import Path


def get_or_train_ensemble(force_retrain: bool = False):
    """Tune the SVD/rating-popularity blend on validation data only."""
    params = {"purpose": "validation_rating_blend", "grid_size": 101}
    if not force_retrain and check_artifact_freshness(
        ENSEMBLE_MODEL_PATH, params, [VAL_DATA_PATH, BASELINE_MODEL_PATH, SVD_MODEL_PATH]
    ):
        return json.loads(ENSEMBLE_MODEL_PATH.read_text(encoding="utf-8"))

    val_df = pd.read_parquet(
        VAL_DATA_PATH, columns=["CustomerID", "Movie_ID", "Rating"]
    )
    actual = val_df["Rating"].to_numpy(dtype=float)
    popularity = get_or_train_popularity()
    pred_pop = (
        val_df["Movie_ID"]
        .map(popularity["movie_avgs"])
        .fillna(popularity["global_mean"])
        .to_numpy()
    )

    svd = get_or_train_svd()
    pred_svd = np.asarray(
        [
            prediction.est
            for prediction in svd.test(
                list(
                    zip(
                        val_df["CustomerID"].astype(str),
                        val_df["Movie_ID"].astype(str),
                        val_df["Rating"],
                    )
                )
            )
        ],
        dtype=np.float32,
    )
    del svd
    gc.collect()

    best_alpha = 0.0
    best_rmse = float("inf")
    for alpha in np.linspace(0, 1, 101):
        prediction = np.clip(alpha * pred_svd + (1.0 - alpha) * pred_pop, 1.0, 5.0)
        rmse = float(np.sqrt(np.mean((actual - prediction) ** 2)))
        if rmse < best_rmse:
            best_alpha, best_rmse = float(alpha), rmse

    artifact = {
        "svd_alpha": best_alpha,
        "popularity_alpha": 1.0 - best_alpha,
        "val_rmse": best_rmse,
    }
    temp_path = ENSEMBLE_MODEL_PATH.with_name(".ensemble_weights.tmp")
    temp_path.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    temp_path.replace(ENSEMBLE_MODEL_PATH)
    # Define source files that affect the ensemble model artifact
    ensemble_source_paths = [
        Path("src/models/ensemble.py"),
        Path("src/models/popularity.py"),
        Path("src/models/svd_model.py"),
        Path("src/utils.py"),
        Path("src/config.py"),
    ]
    save_artifact_metadata(ENSEMBLE_MODEL_PATH, params, [VAL_DATA_PATH, BASELINE_MODEL_PATH, SVD_MODEL_PATH], ensemble_source_paths)
    return artifact
