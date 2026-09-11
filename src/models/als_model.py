from src.utils import check_artifact_freshness, save_artifact_metadata
from src.config import (
    ALS_CONFIDENCE_ALPHA,
    ALS_FACTORS,
    ALS_ITERATIONS,
    ALS_MODEL_PATH,
    ALS_REGULARIZATION,
    RANDOM_STATE,
    TEST_DATA_PATH,
    TRAIN_DATA_PATH,
    VAL_DATA_PATH,
)
from scipy.sparse import csr_matrix
import pandas as pd
import implicit
import gc
import logging
import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")


logger = logging.getLogger(__name__)

ALS_PARAMS = {
    "factors": ALS_FACTORS,
    "iterations": ALS_ITERATIONS,
    "regularization": ALS_REGULARIZATION,
    "confidence_alpha": ALS_CONFIDENCE_ALPHA,
    "random_state": RANDOM_STATE,
}


def get_or_train_als(force_retrain: bool = False):
    """Train ALS on binary implicit interactions derived from training ratings.

    ALS is an offline ranking comparison only; it is not used for explicit-rating
    RMSE/MAE claims or in the serving path.
    """
    if not force_retrain and check_artifact_freshness(
        ALS_MODEL_PATH, ALS_PARAMS, TRAIN_DATA_PATH
    ):
        return implicit.cpu.als.AlternatingLeastSquares.load(str(ALS_MODEL_PATH))

    train_df = pd.read_parquet(
        TRAIN_DATA_PATH, columns=["user_idx", "movie_idx", "Rating"]
    )
    dimension_frames = [train_df[["user_idx", "movie_idx"]]]
    for path in (VAL_DATA_PATH, TEST_DATA_PATH):
        if path.exists():
            dimension_frames.append(
                pd.read_parquet(path, columns=["user_idx", "movie_idx"])
            )
    all_indices = pd.concat(dimension_frames, ignore_index=True)
    n_users = int(all_indices["user_idx"].max()) + 1
    n_movies = int(all_indices["movie_idx"].max()) + 1
    confidence = (
        1.0 + ALS_CONFIDENCE_ALPHA * (train_df["Rating"].astype("float32") / 5.0)
    ).astype("float32")
    matrix = csr_matrix(
        (confidence, (train_df["user_idx"], train_df["movie_idx"])),
        shape=(n_users, n_movies),
        dtype="float32",
    )

    model = implicit.cpu.als.AlternatingLeastSquares(
        factors=ALS_FACTORS,
        iterations=ALS_ITERATIONS,
        regularization=ALS_REGULARIZATION,
        random_state=RANDOM_STATE,
    )
    model.fit(matrix)

    temp_path = ALS_MODEL_PATH.with_suffix(".tmp.npz")
    model.save(str(temp_path))
    temp_path.replace(ALS_MODEL_PATH)
    save_artifact_metadata(ALS_MODEL_PATH, ALS_PARAMS, TRAIN_DATA_PATH)
    del train_df, all_indices, dimension_frames, matrix
    gc.collect()
    return model
