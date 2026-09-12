import gc
import ctypes
import logging
import pickle
from pathlib import Path

import pandas as pd
from surprise import SVD, Dataset, Reader

from src.config import (
    TRAIN_DATA_PATH,
    SVD_MODEL_PATH,
    RANDOM_STATE,
    SVD_N_FACTORS,
    SVD_N_EPOCHS,
    SVD_LR_ALL,
    SVD_REG_ALL,
)
from src.utils import save_artifact_metadata

logger = logging.getLogger(__name__)


SVD_PARAMS = {
    "n_factors": SVD_N_FACTORS,
    "n_epochs": SVD_N_EPOCHS,
    "lr_all": SVD_LR_ALL,
    "reg_all": SVD_REG_ALL,
    "random_state": RANDOM_STATE,
}


# Source files used when creating a new SVD artifact.
SVD_SOURCE_PATHS = [
    Path("src/models/svd_model.py"),
    Path("src/utils.py"),
    Path("src/config.py"),
]


def get_or_train_svd(force_retrain=False):
    """
    Load the existing SVD model artifact when available.

    Training only occurs when:
    1. The artifact does not exist, or
    2. force_retrain=True is explicitly requested.

    This prevents deployment/evaluation environments from unnecessarily
    retraining the large SVD model because of source-file timestamps,
    metadata differences, or regenerated Parquet files.
    """

    # LOAD EXISTING MODEL
    if SVD_MODEL_PATH.exists() and not force_retrain:
        print(f"Saved SVD model found at {SVD_MODEL_PATH}. Loading...")
        logger.info("Saved SVD model found; loading")

        try:
            with open(SVD_MODEL_PATH, "rb") as f:
                svd_model = pickle.load(f)

            print("SVD model loaded successfully.")
            logger.info("SVD model loaded successfully")

            return svd_model

        except Exception as e:
            logger.error("Failed to load saved SVD model: %s", e)
            raise RuntimeError(
                f"Failed to load existing SVD model from "
                f"{SVD_MODEL_PATH}: {e}"
            ) from e

    # TRAIN NEW MODEL
    if force_retrain:
        print("Forced SVD retraining requested.")
        logger.info("Forced SVD retraining requested")
    else:
        print("No saved SVD model found. Starting training pipeline.")
        logger.info("No saved SVD model found; starting training pipeline")

    print("Initiating SVD training pipeline...")
    logger.info("Initiating SVD training pipeline")

    print("Loading SVD training data...")
    logger.info("Loading SVD training data")

    if not TRAIN_DATA_PATH.exists():
        raise FileNotFoundError(
            f"SVD training data not found at {TRAIN_DATA_PATH}. "
            "Ensure train.parquet is available before training."
        )

    train_df = pd.read_parquet(
        TRAIN_DATA_PATH,
        columns=["CustomerID", "Movie_ID", "Rating"],
    )

    train_df["CustomerID"] = train_df["CustomerID"].astype(str)
    train_df["Movie_ID"] = train_df["Movie_ID"].astype(str)

    print("Building SVD dataset...")
    logger.info("Building SVD dataset")

    reader = Reader(rating_scale=(1, 5))

    data = Dataset.load_from_df(
        train_df[["CustomerID", "Movie_ID", "Rating"]],
        reader,
    )

    trainset = data.build_full_trainset()

    del train_df, data
    gc.collect()

    print(
        f"Training SVD model "
        f"(factors={SVD_N_FACTORS}, epochs={SVD_N_EPOCHS})..."
    )
    logger.info("Training SVD model")

    svd_model = SVD(
        n_factors=SVD_N_FACTORS,
        n_epochs=SVD_N_EPOCHS,
        lr_all=SVD_LR_ALL,
        reg_all=SVD_REG_ALL,
        random_state=RANDOM_STATE,
    )

    svd_model.fit(trainset)

    print("SVD model trained successfully.")
    logger.info("SVD model trained successfully")

    print("Pruning raw SVD rating histories...")
    logger.info("Pruning raw SVD rating histories")

    if hasattr(svd_model, "trainset") and svd_model.trainset is not None:
        for u in list(svd_model.trainset.ur.keys()):
            svd_model.trainset.ur[u] = None

        for i in list(svd_model.trainset.ir.keys()):
            svd_model.trainset.ir[i] = None

    del trainset
    gc.collect()

    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)

        print("OS memory trim complete.")
        logger.info("OS memory trim complete")

    except Exception as e:
        print(f"OS memory trim skipped: {e}")
        logger.warning("OS memory trim skipped: %s", e)

    # SAVE MODEL
    print("Saving SVD model artifact...")
    logger.info("Saving SVD model artifact")

    SVD_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    temp_model_path = SVD_MODEL_PATH.with_suffix(".tmp")

    with open(temp_model_path, "wb") as f:
        pickle.dump(
            svd_model,
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    temp_model_path.replace(SVD_MODEL_PATH)

    # Save metadata only when a new model is trained.
    save_artifact_metadata(
        SVD_MODEL_PATH,
        SVD_PARAMS,
        TRAIN_DATA_PATH,
        SVD_SOURCE_PATHS,
    )

    print(f"SVD model artifact saved to {SVD_MODEL_PATH}.")
    logger.info("SVD model artifact saved")

    return svd_model


def predict_batch(svd_model, user_id, movie_ids):
    """
    Predict ratings for one user across many movies using NumPy.

    This reproduces Surprise SVD's prediction equation:

        mu + bu + bi + dot(qi, pu)

    without calling Surprise's Python-level predict() once per movie.

    The trained SVD model itself is unchanged, so this optimization does
    not alter training, RMSE, or model parameters.
    """
    import numpy as np

    trainset = svd_model.trainset

    movie_ids = np.asarray(movie_ids)

    if movie_ids.size == 0:
        return np.empty(0, dtype=float)

    required_attributes = (
        "pu",
        "qi",
        "bu",
        "bi",
        "trainset",
    )

    if not all(hasattr(svd_model, attr) for attr in required_attributes):
        return np.asarray(
            [
                svd_model.predict(
                    str(user_id),
                    str(movie_id),
                ).est
                for movie_id in movie_ids
            ],
            dtype=float,
        )

    raw_user_id = str(user_id)

    try:
        inner_uid = trainset.to_inner_uid(raw_user_id)

    except ValueError:
        global_mean = float(trainset.global_mean)

        return np.full(
            movie_ids.size,
            global_mean,
            dtype=float,
        )

    global_mean = float(trainset.global_mean)

    user_bias = float(
        svd_model.bu[inner_uid]
    )

    user_factors = np.asarray(
        svd_model.pu[inner_uid],
        dtype=np.float64,
    )

    inner_item_id_list = []

    for movie_id in movie_ids:
        try:
            inner_item_id_list.append(
                trainset.to_inner_iid(str(movie_id))
            )

        except ValueError:
            inner_item_id_list.append(-1)

    inner_item_ids = np.asarray(
        inner_item_id_list,
        dtype=np.int64,
    )

    known_mask = inner_item_ids >= 0

    predictions = np.full(
        movie_ids.size,
        global_mean + user_bias,
        dtype=np.float64,
    )

    if known_mask.any():
        known_item_ids = inner_item_ids[known_mask]

        item_biases = np.asarray(
            svd_model.bi[known_item_ids],
            dtype=np.float64,
        )

        item_factors = np.asarray(
            svd_model.qi[known_item_ids],
            dtype=np.float64,
        )

        dot_products = item_factors @ user_factors

        predictions[known_mask] = (
            global_mean
            + user_bias
            + item_biases
            + dot_products
        )

    return np.clip(
        predictions,
        1.0,
        5.0,
    )
