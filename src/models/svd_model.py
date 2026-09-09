import gc
import ctypes
import logging
import pickle
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
from src.utils import check_artifact_freshness, save_artifact_metadata

logger = logging.getLogger(__name__)


SVD_PARAMS = {
    "n_factors": SVD_N_FACTORS,
    "n_epochs": SVD_N_EPOCHS,
    "lr_all": SVD_LR_ALL,
    "reg_all": SVD_REG_ALL,
    "random_state": RANDOM_STATE,
}


def get_or_train_svd(force_retrain=False):
    """Trains the Surprise SVD model or loads an existing one."""
    if (
        SVD_MODEL_PATH.exists()
        and not force_retrain
        and check_artifact_freshness(SVD_MODEL_PATH, SVD_PARAMS)
    ):
        print(f"Saved SVD model found at {SVD_MODEL_PATH}. Loading...")
        logger.info("Saved SVD model found; loading")
        with open(SVD_MODEL_PATH, "rb") as f:
            return pickle.load(f)

    print("Initiating SVD training pipeline...")
    logger.info("Initiating SVD training pipeline")
    print("Loading SVD training data...")
    logger.info("Loading SVD training data")

    train_df = pd.read_parquet(
        TRAIN_DATA_PATH, columns=["CustomerID", "Movie_ID", "Rating"]
    )

    train_df["CustomerID"] = train_df["CustomerID"].astype(str)
    train_df["Movie_ID"] = train_df["Movie_ID"].astype(str)

    print("Building SVD dataset...")
    logger.info("Building SVD dataset")
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(train_df[["CustomerID", "Movie_ID", "Rating"]], reader)

    trainset = data.build_full_trainset()

    del train_df, data
    gc.collect()

    print(f"Training SVD model " f"(factors={SVD_N_FACTORS}, epochs={SVD_N_EPOCHS})...")
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

    print("Saving SVD model artifact...")
    logger.info("Saving SVD model artifact")
    temp_model_path = SVD_MODEL_PATH.with_suffix(".tmp")

    with open(temp_model_path, "wb") as f:
        pickle.dump(svd_model, f, protocol=pickle.HIGHEST_PROTOCOL)

    temp_model_path.replace(SVD_MODEL_PATH)
    save_artifact_metadata(SVD_MODEL_PATH, SVD_PARAMS)
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

    raw_user_id = str(user_id)
    raw_movie_ids = np.asarray(
        [str(movie_id) for movie_id in movie_ids],
        dtype=object,
    )

    inner_uid = trainset.to_inner_uid(raw_user_id)

    user_factors = np.asarray(
        svd_model.pu[inner_uid],
        dtype=np.float64,
    )
    user_bias = float(svd_model.bu[inner_uid])

    predictions = np.full(
        len(raw_movie_ids),
        float(svd_model.trainset.global_mean),
        dtype=np.float64,
    )
    predictions += user_bias

    inner_iids = []

    for raw_movie_id in raw_movie_ids:
        try:
            inner_iids.append(trainset.to_inner_iid(raw_movie_id))
        except ValueError:
            inner_iids.append(None)

    known_mask = np.fromiter(
        (inner_iid is not None for inner_iid in inner_iids),
        dtype=bool,
        count=len(inner_iids),
    )

    if known_mask.any():
        known_inner_iids = np.asarray(
            [inner_iid for inner_iid in inner_iids if inner_iid is not None],
            dtype=np.intp,
        )

        item_biases = np.asarray(
            svd_model.bi[known_inner_iids],
            dtype=np.float64,
        )

        item_factors = np.asarray(
            svd_model.qi[known_inner_iids],
            dtype=np.float64,
        )

        dot_products = item_factors @ user_factors

        predictions[known_mask] += item_biases + dot_products

    return predictions
