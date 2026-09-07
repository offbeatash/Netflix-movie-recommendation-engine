import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
import gc
import pandas as pd
import implicit
from scipy.sparse import csr_matrix
from src.config import (
    TRAIN_DATA_PATH,
    VAL_DATA_PATH,
    TEST_DATA_PATH,
    ALS_MODEL_PATH,
    RANDOM_STATE,
    ALS_FACTORS,
    ALS_ITERATIONS,
    ALS_REGULARIZATION,
)
from src.utils import check_artifact_freshness, save_artifact_metadata


ALS_PARAMS = {
    "factors": ALS_FACTORS,
    "iterations": ALS_ITERATIONS,
    "regularization": ALS_REGULARIZATION,
    "random_state": RANDOM_STATE,
}

def get_or_train_als(force_retrain=False):
    """Trains the implicit ALS matrix factorization model or loads an existing one."""
    if (
        ALS_MODEL_PATH.exists()
        and not force_retrain
        and check_artifact_freshness(ALS_MODEL_PATH, ALS_PARAMS)
    ):
        print(f"Saved ALS model found at {ALS_MODEL_PATH}. Loading...")
        return implicit.cpu.als.AlternatingLeastSquares.load(str(ALS_MODEL_PATH))

    print("Initiating ALS training pipeline...")
    
    train_df = pd.read_parquet(
        TRAIN_DATA_PATH, 
        columns=["user_idx", "movie_idx", "Rating"]
    )
    val_df = pd.read_parquet(VAL_DATA_PATH, columns=["user_idx", "movie_idx"])
    test_df = pd.read_parquet(TEST_DATA_PATH, columns=["user_idx", "movie_idx"])
    
    print("Building sparse CSR matrix...")
    n_users = max(
        train_df["user_idx"].max(),
        val_df["user_idx"].max(),
        test_df["user_idx"].max()
    ) + 1
    n_movies = max(
        train_df["movie_idx"].max(),
        val_df["movie_idx"].max(),
        test_df["movie_idx"].max()
    ) + 1
    
    sparse_train = csr_matrix(
        (train_df["Rating"].values, (train_df["user_idx"].values, train_df["movie_idx"].values)),
        shape=(n_users, n_movies)
    )
    
    user_item_matrix = sparse_train.tocsr().astype("float32")
    
    del train_df, val_df, test_df, sparse_train
    gc.collect()
    
    print("Training Implicit ALS model...")
    als_model = implicit.als.AlternatingLeastSquares(
        factors=ALS_FACTORS,
        iterations=ALS_ITERATIONS,
        regularization=ALS_REGULARIZATION,
        random_state=RANDOM_STATE
    )
    
    als_model.fit(user_item_matrix)

    als_model.user_factors[user_item_matrix.getnnz(axis=1) == 0] = 0
    als_model.item_factors[user_item_matrix.getnnz(axis=0) == 0] = 0
    
    print("Saving ALS model artifact...")
    als_model.save(str(ALS_MODEL_PATH))
    save_artifact_metadata(ALS_MODEL_PATH, ALS_PARAMS)
    print(f"ALS model secured at: {ALS_MODEL_PATH}")
    
    del user_item_matrix
    gc.collect()
    
    return als_model