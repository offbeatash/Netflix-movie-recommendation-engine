import os
import gc
import pandas as pd
import implicit
from scipy.sparse import csr_matrix
from src.config import TRAIN_DATA_PATH, ALS_MODEL_PATH, RANDOM_STATE

def get_or_train_als(force_retrain=False):
    """Trains the implicit ALS matrix factorization model or loads an existing one."""
    if ALS_MODEL_PATH.exists() and not force_retrain:
        print(f"Saved ALS model found at {ALS_MODEL_PATH}. Loading...")
        return implicit.cpu.als.AlternatingLeastSquares.load(str(ALS_MODEL_PATH))

    print("Initiating ALS training pipeline...")
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    
    train_df = pd.read_parquet(
        TRAIN_DATA_PATH, 
        columns=["user_idx", "movie_idx", "Rating"]
    )
    
    print("Building sparse CSR matrix...")
    n_users = train_df["user_idx"].max() + 1
    n_movies = train_df["movie_idx"].max() + 1
    
    sparse_train = csr_matrix(
        (train_df["Rating"].values, (train_df["user_idx"].values, train_df["movie_idx"].values)),
        shape=(n_users, n_movies)
    )
    
    user_item_matrix = sparse_train.tocsr().astype("float32")
    
    del train_df, sparse_train
    gc.collect()
    
    print("Training Implicit ALS model...")
    als_model = implicit.als.AlternatingLeastSquares(
        factors=50,
        iterations=50,
        regularization=0.1,
        random_state=RANDOM_STATE
    )
    
    als_model.fit(user_item_matrix)
    
    print("Saving ALS model artifact...")
    als_model.save(str(ALS_MODEL_PATH))
    print(f"ALS model secured at: {ALS_MODEL_PATH}")
    
    # Flush memory
    del user_item_matrix
    gc.collect()
    
    return als_model