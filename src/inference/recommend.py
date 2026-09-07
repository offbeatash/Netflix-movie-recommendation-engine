import pandas as pd
from src.config import TRAIN_DATA_PATH, ENRICHED_MOVIES_PATH
from src.models.svd_model import get_or_train_svd
from src.models.popularity import get_or_train_popularity

#GLOBAL CACHE: Holds data in RAM across API calls
_CACHE = {}

def _load_artifacts():
    """Loads all models and data into memory exactly once at startup."""
    if not _CACHE:
        print("Initializing Global Cache for Inference API...")
        _CACHE["train_df"] = pd.read_parquet(TRAIN_DATA_PATH, columns=["CustomerID", "Movie_ID"])
        _CACHE["movies_df"] = pd.read_csv(ENRICHED_MOVIES_PATH)
        
        #Pre-explode genres once during startup to save CPU on every call
        movies_exp = _CACHE["movies_df"].copy()
        movies_exp["Genre"] = movies_exp["Genre"].astype(str).str.split(", ")
        movies_exp = movies_exp.explode("Genre")
        _CACHE["movies_exp"] = movies_exp[movies_exp["Genre"].notna() & (movies_exp["Genre"] != "Unknown")]
        
        _CACHE["popularity_artifact"] = get_or_train_popularity()
        _CACHE["svd_model"] = get_or_train_svd()
    return _CACHE

def generate_genre_recommendations(user_id, top_n=1):
    """Predicts ratings utilizing the pre-loaded global memory cache."""
    cache = _load_artifacts()
    
    train_df = cache["train_df"]
    movies_exp = cache["movies_exp"]
    popularity_artifact = cache["popularity_artifact"]
    svd_model = cache["svd_model"]
    
    user_exists = user_id in train_df["CustomerID"].values

    if not user_exists:
        status_msg = f"User '{user_id}' not found. Showing global popular movies (Cold Start Baseline)."
        movie_avgs = popularity_artifact["movie_avgs"]
        
        user_movies = movies_exp.copy()
        user_movies["predicted_rating"] = user_movies["Movie_ID"].map(movie_avgs).fillna(popularity_artifact["global_mean"])
        
        best_per_genre = (
            user_movies.sort_values("predicted_rating", ascending=False)
            .groupby("Genre")[["Genre", "Title", "predicted_rating"]]
            .head(top_n)
        )
    else:
        status_msg = f"Showing personalized results for user: {user_id}"
        seen = set(train_df[train_df["CustomerID"] == user_id]["Movie_ID"])
        unseen_exp = movies_exp[~movies_exp["Movie_ID"].isin(seen)].copy()
        
        id_type = type(next(iter(svd_model.trainset._raw2inner_id_users.keys())))
        safe_uid = id_type(str(user_id)) 
        
        unseen_exp["predicted_rating"] = unseen_exp["Movie_ID"].apply(
            lambda mid: svd_model.predict(safe_uid, id_type(str(mid))).est
        )
        
        best_per_genre = (
            unseen_exp.sort_values("predicted_rating", ascending=False)
            .groupby("Genre")[["Genre", "Title", "predicted_rating"]]
            .head(top_n)
        )
    
    result = best_per_genre.reset_index(drop=True)
    result.columns = ["Genre", "Movie Title", "Predicted Rating"]
    result["Predicted Rating"] = result["Predicted Rating"].round(2)
    
    return status_msg, result