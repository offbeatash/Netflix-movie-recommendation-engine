import pandas as pd
from src.config import TRAIN_DATA_PATH, ENRICHED_MOVIES_PATH
from src.models.svd_model import get_or_train_svd
from src.models.popularity import get_or_train_popularity

def generate_genre_recommendations(user_id, top_n=1):
    """Predicts ratings for unseen movies and returns a status message alongside top picks per genre."""
    train_df = pd.read_parquet(TRAIN_DATA_PATH, columns=["CustomerID", "Movie_ID"])
    movies_df = pd.read_csv(ENRICHED_MOVIES_PATH)
    
    user_exists = user_id in train_df["CustomerID"].values
    
    # Explode genres early for both pathways
    movies_exp = movies_df.copy()
    movies_exp["Genre"] = movies_exp["Genre"].astype(str).str.split(", ")
    movies_exp = movies_exp.explode("Genre")
    movies_exp = movies_exp[movies_exp["Genre"].notna() & (movies_exp["Genre"] != "Unknown")]

    if not user_exists:
        # COLD START: Return the most popular movies globally
        status_msg = f"User '{user_id}' not found. Showing global popular movies (Cold Start Baseline)."
        popularity_artifact = get_or_train_popularity()
        movie_avgs = popularity_artifact["movie_avgs"]
        
        movies_exp["predicted_rating"] = movies_exp["Movie_ID"].map(movie_avgs).fillna(popularity_artifact["global_mean"])
        
        best_per_genre = (
            movies_exp.sort_values("predicted_rating", ascending=False)
            .groupby("Genre")[["Genre", "Title", "predicted_rating"]]
            .head(top_n)
        )
    else:
        # PERSONALIZED: SVD Predictions
        status_msg = f"Showing personalized results for user: {user_id}"
        svd_model = get_or_train_svd()
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