import pandas as pd
from src.config import TRAIN_DATA_PATH, ENRICHED_MOVIES_PATH
from src.models.svd_model import get_or_train_svd

def generate_genre_recommendations(user_id, top_n=1):
    """Predicts ratings for unseen movies and returns top picks per genre."""
    # Load strictly the required columns for lookup
    train_df = pd.read_parquet(TRAIN_DATA_PATH, columns=["CustomerID", "Movie_ID"])
    movies_df = pd.read_csv(ENRICHED_MOVIES_PATH)
    svd_model = get_or_train_svd()
    
    if user_id not in train_df["CustomerID"].values:
        raise ValueError(f"User {user_id} not found in training dataset.")
    
    # Filter out movies the user has already rated
    seen = set(train_df[train_df["CustomerID"] == user_id]["Movie_ID"])
    unseen = movies_df[~movies_df["Movie_ID"].isin(seen)].copy()
    
    # Match the inner ID type required by the Surprise library
    id_type = type(next(iter(svd_model.trainset._raw2inner_id_users.keys())))
    safe_uid = id_type(user_id)
    
    unseen["predicted_rating"] = unseen["Movie_ID"].apply(
        lambda mid: svd_model.predict(safe_uid, id_type(mid)).est
    )
    
    # Explode comma-separated genres into distinct rows
    unseen_exp = unseen.copy()
    unseen_exp["Genre"] = unseen_exp["Genre"].astype(str).str.split(", ")
    unseen_exp = unseen_exp.explode("Genre")
    unseen_exp = unseen_exp[unseen_exp["Genre"].notna() & (unseen_exp["Genre"] != "Unknown")]
    
    best_per_genre = (
        unseen_exp.sort_values("predicted_rating", ascending=False)
        .groupby("Genre")[["Genre", "Title", "predicted_rating"]]
        .head(top_n)
    )
    
    result = best_per_genre.reset_index(drop=True)
    result.columns = ["Genre", "Movie Title", "Predicted Rating"]
    result["Predicted Rating"] = result["Predicted Rating"].round(2)
    
    return result