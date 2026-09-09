import pandas as pd
import gc
from src.config import (
    PROCESSED_DATA_PATH,
    ENRICHED_MOVIES_PATH,
    TRAIN_DATA_PATH,
    VAL_DATA_PATH,
    TEST_DATA_PATH,
    TRAIN_SPLIT_QUANTILE,
    VAL_SPLIT_QUANTILE,
)


def create_splits():
    if TRAIN_DATA_PATH.exists() and VAL_DATA_PATH.exists() and TEST_DATA_PATH.exists():
        print(
            (
                "Train, validation, and test splits already exist. "
                "Skipping feature engineering phase."
            )
        )
        return

    print(f"Loading processed ratings from {PROCESSED_DATA_PATH}...")
    df = pd.read_parquet(PROCESSED_DATA_PATH)

    print(f"Loading enriched movies from {ENRICHED_MOVIES_PATH}...")
    movies = pd.read_csv(ENRICHED_MOVIES_PATH)

    movies["Title"] = movies["Title"].astype("category")
    movies["Genre"] = movies["Genre"].astype("category")

    print("Merging ratings with movie metadata...")
    df = df.merge(movies, left_on="MovieID", right_on="Movie_ID", how="left")

    df.drop(columns="Movie_ID", inplace=True)
    df.rename(columns={"MovieID": "Movie_ID"}, inplace=True)

    del movies
    gc.collect()

    print(
        (
            "Calculating time-based quantiles for splitting FIRST "
            "to prevent temporal leakage..."
        )
    )
    q80 = df["Date"].quantile(TRAIN_SPLIT_QUANTILE)
    q90 = df["Date"].quantile(VAL_SPLIT_QUANTILE)

    print(
        "Filtering inactive users and movies strictly based on the training period..."
    )
    min_user_rating = 10
    min_movie_rating = 50

    train_period_df = df[df["Date"] <= q80]
    user_counts = train_period_df.groupby("CustomerID").size()
    movie_counts = train_period_df.groupby("Movie_ID").size()

    active_users = user_counts[user_counts >= min_user_rating].index
    active_movies = movie_counts[movie_counts >= min_movie_rating].index

    df_model = df[
        df["CustomerID"].isin(active_users) & df["Movie_ID"].isin(active_movies)
    ].copy()

    print(f"Filtered dataset shape: {df_model.shape[0]:,} ratings")
    print(
        f"Users: {df_model['CustomerID'].nunique():,} | "
        f"Movies: {df_model['Movie_ID'].nunique():,}"
    )

    del df, train_period_df, user_counts, movie_counts
    gc.collect()

    print("Creating integer indices for sparse matrix mapping...")
    df_model["user_idx"] = pd.Categorical(df_model["CustomerID"]).codes
    df_model["movie_idx"] = pd.Categorical(df_model["Movie_ID"]).codes

    print(
        f"Physically splitting data ({TRAIN_SPLIT_QUANTILE * 100}% Train | "
        f"{(VAL_SPLIT_QUANTILE - TRAIN_SPLIT_QUANTILE) * 100}% Val | "
        f"{(1 - VAL_SPLIT_QUANTILE) * 100}% Test)..."
    )
    train_df = df_model[df_model["Date"] <= q80]
    val_df = df_model[(df_model["Date"] > q80) & (df_model["Date"] <= q90)]
    test_df = df_model[df_model["Date"] > q90]

    print("Saving splits to disk as Parquet files...")
    train_df.to_parquet(TRAIN_DATA_PATH, compression="snappy")
    val_df.to_parquet(VAL_DATA_PATH, compression="snappy")
    test_df.to_parquet(TEST_DATA_PATH, compression="snappy")

    print(
        f"Success! Train: {len(train_df):,} | Val: {len(val_df):,} | "
        f"Test: {len(test_df):,}"
    )

    del df_model, train_df, val_df, test_df
    gc.collect()


if __name__ == "__main__":
    create_splits()
