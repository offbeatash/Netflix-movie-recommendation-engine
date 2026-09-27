from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

TRAIN_PATH = PROJECT_ROOT / "data" / "train.parquet"
OUTPUT_PATH = PROJECT_ROOT / "artifacts" / "user_seen_movies.npz"


def main():
    print("Loading training interactions...")

    df = pd.read_parquet(
        TRAIN_PATH,
        columns=["CustomerID", "Movie_ID"],
    )

    print(f"Rows loaded: {len(df):,}")
    print(f"Users: {df['CustomerID'].nunique():,}")
    print(f"Movies: {df['Movie_ID'].nunique():,}")

    # Sort once so each user's movie IDs occupy a contiguous range.
    df = df.sort_values(
        ["CustomerID", "Movie_ID"],
        kind="mergesort",
    )

    # Remove duplicate user/movie interactions.
    df = df.drop_duplicates(
        subset=["CustomerID", "Movie_ID"],
        ignore_index=True,
    )

    customer_ids = df["CustomerID"].to_numpy(dtype=np.int32)
    movie_ids = df["Movie_ID"].to_numpy(dtype=np.uint16)

    unique_users, first_indices = np.unique(
        customer_ids,
        return_index=True,
    )

    # Build CSR-style offsets.
    offsets = np.empty(
        len(unique_users) + 1,
        dtype=np.int64,
    )

    offsets[:-1] = first_indices
    offsets[-1] = len(movie_ids)

    np.savez_compressed(
        OUTPUT_PATH,
        user_ids=unique_users.astype(np.int32),
        offsets=offsets,
        movie_ids=movie_ids,
    )

    size_mb = OUTPUT_PATH.stat().st_size / 1024**2

    print()
    print("Deployment history created successfully.")
    print(f"Unique users: {len(unique_users):,}")
    print(f"Unique interactions: {len(movie_ids):,}")
    print(f"Artifact: {OUTPUT_PATH}")
    print(f"Size: {size_mb:.2f} MB")


if __name__ == "__main__":
    main()
