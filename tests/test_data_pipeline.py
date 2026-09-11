import pandas as pd

from src.data import build_features


def test_temporal_split_and_train_only_activity_filtering(tmp_path, monkeypatch):
    """
    Regression test for temporal leakage prevention.

    Verifies that:
    1. Train data is strictly earlier than validation/test data.
    2. User activity is calculated only from the training period.
    3. Movie activity is calculated only from the training period.

    The synthetic dataset deliberately contains:
    - An active user with enough ratings in the training period.
    - A future-only user with 10 ratings. This user must NOT become active.
    - An active movie with >=50 training-period ratings.
    - A future-only movie with >=50 ratings after the training cutoff.
      This movie must NOT become active.
    """

    processed_data_path = tmp_path / "processed_netflix.parquet"
    enriched_movies_path = tmp_path / "movies_with_genres.csv"
    train_path = tmp_path / "train.parquet"
    val_path = tmp_path / "val.parquet"
    test_path = tmp_path / "test.parquet"

    # ------------------------------------------------------------------
    # Build a synthetic chronological dataset.
    #
    # 200 training-period ratings
    # 50 validation-period ratings
    # 50 test-period ratings
    # ------------------------------------------------------------------

    rows = []

    # Active user + active movie:
    # 100 ratings occur during the training period.
    for i in range(100):
        rows.append(
            {
                "MovieID": 1,
                "CustomerID": 1,
                "Rating": 4,
                "Date": pd.Timestamp("2000-01-01") + pd.Timedelta(days=i),
            }
        )

    # Additional training-period ratings to establish the temporal cutoff.
    for i in range(100):
        rows.append(
            {
                "MovieID": 2,
                "CustomerID": 2,
                "Rating": 3,
                "Date": pd.Timestamp("2000-04-01") + pd.Timedelta(days=i),
            }
        )

    # Future-only movie.
    #
    # It has 50 ratings after the training period, which would incorrectly
    # make it "active" if movie activity were calculated on the full dataset.
    for i in range(50):
        rows.append(
            {
                "MovieID": 3,
                "CustomerID": 999 if i < 10 else 3,
                "Rating": 5,
                "Date": pd.Timestamp("2001-01-01") + pd.Timedelta(days=i),
            }
        )

    # Additional future ratings so validation and test periods both exist.
    for i in range(50):
        rows.append(
            {
                "MovieID": 2,
                "CustomerID": 2,
                "Rating": 4,
                "Date": pd.Timestamp("2001-03-01") + pd.Timedelta(days=i),
            }
        )

    ratings = pd.DataFrame(rows)

    movies = pd.DataFrame(
        {
            "Movie_ID": [1, 2, 3],
            "Title": [
                "Active Movie",
                "Second Movie",
                "Future Only Movie",
            ],
            "Genre": [
                "Drama",
                "Comedy",
                "Science Fiction",
            ],
        }
    )

    ratings.to_parquet(processed_data_path, index=False)
    movies.to_csv(enriched_movies_path, index=False)

    # Replace the production paths with the small synthetic test files.
    monkeypatch.setattr(
        build_features,
        "PROCESSED_DATA_PATH",
        processed_data_path,
    )
    monkeypatch.setattr(
        build_features,
        "ENRICHED_MOVIES_PATH",
        enriched_movies_path,
    )
    monkeypatch.setattr(
        build_features,
        "TRAIN_DATA_PATH",
        train_path,
    )
    monkeypatch.setattr(
        build_features,
        "VAL_DATA_PATH",
        val_path,
    )
    monkeypatch.setattr(
        build_features,
        "TEST_DATA_PATH",
        test_path,
    )

    build_features.create_splits()

    train = pd.read_parquet(train_path)
    val = pd.read_parquet(val_path)
    test = pd.read_parquet(test_path)

    # ------------------------------------------------------------------
    # 1. Prove chronological separation.
    # ------------------------------------------------------------------

    assert train["Date"].max() <= val["Date"].min()
    assert val["Date"].max() <= test["Date"].min()

    assert train["Date"].max() < val["Date"].min()
    assert val["Date"].max() < test["Date"].min()

    # ------------------------------------------------------------------
    # 2. Prove user activity filtering uses training-period data only.
    #
    # Customer 999 has exactly 10 ratings, but ALL of them occur after
    # the training cutoff.
    #
    # If the old/leaky implementation counted the full dataset, this
    # customer would incorrectly become an active user.
    # ------------------------------------------------------------------

    assert 999 not in train["CustomerID"].unique()
    assert 999 not in val["CustomerID"].unique()
    assert 999 not in test["CustomerID"].unique()

    # Customer 1 has >=10 ratings during training and therefore remains.
    assert 1 in train["CustomerID"].unique()

    # ------------------------------------------------------------------
    # 3. Prove movie activity filtering uses training-period data only.
    #
    # Movie 3 has exactly 50 ratings, but ALL of them occur after the
    # training cutoff.
    #
    # If the old/leaky implementation counted the full dataset, this
    # movie would incorrectly become an active movie.
    # ------------------------------------------------------------------

    assert 3 not in train["Movie_ID"].unique()
    assert 3 not in val["Movie_ID"].unique()
    assert 3 not in test["Movie_ID"].unique()

    # Movie 1 has >=50 ratings during training and therefore remains.
    assert 1 in train["Movie_ID"].unique()
