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


def test_enrichment_resumability(tmp_path, monkeypatch):
    """Test that genre enrichment can resume from partial state."""
    from src.data import enrich_genres

    # Create temporary paths
    movies_csv = tmp_path / "movie_titles.csv"
    enriched_csv = tmp_path / "movies_with_genres.csv"

    # Create test movie data
    movies_data = """Movie_ID,Year,Title
1,2000,Movie One
2,2001,Movie Two
3,2002,Movie Three
4,2003,Movie Four
"""
    movies_csv.write_text(movies_data, encoding="utf-8")

    # Mock TMDB API key
    monkeypatch.setattr(enrich_genres, "TMDB_API_KEY", "fake-token-for-testing")

    # Mock the get_movie_genres function to return deterministic results
    def mock_get_movie_genres(title, year, token):
        # Return known genres for our test movies
        genre_map = {
            "Movie One": "Action",
            "Movie Two": "Comedy",
            "Movie Three": "Drama",
            "Movie Four": "Unknown"  # Simulate API failure/unknown
        }
        return genre_map.get(title, "Unknown")

    monkeypatch.setattr(enrich_genres, "get_movie_genres", mock_get_movie_genres)

    # Mock the paths
    monkeypatch.setattr(enrich_genres, "MOVIE_TITLES_PATH", movies_csv)
    monkeypatch.setattr(enrich_genres, "ENRICHED_MOVIES_PATH", enriched_csv)

    # First enrichment run - process all movies
    df1 = enrich_genres.process_enrichment()
    assert len(df1) == 4
    assert enriched_csv.exists()

    # Check that we have results for all movies
    assert df1.loc[0, "Genre"] == "Action"   # Movie One
    assert df1.loc[1, "Genre"] == "Comedy"   # Movie Two
    assert df1.loc[2, "Genre"] == "Drama"    # Movie Three
    assert df1.loc[3, "Genre"] == "Unknown"  # Movie Four (mocked as unknown)

    # Second enrichment run - should resume and not re-process successfully enriched movies
    # But should retry the "Unknown" one (though our mock will still return Unknown)
    df2 = enrich_genres.process_enrichment()
    assert len(df2) == 4

    # Results should be identical (resume behavior)
    assert df1.equals(df2)

    # Now let's test true resumability by manually setting some genres to None
    # and see if it fills them in
    df3 = df2.copy()
    df3.loc[1, "Genre"] = None  # Reset Movie Two to None
    df3.to_csv(enriched_csv, index=False)

    # Third run should fill in the missing genre for Movie Two
    df3_result = enrich_genres.process_enrichment()
    assert df3_result.loc[1, "Genre"] == "Comedy"  # Should be filled in
    assert df3_result.loc[0, "Genre"] == "Action"  # Should remain unchanged
    assert df3_result.loc[2, "Genre"] == "Drama"   # Should remain unchanged
    assert df3_result.loc[3, "Genre"] == "Unknown" # Should remain unchanged
