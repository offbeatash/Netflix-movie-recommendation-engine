import pandas as pd

from src.data import build_features


def test_temporal_split_and_train_only_activity_filtering(tmp_path, monkeypatch):
    """Verify temporal leakage prevention and train-only activity filtering."""

    processed_data_path = tmp_path / "processed_netflix.parquet"
    enriched_movies_path = tmp_path / "movies_with_genres.csv"
    train_path = tmp_path / "train.parquet"
    val_path = tmp_path / "val.parquet"
    test_path = tmp_path / "test.parquet"

    rows = []

    for i in range(100):
        rows.append(
            {
                "MovieID": 1,
                "CustomerID": 1,
                "Rating": 4,
                "Date": pd.Timestamp("2000-01-01") + pd.Timedelta(days=i),
            }
        )

    for i in range(100):
        rows.append(
            {
                "MovieID": 2,
                "CustomerID": 2,
                "Rating": 3,
                "Date": pd.Timestamp("2000-04-01") + pd.Timedelta(days=i),
            }
        )

    # Future-only user/movie must not affect training activity filtering.
    for i in range(50):
        rows.append(
            {
                "MovieID": 3,
                "CustomerID": 999 if i < 10 else 3,
                "Rating": 5,
                "Date": pd.Timestamp("2001-01-01") + pd.Timedelta(days=i),
            }
        )

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

    # Temporal separation must be strict.
    assert train["Date"].max() < val["Date"].min()
    assert val["Date"].max() < test["Date"].min()

    # Future-only users must not become active.
    assert 999 not in train["CustomerID"].unique()
    assert 999 not in val["CustomerID"].unique()
    assert 999 not in test["CustomerID"].unique()
    assert 1 in train["CustomerID"].unique()

    # Future-only movies must not become active.
    assert 3 not in train["Movie_ID"].unique()
    assert 3 not in val["Movie_ID"].unique()
    assert 3 not in test["Movie_ID"].unique()
    assert 1 in train["Movie_ID"].unique()


def test_enrichment_resumability(tmp_path, monkeypatch):
    """Verify completed enrichment skips and partial enrichment resumes."""

    from src.data import enrich_genres

    movies_csv = tmp_path / "movie_titles.csv"
    enriched_csv = tmp_path / "movies_with_genres.csv"
    metadata_path = enriched_csv.with_suffix(".enrichment_metadata.json")

    movies_data = """Movie_ID,Year,Title
1,2000,Movie One
2,2001,Movie Two
3,2002,Movie Three
4,2003,Movie Four
"""
    movies_csv.write_text(movies_data, encoding="utf-8")

    monkeypatch.setattr(
        enrich_genres,
        "TMDB_API_KEY",
        "fake-token-for-testing",
    )

    calls = []

    def mock_get_movie_genres(title, year, token):
        calls.append(title)

        genre_map = {
            "Movie One": "Action",
            "Movie Two": "Comedy",
            "Movie Three": "Drama",
            "Movie Four": "Unknown",
        }

        return genre_map.get(title, "Unknown")

    monkeypatch.setattr(
        enrich_genres,
        "get_movie_genres",
        mock_get_movie_genres,
    )

    monkeypatch.setattr(
        enrich_genres,
        "MOVIE_TITLES_PATH",
        movies_csv,
    )
    monkeypatch.setattr(
        enrich_genres,
        "ENRICHED_MOVIES_PATH",
        enriched_csv,
    )
    monkeypatch.setattr(
        enrich_genres,
        "ENRICHMENT_METADATA_PATH",
        metadata_path,
    )

    df1 = enrich_genres.process_enrichment()

    assert len(df1) == 4
    assert df1.loc[0, "Genre"] == "Action"
    assert df1.loc[1, "Genre"] == "Comedy"
    assert df1.loc[2, "Genre"] == "Drama"
    assert df1.loc[3, "Genre"] == "Unknown"
    assert enriched_csv.exists()
    assert metadata_path.exists()
    assert len(calls) == 4

    # Completed enrichment must skip without calling the API.
    calls.clear()

    df2 = enrich_genres.process_enrichment()

    pd.testing.assert_frame_equal(
        df2.reset_index(drop=True),
        df1.reset_index(drop=True),
        check_dtype=False,
    )
    assert calls == []

    # Removing the marker simulates an incomplete enrichment state.
    df3 = df2.copy()
    df3.loc[1, "Genre"] = None
    df3.to_csv(enriched_csv, index=False)
    metadata_path.unlink()

    calls.clear()

    df3_result = enrich_genres.process_enrichment()

    assert df3_result.loc[1, "Genre"] == "Comedy"
    assert df3_result.loc[0, "Genre"] == "Action"
    assert df3_result.loc[2, "Genre"] == "Drama"
    assert df3_result.loc[3, "Genre"] == "Unknown"
    assert sorted(calls) == ["Movie Four", "Movie Two"]
