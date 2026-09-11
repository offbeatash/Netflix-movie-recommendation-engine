import pandas as pd
import pytest
from unittest.mock import patch

from src.inference import recommend
from src.inference.recommend import generate_genre_recommendations

FIXTURE_TRAIN = "tests/fixtures/train.parquet"
FIXTURE_MOVIES = "tests/fixtures/movies_with_genres.csv"


@pytest.fixture(autouse=True)
def clear_recommendation_cache():
    recommend._CACHE.clear()


def mock_popularity():
    movies = pd.read_csv(FIXTURE_MOVIES)

    return {"movie_avgs": pd.Series(4.0, index=movies["Movie_ID"]), "global_mean": 3.5}


def test_cold_start_fallback():
    with patch("src.inference.recommend.TRAIN_DATA_PATH", FIXTURE_TRAIN), patch(
        "src.inference.recommend.ENRICHED_MOVIES_PATH", FIXTURE_MOVIES
    ), patch(
        "src.inference.recommend.get_or_train_popularity",
        return_value=mock_popularity(),
    ), patch(
        "src.inference.recommend.get_or_train_svd"
    ):

        msg, df = generate_genre_recommendations(
            user_id="NON_EXISTENT_USER_999", top_n=2
        )

    assert "not found" in msg.lower()
    assert not df.empty


def test_prediction_sanity_bounds():
    with patch("src.inference.recommend.TRAIN_DATA_PATH", FIXTURE_TRAIN), patch(
        "src.inference.recommend.ENRICHED_MOVIES_PATH", FIXTURE_MOVIES
    ), patch(
        "src.inference.recommend.get_or_train_popularity",
        return_value=mock_popularity(),
    ), patch(
        "src.inference.recommend.get_or_train_svd"
    ):

        _, df = generate_genre_recommendations(user_id="UNKNOWN_123", top_n=10)

    assert df["Predicted Rating"].min() >= 1.0
    assert df["Predicted Rating"].max() <= 5.0


def test_data_schema_types():
    with patch("src.inference.recommend.TRAIN_DATA_PATH", FIXTURE_TRAIN), patch(
        "src.inference.recommend.ENRICHED_MOVIES_PATH", FIXTURE_MOVIES
    ), patch(
        "src.inference.recommend.get_or_train_popularity",
        return_value=mock_popularity(),
    ), patch(
        "src.inference.recommend.get_or_train_svd"
    ):

        _, df = generate_genre_recommendations(user_id="UNKNOWN_123", top_n=1)

    assert list(df.columns) == ["Genre", "Movie Title", "Predicted Rating"]

    assert pd.api.types.is_string_dtype(df["Genre"])
    assert pd.api.types.is_string_dtype(df["Movie Title"])
    assert pd.api.types.is_numeric_dtype(df["Predicted Rating"])
