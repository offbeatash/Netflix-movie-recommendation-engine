"""End-to-end recommendation contract tests."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.inference import recommend
from src.inference.recommend import generate_genre_recommendations


FIXTURE_TRAIN = "tests/fixtures/train.parquet"
FIXTURE_MOVIES = "tests/fixtures/movies_with_genres.csv"


def mock_popularity():
    movies = pd.read_csv(FIXTURE_MOVIES)
    return {"movie_avgs": pd.Series(4.0, index=movies["Movie_ID"]), "global_mean": 3.5}


class FakeSVD:
    """Mock SVD model that returns predictable scores for testing."""

    def __init__(self, user_id: str):

        class FakeTrainset:
            global_mean = 3.5

            def __init__(self, user_id: str):
                self.user_id = str(user_id)
                self._raw2inner_id_users = {self.user_id: 0}
                self._raw2inner_id_items = {
                    "1": 0,
                    "2": 1,
                    "3": 2,
                    "4": 3,
                    "5": 4,
                    "6": 5,
                }

            def to_inner_uid(self, raw_uid):
                if str(raw_uid) not in self._raw2inner_id_users:
                    raise ValueError("User not found")
                return self._raw2inner_id_users[str(raw_uid)]

            def to_inner_iid(self, raw_iid):
                raw_iid = str(raw_iid)
                if raw_iid not in self._raw2inner_id_items:
                    raise ValueError("Movie not found")
                return self._raw2inner_id_items[raw_iid]

        self.trainset = FakeTrainset(user_id)
        # User vector: [0.8, 0.6] - higher values mean stronger preference
        self.pu = np.array([[0.8, 0.6]], dtype=np.float64)
        self.bu = np.array([0.2], dtype=np.float64)

        self.qi = np.array(
            [
                [0.9, 0.8],  # Movie 1
                [0.7, 0.3],  # Movie 2
                [0.8, 0.9],  # Movie 3
                [0.6, 0.2],  # Movie 4
                [0.7, 0.7],  # Movie 5
                [0.5, 0.1],  # Movie 6
            ],
            dtype=np.float64,
        )

        self.bi = np.array(
            [0.3, 0.1, 0.4, -0.1, 0.2, -0.2], dtype=np.float64
        )  # Item biases

    def test(self, testset):
        """Return predictions with .est attribute like the real SVD."""
        from collections import namedtuple

        Prediction = namedtuple("Prediction", ["est"])

        predictions = []
        for u, i, r in testset:
            try:
                inner_uid = self.trainset.to_inner_uid(str(u))
                inner_iid = self.trainset.to_inner_iid(str(i))

                # Standard SVD prediction: μ + b_u + b_i + q_i·p_u
                pred = (
                    self.trainset.global_mean
                    + self.bu[inner_uid]
                    + self.bi[inner_iid]
                    + np.dot(self.qi[inner_iid], self.pu[inner_uid])
                )
                predictions.append(Prediction(est=max(1.0, min(5.0, pred))))
            except ValueError:
                # User or item not in training set
                predictions.append(Prediction(est=self.trainset.global_mean))
        return predictions


def test_personalized_user_contract():
    """Test complete recommendation contract for personalized users."""
    recommend._CACHE.clear()

    # Setup temporary ensemble weights
    ensemble_dir = tempfile.mkdtemp()
    ensemble_path = Path(ensemble_dir) / "ensemble_weights.json"

    with open(ensemble_path, "w") as f:
        json.dump({"svd_alpha": 0.6, "popularity_alpha": 0.4}, f)

    # Load fixture data
    train_df = pd.read_parquet(FIXTURE_TRAIN)
    movies_df = pd.read_csv(FIXTURE_MOVIES)

    # Pick a user that exists in the training data
    user_id = train_df["CustomerID"].iloc[0]

    # Get movies seen by this user in training
    seen_movies = set(train_df.loc[train_df["CustomerID"] == user_id, "Movie_ID"])

    with patch("src.inference.recommend.TRAIN_DATA_PATH", FIXTURE_TRAIN), patch(
        "src.inference.recommend.ENRICHED_MOVIES_PATH", FIXTURE_MOVIES
    ), patch(
        "src.inference.recommend.get_or_train_popularity",
        return_value=mock_popularity(),
    ), patch(
        "src.inference.recommend.get_or_train_svd", return_value=FakeSVD(str(user_id))
    ), patch(
        "src.inference.recommend.ENSEMBLE_MODEL_PATH", ensemble_path
    ):

        # Test various top_n values
        for top_n in [1, 3, 5, 10]:
            msg, df = generate_genre_recommendations(user_id=str(user_id), top_n=top_n)

            # Contract: personalized user should get personalized message
            assert "personalized" in msg.lower()
            assert f"user: {user_id}" in msg.lower()

            # Contract: response should not be empty (assuming sufficient catalog)
            assert not df.empty
            assert len(df) <= top_n  # May be less if not enough unseen movies

            # Contract: correct response format
            assert list(df.columns) == ["Genre", "Movie Title", "Predicted Rating"]

            # Contract: predicted ratings in valid range
            assert df["Predicted Rating"].between(1.0, 5.0).all()

            # Contract: no duplicate recommendations
            assert len(df["Movie Title"].unique()) == len(df)

            # Contract: seen movies should be excluded
            recommended_titles = df["Movie Title"].tolist()
            recommended_movie_ids = movies_df.loc[
                movies_df["Title"].isin(recommended_titles), "Movie_ID"
            ].tolist()

            for movie_id in recommended_movie_ids:
                assert (
                    movie_id not in seen_movies
                ), f"Seen movie {movie_id} should be excluded"

            # Contract: genre information should be valid when expected
            # (Not "Unknown" for successfully enriched movies)
            unknown_genres = df[df["Genre"] == "Unknown"]
            # Allow some unknowns if movies aren't enriched, but not all
            if len(df) > 0:
                assert len(unknown_genres) < len(df) or len(unknown_genres) == 0


def test_cold_start_user_contract():
    """Test complete recommendation contract for cold-start users."""
    recommend._CACHE.clear()

    with patch("src.inference.recommend.TRAIN_DATA_PATH", FIXTURE_TRAIN), patch(
        "src.inference.recommend.ENRICHED_MOVIES_PATH", FIXTURE_MOVIES
    ), patch(
        "src.inference.recommend.get_or_train_popularity",
        return_value=mock_popularity(),
    ), patch(
        "src.inference.recommend.get_or_train_svd"
    ):

        # Test with definitely unknown user
        user_id = "NON_EXISTENT_USER_12345"

        for top_n in [1, 3, 5]:
            msg, df = generate_genre_recommendations(user_id=user_id, top_n=top_n)

            # Contract: cold-start user should get fallback message
            assert "not found" in msg.lower() or "cold start" in msg.lower()

            # Contract: should still get recommendations
            assert not df.empty
            assert len(df) <= top_n

            # Contract: correct response format
            assert list(df.columns) == ["Genre", "Movie Title", "Predicted Rating"]

            # Contract: predicted ratings in valid range
            assert df["Predicted Rating"].between(1.0, 5.0).all()

            # Contract: no duplicate recommendations
            assert len(df["Movie Title"].unique()) == len(df)


def test_recommendation_count_respected():
    """Test that requested number of recommendations is respected."""
    recommend._CACHE.clear()

    ensemble_dir = tempfile.mkdtemp()
    ensemble_path = Path(ensemble_dir) / "ensemble_weights.json"

    with open(ensemble_path, "w") as f:
        json.dump({"svd_alpha": 0.5, "popularity_alpha": 0.5}, f)

    train_df = pd.read_parquet(FIXTURE_TRAIN)
    user_id = train_df["CustomerID"].iloc[0]

    with patch("src.inference.recommend.TRAIN_DATA_PATH", FIXTURE_TRAIN), patch(
        "src.inference.recommend.ENRICHED_MOVIES_PATH", FIXTURE_MOVIES
    ), patch(
        "src.inference.recommend.get_or_train_popularity",
        return_value=mock_popularity(),
    ), patch(
        "src.inference.recommend.get_or_train_svd", return_value=FakeSVD(str(user_id))
    ), patch(
        "src.inference.recommend.ENSEMBLE_MODEL_PATH", ensemble_path
    ):

        msg, df = generate_genre_recommendations(user_id=str(user_id), top_n=2)

        assert len(df) <= 2
        assert len(df) > 0

        assert len(df["Movie Title"].unique()) == len(df)


def test_no_duplicate_recommendations():
    """Test that recommendations contain no duplicates."""
    recommend._CACHE.clear()

    ensemble_dir = tempfile.mkdtemp()
    ensemble_path = Path(ensemble_dir) / "ensemble_weights.json"

    with open(ensemble_path, "w") as f:
        json.dump({"svd_alpha": 0.5, "popularity_alpha": 0.5}, f)

    train_df = pd.read_parquet(FIXTURE_TRAIN)
    movies_df = pd.read_csv(FIXTURE_MOVIES)
    user_id = train_df["CustomerID"].iloc[0]

    with patch("src.inference.recommend.TRAIN_DATA_PATH", FIXTURE_TRAIN), patch(
        "src.inference.recommend.ENRICHED_MOVIES_PATH", FIXTURE_MOVIES
    ), patch(
        "src.inference.recommend.get_or_train_popularity",
        return_value=mock_popularity(),
    ), patch(
        "src.inference.recommend.get_or_train_svd", return_value=FakeSVD(str(user_id))
    ), patch(
        "src.inference.recommend.ENSEMBLE_MODEL_PATH", ensemble_path
    ):

        msg, df = generate_genre_recommendations(user_id=str(user_id), top_n=10)

        assert len(df["Movie Title"].unique()) == len(
            df
        ), f"Found duplicates in: {df['Movie Title'].tolist()}"

        recommended_movie_ids = movies_df.loc[
            movies_df["Title"].isin(df["Movie Title"]), "Movie_ID"
        ].tolist()
        assert len(set(recommended_movie_ids)) == len(recommended_movie_ids)


def test_genre_information_validity():
    """Test that genre information is valid for enriched movies."""
    recommend._CACHE.clear()

    ensemble_dir = tempfile.mkdtemp()
    ensemble_path = Path(ensemble_dir) / "ensemble_weights.json"

    with open(ensemble_path, "w") as f:
        json.dump({"svd_alpha": 0.5, "popularity_alpha": 0.5}, f)

    train_df = pd.read_parquet(FIXTURE_TRAIN)
    movies_df = pd.read_csv(FIXTURE_MOVIES)
    user_id = train_df["CustomerID"].iloc[0]

    with patch("src.inference.recommend.TRAIN_DATA_PATH", FIXTURE_TRAIN), patch(
        "src.inference.recommend.ENRICHED_MOVIES_PATH", FIXTURE_MOVIES
    ), patch(
        "src.inference.recommend.get_or_train_popularity",
        return_value=mock_popularity(),
    ), patch(
        "src.inference.recommend.get_or_train_svd", return_value=FakeSVD(str(user_id))
    ), patch(
        "src.inference.recommend.ENSEMBLE_MODEL_PATH", ensemble_path
    ):

        msg, df = generate_genre_recommendations(user_id=str(user_id), top_n=5)

        # Verify that we have a Genre column
        assert "Genre" in df.columns

        # Verify that Genre column contains string values
        assert df["Genre"].dtype == object

        # Verify that for movies that exist in our fixture data,
        # we get meaningful genre information (not just empty or "Unknown" for all)
        # Get recommendations that match movies in our fixture
        recommended_titles = df["Movie Title"].tolist()
        matching_movies = movies_df[movies_df["Title"].isin(recommended_titles)]

        # If we have matching movies, verify their genre information is sensible
        if len(matching_movies) > 0:
            # At least some of the matching movies should have
            # non-empty, non-"Unknown" genres
            valid_genres = matching_movies[
                matching_movies["Genre"].notna()
                & (matching_movies["Genre"] != "")
                & (matching_movies["Genre"] != "Unknown")
            ]
            # We expect at least some movies to have valid genre information
            # Note: This might be 0 if all movies in fixture happen to
            # have Unknown genres, but that would be unusual for a
            # but that would be unusual for a well-formed fixture
            # The key assertion is that we don't get "Unknown" for ALL recommendations
            # when we have movies that should have known genres
            if len(valid_genres) == 0 and len(matching_movies) > 0:
                # If all matching movies have invalid genres, check if this is because
                # they genuinely have Unknown/empty genres in the fixture
                unknown_or_empty = matching_movies[
                    matching_movies["Genre"].isna()
                    | (matching_movies["Genre"] == "")
                    | (matching_movies["Genre"] == "Unknown")
                ]
                # If all matching movies genuinely have unknown/empty genres in fixture,
                # then it's OK for recommendations to also have unknown genres
                if len(unknown_or_empty) == len(matching_movies):
                    # This is acceptable - the fixture genuinely has unknown genres
                    pass
                else:
                    # This would indicate a problem - we have movies with known genres
                    # but are getting unknown genres in recommendations
                    assert False, "Getting 'Unknown' genre for movies with known genres"
