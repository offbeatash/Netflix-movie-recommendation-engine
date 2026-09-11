import json
import pandas as pd
from unittest.mock import patch

from src.inference import recommend

FIXTURE_TRAIN = "tests/fixtures/train.parquet"
FIXTURE_MOVIES = "tests/fixtures/movies_with_genres.csv"


class FakeSVD:
    def __init__(self, user_id):
        import numpy as np

        class FakeTrainset:
            global_mean = 3.5

            def __init__(self, user_id):
                self.user_id = str(user_id)

                self._raw2inner_id_users = {self.user_id: 0}

                self._raw2inner_id_items = {
                    "1": 0,
                    "2": 1,
                    "3": 2,
                    "4": 3,
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

        # One user, four movies.
        self.pu = np.array(
            [[1.0, 0.0]],
            dtype=np.float64,
        )

        self.bu = np.array(
            [0.5],
            dtype=np.float64,
        )

        self.qi = np.array(
            [
                [1.0, 0.0],
                [0.5, 0.0],
                [0.0, 1.0],
                [0.2, 0.0],
            ],
            dtype=np.float64,
        )

        self.bi = np.array(
            [0.5, 0.2, 0.1, 0.3],
            dtype=np.float64,
        )


def mock_popularity():
    movies = pd.read_csv(FIXTURE_MOVIES)

    return {"movie_avgs": pd.Series(4.0, index=movies["Movie_ID"]), "global_mean": 3.5}


def test_known_user_recommendation(tmp_path):
    recommend._CACHE.clear()
    ensemble_path = tmp_path / "ensemble_weights.json"

    ensemble_path.write_text(
        json.dumps(
            {
                "svd_alpha": 0.7,
                "popularity_alpha": 0.3,
            }
        )
    )

    train_df = pd.read_parquet(FIXTURE_TRAIN)
    movies_df = pd.read_csv(FIXTURE_MOVIES)

    user_id = train_df["CustomerID"].iloc[0]

    seen_movie = train_df.loc[train_df["CustomerID"] == user_id, "Movie_ID"].iloc[0]

    with patch("src.inference.recommend.TRAIN_DATA_PATH", FIXTURE_TRAIN), patch(
        "src.inference.recommend.ENRICHED_MOVIES_PATH", FIXTURE_MOVIES
    ), patch(
        "src.inference.recommend.get_or_train_popularity",
        return_value=mock_popularity(),
    ), patch(
        "src.inference.recommend.get_or_train_svd", return_value=FakeSVD(user_id)
    ), patch(
        "src.inference.recommend.ENSEMBLE_MODEL_PATH", ensemble_path
    ):

        msg, df = recommend.generate_genre_recommendations(user_id=user_id, top_n=1)

    recommended_ids = movies_df.loc[
        movies_df["Title"].isin(df["Movie Title"]), "Movie_ID"
    ]

    assert "personalized" in msg.lower()
    assert not df.empty

    assert list(df.columns) == ["Genre", "Movie Title", "Predicted Rating"]

    assert df["Predicted Rating"].between(1.0, 5.0).all()

    assert seen_movie not in recommended_ids.values
