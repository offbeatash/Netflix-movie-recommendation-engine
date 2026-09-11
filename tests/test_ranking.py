import numpy as np
import pandas as pd
import pytest

from src.evaluation.ranking import (
    catalog_coverage,
    evaluate_top_n,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)


def test_point_metrics():
    recommended = [1, 2, 3]
    relevant = {2, 3}
    assert precision_at_k(recommended, relevant, 2) == pytest.approx(0.5)
    assert recall_at_k(recommended, relevant, 2) == pytest.approx(0.5)
    assert ndcg_at_k([2, 3, 9], relevant, 2) == pytest.approx(1.0)


def test_catalog_coverage():
    assert catalog_coverage([[1, 2], [2, 3]], {1, 2, 3, 4}) == pytest.approx(0.75)


def test_evaluation_masks_training_items_and_is_deterministic():
    train = pd.DataFrame({"CustomerID": [1, 1, 2], "Movie_ID": [1, 2, 1]})
    test = pd.DataFrame(
        {"CustomerID": [1, 1, 2], "Movie_ID": [3, 4, 3], "Rating": [5, 4, 5]}
    )

    def scorer(_user, movie_ids):
        return np.asarray([10.0 - float(movie_id) for movie_id in movie_ids])

    first = evaluate_top_n(test, train, [1, 2, 3, 4], scorer, k=2, random_state=42)
    second = evaluate_top_n(test, train, [1, 2, 3, 4], scorer, k=2, random_state=42)
    assert first == second
    assert 0.0 <= first["precision@2"] <= 1.0
    assert 0.0 <= first["recall@2"] <= 1.0
    assert 0.0 <= first["ndcg@2"] <= 1.0
