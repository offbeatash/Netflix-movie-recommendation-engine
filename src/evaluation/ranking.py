"""Leakage-safe top-N evaluation utilities."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

import numpy as np
import pandas as pd


def precision_at_k(recommended: list[Any], relevant: set[Any], k: int) -> float:
    if k <= 0 or not recommended:
        return 0.0
    return sum(item in relevant for item in recommended[:k]) / min(k, len(recommended))


def recall_at_k(recommended: list[Any], relevant: set[Any], k: int) -> float:
    if not relevant or k <= 0:
        return 0.0
    return sum(item in relevant for item in recommended[:k]) / len(relevant)


def ndcg_at_k(recommended: list[Any], relevant: set[Any], k: int) -> float:
    if not relevant or k <= 0:
        return 0.0
    hits = [1.0 if item in relevant else 0.0 for item in recommended[:k]]
    dcg = sum(gain / np.log2(index + 2) for index, gain in enumerate(hits))
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / np.log2(index + 2) for index in range(ideal_hits))
    return float(dcg / idcg) if idcg else 0.0


def catalog_coverage(
    recommendations: Iterable[Iterable[Any]], catalog: set[Any]
) -> float:
    if not catalog:
        return 0.0
    recommended_items = {item for recs in recommendations for item in recs}
    return len(recommended_items & catalog) / len(catalog)


def intra_list_genre_diversity(
    movie_ids: list[Any], genres: dict[Any, set[str]]
) -> float:
    """Average pairwise genre dissimilarity; 1 means maximally different."""
    if len(movie_ids) < 2:
        return 0.0
    distances: list[float] = []
    for i, left_id in enumerate(movie_ids):
        left = genres.get(left_id, set())
        for right_id in movie_ids[i + 1 :]:
            right = genres.get(right_id, set())
            union = left | right
            distances.append(1.0 if not union else 1.0 - len(left & right) / len(union))
    return float(np.mean(distances)) if distances else 0.0


def evaluate_top_n(
    test_df: pd.DataFrame,
    train_df: pd.DataFrame,
    candidate_movie_ids: Iterable[Any],
    scorer: Callable[[Any, np.ndarray], np.ndarray],
    k: int = 10,
    relevant_threshold: float = 4.0,
    max_users: int = 1000,
    negative_ratio: int = 20,
    random_state: int = 42,
    genres: dict[Any, set[str]] | None = None,
) -> dict[str, float]:
    """Evaluate a scorer using only training-known candidates and past-item masking."""
    rng = np.random.default_rng(random_state)
    candidate = np.asarray(list(candidate_movie_ids))
    train_seen = train_df.groupby("CustomerID")["Movie_ID"].agg(set).to_dict()
    test_relevant = (
        test_df[test_df["Rating"] >= relevant_threshold]
        .groupby("CustomerID")["Movie_ID"]
        .agg(set)
        .to_dict()
    )
    users = sorted(test_relevant)[:max_users]

    precisions: list[float] = []
    recalls: list[float] = []
    ndcgs: list[float] = []
    all_recommendations: list[list[Any]] = []
    diversities: list[float] = []

    for user_id in users:
        relevant = test_relevant[user_id]
        seen = train_seen.get(user_id, set())
        relevant = relevant - seen
        if not relevant:
            continue
        pool = np.asarray([item for item in candidate if item not in seen])
        if len(pool) == 0:
            continue

        negatives = np.asarray(list(relevant))
        negative_candidates = np.asarray(
            [item for item in pool if item not in relevant]
        )
        sample_size = min(len(negative_candidates), max(k * negative_ratio, k))
        if sample_size > 0:
            negatives = np.concatenate(
                [
                    negatives,
                    rng.choice(negative_candidates, size=sample_size, replace=False),
                ]
            )
        else:
            negatives = negatives

        scores = np.asarray(scorer(user_id, negatives), dtype=float)
        order = np.argsort(-scores, kind="stable")[:k]
        recommended = negatives[order].tolist()
        all_recommendations.append(recommended)
        precisions.append(precision_at_k(recommended, relevant, k))
        recalls.append(recall_at_k(recommended, relevant, k))
        ndcgs.append(ndcg_at_k(recommended, relevant, k))
        if genres is not None:
            diversities.append(intra_list_genre_diversity(recommended, genres))

    result = {
        f"precision@{k}": float(np.mean(precisions)) if precisions else 0.0,
        f"recall@{k}": float(np.mean(recalls)) if recalls else 0.0,
        f"ndcg@{k}": float(np.mean(ndcgs)) if ndcgs else 0.0,
        "catalog_coverage": catalog_coverage(all_recommendations, set(candidate)),
    }
    if genres is not None:
        result["diversity"] = float(np.mean(diversities)) if diversities else 0.0
    return result
