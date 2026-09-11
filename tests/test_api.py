import asyncio

import httpx
import pandas as pd

from src.serving import fastapi_app


def request(method, path, **kwargs):
    async def make_request():
        transport = httpx.ASGITransport(app=fastapi_app.app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            return await client.request(method, path, **kwargs)

    return asyncio.run(make_request())


def test_health_endpoint():
    response = request("GET", "/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_ready_endpoint_reports_missing_runtime_files(tmp_path, monkeypatch):
    missing_paths = tuple(
        (name, tmp_path / name) for name in ("train", "movies", "popularity", "svd")
    )
    monkeypatch.setattr(fastapi_app, "REQUIRED_RUNTIME_PATHS", missing_paths)

    response = request("GET", "/ready")

    assert response.status_code == 503
    assert response.json()["status"] == "not_ready"


def test_ready_endpoint_reports_available_runtime_files(tmp_path, monkeypatch):
    runtime_paths = []
    for name in ("train", "movies", "popularity", "svd"):
        path = tmp_path / name
        path.touch()
        runtime_paths.append((name, path))

    monkeypatch.setattr(fastapi_app, "REQUIRED_RUNTIME_PATHS", tuple(runtime_paths))

    response = request("GET", "/ready")

    assert response.status_code == 200
    assert response.json() == {"status": "ready"}


def test_metrics_record_successful_cold_start(monkeypatch):
    results = pd.DataFrame(
        {
            "Genre": ["Drama"],
            "Movie Title": ["Example"],
            "Predicted Rating": [4.0],
        }
    )
    monkeypatch.setattr(
        fastapi_app,
        "generate_genre_recommendations",
        lambda user_id, top_n: (
            "User not found. Showing global popular movies.",
            results,
        ),
    )

    recommendation = request(
        "POST",
        "/recommend",
        json={"user_id": "cold-start-user", "top_n": 1},
    )
    metrics = request("GET", "/metrics")

    assert recommendation.status_code == 200
    assert len(recommendation.json()["recommendations"]) == 1
    assert metrics.status_code == 200
    assert b"recommendation_requests_total" in metrics.content
    assert b"cold_start_requests_total" in metrics.content
    assert b"recommendations_returned_total" in metrics.content
    assert b"http_request_duration_seconds" in metrics.content


def test_metrics_record_personalized_and_errors(monkeypatch):
    results = pd.DataFrame(
        {
            "Genre": ["Comedy", "Drama"],
            "Movie Title": ["Example 1", "Example 2"],
            "Predicted Rating": [4.0, 3.5],
        }
    )
    monkeypatch.setattr(
        fastapi_app,
        "generate_genre_recommendations",
        lambda user_id, top_n: ("Showing personalized results.", results),
    )

    recommendation = request(
        "POST",
        "/recommend",
        json={"user_id": "known-user", "top_n": 1},
    )
    monkeypatch.setattr(
        fastapi_app,
        "generate_genre_recommendations",
        lambda user_id, top_n: (_ for _ in ()).throw(RuntimeError("test failure")),
    )
    failed = request(
        "POST",
        "/recommend",
        json={"user_id": "failing-user", "top_n": 1},
    )
    metrics = request("GET", "/metrics")

    assert recommendation.status_code == 200
    assert failed.status_code == 500
    assert b"personalized_recommendation_requests_total" in metrics.content
    assert b"http_errors_total" in metrics.content


def test_recommend_offloads_sync_inference(monkeypatch):
    results = pd.DataFrame(
        {"Genre": ["Drama"], "Movie Title": ["Example"], "Predicted Rating": [4.0]}
    )
    called = {"value": False}

    async def fake_threadpool(func, **kwargs):
        called["value"] = True
        return func(**kwargs)

    monkeypatch.setattr(fastapi_app, "run_in_threadpool", fake_threadpool)
    monkeypatch.setattr(
        fastapi_app,
        "generate_genre_recommendations",
        lambda user_id, top_n: ("Showing personalized results.", results),
    )
    response = request("POST", "/recommend", json={"user_id": "1", "top_n": 1})
    assert response.status_code == 200
    assert called["value"] is True


def test_api_key_protection(monkeypatch):
    monkeypatch.setattr(fastapi_app, "API_KEY", "secret")
    missing = request("POST", "/recommend", json={"user_id": "1", "top_n": 1})
    valid = request(
        "POST",
        "/recommend",
        headers={"X-API-Key": "secret"},
        json={"user_id": "1", "top_n": 1},
    )
    assert missing.status_code == 401
    assert valid.status_code != 401
    monkeypatch.setattr(fastapi_app, "API_KEY", None)
