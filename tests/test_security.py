from fastapi import HTTPException
from starlette.requests import Request

from src.serving.security import RateLimiter, require_api_key


def request_with_headers(headers):
    scope = {
        "type": "http",
        "method": "POST",
        "path": "/recommend",
        "headers": [
            (key.lower().encode(), value.encode()) for key, value in headers.items()
        ],
        "client": ("127.0.0.1", 1234),
    }
    return Request(scope)


def test_api_key_is_optional_when_unconfigured():
    require_api_key(request_with_headers({}), None)


def test_api_key_rejects_missing_or_invalid():
    try:
        require_api_key(request_with_headers({}), "secret")
        assert False
    except HTTPException as exc:
        assert exc.status_code == 401


def test_api_key_accepts_valid_key():
    require_api_key(request_with_headers({"X-API-Key": "secret"}), "secret")


def test_rate_limiter_enforces_window_limit():
    limiter = RateLimiter(2, 60)
    assert limiter.allow("client")
    assert limiter.allow("client")
    assert not limiter.allow("client")
    assert limiter.allow("other-client")
