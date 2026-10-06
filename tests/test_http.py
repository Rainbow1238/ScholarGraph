"""HTTP 小工具的测试：重试、错误转换、限速。"""

from __future__ import annotations

import time

import pytest
from http_fakes import FakeResponse, FakeSession, connection_error

from scholargraph.search import SearchError, http
from scholargraph.search.http import MinIntervalLimiter, get_json

URL = "https://api.example.org/search"


@pytest.fixture(autouse=True)
def no_retry_wait(monkeypatch):
    monkeypatch.setattr(http, "RETRY_WAIT_SECONDS", 0)


def test_successful_response_is_parsed():
    session = FakeSession(FakeResponse(200, {"results": []}))

    assert get_json(session, URL, params={"q": "x"}) == {"results": []}
    assert session.requests[0]["timeout"] == http.REQUEST_TIMEOUT_SECONDS  # 每个请求都带超时


def test_rate_limited_response_is_retried_once():
    session = FakeSession(FakeResponse(429, text="slow down"), FakeResponse(200, {"ok": True}))

    assert get_json(session, URL, params={}) == {"ok": True}
    assert len(session.requests) == 2


def test_persistent_rate_limit_becomes_a_search_error():
    session = FakeSession(FakeResponse(429, text="daily budget exceeded"))

    with pytest.raises(SearchError, match="HTTP 429"):
        get_json(session, URL, params={})
    assert len(session.requests) == 2  # 首次 + 重试 1 次


def test_client_error_is_not_retried():
    session = FakeSession(FakeResponse(400, text="invalid parameter"))

    with pytest.raises(SearchError, match="HTTP 400"):
        get_json(session, URL, params={})
    assert len(session.requests) == 1


def test_network_failure_becomes_a_search_error():
    with pytest.raises(SearchError):
        get_json(FakeSession(connection_error()), URL, params={})


@pytest.mark.parametrize(
    "response", [FakeResponse(200, None), FakeResponse(200, ["not", "an", "object"])]
)
def test_malformed_body_becomes_a_search_error(response):
    with pytest.raises(SearchError):
        get_json(FakeSession(response), URL, params={})


def test_limiter_spaces_out_consecutive_requests():
    limiter = MinIntervalLimiter(0.05)
    started = time.monotonic()
    for _ in range(3):
        limiter.wait()

    # 3 次请求之间有 2 个间隔；Windows 的计时精度约 15 毫秒，留一点余量
    assert time.monotonic() - started >= 0.09
