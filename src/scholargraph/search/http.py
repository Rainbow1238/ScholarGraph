"""HTTP 检索源共用的小工具：限速器、带重试的 JSON 请求、响应字段的解析。"""

from __future__ import annotations

import threading
import time

import requests

from scholargraph.search.base import SearchError

REQUEST_TIMEOUT_SECONDS = (10.0, 30.0)  # (建立连接, 等待响应)
RETRY_WAIT_SECONDS = 2.0
_TRANSIENT_STATUS_CODES = frozenset({429, 500, 502, 503, 504})  # 限流或服务端临时故障
USER_AGENT = "ScholarGraph (literature research agent)"


class MinIntervalLimiter:
    """保证相邻两次请求之间至少间隔一段时间（线程安全）。"""

    def __init__(self, min_interval_seconds: float) -> None:
        self._min_interval = min_interval_seconds
        self._lock = threading.Lock()
        self._last_request_at = 0.0

    def wait(self) -> None:
        with self._lock:
            remaining = self._last_request_at + self._min_interval - time.monotonic()
            if remaining > 0:
                time.sleep(remaining)
            self._last_request_at = time.monotonic()


def int_or_none(value: object) -> int | None:
    """响应里的数字字段可能缺失或类型不对，只在确实是整数时采用。"""
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def get_json(
    session: requests.Session,
    url: str,
    *,
    params: dict[str, str | int],
    headers: dict[str, str] | None = None,
    limiter: MinIntervalLimiter | None = None,
    retries: int = 1,
) -> dict:
    """发送 GET 请求并返回解析后的 JSON 对象；任何失败都统一转换成 `SearchError`。

    遇到限流或服务端临时故障时稍等片刻重试，其余错误（参数错误、鉴权失败等）立即报告。
    """
    request_headers = {"User-Agent": USER_AGENT, **(headers or {})}

    for attempt in range(retries + 1):
        if limiter is not None:
            limiter.wait()
        try:
            response = session.get(
                url, params=params, headers=request_headers, timeout=REQUEST_TIMEOUT_SECONDS
            )
        except requests.RequestException as error:
            raise SearchError(f"请求 {url} 失败：{error}") from error

        if response.status_code == 200:
            try:
                payload = response.json()
            except ValueError as error:
                raise SearchError(f"{url} 返回的不是合法的 JSON") from error
            if not isinstance(payload, dict):
                raise SearchError(f"{url} 返回的 JSON 不是对象")
            return payload

        is_last_attempt = attempt == retries
        if response.status_code not in _TRANSIENT_STATUS_CODES or is_last_attempt:
            raise SearchError(f"{url} 返回 HTTP {response.status_code}：{response.text[:200]}")
        time.sleep(RETRY_WAIT_SECONDS)

    raise AssertionError("unreachable")  # 循环内每条路径都会 return 或 raise
