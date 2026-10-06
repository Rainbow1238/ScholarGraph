"""HTTP 检索源测试用的替身：假的 requests 会话和响应。"""

from __future__ import annotations

import requests


class FakeResponse:
    def __init__(self, status_code: int = 200, payload: object = None, text: str = "") -> None:
        self.status_code = status_code
        self._payload = payload
        self.text = text

    def json(self):
        if self._payload is None:
            raise ValueError("not json")
        return self._payload


class FakeSession:
    """按顺序返回预设的响应（或抛出预设的异常），并记录每次请求的参数。"""

    def __init__(self, *responses: FakeResponse | Exception) -> None:
        self._responses = list(responses)
        self.requests: list[dict] = []

    def get(self, url, *, params=None, headers=None, timeout=None):
        self.requests.append({"url": url, "params": params, "headers": headers, "timeout": timeout})
        response = self._responses.pop(0) if len(self._responses) > 1 else self._responses[0]
        if isinstance(response, Exception):
            raise response
        return response


def connection_error() -> Exception:
    return requests.ConnectionError("模拟的网络故障")
