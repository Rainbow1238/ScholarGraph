"""大模型访问层的测试：用假的 HTTP 传输层拦截请求，不访问真实接口。"""

from __future__ import annotations

import json

import httpx
import pytest
from openai import OpenAI

from scholargraph.config import Settings
from scholargraph.llm import LLMOutputError, OpenAICompatibleLLM
from scholargraph.schemas import ResearchPlan

VALID_PLAN = json.dumps(
    {
        "scope": "纯文本 RAG",
        "sub_questions": [
            {"question": "什么是 RAG？", "search_queries": ["retrieval augmented generation"]}
        ],
    }
)


class FakeChatAPI:
    """模拟 /chat/completions 接口：按顺序返回预设内容，并记录收到的请求体。"""

    def __init__(self, replies: list[str]) -> None:
        self._replies = list(replies)
        self.requests: list[dict] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "created": 0,
                "model": "deepseek-flash",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": self._replies.pop(0)},
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            },
        )


def make_llm(api: FakeChatAPI, **settings_overrides) -> OpenAICompatibleLLM:
    settings = Settings(api_key="test-key", **settings_overrides)
    client = OpenAI(
        api_key=settings.api_key,
        base_url="http://fake-llm.test",
        http_client=httpx.Client(transport=httpx.MockTransport(api)),
        max_retries=0,
    )
    return OpenAICompatibleLLM(settings, client=client)


def test_structured_output_is_parsed_into_the_schema():
    api = FakeChatAPI([VALID_PLAN])
    plan = make_llm(api).complete_structured("系统提示", "用户问题", ResearchPlan)

    assert plan.sub_questions[0].search_queries == ["retrieval augmented generation"]
    request = api.requests[0]
    assert request["response_format"] == {"type": "json_object"}
    assert request["thinking"] == {"type": "disabled"}
    assert "JSON Schema" in request["messages"][0]["content"]


def test_invalid_output_is_retried_with_the_validation_error():
    api = FakeChatAPI(['{"wrong_field": 1}', VALID_PLAN])
    plan = make_llm(api).complete_structured("系统提示", "用户问题", ResearchPlan)

    assert len(plan.sub_questions) == 1
    assert len(api.requests) == 2
    assert "没有通过校验" in api.requests[1]["messages"][1]["content"]


def test_json_wrapped_in_a_markdown_code_fence_is_accepted():
    api = FakeChatAPI([f"```json\n{VALID_PLAN}\n```"])
    plan = make_llm(api).complete_structured("系统提示", "用户问题", ResearchPlan)

    assert len(plan.sub_questions) == 1
    assert len(api.requests) == 1


def test_output_violating_a_field_constraint_is_retried():
    without_queries = json.dumps(
        {
            "scope": "纯文本 RAG",
            "sub_questions": [{"question": "什么是 RAG？", "search_queries": []}],
        }
    )
    api = FakeChatAPI([without_queries, VALID_PLAN])
    plan = make_llm(api).complete_structured("系统提示", "用户问题", ResearchPlan)

    assert plan.sub_questions[0].search_queries
    assert len(api.requests) == 2


def test_gives_up_after_repeated_invalid_output():
    api = FakeChatAPI(["not json", "still not json", "never json"])
    with pytest.raises(LLMOutputError):
        make_llm(api).complete_structured("系统提示", "用户问题", ResearchPlan)
    assert len(api.requests) == 3


def test_thinking_parameter_is_omitted_when_not_configured():
    api = FakeChatAPI(["一段文本"])
    make_llm(api, thinking="").complete("系统提示", "用户问题")

    assert "thinking" not in api.requests[0]
    assert "response_format" not in api.requests[0]  # 自由文本请求不带 JSON 模式


def test_token_usage_is_accumulated():
    api = FakeChatAPI(["第一段", "第二段"])
    llm = make_llm(api)
    llm.complete("系统提示", "问题一")
    llm.complete("系统提示", "问题二")

    assert (llm.usage.calls, llm.usage.prompt_tokens, llm.usage.completion_tokens) == (2, 20, 10)
