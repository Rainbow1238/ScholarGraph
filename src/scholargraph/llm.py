"""大模型访问层。

节点只依赖 `LLM` 这个协议，不关心背后是 DeepSeek、别的厂商还是测试里的假模型。
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from typing import Protocol, TypeVar

from openai import OpenAI
from pydantic import BaseModel, ValidationError

from scholargraph.config import Settings

ModelT = TypeVar("ModelT", bound=BaseModel)

REQUEST_TIMEOUT_SECONDS = 120.0
MAX_STRUCTURED_ATTEMPTS = 3


class LLMOutputError(RuntimeError):
    """模型的输出无法使用（为空，或多次重试后仍不符合 Schema）。"""


class LLM(Protocol):
    """节点眼中的大模型：只有两种能力，外加一份累计的用量统计。"""

    usage: TokenUsage

    def complete(self, system: str, user: str) -> str:
        """返回自由文本。"""
        ...

    def complete_structured(self, system: str, user: str, schema: type[ModelT]) -> ModelT:
        """返回一个通过 `schema` 校验的对象。"""
        ...


@dataclass
class TokenUsage:
    """累计 token 用量。并行的 Researcher 会同时写入，所以加锁。"""

    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)

    def record(self, prompt_tokens: int, completion_tokens: int) -> None:
        with self._lock:
            self.calls += 1
            self.prompt_tokens += prompt_tokens
            self.completion_tokens += completion_tokens

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class OpenAICompatibleLLM:
    """通过 OpenAI 兼容接口访问大模型（DeepSeek 等）。"""

    def __init__(self, settings: Settings, client: OpenAI | None = None) -> None:
        """`client` 仅供测试注入；正常使用时根据配置自动创建。"""
        self._client = client or OpenAI(
            api_key=settings.api_key,
            base_url=settings.base_url,
            timeout=REQUEST_TIMEOUT_SECONDS,
            max_retries=3,  # 限流、超时等瞬时错误由 SDK 自动退避重试
        )
        self._model = settings.model
        self._temperature = settings.temperature
        self._extra_body = {"thinking": {"type": settings.thinking}} if settings.thinking else None
        self.usage = TokenUsage()

    def complete(self, system: str, user: str) -> str:
        return self._chat(system, user, json_mode=False)

    def complete_structured(self, system: str, user: str, schema: type[ModelT]) -> ModelT:
        """用 JSON 模式拿到输出，再用 Pydantic 校验；不合格就带着错误信息重问。

        重问时发起的是一次全新的单轮请求，而不是在对话里追加一轮：
        这样无需回传上一轮的思考内容，也不会让上下文越滚越长。
        """
        system_with_schema = f"{system}\n\n{_schema_instruction(schema)}"
        prompt = user
        last_error: Exception | None = None

        for _ in range(MAX_STRUCTURED_ATTEMPTS):
            raw = self._chat(system_with_schema, prompt, json_mode=True)
            try:
                return schema.model_validate_json(_strip_code_fence(raw))
            except ValidationError as error:
                last_error = error
                prompt = (
                    f"{user}\n\n你上一次的输出没有通过校验，请修正后重新输出。\n校验错误：\n{error}"
                )

        raise LLMOutputError(
            f"模型连续 {MAX_STRUCTURED_ATTEMPTS} 次未能输出合法的 {schema.__name__}"
        ) from last_error

    def _chat(self, system: str, user: str, *, json_mode: bool) -> str:
        # 只在需要时才带上可选参数，保持对各家 OpenAI 兼容接口的最大兼容性
        options: dict = {}
        if json_mode:
            options["response_format"] = {"type": "json_object"}
        if self._extra_body:
            options["extra_body"] = self._extra_body

        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=self._temperature,
            **options,
        )
        if response.usage is not None:
            self.usage.record(response.usage.prompt_tokens, response.usage.completion_tokens)

        content = response.choices[0].message.content
        if not content or not content.strip():
            raise LLMOutputError("模型返回了空内容。")
        return content


def _strip_code_fence(text: str) -> str:
    """去掉包在 JSON 外面的 Markdown 代码块标记（部分模型即使在 JSON 模式下也会加）。"""
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.removeprefix("```json").removeprefix("```")
        stripped = stripped.removesuffix("```")
    return stripped.strip()


def _schema_instruction(schema: type[BaseModel]) -> str:
    """把 Pydantic 模型转成给模型看的输出格式说明。"""
    json_schema = json.dumps(schema.model_json_schema(), ensure_ascii=False, indent=2)
    return (
        "只输出一个 JSON 对象，不要输出任何其他文字或 Markdown 代码块。\n"
        f"该 JSON 对象必须符合下面的 JSON Schema：\n{json_schema}"
    )
