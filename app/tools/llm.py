"""
llm.py — клиент языковой модели.
Отвечает только за обращение к API. Все настройки приходят из config.py
(объект LLMSettings), сам модуль ничего не читает из окружения.

Два способа использования:

    from config import settings
    from llm import LLMClient

    client = LLMClient(settings.llm_settings())

    # 1. Простой запрос: вопрос -> текст
    client.complete("Ты помощник.", "Привет!")

    # 2. Диалог с инструментами (tool calling): используется в agent_loop.py
    response = client.chat(system, messages, tools=registry.schemas(client.provider))
    response.text, response.tool_calls

Формат сообщений единый для всех провайдеров («нейтральный»):
    {"role": "user", "content": "текст"}
    {"role": "assistant", "content": "текст", "tool_calls": [ToolCall, ...]}
    {"role": "tool", "tool_call_id": "...", "content": "результат инструмента"}
Внутри клиент сам переводит его в формат OpenAI или Anthropic.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from app.core.config import LLMSettings


class LLMError(Exception):
    """Ошибка вызова модели (нет библиотеки, нет ключа, ошибка API)."""


@dataclass
class ToolCall:
    """Просьба модели вызвать инструмент."""
    id: str
    name: str
    arguments: str  # JSON-строка, как её присылает модель


@dataclass
class LLMResponse:
    """Один ответ модели: текст и/или вызовы инструментов."""
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)


# --- Перевод нейтральных сообщений в форматы провайдеров (чистые функции) ---

def to_openai_messages(system: str, messages: list[dict]) -> list[dict]:
    out: list[dict] = [{"role": "system", "content": system}]
    for m in messages:
        if m["role"] == "assistant" and m.get("tool_calls"):
            out.append({
                "role": "assistant",
                "content": m.get("content") or None,
                "tool_calls": [
                    {"id": c.id, "type": "function",
                     "function": {"name": c.name, "arguments": c.arguments or "{}"}}
                    for c in m["tool_calls"]
                ],
            })
        elif m["role"] == "tool":
            out.append({"role": "tool", "tool_call_id": m["tool_call_id"], "content": m["content"]})
        else:
            out.append({"role": m["role"], "content": m["content"]})
    return out


def to_anthropic_messages(messages: list[dict]) -> list[dict]:
    """У Anthropic результаты инструментов — это блоки в сообщении пользователя,
    а подряд идущие сообщения пользователя нужно объединять."""
    out: list[dict] = []

    def add_user(blocks: list[dict]) -> None:
        if out and out[-1]["role"] == "user":
            out[-1]["content"].extend(blocks)
        else:
            out.append({"role": "user", "content": list(blocks)})

    for m in messages:
        if m["role"] == "user":
            add_user([{"type": "text", "text": m["content"]}])
        elif m["role"] == "tool":
            add_user([{"type": "tool_result", "tool_use_id": m["tool_call_id"], "content": m["content"]}])
        else:
            blocks: list[dict] = []
            if m.get("content"):
                blocks.append({"type": "text", "text": m["content"]})
            for c in m.get("tool_calls") or []:
                try:
                    args = json.loads(c.arguments) if c.arguments else {}
                except json.JSONDecodeError:
                    args = {}
                blocks.append({"type": "tool_use", "id": c.id, "name": c.name, "input": args})
            if blocks:
                out.append({"role": "assistant", "content": blocks})
    return out


class LLMClient:
    def __init__(self, settings: LLMSettings):
        self.settings = settings

    @property
    def provider(self) -> str:
        return self.settings.provider

    def describe(self) -> str:
        return self.settings.describe()

    def complete(self, system: str, user: str) -> str:
        """Простой запрос без инструментов: возвращает текст ответа."""
        return self.chat(system, [{"role": "user", "content": user}]).text

    def chat(
        self,
        system: str,
        messages: list[dict],
        tools: list[dict] | None = None,
        allow_tool_calls: bool = True,
    ) -> LLMResponse:
        """Один шаг диалога.
        tools — схемы из registry.schemas(client.provider).
        allow_tool_calls=False — инструменты описаны, но вызывать их запрещено
        (нужно для финального ответа, когда лимит шагов исчерпан)."""
        if self.settings.provider == "anthropic":
            return self._chat_anthropic(system, messages, tools, allow_tool_calls)
        return self._chat_openai(system, messages, tools, allow_tool_calls)

    # --- Anthropic ---
    def _chat_anthropic(self, system, messages, tools, allow_tool_calls) -> LLMResponse:
        try:
            import anthropic
        except ImportError:
            raise LLMError("Не установлена библиотека: pip install anthropic")
        s = self.settings
        if not s.api_key:
            raise LLMError("Не задан ключ: LLM_API_KEY или ANTHROPIC_API_KEY")

        client = anthropic.Anthropic(api_key=s.api_key, base_url=s.base_url)
        params: dict = dict(
            model=s.model,
            max_tokens=s.max_tokens,
            system=system,
            messages=to_anthropic_messages(messages),
        )
        if tools:
            params["tools"] = tools
            if not allow_tool_calls:
                params["tool_choice"] = {"type": "none"}
        if s.temperature is not None:
            params["temperature"] = s.temperature

        try:
            response = client.messages.create(**params)
        except anthropic.APIError as e:
            raise LLMError(f"Ошибка Anthropic API: {e}") from e

        text = "".join(b.text for b in response.content if b.type == "text")
        calls = [
            ToolCall(b.id, b.name, json.dumps(b.input, ensure_ascii=False))
            for b in response.content if b.type == "tool_use"
        ]
        return LLMResponse(text, calls)

    # --- OpenAI и совместимые серверы ---
    def _chat_openai(self, system, messages, tools, allow_tool_calls) -> LLMResponse:
        try:
            import openai
        except ImportError:
            raise LLMError("Не установлена библиотека: pip install openai")
        s = self.settings

        # Локальным серверам (Ollama и т.п.) ключ не нужен, но SDK требует непустую строку
        api_key = s.api_key or ("not-needed" if s.base_url else None)
        if not api_key:
            raise LLMError("Не задан ключ: LLM_API_KEY или OPENAI_API_KEY")

        client = openai.OpenAI(api_key=api_key, base_url=s.base_url)
        # Официальный OpenAI (в том числе новые модели) требует max_completion_tokens,
        # а старый max_tokens отклоняет. Сторонние серверы (Ollama и др.) — наоборот.
        limit_param = "max_tokens" if s.base_url else "max_completion_tokens"
        params: dict = {
            "model": s.model,
            limit_param: s.max_tokens,
            "messages": to_openai_messages(system, messages),
        }
        if tools:
            params["tools"] = tools
            if not allow_tool_calls:
                params["tool_choice"] = "none"
        if s.temperature is not None:
            params["temperature"] = s.temperature

        try:
            response = client.chat.completions.create(**params)
        except openai.OpenAIError as e:
            raise LLMError(f"Ошибка OpenAI-совместимого API: {e}") from e

        msg = response.choices[0].message
        calls = [
            ToolCall(c.id, c.function.name, c.function.arguments or "{}")
            for c in (msg.tool_calls or [])
        ]
        return LLMResponse(msg.content or "", calls)
