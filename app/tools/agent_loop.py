"""
agent_loop.py — агентный цикл.

Модель получает задачу и список инструментов. На каждом шаге она либо просит
вызвать инструменты (мы их выполняем и возвращаем результат), либо отвечает
итоговым текстом. Цикл ограничен max_steps, чтобы не крутиться бесконечно.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Callable

from llm import LLMClient
from app.tools import ToolRegistry

FINAL_PROMPT = (
    "Лимит шагов исчерпан, инструменты больше недоступны. "
    "Составь итоговый отчёт по тому, что уже узнал. "
    "Если чего-то проверить не успел, прямо скажи об этом."
)


@dataclass
class ToolRun:
    """Запись об одном вызове инструмента — для отчёта о ходе работы."""
    step: int
    name: str
    arguments: str
    result_chars: int


@dataclass
class AgentResult:
    answer: str
    steps: int                      # сколько раз вызывалась модель
    tool_runs: list[ToolRun] = field(default_factory=list)
    hit_step_limit: bool = False    # True, если пришлось завершать принудительно


def run_agent(
    client: LLMClient,
    registry: ToolRegistry,
    system: str,
    task: str,
    max_steps: int = 6,
    on_tool: Callable[[ToolRun], None] | None = None,
    history: list[dict] | None = None,
) -> AgentResult:
    """Запускает цикл «модель -> инструменты -> модель» и возвращает итог.
    on_tool вызывается после каждого инструмента (например, чтобы печатать прогресс).
    history — предыдущие сообщения диалога (для режима чата); сам список не изменяется."""
    tools = registry.schemas(client.provider)
    messages: list[dict] = list(history or []) + [{"role": "user", "content": task}]
    runs: list[ToolRun] = []

    for step in range(1, max_steps + 1):
        response = client.chat(system, messages, tools=tools)

        if not response.tool_calls:  # модель закончила и дала ответ
            return AgentResult(response.text, step, runs)

        # Сначала запоминаем просьбу модели, затем добавляем результаты в том же порядке
        messages.append({"role": "assistant", "content": response.text, "tool_calls": response.tool_calls})
        for call in response.tool_calls:
            result = registry.call(call.name, call.arguments)
            run = ToolRun(step, call.name, call.arguments, len(result))
            runs.append(run)
            if on_tool:
                on_tool(run)
            messages.append({"role": "tool", "tool_call_id": call.id, "content": result})

    # Лимит исчерпан: просим итог, запретив новые вызовы инструментов
    messages.append({"role": "user", "content": FINAL_PROMPT})
    final = client.chat(system, messages, tools=tools, allow_tool_calls=False)
    return AgentResult(final.text, max_steps + 1, runs, hit_step_limit=True)


def format_run(run: ToolRun, width: int = 90) -> str:
    """Строка для вывода прогресса: [шаг 2] read_file {"path": "a.py"} -> 1200 симв."""
    try:
        args = json.dumps(json.loads(run.arguments), ensure_ascii=False)
    except (ValueError, TypeError):
        args = run.arguments
    if len(args) > width:
        args = args[: width - 1] + "…"
    return f"[шаг {run.step}] {run.name} {args} -> {run.result_chars} симв."
