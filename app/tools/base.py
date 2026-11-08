"""
tools/base.py — основа системы инструментов.

Инструмент (Tool) — это функция, которую модель может вызвать по имени.
Реестр (ToolRegistry) хранит инструменты, отдаёт их описание модели в нужном формате
и безопасно выполняет вызовы: с таймаутом, обрезкой вывода и без падений.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime
from typing import Any, Callable, Iterator

from project_scanner import IGNORE_DIRS


class ToolError(Exception):
    """Ожидаемая ошибка инструмента. Её текст увидит модель и сможет поправиться."""


# --- Защита секретов: содержимое таких файлов нельзя отправлять в LLM ---

_SENSITIVE_SUFFIXES = (".pem", ".key", ".p12", ".pfx", ".crt")
_SENSITIVE_NAMES = {"credentials", ".netrc", ".pypirc"}


def is_sensitive(path: Path) -> bool:
    name = path.name.lower()
    return (
        name.startswith(".env")
        or name.startswith("id_rsa")
        or name.startswith("id_ed25519")
        or name.endswith(_SENSITIVE_SUFFIXES)
        or name in _SENSITIVE_NAMES
    )


# Папки, которые агент не имеет права изменять
PROTECTED_DIRS = {".git", ".agent_backup"}


@dataclass
class ToolContext:
    """Общее окружение инструментов: корень проекта, кэш и подтверждение действий."""
    root: Path
    max_output_chars: int = 4000
    cache: dict[str, Any] = field(default_factory=dict)
    # confirm(заголовок, подробности) -> разрешить ли действие. None — разрешать всё
    confirm: Callable[[str, str], bool] | None = None
    # метка сессии: резервные копии складываются в .agent_backup/<метка>/
    stamp: str = field(default_factory=lambda: datetime.now().strftime("%Y%m%d-%H%M%S"))

    def approve(self, title: str, details: str = "") -> None:
        """Спрашивает разрешение на изменение. Если пользователь отказал — ToolError,
        и модель увидит отказ."""
        if self.confirm is not None and not self.confirm(title, details):
            raise ToolError(f"Пользователь отклонил действие: {title}. Не повторяй его без новой просьбы.")

    def resolve_new(self, rel: str) -> Path:
        """Путь для изменения. В отличие от resolve() файл может ещё не существовать,
        зато запрещены корень проекта, служебные папки и файлы с секретами."""
        if not rel or rel.strip() in (".", "./"):
            raise ToolError("Нужен путь внутри проекта, а не его корень")
        path = (self.root / rel).resolve()
        if path == self.root:
            raise ToolError("Нельзя изменять корень проекта")
        if not path.is_relative_to(self.root):
            raise ToolError(f"Путь выходит за пределы проекта: {rel}")
        if PROTECTED_DIRS & set(path.relative_to(self.root).parts):
            raise ToolError(f"Служебная папка защищена от изменений: {rel}")
        if is_sensitive(path):
            raise ToolError(f"Файл с возможными секретами защищён от изменений: {rel}")
        return path

    def resolve(self, rel: str) -> Path:
        """Превращает относительный путь в абсолютный и не даёт выйти за пределы проекта."""
        path = (self.root / (rel or ".")).resolve()  # resolve() раскрывает и симлинки
        if path != self.root and not path.is_relative_to(self.root):
            raise ToolError(f"Путь выходит за пределы проекта: {rel}")
        if not path.exists():
            raise ToolError(f"Не найдено: {rel}")
        return path

    def rel(self, path: Path) -> str:
        return path.relative_to(self.root).as_posix()

    def iter_files(self, base: Path, pattern: str = "*") -> Iterator[Path]:
        """Обходит файлы, пропуская служебные папки и секреты."""
        if base.is_file():
            if not is_sensitive(base):
                yield base
            return
        import os
        for dirpath, dirnames, filenames in os.walk(base):
            dirnames[:] = sorted(d for d in dirnames if d not in IGNORE_DIRS)
            for name in sorted(filenames):
                path = Path(dirpath) / name
                if path.match(pattern) and not is_sensitive(path):
                    yield path


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    parameters: dict          # JSON Schema аргументов
    func: Callable[..., str]  # func(ctx, **arguments) -> str
    mutating: bool = False    # True, если инструмент изменяет файлы проекта


def schema(properties: dict[str, dict], required: list[str] | None = None) -> dict:
    """Короткая запись JSON Schema для аргументов инструмента."""
    return {"type": "object", "properties": properties, "required": required or []}


class ToolRegistry:
    def __init__(self, ctx: ToolContext, timeout_s: float = 10.0):
        self.ctx = ctx
        self.timeout_s = timeout_s
        self._tools: dict[str, Tool] = {}
        self._pool = ThreadPoolExecutor(max_workers=4)

    def register(self, *tools: Tool) -> None:
        for t in tools:
            if t.name in self._tools:
                raise ValueError(f"Инструмент уже зарегистрирован: {t.name}")
            self._tools[t.name] = t

    def names(self) -> list[str]:
        return list(self._tools)

    def schemas(self, provider: str = "openai") -> list[dict]:
        """Описание инструментов для модели. Форматы у провайдеров разные."""
        if provider == "anthropic":
            return [
                {"name": t.name, "description": t.description, "input_schema": t.parameters}
                for t in self._tools.values()
            ]
        return [
            {"type": "function",
             "function": {"name": t.name, "description": t.description, "parameters": t.parameters}}
            for t in self._tools.values()
        ]

    def call(self, name: str, arguments: dict | str | None = None) -> str:
        """Выполняет инструмент. Никогда не бросает исключений: любая проблема
        возвращается текстом, чтобы модель увидела её и исправила вызов."""
        tool = self._tools.get(name)
        if tool is None:
            return f"Ошибка: нет инструмента '{name}'. Доступны: {', '.join(self._tools)}"

        try:
            if isinstance(arguments, str):
                arguments = json.loads(arguments) if arguments.strip() else {}
            arguments = arguments or {}
            if not isinstance(arguments, dict):
                raise ValueError("аргументы должны быть объектом")
        except (ValueError, json.JSONDecodeError) as e:
            return f"Ошибка: некорректные аргументы ({e})"

        def run() -> str:
            return tool.func(self.ctx, **arguments)

        try:
            if tool.mutating:
                # Изменяющие инструменты выполняются сразу и без таймаута: внутри может идти
                # диалог подтверждения с пользователем, а сами операции быстрые.
                result = run()
                self.ctx.cache.clear()  # граф зависимостей и прочий кэш устарели
            else:
                result = self._pool.submit(run).result(timeout=self.timeout_s)
        except FutureTimeout:
            return f"Ошибка: инструмент '{name}' не ответил за {self.timeout_s:g} с"
        except ToolError as e:
            return f"Ошибка: {e}"
        except TypeError as e:
            return f"Ошибка: неверные аргументы для '{name}': {e}"
        except Exception as e:  # неожиданная ошибка не должна ронять агента
            return f"Ошибка при выполнении '{name}': {type(e).__name__}: {e}"

        return self._truncate(str(result))

    def _truncate(self, text: str) -> str:
        limit = self.ctx.max_output_chars
        if len(text) <= limit:
            return text
        return text[:limit] + f"\n...[вывод обрезан: показано {limit} из {len(text)} символов]"