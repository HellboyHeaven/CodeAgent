"""
Пакет tools — инструменты, которые агент может вызывать.

    files.py     list_dir, read_file, search_text
    code.py      list_symbols, get_function_code, find_usages
    analysis.py  module_dependencies
    edit.py      create_dir, write_file, edit_file, move_path, delete_path
                 (только при allow_write=True)

Пример:
    from tools import build_registry

    registry = build_registry("./my_project")
    registry.schemas("openai")                       # описание для модели
    registry.call("read_file", {"path": "main.py"})  # выполнение вызова
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from . import analysis, code, edit, files
from .base import Tool, ToolContext, ToolError, ToolRegistry

__all__ = ["build_registry", "Tool", "ToolContext", "ToolError", "ToolRegistry"]


def build_registry(
    root: str | Path,
    max_output_chars: int = 4000,
    timeout_s: float = 10.0,
    allow_write: bool = False,
    confirm: Callable[[str, str], bool] | None = None,
) -> ToolRegistry:
    """Создаёт реестр инструментов для проекта в папке root.

    allow_write — добавить инструменты изменения файлов (по умолчанию только чтение).
    confirm(заголовок, подробности) -> bool — вызывается перед каждым изменением;
    если None, изменения применяются без вопросов."""
    ctx = ToolContext(Path(root).resolve(), max_output_chars, confirm=confirm)
    registry = ToolRegistry(ctx, timeout_s)
    for module in (files, code, analysis):
        registry.register(*module.TOOLS)
    if allow_write:
        registry.register(*edit.TOOLS)
    return registry