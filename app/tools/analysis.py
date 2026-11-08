"""tools/analysis.py — инструменты, использующие анализаторы проекта (граф зависимостей)."""

from __future__ import annotations

import difflib
from pathlib import Path

from dependency_graph import build_dependency_graph, path_to_module

from .base import Tool, ToolContext, ToolError, schema


def module_dependencies(ctx: ToolContext, module: str) -> str:
    # Граф строится один раз и кэшируется на время работы агента
    if "graph" not in ctx.cache:
        ctx.cache["graph"] = build_dependency_graph(ctx.root)
    graph = ctx.cache["graph"]

    # Принимаем и имя модуля (app.core.db), и путь к файлу (app/core/db.py)
    if module.endswith(".py") or "/" in module:
        module = path_to_module(Path(module))

    if module not in graph.modules:
        close = difflib.get_close_matches(module, graph.modules, n=5)
        hint = f" Похожие: {', '.join(close)}" if close else ""
        raise ToolError(f"Модуль '{module}' не найден.{hint}")

    imports = sorted(graph.internal.get(module, ()))
    external = sorted(graph.external.get(module, ()))
    used_by = sorted(graph.imported_by()[module])
    cycles = [c for c in graph.find_cycles() if module in c]

    out = [
        f"Модуль: {module} ({graph.modules[module]})",
        f"Импортирует из проекта: {', '.join(imports) or 'ничего'}",
        f"Сторонние пакеты: {', '.join(external) or 'нет'}",
        f"Импортируется в: {', '.join(used_by) or 'нигде'}",
    ]
    if cycles:
        out.append("Входит в циклы: " + "; ".join(" <-> ".join(c) for c in cycles))
    return "\n".join(out)


TOOLS = [
    Tool(
        "module_dependencies",
        "Показывает связи модуля: что он импортирует, кто импортирует его, "
        "сторонние пакеты и циклические зависимости.",
        schema({"module": {"type": "string", "description": "Имя модуля (app.core.db) или путь к файлу"}}, ["module"]),
        module_dependencies,
    ),
]
