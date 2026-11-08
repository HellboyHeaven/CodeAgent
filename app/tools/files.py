"""tools/files.py — инструменты для работы с файлами: список, чтение, поиск по тексту."""

from __future__ import annotations

import re

from project_scanner import IGNORE_DIRS

from .base import Tool, ToolContext, ToolError, is_sensitive, schema

MAX_FILE_BYTES = 1_000_000
MAX_LINES_PER_READ = 400


def list_dir(ctx: ToolContext, path: str = ".", depth: int = 1) -> str:
    base = ctx.resolve(path)
    if not base.is_dir():
        raise ToolError(f"Это не папка: {path}")
    depth = max(1, min(int(depth), 4))
    lines: list[str] = []

    def walk(directory, level: int, indent: str) -> None:
        entries = sorted(
            (e for e in directory.iterdir() if e.name not in IGNORE_DIRS and not is_sensitive(e)),
            key=lambda e: (e.is_file(), e.name.lower()),
        )
        for e in entries:
            if e.is_dir():
                lines.append(f"{indent}{e.name}/")
                if level < depth:
                    walk(e, level + 1, indent + "  ")
            else:
                lines.append(f"{indent}{e.name}  ({e.stat().st_size} Б)")

    walk(base, 1, "")
    return "\n".join(lines) or "(пусто)"


def read_file(ctx: ToolContext, path: str, start_line: int = 1, end_line: int | None = None) -> str:
    file = ctx.resolve(path)
    if not file.is_file():
        raise ToolError(f"Это не файл: {path}")
    if is_sensitive(file):
        raise ToolError(f"Чтение запрещено (возможные секреты): {path}")
    if file.stat().st_size > MAX_FILE_BYTES:
        raise ToolError("Файл слишком большой (больше 1 МБ)")

    data = file.read_bytes()
    if b"\0" in data:
        raise ToolError("Бинарный файл, прочитать нельзя")
    lines = data.decode("utf-8", errors="replace").splitlines()
    total = len(lines)

    start = max(1, int(start_line))
    end = min(int(end_line) if end_line else total, total, start + MAX_LINES_PER_READ - 1)
    if start > total:
        raise ToolError(f"В файле всего {total} строк")

    body = "\n".join(f"{n:>5}| {lines[n - 1]}" for n in range(start, end + 1))
    header = f"{ctx.rel(file)} (строки {start}-{end} из {total})"
    if end < total:
        header += f"\n[есть продолжение: вызови снова с start_line={end + 1}]"
    return f"{header}\n{body}"


def search_text(
    ctx: ToolContext,
    pattern: str,
    path: str = ".",
    glob: str = "*.py",
    ignore_case: bool = False,
    max_results: int = 30,
) -> str:
    try:
        regex = re.compile(pattern, re.IGNORECASE if ignore_case else 0)
    except re.error as e:
        raise ToolError(f"Некорректное регулярное выражение: {e}")

    base = ctx.resolve(path)
    max_results = max(1, min(int(max_results), 100))
    found: list[str] = []

    for file in ctx.iter_files(base, glob):
        if file.stat().st_size > MAX_FILE_BYTES:
            continue
        try:
            text = file.read_text(encoding="utf-8", errors="strict")
        except (UnicodeDecodeError, OSError):
            continue  # бинарные и нечитаемые файлы пропускаем
        for n, line in enumerate(text.splitlines(), 1):
            if regex.search(line):
                found.append(f"{ctx.rel(file)}:{n}: {line.strip()[:200]}")
                if len(found) >= max_results:
                    return "\n".join(found) + f"\n[достигнут лимит {max_results} совпадений]"
    return "\n".join(found) or "Ничего не найдено"


TOOLS = [
    Tool(
        "list_dir",
        "Показывает содержимое папки проекта (файлы с размером, папки со слэшем).",
        schema({
            "path": {"type": "string", "description": "Папка относительно корня проекта, по умолчанию '.'"},
            "depth": {"type": "integer", "description": "Глубина обхода от 1 до 4, по умолчанию 1"},
        }),
        list_dir,
    ),
    Tool(
        "read_file",
        "Читает текстовый файл с номерами строк. За один вызов не больше 400 строк; "
        "для длинных файлов используй start_line и end_line.",
        schema({
            "path": {"type": "string", "description": "Путь к файлу относительно корня проекта"},
            "start_line": {"type": "integer", "description": "Первая строка, по умолчанию 1"},
            "end_line": {"type": "integer", "description": "Последняя строка"},
        }, ["path"]),
        read_file,
    ),
    Tool(
        "search_text",
        "Ищет регулярное выражение в файлах проекта. Возвращает строки вида файл:номер: текст.",
        schema({
            "pattern": {"type": "string", "description": "Регулярное выражение Python"},
            "path": {"type": "string", "description": "Папка или файл, по умолчанию весь проект"},
            "glob": {"type": "string", "description": "Маска файлов, по умолчанию '*.py'"},
            "ignore_case": {"type": "boolean", "description": "Игнорировать регистр"},
            "max_results": {"type": "integer", "description": "Максимум совпадений, по умолчанию 30"},
        }, ["pattern"]),
        search_text,
    ),
]
