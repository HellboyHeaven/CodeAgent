"""tools/code.py — инструменты понимания Python-кода через ast: структура файла,
исходник функции, места использования имени."""

from __future__ import annotations

import ast
from pathlib import Path

from .base import Tool, ToolContext, ToolError, is_sensitive, schema

MAX_FUNCTION_LINES = 300
_FUNC = (ast.FunctionDef, ast.AsyncFunctionDef)


def _load(ctx: ToolContext, file: str) -> tuple[Path, str, ast.Module]:
    path = ctx.resolve(file)
    if not path.is_file() or path.suffix != ".py":
        raise ToolError(f"Нужен путь к .py файлу: {file}")
    if is_sensitive(path):
        raise ToolError(f"Чтение запрещено: {file}")
    source = path.read_text(encoding="utf-8", errors="replace")
    try:
        return path, source, ast.parse(source)
    except SyntaxError as e:
        raise ToolError(f"Синтаксическая ошибка в {file}: строка {e.lineno}")


def _definitions(tree: ast.Module) -> dict[str, ast.AST]:
    """Все классы, функции и методы: 'func', 'Class', 'Class.method'."""
    result: dict[str, ast.AST] = {}

    def visit(body: list[ast.stmt], prefix: str) -> None:
        for node in body:
            if isinstance(node, ast.ClassDef):
                result[prefix + node.name] = node
                visit(node.body, prefix + node.name + ".")
            elif isinstance(node, _FUNC):
                result[prefix + node.name] = node

    visit(tree.body, "")
    return result


def _first_line(node: ast.AST) -> int:
    """Первая строка определения, включая декораторы."""
    decorators = getattr(node, "decorator_list", [])
    return min([node.lineno] + [d.lineno for d in decorators])


def _signature(node: ast.AST) -> str:
    if isinstance(node, ast.ClassDef):
        return f"class {node.name}"
    a = node.args
    names = [x.arg for x in a.posonlyargs + a.args + a.kwonlyargs]
    if a.vararg:
        names.append("*" + a.vararg.arg)
    if a.kwarg:
        names.append("**" + a.kwarg.arg)
    prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
    return f"{prefix} {node.name}({', '.join(names)})"


def list_symbols(ctx: ToolContext, file: str) -> str:
    path, _, tree = _load(ctx, file)
    lines = [f"{ctx.rel(path)}:"]

    def visit(body: list[ast.stmt], indent: str) -> None:
        for node in body:
            if isinstance(node, (ast.ClassDef, *_FUNC)):
                lines.append(f"{indent}{_signature(node)}  [строки {_first_line(node)}-{node.end_lineno}]")
                if isinstance(node, ast.ClassDef):
                    visit(node.body, indent + "  ")

    visit(tree.body, "  ")
    return "\n".join(lines) if len(lines) > 1 else f"{lines[0]} нет классов и функций"


def get_function_code(ctx: ToolContext, file: str, name: str) -> str:
    path, source, tree = _load(ctx, file)
    defs = _definitions(tree)

    node = defs.get(name)
    if node is None and "." not in name:
        # короткое имя метода: ищем среди 'Class.name'
        matches = [q for q in defs if q.endswith("." + name)]
        if len(matches) == 1:
            name, node = matches[0], defs[matches[0]]
        elif len(matches) > 1:
            raise ToolError(f"Имя неоднозначно, уточни: {', '.join(matches)}")
    if node is None:
        raise ToolError(f"'{name}' не найдено в {file}. Есть: {', '.join(list(defs)[:30])}")

    start, end = _first_line(node), node.end_lineno
    lines = source.splitlines()
    shown_end = min(end, start + MAX_FUNCTION_LINES - 1)
    body = "\n".join(f"{n:>5}| {lines[n - 1]}" for n in range(start, shown_end + 1))
    note = f"\n[показано {shown_end - start + 1} из {end - start + 1} строк]" if shown_end < end else ""
    return f"{ctx.rel(path)}: {name} (строки {start}-{end})\n{body}{note}"


def find_usages(ctx: ToolContext, name: str, path: str = ".", max_results: int = 40) -> str:
    """Ищет по AST, а не по тексту: строки и комментарии не дают ложных совпадений."""
    base = ctx.resolve(path)
    max_results = max(1, min(int(max_results), 100))
    found: list[str] = []

    for file in ctx.iter_files(base, "*.py"):
        try:
            source = file.read_text(encoding="utf-8", errors="replace")
            tree = ast.parse(source)
        except (SyntaxError, OSError):
            continue
        src_lines = source.splitlines()

        for node in ast.walk(tree):
            kind = None
            if isinstance(node, (ast.ClassDef, *_FUNC)) and node.name == name:
                kind = "определение"
            elif isinstance(node, ast.Name) and node.id == name:
                kind = "использование"
            elif isinstance(node, ast.Attribute) and node.attr == name:
                kind = "использование (атрибут)"
            elif isinstance(node, (ast.Import, ast.ImportFrom)) and any(
                (a.asname or a.name).split(".")[-1] == name for a in node.names
            ):
                kind = "импорт"
            if kind:
                text = src_lines[node.lineno - 1].strip()[:160]
                found.append(f"{ctx.rel(file)}:{node.lineno}: [{kind}] {text}")

    found = sorted(set(found))
    if len(found) > max_results:
        return "\n".join(found[:max_results]) + f"\n[показано {max_results} из {len(found)}]"
    return "\n".join(found) or f"'{name}' нигде не встречается"


TOOLS = [
    Tool(
        "list_symbols",
        "Структура Python-файла: классы, функции и методы с номерами строк. "
        "Используй, чтобы быстро понять файл, не читая его целиком.",
        schema({"file": {"type": "string", "description": "Путь к .py файлу"}}, ["file"]),
        list_symbols,
    ),
    Tool(
        "get_function_code",
        "Возвращает исходный код одной функции, метода или класса.",
        schema({
            "file": {"type": "string", "description": "Путь к .py файлу"},
            "name": {"type": "string", "description": "Имя: 'func', 'Class' или 'Class.method'"},
        }, ["file", "name"]),
        get_function_code,
    ),
    Tool(
        "find_usages",
        "Находит, где в проекте определяется и используется функция, класс или переменная "
        "(поиск по AST, без ложных совпадений в строках и комментариях).",
        schema({
            "name": {"type": "string", "description": "Имя без точек, например 'build_tree'"},
            "path": {"type": "string", "description": "Где искать, по умолчанию весь проект"},
            "max_results": {"type": "integer", "description": "Максимум результатов, по умолчанию 40"},
        }, ["name"]),
        find_usages,
    ),
]
