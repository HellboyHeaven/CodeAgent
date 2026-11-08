"""
code_metrics.py — третий модуль ИИ-агента.
Считает метрики качества Python-кода и находит проблемные места:
длинные функции, высокую сложность, «god-классы», глубокую вложенность.

Использование:
    python code_metrics.py /путь/к/проекту
    python code_metrics.py . --json metrics.json
"""

from __future__ import annotations

import argparse
import ast
import json
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path

from project_scanner import IGNORE_DIRS


@dataclass
class Thresholds:
    """Пороги, после которых код считается проблемным."""
    max_function_lines: int = 50
    max_complexity: int = 10
    max_args: int = 5
    max_nesting: int = 4
    max_class_methods: int = 15
    max_class_lines: int = 300
    max_file_lines: int = 500


@dataclass
class FunctionMetrics:
    name: str          # 'func' или 'Class.method'
    file: str
    line: int
    lines: int         # длина в строках
    complexity: int    # цикломатическая сложность
    args: int          # число аргументов (без self/cls)
    nesting: int       # максимальная вложенность
    has_docstring: bool


@dataclass
class ClassMetrics:
    name: str
    file: str
    line: int
    lines: int
    methods: int
    has_docstring: bool


@dataclass
class Issue:
    kind: str       # тип проблемы
    where: str      # 'файл:строка имя'
    message: str    # человекочитаемое описание
    value: int      # измеренное значение (для сортировки)


@dataclass
class MetricsReport:
    files: int = 0
    functions: list[FunctionMetrics] = field(default_factory=list)
    classes: list[ClassMetrics] = field(default_factory=list)
    issues: list[Issue] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)

    @property
    def avg_complexity(self) -> float:
        if not self.functions:
            return 0.0
        return sum(f.complexity for f in self.functions) / len(self.functions)

    @property
    def docstring_coverage(self) -> float:
        """Доля функций и классов с docstring (0..1)."""
        items = [f.has_docstring for f in self.functions] + [c.has_docstring for c in self.classes]
        return sum(items) / len(items) if items else 0.0


# --- Вычисление метрик одной функции ---

_FUNC_TYPES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)
_BRANCH_TYPES = (ast.If, ast.For, ast.AsyncFor, ast.While, ast.ExceptHandler, ast.IfExp, ast.Assert)
_NESTING_TYPES = (ast.If, ast.For, ast.AsyncFor, ast.While, ast.Try, ast.With, ast.AsyncWith)


def complexity_of(func: ast.AST) -> int:
    """Цикломатическая сложность: 1 + число точек ветвления (вложенные функции не считаем)."""
    score = 1
    stack = list(ast.iter_child_nodes(func))
    while stack:
        node = stack.pop()
        if isinstance(node, _FUNC_TYPES):
            continue  # вложенная функция считается отдельно
        if isinstance(node, _BRANCH_TYPES):
            score += 1
        elif isinstance(node, ast.BoolOp):
            score += len(node.values) - 1  # каждое and/or — дополнительный путь
        elif isinstance(node, ast.comprehension):
            score += 1 + len(node.ifs)
        elif isinstance(node, ast.match_case):
            score += 1
        stack.extend(ast.iter_child_nodes(node))
    return score


def nesting_of(node: ast.AST, depth: int = 0) -> int:
    """Максимальная глубина вложенности if/for/while/try/with."""
    deepest = depth
    for child in ast.iter_child_nodes(node):
        if isinstance(child, _FUNC_TYPES):
            continue
        d = depth + 1 if isinstance(child, _NESTING_TYPES) else depth
        deepest = max(deepest, nesting_of(child, d))
    return deepest


def count_args(func: ast.FunctionDef | ast.AsyncFunctionDef) -> int:
    a = func.args
    names = [x.arg for x in a.posonlyargs + a.args + a.kwonlyargs]
    if a.vararg:
        names.append(a.vararg.arg)
    if a.kwarg:
        names.append(a.kwarg.arg)
    return len([n for n in names if n not in ("self", "cls")])


# --- Обход файла ---

def analyze_file(path: Path, rel: str, report: MetricsReport, th: Thresholds) -> None:
    try:
        source = path.read_text(encoding="utf-8", errors="ignore")
        tree = ast.parse(source)
    except SyntaxError:
        report.skipped.append(rel)
        return

    report.files += 1
    total_lines = source.count("\n") + 1
    if total_lines > th.max_file_lines:
        report.issues.append(Issue(
            "big_file", rel, f"Файл слишком большой: {total_lines} строк", total_lines))

    def visit(node: ast.AST, class_name: str | None) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.ClassDef):
                add_class(child)
                visit(child, child.name)
            elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                add_function(child, class_name)
                visit(child, None)  # вложенные функции — уже не методы
            else:
                visit(child, class_name)

    def add_class(node: ast.ClassDef) -> None:
        lines = (node.end_lineno or node.lineno) - node.lineno + 1
        methods = sum(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) for n in node.body)
        report.classes.append(ClassMetrics(
            node.name, rel, node.lineno, lines, methods, ast.get_docstring(node) is not None))
        where = f"{rel}:{node.lineno} {node.name}"
        if methods > th.max_class_methods or lines > th.max_class_lines:
            report.issues.append(Issue(
                "god_class", where,
                f"Слишком большой класс: {methods} методов, {lines} строк", max(methods, lines)))

    def add_function(node: ast.FunctionDef | ast.AsyncFunctionDef, class_name: str | None) -> None:
        name = f"{class_name}.{node.name}" if class_name else node.name
        lines = (node.end_lineno or node.lineno) - node.lineno + 1
        m = FunctionMetrics(
            name, rel, node.lineno, lines, complexity_of(node),
            count_args(node), nesting_of(node), ast.get_docstring(node) is not None)
        report.functions.append(m)

        where = f"{rel}:{node.lineno} {name}"
        if m.lines > th.max_function_lines:
            report.issues.append(Issue("long_function", where, f"Длинная функция: {m.lines} строк", m.lines))
        if m.complexity > th.max_complexity:
            report.issues.append(Issue("complex_function", where, f"Высокая сложность: {m.complexity}", m.complexity))
        if m.args > th.max_args:
            report.issues.append(Issue("many_args", where, f"Много аргументов: {m.args}", m.args))
        if m.nesting > th.max_nesting:
            report.issues.append(Issue("deep_nesting", where, f"Глубокая вложенность: {m.nesting}", m.nesting))

    visit(tree, None)


def analyze_project(root: str | Path, th: Thresholds | None = None) -> MetricsReport:
    root = Path(root).resolve()
    th = th or Thresholds()
    report = MetricsReport()

    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in IGNORE_DIRS]
        for name in filenames:
            if name.endswith(".py"):
                path = Path(dirpath) / name
                analyze_file(path, str(path.relative_to(root)), report, th)

    report.issues.sort(key=lambda i: i.value, reverse=True)
    return report


def print_report(report: MetricsReport, top: int = 15) -> None:
    print(f"Файлов: {report.files} | Функций: {len(report.functions)} | Классов: {len(report.classes)}")
    print(f"Средняя сложность: {report.avg_complexity:.1f}")
    print(f"Покрытие docstring: {report.docstring_coverage:.0%}")

    if report.functions:
        print("\nСамые сложные функции:")
        for f in sorted(report.functions, key=lambda x: x.complexity, reverse=True)[:5]:
            print(f"  сложность {f.complexity:>3} | {f.lines:>3} строк | {f.file}:{f.line} {f.name}")

    print(f"\nПроблемы ({len(report.issues)}):")
    if not report.issues:
        print("  не найдены")
    for issue in report.issues[:top]:
        print(f"  [{issue.kind}] {issue.where} — {issue.message}")
    if len(report.issues) > top:
        print(f"  ... и ещё {len(report.issues) - top}")

    if report.skipped:
        print("\nПропущены (синтаксическая ошибка):", ", ".join(report.skipped))


def main() -> None:
    parser = argparse.ArgumentParser(description="Метрики качества Python-кода")
    parser.add_argument("path", nargs="?", default=".", help="Путь к проекту")
    parser.add_argument("--json", metavar="FILE", help="Сохранить отчёт в JSON")
    args = parser.parse_args()

    report = analyze_project(args.path)
    print_report(report)

    if args.json:
        data = asdict(report)
        data["avg_complexity"] = report.avg_complexity
        data["docstring_coverage"] = report.docstring_coverage
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"\nJSON сохранён: {args.json}")


if __name__ == "__main__":
    main()
