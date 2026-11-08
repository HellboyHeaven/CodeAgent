"""
project_scanner.py — первый модуль ИИ-агента для анализа кода.
Сканирует папки и файлы проекта и собирает структурированный отчёт.

Использование:
    python project_scanner.py /путь/к/проекту
    python project_scanner.py . --depth 3 --json report.json
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from dataclasses import dataclass, field, asdict
from pathlib import Path

# Папки, которые не нужно анализировать
IGNORE_DIRS = {
    ".git", ".idea", ".vscode", "__pycache__", "node_modules", "venv", ".venv",
    "env", "dist", "build", ".mypy_cache", ".pytest_cache", ".tox", ".eggs", ".agent_backup",
}

# Расширение -> язык
LANGUAGES = {
    ".py": "Python", ".js": "JavaScript", ".ts": "TypeScript", ".jsx": "JavaScript",
    ".tsx": "TypeScript", ".java": "Java", ".go": "Go", ".rs": "Rust",
    ".cpp": "C++", ".c": "C", ".cs": "C#", ".rb": "Ruby", ".php": "PHP",
    ".html": "HTML", ".css": "CSS", ".sql": "SQL", ".sh": "Shell",
    ".md": "Markdown", ".json": "JSON", ".yml": "YAML", ".yaml": "YAML",
    ".toml": "TOML", ".txt": "Text",
}

# Файлы, по которым можно понять тип проекта
MARKER_FILES = {
    "requirements.txt": "Python (pip)",
    "pyproject.toml": "Python (pyproject)",
    "setup.py": "Python (setup.py)",
    "Pipfile": "Python (pipenv)",
    "package.json": "Node.js",
    "Dockerfile": "Docker",
    "docker-compose.yml": "Docker Compose",
    "Makefile": "Make",
    "README.md": "README",
    ".gitignore": "Git",
    "pytest.ini": "pytest",
    "tox.ini": "tox",
}

MAX_TEXT_SIZE = 2 * 1024 * 1024  # не читаем файлы больше 2 МБ


@dataclass
class FileInfo:
    path: str
    size: int
    language: str
    lines: int = 0


@dataclass
class PythonStats:
    files: int = 0
    classes: int = 0
    functions: int = 0
    syntax_errors: list[str] = field(default_factory=list)
    top_imports: dict[str, int] = field(default_factory=dict)


@dataclass
class ProjectReport:
    root: str
    total_files: int = 0
    total_dirs: int = 0
    total_size: int = 0
    total_lines: int = 0
    languages: dict[str, int] = field(default_factory=dict)  # язык -> строки
    extensions: dict[str, int] = field(default_factory=dict)  # расширение -> файлы
    markers: list[str] = field(default_factory=list)
    largest_files: list[FileInfo] = field(default_factory=list)
    python: PythonStats = field(default_factory=PythonStats)


def count_lines(path: Path) -> int:
    """Считает строки в текстовом файле, бинарные пропускает."""
    try:
        if path.stat().st_size > MAX_TEXT_SIZE:
            return 0
        with open(path, "rb") as f:
            data = f.read()
        if b"\0" in data:
            return 0
        return data.count(b"\n") + (1 if data and not data.endswith(b"\n") else 0)
    except OSError:
        return 0


def analyze_python_file(path: Path, stats: PythonStats, imports: Counter) -> None:
    """Разбирает Python-файл через ast: классы, функции, импорты."""
    import ast

    stats.files += 1
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="ignore"))
    except SyntaxError as e:
        stats.syntax_errors.append(f"{path}:{e.lineno}")
        return

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            stats.classes += 1
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            stats.functions += 1
        elif isinstance(node, ast.Import):
            for alias in node.names:
                imports[alias.name.split(".")[0]] += 1
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            imports[node.module.split(".")[0]] += 1


def scan_project(root: str | Path, top_n: int = 10) -> ProjectReport:
    root = Path(root).resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"Не папка: {root}")

    report = ProjectReport(root=str(root))
    lang_lines: Counter = Counter()
    ext_count: Counter = Counter()
    imports: Counter = Counter()
    all_files: list[FileInfo] = []

    for dirpath, dirnames, filenames in os.walk(root):
        # Убираем игнорируемые папки прямо на месте, чтобы os.walk в них не заходил
        dirnames[:] = [d for d in dirnames if d not in IGNORE_DIRS]
        report.total_dirs += len(dirnames)

        for name in filenames:
            path = Path(dirpath) / name
            try:
                size = path.stat().st_size
            except OSError:
                continue

            ext = path.suffix.lower()
            language = LANGUAGES.get(ext, "Other")
            lines = count_lines(path)

            report.total_files += 1
            report.total_size += size
            report.total_lines += lines
            ext_count[ext or "(без расширения)"] += 1
            if lines:
                lang_lines[language] += lines
            if name in MARKER_FILES:
                report.markers.append(MARKER_FILES[name])

            all_files.append(FileInfo(str(path.relative_to(root)), size, language, lines))

            if ext == ".py":
                analyze_python_file(path, report.python, imports)

    report.languages = dict(lang_lines.most_common())
    report.extensions = dict(ext_count.most_common())
    report.markers = sorted(set(report.markers))
    report.largest_files = sorted(all_files, key=lambda f: f.size, reverse=True)[:top_n]
    report.python.top_imports = dict(imports.most_common(15))
    return report


def build_tree(root: str | Path, max_depth: int = 3) -> str:
    """Рисует дерево папок и файлов в виде текста."""
    root = Path(root).resolve()
    lines = [f"{root.name}/"]

    def walk(directory: Path, prefix: str, depth: int) -> None:
        if depth > max_depth:
            return
        try:
            entries = sorted(
                (e for e in directory.iterdir() if e.name not in IGNORE_DIRS),
                key=lambda e: (e.is_file(), e.name.lower()),
            )
        except PermissionError:
            return
        for i, entry in enumerate(entries):
            last = i == len(entries) - 1
            lines.append(f"{prefix}{'└── ' if last else '├── '}{entry.name}{'/' if entry.is_dir() else ''}")
            if entry.is_dir():
                walk(entry, prefix + ("    " if last else "│   "), depth + 1)

    walk(root, "", 1)
    return "\n".join(lines)


def human_size(n: float) -> str:
    for unit in ("Б", "КБ", "МБ", "ГБ"):
        if n < 1024:
            return f"{n:.0f} {unit}" if unit == "Б" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} ТБ"


def print_report(report: ProjectReport, tree: str) -> None:
    print("=" * 60)
    print(f"Проект: {report.root}")
    print("=" * 60)
    print(f"Файлов: {report.total_files} | Папок: {report.total_dirs}")
    print(f"Размер: {human_size(report.total_size)} | Строк: {report.total_lines}")

    if report.markers:
        print(f"\nОбнаружено: {', '.join(report.markers)}")

    print("\nЯзыки (по строкам):")
    for lang, n in list(report.languages.items())[:8]:
        print(f"  {lang:<12} {n}")

    py = report.python
    if py.files:
        print(f"\nPython: {py.files} файлов, {py.classes} классов, {py.functions} функций")
        if py.top_imports:
            print("  Частые импорты:", ", ".join(f"{k}({v})" for k, v in py.top_imports.items()))
        if py.syntax_errors:
            print("  Синтаксические ошибки:", ", ".join(py.syntax_errors))

    print("\nСамые большие файлы:")
    for f in report.largest_files[:5]:
        print(f"  {human_size(f.size):>9}  {f.path}")

    print("\nСтруктура:")
    print(tree)


def main() -> None:
    parser = argparse.ArgumentParser(description="Анализ структуры проекта")
    parser.add_argument("path", nargs="?", default=".", help="Путь к проекту")
    parser.add_argument("--depth", type=int, default=3, help="Глубина дерева")
    parser.add_argument("--json", metavar="FILE", help="Сохранить отчёт в JSON")
    args = parser.parse_args()

    report = scan_project(args.path)
    tree = build_tree(args.path, args.depth)
    print_report(report, tree)

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(asdict(report), f, ensure_ascii=False, indent=2)
        print(f"\nJSON-отчёт сохранён: {args.json}")


if __name__ == "__main__":
    main()