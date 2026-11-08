"""
dependency_graph.py — второй модуль ИИ-агента.
Строит граф зависимостей между Python-модулями проекта:
кто кого импортирует, где циклы, какие модули самые «нагруженные».

Использование:
    python dependency_graph.py /путь/к/проекту
    python dependency_graph.py . --json graph.json
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

from project_scanner import IGNORE_DIRS  # переиспользуем список игнорируемых папок


@dataclass
class DependencyGraph:
    # имя модуля -> путь к файлу (относительно корня)
    modules: dict[str, str] = field(default_factory=dict)
    # имя модуля -> внутренние модули, которые он импортирует
    internal: dict[str, set[str]] = field(default_factory=dict)
    # имя модуля -> сторонние пакеты (не stdlib, не из проекта)
    external: dict[str, set[str]] = field(default_factory=dict)
    # файлы, которые не удалось разобрать
    broken: list[str] = field(default_factory=list)

    def imported_by(self) -> dict[str, set[str]]:
        """Обратный граф: имя модуля -> кто его импортирует."""
        result: dict[str, set[str]] = {m: set() for m in self.modules}
        for src, targets in self.internal.items():
            for t in targets:
                result[t].add(src)
        return result

    def most_imported(self, n: int = 5) -> list[tuple[str, int]]:
        """Топ модулей, от которых зависит больше всего других."""
        counts = [(m, len(users)) for m, users in self.imported_by().items()]
        return sorted(counts, key=lambda x: x[1], reverse=True)[:n]

    def most_dependent(self, n: int = 5) -> list[tuple[str, int]]:
        """Топ модулей, которые сами импортируют больше всего."""
        counts = [(m, len(t)) for m, t in self.internal.items()]
        return sorted(counts, key=lambda x: x[1], reverse=True)[:n]

    def find_cycles(self) -> list[list[str]]:
        """Циклические зависимости (алгоритм Тарьяна). Каждый цикл — список модулей."""
        index: dict[str, int] = {}
        low: dict[str, int] = {}
        stack: list[str] = []
        on_stack: set[str] = set()
        cycles: list[list[str]] = []
        counter = [0]

        def visit(v: str) -> None:
            index[v] = low[v] = counter[0]
            counter[0] += 1
            stack.append(v)
            on_stack.add(v)
            for w in self.internal.get(v, ()):
                if w not in index:
                    visit(w)
                    low[v] = min(low[v], low[w])
                elif w in on_stack:
                    low[v] = min(low[v], index[w])
            if low[v] == index[v]:
                component = []
                while True:
                    w = stack.pop()
                    on_stack.discard(w)
                    component.append(w)
                    if w == v:
                        break
                if len(component) > 1:
                    cycles.append(sorted(component))

        sys.setrecursionlimit(max(sys.getrecursionlimit(), 10_000))
        for m in self.modules:
            if m not in index:
                visit(m)
        return cycles

    def to_dict(self) -> dict:
        return {
            "modules": self.modules,
            "internal": {k: sorted(v) for k, v in self.internal.items()},
            "external": {k: sorted(v) for k, v in self.external.items()},
            "cycles": self.find_cycles(),
            "broken": self.broken,
        }


def path_to_module(rel_path: Path) -> str:
    """'pkg/sub/mod.py' -> 'pkg.sub.mod', 'pkg/__init__.py' -> 'pkg'."""
    parts = list(rel_path.with_suffix("").parts)
    if parts and parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def resolve_internal(name: str, known: set[str]) -> str | None:
    """Ищет самый длинный префикс имени, который является модулем проекта."""
    parts = name.split(".")
    for i in range(len(parts), 0, -1):
        candidate = ".".join(parts[:i])
        if candidate in known:
            return candidate
    return None


def extract_imports(tree: ast.AST, module: str, is_package: bool) -> list[str]:
    """Возвращает полные имена всех импортируемых модулей (относительные уже раскрыты)."""
    names: list[str] = []
    package_parts = module.split(".") if module else []
    if not is_package:
        package_parts = package_parts[:-1]  # пакет, в котором лежит файл

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)

        elif isinstance(node, ast.ImportFrom):
            if node.level == 0:
                base = node.module or ""
            else:
                cut = len(package_parts) - (node.level - 1)
                if cut < 0:
                    continue
                base_parts = package_parts[:cut]
                if node.module:
                    base_parts = base_parts + node.module.split(".")
                base = ".".join(base_parts)
            if not base:
                continue
            names.append(base)
            # 'from pkg import mod' — mod может быть подмодулем
            names.extend(f"{base}.{alias.name}" for alias in node.names if alias.name != "*")
    return names


def build_dependency_graph(root: str | Path) -> DependencyGraph:
    root = Path(root).resolve()
    graph = DependencyGraph()
    files: dict[str, tuple[Path, bool]] = {}  # модуль -> (путь, это пакет?)

    # Шаг 1: находим все .py-файлы и даём им имена модулей
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in IGNORE_DIRS]
        for name in filenames:
            if not name.endswith(".py"):
                continue
            path = Path(dirpath) / name
            module = path_to_module(path.relative_to(root))
            if not module:
                continue
            files[module] = (path, name == "__init__.py")
            graph.modules[module] = str(path.relative_to(root))

    known = set(files)
    stdlib = set(sys.stdlib_module_names)

    # Шаг 2: разбираем импорты каждого файла
    for module, (path, is_pkg) in files.items():
        graph.internal[module] = set()
        graph.external[module] = set()
        try:
            tree = ast.parse(path.read_text(encoding="utf-8", errors="ignore"))
        except SyntaxError:
            graph.broken.append(graph.modules[module])
            continue

        for imported in extract_imports(tree, module, is_pkg):
            target = resolve_internal(imported, known)
            if target is not None:
                if target != module:
                    graph.internal[module].add(target)
            else:
                top = imported.split(".")[0]
                if top and top not in stdlib:
                    graph.external[module].add(top)

    return graph


def print_graph(graph: DependencyGraph) -> None:
    print(f"Модулей: {len(graph.modules)}")
    total_edges = sum(len(v) for v in graph.internal.values())
    print(f"Внутренних связей: {total_edges}")

    print("\nСамые импортируемые (от них зависят другие):")
    for m, n in graph.most_imported():
        if n:
            print(f"  {n:>3}  {m}")

    print("\nСамые зависимые (импортируют много модулей):")
    for m, n in graph.most_dependent():
        if n:
            print(f"  {n:>3}  {m}")

    all_ext = sorted({p for s in graph.external.values() for p in s})
    print(f"\nСторонние пакеты: {', '.join(all_ext) or 'нет'}")

    cycles = graph.find_cycles()
    print("\nЦиклические зависимости:")
    if cycles:
        for c in cycles:
            print("  ⚠ " + " <-> ".join(c))
    else:
        print("  не найдены")

    if graph.broken:
        print("\nНе удалось разобрать:", ", ".join(graph.broken))


def main() -> None:
    parser = argparse.ArgumentParser(description="Граф зависимостей Python-проекта")
    parser.add_argument("path", nargs="?", default=".", help="Путь к проекту")
    parser.add_argument("--json", metavar="FILE", help="Сохранить граф в JSON")
    args = parser.parse_args()

    graph = build_dependency_graph(args.path)
    print_graph(graph)

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(graph.to_dict(), f, ensure_ascii=False, indent=2)
        print(f"\nJSON сохранён: {args.json}")


if __name__ == "__main__":
    main()
