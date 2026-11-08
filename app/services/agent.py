"""
agent.py — главный модуль ИИ-агента.
Собирает данные трёх анализаторов (структура, зависимости, метрики),
отправляет их в языковую модель и выводит разбор архитектуры проекта.

Модель и ключи настраиваются через переменные окружения или файл .env
(см. config.py и .env.example).

Использование:
    python agent.py /путь/к/проекту
    python agent.py . --output report.md
    python agent.py . --model gpt-4o     # переопределить LLM_MODEL
    python agent.py . --ask "где используется build_tree?"   # один вопрос о проекте
    python agent.py . --chat             # диалог: вопросы подряд, агент помнит контекст
    python agent.py . --ask "создай tools/utils.py с функцией slugify" --write
    python agent.py . --chat --write     # агент может создавать, править, переносить и удалять файлы
    python agent.py . --chat --write --yes   # то же, но без подтверждений (осторожно!)
    python agent.py . --no-tools         # без инструментов: один запрос по статистике
    python agent.py . --dry-run          # показать, что уйдёт в модель, без запроса
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from project_scanner import scan_project, build_tree
from dependency_graph import build_dependency_graph
from code_metrics import analyze_project, Thresholds
from llm import LLMClient, LLMError
from agent_loop import run_agent, format_run
from app.tools import build_registry
from app.core.config import settings, SettingsError, AnalysisSettings

SYSTEM_PROMPT = """Ты — опытный архитектор ПО и ревьюер Python-кода.
Тебе дают автоматически собранные данные о проекте: структуру файлов, граф зависимостей
между модулями и метрики качества кода. Данные получены статическим анализом,
самого кода ты не видишь, поэтому опирайся только на цифры и не выдумывай детали.

Ответь на русском языке в формате Markdown со следующими разделами:
1. Краткое описание проекта — что это за проект и какой у него стек (по структуре и зависимостям).
2. Архитектура — как организован код, какие слои или модули выделяются.
3. Сильные стороны.
4. Проблемы — самые важные, с привязкой к конкретным файлам и функциям из данных.
5. Рекомендации — не более 5 шагов, отсортированных по приоритету, у каждого коротко «зачем».

Если данных недостаточно для вывода, прямо скажи об этом."""


AGENT_SYSTEM_PROMPT = """Ты — опытный архитектор ПО и ревьюер Python-кода.
Тебе дают автоматически собранные данные о проекте (структура, зависимости, метрики)
и инструменты для чтения кода. Данные — это только отправная точка.

Как работать:
1. Изучи данные и выбери самые подозрительные места: функции с высокой сложностью,
   большие классы, циклические зависимости, самые «нагруженные» модули.
2. Проверь их инструментами: list_symbols, get_function_code, read_file, find_usages,
   module_dependencies, search_text. Не читай всё подряд: бюджет шагов ограничен,
   за один шаг можно вызвать несколько инструментов сразу.
3. Когда информации достаточно, дай итоговый ответ без вызова инструментов.

Правила: ничего не выдумывай, все выводы опирай на увиденный код и данные.
Ссылайся на конкретные места в формате файл:строка.

Итоговый ответ на русском языке в формате Markdown:
1. Краткое описание проекта и его стек.
2. Архитектура — как организован код.
3. Сильные стороны.
4. Проблемы — с привязкой к файлам и строкам, по важности.
5. Рекомендации — не более 5 шагов по приоритету; для главных — что именно изменить в коде.
"""


ASK_SYSTEM_PROMPT = """Ты — помощник разработчика, который отвечает на вопросы о конкретном Python-проекте.
Чтобы ответить, используй инструменты: ищи, читай файлы и функции, смотри связи модулей.
Не читай лишнего: берись только за то, что нужно для ответа, бюджет шагов ограничен.

Правила: отвечай на русском, кратко и по делу. Ссылайся на места в коде в формате файл:строка.
Фрагменты кода приводи только если они нужны для ответа. Ничего не выдумывай:
если ответа в коде нет или ты не успел это проверить, так и скажи."""


WRITE_PROMPT = """Тебе разрешено изменять проект инструментами create_dir, write_file, edit_file,
move_path, delete_path.

Правила изменений:
- Сначала изучи существующий код (read_file, list_symbols) и следуй его стилю.
- Существующие файлы правь через edit_file с минимальным фрагментом, а не перезаписывай целиком.
- Новый код пиши полностью и рабочим: с импортами, без заглушек и многоточий.
- После перемещения или переименования .py-файлов найди и обнови импорты (find_usages, search_text).
- Если инструмент вернул предупреждение о синтаксической ошибке, сразу исправь её.
- Пользователь подтверждает каждое действие. Если он отклонил действие, не повторяй его,
  а коротко спроси, что делать дальше.
- Удалённое и перезаписанное сохраняется в .agent_backup и может быть восстановлено,
  но всё равно не удаляй и не меняй то, о чём тебя не просили.
- Структура проекта ниже показана на момент запуска и могла измениться; актуальную смотри через list_dir.
- В конце коротко перечисли, что изменено."""


def make_confirm():
    """Диалог подтверждения в терминале. Ответ 'a' разрешает все следующие действия сессии."""
    state = {"all": False}

    def confirm(title: str, details: str) -> bool:
        if state["all"]:
            print(f"[авто] {title}", file=sys.stderr)
            return True
        print(f"\n┌─ Агент хочет: {title}", file=sys.stderr)
        for line in details.splitlines():
            print(f"│ {line}", file=sys.stderr)
        try:
            reply = input("└─ Применить? [y — да / N — нет / a — да, и все следующие]: ").strip().lower()
        except EOFError:
            return False  # нет интерактивного ввода — ничего не меняем
        if reply in ("a", "all", "а", "все"):
            state["all"] = True
            return True
        return reply in ("y", "yes", "д", "да")

    return confirm


def build_context(path: str, cfg: AnalysisSettings) -> dict:
    """Запускает три анализатора и собирает компактный словарь для модели."""
    scan = scan_project(path)
    graph = build_dependency_graph(path)
    metrics = analyze_project(path, Thresholds(**cfg.thresholds()))

    scan_data = asdict(scan)
    scan_data["largest_files"] = scan_data["largest_files"][:5]

    return {
        "structure": {
            "summary": scan_data,
            "tree": build_tree(path, cfg.tree_depth),
        },
        "dependencies": {
            "modules_count": len(graph.modules),
            "most_imported": graph.most_imported(7),
            "most_dependent": graph.most_dependent(7),
            "cycles": graph.find_cycles(),
            "external_packages": sorted({p for s in graph.external.values() for p in s}),
        },
        "metrics": {
            "functions": len(metrics.functions),
            "classes": len(metrics.classes),
            "avg_complexity": round(metrics.avg_complexity, 1),
            "docstring_coverage": round(metrics.docstring_coverage, 2),
            "issues_total": len(metrics.issues),
            "top_issues": [asdict(i) for i in metrics.issues[:cfg.max_issues]],
        },
    }


def context_message(context: dict) -> str:
    return "Вот данные анализа проекта в формате JSON:\n\n" + json.dumps(context, ensure_ascii=False, indent=2)


def ask_llm(context: dict, client: LLMClient) -> str:
    """Режим без инструментов: один запрос по статистике."""
    return client.complete(SYSTEM_PROMPT, context_message(context))


def run_question_mode(args) -> None:
    """Режимы --ask (один вопрос) и --chat (диалог). Полный анализ проекта здесь не нужен:
    модель получает только дерево папок и сама ищет остальное инструментами."""
    try:
        client = LLMClient(settings.llm_settings())
    except SettingsError as e:
        sys.exit(f"Ошибка: {e}")

    write = settings.allow_write
    registry = build_registry(
        args.path, settings.max_tool_output_chars, settings.tool_timeout_s,
        allow_write=write,
        confirm=make_confirm() if write and settings.confirm_writes else None,
    )
    tree = build_tree(args.path, settings.analysis.tree_depth)
    prompt = ASK_SYSTEM_PROMPT + ("\n\n" + WRITE_PROMPT if write else "")
    system = f"{prompt}\n\nСтруктура проекта:\n{tree}"
    max_steps = settings.max_write_steps if write else settings.max_agent_steps
    if write:
        mode = "с подтверждением каждого действия" if settings.confirm_writes else "БЕЗ подтверждений"
        print(f"Режим изменений включён ({mode}). Копии удалённого и изменённого: .agent_backup/",
              file=sys.stderr)
    history: list[dict] = []  # храним только вопросы и итоговые ответы, без следов инструментов

    def ask(question: str) -> str:
        result = run_agent(
            client, registry, system, question,
            max_steps=max_steps,
            history=history,
            on_tool=lambda run: print(format_run(run), file=sys.stderr),
        )
        return result.answer

    if args.ask:
        try:
            answer = ask(args.ask)
        except LLMError as e:
            sys.exit(f"Ошибка: {e}")
        print(answer)
        if args.output:
            Path(args.output).write_text(answer, encoding="utf-8")
        return

    print(f"Чат с агентом ({client.describe()}). Пустая строка или 'exit' — выход.", file=sys.stderr)
    while True:
        try:
            question = input("\nвы> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if question.lower() in ("", "exit", "quit", "выход"):
            break
        try:
            answer = ask(question)
        except LLMError as e:
            print(f"Ошибка: {e}", file=sys.stderr)  # в чате ошибка не должна завершать диалог
            continue
        print(f"\n{answer}")
        history += [{"role": "user", "content": question}, {"role": "assistant", "content": answer}]
        del history[:-20]  # помним последние 10 обменов, чтобы запрос не разрастался


def main() -> None:
    parser = argparse.ArgumentParser(description="ИИ-агент для анализа Python-проекта")
    parser.add_argument("path", nargs="?", default=".", help="Путь к проекту")
    parser.add_argument("--model", help="Модель (по умолчанию берётся из LLM_MODEL)")
    parser.add_argument("--output", metavar="FILE", help="Сохранить ответ в .md файл")
    parser.add_argument("--ask", "--task", dest="ask", metavar="ТЕКСТ",
                        help="Один вопрос или задание для агента")
    parser.add_argument("--chat", action="store_true", help="Диалог с агентом о проекте")
    parser.add_argument("--write", action="store_true",
                        help="Разрешить агенту изменять файлы (вместе с --ask или --chat)")
    parser.add_argument("--yes", action="store_true", help="Не спрашивать подтверждение изменений")
    parser.add_argument("--no-tools", action="store_true", help="Без инструментов: один запрос по статистике")
    parser.add_argument("--dry-run", action="store_true", help="Только показать данные, без запроса к API")
    args = parser.parse_args()

    if not Path(args.path).is_dir():
        sys.exit(f"Папка не найдена: {args.path}")

    if args.model:
        settings.llm_model = args.model  # аргумент командной строки важнее .env

    if args.write:
        settings.allow_write = True
    if args.yes:
        settings.confirm_writes = False
    if settings.allow_write and not (args.ask or args.chat):
        sys.exit("Изменение файлов работает только вместе с --ask/--task или --chat")

    if args.ask or args.chat:
        run_question_mode(args)
        return

    print("Анализирую проект...", file=sys.stderr)
    context = build_context(args.path, settings.analysis)

    if args.dry_run:
        text = json.dumps(context, ensure_ascii=False, indent=2)
        print(text)
        print(f"\n[Размер данных: {len(text)} символов]", file=sys.stderr)
        if not args.no_tools:
            registry = build_registry(args.path)
            print(f"[Инструменты агента: {', '.join(registry.names())}]", file=sys.stderr)
        return

    try:
        client = LLMClient(settings.llm_settings())  # настройки модели проверяются здесь
        if args.no_tools:
            print(f"Отправляю данные в {client.describe()}...", file=sys.stderr)
            answer = ask_llm(context, client)
        else:
            registry = build_registry(args.path, settings.max_tool_output_chars, settings.tool_timeout_s)
            print(f"Запускаю агента ({client.describe()}, до {settings.max_agent_steps} шагов)...", file=sys.stderr)
            result = run_agent(
                client, registry, AGENT_SYSTEM_PROMPT, context_message(context),
                max_steps=settings.max_agent_steps,
                on_tool=lambda run: print(format_run(run), file=sys.stderr),
            )
            answer = result.answer
            note = " (достигнут лимит шагов)" if result.hit_step_limit else ""
            print(f"Готово: {result.steps} обращений к модели, {len(result.tool_runs)} вызовов инструментов{note}",
                  file=sys.stderr)
    except (SettingsError, LLMError) as e:
        sys.exit(f"Ошибка: {e}")
    print(answer)

    if args.output:
        Path(args.output).write_text(answer, encoding="utf-8")
        print(f"\nСохранено: {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()