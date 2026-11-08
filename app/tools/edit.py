"""
tools/edit.py — инструменты изменения проекта: создание папок и файлов, правка кода,
переименование, перемещение, удаление.

Меры безопасности:
  * Всё происходит внутри проекта; корень, .git и .agent_backup менять нельзя.
  * Файлы с секретами (.env, ключи) не изменяются.
  * Перед каждым действием вызывается ctx.approve(...) — подтверждение пользователя.
  * Удаление не стирает данные: файл переезжает в .agent_backup/<сессия>/deleted/...
    Перезаписанные и изменённые файлы сохраняются в исходном виде в .agent_backup/<сессия>/changed/...
  * Для .py-файлов после записи проверяется синтаксис, и модель получает предупреждение.
"""

from __future__ import annotations

import ast
import difflib
import functools
import shutil
from pathlib import Path

from .base import Tool, ToolContext, ToolError, schema

MAX_WRITE_CHARS = 300_000
MAX_EDIT_BYTES = 1_000_000
PREVIEW_LINES = 25
DIFF_LINES = 80


# --- Вспомогательные функции ---

def fs_errors(func):
    """Ошибки файловой системы превращаем в понятный для модели ToolError."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except OSError as e:
            raise ToolError(f"Ошибка файловой системы: {e}") from e
    return wrapper


def _backup_root(ctx: ToolContext) -> Path:
    return ctx.root / ".agent_backup" / ctx.stamp


def _backup_copy(ctx: ToolContext, path: Path) -> None:
    """Сохраняет исходное состояние файла (один раз за сессию)."""
    dest = _backup_root(ctx) / "changed" / ctx.rel(path)
    if dest.exists():
        return  # оригинал уже сохранён раньше, не затираем его промежуточной версией
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, dest)


def _unique(dest: Path) -> Path:
    if not dest.exists():
        return dest
    n = 1
    while dest.with_name(f"{dest.name}.{n}").exists():
        n += 1
    return dest.with_name(f"{dest.name}.{n}")


def _diff(old: str, new: str, name: str) -> str:
    lines = list(difflib.unified_diff(
        old.splitlines(), new.splitlines(), f"{name} (было)", f"{name} (стало)", lineterm="", n=2))
    text = "\n".join(lines[:DIFF_LINES])
    if len(lines) > DIFF_LINES:
        text += f"\n... ещё {len(lines) - DIFF_LINES} строк изменений"
    return text


def _preview(content: str) -> str:
    lines = content.splitlines()
    text = "\n".join(lines[:PREVIEW_LINES])
    if len(lines) > PREVIEW_LINES:
        text += f"\n... ещё {len(lines) - PREVIEW_LINES} строк"
    return text


def _read_text(ctx: ToolContext, path: Path) -> str:
    if not path.is_file():
        raise ToolError(f"Это не файл: {ctx.rel(path)}")
    if path.stat().st_size > MAX_EDIT_BYTES:
        raise ToolError("Файл слишком большой (больше 1 МБ)")
    data = path.read_bytes()
    if b"\0" in data:
        raise ToolError("Бинарный файл, изменить нельзя")
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        raise ToolError("Файл не в кодировке UTF-8, изменить нельзя")


def _syntax_warning(ctx: ToolContext, path: Path, text: str) -> str:
    if path.suffix != ".py":
        return ""
    try:
        ast.parse(text)
    except SyntaxError as e:
        return f"\nВНИМАНИЕ: синтаксическая ошибка в {ctx.rel(path)}, строка {e.lineno}: {e.msg}. Исправь её."
    return ""


# --- Инструменты ---

@fs_errors
def create_dir(ctx: ToolContext, path: str) -> str:
    target = ctx.resolve_new(path)
    rel = ctx.rel(target)
    if target.is_dir():
        return f"Папка уже существует: {rel}/"
    if target.exists():
        raise ToolError(f"По этому пути уже есть файл: {rel}")
    ctx.approve(f"Создать папку {rel}/")
    target.mkdir(parents=True)
    return f"Создана папка {rel}/"


@fs_errors
def write_file(ctx: ToolContext, path: str, content: str, overwrite: bool = False) -> str:
    if len(content) > MAX_WRITE_CHARS:
        raise ToolError(f"Слишком большое содержимое ({len(content)} символов, максимум {MAX_WRITE_CHARS})")
    target = ctx.resolve_new(path)
    rel = ctx.rel(target)
    if target.is_dir():
        raise ToolError(f"Это папка, а не файл: {rel}")
    if content and not content.endswith("\n"):
        content += "\n"

    if target.exists():
        if not overwrite:
            raise ToolError(
                f"Файл уже существует: {rel}. Для полной замены передай overwrite=true, "
                "для точечной правки используй edit_file")
        old = _read_text(ctx, target)
        ctx.approve(f"Перезаписать файл {rel}", _diff(old, content, rel))
        _backup_copy(ctx, target)
        verb = "Перезаписан"
    else:
        ctx.approve(f"Создать файл {rel} ({len(content.splitlines())} строк)", _preview(content))
        verb = "Создан"

    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8", newline="\n")
    return f"{verb} файл {rel} ({len(content.splitlines())} строк){_syntax_warning(ctx, target, content)}"


@fs_errors
def edit_file(ctx: ToolContext, path: str, old_text: str, new_text: str, replace_all: bool = False) -> str:
    target = ctx.resolve_new(path)
    rel = ctx.rel(target)
    if not target.exists():
        raise ToolError(f"Файл не найден: {rel}. Чтобы создать новый, используй write_file")
    if not old_text:
        raise ToolError("old_text не может быть пустым")

    text = _read_text(ctx, target)
    if "\r\n" in text:  # сохраняем Windows-переносы строк
        old_text = old_text.replace("\r\n", "\n").replace("\n", "\r\n")
        new_text = new_text.replace("\r\n", "\n").replace("\n", "\r\n")

    count = text.count(old_text)
    if count == 0:
        raise ToolError(
            "old_text не найден в файле. Проверь пробелы и отступы: фрагмент должен совпадать "
            f"с файлом символ в символ. Перечитай нужное место через read_file ({rel}).")
    if count > 1 and not replace_all:
        raise ToolError(
            f"old_text встречается {count} раз. Добавь вокруг него больше соседнего кода, "
            "чтобы фрагмент был уникальным, или передай replace_all=true")

    new_content = text.replace(old_text, new_text) if replace_all else text.replace(old_text, new_text, 1)
    if new_content == text:
        return "Изменений нет: новый текст совпадает со старым"

    ctx.approve(f"Изменить файл {rel}", _diff(text, new_content, rel))
    _backup_copy(ctx, target)
    target.write_text(new_content, encoding="utf-8", newline="")
    return f"Изменён файл {rel}: заменено вхождений — {count if replace_all else 1}{_syntax_warning(ctx, target, new_content)}"


@fs_errors
def move_path(ctx: ToolContext, source: str, destination: str) -> str:
    src = ctx.resolve_new(source)
    dst = ctx.resolve_new(destination)
    src_rel, dst_rel = ctx.rel(src), ctx.rel(dst)
    if not src.exists():
        raise ToolError(f"Не найдено: {src_rel}")
    if dst.exists():
        raise ToolError(f"Назначение уже существует: {dst_rel}. Выбери другое имя или сначала удали его")
    if dst.is_relative_to(src):
        raise ToolError("Нельзя переместить папку внутрь самой себя")

    ctx.approve(f"Переместить/переименовать {src_rel} → {dst_rel}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dst))

    note = ""
    if dst.is_dir() or dst.suffix == ".py":
        note = ("\nИмпорты в коде автоматически НЕ обновлены: найди старые пути через "
                "search_text или find_usages и поправь их через edit_file.")
    return f"Перемещено: {src_rel} → {dst_rel}{note}"


@fs_errors
def delete_path(ctx: ToolContext, path: str, recursive: bool = False) -> str:
    target = ctx.resolve_new(path)
    rel = ctx.rel(target)
    if not target.exists():
        raise ToolError(f"Не найдено: {rel}")

    if target.is_dir():
        files = sum(1 for p in target.rglob("*") if p.is_file())
        if any(target.iterdir()) and not recursive:
            raise ToolError(f"Папка не пуста ({files} файлов). Для удаления вместе с содержимым передай recursive=true")
        what = f"папку {rel}/ ({files} файлов)"
    else:
        what = f"файл {rel} ({target.stat().st_size} Б)"

    backup = _unique(_backup_root(ctx) / "deleted" / rel)
    ctx.approve(f"Удалить {what}", f"Копия сохранится в {ctx.rel(backup)}")
    backup.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(target), str(backup))
    return f"Удалено: {rel}. Резервная копия: {ctx.rel(backup)}"


TOOLS = [
    Tool(
        "create_dir",
        "Создаёт папку (вместе с недостающими родительскими).",
        schema({"path": {"type": "string", "description": "Путь папки относительно корня проекта"}}, ["path"]),
        create_dir,
        mutating=True,
    ),
    Tool(
        "write_file",
        "Создаёт новый файл с заданным содержимым (родительские папки создаются сами). "
        "Для кода пиши полный рабочий текст с импортами. Существующий файл заменяется "
        "только при overwrite=true; для точечных правок лучше edit_file.",
        schema({
            "path": {"type": "string", "description": "Путь файла относительно корня проекта"},
            "content": {"type": "string", "description": "Полное содержимое файла"},
            "overwrite": {"type": "boolean", "description": "Разрешить замену существующего файла"},
        }, ["path", "content"]),
        write_file,
        mutating=True,
    ),
    Tool(
        "edit_file",
        "Правит существующий файл: заменяет фрагмент old_text на new_text. old_text должен "
        "совпадать с файлом символ в символ (включая отступы) и быть уникальным, иначе добавь "
        "соседние строки. Перед правкой прочитай нужное место через read_file.",
        schema({
            "path": {"type": "string", "description": "Путь файла"},
            "old_text": {"type": "string", "description": "Заменяемый фрагмент, точно как в файле"},
            "new_text": {"type": "string", "description": "Новый текст (пустая строка удаляет фрагмент)"},
            "replace_all": {"type": "boolean", "description": "Заменить все вхождения, а не одно"},
        }, ["path", "old_text", "new_text"]),
        edit_file,
        mutating=True,
    ),
    Tool(
        "move_path",
        "Переименовывает или перемещает файл либо папку. Назначение не должно существовать. "
        "Импорты в коде не обновляются, это нужно сделать отдельно.",
        schema({
            "source": {"type": "string", "description": "Что перемещаем"},
            "destination": {"type": "string", "description": "Новый путь (для переименования — новое имя)"},
        }, ["source", "destination"]),
        move_path,
        mutating=True,
    ),
    Tool(
        "delete_path",
        "Удаляет файл или папку. Данные не стираются навсегда: копия сохраняется в .agent_backup. "
        "Непустую папку можно удалить только с recursive=true.",
        schema({
            "path": {"type": "string", "description": "Что удалить"},
            "recursive": {"type": "boolean", "description": "Удалить папку вместе с содержимым"},
        }, ["path"]),
        delete_path,
        mutating=True,
    ),
]
