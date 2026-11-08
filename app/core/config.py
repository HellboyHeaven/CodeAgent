"""
config.py — единое место для всех настроек агента (pydantic-settings).

Значения читаются из переменных окружения и файла .env (регистр не важен:
LLM_MODEL == llm_model). Приоритет: окружение > .env > значения по умолчанию.
Неверные значения (например, MAX_ARGS=abc) дают понятную ошибку валидации при запуске.

Установка:
    pip install pydantic-settings

Использование:
    from config import settings
    settings.max_agent_steps
    settings.llm_settings()      # готовые настройки для LLMClient
"""

from __future__ import annotations

from dataclasses import dataclass, field

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class SettingsError(Exception):
    """Ошибка настройки LLM (не задана модель, ключ, неизвестный провайдер)."""


@dataclass
class LLMSettings:
    """Готовые параметры для LLMClient. Собираются методом Settings.llm_settings()."""
    provider: str                                           # 'anthropic' или 'openai'
    model: str
    api_key: str | None = field(default=None, repr=False)   # не попадёт в логи
    base_url: str | None = None
    max_tokens: int = 2500
    temperature: float | None = None

    def describe(self) -> str:
        """Краткое описание без секретов."""
        where = f" @ {self.base_url}" if self.base_url else ""
        return f"{self.provider}:{self.model}{where}"


@dataclass
class AnalysisSettings:
    """Параметры анализа проекта."""
    tree_depth: int = 3
    max_issues: int = 25
    max_function_lines: int = 50
    max_complexity: int = 10
    max_args: int = 5
    max_nesting: int = 4

    def thresholds(self) -> dict[str, int]:
        """Пороги словарём — для code_metrics.Thresholds(**...)."""
        return {
            "max_function_lines": self.max_function_lines,
            "max_complexity": self.max_complexity,
            "max_args": self.max_args,
            "max_nesting": self.max_nesting,
        }


class Settings(BaseSettings):
    # --- Модель. Все поля необязательны, чтобы --dry-run работал без ключей;
    # наличие модели и ключа проверяется в llm_settings() ---
    llm_provider: str | None = None        # openai | anthropic; если пусто — определяется сам
    llm_model: str | None = None
    llm_base_url: str | None = None        # для Ollama, OpenRouter и других OpenAI-совместимых
    llm_api_key: SecretStr | None = None   # общий ключ; иначе берутся два ключа ниже
    openai_api_key: SecretStr | None = None
    anthropic_api_key: SecretStr | None = None
    llm_max_tokens: int = 2500
    llm_temperature: float | None = None

    # --- Анализ проекта ---
    analysis_tree_depth: int = 3
    analysis_max_issues: int = 25
    max_function_lines: int = 50
    max_complexity: int = 10
    max_args: int = 5
    max_nesting: int = 4

    # --- Агентный цикл (понадобится на следующем шаге) ---
    # Жёсткий лимит обращений к модели и инструментам, защита от бесконечного цикла
    max_agent_steps: int = 6
    tool_timeout_s: float = 10.0
    max_tool_output_chars: int = 4000

    # --- Изменение файлов агентом ---
    allow_write: bool = False      # разрешить инструменты изменения (то же, что флаг --write)
    confirm_writes: bool = True    # спрашивать подтверждение перед каждым изменением
    max_write_steps: int = 15      # лимит шагов в режиме изменений (правки требуют больше шагов)

    # extra="ignore": в .env могут лежать чужие переменные (DATABASE_URL и т.п.),
    # иначе они вызвали бы ошибку валидации.
    # env_ignore_empty=True: пустая строка (LLM_MODEL=) считается «не задано».
    model_config = SettingsConfigDict(env_file=".env", extra="ignore", env_ignore_empty=True)

    @property
    def analysis(self) -> AnalysisSettings:
        return AnalysisSettings(
            tree_depth=self.analysis_tree_depth,
            max_issues=self.analysis_max_issues,
            max_function_lines=self.max_function_lines,
            max_complexity=self.max_complexity,
            max_args=self.max_args,
            max_nesting=self.max_nesting,
        )

    def llm_settings(self) -> LLMSettings:
        """Собирает и проверяет настройки модели. Бросает SettingsError, если чего-то не хватает."""
        if not self.llm_model:
            raise SettingsError(
                "Не задана модель. Укажите LLM_MODEL в окружении или в .env, "
                "например: LLM_MODEL=gpt-4o"
            )

        provider = (self.llm_provider or self._detect_provider()).lower()
        if provider not in ("anthropic", "openai"):
            raise SettingsError(f"Неизвестный LLM_PROVIDER '{provider}'. Допустимо: anthropic, openai")

        provider_key = self.anthropic_api_key if provider == "anthropic" else self.openai_api_key
        key = self.llm_api_key or provider_key

        return LLMSettings(
            provider=provider,
            model=self.llm_model,
            api_key=key.get_secret_value() if key else None,
            base_url=self.llm_base_url,
            max_tokens=self.llm_max_tokens,
            temperature=self.llm_temperature,
        )

    def _detect_provider(self) -> str:
        """Выбирает провайдера, если LLM_PROVIDER не задан."""
        if self.llm_base_url:
            return "openai"  # Ollama, OpenRouter и т.д. говорят на протоколе OpenAI
        if (self.llm_model or "").lower().startswith("claude") or self.anthropic_api_key:
            return "anthropic"
        return "openai"


settings = Settings()