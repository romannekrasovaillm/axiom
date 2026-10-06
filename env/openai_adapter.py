"""OpenAI-совместимый адаптер модели — интерфейс §13 (ENVIRONMENT-V1, ADR-033).

Track-2 Stage A: локальный ``llama-server`` + GGUF открытой instruct-модели
(inference-only обкатка среды, калибровка лесенки L0–L3).  Адаптер переводит
контракт роллаутов §13::

    generate(messages, seed, max_tokens) -> {text, token_ids, behavior_logprobs}

в POST ``{base_url}/chat/completions`` совместимого сервера.  Сообщения
передаются **как есть** (формат v12 §13 уже совпадает с OpenAI chat-форматом),
декодирование детерминировано: ``temperature=0`` и явный ``seed``.

Почему формат результата переиспользует :class:`~env.jaxlm_adapter.GenerationResult`
(а не свой): интерфейс §13 обязан быть ИДЕНТИЧНЫМ прибору JaxLMAdapter, и
структура ответа (состав/порядок полей) уже запиннена тестом структуры.  Для
внешней модели ``behavior_logprobs`` — справочные (тренировка своей политики
идёт на своём чекпойнте); если endpoint их не отдаёт — возвращается пустой
список, а не выдуманные числа.

Гигиена сети (C-011): клиент создаётся с ``trust_env=False`` — прокси-остатки
окружения (``HTTP_PROXY``/``ALL_PROXY``) НЕ наследуются.  Таймауты осознаны по
p99.9: короткий connect и длинный read (генерация длинного хода).  Ретрай —
один, только на ошибку установления соединения; повтор чтения после таймаута не
делается (правило контура: таймауты по p99.9, ретраи в одном слое).  Запасных
endpoint'ов нет (fallback запрещён — расширяет аварию).
"""

from __future__ import annotations

import json
from typing import Any, Optional, Sequence

import httpx

from .jaxlm_adapter import RESULT_FIELDS, GenerationResult

# ── Конфигурация по умолчанию ──────────────────────────────────────────────
# llama-server по умолчанию слушает локальный порт 8080 и открывает
# OpenAI-совместимый префикс ``/v1`` (ключ не требуется).
DEFAULT_BASE_URL = "http://127.0.0.1:8080/v1"

#: Имя модели по умолчанию в теле запроса: llama-server модель из поля не
#: выбирает (она задана запуском), но OpenAI-схема требует непустое значение.
DEFAULT_SERVED_MODEL = "local-model"

#: Таймауты по p99.9 (комментарий обязателен — правило контура): установление
#: соединения к локальному серверу — секунды, чтение ответа — генерация хода
#: может идти минуты, поэтому лимиты разнесены.
TIMEOUT_CONNECT_SECS = 5.0
TIMEOUT_READ_SECS = 120.0

#: Один повтор на ошибку установления соединения (не на чтение).
CONNECT_RETRIES = 1


class OpenAIAdapterError(RuntimeError):
    """Базовая ошибка адаптера внешней модели (понятный текст, не traceback)."""


class OpenAIUnavailableError(OpenAIAdapterError):
    """Endpoint недоступен (connect-ошибка исчерпала единственный ретрай)."""


class OpenAIReadTimeoutError(OpenAIAdapterError):
    """Чтение ответа превысило p99.9-лимит; повтор сознательно не делается."""


class OpenAIResponseError(OpenAIAdapterError):
    """Endpoint ответил не-2xx или телом, которое не разбирается как ответ чата."""


def _build_payload(
    model: str,
    messages: Sequence[dict[str, str]],
    seed: int,
    max_tokens: int,
    temperature: float,
) -> dict[str, Any]:
    """Тело запроса ``chat/completions`` (детерминированное, без потока)."""
    return {
        "model": model,
        # Сообщения §13 — как есть (system/user/assistant/tool).
        "messages": [dict(m) for m in messages],
        # temperature=0 — greedy: детерминизм на фиксированной модели.
        "temperature": float(temperature),
        # seed прокидывается в sampling-параметры, если endpoint его понимает.
        "seed": int(seed),
        "max_tokens": int(max_tokens),
        # llama-server поддерживает поэлементные logprobs — просим их; endpoint
        # без поддержки просто не вернёт блок (обрабатывается в _extract_logprobs).
        "logprobs": True,
        "stream": False,
    }


def _extract_text_and_logprobs(data: dict[str, Any]) -> tuple[str, list[float]]:
    """Разбор ответа чата: текст хода и (если есть) поэлементные logprobs."""
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        raise OpenAIResponseError("ответ endpoint без непустого поля 'choices'")
    choice = choices[0]
    if not isinstance(choice, dict):
        raise OpenAIResponseError("элемент 'choices[0]' не объект")
    message = choice.get("message")
    if not isinstance(message, dict):
        raise OpenAIResponseError("ответ endpoint без 'choices[0].message'")
    content = message.get("content")
    text = "" if content is None else str(content)

    logprobs_block = choice.get("logprobs")
    values: list[float] = []
    if isinstance(logprobs_block, dict):
        content_items = logprobs_block.get("content")
        if isinstance(content_items, list):
            for item in content_items:
                if isinstance(item, dict) and item.get("logprob") is not None:
                    values.append(float(item["logprob"]))
    return text, values


class OpenAIAdapter:
    """Адаптер §13 поверх OpenAI-совместимого endpoint (внешняя модель).

    Конструирование клиента — ленивое по отношению к сети (никаких вызовов в
    ``__init__``).  Для тестов принимается готовый ``transport`` (например,
    ``httpx.MockTransport``) или собранный ``client`` — тогда сеть не трогается.
    """

    def __init__(
        self,
        *,
        base_url: str = DEFAULT_BASE_URL,
        model: str = DEFAULT_SERVED_MODEL,
        api_key: Optional[str] = None,
        seed: int = 0,
        temperature: float = 0.0,
        connect_timeout: float = TIMEOUT_CONNECT_SECS,
        read_timeout: float = TIMEOUT_READ_SECS,
        policy_version: Optional[str] = None,
        transport: Optional[httpx.BaseTransport] = None,
        client: Optional[httpx.Client] = None,
    ) -> None:
        self.base_url = str(base_url).rstrip("/")
        self.model = str(model)
        self._api_key = api_key
        self.seed = int(seed)
        self.temperature = float(temperature)
        self.connect_timeout = float(connect_timeout)
        self.read_timeout = float(read_timeout)
        self._policy_version = policy_version

        if client is not None and transport is not None:
            raise ValueError("укажите либо client, либо transport, не оба")
        if client is not None:
            self._client = client
        else:
            # trust_env=False — не наследовать прокси окружения (C-011).
            timeout = httpx.Timeout(
                self.read_timeout,
                connect=self.connect_timeout,
                write=self.connect_timeout,
                pool=self.connect_timeout,
            )
            self._client = httpx.Client(
                transport=transport, timeout=timeout, trust_env=False
            )

    # -- §13 ----------------------------------------------------------------

    @property
    def policy_version(self) -> str:
        """Предмет замера: имя внешней модели (у адаптера нет локального веса)."""
        if self._policy_version:
            return self._policy_version
        return f"openai-{self.model}"

    @property
    def checkpoint_sha256(self) -> str:
        """Локального чекпойнта нет — пиннинг AD-4 пуст (внешняя модель)."""
        return ""

    def encode(self, text: str) -> list[int]:
        """Прокси-токенизация для учёта бюджета хода (детерминированная).

        Настоящий токенизатор живёт на сервере; журналу роллаута нужны длины
        промптов/наблюдений (бюджет), а не точная разбивка.  UTF-8-байты —
        детерминированная и стабильная аппроксимация.
        """
        return list(str(text).encode("utf-8"))

    def decode(self, ids: Sequence[int]) -> str:
        """Обратная к :meth:`encode` аппроксимация (для симметрии интерфейса)."""
        return bytes(int(i) & 0xFF for i in ids).decode("utf-8", errors="replace")

    def generate(
        self,
        messages: list[dict[str, str]],
        seed: int | None = None,
        max_tokens: int = 256,
    ) -> GenerationResult:
        """Один ход ассистента через внешний endpoint (§13)."""
        effective_seed = self.seed if seed is None else int(seed)
        payload = _build_payload(
            self.model, messages, effective_seed, int(max_tokens), self.temperature
        )
        data = self._post_chat(payload)
        text, logprobs = _extract_text_and_logprobs(data)
        # Точных token_ids сервер в OpenAI-схеме не отдаёт — фиксируем
        # детерминированную прокси-разбивку (см. encode); behavior_logprobs
        # собираются, только если endpoint прислал блок logprobs.
        return GenerationResult(
            text=text,
            token_ids=self.encode(text),
            behavior_logprobs=logprobs,
            policy_version=self.policy_version,
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "OpenAIAdapter":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- транспорт ----------------------------------------------------------

    def _post_chat(self, payload: dict[str, Any]) -> dict[str, Any]:
        """POST ``/chat/completions``: один ретрай на connect, без fallback."""
        url = f"{self.base_url}/chat/completions"
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        attempts = 0
        while True:
            try:
                response = self._client.post(url, json=payload, headers=headers)
            except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
                if attempts < CONNECT_RETRIES:
                    attempts += 1
                    continue
                raise OpenAIUnavailableError(
                    f"не удалось установить соединение с {self.base_url} "
                    f"(после {CONNECT_RETRIES + 1} попыток): {exc}"
                ) from exc
            except httpx.ReadTimeout as exc:
                raise OpenAIReadTimeoutError(
                    f"чтение ответа {self.base_url} превысило "
                    f"{self.read_timeout:.0f} с (p99.9); повтор не делается"
                ) from exc
            except httpx.TimeoutException as exc:  # запись/пул — тоже отказ
                raise OpenAIAdapterError(
                    f"таймаут обращения к {self.base_url}: {exc}"
                ) from exc
            break

        if response.status_code >= 400:
            body = response.text[:200]
            raise OpenAIResponseError(
                f"endpoint {self.base_url} вернул HTTP {response.status_code}: {body}"
            )
        try:
            data = response.json()
        except (ValueError, json.JSONDecodeError) as exc:
            raise OpenAIResponseError(
                f"ответ endpoint {self.base_url} не разбирается как JSON"
            ) from exc
        if not isinstance(data, dict):
            raise OpenAIResponseError("ответ endpoint не JSON-объект")
        return data


__all__ = [
    "CONNECT_RETRIES",
    "DEFAULT_BASE_URL",
    "DEFAULT_SERVED_MODEL",
    "OpenAIAdapter",
    "OpenAIAdapterError",
    "OpenAIReadTimeoutError",
    "OpenAIResponseError",
    "OpenAIUnavailableError",
    "RESULT_FIELDS",
    "TIMEOUT_CONNECT_SECS",
    "TIMEOUT_READ_SECS",
]
