"""OpenAI-совместимый адаптер модели — интерфейс §13 (ENVIRONMENT-V1, ADR-033).

Track-2 Stage A: обкатка среды на готовой открытой модели через внешний
inference-endpoint (llama-server + GGUF), без тренировки.  Адаптер переводит
контракт роллаутов §13::

    generate(messages, seed, max_tokens) -> {text, token_ids, behavior_logprobs}

в POST ``{base_url}/chat/completions`` совместимого сервера.  Сообщения
передаются **как есть** (формат v12 §13 совпадает с форматом чата), декодирование
детерминировано: ``temperature=0`` и явный ``seed``.

Границы (порт и адаптер)
------------------------

Этот модуль — **порт**: политика (формат запроса, таймауты, ретраи, разбор
ответа, коды ошибок) без привязки к HTTP-библиотеке.  Механизм — конкретный
транспорт — поставляется вызывающим (``transport=``), потому что пакет ``env/``
признан контуром награды RL (C-039/AD-2): импорт HTTP-клиента сюда недопустим.
Готовый транспорт на httpx с ``trust_env=False`` живёт вне контура —
:mod:`clients.openai_http`; он же даёт фабрику для калибровки.

Таймауты по p99.9: короткий connect и длинный read (генерация длинного хода).
Ретрай — один, только на ошибку установления соединения; повтор чтения после
таймаута не делается (правило контура: таймауты по p99.9, ретраи в одном слое).
Запасных endpoint'ов нет (fallback расширяет аварию).

Почему результат переиспользует :class:`~env.jaxlm_adapter.GenerationResult`:
интерфейс §13 обязан быть ИДЕНТИЧНЫМ прибору JaxLMAdapter, и структура ответа
(состав/порядок полей) уже запиннена тестом структуры.  Для внешней модели
``behavior_logprobs`` — справочные; если endpoint их не отдаёт, возвращается
пустой список (не выдуманные числа), а не ``None``: журнал роллаута итерирует
поле (``list(gen.behavior_logprobs)``), ``None`` уронил бы эпизод.
"""

from __future__ import annotations

import json
from typing import Any, Optional, Protocol, Sequence, runtime_checkable

from .jaxlm_adapter import RESULT_FIELDS, GenerationResult

# ── Конфигурация по умолчанию ──────────────────────────────────────────────
# llama-server по умолчанию слушает локальный порт 8080 и открывает
# OpenAI-совместимый префикс ``/v1`` (ключ не требуется).
DEFAULT_BASE_URL = "http://127.0.0.1:8080/v1"

#: Имя модели по умолчанию в теле запроса: llama-server модель из поля не
#: выбирает (она задана запуском), но схема чата требует непустое значение.
DEFAULT_SERVED_MODEL = "local-model"

#: Таймауты по p99.9: установление соединения к локальному серверу — секунды,
#: чтение ответа — генерация хода может идти минуты, поэтому лимиты разнесены.
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


# ── Порт транспорта ────────────────────────────────────────────────────────
# Транспорт поставляет вызывающий (вне контура награды). Он обязан перевести
# сбои своей библиотеки в эти типы, чтобы политика ретраев осталась здесь.


class TransportError(Exception):
    """Сбой транспорта (базовый)."""


class TransportConnectError(TransportError):
    """Не удалось установить соединение (единственный ретраируемый случай)."""


class TransportReadTimeout(TransportError):
    """Ответ не пришёл в лимит чтения (не ретраируется)."""


@runtime_checkable
class HttpResponse(Protocol):
    """Минимальный ответ транспорта (совместим с ``httpx.Response``)."""

    status_code: int
    text: str

    def json(self) -> Any:  # pragma: no cover - протокол
        ...


@runtime_checkable
class HttpTransport(Protocol):
    """Минимальный POST-транспорт (совместим с httpx-обёрткой clients/)."""

    def post(self, url: str, *, json: dict[str, Any], headers: dict[str, str]) -> HttpResponse:
        ...  # pragma: no cover - протокол


def build_payload(
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
        # Поэлементные logprobs: endpoint без поддержки просто не вернёт блок.
        "logprobs": True,
        "stream": False,
    }


def extract_text_and_logprobs(data: dict[str, Any]) -> tuple[str, list[float]]:
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

    Сети в конструкторе не трогает: HTTP делает переданный ``transport``.
    Без транспорта генерация честно отказывает
    :class:`OpenAIAdapterError` (а не падает ImportError'ом библиотеки).
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
        transport: Optional[HttpTransport] = None,
    ) -> None:
        self.base_url = str(base_url).rstrip("/")
        self.model = str(model)
        self._api_key = api_key
        self.seed = int(seed)
        self.temperature = float(temperature)
        self.connect_timeout = float(connect_timeout)
        self.read_timeout = float(read_timeout)
        self._policy_version = policy_version
        self.transport = transport

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
        payload = build_payload(
            self.model, messages, effective_seed, int(max_tokens), self.temperature
        )
        data = self._post_chat(payload)
        text, logprobs = extract_text_and_logprobs(data)
        # Точных token_ids схема ответа чата не отдаёт — фиксируем
        # детерминированную прокси-разбивку (см. encode); behavior_logprobs
        # собираются, только если endpoint прислал блок logprobs.
        return GenerationResult(
            text=text,
            token_ids=self.encode(text),
            behavior_logprobs=logprobs,
            policy_version=self.policy_version,
        )

    def close(self) -> None:
        close = getattr(self.transport, "close", None)
        if callable(close):
            close()

    def __enter__(self) -> "OpenAIAdapter":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- транспорт ----------------------------------------------------------

    def _post_chat(self, payload: dict[str, Any]) -> dict[str, Any]:
        """POST ``/chat/completions``: один ретрай на connect, без fallback."""
        if self.transport is None:
            raise OpenAIAdapterError(
                "не задан HTTP-транспорт: пакет env/ — контур награды (C-039), "
                "HTTP-клиент сюда не импортируется. Передайте transport= "
                "(готовый httpx-транспорт — clients.openai_http.HttpxTransport)."
            )
        url = f"{self.base_url}/chat/completions"
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        attempts = 0
        while True:
            try:
                response = self.transport.post(url, json=payload, headers=headers)
            except TransportConnectError as exc:
                if attempts < CONNECT_RETRIES:
                    attempts += 1
                    continue
                raise OpenAIUnavailableError(
                    f"не удалось установить соединение с {self.base_url} "
                    f"(после {CONNECT_RETRIES + 1} попыток): {exc}"
                ) from exc
            except TransportReadTimeout as exc:
                raise OpenAIReadTimeoutError(
                    f"чтение ответа {self.base_url} превысило "
                    f"{self.read_timeout:.0f} с (p99.9); повтор не делается"
                ) from exc
            except TransportError as exc:
                raise OpenAIAdapterError(
                    f"сбой транспорта при обращении к {self.base_url}: {exc}"
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
    "HttpResponse",
    "HttpTransport",
    "OpenAIAdapter",
    "OpenAIAdapterError",
    "OpenAIReadTimeoutError",
    "OpenAIResponseError",
    "OpenAIUnavailableError",
    "RESULT_FIELDS",
    "TIMEOUT_CONNECT_SECS",
    "TIMEOUT_READ_SECS",
    "TransportConnectError",
    "TransportError",
    "TransportReadTimeout",
    "build_payload",
    "extract_text_and_logprobs",
]
