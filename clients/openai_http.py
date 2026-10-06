"""Конкретный HTTP-транспорт для порта ``env.openai_adapter`` (track-2 Stage A).

Пакет ``env/`` — контур награды RL (C-039/AD-2): импорт HTTP-клиента внутрь
него запрещён гейтом.  Поэтому механизм (httpx) живёт здесь, а политика
(формат запроса, таймауты, ретраи, разбор ответа) — в
:mod:`env.openai_adapter`.  Транспорт переводит сбои httpx в типы порта
(``TransportConnectError`` / ``TransportReadTimeout`` / ``TransportError``),
чтобы решение о ретрае принимал порт, а не библиотека.

Гигиена сети (C-011): клиент создаётся с ``trust_env=False`` — прокси-остатки
окружения (``HTTP_PROXY``/``ALL_PROXY``) не наследуются.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

import httpx

from env.openai_adapter import (
    DEFAULT_BASE_URL,
    DEFAULT_SERVED_MODEL,
    TIMEOUT_CONNECT_SECS,
    TIMEOUT_READ_SECS,
    OpenAIAdapter,
    TransportConnectError,
    TransportError,
    TransportReadTimeout,
)


class HttpxTransport:
    """POST-транспорт на httpx с ``trust_env=False`` и p99.9-таймаутами.

    ``transport`` — подменяемый нижележащий httpx-транспорт (тесты передают
    ``httpx.MockTransport``); ``None`` — обычная сеть.
    """

    def __init__(
        self,
        *,
        connect_timeout: float = TIMEOUT_CONNECT_SECS,
        read_timeout: float = TIMEOUT_READ_SECS,
        trust_env: bool = False,
        transport: Optional[httpx.BaseTransport] = None,
    ) -> None:
        timeout = httpx.Timeout(
            read_timeout,
            connect=connect_timeout,
            write=connect_timeout,
            pool=connect_timeout,
        )
        self._client = httpx.Client(
            timeout=timeout, trust_env=trust_env, transport=transport
        )

    @property
    def client(self) -> httpx.Client:
        """Нижележащий клиент (для инспекции настроек, например trust_env)."""
        return self._client

    def post(
        self, url: str, *, json: dict[str, Any], headers: dict[str, str]
    ) -> httpx.Response:
        try:
            return self._client.post(url, json=json, headers=headers)
        except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
            raise TransportConnectError(str(exc)) from exc
        except httpx.ReadTimeout as exc:
            raise TransportReadTimeout(str(exc)) from exc
        except httpx.TimeoutException as exc:  # запись/пул — отказ, но не ретраим
            raise TransportError(str(exc)) from exc

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "HttpxTransport":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def build_openai_factory(
    *,
    base_url: str = DEFAULT_BASE_URL,
    model: str = DEFAULT_SERVED_MODEL,
    api_key: Optional[str] = None,
    seed: int = 0,
    transport: Optional[Any] = None,
) -> Callable[[], OpenAIAdapter]:
    """Фабрика эпизода: свежий адаптер §13 на задачу (шов ``model_factory``).

    ``transport`` — заранее собранный транспорт (тесты/повторное
    использование).  ``None`` — на каждый эпизод создаётся свой
    :class:`HttpxTransport` (изоляция соединений между задачами).
    """

    def factory() -> OpenAIAdapter:
        active = transport if transport is not None else HttpxTransport()
        return OpenAIAdapter(
            base_url=base_url,
            model=model,
            api_key=api_key,
            seed=seed,
            transport=active,
        )

    return factory


__all__ = ["HttpxTransport", "build_openai_factory"]
