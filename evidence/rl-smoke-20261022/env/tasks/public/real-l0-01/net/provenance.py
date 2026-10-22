"""Провенанс-метки диагностики шага — gated name scopes (ADR-054 Am.31, цикл-19).

Зачем.  HLO/nsys-трейс клетки B=4 показывает хвост ``command_buffer_1229/985``
(9.75 с GPU, 512 графов, 57.9k ядер), но не отвечает на вопрос, **какому
компоненту** принадлежат эти ядра: имена XLA-операций несут только форму
выражения (``jit(step)/mul``), а границы слоёв в них не видны.  Гейт
``AXIOM_TRACE_MARKERS=1`` оборачивает *границы вызовов* компонентов
(:func:`mark`) в ``jax.named_call``: имя scope попадает в ``op_name`` каждой
HLO-операции внутри как ``probe_<name>`` (``probe_kda``, ``probe_mla``,
``probe_moe``, ``probe_ce``, ...), а по ``op_name`` ядра трейса и графы
command buffer'ов относятся к компоненту механически, без догадок.

Почему выключенный гейт не меняет граф.  При ``off`` (дефолт) :func:`mark`
делает **прямой** вызов ``fn(*args, **kwargs)``: ни ``jax.named_call``, ни
какого-либо другого примитива в jaxpr не добавляется.  Граф, его фьюзинг,
порядок операций и числа побитово прежние — метка это чистая диагностика,
а не изменение вычисления.

Почему fail-closed.  Допустимы только ``"0"``/``"1"`` (пусто/не задано =
``"0"``); неизвестное значение — :class:`TraceMarkersError` на трейсе, без
тихого отката к ``off``.  Опечатка в имени переменной или в значении не
должна дать трейс, который выглядит размеченным, а на деле снят с прежнего
(немаркированного) графа: такой замер был бы замером другого объекта.
Форма — как у ``net.mla.persist_mode``/``flash_bwd_mode``.
"""

from __future__ import annotations

import os

import jax

#: Env-гейт провенанс-меток (``0`` — дефолт, граф побитово прежний).
TRACE_MARKERS_ENV = "AXIOM_TRACE_MARKERS"

#: Допустимые значения; неизвестное — ошибка (fail-closed, без догадок).
TRACE_MARKERS_VALUES = ("0", "1")
TRACE_MARKERS_OFF = "0"
TRACE_MARKERS_ON = "1"

#: Префикс имени scope в HLO (``probe_<name>``) — по нему трейс ищет компонент.
TRACE_MARKER_PREFIX = "probe_"


class TraceMarkersError(ValueError):
    """Неизвестное значение ``AXIOM_TRACE_MARKERS`` (fail-closed, без отката)."""


def markers_mode(value: str | None = None) -> bool:
    """Провенанс-метки: ``False`` (дефолт) | ``True``.

    ``AXIOM_TRACE_MARKERS=1`` оборачивает границы вызовов компонентов в
    ``jax.named_call`` с именем ``probe_<name>``; ``0``/пусто/не задано —
    прямой вызов (граф побитово прежний).  Значения — только ``"0"``/``"1"``,
    регистр и пробелы нормализуются; неизвестное — :class:`TraceMarkersError`
    (fail-closed: опечатка не должна тихо оставить немаркированный граф —
    трейс выдал бы себя за размеченный).  Явный аргумент сильнее окружения
    (форма ``net.mla.persist_mode``).  Читается на трейсе, чтобы тест мог
    переключить гейт в процессе.
    """
    raw = value if value is not None else os.environ.get(TRACE_MARKERS_ENV)
    mode = (raw or "").strip().lower() or TRACE_MARKERS_OFF
    if mode not in TRACE_MARKERS_VALUES:
        raise TraceMarkersError(
            f"{TRACE_MARKERS_ENV}={mode!r}: ожидается одно из "
            f"{list(TRACE_MARKERS_VALUES)}"
        )
    return mode == TRACE_MARKERS_ON


def mark(name: str, fn, *args, **kwargs):
    """Вызвать ``fn`` в провенанс-scope ``probe_<name>`` под гейтом.

    ``off`` (дефолт) — прямой ``fn(*args, **kwargs)``: ничего не оборачивается,
    jaxpr/граф/формы/числа побитово прежние (метка — диагностика, не операция).
    ``on`` — ``jax.named_call(fn, name="probe_<name>")(*args, **kwargs)``: имя
    scope уходит в ``op_name`` HLO-операций (и далее в имена ядер трейса),
    поэтому component-атрибуция command buffer'ов становится механической.
    Неизвестное значение гейта — :class:`TraceMarkersError` до трейса
    (fail-closed, см. :func:`markers_mode`).
    """
    if not markers_mode():
        return fn(*args, **kwargs)
    return jax.named_call(fn, name=f"{TRACE_MARKER_PREFIX}{name}")(*args, **kwargs)
