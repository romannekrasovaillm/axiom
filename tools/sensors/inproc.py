"""Встраиваемые датчики учебного цикла (дельта C5, ADR-037).

Библиотека для вставки в цикл обучения: ``PhaseTimer`` (S-023) и
``DeviceMemory`` (S-024). Правится только этот модуль — ``tools/pretrain_run.py``
и ``net/`` линии-1 не трогаются; интеграция (когда дельта фазовых таймеров
линии-1 будет влита) переводит их выход в контракт фактов через :meth:`PhaseTimer.commit`.

Границы фаз — через ``jax.block_until_ready``, иначе таймер мерит постановку в
очередь, а не исполнение. Режим выборки (каждый K-й шаг) снижает искажение
скорости синхронизациями; накладные расходы таймера пишутся фактом
``timer_overhead_pct``. Всё, что не измерилось, пишется ``unverified``.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator, Optional

from ._common import config_subject
from .fact import write_fact

#: Фазы шага учебного цикла.
PHASES: tuple[str, ...] = ("data", "forward", "backward", "ce", "optimizer", "host_sync")


class PhaseTimer:
    """Пофазовый таймер шага (S-023).

    Использование::

        timer = PhaseTimer(every=4)
        timer.begin_step(step)
        with timer.phase("forward", block=lambda: jax.block_until_ready(out)):
            out = forward(...)
        ...
        timer.commit(step, overhead_pct=...)
    """

    def __init__(
        self,
        *,
        every: int = 1,
        out_dir: Optional[str | Path] = None,
        subject: Optional[dict[str, Any]] = None,
        sensor: str = "S-023",
    ) -> None:
        self.every = max(1, int(every))
        self.out_dir = out_dir
        self.subject = subject
        self.sensor = sensor
        self._counter = 0
        self._sample = False
        self._seconds: dict[str, float] = {}

    def should_sample(self, step: int) -> bool:
        """Режим выборки: сэмплируем каждый ``every``-й шаг (детерминированно по шагу)."""
        return (step % self.every) == 0

    def begin_step(self, step: int) -> None:
        self._sample = self.should_sample(step)
        self._seconds = {}

    @contextmanager
    def phase(self, name: str, block: Optional[Callable[[], Any]] = None) -> Iterator[None]:
        """Измеряет фазу ``name``; ``block`` вызывается перед остановкой часов."""
        if not self._sample:
            yield
            return
        started = time.perf_counter()
        try:
            yield
        finally:
            if block is not None:
                block()  # jax.block_until_ready — граница фазы, иначе меряется очередь
            self._seconds[name] = time.perf_counter() - started

    @staticmethod
    def overhead_pct(baseline_seconds: float, measured_seconds: float) -> Optional[float]:
        """Накладные расходы таймера: (measured-baseline)/baseline × 100."""
        if baseline_seconds <= 0:
            return None
        return (measured_seconds - baseline_seconds) / baseline_seconds * 100.0

    def commit(
        self,
        step: int,
        *,
        overhead_pct_value: Optional[float] = None,
        write: bool = True,
    ) -> Optional[dict[str, Any]]:
        """Пишет факты шага (только на сэмплируемом шаге). ``None`` — не сэмпл."""
        if not self._sample:
            return None
        subject = self.subject or config_subject()
        record: dict[str, Any] = {"step": step, "phase_seconds": dict(self._seconds)}
        if write:
            write_fact(
                self.sensor, "step_phase_seconds", record, unit="seconds", quality="measured",
                method="PhaseTimer: границы фаз через jax.block_until_ready, режим выборки every=%d" % self.every,
                subject=subject, out_dir=self.out_dir,
            )
            if overhead_pct_value is not None:
                write_fact(
                    self.sensor, "timer_overhead_pct", float(overhead_pct_value), unit="pct",
                    quality="measured", method="PhaseTimer.overhead_pct", subject=subject,
                    out_dir=self.out_dir,
                )
        return record


class DeviceMemory:
    """Память устройства через JAX ``device.memory_stats()`` (S-024)."""

    def __init__(
        self,
        *,
        out_dir: Optional[str | Path] = None,
        subject: Optional[dict[str, Any]] = None,
        sensor: str = "S-024",
    ) -> None:
        self.out_dir = out_dir
        self.subject = subject
        self.sensor = sensor

    def sample(self, device: Any = None, *, write: bool = True) -> dict[str, Any]:
        """Снимает ``device.memory_stats()``; недоступно → ``unverified``."""
        stats = None
        reason = ""
        try:  # тяжёлый импорт по требованию
            if device is None:
                import jax

                device = jax.local_devices()[0]
            stats = device.memory_stats()
        except Exception as exc:  # noqa: BLE001 — любая ошибка окружения = «неизвестно»
            reason = f"memory_stats недоступны: {type(exc).__name__}"
        if not isinstance(stats, dict) or not stats:
            reason = reason or "memory_stats вернули пусто (не GPU/unified memory)"
        peak = stats.get("peak_bytes_in_use") if isinstance(stats, dict) else None
        in_use = stats.get("bytes_in_use") if isinstance(stats, dict) else None
        subject = self.subject or config_subject()
        result: dict[str, Any] = {}
        if write:
            for fact, value in (("device_peak_bytes", peak), ("device_bytes_in_use", in_use)):
                result[fact] = write_fact(
                    self.sensor, fact, value, unit="bytes", quality="measured",
                    method="jax device.memory_stats()", subject=subject, out_dir=self.out_dir,
                    status="ok" if value is not None else "unverified",
                    note="" if value is not None else reason,
                )
        return result


__all__ = ["PHASES", "PhaseTimer", "DeviceMemory"]
