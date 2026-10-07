"""Датчики поведенческого слоя Spine (ADR-037).

Датчик — код, который срабатывает во время работы системы (или по её
артефактам), измеряет и пишет **запись факта** (:mod:`tools.sensors.fact`).
Вердикт датчик не выносит: это делает предикат утверждения
(``tools/check_claims.py``). Датчик, который не может измерить, пишет честную
запись ``unverified`` с причиной, а не заглушку (C-007).

Пакет собирает три вещи:

* :mod:`tools.sensors.subject` — пин предмета измерения (git sha, хеши файлов,
  хост, устройство);
* :mod:`tools.sensors.fact` — контракт записи факта (JSONL, цепочка
  ``prev_sha256``);
* :mod:`tools.sensors.registry` + :mod:`tools.sensors.probe` — реестр
  ``model/sensors.yaml`` и проверка его исполнения.

Конкретные датчики (S-001…S-030) — соседние модули пакета; их запуск
``python3 -m tools.sensors.<name>``.
"""

from __future__ import annotations

__all__ = ["fact", "registry", "subject"]
