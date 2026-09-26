"""Пайплайн датасета агентных эпизодов ``axiom-domain-ds-v1`` (ADR-020, дельта-1).

Компонент E датасета: сессии кодовых агентов владельца → верифицированные
агентные эпизоды. Порядок ступеней строгий (ADR-020, «Пайплайн»):

1. :mod:`~axiom_ds.scrub` — deny-list каталогов + скраб секретов (**до всего**);
2. :mod:`~axiom_ds.episodes` — эпизодизация по границам tool-циклов;
3. :mod:`~axiom_ds.verify` — механический класс исхода (контракт сессии либо
   парный отчёт турникета, :mod:`~axiom_ds.harness`);
4. :mod:`~axiom_ds.dedup` — точные хеши блоков + MinHash near-dup;
5. :mod:`~axiom_ds.build` — CLI и числовой отчёт.

Сырьё (сессии) и выход датасета — вне репозитория: ``~/gb10-shared/datasets``
(C-032/C-033), приватность — AD-6.
"""

from __future__ import annotations

__all__ = ["build", "dedup", "episodes", "harness", "scrub", "verify"]

PIPELINE_VERSION = "axiom-ds-episodes/1"
