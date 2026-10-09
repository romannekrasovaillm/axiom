"""Политика рематериализации: что сохранять внутри ``jax.checkpoint`` (ADR-049).

``jax.checkpoint(fn)`` без политики сохраняет входы/выходы обёрнутой функции и
**пересчитывает всё** промежуточное при backward — максимальная экономия
памяти и максимальная цена пересчёта.  JAX позволяет политикой сказать, какие
промежуточные результаты сохранять:

* ``none`` — текущее поведение: ``jax.checkpoint(fn)`` вызывается **без**
  аргумента ``policy``, поэтому граф побитово тот же, что до дельты (ADR-049,
  «Reversibility»: дефолт не двигается);
* ``dots_saveable`` — сохранять выходы matmul/conv, пересчитывать elementwise-хвост;
* ``dots_with_no_batch_dims_saveable`` — то же, но только для matmul без
  batch-измерений (обычно безопаснее по памяти).

Единственная точка, где имя политики превращается в callable JAX: и
``net/model.py`` (backbone-слой), и ``net/kda.py`` (тела scan) ходят сюда, так
что «согласованно» (ADR-049 п. 2) — это свойство кода, а не договорённость.

Значение вне списка — **ошибка**, а не молчаливый ``none``: молчаливая подмена
вернула бы прогон к полному пересчёту, и матрица «политика → время/память»
мерила бы не то, что объявлено (ADR-049, «Consequences»: риск ошибки
конфигурации закрывается fail-closed валидацией).

``none`` остаётся дефолтом и на уровне схемы (``net.config.ModelConfig``), и на
уровне раннера (``--remat-policy`` не задан → значение конфига).
"""

from __future__ import annotations

import jax

from .config import REMAT_POLICIES

#: Значение по умолчанию — текущее поведение (save-nothing-политика внутри
#: обёртки: сохраняются только входы/выходы, промежуточное пересчитывается).
DEFAULT_REMAT_POLICY = "none"


def remat_policy_fn(policy: str):
    """Callable политики JAX по имени; ``None`` для ``none`` (policy не задаётся).

    Неизвестное имя — :class:`ValueError` (fail-closed): значение обязано быть
    объявлено в :data:`net.config.REMAT_POLICIES`, иначе потребитель не должен
    «догадываться» о намерении архитектора.
    """
    if policy == "none":
        return None
    if policy == "dots_saveable":
        return jax.checkpoint_policies.dots_saveable
    if policy == "dots_with_no_batch_dims_saveable":
        return jax.checkpoint_policies.dots_with_no_batch_dims_saveable
    raise ValueError(
        f"неизвестная политика рематериализации: {policy!r}; "
        f"ожидается одна из {REMAT_POLICIES}"
    )


def validate_remat_policy(policy: str) -> None:
    """Проверить значение, ничего не строя (fail-closed на неизвестном).

    Нужна там, где remat-граница стоит под выключенным переключателем
    (``grad_ckpt_policy="none"``, ``kda_chunked_backward=False``): объявленное,
    но неизвестное значение обязано падать сразу, а не тогда, когда механика
    однажды включится.
    """
    remat_policy_fn(policy)


def remat_checkpoint(fn, policy: str = DEFAULT_REMAT_POLICY):
    """``jax.checkpoint`` с объявленной политикой рематериализации.

    ``none`` вызывает ``jax.checkpoint(fn)`` ровно так же, как это делал код до
    ADR-049 — аргумент ``policy`` не передаётся вовсе.  Это и есть «дефолт
    сохраняется побитово»: у обеих форм совпадает не только число remat-границ,
    но и параметры примитивов (``policy=None`` в графе).
    """
    resolved = remat_policy_fn(policy)
    if resolved is None:
        return jax.checkpoint(fn)
    return jax.checkpoint(fn, policy=resolved)
