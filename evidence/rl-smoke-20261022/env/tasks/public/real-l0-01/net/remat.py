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
  batch-измерений (обычно безопаснее по памяти);
* ``everything_saveable`` — сохранять всё промежуточное: внутри границы не
  пересчитывается ничего (как если бы ``jax.checkpoint`` не стоял).  Крайняя
  точка шкалы для матрицы ADR-049: на целевой геометрии ожидается OOM, но без
  этой строки матрица не отличала бы «политика дорога» от «политики нет».

Политики выгрузки в хост-память (``offload_dots_saveable``) в установленной
версии JAX **нет**: у ``jax.checkpoint_policies`` есть фабрика
``offload_dot_with_no_batch_dims(offload_src, offload_dst)`` с двумя аргументами
пространств памяти, а не готовый callable с этим именем.  Одноимённая строка
поэтому отвергается как неизвестная (fail-closed), а не подменяется похожей.

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

#: Имя объявленной политики → атрибут :mod:`jax.checkpoint_policies`.  Карта
#: явная (а не «имя равно атрибуту»), чтобы переименование политики в JAX было
#: видно здесь, а не падало ``AttributeError`` в случайном месте вызова.
#: Согласованность с :data:`net.config.REMAT_POLICIES` проверяется при импорте:
#: объявленная политика без карты — ошибка конфигурации, а не «смолчит до
#: первого прогона» (spine AD-9).
_POLICY_ATTRS = {
    "dots_saveable": "dots_saveable",
    "dots_with_no_batch_dims_saveable": "dots_with_no_batch_dims_saveable",
    "everything_saveable": "everything_saveable",
}

_UNMAPPED = set(REMAT_POLICIES) - {"none"} - set(_POLICY_ATTRS)
assert not _UNMAPPED, (
    "объявленные политики без карты на jax.checkpoint_policies: "
    f"{sorted(_UNMAPPED)}"
)


def remat_policy_fn(policy: str):
    """Callable политики JAX по имени; ``None`` для ``none`` (policy не задаётся).

    Неизвестное имя — :class:`ValueError` (fail-closed): значение обязано быть
    объявлено в :data:`net.config.REMAT_POLICIES`, иначе потребитель не должен
    «догадываться» о намерении архитектора.

    Объявленное, но **отсутствующее** в установленной версии JAX имя — тоже
    :class:`ValueError` (проверка ``hasattr``): политику нельзя подменить
    похожей.  Ровно этот случай зафиксирован в ADR-049 Amendment про
    ``offload_dots_saveable`` (в JAX 0.10.2 атрибута нет) — здесь он ловится
    механизмом, а не только пином-тестом конкретного имени.
    """
    if policy == "none":
        return None
    if policy not in REMAT_POLICIES:
        raise ValueError(
            f"неизвестная политика рематериализации: {policy!r}; "
            f"ожидается одна из {REMAT_POLICIES}"
        )
    attr = _POLICY_ATTRS[policy]
    if not hasattr(jax.checkpoint_policies, attr):
        raise ValueError(
            f"политика рематериализации {policy!r} объявлена, но отсутствует в "
            f"установленной версии JAX {jax.__version__}: "
            f"jax.checkpoint_policies.{attr} не найден; подмена похожей "
            f"политикой запрещена (fail-closed)"
        )
    return getattr(jax.checkpoint_policies, attr)


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
