"""bf16-мастер дерева параметров — гейты ``AXIOM_MASTER_DTYPE`` / ``AXIOM_MASTER_COMP``.

ADR-054 Am.13 (цикл-5 кампании MFU-55).  Барьер шага — память-стена LPDDR5x:
~55–60 ГБ трафика деревьев params/grads/state на шаг (мастер fp32 ⇄ рабочая
копия bf16, касты ×2 направления, DtoD-перетасовки от dtype-конфликта).  Режим
этого модуля снимает касты: **дерево параметров живёт в bf16 как единственное
представление**, fp32 остаётся только в моментах оптимизатора и в арифметике
применения обновления.

Модуль держит три вещи и ничего больше:

* чтение гейтов (:func:`master_dtype_mode`, :func:`master_comp_mode`) —
  fail-closed, как ``AXIOM_COMPUTE_DTYPE``/``AXIOM_OPT_BATCH``: опечатка падает
  на сборке, а не тихо выбирает путь;
* **применение обновления к bf16-мастеру** — три плеча
  (:func:`rn_apply`, :func:`kahan_apply`, :func:`sr_apply`).  Все три получают
  уже посчитанное в fp32 обновление ``update`` (дельту целевого значения) и
  возвращают хранимое bf16-значение (+ буфер/ключ, где он есть);
* стохастическое округление fp32→bf16 (:func:`stochastic_round_bf16`).

Три плеча — три судьбы мелкого обновления
-----------------------------------------

``lr·update < |w|·2^-8`` — обновление ниже половины шага сетки bf16 при
``|w|``.  Прямое ``bf16(w_f32 + update)`` теряет его целиком: это плато
чистого bf16-мастера (Am.13).  Плечи:

``none``
    round-to-nearest, ``bf16(w_f32 + update)`` — контрольный замер ущерба.
``kahan``
    компенсационный буфер ``comp`` (bf16), в состоянии оптимизатора:
    пара ``(w, comp)`` хранит не только ``w``, а и недобранный остаток
    обновления, который следующий шаг возвращает в арифметику.  Значение пары
    — ``w_f32 − comp``.
``sr``
    стохастическое округление: вверх с вероятностью дробной части, так что
    математическое ожидание совпадает с точной суммой, а сам остаток не
    хранится (RNG — ключ в состоянии, seeded).

Формула kahan и её прочтение (важно)
------------------------------------

В TASK.md формула записана как ``comp_new = bf16((t − master_f32) − y)``, где
``t = master_f32 + y`` — **до** каста в bf16.  Выражение ``(t − master_f32) − y``
есть в точности ошибка округления сложения в fp32 (Knuth TwoSum), относительный
порядок которой ``2^-24``, тогда как потеря bf16-хранилища — ``2^-9``: в этом
прочтении буфер компенсирует не то, что теряется, и плечо ``kahan``
побитово совпадает с ``none`` (измерено: 2000 шагов с обновлением ниже
полушага сетки — 0 применённых обновлений у обоих; прогон на CPU с real bf16).
Реализовано чтение, в котором ``t`` — **уже сохранённый** мастер
(``t = bf16(master_f32 + y)``), то есть ``comp_new = bf16((t_f32 − master_f32) − y)``
= знак перевёрнутого остатка хранилища (``−r`` при ``r = t − bf16(t)``).  Оно
сохраняет букву формулы (``y = update − comp``, те же множители, тот же порядок
операций) и делает пару ``(master − comp)`` точным трекером fp32-траектории:
``eff' = master'_f32 − comp' = master_f32 + y`` — ровно то, что предписано
бюджетом обновления, а мелкое обновление не теряется, а переносится в
следующий шаг.  Расхождение зафиксировано в результате прогона
(``conflicts_with_prior_decisions``) и в докстринге теста
``net/tests/test_35_bf16_master.py``: побитовое совпадение буквального
прочтения с ``none`` измеримо и является тестовым утверждением.

Граница честности: компенсация восстанавливает только то, что потерял каст
bf16 (остаток ``r``, точно возвращаемый в арифметику в пределах bf16-точности
самого ``comp``: остаточная ошибка пары ~ ``|w|·2^-18`` на шаг, случайная).
Ни ADR-013 (детерминизм), ни ADR-048 (классификация) здесь не трогаются;
``sr`` меняет поток случайности шага по построению, поэтому в гейтовый
профиль детерминизма он не входит (Am.13: только ``kahan``/``none``).
"""

from __future__ import annotations

import os
import warnings

import jax
import jax.numpy as jnp

# ---------------------------------------------------------------------------
# Гейты
# ---------------------------------------------------------------------------

#: Env-гейт представления мастера: ``fp32`` (дефолт, побитово прежнее
#: поведение) | ``bf16`` (мастер — единственное дерево, bf16).
MASTER_DTYPE_ENV = "AXIOM_MASTER_DTYPE"

#: Env-гейт компенсации потери мелких обновлений: ``kahan`` (дефолт при
#: bf16-мастере) | ``sr`` | ``none``.  При ``fp32``-мастере не осмыслен.
MASTER_COMP_ENV = "AXIOM_MASTER_COMP"

#: Значения ``AXIOM_MASTER_DTYPE`` в порядке отчёта.
FP32 = "fp32"
BF16 = "bf16"
MASTER_DTYPES = (FP32, BF16)

#: Значения ``AXIOM_MASTER_COMP`` в порядке отчёта.
KAHAN = "kahan"
SR = "sr"
NONE = "none"
COMPS = (KAHAN, SR, NONE)

#: Дефолт ``AXIOM_MASTER_DTYPE``: прежнее поведение (fp32-мастер).
DEFAULT_MASTER_DTYPE = FP32

#: Дефолт ``AXIOM_MASTER_COMP`` при bf16-мастере (Am.13).
DEFAULT_COMP = KAHAN


class MasterDtypeError(ValueError):
    """Неизвестное значение гейта мастера (fail-closed, без догадок)."""


def _env_mode(name: str, raw: str | None) -> str | None:
    """Прочитать переменную: ``None`` — не задана, иначе нормализованное значение."""
    value = raw if raw is not None else os.environ.get(name)
    if value is None:
        return None
    value = value.strip().lower()
    return value or None


def master_dtype_mode(value: str | None = None) -> str:
    """Режим мастера: ``fp32`` (дефолт) | ``bf16``.

    Читается на **сборке** шага/состояния (режим меняет структуру состояния и
    dtype дерева — в середине прогона он не меняется).  Неизвестное значение —
    :class:`MasterDtypeError`, а не тихий откат в fp32: опечатка, молча
    оставляющая fp32, сделала бы каждый замер bf16-ноги замером другой модели.
    Пустая переменная = «не задана» (дефолт).  Явный аргумент сильнее env.
    """
    raw = _env_mode(MASTER_DTYPE_ENV, value)
    mode = raw or DEFAULT_MASTER_DTYPE
    if mode not in MASTER_DTYPES:
        raise MasterDtypeError(
            f"{MASTER_DTYPE_ENV}={mode!r}: ожидается одно из {list(MASTER_DTYPES)}"
        )
    return mode


def master_comp_mode(value: str | None = None, *, master_dtype: str | None = None) -> str:
    """Режим компенсации: ``kahan`` (дефолт) | ``sr`` | ``none``.

    Осмыслен только при ``AXIOM_MASTER_DTYPE=bf16`` (Am.13): при fp32-мастере
    обновление применяется напрямую, терять нечего.  Значение, отличное от
    ``none``, при fp32-мастере **игнорируется с предупреждением** (возврат
    ``none``), а не отвергается: это диагностическая ручка, и её наличие в
    окружении при выключенном bf16-режиме не должно ронять прогон (но и не
    должно проходить молча — иначе «компенсация» в отчёте будет неправдой).
    Предупреждение называет источник (переменную окружения или аргумент) и не
    срабатывает на согласованном ``none``: явный ``none`` при fp32-мастере —
    это не «проигнорированная компенсация», а то же самое поведение.

    Неизвестное значение — :class:`MasterDtypeError` (fail-closed) **в обоих**
    направлениях: и при bf16, и при fp32 (опечатка в имени плеча не должна
    прятаться за игнором).
    """
    dtype = master_dtype_mode(master_dtype)
    raw = _env_mode(MASTER_COMP_ENV, value)
    if raw is not None and raw not in COMPS:
        raise MasterDtypeError(
            f"{MASTER_COMP_ENV}={raw!r}: ожидается одно из {list(COMPS)}"
        )
    if dtype != BF16:
        if raw is not None and raw != NONE:
            source = MASTER_COMP_ENV if value is None else "master_comp"
            warnings.warn(
                f"{source}={raw!r} при {MASTER_DTYPE_ENV}={dtype!r}: "
                "компенсация осмысленна только при bf16-мастере — значение "
                "игнорируется (применение update идёт напрямую, как прежде)",
                stacklevel=2,
            )
        return NONE
    return raw or DEFAULT_COMP


def uses_comp_buffer(comp: str) -> bool:
    """``True`` — плечо держит bf16-буфер компенсации в состоянии (kahan)."""
    if comp not in COMPS:
        raise MasterDtypeError(f"компенсация {comp!r}: ожидается одно из {list(COMPS)}")
    return comp == KAHAN


def uses_rng_key(comp: str) -> bool:
    """``True`` — плечо держит RNG-ключ в состоянии (sr)."""
    if comp not in COMPS:
        raise MasterDtypeError(f"компенсация {comp!r}: ожидается одно из {list(COMPS)}")
    return comp == SR


def extra_is_state(comp: str) -> bool:
    """Есть ли у листа состояния «дополнительный» элемент (буфер или ключ)."""
    return uses_comp_buffer(comp) or uses_rng_key(comp)


# ---------------------------------------------------------------------------
# Примитивы применения обновления (на один лист; jit-совместимы)
# ---------------------------------------------------------------------------


def rn_apply(master, update):
    """``none``: ``master_new = bf16(master_f32 + update)`` (round-to-nearest)."""
    return (master.astype(jnp.float32) + update.astype(jnp.float32)).astype(jnp.bfloat16)


def kahan_apply(master, update, comp):
    """``kahan``: компенсированное применение ``update`` к bf16-мастеру.

    ``y = update − comp``; ``t = bf16(master_f32 + y)`` (fp32-сложение, затем
    каст) — **хранимое** новое значение; ``comp_new = bf16((t_f32 − master_f32) − y)``
    — знак перевёрнутого остатка хранилища.  Пара ``(master, comp)`` трекает
    значение ``master_f32 − comp``: подстановкой ``t_f32 = master_f32 + y``
    (мастер bf16 — точное fp32-значение) получаем
    ``t_f32 − comp_new = master_f32 + y``, то есть трекер продвинулся ровно на
    ``y = update − comp``, а ``comp`` при этом несёт ``−r`` (``r = t_f32 − t``) —
    недобранный остаток обновления, который вернётся следующим шагом.

    Оба операнда ``comp`` (bf16) и ``master`` (bf16) в fp32 точны: ошибка
    арифметики здесь — только fp32-округления самих сложений, ``2^-24``
    относительно, — они и уезжают в ``comp`` следующим шагом (в bf16-точности
    остатка, ``2^-18`` относительно ``master``).
    """
    m32 = master.astype(jnp.float32)
    y = update.astype(jnp.float32) - comp.astype(jnp.float32)
    new_master = (m32 + y).astype(jnp.bfloat16)
    new_comp = ((new_master.astype(jnp.float32) - m32) - y).astype(jnp.bfloat16)
    return new_master, new_comp


def stochastic_round_bf16(x, key):
    """SR fp32→bf16: вверх с вероятностью дробной части шага сетки.

    Трюк «прибавить случайные 16 бит и отрезать их»: для fp32-значения с
    младшими 16 битами ``l`` перенос в старшие возникает ровно при
    ``r ≥ 2^16 − l``, то есть с вероятностью ``l/2^16`` — точной дробной частью
    (середина округляется вверх/вниз равновероятно), без смещения на точном
    нуле (``l = 0`` — переноса нет).  Знак float в IEEE — отдельный бит: для
    отрицательного значения трюк округляет модуль, то есть математическое
    ожидание результата равно исходному значению.

    Границы: NaN/±inf не обрабатываются отдельно (экспонента насыщена —
    значение остаётся насыщенным); денормали дают денормали bf16.  Мастер-веса
    модели — конечные величины O(1), и это записано, а не подразумевается.
    """
    x32 = x.astype(jnp.float32)
    bits = jax.lax.bitcast_convert_type(x32, jnp.uint32)
    rnd = jax.random.bits(key, shape=x32.shape, dtype=jnp.uint32)
    rnd16 = jax.lax.shift_right_logical(rnd, jnp.uint32(16))
    rounded = (bits + rnd16) & jnp.uint32(0xFFFF0000)
    return jax.lax.bitcast_convert_type(rounded, jnp.float32).astype(jnp.bfloat16)


def sr_apply(master, update, key):
    """``sr``: ``master_new = SR_bf16(master_f32 + update)``; ключ обновляется.

    Ключ — часть состояния оптимизатора (Am.13): поток случайности
    воспроизводим при фиксированном сиде и переживает resume (ключ лежит в
    чекпойнте).  ``jax.random.split`` на каждом шаге не даёт одному и тому же
    шуму повторяться на соседних шагах.
    """
    next_key, subkey = jax.random.split(key)
    total = master.astype(jnp.float32) + update.astype(jnp.float32)
    return stochastic_round_bf16(total, subkey), next_key


def apply_update(comp: str, master, update, extra=None):
    """Применить ``update`` (fp32) к bf16-мастеру плечом ``comp``.

    ``extra`` — буфер компенсации (``kahan``) или RNG-ключ (``sr``); у ``none``
    его нет.  Возврат — ``(master_new, extra_new)``; структура пары одна на все
    плечи, поэтому вызывающий (оптимизатор) не ветвится по режиму дважды.
    """
    if comp == KAHAN:
        return kahan_apply(master, update, extra)
    if comp == SR:
        return sr_apply(master, update, extra)
    if comp == NONE:
        return rn_apply(master, update), extra
    raise MasterDtypeError(f"компенсация {comp!r}: ожидается одно из {list(COMPS)}")


__all__ = [
    "BF16",
    "COMPS",
    "DEFAULT_COMP",
    "DEFAULT_MASTER_DTYPE",
    "FP32",
    "KAHAN",
    "MASTER_COMP_ENV",
    "MASTER_DTYPES",
    "MASTER_DTYPE_ENV",
    "MasterDtypeError",
    "NONE",
    "SR",
    "apply_update",
    "extra_is_state",
    "kahan_apply",
    "master_comp_mode",
    "master_dtype_mode",
    "rn_apply",
    "sr_apply",
    "stochastic_round_bf16",
    "uses_comp_buffer",
    "uses_rng_key",
]
