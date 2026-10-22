"""ADR-054 Am.23 (цикл-14, веха-11) — фьюзинг применения шага оптимизатора.

Разбор остатка ядер (стационар, Am.23): шаг хост-bound (Am.15), по количеству
доминирует ``wrapped_convert`` (dtype-конверсии fp32↔bf16 **на каждый лист**
оптимизатора: апкаст градиента/мастера, применение плеча компенсации, запись
мастера и буфера) и CUB radix-sort MoE-группировки.  Дельта вехи-11: применение
обновления фьюзится на **виртуальные shape-группы** (``AXIOM_OPT_FUSE=1``,
дефолт 0) — апкасты и запись на группу одним кастом/плечом вместо цепочки на
лист; NS5 остаётся **листовым** (dot_general в графе не меняется).

Выбор (а)/(б) — задокументированное измерение
---------------------------------------------

Milestone требует: либо (а) снять запрет ``bf16`` × ``grouped``, доказав
симметрию с leaf в пределах одного bf16-округления на новом пути, либо
(б) построить фьюзинг в leaf-режиме с единым кастом на виртуальную группу.
Измерено:

* путь (а) в своей сути — batched NS5 поверх bf16-хранилища: дельта обновления
  против листового NS5 ~1.1e-5 абс. ≈ 2e-5 отн. (Am.11; пин диагноза —
  ``test_35_bf16_master.py::test_h2_...``), после ``lr`` микро-сдвиг
  перешагивает середину округления и сохранённые листья расходятся **ровно на
  шаг сетки bf16**.  Критерий «не больше одного округления хранилища» для этого
  пути веха-6 объявила невыразимым (шаг сетки в 1..2 раза больше ``|y|·2^-8``;
  «одно округление» неотличимо от тихой расходимости) — доказательства
  симметрии (а) дать нельзя, только замер, который уже был признан
  недостаточным;
* путь (б) — этот файл: NS5 листовой (``dot_general`` в трассированном графе
  leaf и fused совпадает числом — пин :func:`test_d_fused_drops_convert_ops_...`),
  фьюзится только применение.  Арифметика тождественна: eager-путь побитов
  (и в динамике 3 шагов — :func:`test_eager_three_steps_kahan_is_bitwise`).
  Под jit расхождения есть и они измерены: на tiny — master/моменты побитово,
  хвосты ≤4.8e-7; на small (3 шага, худшие сиды) — редкие «щелчки хранилища»
  (доли процента элементов, ≤1.22e-4 абс. на мастерах, лоссы ≤4.8e-7) — класс
  lowered-расхождений XLA, пин :func:`test_jit_characterization_small_flip_cascade`.
  То есть критерий вехи-6 «не больше одного округления представления» здесь
  выразим и выполнен на уровне источника (eager — 0; jit — щелчки редкие и
  ограниченные), тогда как для (а) источник — систематическая дельта 2e-5 отн.
  на каждом элементе.

Выбор: **(б)** — фьюзинг в leaf-режиме; запрет ``grouped`` × ``bf16``
сохранён без изменений (фикс-раунд вехи-6).

Что пинуется здесь (CPU):

* **(а) гейты off — побитово прежние пути**: прогон против ``net/`` базовой
  ревизии (``git archive`` в подпроцесс, как в ``test_35_bf16_master.py``) —
  та же нога с bf16-мастером, те же батчи, fuse/counting не заданы: loss'ы и
  хеши деревьев (params + состояние) совпадают строка к строке;
* **(б) гейт-контракт** ``AXIOM_OPT_FUSE`` — дефолт 0, опечатка падает на
  сборке (fail-closed), явный аргумент сильнее окружения, при fp32-мастере
  значение игнорируется с предупреждением (фьюзить нечего), fp32-путь
  остаётся побитово прежним;
* **(в) паритет** fused ↔ leaf: eager (``jit=False``) — побитово на всех трёх
  плечах (kahan/none/sr), в том числе в динамике 3 шагов; под jit на tiny —
  master/моменты в допуске паритет-тестов оптимизатора (rtol=1e-6, atol=1e-7),
  хвосты компенсации ≤1e-5 (измерено ≤4.8e-7), лоссы ≤1e-5; под jit на small —
  редкие «щелчки хранилища» от lowering'а, ограниченные FLIP_ATOL (измерено
  ≤1.22e-4 абс., доли процента элементов, лоссы ≤4.8e-7);
* **(г) census**: на геометрии l3-full (абстрактное дерево) трассированный
  fused-шаг несёт в разы меньше ``convert_element_type`` при **неизменном**
  числе ``dot_general`` (NS5 не расщеплён/не переупорядочен);
* **(д) контракты шага** — два выхода в bf16-мастере (``work_dtype`` —
  по-прежнему ошибка сборки), ``donate`` — паритет, повторные вызовы — один
  executable, смена ``lr`` не перекомпилирует, fail-closed классификации и
  бан ``grouped`` × ``bf16`` живы (с подсказкой про fuse);
* **(е) маркировка** — строка метрик несёт ``opt_fuse`` (и ``moe_counting`` из
  вехи-11) в формате master_dtype/accum_jit: A/B-ноги различимы по журналу.
"""

from __future__ import annotations

import collections
import hashlib
import io
import json
import os
import subprocess
import sys
import tarfile
import warnings
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from conftest import small_config, tiny_config

from net import master_dtype as md
from net import model, optimizer
from net import train_loop as tl

ROOT = Path(__file__).resolve().parents[2]
NET_DIR = Path(__file__).resolve().parents[1]
CASE_DIR = NET_DIR.parent
L3_CONFIG = NET_DIR / "config.json"

#: Базовая ревизия гейт-off (родитель дельты вехи-11 = rollback-якорь).
BASELINE_COMMIT = "2e47faf"

#: Допуск паритет-тестов оптимизатора (test_30/test_34/test_donate_step).
RTOL, ATOL = 1e-6, 1e-7

#: Допуск хвостов компенсации (bf16-буферы динамического диапазона ~1e-4):
#: измеренное расхождение — 0..2 ulp bf16 (≤2.4e-7), 1e-5 = ~25 ulp; тест
#: ловит подмену арифметики плеча, а не последний бит (тот же смысл, что у
#: допуска lowered-контракций test_32).
COMP_ATOL = 1e-5

#: Допуск лоссов паритета (пары «одна арифметика, разные lowering-решения XLA»).
LOSS_ATOL = 1e-5

#: Допуск «каскада щелчка хранилища» на малой геометрии (small cfg, 3 шага,
#: сиды 11/21 — где щелчки измерены): измерено max|Δp| = 6.1e-5 / 1.22e-4,
#: max|Δmoment| = 6.1e-6 / 4.9e-5, max|Δcomp| = 6.1e-5 / 1.22e-4.  1e-3 — 8x
#: запас над худшим; порог ловит подмену формул (она даёт O(1) на этих
#: геометриях), а не редкий однократный щелчок округления хранилища.
FLIP_ATOL = 1e-3


def _inputs(cfg, *, seed: int = 0, bf16: bool = True):
    """Детерминированное дерево параметров + градиенты того же dtype."""
    params = model.init_params(jr.PRNGKey(seed), cfg)
    grads = jax.tree_util.tree_map(
        lambda leaf: jr.normal(jr.PRNGKey(int(leaf.size) + seed + 1), leaf.shape),
        params,
    )
    if bf16:
        params = jax.tree_util.tree_map(lambda x: x.astype(jnp.bfloat16), params)
        grads = jax.tree_util.tree_map(lambda x: x.astype(jnp.bfloat16), grads)
    return params, grads


def _state(cfg, params, *, comp: str = "kahan"):
    return optimizer.init_state(params, master_dtype="bf16", master_comp=comp)


def _leaves_with_path(tree):
    return jax.tree_util.tree_flatten_with_path(tree)[0]


def _max_abs_diff(a, b) -> float:
    la, lb = jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b)
    assert len(la) == len(lb), f"разные деревья: {len(la)} против {len(lb)} листьев"
    worst = 0.0
    for x, y in zip(la, lb):
        if x is None and y is None:
            continue
        assert (x is None) == (y is None), "структура None-слотов разошлась"
        if x is None:
            continue
        xa, ya = np.asarray(x), np.asarray(y)
        assert xa.shape == ya.shape, f"формы разошлись: {xa.shape} vs {ya.shape}"
        if xa.size:
            worst = max(worst, float(np.max(np.abs(xa.astype(np.float64) - ya.astype(np.float64)))))
    return worst


def _assert_trees_close(a, b, *, rtol: float = RTOL, atol: float = ATOL, msg: str = ""):
    for x, y in zip(jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b)):
        if x is None or y is None:
            continue
        assert bool(jnp.allclose(x, y, rtol=rtol, atol=atol)), (
            f"{msg}: max|Δ|={float(jnp.max(jnp.abs(np.asarray(x, np.float32) - np.asarray(y, np.float32)))):.3e} "
            f"(rtol={rtol}, atol={atol})"
        )


def _state_categories(state_l, state_f):
    """Расхождения состояния по категориям слотов (moments/comp/keys).

    Слоты bf16-режима: fp32 — моменты (m, v), bf16 — буфер компенсации, uint32 —
    ключ sr.  Возврат: ``{категория: (max|Δ|, #различных элементов)}``.
    """
    worst: dict[str, float] = {}
    ndiff: dict[str, int] = {}
    for (path, x), (_, y) in zip(_leaves_with_path(state_l), _leaves_with_path(state_f)):
        if x is None or y is None:
            continue
        dtype = np.asarray(x).dtype
        # Имена dtype, не np.bfloat16 (в numpy 2.x его нет — форма 0112-venv).
        category = {"float32": "moment", "bfloat16": "comp"}.get(dtype.name, dtype.name)
        xa, ya = np.asarray(x), np.asarray(y)
        if xa.size and not np.array_equal(xa, ya):
            worst[category] = max(
                worst.get(category, 0.0),
                float(np.max(np.abs(xa.astype(np.float64) - ya.astype(np.float64)))),
            )
            ndiff[category] = ndiff.get(category, 0) + int(np.count_nonzero(xa != ya))
    return worst, ndiff


def _eqn_counts(fn, *args) -> collections.Counter:
    closed = jax.make_jaxpr(fn)(*args)
    return collections.Counter(str(eqn.primitive) for eqn in closed.jaxpr.eqns)


def _leg(cfg, *, comp: str, steps: int, seed: int, lr: float = 1e-3, fuse: bool, jit: bool = True):
    """Нога мини-обучения: ``steps`` × (loss → градиент → шаг) на одном входе."""
    params, grads0 = _inputs(cfg, seed=seed)
    ids = jr.randint(jr.PRNGKey(2 + seed), (2, 16), 0, cfg.vocab_size)
    state = _state(cfg, params, comp=comp)
    step = optimizer.make_step(cfg, master_dtype="bf16", master_comp=comp, fuse=fuse, jit=jit)
    losses = []
    for _ in range(steps):
        loss, grads = jax.value_and_grad(
            lambda p: model.compute_loss(p, cfg, ids, chunk_size=8)
        )(params)
        losses.append(float(loss))
        params, state = step(params, grads, state, lr)
    return params, state, losses


def _budget():
    return tl.Budget(run_ref="opt-fuse", path=Path("/dev/null"), present=True)


def _hash_tree(tree) -> str:
    """Стабильный хеш дерева (dtype+shape+байты листьев) для вызовов вне процесса."""
    digest = hashlib.sha256()
    for leaf in jax.tree_util.tree_leaves(tree):
        if leaf is None:
            digest.update(b"none")
            continue
        arr = np.asarray(leaf)
        digest.update(str(arr.dtype).encode())
        digest.update(str(arr.shape).encode())
        digest.update(np.ascontiguousarray(arr).tobytes())
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# (б) Гейт AXIOM_OPT_FUSE: дефолт 0, fail-closed, явный аргумент сильнее env
# ---------------------------------------------------------------------------


def test_gate_default_off_fail_closed_and_explicit_wins(monkeypatch, cfg):
    """``AXIOM_OPT_FUSE``: дефолт ``0``; опечатка — ошибка сборки шага.

    Fail-closed: неизвестное значение не должно тихо оставить leaf-путь —
    замер ноги «фьюзинг» стал бы замером прежнего пути (форма
    ``AXIOM_ACCUM_JIT``/``AXIOM_OPT_BATCH``).  Пустая переменная = «не
    задано» = безопасный ``0``; явный аргумент сильнее окружения.
    """
    monkeypatch.delenv(optimizer.OPT_FUSE_ENV, raising=False)
    assert optimizer.opt_fuse_mode() is False
    assert optimizer.opt_fuse_mode(" 1 ") is True
    assert optimizer.opt_fuse_mode("0") is False
    assert optimizer.opt_fuse_mode(True) is True
    assert optimizer.opt_fuse_mode(False) is False
    monkeypatch.setenv(optimizer.OPT_FUSE_ENV, "")
    assert optimizer.opt_fuse_mode() is False

    monkeypatch.setenv(optimizer.OPT_FUSE_ENV, "on")  # опечатка
    with pytest.raises(ValueError, match=optimizer.OPT_FUSE_ENV):
        optimizer.opt_fuse_mode()
    with pytest.raises(ValueError, match=optimizer.OPT_FUSE_ENV):
        optimizer.make_step(cfg, master_dtype="bf16")
    with pytest.raises(ValueError):
        optimizer.opt_fuse_mode("nonsense")

    # явный аргумент сильнее окружения (ручка отката без правки env).
    monkeypatch.setenv(optimizer.OPT_FUSE_ENV, "0")
    assert optimizer.opt_fuse_mode(True) is True
    monkeypatch.setenv(optimizer.OPT_FUSE_ENV, "1")
    assert optimizer.opt_fuse_mode(False) is False
    optimizer.make_step(cfg, master_dtype="bf16", fuse="0")  # сборка проходит


def test_fuse_at_fp32_master_is_ignored_with_warning(cfg):
    """``AXIOM_OPT_FUSE=1`` при fp32-мастере — предупреждение и прежний путь.

    Фьюзить при fp32 нечего (кастов fp32↔bf16 в применении нет): значение
    игнорируется с предупреждением (форма ``master_comp_mode``), а результат
    шага побитово равен leaf-пути — «тихо другой режим» не появляется.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert optimizer.opt_fuse_mode("1", master_dtype="fp32") is False
    assert any("фьюзинг" in str(w.message) for w in caught), "нет предупреждения об игноре"

    params, grads = _inputs(cfg, seed=3, bf16=False)
    state = optimizer.init_state(params)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fused = optimizer.make_step(cfg, master_dtype="fp32", fuse=True)
        assert any("фьюзинг" in str(w.message) for w in caught), (
            "make_step не предупредил об игноре fuse при fp32-мастере"
        )
    leaf = optimizer.make_step(cfg, master_dtype="fp32")
    p_f, s_f = fused(params, grads, state, 1e-3)
    p_l, s_l = leaf(params, grads, state, 1e-3)
    assert _max_abs_diff(p_f, p_l) == 0.0, "fp32-путь с fuse=1 не побитово прежний"
    assert _max_abs_diff(s_f, s_l) == 0.0


# ---------------------------------------------------------------------------
# (в) Паритет: eager — побитово; jit — допуск (master/moments/comp/лоссы)
# ---------------------------------------------------------------------------


def test_eager_fused_is_bitwise_equal_to_leaf(cfg):
    """``jit=False``: fused ↔ leaf побитово на всех трёх плечах.

    Eager исполняет каждый примитив отдельно (нет re-fusion графа), поэтому
    здесь видна арифметика путей «как написана»: стёк элементов считается теми
    же операциями с теми же значениями, NS5 листовой, применение — vmap той же
    функции.  Расхождения под jit (ниже) — решения lowering'а XLA, а не
    подмена формул: этот тест — граница между ними.
    """
    params, grads = _inputs(cfg, seed=0)
    for comp in ("kahan", "none", "sr"):
        state = _state(cfg, params, comp=comp)
        leaf = optimizer.make_step(cfg, master_dtype="bf16", master_comp=comp, jit=False)
        fused = optimizer.make_step(
            cfg, master_dtype="bf16", master_comp=comp, fuse=True, jit=False
        )
        p_l, s_l = leaf(params, grads, state, 1e-3)
        p_f, s_f = fused(params, grads, state, 1e-3)
        dp, ds = _max_abs_diff(p_l, p_f), _max_abs_diff(s_l, s_f)
        print(f"[eager] comp={comp}: max|dp|={dp:.3e} max|ds|={ds:.3e}")
        assert dp == 0.0 and ds == 0.0, f"eager-путь не побитовый: {dp:.3e}/{ds:.3e}"


def test_eager_three_steps_kahan_is_bitwise(cfg):
    """``jit=False``, 3 шага: fused ↔ leaf побитово и в динамике (kahan).

    Один шаг мог бы пройти «случайно» совпав; три шага с пересчётом лосса и
    градиентов на каждом — проверка, что формулы тождественны и в каскаде
    (мастер → градиент → момент → обновление → хранилище).  Измерено на
    small-конфиге: max|Δp| = max|Δstate| = 0.0 — именно этот факт отделяет
    арифметику фьюзинга от lowering-различий jit-графа ниже.
    """
    p_l, s_l, ll = _leg(cfg, comp="kahan", steps=3, seed=11, fuse=False, jit=False)
    p_f, s_f, lf = _leg(cfg, comp="kahan", steps=3, seed=11, fuse=True, jit=False)
    dp, ds = _max_abs_diff(p_l, p_f), _max_abs_diff(s_l, s_f)
    print(f"[eager 3 шага] dloss={max(abs(a - b) for a, b in zip(ll, lf)):.3e} "
          f"max|dp|={dp:.3e} max|ds|={ds:.3e}")
    assert ll == lf and dp == 0.0 and ds == 0.0, (
        f"eager-динамика не побитова: dloss={max(abs(a - b) for a, b in zip(ll, lf)):.3e} "
        f"max|dp|={dp:.3e} max|ds|={ds:.3e}"
    )


@pytest.mark.parametrize("seed", [0, 11])
def test_jit_parity_tiny_master_moments_tight(seed):
    """Под jit: master/moments в допуске оптимизатора, хвосты comp — в своём.

    Tiny-геометрия — «чистый» контур: измерено (3 шага, сиды 0/11/21) master и
    моменты побитово, буферы компенсации ≤4.8e-7 (≤2 ulp их масштаба), лоссы
    побитово.  Здесь действует тот же порядок допусков, что у паритет-тестов
    оптимизатора (rtol=1e-6/atol=1e-7); расхождения — решения контракции/
    фьюжна XLA для объединённого графа (eager-путь побитов — см. выше), а не
    подмена формул: ``dot_general`` не переупорядочен (census ниже).
    """
    cfg = tiny_config()
    comp = "kahan"
    p_l, s_l, losses_l = _leg(cfg, comp=comp, steps=3, seed=seed, fuse=False)
    p_f, s_f, losses_f = _leg(cfg, comp=comp, steps=3, seed=seed, fuse=True)
    worst, ndiff = _state_categories(s_l, s_f)
    print(
        f"[jit tiny seed={seed}] dloss={max(abs(a - b) for a, b in zip(losses_l, losses_f)):.3e} "
        f"max|dp|={_max_abs_diff(p_l, p_f):.3e} state={worst} ndiff={ndiff}"
    )
    _assert_trees_close(p_l, p_f, msg=f"master после 3 шагов (seed={seed})")
    for (path, x), (_, y) in zip(_leaves_with_path(s_l), _leaves_with_path(s_f)):
        if x is None or y is None:
            continue
        arr_dtype = np.asarray(x).dtype
        if arr_dtype.name == "float32":  # моменты
            assert bool(jnp.allclose(x, y, rtol=RTOL, atol=ATOL)), (
                f"момент {jax.tree_util.keystr(path)} вне допуска"
            )
        elif arr_dtype.name == "bfloat16":  # буфер компенсации
            assert _max_abs_diff(x, y) <= COMP_ATOL, (
                f"comp {jax.tree_util.keystr(path)} вне допуска {COMP_ATOL}"
            )
    assert all(abs(a - b) <= LOSS_ATOL for a, b in zip(losses_l, losses_f)), (
        f"лоссы разошлись: {losses_l} vs {losses_f}"
    )


@pytest.mark.parametrize("seed", [11, 21])
def test_jit_characterization_small_flip_cascade(seed):
    """Малая геометрия: редкие «щелчки хранилища» от lowering'а — в допуске.

    На small-конфиге (больше слоёв и листьев, больше шансов попасть в середину
    округления) jit-пары расходятся **редкими единичными щелчками** bf16-
    хранилища: элемент, чей апдейт лёг у границы округления, получает соседнее
    представимое значение; дальше щелчок едет через градиент (вход следующего
    шага) и NS5 — малым каскадом (измерено: n_diff мастеров 18/152 элемента —
    доли процента; max|Δp| 6.1e-5/1.22e-4 = единицы ulp собственных элементов,
    у малых величин — до ~14 ulp; max|Δmoment| 6.1e-6/4.9e-5; лоссы
    0/4.8e-7).  Это класс lowered-расхождений (eager побитов — выше), не
    подмена арифметики: «каскад щелчка» ограничен FLIP_ATOL, число
    разъехавшихся мастер-элементов — долей от дерева, и печатается в отчёт.

    Контраст с запретом вехи-6: там источник — batched NS5 с систематической
    дельтой ~2e-5 отн. на каждом элементе/шаге; здесь источник — различия
    lowering'а уровня fp32-ulp, а щелчки — редкие (сотни из сотен тысяч
    элементов за 3 шага).
    """
    cfg = small_config()
    comp = "kahan"
    p_l, s_l, losses_l = _leg(cfg, comp=comp, steps=3, seed=seed, fuse=False)
    p_f, s_f, losses_f = _leg(cfg, comp=comp, steps=3, seed=seed, fuse=True)
    total_p = sum(int(np.size(leaf)) for leaf in jax.tree_util.tree_leaves(p_l))
    n_p = 0
    for x, y in zip(jax.tree_util.tree_leaves(p_l), jax.tree_util.tree_leaves(p_f)):
        if x is None:
            continue
        n_p += int(np.count_nonzero(np.asarray(x) != np.asarray(y)))
    worst, ndiff = _state_categories(s_l, s_f)
    dloss = max(abs(a - b) for a, b in zip(losses_l, losses_f))
    print(
        f"[jit small seed={seed}] dloss={dloss:.3e} max|dp|={_max_abs_diff(p_l, p_f):.3e} "
        f"n_diff={n_p}/{total_p} ({100.0 * n_p / total_p:.3f}%) state={worst} ndiff={ndiff}"
    )
    assert dloss <= LOSS_ATOL, f"лоссы вне допуска: {dloss:.3e}"
    assert n_p <= 0.01 * total_p, (
        f"щелчки мастеров вышли за долю 1%: {n_p}/{total_p}"
    )
    assert _max_abs_diff(p_l, p_f) <= FLIP_ATOL, "каскад мастеров вне FLIP_ATOL"
    for category, value in worst.items():
        assert value <= FLIP_ATOL, f"каскад {category} вне FLIP_ATOL: {value:.3e}"
    # Содержательная граница: каскад остаётся каскадом щелчков, а не сдвигом
    # траектории — число разъехавшихся хвостов тоже ограничено (не все листья).
    assert all(count <= 0.5 * total_p for count in ndiff.values()), (
        f"хвосты разъехались массово: {ndiff}"
    )
def test_sr_arm_keeps_per_leaf_key_stream_under_fuse(cfg):
    """``sr``: поток ключей per-leaf сохраняется, фьюзинг воспроизводим.

    Плечо sr держит RNG-ключ в состоянии: vmap-применение не должно склеивать
    потоки (иначе «seeded, воспроизводимо» стало бы заявлением без носителя).
    Проверка: ключи после шага у leaf и fused совпадают побитово (тот же
    per-leaf split), а повторный fused-прогон с тем же сидом — тот же результат.
    """
    params, grads = _inputs(cfg, seed=4)
    state = _state(cfg, params, comp="sr")
    leaf = optimizer.make_step(cfg, master_dtype="bf16", master_comp="sr", fuse=False)
    fused = optimizer.make_step(cfg, master_dtype="bf16", master_comp="sr", fuse=True)
    _, s_l = leaf(params, grads, state, 1e-3)
    _, s_f = fused(params, grads, state, 1e-3)
    for (path, x), (_, y) in zip(_leaves_with_path(s_l), _leaves_with_path(s_f)):
        if x is None or y is None:
            continue
        if np.asarray(x).dtype.kind == "u":  # ключ
            assert np.array_equal(np.asarray(x), np.asarray(y)), (
                f"поток ключей sr разъехался: {jax.tree_util.keystr(path)}"
            )
    p_a, s_a = fused(params, grads, state, 1e-3)
    p_b, s_b = fused(params, grads, state, 1e-3)
    assert _max_abs_diff(p_a, p_b) == 0.0 and _max_abs_diff(s_a, s_b) == 0.0, (
        "fused+sr не воспроизводим при том же сиде"
    )


def _assert_state_close(a, b, *, msg: str = "состояние"):
    """Состояние: моменты (fp32) — допуск паритета; хвосты (bf16) — COMP_ATOL."""
    worst, ndiff = _state_categories(a, b)
    print(f"[state] {msg}: worst={worst} ndiff={ndiff}")
    for (path, x), (_, y) in zip(_leaves_with_path(a), _leaves_with_path(b)):
        if x is None or y is None:
            continue
        dtype = np.asarray(x).dtype
        if dtype.name == "float32":
            assert bool(jnp.allclose(x, y, rtol=RTOL, atol=ATOL)), (
                f"{msg}: момент {jax.tree_util.keystr(path)} вне допуска "
                f"(max|Δ|={_max_abs_diff(x, y):.3e})"
            )
        elif dtype.name == "bfloat16":
            assert _max_abs_diff(x, y) <= COMP_ATOL, (
                f"{msg}: хвост {jax.tree_util.keystr(path)} вне допуска {COMP_ATOL}"
            )
        else:  # ключи RNG — поток обязан совпасть побитово
            assert np.array_equal(np.asarray(x), np.asarray(y)), (
                f"{msg}: ключ {jax.tree_util.keystr(path)} разъехался"
            )


# ---------------------------------------------------------------------------
# (г) Census: конверсии падают, dot_general (NS5) не меняется
# ---------------------------------------------------------------------------


def _l3_full_abstract(*, comp: str = "kahan"):
    """Абстрактное дерево l3-full: bf16-мастер + состояние режима (без аллокации)."""
    from net.config import load_config

    cfg = load_config(L3_CONFIG)
    shapes = jax.eval_shape(
        lambda key: model.init_params(key, cfg),
        jax.ShapeDtypeStruct((2,), jnp.uint32),
    )
    tree = jax.tree_util.tree_map(
        lambda x: jax.ShapeDtypeStruct(x.shape, jnp.bfloat16), shapes
    )
    state = jax.eval_shape(
        lambda p: optimizer.init_state(p, master_dtype="bf16", master_comp=comp), tree
    )
    return cfg, tree, state


def test_d_fused_drops_convert_ops_and_keeps_leaf_ns5_l3_full():
    """Геометрия l3-full: fused резко сжимает конверсии, NS5 остаётся листовым.

    Измерено на абстрактном дереве (трассировка, без аллокации): конверсии
    6507 → 279 (×23), всего примитивов 44817 → 33364, а ``dot_general`` —
    **не меняется** (7995 = 7995): NS5 листовой, переупорядочивания редукций
    (запрещённого Am.11 для bf16) фьюзинг не вносит.  Числа печатаются: смена
    геометрии/путей обязана двигать их вместе с тестом.
    """
    cfg, tree, state = _l3_full_abstract()
    c_leaf = _eqn_counts(
        optimizer.make_step(cfg, master_dtype="bf16", master_comp="kahan", jit=False),
        tree, tree, state, 1e-3,
    )
    c_fused = _eqn_counts(
        optimizer.make_step(
            cfg, master_dtype="bf16", master_comp="kahan", fuse=True, jit=False
        ),
        tree, tree, state, 1e-3,
    )
    print(
        f"[census l3-full] convert: {c_leaf['convert_element_type']} → "
        f"{c_fused['convert_element_type']}; eqns: {sum(c_leaf.values())} → "
        f"{sum(c_fused.values())}; dot: {c_leaf['dot_general']} → {c_fused['dot_general']}"
    )
    assert c_fused["convert_element_type"] * 10 <= c_leaf["convert_element_type"], (
        "фьюзинг не сжал конверсии: "
        f"{c_fused['convert_element_type']} против {c_leaf['convert_element_type']}"
    )
    assert c_fused["dot_general"] == c_leaf["dot_general"], (
        "число dot_general изменилось — NS5 перестал быть листовым"
    )
    assert sum(c_fused.values()) < sum(c_leaf.values())


# ---------------------------------------------------------------------------
# (д) Контракты шага под fuse: work_dtype, donate, кеш, бан, fail-closed
# ---------------------------------------------------------------------------


def test_d2_fuse_contracts_work_dtype_donate_and_cache(cfg):
    """fuse не переоткрывает контракты bf16-шага (Am.13) и держит один executable."""
    params, grads = _inputs(cfg, seed=5)
    state = _state(cfg, params)
    with pytest.raises(ValueError, match="work_dtype"):
        optimizer.make_step(cfg, master_dtype="bf16", fuse=True, work_dtype="bfloat16")

    step = optimizer.make_step(cfg, master_dtype="bf16", fuse=True)
    out_a = step(params, grads, state, 1e-3)
    assert len(out_a) == 2, "в bf16-мастере шаг двухвыходной (Am.13)"
    out_b = step(params, grads, state, 1e-3)
    assert step._cache_size() == 1, "повторный вызов перекомпилировал fused-шаг"
    assert _max_abs_diff(out_a[0], out_b[0]) == 0.0 and _max_abs_diff(out_a[1], out_b[1]) == 0.0
    step(params, grads, state, 7e-3)  # смена lr
    assert step._cache_size() == 1, "смена lr перекомпилировала fused-шаг"

    donated = optimizer.make_step(cfg, master_dtype="bf16", fuse=True, donate=True)
    fresh = lambda x: jax.tree_util.tree_map(lambda y: y + 0, x)
    p_d, s_d = donated(fresh(params), fresh(grads), fresh(state), 1e-3)
    _assert_trees_close(out_a[0], p_d, msg="fused: donate vs plain (params)")
    _assert_state_close(out_a[1], s_d, msg="fused: donate vs plain (state)")


def test_d3_grouped_ban_kept_and_points_to_fuse(cfg):
    """Запрет ``grouped`` × ``bf16`` сохранён (решение (б)); сообщение называет выход.

    Фьюзинг не трогает batched NS5 — причину запрета вехи-6 не снимает; вместо
    «снять запрет» выбран путь (б) (фьюзинг в leaf-режиме), поэтому шаг
    по-прежнему падает на сборке, а текст ошибки указывает на ``AXIOM_OPT_FUSE``.
    """
    with pytest.raises(optimizer.IncompatibleModeError, match="grouped") as exc:
        optimizer.make_step(cfg, master_dtype="bf16", opt_batch="grouped")
    assert optimizer.OPT_FUSE_ENV in str(exc.value), "ошибка не называет выход (fuse)"
    # Половины осей живы: leaf × bf16 (в т.ч. fuse) и grouped × fp32.
    optimizer.make_step(cfg, master_dtype="bf16", fuse=True)
    optimizer.make_step(cfg, master_dtype="fp32", opt_batch="grouped")


def test_d4_fused_unclassified_2d_name_fails_closed(cfg):
    """Fail-closed ADR-048 переживает fused-трассировку; legacy-режим жив."""
    tree = {"embedding": jnp.zeros((8, 4)), "W_mystery": jnp.zeros((4, 4))}
    tree = jax.tree_util.tree_map(lambda x: x.astype(jnp.bfloat16), tree)
    state = optimizer.init_state(tree, master_dtype="bf16", legacy_muon_all_2d=True)
    with pytest.raises(optimizer.UnclassifiedMatrixError):
        optimizer.make_step(cfg, master_dtype="bf16", fuse=True)(tree, tree, state, 1e-3)
    legacy_state = optimizer.init_state(tree, master_dtype="bf16", legacy_muon_all_2d=True)
    p, s = optimizer.make_step(
        cfg, master_dtype="bf16", fuse=True, legacy_muon_all_2d=True
    )(tree, tree, legacy_state, 1e-3)
    assert set(p) == set(tree)


# ---------------------------------------------------------------------------
# (е) Маркировка режимов в строке метрик
# ---------------------------------------------------------------------------


def _train_leg(cfg, steps: int, *, metrics_path, seed: int = 7, lr: float = 1e-2):
    rng = np.random.default_rng(seed)
    batches = [
        rng.integers(0, cfg.vocab_size, size=(1, 16), dtype=np.int32)
        for _ in range(steps)
    ]
    return tl.train(
        cfg,
        iter(batches),
        train_config=tl.TrainConfig(
            steps=steps, lr=lr, seed=seed, schedule="cosine", param_dtype="float32",
            metrics_path=metrics_path, log_every=0, kpi_every=1,
        ),
        budget=_budget(),
    )


def test_e_metrics_rows_carry_opt_fuse_and_moe_counting(cfg, tmp_path, monkeypatch):
    """Каждая строка метрик несёт ``opt_fuse`` и ``moe_counting`` (Am.23).

    Формат master_dtype/accum_jit: A/B-ноги цикла-14 («фьюзинг вкл/выкл»,
    «counting-sort vs argsort») различимы по журналу, а не по памяти
    исполнителя.  fp32 + fuse=1 — маркер 0: значение игнорируется (с
    предупреждением), фактический путь прежний.
    """
    from net import moe as moe_mod

    metrics = tmp_path / "on.jsonl"
    monkeypatch.setenv(md.MASTER_DTYPE_ENV, "bf16")
    monkeypatch.setenv(md.MASTER_COMP_ENV, "kahan")
    monkeypatch.setenv(optimizer.OPT_FUSE_ENV, "1")
    monkeypatch.setenv(moe_mod.COUNTING_MODE_ENV, "1")
    assert _train_leg(cfg, 2, metrics_path=metrics, seed=7).steps_done == 2
    rows = tl.MetricsWriter.read(metrics)
    assert len(rows) == 2
    for row in rows:
        assert row["opt_fuse"] == 1
        assert row["moe_counting"] == 1
        assert row["master_dtype"] == "bf16" and row["master_comp"] == "kahan"

    metrics_off = tmp_path / "off.jsonl"
    monkeypatch.delenv(optimizer.OPT_FUSE_ENV, raising=False)
    monkeypatch.delenv(moe_mod.COUNTING_MODE_ENV, raising=False)
    _train_leg(cfg, 1, metrics_path=metrics_off, seed=7)
    row = tl.MetricsWriter.read(metrics_off)[0]
    assert row["opt_fuse"] == 0 and row["moe_counting"] == 0

    metrics_fp32 = tmp_path / "fp32.jsonl"
    monkeypatch.delenv(md.MASTER_DTYPE_ENV, raising=False)
    monkeypatch.delenv(md.MASTER_COMP_ENV, raising=False)
    monkeypatch.setenv(optimizer.OPT_FUSE_ENV, "1")
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        _train_leg(cfg, 1, metrics_path=metrics_fp32, seed=7)
    row = tl.MetricsWriter.read(metrics_fp32)[0]
    assert row["opt_fuse"] == 0, "fp32-нога пометила fuse активным (значение не игнорируется)"
    assert row["master_dtype"] == "fp32"


# ---------------------------------------------------------------------------
# (а) Гейт-off побитово прежний — против базовой ревизии (git archive)
# ---------------------------------------------------------------------------

#: Проба гейт-off (fuse/counting не заданы): bf16-мастер + kahan, 2 шага,
#: печатает JSON с loss'ами и хешами дерева параметров и состояния.  Запускается
#: в подпроцессе с ``net`` = архив базовой ревизии; ``conftest`` импортируется
#: ДО ``jax`` — пины ADR-010 действуют на процесс только до создания клиента.
_GATE_OFF_PROBE = r'''
import hashlib, json, sys
sys.path.insert(0, "net/tests")
import conftest  # noqa: F401 — пины ADR-010 до import jax
import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr
from conftest import tiny_config
from net import model, optimizer

def hash_tree(tree):
    digest = hashlib.sha256()
    for leaf in jax.tree_util.tree_leaves(tree):
        if leaf is None:
            digest.update(b"none")
            continue
        arr = np.asarray(leaf)
        digest.update(str(arr.dtype).encode())
        digest.update(str(arr.shape).encode())
        digest.update(np.ascontiguousarray(arr).tobytes())
    return digest.hexdigest()

cfg = tiny_config()
params = jax.tree_util.tree_map(
    lambda x: x.astype(jnp.bfloat16), model.init_params(jr.PRNGKey(0), cfg))
grads = jax.tree_util.tree_map(
    lambda leaf: jr.normal(jr.PRNGKey(int(leaf.size) + 1), leaf.shape), params)
grads = jax.tree_util.tree_map(lambda x: x.astype(jnp.bfloat16), grads)
state = optimizer.init_state(params, master_dtype="bf16", master_comp="kahan")
ids = jr.randint(jr.PRNGKey(2), (2, 16), 0, cfg.vocab_size)
step = optimizer.make_step(cfg, master_dtype="bf16", master_comp="kahan")
losses = []
for _ in range(2):
    loss, grads = jax.value_and_grad(
        lambda p: model.compute_loss(p, cfg, ids, chunk_size=8))(params)
    losses.append(repr(float(loss)))
    params, state = step(params, grads, state, 1e-3)
print(json.dumps({
    "losses": losses,
    "params_hash": hash_tree(params),
    "state_hash": hash_tree(state),
}))
'''


@pytest.fixture(scope="module")
def baseline_gate_off() -> dict:
    """Прогон ``net/`` ревизии ``BASELINE_COMMIT`` в подпроцессе (см. пробу)."""
    root = NET_DIR / "tests" / ".baseline-fuse-gate-off"
    if root.exists():
        import shutil

        shutil.rmtree(root)
    root.mkdir()
    try:
        try:
            archive = subprocess.run(
                ["git", "-C", str(CASE_DIR), "archive", BASELINE_COMMIT, "net/"],
                check=True, capture_output=True,
            )
        except (subprocess.CalledProcessError, FileNotFoundError) as exc:  # pragma: no cover
            pytest.skip(f"baseline {BASELINE_COMMIT} недоступен: {exc}")
        with tarfile.open(fileobj=io.BytesIO(archive.stdout)) as bundle:
            bundle.extractall(root)
        env = dict(os.environ)
        env[md.MASTER_DTYPE_ENV] = "bf16"
        env[md.MASTER_COMP_ENV] = "kahan"
        env.pop(optimizer.OPT_FUSE_ENV, None)
        env.pop("AXIOM_MOE_COUNTING", None)
        probe = root / "probe.py"
        probe.write_text(_GATE_OFF_PROBE, encoding="utf-8")
        completed = subprocess.run(
            [sys.executable, str(probe)], cwd=root, env=env,
            capture_output=True, text=True,
        )
        if completed.returncode != 0:  # pragma: no cover
            raise AssertionError(f"прогон базовой ревизии упал: {completed.stderr[-2000:]}")
        yield json.loads(completed.stdout.strip().splitlines()[-1])
    finally:
        import shutil

        shutil.rmtree(root, ignore_errors=True)


def test_a_gate_off_is_bitwise_identical_to_baseline(baseline_gate_off, monkeypatch):
    """Гейты (fuse/counting) не заданы — loss'ы и деревья побитово прежние (Am.23).

    Та же нога (bf16-мастер + kahan, 2 шага, одни батчи), что у пробы базовой
    ревизии: дефолт ``AXIOM_OPT_FUSE=0`` не меняет ни одного листа графа — и
    потерь, ни параметров, ни состояния (моменты + буферы компенсации).
    """
    monkeypatch.delenv(optimizer.OPT_FUSE_ENV, raising=False)
    monkeypatch.delenv("AXIOM_MOE_COUNTING", raising=False)
    cfg = tiny_config()
    params, grads = _inputs(cfg, seed=0)
    state = _state(cfg, params, comp="kahan")
    ids = jr.randint(jr.PRNGKey(2), (2, 16), 0, cfg.vocab_size)
    step = optimizer.make_step(cfg, master_dtype="bf16", master_comp="kahan")
    losses = []
    for _ in range(2):
        loss, grads = jax.value_and_grad(
            lambda p: model.compute_loss(p, cfg, ids, chunk_size=8)
        )(params)
        losses.append(repr(float(loss)))
        params, state = step(params, grads, state, 1e-3)
    current = {
        "losses": losses,
        "params_hash": _hash_tree(params),
        "state_hash": _hash_tree(state),
    }
    assert current["losses"] == baseline_gate_off["losses"], (
        f"лоссы разошлись: {current['losses']} vs {baseline_gate_off['losses']}"
    )
    assert current["params_hash"] == baseline_gate_off["params_hash"], (
        "дерево параметров гейт-off не побитово прежнее"
    )
    assert current["state_hash"] == baseline_gate_off["state_hash"], (
        "состояние оптимизатора гейт-off не побитово прежнее"
    )
