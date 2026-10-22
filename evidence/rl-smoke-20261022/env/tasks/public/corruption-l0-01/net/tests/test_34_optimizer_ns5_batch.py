"""ADR-054 (цикл-4, Am.11, веха-5) — батчинг NS5 оптимизатора по shape-группам.

После победы над KDA-барьером шаг оптимизатора держат ~470 «больших» XLA-графов
не-KDA части (~45 мс front-end каждый ≈ 21 с/шаг, Am.10).  Вклад оптимизатора:
NS5 разворачивался **по каждому** Muon-листу отдельно (``tree_map`` +
unrolled Newton–Schulz) — на геометрии l3-full это 533 цепочки NS5 ⇒ сотни
мелких фьюзий в ``step_fn``.  Дельта вехи-5: ``AXIOM_OPT_BATCH=grouped``
собирает Muon-листья одной (группы, формы) в стёк и исполняет NS5 **одним**
``vmap``-вызовом на группу (одно batched ``dot_general`` вместо N) —
533 цепочки → 23 (реестр — :func:`optimizer.ns5_shape_groups`).

Что пинуется здесь (CPU, без GPU):

* **(а) паритет leaf ↔ grouped** — три шага мини-обучения (loss → градиент →
  шаг) с одного сида дают одинаковые параметры/состояние/лосс в допуске
  паритет-тестов оптимизатора (rtol=1e-6, atol=1e-7); отдельно — паритет на
  лист против независимых формул по всем трём Muon-веткам (matrix, per-head,
  batched) с негативным контролем «не NS5 по всему стёку»;
* **(б) группировка сворачивает граф** — число ``dot_general`` в трассированном
  шаге падает в разы (на микро-конфиге ×5), а на геометрии l3-full реестр групп
  показывает 533 листа → 23 вызова (21 батчированный + 2 одиночных);
* **(в) режим-флаг** ``AXIOM_OPT_BATCH`` — дефолт ``leaf`` (безопасный откат),
  ``grouped`` выбирает новую дорогу, неизвестное значение — ошибка на сборке
  шага (fail-closed), явный аргумент ``opt_batch`` сильнее окружения;
* **контракт возвратов шага не изменён** — ``work_dtype`` (третий выход, каст
  того же графа), ``donate`` (паритет), повторные вызовы (один executable,
  ``lr`` динамичен), fail-closed на неклассифицированном 2-D имени;
* **инвариант ADR-048** под новой дорогой — ``embedding``/связанный LM head
  остаётся AdamW, NS5 его не касается ни в leaf, ни в grouped.

Не пинуется здесь (замер архитектора на GB10): сколько именно submissions/графов
в NVTX исчезает, ток/с и время шага.  Здесь — форма графа и паритет.
"""

from __future__ import annotations

import collections
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from net import model, optimizer

ROOT = Path(__file__).resolve().parents[2]
L3_CONFIG = ROOT / "net" / "config.json"

#: Флаг режима (та же переменная, что читает ``make_step``).
FLAG = optimizer.OPT_BATCH_ENV

#: Допуск паритет-тестов оптимизатора (``test_30_optimizer_step_jit`` —
#: rtol=1e-6/atol=1e-7; та же пара у ``test_donate_step`` и ``test_bf16_cast_in_step``).
RTOL, ATOL = 1e-6, 1e-7


# ---------------------------------------------------------------------------
# Хелперы
# ---------------------------------------------------------------------------


def _inputs(cfg, *, seed: int = 0):
    """Детерминированное дерево параметров + градиенты + состояние."""
    params = model.init_params(jr.PRNGKey(seed), cfg)
    grads = jax.tree_util.tree_map(
        lambda leaf: jr.normal(jr.PRNGKey(int(leaf.size) + seed + 1), leaf.shape),
        params,
    )
    return params, grads, optimizer.init_state(params)


def _leg(cfg, mode, *, steps: int = 3, lr: float = 1e-3, seed: int = 11):
    """Нога мини-обучения: ``steps`` × (loss → градиент → шаг) на одном сиде."""
    ids = jr.randint(jr.PRNGKey(2), (2, 16), 0, cfg.vocab_size)
    params = model.init_params(jr.PRNGKey(seed), cfg)
    state = optimizer.init_state(params)
    step = optimizer.make_step(cfg, opt_batch=mode)
    losses = []
    for _ in range(steps):
        loss, grads = jax.value_and_grad(
            lambda p: model.compute_loss(p, cfg, ids, chunk_size=8)
        )(params)
        params, state = step(params, grads, state, lr)
        losses.append(float(loss))
    return params, state, losses


def _max_abs_diff(a, b) -> float:
    la, lb = jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b)
    assert len(la) == len(lb), "разная структура деревьев"
    worst = 0.0
    for x, y in zip(la, lb):
        assert x.shape == y.shape, f"формы разошлись: {x.shape} vs {y.shape}"
        worst = max(worst, float(jnp.max(jnp.abs(x - y))))
    return worst


def _assert_trees_close(a, b, *, rtol: float = RTOL, atol: float = ATOL, msg: str = ""):
    for x, y in zip(jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b)):
        assert bool(jnp.allclose(x, y, rtol=rtol, atol=atol)), (
            f"{msg}: max|Δ|={float(jnp.max(jnp.abs(x - y))):.3e} "
            f"(rtol={rtol}, atol={atol})"
        )


def _eqn_counts(fn, *args) -> collections.Counter:
    """Примитивы трассированного (не компилированного) шага — форма графа."""
    closed = jax.make_jaxpr(fn)(*args)
    return collections.Counter(str(eqn.primitive) for eqn in closed.jaxpr.eqns)


def _graph_counts(cfg, *, opt_batch=None) -> collections.Counter:
    """Primitive-состав eager-шага (режим — аргументом или из окружения)."""
    params, grads, state = _inputs(cfg)
    step = optimizer.make_step(cfg, opt_batch=opt_batch, jit=False)
    return _eqn_counts(step, params, grads, state, 1e-3)


# ---------------------------------------------------------------------------
# (а) Паритет: 3 шага мини-обучения, leaf против grouped
# ---------------------------------------------------------------------------


def test_three_training_steps_grouped_equals_leaf(tiny_cfg):
    """Главный паритет вехи-5: обучение не различает режимы.

    Один сид, один поток батчей, три шага (loss → градиент → шаг).  Параметры
    обязаны совпасть в допуске паритет-тестов оптимизатора, состояние — тоже
    (момент Muon/AdamW — элементные цепочки, batched-стёк их не меняет), лоссы —
    тем более: расхождение параметров уровня 1e-7 не должно двигать loss.
    """
    p_leaf, s_leaf, l_leaf = _leg(tiny_cfg, "leaf")
    p_grouped, s_grouped, l_grouped = _leg(tiny_cfg, "grouped")

    assert jax.tree_util.tree_structure(p_leaf) == jax.tree_util.tree_structure(p_grouped)
    assert jax.tree_util.tree_structure(s_leaf) == jax.tree_util.tree_structure(s_grouped)

    worst_p = _max_abs_diff(p_leaf, p_grouped)
    worst_s = _max_abs_diff(s_leaf, s_grouped)
    _assert_trees_close(p_leaf, p_grouped, msg="параметры после 3 шагов")
    _assert_trees_close(s_leaf, s_grouped, msg="состояние после 3 шагов")
    assert np.allclose(l_leaf, l_grouped, rtol=RTOL, atol=ATOL), (
        f"лоссы разошлись: {l_leaf} vs {l_grouped}"
    )
    print(f"[parity] 3 шага: max|Δp|={worst_p:.3e}, max|Δstate|={worst_s:.3e}")


def test_single_step_parity_is_tight_on_tiny(tiny_cfg):
    """Один шаг с идентичного входа: расхождение grouped ↔ leaf — шум понижения.

    Батчинг меняет только порядок редукций/фьюжнов NS5 (не формулы): на
    микро-геометрии расхождение параметров ≤ 1e-8 (измерено 0 … 3.7e-9 —
    показывается на ViT-``fc1`` при части сидов), состояние (момент Muon и
    AdamW-пары — элементные цепочки, batched-стёк их не переупорядочивает) —
    побитово то же.  Это и есть «арифметика листа не изменилась» числом:
    масштаб 1e-9, а не сдвиг обновления; допуск паритет-тестов (1e-6/1e-7)
    взят в ``test_three_training_steps...``.

    Порог 1e-8 намеренно жёсткий: рост расхождения до уровня обновления
    (в ~100 раз) обязан валить тест, а не проходить «в допуске».
    """
    params, grads, state = _inputs(tiny_cfg, seed=3)

    p_leaf, s_leaf = optimizer.make_step(tiny_cfg, opt_batch="leaf")(params, grads, state, 1e-3)
    p_grouped, s_grouped = optimizer.make_step(tiny_cfg, opt_batch="grouped")(params, grads, state, 1e-3)

    worst = _max_abs_diff(p_leaf, p_grouped)
    _assert_trees_close(p_leaf, p_grouped, msg="один шаг, leaf vs grouped")
    assert worst <= 1e-8, (
        f"расхождение одного шага {worst:.3e} — это не шум батчинга (порог 1e-8)"
    )
    assert all(
        np.array_equal(np.asarray(a), np.asarray(b))
        for a, b in zip(jax.tree_util.tree_leaves(s_leaf), jax.tree_util.tree_leaves(s_grouped))
    ), "состояние (момент/AdamW-пары) после одного шага разошлось — элементная цепочка задета"
    print(f"[parity] 1 шаг (tiny): max|Δp|={worst:.3e}, состояние — побитово")
    # Путь отката обязан остаться тем же и по режиму: leaf-нога не зависит от grouped.
    assert optimizer.opt_batch_mode("leaf") == optimizer.OPT_BATCH_LEAF


def _mixing_tree(cfg):
    """Tree с листьями всех трёх Muon-веток и обеих AdamW-групп (одинаковые формы рядом)."""
    def rnd(shape, seed):
        return jr.normal(jr.PRNGKey(seed), shape)

    return {
        "W_o": rnd((8, 4), 1),        # muon_matrix
        "W_g": rnd((8, 4), 2),        # muon_matrix — та же форма, идёт в стёк
        "W_q": rnd((8, 4), 3),        # muon_per_head
        "W_v": rnd((8, 4), 4),        # muon_per_head — та же форма, идёт в стёк
        "expert_g": rnd((2, 4, 4), 5),  # muon_batched
        "expert_u": rnd((2, 4, 4), 6),  # muon_batched — та же форма, идёт в стёк
        "embedding": rnd((8, 4), 7),  # adamw_embed
        "norm": rnd((4,), 8),         # adamw_vector
    }


def test_grouped_matches_the_per_leaf_reference_formulas(tiny_cfg):
    """Каждый лист стека — это его собственный NS5, а не NS5 по всему стёку.

    Формулы-эталоны независимы от ``_walk_grouped``: момент, NS5 по каждой
    матрице (per-head — по головам), wd, clip.  Негативный контроль — NS5,
    посчитанный по склеенной матрице (та же форма стека, но одна матрица):
    он обязан отличаться — иначе «батч» подменил бы арифметику листа.
    """
    tree = _mixing_tree(tiny_cfg)
    grads = jax.tree_util.tree_map(jnp.ones_like, tree)
    state = optimizer.init_state(tree)
    lr, wd = 1e-3, tiny_cfg.weight_decay
    moment = 0.95
    clip = lambda p: jnp.clip(p, -1.0, 1.0)
    upd = optimizer.make_step(tiny_cfg, opt_batch="grouped")(tree, grads, state, lr)[0]

    def muon_ref(leaf, head=False):
        m = moment * 0.0 + (1.0 - moment) * grads[leaf]
        u = (
            optimizer.per_head_newtonschulz5(m, tiny_cfg.num_heads)
            if head
            else optimizer.newtonschulz5(m)
        )
        return clip(tree[leaf] * (1.0 - lr * wd) - lr * u)

    def batched_ref(leaf):
        m = (1.0 - moment) * grads[leaf]
        u = jax.vmap(optimizer.newtonschulz5)(m.reshape(-1, *m.shape[-2:])).reshape(m.shape)
        return clip(tree[leaf] * (1.0 - lr * wd) - lr * u)

    for leaf in ("W_o", "W_g"):
        assert bool(jnp.allclose(upd[leaf], muon_ref(leaf), rtol=RTOL, atol=ATOL))
    for leaf in ("W_q", "W_v"):
        assert bool(jnp.allclose(upd[leaf], muon_ref(leaf, head=True), rtol=RTOL, atol=ATOL))
    for leaf in ("expert_g", "expert_u"):
        assert bool(jnp.allclose(upd[leaf], batched_ref(leaf), rtol=RTOL, atol=ATOL))
    # AdamW-ветки новой дорогой не тронуты (в т.ч. инвариант ADR-048 на embedding).
    for leaf in ("embedding", "norm"):
        m = (1.0 - 0.9) * grads[leaf]
        v = (1.0 - 0.95) * (grads[leaf] ** 2)
        ref = tree[leaf] * (1.0 - lr * wd) - lr * (m / 0.1) / (jnp.sqrt(v / 0.05) + 1e-8)
        if leaf == "embedding":
            ref = clip(ref)
        assert bool(jnp.allclose(upd[leaf], ref, rtol=RTOL, atol=ATOL)), (
            f"{leaf}: AdamW-ветка под grouped разошлась"
        )

    # Негативный контроль: один NS5 по склейке стека — это НЕ то, что делает grouped.
    glued = jnp.concatenate(
        [
            (1.0 - moment) * grads["W_o"],
            (1.0 - moment) * grads["W_g"],
        ],
        axis=0,
    )
    glued_upd = optimizer.newtonschulz5(glued)
    assert not bool(jnp.allclose(upd["W_o"], glued_upd[:8], rtol=1e-3, atol=1e-5)), (
        "grouped посчитал NS5 по всему стёку — арифметика листа подменена"
    )
    # и per-head ветка — не цельный NS5 по матрице
    assert not bool(
        jnp.allclose(
            upd["W_q"],
            clip(tree["W_q"] * (1.0 - lr * wd) - lr * optimizer.newtonschulz5((1.0 - moment) * grads["W_q"])),
            rtol=1e-3,
            atol=1e-5,
        )
    )


def test_grouped_keeps_embedding_out_of_ns5(tiny_cfg):
    """ADR-048 под новой дорогой: embedding/LM head — AdamW, не NS5.

    Реестр групп (инструмент режима) не содержит ``embedding`` среди Muon-цепочек,
    и обновление листа совпадает с AdamW-формулой, а не с Muon.
    """
    tree = _mixing_tree(tiny_cfg)
    report = optimizer.ns5_shape_groups(tree)
    assert all("embedding" not in e["examples"] for e in report["groups"])
    assert all(e["group"] in optimizer._MUON_GROUPS for e in report["groups"])

    grads = jax.tree_util.tree_map(jnp.ones_like, tree)
    state = optimizer.init_state(tree)
    lr, wd = 1e-3, tiny_cfg.weight_decay
    upd = optimizer.make_step(tiny_cfg, opt_batch="grouped")(tree, grads, state, lr)[0]

    m, v = (1 - 0.9) * grads["embedding"], (1 - 0.95) * (grads["embedding"] ** 2)
    adamw = jnp.clip(
        tree["embedding"] * (1.0 - lr * wd) - lr * (m / 0.1) / (jnp.sqrt(v / 0.05) + 1e-8),
        -1.0,
        1.0,
    )
    assert bool(jnp.allclose(upd["embedding"], adamw, rtol=RTOL, atol=ATOL))
    assert not bool(
        jnp.allclose(
            upd["embedding"],
            tree["embedding"] * (1.0 - lr * wd) - lr * optimizer.newtonschulz5(0.05 * grads["embedding"]),
            rtol=1e-3,
            atol=1e-5,
        )
    ), "embedding под grouped ушёл в NS5 — инвариант ADR-048 нарушен"


# ---------------------------------------------------------------------------
# (б) Группировка сворачивает граф: dot_general и реестр групп
# ---------------------------------------------------------------------------


def test_grouped_has_far_fewer_dot_generals_in_the_step_graph(tiny_cfg):
    """Трассированный шаг: batched-NS5 вместо N развёрнутых цепочек.

    Счёт — по frontend-jaxpr (платформо-независимо): каждый NS5-вызов — 3
    ``dot_general`` на итерацию; grouped сворачивает цепочки одинаковых форм в
    один batched-dot на группу.  Порог намеренно свободный (×3 по dot и ×2 по
    всем примитивам): тест ловит возврат к разворачиванию по листьям, а не
    дрейф на единицы операций.
    """
    leaf = _graph_counts(tiny_cfg, opt_batch="leaf")
    grouped = _graph_counts(tiny_cfg, opt_batch="grouped")
    leaf_dots, grouped_dots = leaf["dot_general"], grouped["dot_general"]
    leaf_eqns, grouped_eqns = sum(leaf.values()), sum(grouped.values())
    print(
        f"[graph] tiny: leaf dots={leaf_dots} eqns={leaf_eqns} | "
        f"grouped dots={grouped_dots} eqns={grouped_eqns}"
    )
    assert leaf_dots > 0 and grouped_dots > 0
    assert grouped_dots * 3 <= leaf_dots, (
        f"grouped не свернул NS5-цепочки: {grouped_dots} dot против {leaf_dots} у leaf"
    )
    assert grouped_eqns * 2 <= leaf_eqns, (
        f"группировка не сократила граф: {grouped_eqns} примитивов против {leaf_eqns}"
    )


def test_l3_full_ns5_census_533_chains_to_23():
    """Геометрия l3-full: сколько NS5-цепочек остаётся у grouped (расчёт Am.11).

    Реестр (:func:`optimizer.ns5_shape_groups`) — носитель числа: 533 Muon-листа
    (320 ``muon_matrix`` + 75 ``per-head`` + 138 ``muon_batched``) сворачиваются в
    23 вызова (21 батчированный стёк + 2 листа-одиночки своей формы).  Смена
    геометрии модели обязана двигать эти числа вместе с тестом — та же цена,
    что у реестра ADR-048 (``test_29_optimizer_groups``).
    """
    from net.config import load_config

    shapes = jax.eval_shape(
        lambda key: model.init_params(key, load_config(L3_CONFIG)),
        jax.ShapeDtypeStruct((2,), jnp.uint32),
    )
    report = optimizer.ns5_shape_groups(shapes)

    assert report["chains_leaf"] == report["muon_leaves"] == 533
    assert report["chains_grouped"] == 23
    assert report["batched_chains"] == 21
    assert sum(e["leaves"] for e in report["groups"]) == 533
    by_group = collections.Counter()
    for entry in report["groups"]:
        by_group[entry["group"]] += entry["leaves"]
    assert by_group == {
        optimizer.GROUP_MUON_MATRIX: 320,
        optimizer.GROUP_MUON_PER_HEAD: 75,
        optimizer.GROUP_MUON_BATCHED: 138,
    }
    # per-head семейство l3-full — 63×(1536,1536) + 12×(512,1536) (12 слоёв KDA/MLA)
    assert {
        tuple(e["shape"]): e["leaves"]
        for e in report["groups"]
        if e["group"] == optimizer.GROUP_MUON_PER_HEAD
    } == {(1536, 1536): 63, (512, 1536): 12}
    # в muon_matrix — 17 различных форм (складываются в 17 вызовов, 2 из них одиночные)
    assert sum(1 for e in report["groups"] if e["group"] == optimizer.GROUP_MUON_MATRIX) == 17
    # legacy-режим классификации (ADR-048) сворачивается так же честно
    assert optimizer.ns5_shape_groups(shapes, legacy_muon_all_2d=True)["muon_leaves"] == 536


# ---------------------------------------------------------------------------
# (в) Режим-флаг: дефолт leaf, grouped — новая дорога, мусор — ошибка
# ---------------------------------------------------------------------------


def test_default_mode_is_leaf_and_unknown_value_fails_closed(monkeypatch, tiny_cfg):
    """``AXIOM_OPT_BATCH``: дефолт ``leaf``; неизвестное значение — явная ошибка.

    Fail-closed: опечатка не должна тихо выбрать режим.  Пустая переменная —
    «не задано» (ручка отката, как у ``AXIOM_KDA_*``), это тоже безопасный
    ``leaf``, а не ошибка.
    """
    monkeypatch.delenv(FLAG, raising=False)
    assert optimizer.opt_batch_mode() == optimizer.OPT_BATCH_LEAF

    monkeypatch.setenv(FLAG, "groupd")  # опечатка
    with pytest.raises(ValueError, match=FLAG):
        optimizer.opt_batch_mode()
    with pytest.raises(ValueError, match=FLAG):
        optimizer.make_step(tiny_cfg)
    with pytest.raises(ValueError):
        optimizer.opt_batch_mode("nonsense")

    monkeypatch.setenv(FLAG, "")
    assert optimizer.opt_batch_mode() == optimizer.OPT_BATCH_LEAF
    assert optimizer.opt_batch_mode(" Grouped ") == optimizer.OPT_BATCH_GROUPED


def test_env_selects_grouped_and_explicit_argument_wins(monkeypatch, tiny_cfg):
    """Окружение выбирает дорогу; явный ``opt_batch`` сильнее окружения.

    Наблюдаемое — форма графа (число ``dot_general``): дефолт совпадает с
    ``leaf``, ``AXIOM_OPT_BATCH=grouped`` — с ``grouped``, а ``opt_batch="leaf"``
    при ``AXIOM_OPT_BATCH=grouped`` возвращает leaf-граф (ручка отката работает
    и на запущенном стенде без правки кода).
    """
    monkeypatch.delenv(FLAG, raising=False)
    leaf_dots = _graph_counts(tiny_cfg, opt_batch="leaf")["dot_general"]
    grouped_dots = _graph_counts(tiny_cfg, opt_batch="grouped")["dot_general"]
    assert grouped_dots < leaf_dots
    assert _graph_counts(tiny_cfg, opt_batch=None)["dot_general"] == leaf_dots, (
        "дефолт (без флага) не leaf — безопасный откат сломан"
    )

    monkeypatch.setenv(FLAG, "grouped")
    assert _graph_counts(tiny_cfg, opt_batch=None)["dot_general"] == grouped_dots
    assert _graph_counts(tiny_cfg, opt_batch="leaf")["dot_general"] == leaf_dots, (
        "явный opt_batch='leaf' не перебил AXIOM_OPT_BATCH=grouped"
    )
    monkeypatch.setenv(FLAG, "leaf")
    assert _graph_counts(tiny_cfg, opt_batch="grouped")["dot_general"] == grouped_dots


# ---------------------------------------------------------------------------
# Контракт возвратов шага сохранён (work_dtype / donate / повторы / fail-closed)
# ---------------------------------------------------------------------------


def test_grouped_repeated_calls_work_dtype_and_dynamic_lr(tiny_cfg):
    """Повторные вызовы grouped-шага: один executable, lr динамичен, каст внутри.

    ``work_dtype`` — тот же третий выход, что у leaf-пути: bf16-каст возвращённого
    мастера, побитово (тот же ``astype`` в том же графе).  Повторный вызов с теми
    же входами даёт те же числа (группировка не состояние), смена ``lr`` не
    перекомпилирует.
    """
    params, grads, state = _inputs(tiny_cfg, seed=4)
    step = optimizer.make_step(tiny_cfg, opt_batch="grouped", work_dtype="bfloat16")

    new_p, new_s, work = step(params, grads, state, 1e-3)
    again_p, again_s, again_work = step(params, grads, state, 1e-3)
    assert _max_abs_diff(new_p, again_p) == 0.0, "повторный вызов дал другие числа"
    assert _max_abs_diff(new_s, again_s) == 0.0
    assert _max_abs_diff(work, again_work) == 0.0
    assert step._cache_size() == 1, "повторные вызовы перекомпилируют grouped-шаг"

    cast = jax.tree_util.tree_map(lambda leaf: leaf.astype(jnp.bfloat16), new_p)
    for a, b in zip(jax.tree_util.tree_leaves(work), jax.tree_util.tree_leaves(cast)):
        assert a.dtype == jnp.bfloat16
        assert np.array_equal(np.asarray(a), np.asarray(b)), (
            "третий выход grouped-шага — не каст его же мастер-дерева"
        )
    step(params, grads, state, 7e-3)  # другой lr
    assert step._cache_size() == 1, "смена lr перекомпилировала grouped-шаг"


def test_grouped_donate_parity(tiny_cfg):
    """``donate=True`` под grouped — контракт владения буферами, не математика."""
    params, grads, state = _inputs(tiny_cfg, seed=5)
    plain = optimizer.make_step(tiny_cfg, opt_batch="grouped")
    donated = optimizer.make_step(tiny_cfg, opt_batch="grouped", donate=True)

    p_plain, s_plain = plain(params, grads, state, 1e-3)
    p_don, s_don = donated(
        jax.tree_util.tree_map(lambda x: x + 0, params),
        jax.tree_util.tree_map(lambda x: x + 0, grads),
        jax.tree_util.tree_map(lambda x: x + 0, state),
        1e-3,
    )
    worst = _max_abs_diff(p_plain, p_don)
    _assert_trees_close(p_plain, p_don, msg="grouped: donate vs plain")
    _assert_trees_close(s_plain, s_don, msg="grouped: donate vs plain (state)")
    print(f"[donate] grouped: max|Δ|={worst:.3e}")


def test_grouped_unclassified_2d_name_fails_closed(tiny_cfg):
    """Fail-closed ADR-048 переживает grouped-трассировку и jit."""
    tree = {"embedding": jnp.zeros((8, 4)), "W_mystery": jnp.zeros((4, 4))}
    state = optimizer.init_state(tree, legacy_muon_all_2d=True)  # состояние — legacy
    with pytest.raises(optimizer.UnclassifiedMatrixError):
        optimizer.make_step(tiny_cfg, opt_batch="grouped")(tree, tree, state, 1e-3)
    with pytest.raises(optimizer.UnclassifiedMatrixError):
        optimizer.make_step(tiny_cfg, opt_batch="grouped", jit=False)(tree, tree, state, 1e-3)
    # и реестр групп не прячет пропуск: легаси-классификация — вопрос отдельный
    with pytest.raises(optimizer.UnclassifiedMatrixError):
        optimizer.init_state(tree)


def test_grouped_legacy_flag_parity(tiny_cfg):
    """``legacy_muon_all_2d`` + grouped: прежняя классификация, тот же паритет.

    В legacy-режиме Muon забирает и 2-D embeddings (ADR-048 undo) — они обязаны
    ехать в стёк так же честно; результат обязан совпасть с leaf-legacy.
    """
    tree = _mixing_tree(tiny_cfg)
    grads = jax.tree_util.tree_map(lambda x: jr.normal(jr.PRNGKey(int(x.size) + 9), x.shape), tree)
    state = optimizer.init_state(tree, legacy_muon_all_2d=True)

    p_leaf, s_leaf = optimizer.make_step(
        tiny_cfg, opt_batch="leaf", legacy_muon_all_2d=True
    )(tree, grads, state, 1e-3)
    p_grouped, s_grouped = optimizer.make_step(
        tiny_cfg, opt_batch="grouped", legacy_muon_all_2d=True
    )(tree, grads, state, 1e-3)
    _assert_trees_close(p_leaf, p_grouped, msg="legacy: leaf vs grouped")
    _assert_trees_close(s_leaf, s_grouped, msg="legacy: leaf vs grouped (state)")
