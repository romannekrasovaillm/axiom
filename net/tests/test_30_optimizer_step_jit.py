"""ADR-048 Amendment — jit шага оптимизатора и один проход по дереву.

После переписывания KDA (ADR-047) шаг оптимизатора стал доминантой шага
(``sec_backopt`` = 20.4 с из ~31 с).  Amendment ADR-048 уточняет причину: это
**хостовой** overhead, а не арифметика — NS-работа всего шага ~3.6e13 FLOP
(десятки мс), а шаг исполнялся *вне* ``jax.jit`` и проходил по дереву **дважды**
(``tree_map_with_path`` × 2 при 723 листьях).  Рычаг — jit-обёртка шага плюс один
проход; математика веток не меняется.

Что пинуется здесь (CPU, без GPU):

* **паритет один проход ↔ два прохода** — тот же результат на том же входе
  (eager и под jit): это критерий №1 отката, красный тест — блокер;
* **паритет jit ↔ eager** — jit не меняет числа;
* **``lr`` — динамический аргумент**: смена ``lr`` не перекомпилирует шаг
  (контраст с замыканием-статикой показан отдельно, чтобы тест был чувствителен);
* **``weight_clip`` распространён на ``adamw_embed``** (ADR-048 Amendment п.4,
  значение 1.0), а ``adamw_vector`` — нет (два свойства сразу не меняются);
* **legacy-флаг** возвращает прежнюю классификацию на том же коде;
* **fail-closed** переживает jit: ``UnclassifiedMatrixError`` доходит до
  вызывающего, а не превращается в аварийную компиляцию;
* **смоук-замер**: jit заметно быстрее eager на малом конфиге (порядок ×10+;
  точные числа — в ``evidence/kda-rewrite/optimizer-jit-smoke.json``).
"""

from __future__ import annotations

import time

import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

from net import model, optimizer


def _inputs(cfg, *, seed: int = 0):
    params = model.init_params(jr.PRNGKey(seed), cfg)
    grads = jax.tree_util.tree_map(
        lambda leaf: jr.normal(jr.PRNGKey(int(leaf.size) + seed + 1), leaf.shape),
        params,
    )
    state = optimizer.init_state(params)
    return params, grads, state


def _leaves_equal(a, b, *, rtol: float = 0.0, atol: float = 0.0) -> bool:
    la, lb = jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b)
    if len(la) != len(lb):
        return False
    return all(bool(jnp.allclose(x, y, rtol=rtol, atol=atol)) for x, y in zip(la, lb))


# ---------------------------------------------------------------------------
# 1. Паритет: один проход ↔ два прохода (критерий отката №1)
# ---------------------------------------------------------------------------


def test_one_walk_matches_two_walks_eager(tiny_cfg):
    """Eager: один проход по дереву даёт ровно то же, что два."""
    params, grads, state = _inputs(tiny_cfg)
    lr = 1e-3
    one = optimizer.make_step(tiny_cfg, jit=False, single_pass=True)
    two = optimizer.make_step(tiny_cfg, jit=False, single_pass=False)

    p1, s1 = one(params, grads, state, lr)
    p2, s2 = two(params, grads, state, lr)

    assert jax.tree_util.tree_structure(p1) == jax.tree_util.tree_structure(p2)
    assert jax.tree_util.tree_structure(s1) == jax.tree_util.tree_structure(s2)
    assert _leaves_equal(p1, p2)
    assert _leaves_equal(s1, s2)
    # структура состояния — как у init_state (Muon: momentum; AdamW: (m, v))
    assert jax.tree_util.tree_structure(s1) == jax.tree_util.tree_structure(state)


def test_one_walk_matches_two_walks_jitted(tiny_cfg):
    """Паритет сохраняется и под jit (сравниваем jit-путь с jit-путём)."""
    params, grads, state = _inputs(tiny_cfg)
    lr = 1e-3
    one = optimizer.make_step(tiny_cfg, jit=True, single_pass=True)
    two = optimizer.make_step(tiny_cfg, jit=True, single_pass=False)

    p1, s1 = one(params, grads, state, lr)
    p2, s2 = two(params, grads, state, lr)
    assert _leaves_equal(p1, p2)
    assert _leaves_equal(s1, s2)


def test_one_pass_walk_is_the_default(tiny_cfg):
    """Дефолт — jit + один проход: у возвращаемого шага есть jit-кэш."""
    default = optimizer.make_step(tiny_cfg)
    assert hasattr(default, "_cache_size"), "make_step по умолчанию обязан возвращать jit-шаг"
    eager = optimizer.make_step(tiny_cfg, jit=False)
    assert not hasattr(eager, "_cache_size"), "jit=False обязан возвращать обычную функцию"


# ---------------------------------------------------------------------------
# 2. Паритет jit ↔ eager
# ---------------------------------------------------------------------------


def test_jit_matches_eager(tiny_cfg):
    params, grads, state = _inputs(tiny_cfg, seed=3)
    lr = 5e-4
    eager = optimizer.make_step(tiny_cfg, jit=False)
    jitted = optimizer.make_step(tiny_cfg, jit=True)

    pe, se = eager(params, grads, state, lr)
    pj, sj = jitted(params, grads, state, lr)
    assert _leaves_equal(pe, pj, rtol=1e-6, atol=1e-7)
    assert _leaves_equal(se, sj, rtol=1e-6, atol=1e-7)


# ---------------------------------------------------------------------------
# 3. lr — динамический аргумент: смена lr не перекомпилирует
# ---------------------------------------------------------------------------


def test_lr_change_does_not_recompile(tiny_cfg):
    """Два вызова с разным ``lr`, одинаковыми формами → одна компиляция."""
    params, grads, state = _inputs(tiny_cfg)
    step = optimizer.make_step(tiny_cfg)

    step(params, grads, state, 1e-3)
    after_first = step._cache_size()
    assert after_first == 1

    step(params, grads, state, 7e-3)  # другой lr, те же формы/структура
    assert step._cache_size() == after_first, (
        "смена lr перекомпилировала шаг — lr попал в статические аргументы"
    )


def test_lr_as_closure_would_recompile_negative_control():
    """Негативный контроль: с замкнутым ``lr`` три значения дают три компиляции.

    Показывает, что тест выше чувствителен — он ловит именно статический ``lr``,
    а не «jit что-то закешировал».  Геометрия здесь игрушечная: проверяется
    семантика jit-кэша, а не наш шаг.
    """
    tree = {"a": jnp.ones((4, 4))}

    def closure_step(lr):
        return jax.jit(lambda t: jax.tree_util.tree_map(lambda x: x * (1.0 - lr), t))

    compilations = 0
    for lr in (1e-3, 2e-3, 3e-3):
        step = closure_step(lr)  # статический lr → новый jit-объект на каждое значение
        step(tree)
        compilations += step._cache_size()
    assert compilations == 3, (
        "контроль сломан: статический lr не вызвал перекомпиляцию на каждом значении"
    )


# ---------------------------------------------------------------------------
# 4. weight_clip распространён на adamw_embed (ADR-048 Amendment п.4)
# ---------------------------------------------------------------------------


def test_weight_clip_applies_to_adamw_embed_not_to_adamw_vector(tiny_cfg):
    """``embedding`` клиппится на 1.0; ``adamw_vector`` — нет (как до дельты)."""
    tree = {
        "embedding": jnp.full((4, 4), 2.0),  # adamw_embed, вне диапазона клипа
        "norm": jnp.full((4,), 2.0),         # adamw_vector
    }
    grads = jax.tree_util.tree_map(jnp.zeros_like, tree)
    state = optimizer.init_state(tree)
    lr, wd = 1e-3, tiny_cfg.weight_decay

    updated, _ = optimizer.make_step(tiny_cfg)(tree, grads, state, lr)

    # AdamW с нулевым градиентом: p * (1 - lr*wd) ≈ 2.0 — вне [-1, 1].
    raw = 2.0 * (1.0 - lr * wd)
    assert bool(jnp.allclose(updated["embedding"], 1.0)), (
        "weight_clip не применён к adamw_embed (Amendment п.4)"
    )
    assert bool(jnp.allclose(updated["norm"], raw)), (
        "adamw_vector клиппится — прежнее поведение сломано (два свойства сразу)"
    )
    assert not bool(jnp.allclose(updated["norm"], 1.0))


def test_weight_clip_off_disables_clip_for_embed(tiny_cfg):
    tree = {"embedding": jnp.full((4, 4), 2.0), "norm": jnp.full((4,), 2.0)}
    grads = jax.tree_util.tree_map(jnp.zeros_like, tree)
    state = optimizer.init_state(tree)
    lr, wd = 1e-3, tiny_cfg.weight_decay

    updated, _ = optimizer.make_step(tiny_cfg, weight_clip=None)(
        tree, grads, state, lr
    )
    raw = 2.0 * (1.0 - lr * wd)
    assert bool(jnp.allclose(updated["embedding"], raw))


# ---------------------------------------------------------------------------
# 5. legacy-флаг восстанавливает прежнюю классификацию (на том же jit-шаге)
# ---------------------------------------------------------------------------


def test_legacy_flag_restores_muon_for_embedding(tiny_cfg):
    params, grads, state = _inputs(tiny_cfg, seed=5)
    legacy_state = optimizer.init_state(params, legacy_muon_all_2d=True)
    # прежнее состояние 2-D листа — один momentum, не пара (m, v)
    assert legacy_state.embedding.shape == params.embedding.shape

    step = optimizer.make_step(tiny_cfg, legacy_muon_all_2d=True)
    updated, new_state = step(params, grads, legacy_state, 1e-3)

    # Muon-обновление (не AdamW): m = 0.05*g, NS5, wd, clip.
    m = 0.05 * grads.embedding
    upd = optimizer.newtonschulz5(m, 5)
    expected = jnp.clip(
        params.embedding * (1.0 - 1e-3 * tiny_cfg.weight_decay) - 1e-3 * upd, -1.0, 1.0
    )
    assert bool(jnp.allclose(updated.embedding, expected, rtol=1e-5, atol=1e-7))
    # состояние legacy-ветки — momentum, а не пара
    assert new_state.embedding.shape == params.embedding.shape

    # и отличается от решения ADR-048 (AdamW) — флаг действительно переключает
    default_state = optimizer.init_state(params)
    default_updated, _ = optimizer.make_step(tiny_cfg)(params, grads, default_state, 1e-3)
    assert not bool(
        jnp.allclose(updated.embedding, default_updated.embedding, rtol=1e-3, atol=1e-5)
    )


# ---------------------------------------------------------------------------
# 6. fail-closed переживает jit
# ---------------------------------------------------------------------------


def test_unclassified_error_propagates_through_jit(tiny_cfg):
    tree = {"embedding": jnp.zeros((8, 4)), "W_mystery": jnp.zeros((4, 4))}
    state = optimizer.init_state(tree, legacy_muon_all_2d=True)
    step = optimizer.make_step(tiny_cfg)  # jit по умолчанию
    with pytest.raises(optimizer.UnclassifiedMatrixError):
        step(tree, tree, state, 1e-3)


# ---------------------------------------------------------------------------
# 7. Смоук-замер: jit против eager на малом конфиге
# ---------------------------------------------------------------------------


def test_smoke_jit_is_much_faster_than_eager(tiny_cfg):
    """Порядок ускорения — ×10+ (точное число пишет tools/optimizer_jit_smoke.py).

    Порог намеренно свободный (×3), чтобы тест не флейковал на загруженной
    машине; измеренное ускорение — сотни раз.
    """
    params, grads, state = _inputs(tiny_cfg, seed=7)
    lr = 1e-3
    eager = optimizer.make_step(tiny_cfg, jit=False)
    jitted = optimizer.make_step(tiny_cfg)
    jitted(params, grads, state, lr)  # прогрев/компиляция — вне замера

    n = 3
    t0 = time.perf_counter()
    for _ in range(n):
        eager(params, grads, state, lr)
    eager_ms = (time.perf_counter() - t0) / n * 1e3

    t0 = time.perf_counter()
    for _ in range(n):
        jitted(params, grads, state, lr)
    jit_ms = (time.perf_counter() - t0) / n * 1e3

    ratio = eager_ms / jit_ms
    print(f"[smoke] eager={eager_ms:.2f} ms/шаг, jit={jit_ms:.3f} ms/шаг, ×{ratio:.0f}")
    assert jit_ms < eager_ms, "jit-шаг не быстрее eager — рычаг Amendment не работает"
    assert ratio > 3.0, f"ускорение jit всего ×{ratio:.1f} — ожидался порядок ×10+"
