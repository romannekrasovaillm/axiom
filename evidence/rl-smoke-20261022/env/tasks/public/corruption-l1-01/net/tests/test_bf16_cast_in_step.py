"""ADR-054 (цикл-4, Am.10 п.1) — bf16-каст параметров внутри jitted-шага.

Дельта: ``params = working_params(master)`` (``tree_map astype→bf16`` по ~330
листьям дерева параметров) исполнялся как ~330 **отдельных**
``jit(convert_element_type)`` модулей на шаг — крупнейший одиночный источник
лишних submissions после победы над KDA (Am.10: ~3300 сабмишенов/шаг, ~8 мс на
большой граф).  Вариант A: каст уезжает **третьим выходом** ``step_fn``
(``(master_new, state_new, params_bf16)``) — тот же jitted-граф, отдельного
submission нет; шаг-0 (до первого ``step_fn``) остаётся одним явным кастом вне
цикла.

Что пинуется здесь:

* **(а) паритет**: третий выход шага побитово равен старому вычислению
  ``working_params(master_new)`` (тот же ``astype``, те же формы/значения), а
  сам мастер-путь не изменился (fp32-возврат и математика те же);
* **(б) каст — часть графа шага**: jaxpr шага с ``work_dtype`` несёт bf16-каст
  на каждый лист, выходов три, дефолтный контракт (два выхода) цел; повторные
  шаги не перекомпилируют (один executable);
* **(в) проводка цикла**: bf16-прогон ``train()`` собирает шаг с
  ``work_dtype="bfloat16"`` и исполняется (params — выход шага, в т.ч. под
  donate), fp32-прогон не трогает контракт (``work_dtype=None``).

Не пинуется здесь: исчезновение ~330 submissions на GPU-профиле (nsys/NVTX,
приёмка цикла-4); здесь — контракт возврата, паритет и проводка.
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from conftest import tiny_config

from net import model, optimizer
from net import train_loop as tl

#: Геометрия смоука: одна последовательность, короткий контекст (CPU-время).
SEQ_LEN = 16
#: ``chunk_size`` — как в ``TrainConfig`` по умолчанию (на T=16 не режет).
CHUNK = 64


def _batches(cfg, steps: int, *, seed: int = 5):
    """Детерминированный поток ``(1, T) int32`` — одинаковые данные для ног."""
    rng = np.random.default_rng(seed)
    return [
        rng.integers(0, cfg.vocab_size, size=(1, SEQ_LEN), dtype=np.int32)
        for _ in range(steps)
    ]


def _free_budget() -> "tl.Budget":
    """Смета без границ: тест гоняет провод, а не стоп-правило AD-8."""
    return tl.Budget(run_ref="bf16-cast-in-step", path=Path("/dev/null"), present=True)


def _grads_like(master):
    """Ненулевые fp32-градиенты той же формы, что мастер (шаг без модели)."""
    return jax.tree_util.tree_map(lambda leaf: jnp.full_like(leaf, 1e-3), master)


def _old_working(master):
    """Старое вычисление цикла: ``tree_map astype→bf16`` по мастеру."""
    return jax.tree_util.tree_map(lambda leaf: leaf.astype(jnp.bfloat16), master)


# ---------------------------------------------------------------------------
# (а) Паритет: третий выход == старый каст мастера
# ---------------------------------------------------------------------------


def test_a_returned_work_matches_the_old_external_cast():
    """Каст в шаге не меняет ни значения, ни формы: паритет со старым путём.

    Обе ноги идут по одним и тем же входам: старая — ``step`` без ``work_dtype``
    плюс ``working_params(master)`` в питоне, новая — ``work_dtype="bfloat16"``,
    третий выход.  На одном вызове третий выход обязан быть **побитово** равен
    касту возвращённого мастера (тот же ``astype`` на том же массиве); между
    ногами — совпадение в допуске паритет-тестов (rtol=1e-6, atol=1e-7).
    """
    cfg = tiny_config()
    master = model.init_params(jr.PRNGKey(3), cfg)
    state = optimizer.init_state(master)
    grads = _grads_like(master)
    lr = 1e-2

    old_master, _ = optimizer.make_step(cfg)(master, grads, state, lr)
    old_work = _old_working(old_master)

    new_master, _, new_work = optimizer.make_step(
        cfg, work_dtype="bfloat16"
    )(master, grads, state, lr)

    # Тот же astype на том же массиве — побитово: каст не переехал, а лишь
    # перестал быть отдельным модулем на лист.
    assert jax.tree_util.tree_structure(new_work) == jax.tree_util.tree_structure(
        old_work
    ), "третий выход шага — не дерево параметров той же формы"
    for returned, ref in zip(
        jax.tree_util.tree_leaves(new_work), jax.tree_util.tree_leaves(_old_working(new_master))
    ):
        assert returned.dtype == ref.dtype == jnp.bfloat16
        assert np.array_equal(np.asarray(returned), np.asarray(ref)), (
            "третий выход шага разошёлся с кастом его же мастера — это другой astype"
        )

    # Мастер-путь не изменился: те же значения в допуске паритет-тестов.
    leaves_new = jax.tree_util.tree_leaves(new_master)
    leaves_old = jax.tree_util.tree_leaves(old_master)
    assert len(leaves_new) == len(leaves_old)
    for a, b in zip(leaves_new, leaves_old):
        assert a.dtype == b.dtype == jnp.float32
        assert bool(jnp.allclose(a, b, rtol=1e-6, atol=1e-7)), (
            "fp32-мастер новой ноги разошёлся со старой — каст задел математику"
        )
    for returned, ref in zip(
        jax.tree_util.tree_leaves(new_work), jax.tree_util.tree_leaves(old_work)
    ):
        assert bool(jnp.allclose(returned, ref, rtol=1e-6, atol=1e-7)), (
            "bf16-рабочая копия новой ноги разошлась со старой"
        )


# ---------------------------------------------------------------------------
# (б) Каст внутри графа шага: три выхода, один executable, jaxpr несёт касты
# ---------------------------------------------------------------------------


def test_b_cast_is_inside_the_step_graph_not_per_leaf_modules():
    """Каст — часть jitted-шага, а не ~330 отдельных модулей.

    ``jax.make_jaxpr`` под ``work_dtype`` обязан показать bf16-конверты в самом
    графе шага (по конверту на лист), а ``eval_shape`` — три выхода: прежние
    два плюс bf16-дерево.  Дефолтная сборка (без ``work_dtype``) остаётся
    двухвыходной — контракт существующих вызывающих цел.  Серия повторных
    шагов не перекомпилирует граф (``_cache_size() == 1``).
    """
    cfg = tiny_config()
    master = model.init_params(jr.PRNGKey(4), cfg)
    state = optimizer.init_state(master)
    grads = _grads_like(master)
    lr = 1e-2

    eager = optimizer.make_step(cfg, work_dtype="bfloat16", jit=False)
    closed = jax.make_jaxpr(eager)(master, grads, state, lr)
    n_leaves = len(jax.tree_util.tree_leaves(master))
    n_bf16 = sum(
        1
        for eqn in closed.jaxpr.eqns
        if str(eqn.primitive) == "convert_element_type"
        and eqn.params.get("new_dtype") == jnp.bfloat16
    )
    assert n_bf16 >= n_leaves, (
        f"в графе шага {n_bf16} bf16-конвертов на {n_leaves} листьев — "
        "каст не весь внутри шага"
    )

    shapes = jax.eval_shape(eager, master, grads, state, lr)
    assert len(shapes) == 3, "work_dtype обязан добавлять третий выход шага"
    work_dtypes = {str(leaf.dtype) for leaf in jax.tree_util.tree_leaves(shapes[2])}
    assert work_dtypes == {"bfloat16"}, (
        f"третий выход — не bf16-дерево: {sorted(work_dtypes)}"
    )

    default_shapes = jax.eval_shape(optimizer.make_step(cfg), master, grads, state, lr)
    assert len(default_shapes) == 2, "дефолтный контракт шага (два выхода) сломан"

    step = optimizer.make_step(cfg, work_dtype="bfloat16")
    step(master, grads, state, lr)
    step(master, grads, state, lr)
    assert step._cache_size() == 1, (
        "повторные шаги перекомпилируют граф — кэш исполнителя не один"
    )


# ---------------------------------------------------------------------------
# (в) Проводка цикла: bf16 -> work_dtype="bfloat16", fp32 -> None
# ---------------------------------------------------------------------------


def _spy_make_step(monkeypatch):
    """Обернуть ``optimizer.make_step`` наблюдателем kwargs (work_dtype/donate)."""
    seen: list[dict] = []
    real = optimizer.make_step

    def spy(*args, **kwargs):
        seen.append(dict(kwargs))
        return real(*args, **kwargs)

    # train() импортирует модуль (``from net import model, optimizer``) и читает
    # атрибут в момент вызова — патч модуля виден циклу.
    monkeypatch.setattr(optimizer, "make_step", spy)
    return seen


def test_c_train_loop_requests_bf16_copy_from_the_step(monkeypatch):
    """bf16-цикл просит рабочую копию у шага; fp32-цикл контракт не трогает.

    bf16: ``work_dtype="bfloat16"`` и исполнение без ошибок — params шага
    приходят третьим выходом (в т.ч. через donate-цепочку, буфер градиента —
    прежний).  fp32: ``work_dtype=None``, ``params is master`` — старый путь,
    отдельного каста в графе нет.
    """
    cfg = tiny_config()
    seen = _spy_make_step(monkeypatch)

    bf16_result = tl.train(
        cfg,
        iter(_batches(cfg, 3, seed=29)),
        train_config=tl.TrainConfig(
            steps=3,
            lr=1e-2,
            seed=7,
            schedule="cosine",
            param_dtype="bfloat16",
        ),
        budget=_free_budget(),
    )
    assert seen, "make_step не вызван — spy не увидел цикл"
    assert seen[0].get("work_dtype") == "bfloat16", (
        "bf16-цикл не просит у шага рабочую копию — каст остался вне jit "
        "(~330 отдельных модулей/шаг)"
    )
    assert bf16_result.steps_done == 3
    assert all(np.isfinite(loss) for loss in bf16_result.losses)

    fp32_result = tl.train(
        cfg,
        iter(_batches(cfg, 2, seed=31)),
        train_config=tl.TrainConfig(
            steps=2,
            lr=1e-2,
            seed=7,
            schedule="cosine",
            param_dtype="float32",
        ),
        budget=_free_budget(),
    )
    assert seen[-1].get("work_dtype") is None, (
        "fp32-цикл просит bf16-копию у шага — fp32-режим тронут"
    )
    assert fp32_result.steps_done == 2
    assert all(np.isfinite(loss) for loss in fp32_result.losses)
