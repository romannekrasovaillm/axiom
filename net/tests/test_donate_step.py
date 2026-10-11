"""ADR-054 стадия-1 — donate-аргументы jit: шаг без копий деревьев (CPU).

Дельтa устраняет ~28k DtoD-копий/шаг (31.5 ГиБ) профиля c64_B2: ``grad_fn``
и шаг оптимизатора вызывались без ``donate_argnums``, и каждое дерево
(params/grads/state) уезжало в копии буферов.  Donate — **контракт владения
буферами, не математика**: донированный вызов вправе переиспользовать буферы
аргументов под свои выходы, а вызывающий обязан не читать их после вызова
(иначе «Array has been deleted» / «donated buffer»).  Премисса дельты уже в
коде: ``params = working_params(master)`` пересоздаётся каждый шаг
(``net/train_loop.py:3025``).

Что пинуется здесь:

* **(а) паритет 3 шагов donate=True vs False** — loss, bf16-params и fp32-мастер
  совпадают (bf16 — режим кампании, там действует премисса «params — своё
  дерево, мёртв после вызова»): donate не меняет арифметику.  Порог — как у
  соседних паритет-тестов (``test_30_optimizer_step_jit``: rtol=1e-6,
  atol=1e-7); измеренная разница на CPU — 0.0 (одни и те же ядра);
* **(б) 5 последовательных шагов donate-пути ``train()``** (bf16, без
  профайлера) без RuntimeError «donated buffer reused»: буфер переживает ровно
  один шаг, пересоздание из возврата шага работает; spy на ``make_step``
  фиксирует, что цикл зовёт шаг с ``donate=True``;
* **(в) accum-ветка не берёт donated-градиент**: 2 микробатча на шаг держат
  одни ``params`` живыми на обоих микровызовах — донация в этой ветке упала бы
  на втором микровызове, поэтому исполнение 2-го микробатча и есть предикат
  «ветка использует недонированный ``grad_fn``»;
* **(г) ``phase_profile=True`` → ``donate=False``**: профайлер читает
  ``grads``/``master``/``state`` после ``step_fn`` (``profiler.measure``); spy
  фиксирует флаг, а нога ``sec_backopt`` обязана быть измерена — с донацией
  профайлер получил бы удалённый буфер, ошибку проглотил бы (диагностика ног
  не падает) и записал ``null``;
* **fp32-одиночная ветка остаётся без donate**: ``working_params`` отдаёт сам
  мастер (``params is master``), его буфер читает ``step_fn`` после градиента —
  донация там была бы use-after-free; прогон fp32 обязан исполняться.

Не пинуется здесь: исчезновение DtoD-копий — GPU-замер (nsys, A/B c64_B2,
приёмка стадии-1 по ADR-054); здесь — контракт владения буферами и паритет.
"""

from __future__ import annotations

import math
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
    return tl.Budget(run_ref="donate-step-test", path=Path("/dev/null"), present=True)


def _working(master):
    """bf16-рабочая копия мастера — ровно как ``working_params`` цикла."""
    return jax.tree_util.tree_map(lambda leaf: leaf.astype(jnp.bfloat16), master)


def _mini_loop(cfg, batches, *, donate: bool, seed: int = 7, lr: float = 1e-2):
    """Арифметика ветки ``len(group) == 1`` один-в-один, donate — разница ног.

    Копирует шаг цикла: ``loss, grads = grad_fn(params, batch)`` → bf16→fp32
    каст градиентов (как :2997 цикла) → ``step_fn(master, grads, state, lr)`` →
    ``params = working_params(master)``.  Донация включается на обеих точках
    (градиент и шаг) либо ни на одной.
    """
    master = model.init_params(jr.PRNGKey(seed), cfg)

    def loss_fn(params, batch):
        return model.compute_loss(params, cfg, batch, chunk_size=CHUNK)

    plain = jax.jit(jax.value_and_grad(loss_fn))
    donated = jax.jit(jax.value_and_grad(loss_fn), donate_argnums=(0,))
    step = optimizer.make_step(cfg, donate=donate)
    state = optimizer.init_state(master)
    params = _working(master)
    losses: list[float] = []
    for batch in batches:
        loss, grads = (donated if donate else plain)(params, batch)
        losses.append(float(jax.device_get(loss)))
        grads = jax.tree_util.tree_map(lambda leaf: leaf.astype(jnp.float32), grads)
        master, state = step(master, grads, state, lr)
        params = _working(master)
    return losses, params, master


def _leaves_equal(a, b, *, rtol: float = 0.0, atol: float = 0.0) -> bool:
    la, lb = jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b)
    if len(la) != len(lb):
        return False
    return all(bool(jnp.allclose(x, y, rtol=rtol, atol=atol)) for x, y in zip(la, lb))


def _spy_make_step(monkeypatch):
    """Обернуть ``optimizer.make_step`` наблюдателем kwargs (флаг donate)."""
    seen: list[dict] = []
    real = optimizer.make_step

    def spy(*args, **kwargs):
        seen.append(dict(kwargs))
        return real(*args, **kwargs)

    # train() импортирует модуль (``from net import model, optimizer``, :2787)
    # и читает атрибут в момент вызова — патч модуля виден циклу.
    monkeypatch.setattr(optimizer, "make_step", spy)
    return seen


# ---------------------------------------------------------------------------
# (а) Паритет: donate=True vs donate=False за 3 шага
# ---------------------------------------------------------------------------


def test_a_parity_three_steps_donate_vs_plain():
    """Donate не меняет арифметику: те же loss и params на тех же данных.

    bf16 — обязательный режим donate-ноги: в fp32 ``params is master``, и
    градиентная донация в цикле сознательно отключена (см. тест fp32 ниже).
    """
    cfg = tiny_config()
    batches = _batches(cfg, 3, seed=5)

    losses_donated, params_donated, master_donated = _mini_loop(
        cfg, batches, donate=True
    )
    losses_plain, params_plain, master_plain = _mini_loop(cfg, batches, donate=False)

    assert len(losses_donated) == len(losses_plain) == 3
    for index, (a, b) in enumerate(zip(losses_donated, losses_plain)):
        assert math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-7), (
            f"шаг {index}: loss donate={a!r} vs plain={b!r} — donate изменил арифметику"
        )
    assert _leaves_equal(params_donated, params_plain, rtol=1e-6, atol=1e-7), (
        "bf16-params donate-пути разошлись с безопасным путём"
    )
    assert _leaves_equal(master_donated, master_plain, rtol=1e-6, atol=1e-7), (
        "fp32-мастер donate-пути разошёлся с безопасным путём"
    )


# ---------------------------------------------------------------------------
# (б) Повторные donate-шаги реального цикла: буфер переживает ровно один шаг
# ---------------------------------------------------------------------------


def test_b_train_five_donated_steps_without_buffer_error(monkeypatch):
    """5 шагов ``train()`` (bf16) с donate=True — без «donated buffer reused».

    Премисса дельты: ``params`` пересоздаётся каждый шаг, ``master/grads/state``
    после donate не читаются (профайлер выключен).  Любое нарушение контракта
    владения даёт RuntimeError об удалённом буфере уже на втором шаге.
    """
    cfg = tiny_config()
    seen = _spy_make_step(monkeypatch)

    result = tl.train(
        cfg,
        iter(_batches(cfg, 5, seed=11)),
        train_config=tl.TrainConfig(
            steps=5,
            lr=1e-2,
            seed=7,
            schedule="cosine",
            param_dtype="bfloat16",
        ),
        budget=_free_budget(),
    )

    assert seen, "make_step не вызван — spy не увидел цикл"
    assert seen[0].get("donate") is True, (
        "цикл без профайлера не включил donate шага (ожидалось donate=True)"
    )
    assert result.steps_done == 5
    assert len(result.losses) == 5
    assert all(math.isfinite(loss) for loss in result.losses)
    assert isinstance(result.master_tree_hash, str) and result.master_tree_hash


# ---------------------------------------------------------------------------
# (в) Accum-ветка: params живы на всех микровызовах — donated-градиент запрещён
# ---------------------------------------------------------------------------


def test_c_accum_branch_keeps_params_alive_across_microbatches():
    """2 микробатча на шаг: ветка не берёт ``grad_fn_donated``.

    В accum-ветке ``grad_fn`` зовётся на каждом микробатче с теми же ``params``
    (token-weighted mean).  Донированный градиент пометил бы ``params``
    удалёнными, и второй микровызов упал бы («Array has been deleted») —
    поэтому исполнение обоих микробатчей на каждом из 3 шагов и есть предикат
    «ветка использует недонированный ``grad_fn``».  bf16 — режим, где
    одиночная ветка донирует, так что подмена веток не прошла бы молча.
    """
    cfg = tiny_config()
    result = tl.train(
        cfg,
        iter(_batches(cfg, 6, seed=13)),
        train_config=tl.TrainConfig(
            steps=3,
            lr=1e-2,
            seed=7,
            schedule="cosine",
            param_dtype="bfloat16",
            accum_tokens=2 * SEQ_LEN,  # target = 2 микробатча на шаг оптимизатора
            micro_batch=1,
        ),
        budget=_free_budget(),
    )

    assert result.steps_done == 3, (
        "accum-прогон не доехал — второй микровызов получил мёртвый буфер params?"
    )
    assert len(result.losses) == 3
    assert all(math.isfinite(loss) for loss in result.losses)


# ---------------------------------------------------------------------------
# (г) phase_profile=True → donate=False: профайлер читает буферы после шага
# ---------------------------------------------------------------------------


def test_d_phase_profile_disables_step_donation(monkeypatch, tmp_path: Path):
    """Живой профайлер запрещает donate шага; его ноги обязаны быть измерены.

    ``profiler.measure`` (:3037) получает ``grads`` после ``step_fn`` — при
    donate он читал бы удалённый буфер, ошибку проглотил бы (ноги
    диагностические, ``attempt`` ловит исключения) и записал ``null``.  Spy
    фиксирует флаг контракта (``donate=False``), а ``sec_backopt`` — что нога
    шага реально измерена, т.е. буферы живы.
    """
    cfg = tiny_config()
    seen = _spy_make_step(monkeypatch)
    metrics_path = tmp_path / "metrics.jsonl"

    tl.train(
        cfg,
        iter(_batches(cfg, 2, seed=17)),
        train_config=tl.TrainConfig(
            steps=2,
            lr=1e-2,
            seed=7,
            schedule="cosine",
            param_dtype="bfloat16",
            phase_profile=True,
            kpi_every=1,
            metrics_path=metrics_path,
        ),
        budget=_free_budget(),
    )

    assert seen, "make_step не вызван — spy не увидел цикл"
    assert seen[0].get("donate") is False, (
        "при phase_profile=True шаг собран с donate — профайлеру нельзя"
    )
    rows = tl.MetricsWriter.read(metrics_path)
    assert len(rows) == 2
    for row in rows:
        assert row.get("phase_profile") is True
        assert row.get("sec_backopt") is not None, (
            f"шаг {row.get('step')}: нога sec_backopt не измерена — "
            "профайлер получил мёртвый буфер (donate шага не выключен?)"
        )


# ---------------------------------------------------------------------------
# fp32-одиночная ветка: donate градиента запрещён (params is master)
# ---------------------------------------------------------------------------


def test_fp32_single_branch_stays_executable():
    """fp32: ``working_params`` отдаёт сам мастер — донация градиента запрещена.

    В fp32 буфер ``params`` — это буфер ``master``, который читает ``step_fn``
    сразу после градиента (:3024); donate там был бы use-after-free.  Прогон
    обязан исполняться (регрессия «включили donate в fp32» падает здесь же,
    с удалённым буфером).
    """
    cfg = tiny_config()
    result = tl.train(
        cfg,
        iter(_batches(cfg, 2, seed=23)),
        train_config=tl.TrainConfig(
            steps=2,
            lr=1e-2,
            seed=7,
            schedule="cosine",
            param_dtype="float32",
        ),
        budget=_free_budget(),
    )
    assert result.steps_done == 2
    assert all(math.isfinite(loss) for loss in result.losses)
