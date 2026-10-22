"""Цикл-29 кампании MFU-55 (ADR-054 Am.51): видимая причина отказа ног
phase-profile и per-layer remat MoE-стека.

Мотив дельты:

* ``sec_bwd_moe`` был ``null``, а ``PhaseProfiler.attempt`` глотал исключение —
  причина (OOM или ошибка компиляции) не была видна.  Теперь тип и текст
  исключения ноги едут в аддитивное поле ``phase_errors`` записи
  phase-profile: время ноги остаётся ``null``, но запись отвечает «почему».
* Внешний ``jax.checkpoint`` Am.49 вокруг всего MoE-стека заменён per-layer
  remat внутри ``_phase_moe_stack(remat=True)`` — по образцу per-layer remat
  ``model.forward``: граница на слой, живой набор backward — вход слоя, а не
  активации всех слоёв.  Forward-нога (``sec_moe``) идёт без remat.

Что пиннуется здесь (CPU, jax 0.11.2):

* **(а)** ошибка ноги: искусственное исключение → ``phase_errors[field]``
  содержит тип и текст (усечённый), поле времени ``null``, замер не падает;
* **(б)** ``remat=True``: значения совпадают с нерематным стеком, в jaxpr —
  ровно один remat-примитив на слой; нога ``bwd_moe`` компилируется и
  возвращает градиенты той же структуры/формы, что ``params``;
* **(в)** forward-нога ``sec_moe`` побитово равна до-дельта-ревизии стека;
* **(г)** ``phase_errors`` присутствует в записи phase-profile и пуст при
  успехе (ошибка прошлого замера не протекает в следующий).

Численные пути шага не меняются: правки — только диагностическая нога.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from conftest import small_config

from net import mlp as mlp_mod
from net import model
from net import moe as moe_mod
from net import train_loop as tl

# --------------------------------------------------------------------------- #
# Общие помощники
# --------------------------------------------------------------------------- #


def _bits(x: jnp.ndarray) -> np.ndarray:
    """Сырые fp32-биты массива (NaN-безопасное побитовое сравнение)."""
    arr = np.asarray(jnp.asarray(x))
    assert arr.dtype == np.float32, f"побитовое сравнение ждёт fp32, получено {arr.dtype}"
    return arr.view(np.uint32)


def _bitwise_equal(a: jnp.ndarray, b: jnp.ndarray) -> bool:
    return a.shape == b.shape and bool(np.array_equal(_bits(a), _bits(b)))


def _primitives(obj, out=None, depth=0) -> list[str]:
    """Имена примитивов jaxpr, включая вложенные (jit/custom_vjp/remat)."""
    if out is None:
        out = []
    if depth > 30:
        return out
    eqns = getattr(obj, "eqns", None)
    if eqns is not None:
        for eqn in eqns:
            out.append(eqn.primitive.name)
            for param in eqn.params.values():
                if hasattr(param, "eqns") or hasattr(param, "jaxpr"):
                    _primitives(param, out, depth + 1)
            for var in eqn.invars:
                inner = getattr(var, "aval", None)
                if inner is not None and hasattr(inner, "jaxpr"):
                    _primitives(inner.jaxpr, out, depth + 1)
    elif hasattr(obj, "jaxpr"):
        _primitives(obj.jaxpr, out, depth + 1)
    return out


def _moe_batch(cfg, *, batch: int = 2, seq: int = 8):
    """Параметры малой модели и батч ``(B, T) int32`` для MoE-стека."""
    params = model.init_params(jr.PRNGKey(0), cfg)
    ids = jr.randint(
        jr.PRNGKey(1), (batch, seq), 0, cfg.vocab_size, dtype=jnp.int32
    )
    return params, ids


def _profiler(cfg):
    """``PhaseProfiler`` той же сборки, что в ``train`` (ноги не исполняются)."""
    noop = lambda p, ids: jnp.zeros((), jnp.float32)  # noqa: E731
    return tl._PhaseProfiler(
        cfg,
        tl.TrainConfig(steps=1, chunk_size=8),
        grad_fn=noop,
        forward_fn=noop,
    )


def _pre_delta_moe_stack(params, cfg, input_ids):
    """До-дельта-ревизия ``_phase_moe_stack`` (Am.49): эталон forward-ноги.

    Копия побитового поведения до Am.51: тот же порядок операций, без remat.
    ``sec_moe`` обязан остаться ей равным — граница нужна только backward-ноге.
    """
    from net.norm import rms_norm

    h = params.embedding[input_ids]
    for block in params.layers:
        x = rms_norm(h, block.norm_mlp)
        if isinstance(block.mlp, moe_mod.LatentMoEParams):
            h = h + moe_mod.apply(block.mlp, cfg, x)
        else:
            h = h + mlp_mod.apply(block.mlp, cfg, x)
    return h


def _stub_zero(*args):
    """Заглушка ноги: любая арность, скалярный ноль (провод, не арифметика)."""
    return jnp.zeros((), jnp.float32)


def _stub_profiler(cfg):
    """Профайлер с дешёвыми ногами: ``attempt``/``measure`` без компиляции модели.

    Все ноги заменены jit-заглушками; цены не важны — проверяется, что причина
    отказа попадает в запись, а успешный замер несёт пустой ``phase_errors``.
    """
    profiler = tl._PhaseProfiler(
        cfg,
        tl.TrainConfig(steps=1, chunk_size=8),
        grad_fn=jax.jit(_stub_zero),
        forward_fn=jax.jit(_stub_zero),
    )
    profiler._fns = {
        name: jax.jit(_stub_zero) for name in ("kda", "mla", "moe", "ce")
    }
    profiler._bwd_fns = {
        name: jax.jit(_stub_zero)
        for name in ("bwd_kda", "bwd_mla", "bwd_moe", "bwd_ce", "bwd_other")
    }
    return profiler


def _measure(profiler):
    """Один замер профайлера на фиктивных данных (все ноги — заглушки)."""
    return profiler.measure(
        params={"w": jnp.zeros((2,), jnp.float32)},
        batch=jnp.zeros((1, 2), jnp.int32),
        grads={"w": jnp.zeros((2,), jnp.float32)},
        master={"w": jnp.zeros((2,), jnp.float32)},
        state={},
        lr=0.0,
        step_fn=jax.jit(_stub_zero),
        step_seconds=1.0,
        loader_seconds=0.5,
    )


# --------------------------------------------------------------------------- #
# (а) причина отказа ноги видна, время null, замер не падает
# --------------------------------------------------------------------------- #


def test_attempt_error_is_recorded_and_time_stays_null():
    """Исключение ноги → ``phase_errors[field]`` с типом/текстом, время ``None``.

    До Am.51 ``except Exception`` глотал причину: ``sec_bwd_moe=null`` не
    отличал OOM от ошибки компиляции.  Текст усечён до 200 символов, форма
    возврата ``attempt`` прежняя — замер продолжается.
    """
    detail = "искусственный отказ bwd_moe: " + "x" * 500

    def boom(p, ids):
        raise RuntimeError(detail)

    profiler = _stub_profiler(small_config())
    profiler._bwd_fns["bwd_moe"] = boom

    record = _measure(profiler)

    assert record["sec_bwd_moe"] is None, "упавшая нога получила время"
    errors = record[tl.PHASE_ERRORS_FIELD]
    assert "sec_bwd_moe" in errors, f"причина отказа не записана: {errors}"
    assert errors["sec_bwd_moe"] == f"RuntimeError: {detail[:200]}"
    # Замер не упал: остальные ноги измерены заглушками.
    assert record["sec_moe"] is not None
    assert record["sec_bwd_kda"] is not None


# --------------------------------------------------------------------------- #
# (б) per-layer remat: значения, граф, компиляция backward-ноги
# --------------------------------------------------------------------------- #


def test_remat_stack_values_match_plain_stack():
    """``remat=True`` не меняет значения стека: побитово равно нерематному."""
    cfg = small_config()
    params, ids = _moe_batch(cfg)

    plain = jax.jit(lambda p, i: tl._phase_moe_stack(p, cfg, i))(params, ids)
    checked = jax.jit(lambda p, i: tl._phase_moe_stack(p, cfg, i, remat=True))(
        params, ids
    )

    assert plain.shape == checked.shape
    assert _bitwise_equal(plain, checked), "remat сдвинул значения MoE-стека"


def test_remat_boundary_per_layer_in_jaxpr():
    """В jaxpr remat-стека — по границе на слой; без remat границ нет.

    Примитив JAX 0.11.2 называется ``remat2``; проверяется префикс ``remat``,
    чтобы тест не привязывался к номеру схемы.
    """
    cfg = small_config()
    params, ids = _moe_batch(cfg)

    checked = jax.make_jaxpr(
        lambda p, i: tl._phase_moe_stack(p, cfg, i, remat=True)
    )(params, ids)
    names = _primitives(checked.jaxpr)
    remat = [name for name in names if name.startswith("remat")]
    assert len(remat) == len(params.layers), (
        f"границ remat {len(remat)} на {len(params.layers)} слоёв: {remat}"
    )

    plain = jax.make_jaxpr(lambda p, i: tl._phase_moe_stack(p, cfg, i))(params, ids)
    assert not [name for name in _primitives(plain.jaxpr) if name.startswith("remat")]


def test_bwd_moe_leg_compiles_and_returns_param_shaped_grads():
    """Нога ``bwd_moe`` компилируется; градиенты — pytree формы ``params``."""
    cfg = small_config()
    params, ids = _moe_batch(cfg)
    leg = _profiler(cfg)._bwd_fns["bwd_moe"]

    grads = leg.lower(params, ids).compile()(params, ids)

    assert grads is not None, "нога bwd_moe вернула None"
    assert jax.tree_util.tree_structure(grads) == jax.tree_util.tree_structure(
        params
    ), "pytree градиентов не совпал со структурой params"
    for index, (grad, param) in enumerate(
        zip(jax.tree_util.tree_leaves(grads), jax.tree_util.tree_leaves(params))
    ):
        assert grad is not None, f"лист {index}: градиент None"
        assert grad.shape == param.shape, (
            f"лист {index}: форма {grad.shape} != {param.shape}"
        )
        assert bool(jnp.isfinite(grad).all()), f"лист {index}: nan/inf в градиенте"
    assert any(
        float(jnp.max(jnp.abs(grad))) > 0.0
        for grad in jax.tree_util.tree_leaves(grads)
    ), "все градиенты нулевые — нога не дифференцирует MoE-стек"


# --------------------------------------------------------------------------- #
# (в) forward-нога sec_moe не изменилась
# --------------------------------------------------------------------------- #


def test_forward_moe_leg_is_bitwise_unchanged():
    """``sec_moe`` побитово равен до-дельта-ревизии стека (без remat)."""
    cfg = small_config()
    params, ids = _moe_batch(cfg)

    current = _profiler(cfg)._fns["moe"](params, ids)
    reference = jax.jit(lambda p, i: _pre_delta_moe_stack(p, cfg, i))(params, ids)

    assert _bitwise_equal(current, reference), "forward-нога sec_moe сдвинулась"


# --------------------------------------------------------------------------- #
# (г) phase_errors в записи: присутствует и пуст при успехе
# --------------------------------------------------------------------------- #


def test_phase_errors_field_is_present_and_empty_on_success():
    """Успешный замер несёт ``phase_errors == {}``; ошибка не протекает дальше."""
    profiler = _stub_profiler(small_config())

    record = _measure(profiler)
    assert tl.PHASE_ERRORS_FIELD == "phase_errors"
    assert "phase_errors" in record, "поля phase_errors нет в записи phase-profile"
    assert record["phase_errors"] == {}
    # Все ноги исполнены: поле времени не null (провод замера жив).
    assert all(record[field] is not None for field in tl.PHASE_PROFILE_FIELDS)

    # Ошибка относится к текущему замеру: после отказа и восстановления ноги
    # следующий замер снова пуст (причина не тянется из прошлого kpi-шага).
    def boom(p, ids):
        raise RuntimeError("разовый отказ")

    profiler._bwd_fns["bwd_moe"] = boom
    failed = _measure(profiler)
    assert "sec_bwd_moe" in failed["phase_errors"]
    profiler._bwd_fns["bwd_moe"] = jax.jit(_stub_zero)
    assert _measure(profiler)["phase_errors"] == {}
