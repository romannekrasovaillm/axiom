"""Цикл-28 кампании MFU-55 (ADR-054 Am.49): memory-lean ``bwd_moe``-нога
phase-profile + fail-closed валидация дормантных FP8-гейтов.

Мотив дельты:

* ``sec_bwd_moe`` был ``null`` с Am.35 — ``jax.grad`` по ``_phase_moe_stack``
  держал активации всех MoE-слоёв и падал OOM при B=4.  Нога оборачивает стек
  в :func:`jax.checkpoint`: память ≤ одного слоя ценой пересчёта.  Время ноги
  поэтому — **ориентир**, а не партиция (``bwd_reconciliation`` уже помечает
  это); численные пути шага не меняются (нога диагностическая).
* ``AXIOM_FP8_ROW_SCALE`` читался только в fp8-ветке backward
  (``AXIOM_GEMM_FP8_BWD=1``), поэтому опечатка дормантного гейта не ловилась
  ни при ``BWD=0``, ни при ``AXIOM_GEMM_FP8=0``.  Теперь все FP8-гейты
  валидируются на каждом трейсе ``gemm``/``gemm_fp8`` и в правиле backward
  (fail-closed); корректные ``0``/``1`` поведения не меняют.

Что пиннуется здесь (CPU, jax 0.11.2, шов ``_fp8_matmul`` — деквантованным
f32-матмулом, значения e4m3 в f32 точны):

* **(а) нога ``bwd_moe``**: ``PhaseProfiler`` на малой конфигурации строит
  ногу, она компилируется и исполняется; результат — pytree градиентов той же
  структуры/формы, что ``params`` (не ``None``); в jaxpr ноги присутствует
  remat-примитив (``jax.checkpoint`` в графе);
* **(б) гейты**: ``AXIOM_FP8_ROW_SCALE=мусор`` — ошибка и при ``BWD=0``, и при
  ``AXIOM_GEMM_FP8=0`` (на ``gemm``, ``gemm(..., fp8=True)``, ``gemm_fp8`` и
  правиле backward); неизвестные ``AXIOM_GEMM_FP8``/``AXIOM_GEMM_FP8_BWD`` —
  ошибка всегда; корректные ``0``/``1`` (пробелы нормализуются) — без ошибок
  и без изменения дормантного поведения (бит-в-бит);
* **(в) регресс** ``test_50_gemm_fp8``/``test_51_gemm_fp8_bwd`` — отдельные
  файлы, здесь не дублируются.

Окружение: CPU-интерпретатор, ``NET_JAX_BACKEND=cpu``/``JAX_PLATFORMS=cpu``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from conftest import small_config

from net import compute_dtype, model
from net import train_loop as tl

#: Гейты дельты.
FP8_GATE = compute_dtype.GEMM_FP8_ENV
BWD_GATE = compute_dtype.GEMM_FP8_BWD_ENV
ROW_GATE = compute_dtype.FP8_ROW_SCALE_ENV


# --------------------------------------------------------------------------- #
# Общие помощники
# --------------------------------------------------------------------------- #


def _dequant_matmul(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    """Шов CPU: fp8-операнды деквантуются в f32 (значения e4m3 точны в f32)."""
    return jnp.matmul(a.astype(jnp.float32), b.astype(jnp.float32))


@pytest.fixture(autouse=True)
def _cpu_bodies(monkeypatch):
    """CPU-исполнение: гейты сняты, шов CPU подменён (форма test_50/test_51)."""
    for env in (
        compute_dtype.MODE_ENV,
        FP8_GATE,
        BWD_GATE,
        ROW_GATE,
    ):
        monkeypatch.delenv(env, raising=False)
    monkeypatch.setattr(compute_dtype, "_fp8_matmul", _dequant_matmul)


def _set_gates(
    monkeypatch,
    *,
    fp8: str | None = None,
    bwd: str | None = None,
    row: str | None = None,
) -> None:
    """Явно выставить оси гейтов; ``None`` — снять переменную (дефолт ``0``)."""
    for env, value in ((FP8_GATE, fp8), (BWD_GATE, bwd), (ROW_GATE, row)):
        if value is None:
            monkeypatch.delenv(env, raising=False)
        else:
            monkeypatch.setenv(env, value)


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
    """Параметры малой модели и батч ``(B, T) int32`` для MoE-ноги."""
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


# --------------------------------------------------------------------------- #
# (а) нога bwd_moe: checkpoint, компиляция, pytree градиентов
# --------------------------------------------------------------------------- #


def test_bwd_moe_leg_compiles_runs_and_returns_param_pytree():
    """Нога ``bwd_moe`` исполняется; градиенты — pytree формы ``params``.

    ``jax.checkpoint`` вокруг ``_phase_moe_stack`` — ремат: память ноги ≤ одного
    слоя вместо активаций всех MoE-слоёв (Am.49).  Проверяется структура
    результата (не ``None``, тот же treedef и формы листьев) и конечность.
    """
    cfg = small_config()
    params, ids = _moe_batch(cfg)
    profiler = _profiler(cfg)
    leg = profiler._bwd_fns["bwd_moe"]

    lowered = leg.lower(params, ids)
    grads = lowered.compile()(params, ids)

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
        assert jnp.issubdtype(grad.dtype, jnp.floating), (
            f"лист {index}: не float dtype {grad.dtype}"
        )
        assert bool(jnp.isfinite(grad).all()), f"лист {index}: nan/inf в градиенте"
    assert any(
        float(jnp.max(jnp.abs(grad))) > 0.0
        for grad in jax.tree_util.tree_leaves(grads)
    ), "все градиенты нулевые — нога не дифференцирует MoE-стек"


def test_bwd_moe_leg_contains_remat_in_jaxpr():
    """``jax.checkpoint`` попал в граф: в jaxpr ноги есть remat-примитив.

    Примитив JAX 0.11.2 называется ``remat2``; проверяется префикс ``remat``,
    чтобы тест не привязывался к номеру версии схемы.
    """
    cfg = small_config()
    params, ids = _moe_batch(cfg)
    leg = _profiler(cfg)._bwd_fns["bwd_moe"]

    names = _primitives(jax.make_jaxpr(leg)(params, ids))
    remat = [name for name in names if name.startswith("remat")]
    assert remat, f"в jaxpr ноги нет remat-примитива: {sorted(set(names))}"


def test_bwd_moe_leg_differs_from_uncheckpointed_stack():
    """Ремат не меняет значения: градиенты бит-в-бит равны нерематной ноге.

    ``jax.checkpoint`` — только про память/пересчёт; численный результат
    диагностической ноги обязан совпадать с прежней формой (Am.49: численные
    пути не меняются).
    """
    cfg = small_config()
    params, ids = _moe_batch(cfg)

    checked = _profiler(cfg)._bwd_fns["bwd_moe"](params, ids)
    plain = jax.jit(
        jax.grad(lambda p, i: jnp.sum(tl._phase_moe_stack(p, cfg, i)))
    )(params, ids)

    for index, (a, b) in enumerate(
        zip(jax.tree_util.tree_leaves(checked), jax.tree_util.tree_leaves(plain))
    ):
        assert _bitwise_equal(a, b), f"лист {index}: ремат изменил градиент"


# --------------------------------------------------------------------------- #
# (б) fail-closed валидация FP8-гейтов
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "fp8,bwd",
    [("0", "0"), ("0", "1"), ("1", "0"), ("1", "1")],
    ids=["off-off", "off-bwd", "fp8-off", "fp8-bwd"],
)
def test_row_gate_typo_fails_closed_on_every_path(monkeypatch, fp8, bwd):
    """``AXIOM_FP8_ROW_SCALE=мусор`` — ошибка при любых ``FP8``/``BWD``.

    До Am.49 ROW читался только в ``_gemm_fp8_bwd`` под ``BWD=1``: при
    ``BWD=0`` опечатка молчала.  Теперь валидация идёт на входе ``gemm``,
    ``gemm(..., fp8=True)``, ``gemm_fp8`` и правила backward — независимо от
    значений прочих гейтов (fail-closed).
    """
    _set_gates(monkeypatch, fp8=fp8, bwd=bwd, row="garbage")
    a = jr.normal(jr.PRNGKey(0), (8, 6))
    b = jr.normal(jr.PRNGKey(1), (6, 4))
    g = jr.normal(jr.PRNGKey(2), (8, 4))

    with pytest.raises(compute_dtype.Fp8RowScaleError):
        compute_dtype.gemm(a, b)
    with pytest.raises(compute_dtype.Fp8RowScaleError):
        compute_dtype.gemm(a, b, fp8=True)
    with pytest.raises(compute_dtype.Fp8RowScaleError):
        compute_dtype.gemm_fp8(a, b)
    with pytest.raises(compute_dtype.Fp8RowScaleError):
        compute_dtype._gemm_fp8_bwd((a, b), g)


def test_all_gate_typos_fail_closed_even_when_others_are_off(monkeypatch):
    """Каждый гейт валидируется сам, а не «если включён сосед» (Am.49)."""
    a = jr.normal(jr.PRNGKey(0), (8, 6))
    b = jr.normal(jr.PRNGKey(1), (6, 4))

    _set_gates(monkeypatch, fp8="0", bwd="0", row="0")
    monkeypatch.setenv(FP8_GATE, "garbage")
    with pytest.raises(compute_dtype.GemmFp8Error):
        compute_dtype.gemm(a, b)

    _set_gates(monkeypatch, fp8="0", bwd="0", row="0")
    monkeypatch.setenv(BWD_GATE, "garbage")
    with pytest.raises(compute_dtype.GemmFp8BwdError):
        compute_dtype.gemm(a, b)

    _set_gates(monkeypatch, fp8="0", bwd="0", row="0")
    monkeypatch.setenv(ROW_GATE, "garbage")
    with pytest.raises(compute_dtype.Fp8RowScaleError):
        compute_dtype.gemm(a, b)


@pytest.mark.parametrize(
    "row", ["0", "1", " 0 ", " 1 ", "\t1\n", "\n0\t"], ids=repr
)
def test_valid_row_values_do_not_change_dormant_behavior(monkeypatch, row):
    """Корректные ``0``/``1`` (пробелы нормализуются) — без ошибок и без сдвига.

    При ``BWD=0`` ROW дормантен: forward и backward ``gemm_fp8`` обязаны
    совпасть бит-в-бит с ROW=0.  Заодно проверяется, что пробельные формы
    валидны и не включают ветку.
    """
    a = jr.normal(jr.PRNGKey(0), (8, 6))
    b = jr.normal(jr.PRNGKey(1), (6, 4)) * 0.02
    g = jr.normal(jr.PRNGKey(2), (8, 4))

    def loss(x, y):
        return jnp.sum(compute_dtype.gemm_fp8(x, y) * g)

    _set_gates(monkeypatch, fp8="1", bwd="0", row="0")
    c_ref = compute_dtype.gemm_fp8(a, b)
    ga_ref, gb_ref = jax.grad(loss, argnums=(0, 1))(a, b)

    _set_gates(monkeypatch, fp8="1", bwd="0", row=row)
    c = compute_dtype.gemm_fp8(a, b)
    ga, gb = jax.grad(loss, argnums=(0, 1))(a, b)

    assert _bitwise_equal(c, c_ref), f"ROW={row!r} сдвинул forward при BWD=0"
    assert _bitwise_equal(ga, ga_ref), f"ROW={row!r} сдвинул ga при BWD=0"
    assert _bitwise_equal(gb, gb_ref), f"ROW={row!r} сдвинул gb при BWD=0"


@pytest.mark.parametrize("raw", ["TRUE", "On", "01", "-1", "2", "fP8"])
def test_gate_values_are_case_insensitive_but_strict(monkeypatch, raw):
    """Регистр нормализуется, но нечисловые значения — ошибка (без догадок)."""
    a = jr.normal(jr.PRNGKey(0), (8, 6))
    b = jr.normal(jr.PRNGKey(1), (6, 4))

    _set_gates(monkeypatch, fp8="0", bwd="0", row=raw)
    with pytest.raises(compute_dtype.Fp8RowScaleError):
        compute_dtype.gemm(a, b)


def test_correct_gate_combinations_run(monkeypatch):
    """Все корректные комбинации ``0``/``1`` исполняются без ошибок."""
    a = jr.normal(jr.PRNGKey(0), (8, 6))
    b = jr.normal(jr.PRNGKey(1), (6, 4)) * 0.02
    g = jr.normal(jr.PRNGKey(2), (8, 4))

    for fp8 in ("0", "1"):
        for bwd in ("0", "1"):
            for row in ("0", "1"):
                _set_gates(monkeypatch, fp8=fp8, bwd=bwd, row=row)
                out = compute_dtype.gemm(a, b, fp8=True)
                assert bool(jnp.isfinite(out).all())

                def loss(x, y):
                    return jnp.sum(compute_dtype.gemm_fp8(x, y) * g)

                ga, gb = jax.grad(loss, argnums=(0, 1))(a, b)
                for grad in (ga, gb):
                    assert bool(jnp.isfinite(grad).all())
