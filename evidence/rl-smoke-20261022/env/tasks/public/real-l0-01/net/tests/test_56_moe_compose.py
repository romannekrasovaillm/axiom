"""Цикл-36 (ADR-054 Am.59): модельный композит fused dispatch-пути MoE.

Стендовый A/B ``AXIOM_MOE_DISPATCH=fused`` (вместе с ``AXIOM_GEMM_FP8=1`` и
``AXIOM_MOE_BWD=fused``) упал на первом шаге:

    sum requires ndarray or scalar arguments, got <class 'tuple'> at position 0

Dispatch-кернел при этом проверен на GPU побитово (Am.59), поэтому дефект жил
в обвязке combine/scatter: ``out_type`` Mosaic-кернела — **дерево**, и у
scatter оно состоит из одного ``ShapeDtypeStruct``, поэтому Pallas возвращает
кортеж ``(dz_k,)``, а не голый массив.  Обёртка :func:`moe_dispatch.scatter`
кормила этот кортеж в ``jnp.sum(dz_k, axis=1)`` — отсюда стендовая ошибка.
Цикл-35 этого не ловил: ``test_55`` подменяет *сами обёртки*
``combine``/``scatter`` jnp-эквивалентами, то есть исполняет reference, а не
проводку возврата.

Что пинуется здесь (CPU, jax 0.11.2, ``NET_JAX_BACKEND=cpu``):

* **(а) корень дефекта**: ``jax.eval_shape`` scatter-кернела возвращает
  кортеж из одного элемента — структуру ``out_type``; обёртки
  :func:`moe_dispatch.combine`/:func:`moe_dispatch.scatter` распаковывают её
  явно и отдают ровно по одному ndarray;
* **(б) модельный композит** ``moe.apply`` с **настоящими** обёртками
  kernel/combine/scatter и низкоуровневыми jnp-швами
  ``_kernel_run``/``_combine_run``/``_scatter_run`` (шов ниже дефекта, поэтому
  возврат не подменяется): ``AXIOM_MOE_DISPATCH=fused`` — forward побитово
  равен off-пути, градиенты по параметрам/x scale-relative ``≤1e-6`` (это
  перестановка строк, а не приближение); ``AXIOM_MOE_BWD=fused`` отдельно и
  вместе с fused-dispatch — ``≤2e-2`` (bf16-операнды экспертного кернела);
* **(в) ``compute_loss``/``jax.grad`` на tiny-модели** (``moe_latent_dim=128``
  — требование сита ``valid_config``: ``latent % bc == 0``, ``bc=128``) с
  fused-гейтом — без исключений, потеря побитово прежняя, форма градиентов
  совпадает с параметрами, значения — в кампанийном допуске ``2e-2``
  (строгий пин перестановки ``≤1e-6`` стоит на самом ``moe.apply``);
* **(г) регресс**: off-путь (``AXIOM_MOE_DISPATCH`` не задан) побитово
  совпадает с ревизией до дельты (``baseline_commit`` пакета, ``290d820``) —
  значение и все листья градиентов.

Границы честности: численный GPU-паритет исполненных Mosaic-кернелов и
стендовый A/B — за архитектором; здесь пинуется CPU-семантика, структура
возврата Pallas и отсутствие регрессии off-пути.  Компромисс «dispatch-only»
не выбран: fused combine/scatter чинится явной распаковкой одного выхода.
"""

from __future__ import annotations

import dataclasses
import hashlib
import io
import json
import os
import subprocess
import sys
import tarfile
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

from net import model, moe
from net.kernels import moe_dispatch, moe_experts_bwd

#: Ревизия до дельты (``baseline_commit`` handoff-пакета цикла-36): off-путь
#: обязан воспроизводить её побитово.  ``git archive`` + skip, если недоступна.
BASELINE_COMMIT = "290d820"

CASE_DIR = Path(__file__).resolve().parents[2]

#: Допуск fused-dispatch (только перестановка строк): практически побитово.
DISPATCH_TOL = 1e-6

#: Кампанийный допуск bf16-операндов экспертного backward'а (Am.52/59).
BWD_TOL = 2e-2


@pytest.fixture(autouse=True)
def _cpu_bodies(monkeypatch):
    """CPU-исполнение: KDA — ``jax``; гейты цикла-36 сняты по умолчанию."""
    monkeypatch.setenv("AXIOM_KDA_CC_IMPL", "jax")
    monkeypatch.delenv(moe.MOE_DISPATCH_ENV, raising=False)
    monkeypatch.delenv(moe.MOE_BWD_ENV, raising=False)
    monkeypatch.delenv(moe.COUNTING_MODE_ENV, raising=False)


def _composite_config(**overrides):
    """Малая MoE-конфигурация, проходящая сито fused-пути.

    ``valid_config`` требует ``latent % bc == 0`` при ``bc=128``, поэтому
    ``moe_latent_dim=128``; ``tokens·k`` обязано делиться на ``bm=16``.
    """
    from conftest import small_config

    return small_config(moe_latent_dim=128, **overrides)


# --------------------------------------------------------------------------- #
# Низкоуровневые швы: jnp вместо Mosaic-run'ов, настоящие обёртки остаются
# --------------------------------------------------------------------------- #


def _pad_dump(a: jnp.ndarray, n_exp: int, fill) -> jnp.ndarray:
    """Дописать dump-слоты (``rows + n_exp``): обёртка обязана их срезать.

    Значащие строки — первые ``rows``; dump-слоты заполняются мусором
    (``nan``/``-1``), чтобы случайное использование срезов за пределами
    ``[:rows]`` падало, а не совпадало с нулями.
    """
    tail = jnp.full((n_exp,) + a.shape[1:], fill, a.dtype)
    return jnp.concatenate([a, tail], axis=0)


def _fake_kernel_run(tokens, k, latent, n_exp, bm, bc, dtype):
    """Шов dispatch-run'а: каноническая раскладка + dump-слоты."""

    def run(z, topk_flat, group_sizes, offsets):
        zg, sizes, assign, pair = moe_dispatch.reference_dispatch(
            z, jnp.asarray(topk_flat).reshape(tokens, k), None, n_exp=n_exp)
        return (_pad_dump(zg, n_exp, jnp.nan), sizes,
                _pad_dump(assign, n_exp, -1), _pad_dump(pair, n_exp, -1))

    return run


def _fake_combine_run(tokens, k, latent, n_exp, bm, bc, dtype):
    """Шов combine-run'а: два выхода — ``(de_grouped, dsel_p)``, как у Pallas."""

    def run(du, e_grouped, pair, sel_p, group_sizes, offsets):
        return moe_dispatch.reference_combine(
            du, e_grouped, pair, sel_p, group_sizes, n_exp=n_exp)

    return run


def _fake_scatter_run(tokens, k, latent, n_exp, bm, bc, dtype):
    """Шов scatter-run'а: **буфер** ``(tokens, k, latent)`` в кортеже из одного.

    Это ровно структура ``out_type`` кернела (Pallas-дерево из одного листа) —
    именно её цикл-35 скормил ``jnp.sum``.  Сумма по ``k`` остаётся в
    обёртке :func:`moe_dispatch.scatter`, поэтому шов не подменяет её.
    """

    def run(dz_grouped, pair, group_sizes, offsets):
        pair = pair.astype(jnp.int32)
        tok, slot = pair // k, pair % k
        dz_k = jnp.zeros((tokens, k, latent), jnp.float32).at[tok, slot].set(
            dz_grouped.astype(jnp.float32))
        return (dz_k,)

    return run


def _install_cpu_seams(monkeypatch):
    """Подменить только исполнителей Mosaic; обёртки и их возврат — настоящие."""
    monkeypatch.setattr(moe_dispatch, "device_available", lambda: True)
    monkeypatch.setattr(moe_dispatch, "_kernel_run", _fake_kernel_run)
    monkeypatch.setattr(moe_dispatch, "_combine_run", _fake_combine_run)
    monkeypatch.setattr(moe_dispatch, "_scatter_run", _fake_scatter_run)


def _set_gates(monkeypatch, mode: str):
    """Включить гейты цикла-36: ``dispatch`` | ``bwd`` | ``both``."""
    monkeypatch.delenv(moe.MOE_DISPATCH_ENV, raising=False)
    monkeypatch.delenv(moe.MOE_BWD_ENV, raising=False)
    if mode in ("dispatch", "both"):
        monkeypatch.setenv(moe.MOE_DISPATCH_ENV, moe.MOE_DISPATCH_FUSED)
    if mode in ("bwd", "both"):
        monkeypatch.setenv(moe.MOE_BWD_ENV, moe.MOE_BWD_FUSED)
        monkeypatch.setattr(moe_experts_bwd, "kernel",
                            moe_experts_bwd.reference_bwd)


def _tolerance(mode: str) -> float:
    """Dispatch-only — перестановка (``1e-6``), fused-bwd — bf16 (``2e-2``)."""
    return DISPATCH_TOL if mode == "dispatch" else BWD_TOL


def _spy(monkeypatch, name: str) -> list:
    """Обернуть настоящую функцию модуля счётчиком вызовов (проводка, не шов)."""
    real = getattr(moe_dispatch, name)
    calls: list = []

    def wrapper(*args, **kwargs):
        calls.append((args, kwargs))
        return real(*args, **kwargs)

    monkeypatch.setattr(moe_dispatch, name, wrapper)
    return calls


def _tree_rel(on, off) -> float:
    """Худшее scale-relative ``max|d|`` по всем листьям дерева."""
    worst = 0.0
    for a, b in zip(jax.tree_util.tree_leaves(on), jax.tree_util.tree_leaves(off)):
        delta = float(jnp.max(jnp.abs(a - b)))
        scale = max(float(jnp.max(jnp.abs(a))), float(jnp.max(jnp.abs(b))))
        worst = max(worst, delta / scale if scale else 0.0)
    return worst


# --------------------------------------------------------------------------- #
# (а) корень дефекта: структура возврата Pallas и распаковка в обёртках
# --------------------------------------------------------------------------- #


def test_scatter_kernel_out_type_is_a_single_output_tree():
    """``out_type`` scatter — дерево из одного листа, значит возврат — кортеж.

    Это структурный пин дефекта цикла-36: ``jnp.sum`` на таком кортеже падает
    с "sum requires ndarray or scalar arguments".  Lowering/исполнение не
    нужны — ``eval_shape`` доступен на CPU.
    """
    tokens, k, latent, n_exp, bm, bc = 16, 2, 128, 4, 16, 128
    kernel = moe_dispatch._build_scatter_kernel(
        tokens, k, latent, n_exp, bm, bc, "float32")
    out = jax.eval_shape(
        kernel,
        jax.ShapeDtypeStruct((tokens * k, latent), jnp.float32),
        jax.ShapeDtypeStruct((tokens * k,), jnp.int32),
        jax.ShapeDtypeStruct((n_exp,), jnp.int32),
        jax.ShapeDtypeStruct((n_exp,), jnp.int32),
    )
    assert isinstance(out, tuple) and len(out) == 1
    assert out[0].shape == (tokens, k, latent)
    assert jnp.dtype(out[0].dtype) == jnp.float32


def _layout(tokens=16, k=2, latent=128, n_exp=4, dtype="bfloat16"):
    """``(z, topk, sel_p, zg, sizes, assign, pair)`` для прямой проверки обёрток."""
    z, topk, sel_p = moe_dispatch.make_inputs(
        jr.PRNGKey(0), tokens=tokens, k=k, latent=latent, n_exp=n_exp,
        dtype=dtype)
    zg, sizes, assign, pair = moe_dispatch.reference_dispatch(
        z, topk, sel_p, n_exp=n_exp)
    return z, topk, sel_p, zg, sizes, assign, pair


def test_combine_wrapper_returns_two_arrays(monkeypatch):
    """``combine`` отдаёт ровно ``de_grouped`` и ``dsel_p`` — по одному массиву."""
    _install_cpu_seams(monkeypatch)
    tokens, k, latent, n_exp = 16, 2, 128, 4
    z, topk, sel_p, zg, sizes, assign, pair = _layout()
    du = jr.normal(jr.PRNGKey(5), (tokens, latent)).astype(z.dtype)

    de, dsel = moe_dispatch.combine(
        du, zg, pair, sel_p, sizes, n_exp=n_exp)
    de_ref, dsel_ref = moe_dispatch.reference_combine(
        du, zg, pair, sel_p, sizes, n_exp=n_exp)

    assert isinstance(de, jax.Array) and isinstance(dsel, jax.Array)
    assert de.shape == (tokens * k, latent)
    assert dsel.shape == (tokens, k)
    assert jnp.dtype(dsel.dtype) == jnp.float32
    assert jnp.array_equal(de, de_ref)
    assert jnp.array_equal(dsel, dsel_ref)


def test_scatter_wrapper_unpacks_the_single_output(monkeypatch):
    """``scatter`` распаковывает ``(dz_k,)`` и возвращает голый ``dz``.

    Шов возвращает кортеж из одного элемента — как настоящий Pallas-кернел;
    обёртка обязана отдать ndarray ``(tokens, latent)``, а не кортеж.
    """
    _install_cpu_seams(monkeypatch)
    tokens, k, latent, n_exp = 16, 2, 128, 4
    _, _, _, zg, sizes, _, pair = _layout()
    dz_grouped = jr.normal(
        jr.PRNGKey(6), (tokens * k, latent), jnp.float32)

    dz = moe_dispatch.scatter(
        dz_grouped, pair, sizes, n_tokens=tokens, k=k, n_exp=n_exp)
    dz_ref = moe_dispatch.reference_scatter(
        dz_grouped, pair, sizes, n_tokens=tokens, k=k, n_exp=n_exp)

    assert isinstance(dz, jax.Array), "в обёртку просочился кортеж Pallas"
    assert dz.shape == (tokens, latent)
    assert jnp.array_equal(dz, dz_ref)


# --------------------------------------------------------------------------- #
# (б) модельный композит: moe.apply, настоящие обёртки
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("mode", ["dispatch", "bwd", "both"])
def test_moe_apply_composite_matches_off(monkeypatch, mode):
    """``moe.apply``: forward побитово, градиенты по параметрам/x в допуске.

    ``dispatch`` — только перестановка строк (fused dispatch + jnp-швы
    combine/scatter): ``≤1e-6``.  ``bwd``/``both`` дополнительно включают
    эталонный jnp-адъюнкт экспертного кернела (bf16-точки округления):
    ``≤2e-2``.  Именно эта связка падала на стенде цикл-35 → цикл-36.
    """
    cfg = _composite_config()
    params = moe.init_moe(jr.PRNGKey(2), cfg)
    x = jr.normal(jr.PRNGKey(4), (16, cfg.hidden))

    def loss(p, xx):
        return jnp.sum(moe.apply(p, cfg, xx))

    off_value = moe.apply(params, cfg, x)
    off_grads = jax.grad(loss, argnums=(0, 1))(params, x)

    # Швы — только исполнители Mosaic; обёртки остаются настоящими, и их
    # вызовы считаются: композит обязан пройти именно через проводку цикла-36.
    calls = {}
    if mode in ("dispatch", "both"):
        for name in ("kernel", "combine", "scatter"):
            calls[name] = _spy(monkeypatch, name)
    _install_cpu_seams(monkeypatch)
    _set_gates(monkeypatch, mode)
    on_value = moe.apply(params, cfg, x)
    on_grads = jax.grad(loss, argnums=(0, 1))(params, x)

    if mode in ("dispatch", "both"):
        assert all(calls[name] for name in ("kernel", "combine", "scatter")), (
            "композит не прошёл через обёртки moe_dispatch: "
            f"{ {k: len(v) for k, v in calls.items()} }")
    assert jnp.array_equal(on_value, off_value), "forward сдвинулся"
    rel = _tree_rel(on_grads, off_grads)
    print(f"moe.apply mode={mode}: max|d|/scale={rel:.4e} (tol {_tolerance(mode):.0e})")
    assert rel <= _tolerance(mode)


# --------------------------------------------------------------------------- #
# (в) compute_loss/jax.grad на tiny-модели с fused-гейтом
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("mode", ["dispatch", "both"])
def test_compute_loss_grad_on_tiny_model(monkeypatch, mode):
    """Полный ``compute_loss`` + ``jax.grad``: без исключений, форма и допуск.

    Гейт ``AXIOM_MOE_DISPATCH=fused`` (и вместе с ``AXIOM_MOE_BWD=fused``)
    проходит через все MoE-слои tiny-модели: потеря побитово прежняя, дерево
    градиентов повторяет дерево параметров, значения — в объявленном допуске.
    """
    from conftest import tiny_config

    cfg = dataclasses.replace(tiny_config(), moe_latent_dim=128)
    params = model.init_params(jr.PRNGKey(0), cfg)
    ids = jr.randint(jr.PRNGKey(1), (2, 16), 0, cfg.vocab_size)

    def loss(p):
        return model.compute_loss(p, cfg, ids, chunk_size=8)

    off_value = loss(params)
    off_grads = jax.grad(loss)(params)

    _install_cpu_seams(monkeypatch)
    _set_gates(monkeypatch, mode)
    on_value = loss(params)
    on_grads = jax.grad(loss)(params)

    assert bool(jnp.isfinite(on_value))
    assert jnp.array_equal(on_value, off_value), "потеря сдвинулась"
    on_leaves = jax.tree_util.tree_leaves(on_grads)
    param_leaves = jax.tree_util.tree_leaves(params)
    assert len(on_leaves) == len(param_leaves)
    assert all(a.shape == b.shape for a, b in zip(on_leaves, param_leaves))
    rel = _tree_rel(on_grads, off_grads)
    print(f"compute_loss mode={mode}: max|d|/scale={rel:.4e} "
          f"(tol {BWD_TOL:.0e})")
    # Строгий пин «перестановки» (≤1e-6) стоит на самом moe.apply; в полной
    # модели разный порядок редукций слоёв усиливает ту же дельту до ~1e-6,
    # поэтому здесь действует кампанийный допуск bf16-композита.
    assert rel <= BWD_TOL


# --------------------------------------------------------------------------- #
# (г) регресс: off-путь побитово прежний (baseline_commit пакета)
# --------------------------------------------------------------------------- #

#: Проба off-пути: значение ``moe.apply`` и все листья градиентов (params, x).
_PROBE = '''
import hashlib
import json

import jax
import jax.numpy as jnp
import numpy as np

from net import config, moe

def digest(x):
    a = np.asarray(x)
    return {"shape": list(a.shape), "dtype": str(a.dtype),
            "sha": hashlib.sha256(a.tobytes()).hexdigest()}

cfg = config.ModelConfig(
    vocab_size=512, hidden=64, num_layers=4, num_kda_layers=3,
    num_mla_layers=1, num_heads=4, head_dim=16, kda_dk=16, kda_dv=16,
    kda_decay_rank=16, kda_short_conv_kernel=4, kda_g_min=-5.0,
    mla_latent_dim=32, mla_head_dim=16, mlp_intermediate=128,
    siti_beta_gate=4.0, siti_beta_up=25.0, mtp_layers=1,
    mtp_loss_weight=0.1, attnres_blocks=1, attnres_block_size=4,
    vit_patch=14, vit_hidden=32, vit_depth=2, vit_heads=2, vit_mlp=64,
    image_size=56, moe_dense_layers=1, moe_latent_dim=128,
    moe_num_routed=6, moe_num_shared=2, moe_top_k=2,
    moe_expert_intermediate=16, moe_shared_intermediate=32,
    qb_weight=0.01, routing_seed=0,
)
params = moe.init_moe(jax.random.PRNGKey(0), cfg)
x = jax.random.normal(jax.random.PRNGKey(1), (16, cfg.hidden))

value = jnp.sum(moe.apply(params, cfg, x))
grads = jax.grad(
    lambda p, xx: jnp.sum(moe.apply(p, cfg, xx)), argnums=(0, 1))(params, x)
out = {"value": digest(value),
       "grads": [digest(g) for g in jax.tree_util.tree_leaves(grads)]}
print(json.dumps(out, sort_keys=True))
'''


def _extract_baseline(tmp_path: Path) -> Path:
    """``net/`` ревизии до дельты в tmp; проба запускается оттуда же."""
    root = tmp_path / "baseline"
    root.mkdir()
    try:
        archive = subprocess.run(
            ["git", "-C", str(CASE_DIR), "archive", BASELINE_COMMIT, "net/"],
            check=True, capture_output=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        pytest.skip(f"baseline {BASELINE_COMMIT} недоступен: {exc}")
    with tarfile.open(fileobj=io.BytesIO(archive.stdout)) as bundle:
        bundle.extractall(root)
    return root


def _run_probe(root: Path, script: Path):
    """Запуск пробы в отдельном процессе (``PYTHONPATH`` — корень ревизии)."""
    env = {
        key: value
        for key, value in os.environ.items()
        if key not in {moe.MOE_DISPATCH_ENV, moe.MOE_BWD_ENV,
                       moe.COUNTING_MODE_ENV, "AXIOM_COMPUTE_DTYPE",
                       "PYTHONPATH"}
    }
    env.update({
        "NET_JAX_BACKEND": "cpu",
        "JAX_PLATFORMS": "cpu",
        "PYTHONPATH": str(root),
    })
    done = subprocess.run(
        [sys.executable, str(script)], cwd=root, env=env,
        capture_output=True, text=True,
    )
    if done.returncode != 0:  # pragma: no cover — диагностика прогона
        raise AssertionError(f"проба упала ({root}):\n{done.stderr[-2000:]}")
    lines = [line for line in done.stdout.strip().splitlines() if line.strip()]
    return json.loads(lines[-1])


def test_off_path_is_bitwise_the_baseline(tmp_path):
    """``AXIOM_MOE_DISPATCH`` не задан — значение и градиенты как до дельты."""
    baseline = _extract_baseline(tmp_path)
    script = tmp_path / "probe.py"
    script.write_text(_PROBE, encoding="utf-8")
    assert _run_probe(CASE_DIR, script) == _run_probe(baseline, script)
