"""Цикл-35 (ADR-054 Am.58): fused dispatch/scatter MoE — гейт ``AXIOM_MOE_DISPATCH``.

Am.56/57 разложили ``sec_bwd_moe`` = 7.144 с на fused-кернел экспертного блока
3.55 с и XLA-часть ~3.6 с (``expert_of → stable-sort → take(z) → ragged →
scatter(dz)`` — мелкие кернелы и латентность на 24 слоях).  Цикл-35 заменяет
эту последовательность Mosaic-кернелами :mod:`net.kernels.moe_dispatch`:
dispatch (каноническая группировка строк), combine (``de_grouped``/``dsel_p``),
scatter (``dz``).  Гейт ``AXIOM_MOE_DISPATCH`` по умолчанию ``"jax"`` —
прежний путь побитово.

Что пиннуется здесь (CPU, jax 0.11.2):

* **(а) контракт гейта**: пусто/не задано = ``"jax"``; ``"fused"``; регистр и
  пробелы нормализуются; неизвестное — :class:`net.moe.MoeDispatchError` до
  трейса; явный аргумент сильнее env; значение читается на трейсе;
* **(б) off — побитово ревизия до дельты** (``53a1336``, ``baseline_commit``
  пакета): значение, градиенты и jaxpr ``_routed_experts_grouped`` совпадают с
  извлечённым ``net/`` baseline — оба режима ``AXIOM_MOE_COUNTING``;
  блокирующий критерий цикла;
* **(в) on — канонический порядок**: ``z_grouped``/``group_sizes``/
  ``assign_rows``/``pair_of_row`` совпадают с прежней раскладкой (стабильно по
  эксперту, внутри — порядок появления); forward ``_routed_experts_grouped``
  побитово равен off; ``dz``/``dsel_p`` и градиенты весов — scale-relative
  ≤ 2e-2 (кампанийный допуск bf16) на нескольких сидах, пустой/неполной
  группе, интерливинге маршрутов, ``k=1``/``k=2`` и обоих режимах группировки;
  два запуска побитово совпадают (детерминизм);
* **(г) отказ типизирован**: ``valid_config`` вне живых случаев (некратные
  ``rows``/``latent``, не bf16/fp32) и ``device_available=False`` →
  :class:`net.kernels.moe_dispatch.MoeDispatchKernelError` — на CPU fused-режим
  падает до какого-либо исполнения, без тихого отката;
* **(д) lower_check-адаптер** ``net/tests/moe_dispatch_lower.py`` (форма
  ``moe_experts_bwd_lower.py``) и lowering живого случая для sm_121 на этом
  CPU-узле (патч arch/Mosaic SMEM, как ``test_54_moe_bwd_fused.py``).

Реальный GPU-паритет, время и детерминизм исполненного кернела — за
архитектором (стенд): здесь пинуется раскладка/адресация и CPU-семантика, не
``mma``.

Окружение: CPU-интерпретатор, ``NET_JAX_BACKEND=cpu``/``JAX_PLATFORMS=cpu``.
"""

from __future__ import annotations

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

from net import compute_dtype, moe
from net.kernels import moe_dispatch, moe_experts_bwd

#: Ревизия до дельты (``baseline_commit`` handoff-пакета): off обязан
#: воспроизводить её побитово, включая jaxpr.  ``git archive`` + skip, если
#: ревизия недоступна (форма test_54).
BASELINE_COMMIT = "53a1336"

CASE_DIR = Path(__file__).resolve().parents[2]

#: Кампанийный допуск fused-пути (ADR-054 Am.58): scale-relative ``max|d|``
#: при bf16-операндах экспертного kernel'а.
BWD_TOL = 2e-2


@pytest.fixture(autouse=True)
def _cpu_bodies(monkeypatch):
    """CPU-исполнение: KDA — ``jax``; гейты цикла-35 сняты по умолчанию."""
    monkeypatch.setenv("AXIOM_KDA_CC_IMPL", "jax")
    monkeypatch.delenv(moe.MOE_DISPATCH_ENV, raising=False)
    monkeypatch.delenv(moe.MOE_BWD_ENV, raising=False)
    monkeypatch.delenv(moe.COUNTING_MODE_ENV, raising=False)


def _small_config(**overrides):
    from conftest import small_config

    return small_config(**overrides)


def _operands(cfg, seed: int = 3, n_tokens: int = 16):
    """``(params, z, topk, sel_p)`` — маршрутизированная подзадача MoE."""
    params = moe.init_moe(jr.PRNGKey(cfg.routing_seed), cfg)
    x = jr.normal(jr.PRNGKey(seed), (n_tokens, cfg.hidden))
    z = compute_dtype.gemm(x, params.W_down)
    _, topk, sel_p, _ = moe._dispatch(params, cfg, x)
    return params, z, topk, sel_p


def _routed_loss(params, cfg, z, topk, sel_p, **kwargs):
    """Скалярная потеря экспертной части — носитель градиентов для сверки."""
    return jnp.sum(moe._routed_experts_grouped(
        params, cfg, z, topk, sel_p, **kwargs))


def _grads(params, cfg, z, topk, sel_p, **kwargs):
    """``(dz, dwg, dwu, dwd, dsel_p)`` экспертной части."""

    def f(zz, wg, wu, wd, sp):
        pp = params._replace(expert_g=wg, expert_u=wu, expert_d=wd)
        return _routed_loss(pp, cfg, zz, topk, sp, **kwargs)

    return jax.grad(f, argnums=(0, 1, 2, 3, 4))(
        z, params.expert_g, params.expert_u, params.expert_d, sel_p)


def _scale_relative(on, off) -> tuple[float, float, float]:
    """``(max|d|, scale, max|d|/scale)``; scale — максимум модуля обоих."""
    delta = float(jnp.max(jnp.abs(on - off)))
    scale = max(float(jnp.max(jnp.abs(on))), float(jnp.max(jnp.abs(off))))
    return delta, scale, (delta / scale if scale else 0.0)


def _tree_max_rel(on, off) -> float:
    worst = 0.0
    for a, b in zip(on, off):
        _, _, rel = _scale_relative(a, b)
        worst = max(worst, rel)
    return worst


def _install_reference_seam(monkeypatch):
    """Шов fused-кернелов: jnp-эквиваленты вместо Mosaic (CPU-паритет)."""
    monkeypatch.setattr(moe_dispatch, "kernel", moe_dispatch.reference_dispatch)
    monkeypatch.setattr(moe_dispatch, "combine", moe_dispatch.reference_combine)
    monkeypatch.setattr(moe_dispatch, "scatter", moe_dispatch.reference_scatter)


# --------------------------------------------------------------------------- #
# (а) контракт гейта
# --------------------------------------------------------------------------- #


def test_gate_default_and_normalisation(monkeypatch):
    """Пусто/не задано = ``jax``; регистр и окружающие пробелы нормализуются."""
    monkeypatch.delenv(moe.MOE_DISPATCH_ENV, raising=False)
    assert moe.moe_dispatch_mode() == moe.MOE_DISPATCH_JAX

    cases = {
        "": "jax",
        "   ": "jax",
        "jax": "jax",
        "Jax": "jax",
        " JAX ": "jax",
        "fused": "fused",
        "Fused": "fused",
        " FUSED ": "fused",
        # легаси-значения прежнего исполнителя: fused-раскладка их не подменяет
        "grouped": "jax",
        "dense": "jax",
    }
    for raw, expected in cases.items():
        monkeypatch.setenv(moe.MOE_DISPATCH_ENV, raw)
        assert moe.moe_dispatch_mode() == expected, raw


@pytest.mark.parametrize("raw", ["fusion", "fuzed", "on", "1", "0", "auto",
                                 "jax2", "fused2"])
def test_gate_unknown_fails_closed(monkeypatch, raw):
    """Неизвестное значение — :class:`MoeDispatchError` (тихий откат запрещён)."""
    monkeypatch.setenv(moe.MOE_DISPATCH_ENV, raw)
    with pytest.raises(moe.MoeDispatchError) as excinfo:
        moe.moe_dispatch_mode()
    assert moe.MOE_DISPATCH_ENV in str(excinfo.value)
    assert repr(raw.strip().lower()) in str(excinfo.value)


def test_unknown_gate_raises_before_any_trace(monkeypatch):
    """Ошибка гейта — до входа в группировку и до jax-трейса."""

    def _reached(*_args, **_kwargs):
        raise AssertionError("grouping reached after an unknown gate value")

    monkeypatch.setattr(moe, "_grouped_layout", _reached)
    monkeypatch.setenv(moe.MOE_DISPATCH_ENV, "fuzed")
    with pytest.raises(moe.MoeDispatchError):
        moe._routed_experts_grouped(None, None, None, None, None)


def test_explicit_argument_beats_env(monkeypatch):
    """Явный аргумент сильнее env, включая env с неизвестным значением."""
    monkeypatch.setenv(moe.MOE_DISPATCH_ENV, "fused")
    assert moe.moe_dispatch_mode("jax") == "jax"

    monkeypatch.setenv(moe.MOE_DISPATCH_ENV, "jax")
    assert moe.moe_dispatch_mode("fused") == "fused"

    monkeypatch.setenv(moe.MOE_DISPATCH_ENV, "typo")
    assert moe.moe_dispatch_mode(" FUSED ") == "fused"


def test_explicit_argument_selects_the_path(monkeypatch):
    """``dispatch`` управляет веткой: fused — Mosaic-фабрика, jax — прежний путь."""
    cfg = _small_config()
    params, z, topk, sel_p = _operands(cfg, n_tokens=8)
    calls: list[tuple] = []
    real_factory = moe._grouped_dispatch_fused_impl

    def _spy(*args):
        calls.append(args)
        return real_factory(*args)

    monkeypatch.setattr(moe, "_grouped_dispatch_fused_impl", _spy)
    _install_reference_seam(monkeypatch)

    monkeypatch.setenv(moe.MOE_DISPATCH_ENV, "jax")
    moe._routed_experts_grouped(params, cfg, z, topk, sel_p, dispatch="fused")
    assert len(calls) == 1, "явный dispatch=fused обязан построить fused-фабрику"

    calls.clear()
    monkeypatch.setenv(moe.MOE_DISPATCH_ENV, "fused")
    moe._routed_experts_grouped(params, cfg, z, topk, sel_p, dispatch="jax")
    assert calls == [], "явный dispatch=jax не должен трогать fused-фабрику"


def test_gate_is_read_on_trace(monkeypatch):
    """Значение гейта читается на трейсе, а не при импорте: смена env видна."""
    cfg = _small_config()
    params, z, topk, sel_p = _operands(cfg, n_tokens=8)
    _install_reference_seam(monkeypatch)

    monkeypatch.setenv(moe.MOE_DISPATCH_ENV, "jax")
    off = _routed_loss(params, cfg, z, topk, sel_p)
    monkeypatch.setenv(moe.MOE_DISPATCH_ENV, "fused")
    on = _routed_loss(params, cfg, z, topk, sel_p)
    assert jnp.array_equal(off, on)


def test_legacy_dispatch_mode_still_rolls_back(monkeypatch):
    """Прежний флаг ``grouped``/``dense`` не сломан fused-гейтом."""
    monkeypatch.delenv(moe.MOE_DISPATCH_ENV, raising=False)
    assert moe.dispatch_mode() == "grouped"
    monkeypatch.setenv(moe.MOE_DISPATCH_ENV, "dense")
    assert moe.dispatch_mode() == "dense"
    monkeypatch.setenv(moe.MOE_DISPATCH_ENV, "fused")
    assert moe.dispatch_mode() == "grouped"
    with pytest.raises(ValueError):
        monkeypatch.setenv(moe.MOE_DISPATCH_ENV, "banana")
        moe.dispatch_mode()


# --------------------------------------------------------------------------- #
# (б) off — побитово ревизия до дельты (блокирующий критерий)
# --------------------------------------------------------------------------- #

#: Проба для ревизии до дельты: значение, градиенты (z, три веса, sel_p) и jaxpr.
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
    vocab_size=64, hidden=32, num_heads=4, head_dim=8,
    moe_latent_dim=8, moe_num_routed=4, moe_top_k=2, moe_num_shared=1,
    moe_expert_intermediate=16, moe_shared_intermediate=32,
    siti_beta_gate=4.0, siti_beta_up=25.0, routing_seed=0,
)
params = moe.init_moe(jax.random.PRNGKey(0), cfg)
x = jax.random.normal(jax.random.PRNGKey(1), (8, cfg.hidden))
z = jnp.dot(x, params.W_down)
_, topk, sel_p, _ = moe._dispatch(params, cfg, x)

def f(zz, wg, wu, wd, sp):
    pp = params._replace(expert_g=wg, expert_u=wu, expert_d=wd)
    return jnp.sum(moe._routed_experts_grouped(pp, cfg, zz, topk, sp))

value = f(z, params.expert_g, params.expert_u, params.expert_d, sel_p)
gz, gg, gu, gd, gs = jax.grad(f, argnums=(0, 1, 2, 3, 4))(
    z, params.expert_g, params.expert_u, params.expert_d, sel_p)
out = {"v": digest(value), "gz": digest(gz), "gg": digest(gg),
       "gu": digest(gu), "gd": digest(gd), "gs": digest(gs),
       "jaxpr": str(jax.make_jaxpr(jax.grad(f, argnums=(0, 1, 2, 3, 4)))(
           z, params.expert_g, params.expert_u, params.expert_d, sel_p))}
print(json.dumps(out, sort_keys=True))
'''


def _extract_baseline(tmp_path: Path) -> Path:
    """``net/`` базовой ревизии в tmp; проба запускается оттуда же."""
    root = tmp_path / "baseline"
    root.mkdir()
    try:
        archive = subprocess.run(
            ["git", "-C", str(CASE_DIR), "archive", BASELINE_COMMIT, "net/"],
            check=True,
            capture_output=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        pytest.skip(f"baseline {BASELINE_COMMIT} недоступен: {exc}")
    with tarfile.open(fileobj=io.BytesIO(archive.stdout)) as bundle:
        bundle.extractall(root)
    return root


def _run_probe(root: Path, script: Path, *, extra_env: dict | None = None):
    """Запуск пробы в отдельном процессе (``PYTHONPATH`` — корень ревизии)."""
    env = {
        key: value
        for key, value in os.environ.items()
        if key not in {moe.MOE_DISPATCH_ENV, moe.MOE_BWD_ENV,
                       "AXIOM_COMPUTE_DTYPE", "PYTHONPATH"}
    }
    env.update({
        "NET_JAX_BACKEND": "cpu",
        "JAX_PLATFORMS": "cpu",
        "PYTHONPATH": str(root),
    })
    env.update(extra_env or {})
    done = subprocess.run(
        [sys.executable, str(script)], cwd=root, env=env,
        capture_output=True, text=True,
    )
    if done.returncode != 0:  # pragma: no cover — диагностика прогона
        raise AssertionError(f"проба упала ({root}):\n{done.stderr[-2000:]}")
    lines = [line for line in done.stdout.strip().splitlines() if line.strip()]
    return json.loads(lines[-1])


def test_off_path_is_bitwise_the_baseline(tmp_path):
    """``jax``-ветка — побитово ревизия до дельты: числа, градиенты и jaxpr.

    Сверяются оба режима ``AXIOM_MOE_COUNTING`` и явный ``AXIOM_MOE_DISPATCH=jax``:
    дельта касается только выбора раскладки, поэтому ни одна из прежних ног не
    имеет права сдвинуться.
    """
    baseline = _extract_baseline(tmp_path)
    script = tmp_path / "probe.py"
    script.write_text(_PROBE, encoding="utf-8")

    for extra in ({}, {moe.MOE_DISPATCH_ENV: "jax"},
                  {moe.COUNTING_MODE_ENV: "1"},
                  {moe.MOE_DISPATCH_ENV: "jax", moe.COUNTING_MODE_ENV: "1"}):
        base_out = _run_probe(baseline, script, extra_env=extra)
        current_out = _run_probe(CASE_DIR, script, extra_env=extra)
        assert current_out == base_out, (
            "off-путь разошёлся с ревизией до дельты "
            f"(env={extra})")


# --------------------------------------------------------------------------- #
# (в) on: канонический порядок, forward побитово, градиенты в допуске
# --------------------------------------------------------------------------- #


def _off_layout(z, topk):
    """Прежняя раскладка forward'а: stable argsort, порядок появления."""
    n_tokens, k = topk.shape
    expert_of = topk.reshape(-1)
    token_of = jnp.repeat(jnp.arange(n_tokens, dtype=expert_of.dtype), k)
    group_sizes = jnp.bincount(expert_of, length=int(topk.max()) + 1)
    order = jnp.argsort(expert_of, stable=True)
    return z[token_of[order]], group_sizes.astype(jnp.int32), order


def test_reference_dispatch_is_the_canonical_layout():
    """``z_grouped``/``group_sizes``/``assign_rows``/``pair_of_row`` — канон.

    Порядок групп — по эксперту, внутри эксперта — по порядку появления пары
    (стабильный argsort).  ``assign_rows`` — обратная перестановка,
    ``pair_of_row`` — прямая; ``row_of_token`` — та же позиция в форме
    ``(tokens, k)``.
    """
    z, topk, sel_p = moe_dispatch.make_inputs(
        jr.PRNGKey(0), tokens=128, k=2, latent=768, n_exp=12)
    zg, sizes, assign, pair = moe_dispatch.reference_dispatch(
        z, topk, sel_p, n_exp=12)
    z_off, sizes_off, order = _off_layout(z, topk)

    assert jnp.array_equal(zg, z_off)
    assert jnp.array_equal(sizes, sizes_off)
    assert int(jnp.sum(sizes)) == z.shape[0] * topk.shape[1]
    assert jnp.array_equal(pair, order.astype(jnp.int32))
    assert jnp.array_equal(assign[order], jnp.arange(assign.shape[0]))
    assert jnp.array_equal(pair[assign], jnp.arange(assign.shape[0]))
    row_of_token = assign.reshape(topk.shape)
    assert row_of_token.shape == topk.shape
    # Внутри эксперта строки идут в порядке появления пары.
    expert_of = topk.reshape(-1)
    for e in range(12):
        rows = order[expert_of[order] == e]
        expected = jnp.nonzero(expert_of == e, size=int(sizes[e]))[0]
        assert jnp.array_equal(rows, expected), f"порядок строк эксперта {e}"


def _mirror_scan(topk, n_exp):
    """CPU-зеркало скана кернела: per-expert счётчик ``start + cnt``.

    Повторяет алгоритм тела ``_build_kernel`` (блоки ``bm`` не влияют на
    результат): программа эксперта идёт по всем парам, свои уплотняет в
    ``[start, start+size)``, чужие — в dump-слот ``rows + e``.  Возвращает
    ровно те величины, которые кернел пишет в ``group_sizes``/``assign_rows``/
    ``pair_of_row``.
    """
    n_tokens, k = (int(v) for v in topk.shape)
    rows = n_tokens * k
    expert_of = [int(v) for v in topk.reshape(-1)]
    sizes = [0] * int(n_exp)
    for e in expert_of:
        sizes[e] += 1
    assign = [-1] * rows
    pair = [-1] * rows
    for e in range(int(n_exp)):
        start = sum(sizes[:e])
        cnt = 0
        for i, ex in enumerate(expert_of):
            if ex == e:
                assign[i] = start + cnt
                pair[start + cnt] = i
                cnt += 1
        assert cnt == sizes[e], f"эксперт {e}: счётчик {cnt} != size {sizes[e]}"
    return (jnp.asarray(sizes, jnp.int32), jnp.asarray(assign, jnp.int32),
            jnp.asarray(pair, jnp.int32))


@pytest.mark.parametrize("layout", ["empty", "tail", "interleaved"])
def test_kernel_scan_mirror_matches_the_canonical_layout(layout):
    """Адресация кернела (скан + префикс) — тот же канон, что reference.

    Пустая группа, неполный хвост и интерливинг: ``group_sizes``,
    ``assign_rows`` и ``pair_of_row`` совпадают с jnp-эквивалентом; живые
    позиции не пересекаются с dump-слотами ``rows + e``.
    """
    n_tokens, k, n_exp = 16, 2, 6
    rows = n_tokens * k
    if layout == "empty":
        topk = jnp.tile(jnp.arange(k, dtype=jnp.int32), (n_tokens, 1))
    elif layout == "tail":
        topk = jnp.zeros((n_tokens, k), jnp.int32)
        topk = topk.at[:9, 1].set(1)  # 23 пары 0 / 9 пар 1, хвост неполный
    else:
        topk = (k * jnp.arange(n_tokens)[:, None]
                + jnp.arange(k)[None, :]) % n_exp
    z = jnp.zeros((n_tokens, 768), jnp.bfloat16)
    sed = jnp.full((n_tokens, k), 0.5, jnp.float32)
    _, sizes_ref, assign_ref, pair_ref = moe_dispatch.reference_dispatch(
        z, topk.astype(jnp.int32), sed, n_exp=n_exp)
    sizes, assign, pair = _mirror_scan(topk, n_exp)
    assert jnp.array_equal(sizes, sizes_ref)
    assert jnp.array_equal(assign, assign_ref)
    assert jnp.array_equal(pair, pair_ref)
    assert int(jnp.max(assign)) < rows and int(jnp.max(pair)) < rows
    assert int(jnp.sum(sizes)) == rows


@pytest.mark.parametrize("seed", [3, 11])
@pytest.mark.parametrize("counting", ["0", "1"])
def test_fused_forward_is_bitwise_the_jax_path(monkeypatch, seed, counting):
    """Forward fused-ветки — дословно прежний путь: ``array_equal``."""
    cfg = _small_config()
    params, z, topk, sel_p = _operands(cfg, seed=seed, n_tokens=16)
    monkeypatch.setenv(moe.COUNTING_MODE_ENV, counting)
    off = moe._routed_experts_grouped(params, cfg, z, topk, sel_p)

    monkeypatch.setenv(moe.MOE_DISPATCH_ENV, "fused")
    _install_reference_seam(monkeypatch)
    on = moe._routed_experts_grouped(params, cfg, z, topk, sel_p)
    assert jnp.array_equal(on, off)


@pytest.mark.parametrize("seed", [3, 11])
@pytest.mark.parametrize("counting", ["0", "1"])
def test_fused_grads_match_autodiff(monkeypatch, seed, counting):
    """Градиенты z/весов/routing — scale-relative ≤ 2e-2 (bf16-операнды)."""
    cfg = _small_config()
    params, z, topk, sel_p = _operands(cfg, seed=seed, n_tokens=16)
    monkeypatch.setenv(moe.COUNTING_MODE_ENV, counting)
    off = _grads(params, cfg, z, topk, sel_p)

    monkeypatch.setenv(moe.MOE_DISPATCH_ENV, "fused")
    _install_reference_seam(monkeypatch)
    on = _grads(params, cfg, z, topk, sel_p)

    rel = _tree_max_rel(on, off)
    print(f"seed={seed} counting={counting}: max|d|/scale={rel:.4e}")
    assert rel <= BWD_TOL


def test_fused_grads_match_autodiff_with_fused_expert_bwd(monkeypatch):
    """Оба гейта fused: combine/scatter + fused-кернел экспертного backward'а.

    Экспертный кернел заменён его jnp-адъюнктом (шов), поэтому сверяется
    связка: раскладка combine/scatter не сдвигает градиенты за bf16-допуск.
    """
    cfg = _small_config()
    params, z, topk, sel_p = _operands(cfg, seed=5, n_tokens=16)
    off = _grads(params, cfg, z, topk, sel_p)

    monkeypatch.setenv(moe.MOE_DISPATCH_ENV, "fused")
    monkeypatch.setenv(moe.MOE_BWD_ENV, "fused")
    _install_reference_seam(monkeypatch)
    monkeypatch.setattr(moe_experts_bwd, "kernel", moe_experts_bwd.reference_bwd)
    on = _grads(params, cfg, z, topk, sel_p)

    rel = _tree_max_rel(on, off)
    print(f"both fused: max|d|/scale={rel:.4e}")
    assert rel <= BWD_TOL


def test_fused_path_is_deterministic(monkeypatch):
    """Два запуска fused-пути побитово совпадают (порядок сумм фиксирован)."""
    cfg = _small_config()
    params, z, topk, sel_p = _operands(cfg, n_tokens=16)
    monkeypatch.setenv(moe.MOE_DISPATCH_ENV, "fused")
    _install_reference_seam(monkeypatch)

    first = moe._routed_experts_grouped(params, cfg, z, topk, sel_p)
    second = moe._routed_experts_grouped(params, cfg, z, topk, sel_p)
    assert jnp.array_equal(first, second)

    g1 = _grads(params, cfg, z, topk, sel_p)
    g2 = _grads(params, cfg, z, topk, sel_p)
    for a, b in zip(g1, g2):
        assert jnp.array_equal(a, b)


@pytest.mark.parametrize("layout", ["random", "empty", "interleaved"])
@pytest.mark.parametrize("k", [1, 2])
def test_group_layouts_and_k(monkeypatch, layout, k):
    """Пустые/неполные группы, интерливинг маршрутов и ``k=1``/``k=2``."""
    cfg = _small_config(moe_top_k=k)
    params, z, _, _ = _operands(cfg, n_tokens=16)
    n = cfg.moe_num_routed
    rows = z.shape[0] * k
    if layout == "empty":
        # Только эксперты 0 и 1; остальные группы пусты.
        topk = jnp.tile(jnp.arange(k, dtype=jnp.int32), (z.shape[0], 1))
    elif layout == "interleaved":
        topk = (k * jnp.arange(z.shape[0])[:, None]
                + jnp.arange(k)[None, :]) % n
    else:
        topk = jr.randint(jr.PRNGKey(7), (z.shape[0], k), 0, n)
    topk = topk.astype(jnp.int32)
    sel_p = jr.uniform(jr.PRNGKey(9), (z.shape[0], k)) + 0.1
    sel_p = sel_p / jnp.sum(sel_p, axis=-1, keepdims=True)
    if layout == "empty":
        sizes = jnp.bincount(topk.reshape(-1), length=n)
        assert int(jnp.sum(sizes[2:])) == 0, sizes
    elif layout == "interleaved":
        sizes = jnp.bincount(topk.reshape(-1), length=n)
        assert int(sizes.min()) > 0

    off = _grads(params, cfg, z, topk, sel_p)
    off_fwd = moe._routed_experts_grouped(params, cfg, z, topk, sel_p)
    monkeypatch.setenv(moe.MOE_DISPATCH_ENV, "fused")
    _install_reference_seam(monkeypatch)
    on_fwd = moe._routed_experts_grouped(params, cfg, z, topk, sel_p)
    on = _grads(params, cfg, z, topk, sel_p)

    assert jnp.array_equal(on_fwd, off_fwd), "forward/layout сдвинулись"
    rel = _tree_max_rel(on, off)
    print(f"{layout} k={k} rows={rows}: max|d|/scale={rel:.4e}")
    assert rel <= BWD_TOL


def test_fused_apply_is_bitwise_with_the_reference_seam(monkeypatch):
    """Полный ``moe.apply``: fused-раскладка не меняет forward побитово."""
    cfg = _small_config()
    params = moe.init_moe(jr.PRNGKey(2), cfg)
    x = jr.normal(jr.PRNGKey(4), (8, cfg.hidden))
    off = moe.apply(params, cfg, x)

    monkeypatch.setenv(moe.MOE_DISPATCH_ENV, "fused")
    _install_reference_seam(monkeypatch)
    on = moe.apply(params, cfg, x)
    assert jnp.array_equal(on, off)


# --------------------------------------------------------------------------- #
# (г) типизированные отказы
# --------------------------------------------------------------------------- #


def test_valid_config_sieve():
    """Сито: кампанийная геометрия — да; некратные формы/типы — нет."""
    params = {"bm": 16, "bc": 128}
    assert moe_dispatch.valid_config(moe_dispatch.CASES[0], params)
    assert moe_dispatch.valid_config(moe_dispatch.CASES[1], params)
    # CASES[2]: rows = 300 не делится на bm=16 — fail-closed.
    assert not moe_dispatch.valid_config(moe_dispatch.CASES[2], params)
    # fp32 — движение строк (бит-в-бит forward), допустим.
    assert moe_dispatch.valid_config(
        dict(tokens=32, k=2, latent=768, n_exp=12, dtype="float32"), params)
    # latent не кратен bc / bc не кратен 128 / int8.
    assert not moe_dispatch.valid_config(
        dict(tokens=32, k=2, latent=96, n_exp=12, dtype="bfloat16"), params)
    assert not moe_dispatch.valid_config(
        dict(tokens=32, k=2, latent=768, n_exp=12, dtype="bfloat16"),
        {"bm": 16, "bc": 100})
    assert not moe_dispatch.valid_config(
        dict(tokens=32, k=2, latent=768, n_exp=12, dtype="int8"), params)
    # rows % bm != 0 — типизированный отказ, а не кламп.
    assert not moe_dispatch.valid_config(
        dict(tokens=10, k=3, latent=768, n_exp=12, dtype="bfloat16"), params)


def _kernel_inputs(tokens: int = 128, k: int = 2, latent: int = 768,
                   n_exp: int = 12, dtype: str = "bfloat16"):
    z, topk, sel_p = moe_dispatch.make_inputs(
        jr.PRNGKey(0), tokens=tokens, k=k, latent=latent, n_exp=n_exp,
        dtype=dtype)
    return z, topk, sel_p


def test_kernel_refuses_a_broken_config_typed(monkeypatch):
    """Форма вне сита — :class:`MoeDispatchKernelError` (даже при «есть GPU»)."""
    monkeypatch.setattr(moe_dispatch, "device_available", lambda: True)
    z, topk, sel_p = _kernel_inputs(tokens=100, k=3, latent=768)
    with pytest.raises(moe_dispatch.MoeDispatchKernelError) as excinfo:
        moe_dispatch.kernel(z, topk, sel_p, n_exp=12)
    assert "valid_config" in str(excinfo.value)

    z, topk, sel_p = _kernel_inputs(tokens=32, k=2, latent=96)
    with pytest.raises(moe_dispatch.MoeDispatchKernelError) as excinfo:
        moe_dispatch.kernel(z, topk, sel_p, n_exp=12)
    assert "valid_config" in str(excinfo.value)


def test_kernel_refuses_without_a_device_typed(monkeypatch):
    """``device_available=False`` — :class:`MoeDispatchKernelError` без отката."""
    monkeypatch.setattr(moe_dispatch, "device_available", lambda: False)
    z, topk, sel_p = _kernel_inputs()
    with pytest.raises(moe_dispatch.MoeDispatchKernelError) as excinfo:
        moe_dispatch.kernel(z, topk, sel_p, n_exp=12)
    assert "GPU" in str(excinfo.value)


def test_combine_and_scatter_refuse_on_cpu_typed(monkeypatch):
    """Combine/scatter на CPU — типизированный отказ, не тихий jax-путь."""
    monkeypatch.setattr(moe_dispatch, "device_available", lambda: False)
    z, topk, sel_p = _kernel_inputs()
    zg, gs, assign, pair = moe_dispatch.reference_dispatch(
        z, topk, sel_p, n_exp=12)
    du = jnp.zeros_like(z)
    with pytest.raises(moe_dispatch.MoeDispatchKernelError) as excinfo:
        moe_dispatch.combine(du, zg, pair, sel_p, gs, n_exp=12)
    assert "GPU" in str(excinfo.value)
    with pytest.raises(moe_dispatch.MoeDispatchKernelError) as excinfo:
        moe_dispatch.scatter(zg.astype(jnp.float32), pair, gs,
                             n_tokens=z.shape[0], k=2, n_exp=12)
    assert "GPU" in str(excinfo.value)


def test_fused_mode_refuses_on_cpu_typed(monkeypatch):
    """Fused-режим на CPU падает типизированно, а не молча берёт autodiff."""
    cfg = _small_config(moe_latent_dim=768, moe_expert_intermediate=384,
                        moe_num_routed=12)
    params = moe.init_moe(jr.PRNGKey(0), cfg)
    z = jr.normal(jr.PRNGKey(1), (16, 768))
    topk = jnp.zeros((16, 2), jnp.int32)
    sel_p = jnp.full((16, 2), 0.5)
    monkeypatch.setenv(moe.MOE_DISPATCH_ENV, "fused")
    with pytest.raises(moe_dispatch.MoeDispatchKernelError):
        jax.grad(lambda zz: _routed_loss(params, cfg, zz, topk, sel_p))(z)


# --------------------------------------------------------------------------- #
# (д) lower_check-адаптер и lowering на этом узле
# --------------------------------------------------------------------------- #


def test_lower_adapter_contract():
    """Адаптер ``moe_dispatch_lower`` повторяет контракт боевого модуля."""
    from net.tests import moe_dispatch_lower as adapter

    assert adapter.CASES is moe_dispatch.CASES
    assert adapter.TUNE_SPACE == moe_dispatch.TUNE_SPACE
    assert callable(adapter.make_inputs)
    assert callable(adapter.valid_config)
    assert callable(adapter.kernel)
    assert callable(adapter.reference)


def _lower(builder, *shapes):
    """Lowering построителя для sm_121 ровно как у скилла lower_check."""
    from jax.experimental.mosaic import gpu as mosaic_mod
    from jax.experimental.mosaic.gpu import utils as mosaic_utils

    mosaic_utils._infer_arch = lambda: (12, 1)
    with mosaic_mod.core.artificial_shared_memory_limit(
            moe_dispatch.SMEM_LIMIT_BYTES):
        fn = jax.jit(lambda *a: builder(*a))
        return fn.trace(*shapes).lower(lowering_platforms=("cuda",)).as_text()


def test_dispatch_kernel_lowers_for_sm121():
    """Живой (дешёвый) случай dispatch собирается в Mosaic-модуль."""
    pytest.importorskip("jax.experimental.pallas.mosaic_gpu")
    if any(d.platform == "gpu" for d in jax.devices()):  # pragma: no cover
        pytest.skip("GPU присутствует: настоящая проверка — check_kernel.py")
    case = moe_dispatch.CASES[1]
    tokens, k = case["tokens"], case["k"]
    latent, n_exp = case["latent"], case["n_exp"]
    rows = tokens * k
    gs = jnp.full((n_exp,), rows // n_exp, jnp.int32)
    offs = moe_dispatch.group_offsets(gs)
    build = moe_dispatch._build_kernel(
        tokens, k, latent, n_exp, 16, 128, case["dtype"])
    text = _lower(build, jnp.zeros((tokens, latent), jnp.bfloat16),
                  jnp.zeros((rows,), jnp.int32), gs, offs)
    assert "mosaic_gpu" in text, "lowering не дал Mosaic-кернел"


def test_combine_and_scatter_kernels_lower_for_sm121():
    """Combine и scatter живого случая тоже собираются в Mosaic-модуль."""
    pytest.importorskip("jax.experimental.pallas.mosaic_gpu")
    if any(d.platform == "gpu" for d in jax.devices()):  # pragma: no cover
        pytest.skip("GPU присутствует: настоящая проверка — check_kernel.py")
    case = moe_dispatch.CASES[1]
    tokens, k = case["tokens"], case["k"]
    latent, n_exp = case["latent"], case["n_exp"]
    rows = tokens * k
    gs = jnp.full((n_exp,), rows // n_exp, jnp.int32)
    offs = moe_dispatch.group_offsets(gs)
    dt = jnp.dtype(case["dtype"])
    pair = jnp.zeros((rows,), jnp.int32)

    build = moe_dispatch._build_combine_kernel(
        tokens, k, latent, n_exp, 16, 128, case["dtype"])
    text = _lower(build, jnp.zeros((tokens, latent), dt),
                  jnp.zeros((rows, latent), dt), pair,
                  jnp.zeros((tokens, k), jnp.float32), gs, offs)
    assert "mosaic_gpu" in text

    build = moe_dispatch._build_scatter_kernel(
        tokens, k, latent, n_exp, 16, 128, case["dtype"])
    text = _lower(build, jnp.zeros((rows, latent), jnp.float32), pair, gs, offs)
    assert "mosaic_gpu" in text


def test_kernel_out_types_carry_dump_slots():
    """Объявленные выходы: ``z_grouped``/книги — ``rows + n_exp`` и ``n_exp``."""
    case = moe_dispatch.CASES[1]
    tokens, k = case["tokens"], case["k"]
    latent, n_exp = case["latent"], case["n_exp"]
    rows = tokens * k
    shapes = jax.eval_shape(
        lambda: moe_dispatch.make_inputs(jax.random.key(0), **case))
    build = moe_dispatch._build_kernel(
        tokens, k, latent, n_exp, 16, 128, case["dtype"])
    gs = jnp.full((n_exp,), rows // n_exp, jnp.int32)
    out = jax.eval_shape(lambda *a: build(*a), shapes[0],
                         jax.ShapeDtypeStruct((rows,), jnp.int32), gs,
                         moe_dispatch.group_offsets(gs))
    assert [tuple(o.shape) for o in out] == [
        (rows + n_exp, latent),   # z_grouped + dump-слоты
        (n_exp,),                 # group_sizes
        (rows + n_exp,),          # assign_rows + dump-слоты
        (rows + n_exp,),          # pair_of_row + dump-слоты
    ]


def test_invalid_case_is_not_lowered():
    """Обещание сита: случай с ``rows % bm != 0`` не доходит до lowering."""
    case = moe_dispatch.CASES[2]
    assert not moe_dispatch.valid_config(case, {"bm": 16, "bc": 128})
    assert (case["tokens"] * case["k"]) % 16
