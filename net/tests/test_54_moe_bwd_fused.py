"""Цикл-30 (ADR-054 Am.52): fused backward экспертного блока MoE — гейт ``AXIOM_MOE_BWD``.

Пофазовый профиль цикла-29 отнёс ``sec_bwd_moe`` = 10.235 с (49 % полного
backward, 38 % шага) при ~0.06 TFLOPS: autodiff трёх ``ragged_dot`` и
``siti_glu`` — это десятки мелких кернелов на слой.  Цикл-30 добавляет
``jax.custom_vjp`` вокруг экспертной части: forward — дословно прежний путь
(grouping + три ``ragged_dot`` + ``siti_glu``; числа те же), backward — один
fused-кернел ``net/kernels/moe_experts_bwd.py`` плюс XLA для группировки/
сборки строк и градиента маршрутизации.

Что пиннуется здесь (CPU, jax 0.11.2):

* **(а) контракт гейта**: пусто/не задано = ``"jax"``; ``"fused"``; регистр и
  пробелы нормализуются; неизвестное — :class:`net.moe.MoeBwdError` до
  трейса; явный аргумент сильнее env; значение читается на трейсе;
* **(б) off — побитово ревизия до дельты** (``439709a``, ``baseline_commit``
  пакета): значение и градиенты ``_routed_experts_grouped`` и jaxpr графа
  совпадают с извлечённым ``net/`` baseline — оба режима
  ``AXIOM_MOE_COUNTING``; блокирующий критерий цикла;
* **(в) on — forward побитово равен off**, градиенты — scale-relative
  ``max|d|/scale ≤ 2e-2`` (кампанийный допуск bf16) на нескольких сидах и
  формах, включая хвостовую неполную группу, пустой эксперт
  (``group_size=0``), интерливинг маршрутов (порядок групп/строк) и оба
  режима группировки; фактический ``max|d|/scale`` печатается;
* **(г) отказ типизирован**: ``valid_config`` вне живых случаев и
  ``device_available=False`` → :class:`net.kernels.moe_experts_bwd.MoeBwdKernelError`
  (на CPU fused-режим падает до какого-либо исполнения, без тихого отката);
* **(д) lower_check-адаптер** ``net/tests/moe_experts_bwd_lower.py`` (форма
  ``kda_persist_bwd_lower.py``) и lowering живого случая для sm_121 на этом
  CPU-узле (патч arch/Mosaic SMEM, как ``test_kda_cc_fused.py``).

Окружение: CPU-интерпретатор, ``NET_JAX_BACKEND=cpu``/``JAX_PLATFORMS=cpu``;
KDA-тело — ``jax``; fused-кернел не исполняется (только lowering), численный
пин — jnp-адъюнкт ``reference_bwd``.
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

from net import compute_dtype, mlp, moe
from net.kernels import moe_experts_bwd

#: Ревизия до дельты (``baseline_commit`` handoff-пакета): off обязан
#: воспроизводить её побитово, включая jaxpr.  ``git archive`` + skip, если
#: ревизия недоступна (форма test_47).
BASELINE_COMMIT = "439709a"

CASE_DIR = Path(__file__).resolve().parents[2]

#: Кампанийный допуск fused-backward (ADR-054 Am.52): scale-relative ``max|d|``
#: при bf16-операндах кернела.
BWD_TOL = 2e-2


# --------------------------------------------------------------------------- #
# Общие помощники
# --------------------------------------------------------------------------- #


@pytest.fixture(autouse=True)
def _cpu_bodies(monkeypatch):
    """CPU-исполнение: KDA — ``jax``; гейт цикла-30 снят по умолчанию."""
    monkeypatch.setenv("AXIOM_KDA_CC_IMPL", "jax")
    monkeypatch.delenv(moe.MOE_BWD_ENV, raising=False)
    monkeypatch.delenv(moe.COUNTING_MODE_ENV, raising=False)


def _small_config():
    from conftest import small_config

    return small_config()


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
    """``(dz, dwg, dwu, dwd)`` экспертной части."""

    def f(zz, wg, wu, wd):
        pp = params._replace(expert_g=wg, expert_u=wu, expert_d=wd)
        return _routed_loss(pp, cfg, zz, topk, sel_p, **kwargs)

    return jax.grad(f, argnums=(0, 1, 2, 3))(
        z, params.expert_g, params.expert_u, params.expert_d)


def _scale_relative(on, off) -> tuple[float, float, float]:
    """``(max|d|, scale, max|d|/scale)``; scale — максимум модуля обоих."""
    delta = float(jnp.max(jnp.abs(on - off)))
    scale = max(float(jnp.max(jnp.abs(on))), float(jnp.max(jnp.abs(off))))
    return delta, scale, (delta / scale if scale else 0.0)


def _install_reference_seam(monkeypatch):
    """Шов fused-кернела: jnp-адъюнкт вместо Mosaic-кернела (CPU-паритет)."""
    monkeypatch.setattr(moe_experts_bwd, "kernel", moe_experts_bwd.reference_bwd)


def _tree_max_rel(on, off) -> float:
    worst = 0.0
    for a, b in zip(on, off):
        _, _, rel = _scale_relative(a, b)
        worst = max(worst, rel)
    return worst


# --------------------------------------------------------------------------- #
# (а) контракт гейта
# --------------------------------------------------------------------------- #


def test_gate_default_and_normalisation(monkeypatch):
    """Пусто/не задано = ``jax``; регистр и окружающие пробелы нормализуются."""
    monkeypatch.delenv(moe.MOE_BWD_ENV, raising=False)
    assert moe.moe_bwd_mode() == moe.MOE_BWD_JAX

    cases = {
        "": "jax",
        "   ": "jax",
        "jax": "jax",
        "Jax": "jax",
        " JAX ": "jax",
        "fused": "fused",
        "Fused": "fused",
        " FUSED ": "fused",
    }
    for raw, expected in cases.items():
        monkeypatch.setenv(moe.MOE_BWD_ENV, raw)
        assert moe.moe_bwd_mode() == expected, raw


@pytest.mark.parametrize("raw", ["fusion", "fuzed", "on", "1", "0", "auto", "jax2"])
def test_gate_unknown_fails_closed(monkeypatch, raw):
    """Неизвестное значение — :class:`MoeBwdError` (тихий откат запрещён)."""
    monkeypatch.setenv(moe.MOE_BWD_ENV, raw)
    with pytest.raises(moe.MoeBwdError) as excinfo:
        moe.moe_bwd_mode()
    assert moe.MOE_BWD_ENV in str(excinfo.value)
    assert repr(raw.strip().lower()) in str(excinfo.value)


def test_unknown_gate_raises_before_any_trace(monkeypatch):
    """Ошибка гейта — до входа в группировку и до jax-трейса."""

    def _reached(*_args, **_kwargs):
        raise AssertionError("grouping reached after an unknown gate value")

    monkeypatch.setattr(moe, "_grouped_layout", _reached)
    monkeypatch.setenv(moe.MOE_BWD_ENV, "fuzed")
    with pytest.raises(moe.MoeBwdError):
        moe._routed_experts_grouped(None, None, None, None, None)


def test_explicit_argument_beats_env(monkeypatch):
    """Явный аргумент сильнее env, включая env с неизвестным значением."""
    monkeypatch.setenv(moe.MOE_BWD_ENV, "fused")
    assert moe.moe_bwd_mode("jax") == "jax"

    monkeypatch.setenv(moe.MOE_BWD_ENV, "jax")
    assert moe.moe_bwd_mode("fused") == "fused"

    monkeypatch.setenv(moe.MOE_BWD_ENV, "typo")
    assert moe.moe_bwd_mode(" FUSED ") == "fused"


def test_explicit_argument_selects_the_path(monkeypatch):
    """``bwd`` управляет веткой: fused — фабрика custom_vjp, jax — прежний путь."""
    cfg = _small_config()
    params, z, topk, sel_p = _operands(cfg, n_tokens=8)
    calls: list[tuple] = []
    real_factory = moe._grouped_fused_impl

    def _spy(*args):
        calls.append(args)
        return real_factory(*args)

    monkeypatch.setattr(moe, "_grouped_fused_impl", _spy)
    monkeypatch.setattr(moe_experts_bwd, "kernel", moe_experts_bwd.reference_bwd)

    monkeypatch.setenv(moe.MOE_BWD_ENV, "jax")
    moe._routed_experts_grouped(params, cfg, z, topk, sel_p, bwd="fused")
    assert len(calls) == 1, "явный bwd=fused обязан построить fused-фабрику"

    calls.clear()
    monkeypatch.setenv(moe.MOE_BWD_ENV, "fused")
    moe._routed_experts_grouped(params, cfg, z, topk, sel_p, bwd="jax")
    assert calls == [], "явный bwd=jax не должен трогать fused-фабрику"


def test_gate_is_read_on_trace(monkeypatch):
    """Значение гейта читается на трейсе, а не при импорте: смена env видна."""
    cfg = _small_config()
    params, z, topk, sel_p = _operands(cfg, n_tokens=8)
    monkeypatch.setattr(moe_experts_bwd, "kernel", moe_experts_bwd.reference_bwd)

    monkeypatch.setenv(moe.MOE_BWD_ENV, "jax")
    off = _routed_loss(params, cfg, z, topk, sel_p)
    monkeypatch.setenv(moe.MOE_BWD_ENV, "fused")
    on = _routed_loss(params, cfg, z, topk, sel_p)
    assert jnp.array_equal(off, on)


# --------------------------------------------------------------------------- #
# (б) off — побитово ревизия до дельты (блокирующий критерий)
# --------------------------------------------------------------------------- #

#: Проба для ревизии до дельты: значение, градиенты (z и три веса) и jaxpr.
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

def f(zz, wg, wu, wd):
    pp = params._replace(expert_g=wg, expert_u=wu, expert_d=wd)
    return jnp.sum(moe._routed_experts_grouped(pp, cfg, zz, topk, sel_p))

value = f(z, params.expert_g, params.expert_u, params.expert_d)
gz, gg, gu, gd = jax.grad(f, argnums=(0, 1, 2, 3))(
    z, params.expert_g, params.expert_u, params.expert_d)
out = {"v": digest(value), "gz": digest(gz), "gg": digest(gg),
       "gu": digest(gu), "gd": digest(gd),
       "jaxpr": str(jax.make_jaxpr(jax.grad(f, argnums=(0, 1, 2, 3)))(
           z, params.expert_g, params.expert_u, params.expert_d))}
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
        if key not in {moe.MOE_BWD_ENV, "AXIOM_COMPUTE_DTYPE", "PYTHONPATH"}
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

    Сверяются оба режима ``AXIOM_MOE_COUNTING`` (0 и 1): дельта касается
    только выбора backward, поэтому ни одна из прежних ног не имеет права
    сдвинуться.
    """
    baseline = _extract_baseline(tmp_path)
    script = tmp_path / "probe.py"
    script.write_text(_PROBE, encoding="utf-8")

    for extra in ({}, {moe.COUNTING_MODE_ENV: "1"}):
        base_out = _run_probe(baseline, script, extra_env=extra)
        current_out = _run_probe(CASE_DIR, script, extra_env=extra)
        assert current_out == base_out, (
            "off-путь разошёлся с ревизией до дельты "
            f"(AXIOM_MOE_COUNTING={extra.get(moe.COUNTING_MODE_ENV, '<unset>')})"
        )


# --------------------------------------------------------------------------- #
# (в) on: forward побитово, градиенты в кампанийном допуске
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("seed", [3, 11])
def test_fused_forward_is_bitwise_the_jax_path(monkeypatch, seed):
    """Forward fused-ветки — дословно прежний путь: ``array_equal``."""
    cfg = _small_config()
    params, z, topk, sel_p = _operands(cfg, seed=seed, n_tokens=16)
    off = moe._routed_experts_grouped(params, cfg, z, topk, sel_p)

    monkeypatch.setenv(moe.MOE_BWD_ENV, "fused")
    _install_reference_seam(monkeypatch)
    on = moe._routed_experts_grouped(params, cfg, z, topk, sel_p)
    assert jnp.array_equal(on, off)


@pytest.mark.parametrize("seed", [3, 11])
@pytest.mark.parametrize("counting", [False, True])
def test_fused_grads_match_autodiff(monkeypatch, seed, counting):
    """Градиенты z и весов — scale-relative ≤ 2e-2 (bf16-операнды кернела)."""
    cfg = _small_config()
    params, z, topk, sel_p = _operands(cfg, seed=seed, n_tokens=16)
    if counting:
        monkeypatch.setenv(moe.COUNTING_MODE_ENV, "1")
    off = _grads(params, cfg, z, topk, sel_p)

    monkeypatch.setenv(moe.MOE_BWD_ENV, "fused")
    _install_reference_seam(monkeypatch)
    on = _grads(params, cfg, z, topk, sel_p)

    rel = _tree_max_rel(on, off)
    print(f"seed={seed} counting={counting}: max|d|/scale={rel:.4e}")
    assert rel <= BWD_TOL


def test_group_order_and_rows_are_preserved(monkeypatch):
    """Интерливинг маршрутов: порядок групп/строк — входной, градиенты сходятся.

    Маршруты чередуются между экспертами (соседние назначения — разные
    эксперты), поэтому перестановка строк внутри эксперта или потеря порядка
    появления ломает равенство с autodiff, а не маскируется симметрией.
    """
    cfg = _small_config()
    params, z, _, _ = _operands(cfg, n_tokens=16)
    k = cfg.moe_top_k
    n = cfg.moe_num_routed
    n_tokens = z.shape[0]
    # topk[i, j] = (2*i + j) % n — назначения чередуются по экспертам.
    idx = (2 * jnp.arange(n_tokens)[:, None] + jnp.arange(k)[None, :]) % n
    topk = idx.astype(jnp.int32)
    w = jr.uniform(jr.PRNGKey(5), (n_tokens, k))
    sel_p = w / jnp.sum(w, axis=-1, keepdims=True)

    # Раскладка: внутри эксперта — порядок появления (стабильный argsort).
    expert_of = topk.reshape(-1)
    sizes = jnp.bincount(expert_of, length=n)
    order = jnp.argsort(expert_of, stable=True)
    for e in range(n):
        rows = order[expert_of[order] == e]
        expected = jnp.nonzero(expert_of == e, size=int(sizes[e]))[0]
        assert jnp.array_equal(rows, expected), (
            f"порядок строк эксперта {e} не совпал с порядком появления")

    off = _grads(params, cfg, z, topk, sel_p)
    monkeypatch.setenv(moe.MOE_BWD_ENV, "fused")
    _install_reference_seam(monkeypatch)
    on = _grads(params, cfg, z, topk, sel_p)
    rel = _tree_max_rel(on, off)
    print(f"interleaved: max|d|/scale={rel:.4e}")
    assert rel <= BWD_TOL


@pytest.mark.parametrize("layout", ["empty", "tail"])
def test_special_groups(monkeypatch, layout):
    """Особые случаи: пустой эксперт (``group_size=0``) и неполная группа."""
    cfg = _small_config()
    params, z, _, _ = _operands(cfg, n_tokens=16)
    k = cfg.moe_top_k
    n = cfg.moe_num_routed
    n_tokens = z.shape[0]
    if layout == "empty":
        # Только эксперты 0 и 1; эксперты 2..n-1 пусты.
        idx = jnp.stack([jnp.zeros(n_tokens, jnp.int32),
                         jnp.ones(n_tokens, jnp.int32)], axis=1)
    else:
        # Неполная последняя группа: 17 назначений эксперта 0 и 15 эксперта 1
        # при rows = 32 (не делится на n и не кратно числу экспертов).
        base = jnp.zeros((n_tokens, k), jnp.int32)
        base = base.at[:9, 1].set(1)              # 23 назначения 0 / 9 — 1
        idx = base
    topk = idx.astype(jnp.int32)
    sizes = jnp.bincount(topk.reshape(-1), length=n)
    if layout == "empty":
        assert int(jnp.sum(sizes[2:])) == 0, sizes
    else:
        assert int(sizes[0]) != int(sizes[1]), sizes
    w = jr.uniform(jr.PRNGKey(7), (n_tokens, k)) + 0.1
    sel_p = w / jnp.sum(w, axis=-1, keepdims=True)

    off = _grads(params, cfg, z, topk, sel_p)
    monkeypatch.setenv(moe.MOE_BWD_ENV, "fused")
    _install_reference_seam(monkeypatch)
    on = _grads(params, cfg, z, topk, sel_p)
    rel = _tree_max_rel(on, off)
    print(f"{layout}: sizes={sizes.tolist()} max|d|/scale={rel:.4e}")
    assert rel <= BWD_TOL
    # Пустой эксперт: его весовые градиенты — ровно нуль (а не NaN/мусор).
    if layout == "empty":
        for grad in on[1:]:
            assert jnp.all(grad[2:] == 0), "пустой эксперт получил ненулевой градиент"


def test_reference_bwd_is_the_adjoint_of_the_expert_block():
    """``reference_bwd`` — адъюнкт экспертного блока: сверка с ``jax.vjp``.

    Прямая блок-функция — ragged-путь (``ragged_dot`` + ``siti_glu``) на тех же
    (hg, hu, a), а котангент выхода задан явно.  Это пинует и формулы, и
    порядок строк: перестановка внутри группы разошлась бы с ``ragged_dot``.
    """
    cfg = _small_config()
    params, z, _, _ = _operands(cfg, n_tokens=12)
    k = cfg.moe_top_k
    n = cfg.moe_num_routed
    n_tokens = z.shape[0]
    # Пустой эксперт (2) и неровные группы: 0 -> 9 строк, 1 -> 15.
    topk = jnp.zeros((n_tokens, k), jnp.int32)
    topk = topk.at[5:, 1].set(1)
    expert_of = topk.reshape(-1)
    group_sizes = jnp.bincount(expert_of, length=n).astype(jnp.int32)
    assert int(group_sizes[2]) == 0

    from net.moe import _ragged_gemm

    z_grouped = jr.normal(jr.PRNGKey(2), (n_tokens * k, cfg.moe_latent_dim))
    de = jr.normal(jr.PRNGKey(5), (n_tokens * k, cfg.moe_latent_dim))
    bg, bu = cfg.siti_beta_gate, cfg.siti_beta_up

    def block(zz, wg, wu, wd):
        """Прямая алгебра экспертного блока — ровно путь forward'а."""
        h_g = _ragged_gemm(zz, wg, group_sizes)
        h_u = _ragged_gemm(zz, wu, group_sizes)
        aa = mlp.siti_glu((h_g, h_u), bg, bu)
        return _ragged_gemm(aa, wd, group_sizes)

    h_g = _ragged_gemm(z_grouped, params.expert_g, group_sizes)
    h_u = _ragged_gemm(z_grouped, params.expert_u, group_sizes)
    a = mlp.siti_glu((h_g, h_u), bg, bu)

    _, vjp = jax.vjp(block, z_grouped, params.expert_g, params.expert_u,
                     params.expert_d)
    dz_ref, dwg_ref, dwu_ref, dwd_ref = vjp(de)

    dwg, dwu, dwd, dz = moe_experts_bwd.reference_bwd(
        z_grouped, h_g, h_u, a, de, params.expert_g, params.expert_u,
        params.expert_d, group_sizes)
    rel = _tree_max_rel((dz, dwg, dwu, dwd), (dz_ref, dwg_ref, dwu_ref, dwd_ref))
    print(f"reference_bwd vs vjp: max|d|/scale={rel:.4e}")
    assert rel <= BWD_TOL


# --------------------------------------------------------------------------- #
# (г) типизированные отказы
# --------------------------------------------------------------------------- #


def _kernel_inputs(rows: int = 64, latent: int = 768, inter: int = 384,
                   n_exp: int = 2):
    dt = jnp.bfloat16
    z = jnp.zeros((rows, latent), dt)
    hg = jnp.zeros((rows, inter), dt)
    a = jnp.zeros((rows, inter), dt)
    de = jnp.zeros((rows, latent), dt)
    wg = jnp.zeros((n_exp, latent, inter), dt)
    wu = jnp.zeros((n_exp, latent, inter), dt)
    wd = jnp.zeros((n_exp, inter, latent), dt)
    sizes = jnp.full((n_exp,), rows // n_exp, jnp.int32)
    sizes = sizes.at[0].add(rows - int(sizes.sum()))
    return z, hg, hg, a, de, wg, wu, wd, sizes


def test_valid_config_sieve():
    """Сито: живая кампанийная геометрия — да, oversize/битые формы — нет."""
    assert moe_experts_bwd.valid_config(moe_experts_bwd.CASES[0], {"bm": 16, "bt": 128, "bk": 32})
    assert moe_experts_bwd.valid_config(moe_experts_bwd.CASES[1], {"bm": 16, "bt": 128, "bk": 32})
    oversize = dict(rows=2048, latent=1024, inter=768, n_exp=16, dtype="bfloat16")
    assert not moe_experts_bwd.valid_config(oversize, {"bm": 16, "bt": 128, "bk": 32})
    broken = dict(rows=64, latent=96, inter=384, n_exp=2, dtype="bfloat16")
    assert not moe_experts_bwd.valid_config(broken, {"bm": 16, "bt": 128, "bk": 32})
    fp32 = dict(rows=64, latent=768, inter=384, n_exp=2, dtype="float32")
    assert not moe_experts_bwd.valid_config(fp32, {"bm": 16, "bt": 128, "bk": 32})


def test_kernel_refuses_a_broken_config_typed(monkeypatch):
    """Форма вне сита — :class:`MoeBwdKernelError` (даже при «есть GPU»)."""
    monkeypatch.setattr(moe_experts_bwd, "device_available", lambda: True)
    args = _kernel_inputs(latent=96, inter=384)
    with pytest.raises(moe_experts_bwd.MoeBwdKernelError) as excinfo:
        moe_experts_bwd.kernel(*args)
    assert "valid_config" in str(excinfo.value)


def test_kernel_refuses_without_a_device_typed(monkeypatch):
    """``device_available=False`` — :class:`MoeBwdKernelError` без тихого отката."""
    monkeypatch.setattr(moe_experts_bwd, "device_available", lambda: False)
    args = _kernel_inputs()
    with pytest.raises(moe_experts_bwd.MoeBwdKernelError) as excinfo:
        moe_experts_bwd.kernel(*args)
    assert "GPU" in str(excinfo.value)


def test_fused_mode_refuses_on_cpu_typed(monkeypatch):
    """Fused-режим на CPU падает типизированно, а не молча берёт autodiff."""
    cfg = _small_config()
    params, z, topk, sel_p = _operands(cfg, n_tokens=8)
    monkeypatch.setenv(moe.MOE_BWD_ENV, "fused")
    with pytest.raises(moe_experts_bwd.MoeBwdKernelError):
        jax.grad(lambda zz: _routed_loss(params, cfg, zz, topk, sel_p))(z)


# --------------------------------------------------------------------------- #
# (д) lowering на этом узле: живые случаи собираются для sm_121
# --------------------------------------------------------------------------- #


def _lower_live_case(case):
    """Lowering ``_build_kernel`` для sm_121 ровно как у скилла lower_check."""
    from jax.experimental.mosaic import gpu as mosaic_mod
    from jax.experimental.mosaic.gpu import utils as mosaic_utils

    mosaic_utils._infer_arch = lambda: (12, 1)
    ctx = mosaic_mod.core.artificial_shared_memory_limit(
        moe_experts_bwd.SMEM_LIMIT_BYTES)
    with ctx:
        shapes = jax.eval_shape(lambda: moe_experts_bwd.make_inputs(
            jax.random.key(0), **case))
        spec = moe_experts_bwd._KernelSpec(
            case["rows"], case["latent"], case["inter"], case["n_exp"],
            16, 128, 32, case["dtype"], 4.0, 25.0)
        build = moe_experts_bwd._build_kernel(
            spec.rows, spec.latent, spec.inter, spec.n_exp, spec.bm, spec.bt,
            spec.bk, spec.dtype, spec.beta_gate, spec.beta_up)
        fn = jax.jit(lambda *a: build(*a))
        return fn.trace(*shapes).lower(lowering_platforms=("cuda",)).as_text()


def test_kernel_lowers_for_sm121():
    """Живой (дешёвый) случай собирается в Mosaic-модуль; oversize — скип."""
    pytest.importorskip("jax.experimental.pallas.mosaic_gpu")
    if any(d.platform == "gpu" for d in jax.devices()):  # pragma: no cover — стенд
        pytest.skip("GPU присутствует: настоящая проверка — check_kernel.py")
    text = _lower_live_case(moe_experts_bwd.CASES[1])
    assert "mosaic_gpu" in text, "lowering не дал Mosaic-кернел"


def test_oversize_case_is_not_lowered():
    """Обещание сита: oversize-случай не доходит до lowering."""
    for case in moe_experts_bwd.CASES:
        if not moe_experts_bwd.valid_config(case, {"bm": 16, "bt": 128, "bk": 32}):
            assert moe_experts_bwd._smem_floor_bytes(
                16, 128, 32, case["latent"], case["inter"]
            ) > moe_experts_bwd.SMEM_LIMIT_BYTES or case["latent"] % 128
