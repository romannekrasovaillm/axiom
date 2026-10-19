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
* **(е) структурная проба dz-эпилога** (цикл-31, ADR-054 Am.53): колонки
  тайлов покрывают ``[0, latent)`` ровно один раз; адресация строк
  (префиксные суммы ``group_sizes`` + маска + dump-слот) проверяется
  целочисленно на кампанийных и граничных раскладках; запись, которую
  исполняет **само тело кернела**, проверяется подменой ``_store_row`` на
  lowering — тайл ``tj`` обязан писать в колонки ``[tj·bt, (tj+1)·bt)``;
  CPU-зеркало ``mirror_dz_path`` сверяется с ``reference_bwd``, а
  историческая политика «все тайлы в колонку 0» даёт сигнатуру стенда
  (``‖Δ‖/‖·‖ = 1.08`` на пустом буфере … ``1.41`` на остаточной памяти —
  наблюдённая 1.28 внутри).  Реальный GPU-паритет по-прежнему за
  архитектором (ADR-011): здесь пинуется адресация и формулы, не ``mma``.
* **(ж) guard'ы цикла-32** (ADR-054 Am.54): пустая группа эксперта даёт
  ровно нулевые ``dwg``/``dwu``/``dwd`` без nan/inf — и на «боевом» условии
  стенда (скретч соседней группы не инициализирован: ``scratch_init="nan"``
  в ``mirror_weight_path``); маскированный операнд — select, а не умножение
  (``0·NaN = NaN``); ``valid_config`` fail-closed отвергает
  ``rows % bm != 0`` (кламп последнего блока перекрывает строки, весовые
  фазы считали бы их дважды); неполная группа пишет каждую живую строку
  ровно один раз, dump-слоты не пересекаются с живыми; весовая CPU-эмуляция
  сходится с ``reference_bwd`` на полной/хвостовой группе.

Окружение: CPU-интерпретатор, ``NET_JAX_BACKEND=cpu``/``JAX_PLATFORMS=cpu``;
KDA-тело — ``jax``; fused-кернел не исполняется (только lowering), численный
пин — jnp-адъюнкт ``reference_bwd``.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
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


# --------------------------------------------------------------------------- #
# (е) структурная проба dz-эпилога (цикл-31, ADR-054 Am.53)
# --------------------------------------------------------------------------- #


def _dz_signature(rows: int = 32, latent: int = 768, inter: int = 128,
                  n_exp: int = 3, seed: int = 0):
    """Малая форма для проб dz-эпилога: пустая группа и неполный хвост."""
    return moe_experts_bwd.make_inputs(
        jr.PRNGKey(seed), rows=rows, latent=latent, inter=inter, n_exp=n_exp)


def _rel_l2(on, off) -> float:
    """``‖on − off‖ / ‖off‖`` — относительная L2-норма (метрика стенда)."""
    return float(jnp.linalg.norm(on - off) / jnp.linalg.norm(off))


def _group_layout(rows: int, n_exp: int) -> list[int]:
    """Раскладка ``group_sizes`` как у :func:`make_inputs`: пустая группа + хвост."""
    base = rows // n_exp
    sizes = [base] * n_exp
    sizes[0] += rows - base * n_exp
    sizes[1] = 0
    sizes[0] += base
    assert sum(sizes) == rows
    return sizes


@pytest.mark.parametrize("latent,bt", [(768, 128), (768, 384), (384, 128),
                                       (256, 256)])
def test_dz_tile_columns_partition_latent(latent, bt):
    """Колонки тайлов dz покрывают ``[0, latent)`` ровно один раз.

    Гард дефекта цикла-30: каждый тайл писался в колонки ``[0, bt)``, поэтому
    ``latent − bt`` колонок оставались незаписанными, а первая — перезаписанной
    последним тайлом.  Покрытие считается по тем же ``(c0, width)``, которые
    кернел отдаёт в ``pl.ds(c0, width)`` своей записи.
    """
    cols = moe_experts_bwd.dz_tile_columns(latent, bt)
    assert sum(width for _, width in cols) == latent
    assert [c0 for c0, _ in cols] == list(range(0, latent, bt))
    written = [c for c0, width in cols for c in range(c0, c0 + width)]
    assert written == list(range(latent)), (latent, bt)


def test_dz_tile_columns_reject_a_partial_tile():
    """Тайлинг, не покрывающий latent, — типизированный отказ, а не тихая дыра."""
    with pytest.raises(ValueError):
        moe_experts_bwd.dz_tile_columns(768, 100)


def test_kernel_out_types_carry_expert_rows_and_dump_slots():
    """Объявленные выходы кернела: ``dz`` — ``(rows + n_exp, latent)`` fp32.

    ``kernel()`` возвращает ``dz[:rows]``: строки экспертов — первые ``rows``
    (их адресует :func:`expert_row_slot`), хвост ``n_exp`` — dump-слоты для
    строк вне группы.  ``jax.eval_shape`` пинует форму выходов без устройства
    (lowering не нужен) — это CPU-часть проверки «правильная форма/строки».
    """
    case = moe_experts_bwd.CASES[1]
    rows, latent, inter = case["rows"], case["latent"], case["inter"]
    n_exp = case["n_exp"]
    shapes = jax.eval_shape(
        lambda: moe_experts_bwd.make_inputs(jax.random.key(0), **case))
    build = moe_experts_bwd._build_kernel(rows, latent, inter, n_exp, 16, 128,
                                          32, case["dtype"], 4.0, 25.0)
    out = jax.eval_shape(lambda *a: build(*a), *shapes)
    assert [tuple(o.shape) for o in out] == [
        (n_exp, latent, inter),                 # dwg
        (n_exp, latent, inter),                 # dwu
        (n_exp, inter, latent),                 # dwd
        (rows + n_exp, latent),                 # dz + dump-слоты
        (rows + n_exp, inter),                  # dhg скретч
        (rows + n_exp, inter),                  # dhu скретч
    ]
    assert [jnp.dtype(o.dtype) for o in out[3:]] == [
        jnp.float32, jnp.bfloat16, jnp.bfloat16]


@pytest.mark.parametrize("rows,n_exp,bm", [(2048, 12, 16), (256, 12, 16),
                                           (48, 3, 16), (50, 3, 16)])
def test_dz_row_slots_cover_the_expert_range(rows, n_exp, bm):
    """Строки группы пишутся каждая в свою строку, строки вне группы — в dump-слот.

    Целочисленная проба адресации, которую исполняет кернел: границы блоков
    (``expert_block_range``), смещение блока (``expert_block_offset``) и выбор
    строки записи (``expert_row_slot``) на префиксных суммах ``group_sizes``.
    Перекрытие строк между экспертами или запись строки группы в dump-слот
    ломают равенство с ``reference_bwd`` — это и есть проверяемое свойство.
    """
    sizes = _group_layout(rows, n_exp)
    start = 0
    for e, size in enumerate(sizes):
        end = start + size
        i0, i1 = moe_experts_bwd.expert_block_range(start, end, bm)
        counts: dict[int, int] = {}
        for i in range(int(i0), int(i1)):
            off = int(moe_experts_bwd.expert_block_offset(jnp.int32(i), rows, bm))
            assert off + bm <= rows
            for r in range(bm):
                gr = off + r
                slot = int(moe_experts_bwd.expert_row_slot(
                    jnp.int32(gr), jnp.int32(start), jnp.int32(end), rows, e))
                if start <= gr < end:
                    counts[slot] = counts.get(slot, 0) + 1
                else:
                    assert slot == rows + e, (e, gr, slot)
        assert set(counts) == set(range(start, end)), (e, sizes, sorted(counts))
        if size:
            assert min(counts.values()) >= 1
            # Кампанийные формы кратны bm: строка пишется ровно один раз.
            if rows % bm == 0:
                assert max(counts.values()) == 1, (e, sizes)
        start = end
    assert start == rows


def test_dz_kernel_writes_each_tile_into_its_own_columns(monkeypatch):
    """Запись, исполняемая самим телом кернела: тайл ``tj`` → колонки ``[tj·bt, (tj+1)·bt)``.

    ``_store_row`` — единственный путь записи строк в GMEM (скретч ``dhg``/
    ``dhu`` и dz-эпилог), поэтому проба подменяет его регистрирующей обёрткой
    на время трейса ``_build_kernel``: проверяется не jnp-эмуляция, а тот код,
    который уходит в Mosaic-нижний уровень.  При дефекте цикла-30 набор баз
    dz-тайлов был бы ``{0}``, а покрытие — ``[0, bt)``.
    """
    pytest.importorskip("jax.experimental.pallas.mosaic_gpu")
    if any(d.platform == "gpu" for d in jax.devices()):  # pragma: no cover — стенд
        pytest.skip("GPU присутствует: запись исполняется стендом (check_kernel.py)")

    case = moe_experts_bwd.CASES[1]                     # rows=256, latent=768, inter=384
    bm, bt = moe_experts_bwd.BM_DEFAULT, moe_experts_bwd.BT_DEFAULT
    calls: list[tuple[int, int, int]] = []
    real = moe_experts_bwd._store_row

    def _spy(out, idx, row_s, r, c0, width):
        calls.append((int(c0), int(width), int(r)))
        return real(out, idx, row_s, r, c0, width)

    monkeypatch.setattr(moe_experts_bwd, "_store_row", _spy)
    moe_experts_bwd._build_kernel.cache_clear()
    try:
        text = _lower_live_case(case)
    finally:
        moe_experts_bwd._build_kernel.cache_clear()
    assert "mosaic_gpu" in text, "lowering не дал Mosaic-кернел"

    expected = [tj * bt for tj in range(case["latent"] // bt)]
    dz_calls = [(c0, width, r) for c0, width, r in calls if width == bt]
    scratch = [(c0, width, r) for c0, width, r in calls if width == case["inter"]]
    bases = sorted({c0 for c0, _, _ in dz_calls})
    assert bases == expected, f"dz-тайлы пишутся в колонки {bases}, ожидались {expected}"
    assert {width for _, width, _ in dz_calls} == {bt}
    assert sorted({r for _, _, r in dz_calls}) == list(range(bm))
    per_tile = {c0: sum(1 for cc, _, _ in dz_calls if cc == c0) for c0 in expected}
    assert len(set(per_tile.values())) == 1, per_tile
    # Скретч dhg/dhu — всегда колонка 0 и полная ширина inter.
    assert {c0 for c0, _, _ in scratch} == {0}
    assert sorted({r for _, _, r in scratch}) == list(range(bm))
    print(f"dz store: bases={bases} calls/tile={per_tile[expected[0]]} "
          f"scratch_calls={len(scratch)}")


@pytest.mark.parametrize("weights", ["tiled", "full"])
def test_dz_mirror_matches_reference(weights):
    """CPU-зеркало dz-пути совпадает с ``reference_bwd`` (пустой эксперт + хвост).

    ``tiled`` — операнд-путь кернела (K-тайлы ``bk``, транспонированные тайлы
    ``wg``/``wu``), ``full`` — заведомо верная jnp-версия ``dhg·wgᵀ + dhu·wuᵀ``
    одним matmul.  Совпадение обоих с ``reference_bwd`` пинует формулы,
    ориентацию и порядок строк.
    """
    z, hg, hu, a, de, wg, wu, wd, gs = _dz_signature()
    _, _, _, dz_ref = moe_experts_bwd.reference_bwd(z, hg, hu, a, de, wg, wu, wd, gs)
    rows, latent = dz_ref.shape
    out = moe_experts_bwd.mirror_dz_path(hg, hu, de, wd, wg, wu, gs, weights=weights)
    rel = _rel_l2(out[:rows], dz_ref)
    print(f"mirror({weights}) vs reference_bwd: rel_l2={rel:.3e}")
    assert out.shape == (rows + wg.shape[0], latent)
    assert rel <= 1e-5


def test_dz_error_is_addressing_not_wg_orientation():
    """Ошибка — в адресации записи, а не в ориентации ``wgᵀ``/``wuᵀ``.

    Две сверки на одной форме:

    * операнд-путь кернела (``tiled``) и заведомо верная jnp-версия (``full``)
      дают одну и ту же dz: расхождение ≤ 1e-5 — ориентация и K-тайлинг не
      источник ошибки;
    * историческая политика записи (все тайлы в колонку 0) уводит **оба**
      режима одинаково — подмена ``wgᵀ`` на заведомо верную jnp-версию ошибку
      не снимает, значит, её источник — запись (строка/колонка/offset).
    """
    z, hg, hu, a, de, wg, wu, wd, gs = _dz_signature(latent=384, inter=128)
    _, _, _, dz_ref = moe_experts_bwd.reference_bwd(z, hg, hu, a, de, wg, wu, wd, gs)
    rows = dz_ref.shape[0]

    tiled = moe_experts_bwd.mirror_dz_path(hg, hu, de, wd, wg, wu, gs)
    full = moe_experts_bwd.mirror_dz_path(hg, hu, de, wd, wg, wu, gs, weights="full")
    assert _rel_l2(tiled[:rows], full[:rows]) <= 1e-5
    assert _rel_l2(full[:rows], dz_ref) <= 1e-5

    zero_column = lambda tj, c0, width: 0      # noqa: E731 — исторический дефект
    buggy = moe_experts_bwd.mirror_dz_path(
        hg, hu, de, wd, wg, wu, gs, store_column=zero_column)
    buggy_full = moe_experts_bwd.mirror_dz_path(
        hg, hu, de, wd, wg, wu, gs, weights="full", store_column=zero_column)
    rel_buggy = _rel_l2(buggy[:rows], dz_ref)
    rel_buggy_full = _rel_l2(buggy_full[:rows], dz_ref)
    print(f"store-at-column-0: tiled={rel_buggy:.4f} full={rel_buggy_full:.4f}")
    assert rel_buggy > 0.5
    assert rel_buggy_full == pytest.approx(rel_buggy, rel=1e-5)


def test_dz_store_defect_reproduces_the_cycle30_signature():
    """Историческая политика записи воспроизводит стендовую сигнатуру rel ≈ 1.28.

    ``latent=768``, ``bt=128``: незаписанных колонок 640 из 768, первая
    перезаписана последним тайлом, поэтому на пустом буфере
    ``‖Δ‖/‖dz‖ = sqrt((latent − bt + 2·bt)/latent) = sqrt(1 + bt/latent) =
    1.080``, а на остаточной памяти с RMS ≈ RMS(dz) — ``sqrt(2) = 1.414``.
    Наблюдённая на стенде 1.28 лежит между границами (остаточная память с
    RMS ≈ 0.75·RMS(dz)); исправленная политика даёт ≤ 1e-5.  Это объяснение
    сигнатуры, а не заявление паритета исполненного ядра (ADR-011).
    """
    z, hg, hu, a, de, wg, wu, wd, gs = _dz_signature()
    _, _, _, dz_ref = moe_experts_bwd.reference_bwd(z, hg, hu, a, de, wg, wu, wd, gs)
    rows, latent = dz_ref.shape
    bt = moe_experts_bwd.BT_DEFAULT

    fixed = moe_experts_bwd.mirror_dz_path(hg, hu, de, wd, wg, wu, gs)[:rows]
    buggy = moe_experts_bwd.mirror_dz_path(
        hg, hu, de, wd, wg, wu, gs, store_column=lambda tj, c0, width: 0)[:rows]
    rel_fixed = _rel_l2(fixed, dz_ref)
    rel_zero = _rel_l2(buggy, dz_ref)
    # Остаточная память: детерминированный «мусор» той же RMS в незаписанных колонках.
    touched = jnp.any(buggy != 0, axis=0, keepdims=True)
    leftover = jr.normal(jr.PRNGKey(13), buggy.shape) * jnp.sqrt(jnp.mean(dz_ref ** 2))
    rel_residual = _rel_l2(buggy + leftover * (~touched), dz_ref)
    print(f"dz store: fixed={rel_fixed:.3e} zero-init={rel_zero:.4f} "
          f"residual={rel_residual:.4f} analytic={math.sqrt(1.0 + bt / latent):.4f}")
    assert rel_fixed <= 1e-5
    # Аналитическая оценка — с точностью до выборочных кросс-членов.
    assert rel_zero == pytest.approx(math.sqrt(1.0 + bt / latent), abs=5e-3)
    assert 1.0 <= rel_residual <= 1.45
    # Стендовая 1.28 — внутри интервала [1.080, 1.414]: сигнатура объяснена.
    assert rel_zero <= 1.28 <= math.sqrt(2.0)


# --------------------------------------------------------------------------- #
# (ж) guard'ы цикла-32: пустая/неполная группа, rows % bm fail-closed
# --------------------------------------------------------------------------- #


def test_valid_config_requires_rows_multiple_of_bm():
    """``rows % bm != 0`` — fail-closed: сито отвергает форму, CASES не меняются.

    При некратных ``rows`` кламп последнего блока (``expert_block_offset``)
    перекрывает строки, и весовые фазы с K-потоком по строкам посчитали бы
    их дважды.  Кампанийная и тестовая геометрия ``CASES`` кратна ``bm=16`` и
    проходит.
    """
    params = {"bm": 16, "bt": 128, "bk": 32}
    base = dict(rows=256, latent=768, inter=384, n_exp=12, dtype="bfloat16")
    assert moe_experts_bwd.valid_config(base, params)
    for rows in (257, 264, 100):
        assert not moe_experts_bwd.valid_config(dict(base, rows=rows), params), rows
    # Проверка идёт по действующему bm: bm=32 требует rows % 32 == 0.
    small = dict(rows=2048, latent=128, inter=128, n_exp=12, dtype="bfloat16")
    wide = {"bm": 32, "bt": 128, "bk": 32}
    assert moe_experts_bwd.valid_config(small, wide)
    assert not moe_experts_bwd.valid_config(dict(small, rows=2064), wide)


def test_kernel_refuses_non_multiple_rows_typed(monkeypatch):
    """Некратные rows — :class:`MoeBwdKernelError` до построения/исполнения."""
    monkeypatch.setattr(moe_experts_bwd, "device_available", lambda: True)
    args = _kernel_inputs(rows=2050)
    with pytest.raises(moe_experts_bwd.MoeBwdKernelError) as excinfo:
        moe_experts_bwd.kernel(*args)
    message = str(excinfo.value)
    assert "valid_config" in message and "rows % bm" in message


def test_masked_operand_selects_zero_instead_of_nan():
    """Маска строки — select: nan/inf вне группы не попадают в аккумулятор.

    Стендовый механизм цикла-32: весовая фаза читает скретч блоками строк, а
    строки соседней группы могли быть ещё не записаны (неинициализированный
    GMEM).  ``0·NaN = NaN``; select выбирает ноль и не протаскивает nan.
    """
    values = jnp.array([[jnp.nan, 2.0], [jnp.inf, -1.0]], jnp.float32)
    mask = jnp.array([[0.0], [1.0]], jnp.float32)
    out = moe_experts_bwd.masked_operand(values, mask)
    assert jnp.all(jnp.isfinite(out[0]))
    assert jnp.all(out[0] == 0)
    assert jnp.array_equal(out[1], values[1])
    # Контроль: умножение на маску — источник nan (0·nan = nan).
    assert jnp.isnan(values * mask).any()


def test_empty_group_weight_guard_ignores_the_accumulator():
    """Пустая группа: выход весового тайла — ровно нули, carry не читается."""
    acc = jnp.full((4, 4), jnp.nan, jnp.float32)
    zero = moe_experts_bwd.empty_group_weight_guard(acc, jnp.int32(0))
    assert jnp.all(zero == 0) and jnp.all(jnp.isfinite(zero))
    # Непустая группа: аккумулятор проходит без изменений.
    keep = moe_experts_bwd.empty_group_weight_guard(acc, jnp.int32(3))
    assert jnp.all(jnp.isnan(keep))


def test_mirror_weight_empty_group_is_zero_and_finite():
    """Пустая группа стендовой формы (rows=256, 12 экспертов): нули без nan.

    ``make_inputs`` кампанийной формы отдаёт эксперту 1 ноль строк.  Зеркало
    весовых фаз прогоняется дважды: с нулевым скретчем и с nan-заполненным
    (неинициализированный GMEM соседних групп — условие стенда).  Оба прогона
    обязаны дать одинаковые конечные ``dwg``/``dwu``/``dwd`` и ровно нулевой
    срез пустого эксперта; строки его блоков адресуются только в dump-слот
    ``rows + e`` и не пересекаются с живыми.
    """
    case = moe_experts_bwd.CASES[1]                     # rows=256, 12 экспертов
    z, hg, hu, a, de, wg, wu, wd, gs = moe_experts_bwd.make_inputs(
        jr.PRNGKey(0), **case)
    rows = case["rows"]
    bm = moe_experts_bwd.BM_DEFAULT
    assert int(gs[1]) == 0
    assert int(jnp.sum(gs)) == rows

    grads = {}
    for init in ("zeros", "nan"):
        dwg, dwu, dwd, dhg_s, dhu_s = moe_experts_bwd.mirror_weight_path(
            z, hg, hu, a, de, wg, wu, wd, gs, scratch_init=init)
        for name, grad in (("dwg", dwg), ("dwu", dwu), ("dwd", dwd)):
            assert jnp.all(jnp.isfinite(grad)), (init, name)
            assert jnp.all(grad[1] == 0), (init, name)
        # Guard не занулил всю программу: непустая группа живая.
        assert jnp.any(dwg[0] != 0)
        grads[init] = (dwg, dwu, dwd, dhg_s, dhu_s)
    for lhs, rhs in zip(grads["zeros"][:3], grads["nan"][:3]):
        assert jnp.array_equal(lhs, rhs), "nan-скретч изменил весовой градиент"

    # Адресация строк пустой группы: блок(и) пишут только собственный dump-слот.
    start = int(jnp.cumsum(gs.astype(jnp.int32))[0])
    end = start
    i0, i1 = moe_experts_bwd.expert_block_range(start, end, bm)
    slots = set()
    for i in range(int(i0), int(i1)):
        off = int(moe_experts_bwd.expert_block_offset(jnp.int32(i), rows, bm))
        for r in range(bm):
            slots.add(int(moe_experts_bwd.expert_row_slot(
                jnp.int32(off + r), jnp.int32(start), jnp.int32(end), rows, 1)))
    assert slots == {rows + 1}
    assert slots.isdisjoint(set(range(rows)))


@pytest.mark.parametrize("rows,bm,sizes", [
    (64, 16, [23, 0, 41]),
    (64, 16, [17, 47]),
    (96, 16, [1, 0, 95]),
])
def test_partial_group_rows_written_once_and_dump_disjoint(rows, bm, sizes):
    """Неполная группа: каждая живая строка — ровно один раз, dump вне живых.

    Кратные ``rows`` (условие сита) и группы, не выровненные по ``bm``: блоки
    с запасом по краям пересекаются с соседними группами, но маска оставляет
    каждую живую строку ровно в одном блоке, а строки вне группы уходят в
    dump-слот эксперта ``rows + e`` (он не пересекается с ``[0, rows)``).
    """
    assert sum(sizes) == rows and rows % bm == 0
    start = 0
    dump = set()
    for e, size in enumerate(sizes):
        end = start + size
        i0, i1 = moe_experts_bwd.expert_block_range(start, end, bm)
        counts: dict[int, int] = {}
        for i in range(int(i0), int(i1)):
            off = int(moe_experts_bwd.expert_block_offset(jnp.int32(i), rows, bm))
            assert 0 <= off and off + bm <= rows
            for r in range(bm):
                gr = off + r
                slot = int(moe_experts_bwd.expert_row_slot(
                    jnp.int32(gr), jnp.int32(start), jnp.int32(end), rows, e))
                if start <= gr < end:
                    counts[slot] = counts.get(slot, 0) + 1
                else:
                    assert slot == rows + e, (e, gr, slot)
                    dump.add(slot)
        assert sorted(counts) == list(range(start, end)), (e, sizes)
        if size:
            assert max(counts.values()) == 1, (e, sizes)
        start = end
    assert start == rows
    assert dump and dump.isdisjoint(set(range(rows)))


def test_non_multiple_rows_overlap_is_why_fail_closed():
    """Свидетель причины fail-closed: rows=50, bm=16 — строки 34..47 дважды.

    Последний блок кламплится к ``rows − bm`` и накрывает хвост предыдущего;
    весовые фазы с K-потоком по строкам посчитали бы перекрытие дважды.
    Поэтому :func:`valid_config` и зеркало отвергают некратные формы.
    """
    rows, bm = 50, 16
    i0, i1 = moe_experts_bwd.expert_block_range(0, rows, bm)
    offsets = [int(moe_experts_bwd.expert_block_offset(jnp.int32(i), rows, bm))
               for i in range(int(i0), int(i1))]
    assert offsets == [0, 16, 32, 34]
    seen: dict[int, int] = {}
    for off in offsets:
        for gr in range(off, off + bm):
            seen[gr] = seen.get(gr, 0) + 1
    assert sorted(g for g, n in seen.items() if n > 1) == list(range(34, 48))

    case = dict(rows=rows, latent=768, inter=384, n_exp=2, dtype="bfloat16")
    assert not moe_experts_bwd.valid_config(
        case, {"bm": bm, "bt": 128, "bk": 32})
    z, hg, hu, a, de, wg, wu, wd, _ = moe_experts_bwd.make_inputs(
        jr.PRNGKey(0), rows=50, latent=256, inter=128, n_exp=2)
    with pytest.raises(ValueError):
        moe_experts_bwd.mirror_weight_path(
            z, hg, hu, a, de, wg, wu, wd, jnp.array([25, 25], jnp.int32))


@pytest.mark.parametrize("sizes", [[32, 32], [23, 41], [17, 47], [64, 0]])
def test_weight_mirror_matches_reference(sizes):
    """Регресс цикла-32: весовая эмуляция равна ``reference_bwd``, dz не тронут.

    Полная группа, хвостовая (неполная) и пустая — все три весовых
    градиента совпадают с jnp-адъюнктом; dz-паритет зеркала
    (``mirror_dz_path``) сохраняется — путь цикла-31 не сломан.
    """
    rows, latent, inter, n_exp = 64, 256, 128, 2
    z, hg, hu, a, de, wg, wu, wd, _ = moe_experts_bwd.make_inputs(
        jr.PRNGKey(0), rows=rows, latent=latent, inter=inter, n_exp=n_exp)
    gs = jnp.asarray(sizes, jnp.int32)
    dwg, dwu, dwd, dz_ref = moe_experts_bwd.reference_bwd(
        z, hg, hu, a, de, wg, wu, wd, gs)
    mdwg, mdwu, mdwd, _dhg_s, _dhu_s = moe_experts_bwd.mirror_weight_path(
        z, hg, hu, a, de, wg, wu, wd, gs)
    for name, on, off in (("dwg", mdwg, dwg), ("dwu", mdwu, dwu),
                          ("dwd", mdwd, dwd)):
        rel = _rel_l2(on, off)
        assert rel <= 1e-5, (name, rel)
    dz = moe_experts_bwd.mirror_dz_path(hg, hu, de, wd, wg, wu, gs)[:rows]
    assert _rel_l2(dz, dz_ref) <= 1e-5


def test_kernel_weight_tiles_are_guarded_on_trace(monkeypatch):
    """Тело кернела: каждый из трёх весовых выходов обёрнут guard'ом.

    Регистрирующая обёртка ``empty_group_weight_guard`` на трейсе
    ``_build_kernel``: три весовые фазы (dwd, dwg, dwu) обязаны пройти через
    guard пустой группы — иначе на стенде rows=256 вновь появится ``dwg=nan``.
    """
    pytest.importorskip("jax.experimental.pallas.mosaic_gpu")
    case = moe_experts_bwd.CASES[1]
    calls: list[tuple[int, ...]] = []
    real = moe_experts_bwd.empty_group_weight_guard

    def _spy(acc, size):
        calls.append(tuple(acc.shape))
        return real(acc, size)

    monkeypatch.setattr(moe_experts_bwd, "empty_group_weight_guard", _spy)
    moe_experts_bwd._build_kernel.cache_clear()
    try:
        shapes = jax.eval_shape(lambda: moe_experts_bwd.make_inputs(
            jax.random.key(0), **case))
        build = moe_experts_bwd._build_kernel(
            case["rows"], case["latent"], case["inter"], case["n_exp"],
            16, 128, 32, case["dtype"], 4.0, 25.0)
        jax.eval_shape(lambda *args: build(*args), *shapes)
    finally:
        moe_experts_bwd._build_kernel.cache_clear()
    assert len(calls) == 3, calls
    assert all(shape == (128, 128) for shape in calls), calls
