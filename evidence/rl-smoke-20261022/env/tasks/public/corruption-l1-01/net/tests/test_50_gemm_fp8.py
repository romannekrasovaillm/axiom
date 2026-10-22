"""Цикл-26 (ADR-054 Am.45): full-step FP8 GEMM плотных проекций/MLP + STE.

Мотив: FP8-CE in situ дал минус (Am.44), изолированные 85 TFLOPS не
переносятся.  Цикл-26 включает FP8 на **тяжёлых плотных проекциях шага**
(проекции attention, MLP) и оставляет CE/lm_head, MoE и oracle-пути на
прежнем fp32/bf16-пути побитово.  Гейт ``AXIOM_GEMM_FP8`` (default ``0``):

* ``net/compute_dtype.gemm_fp8`` — ``jax.custom_vjp`` с границей
  ``(a, b) -> c``: forward ``matmul(q(a), q(b))·(s_a·s_b)`` (e4m3,
  per-tensor скейлы, fp32-аккум), backward — bf16/fp32-граница по
  **неквантованным** операндам (STE; скейлы — константы);
* ``compute_dtype.gemm(..., fp8=True)`` — объявленные проекционные callsite-ы
  (``kda._project``/``_output_gate``/окно, ``mla`` проекции латента/выхода,
  ``mlp.apply``).  Дефолт ``fp8=False`` оставляет CE/lm_head и MoE нетронутыми:
  глобальная подмена ``gemm`` под гейтом смешала бы замер FP8-проекций с
  измеренным минусом FP8-CE.

Что пиннуется здесь (CPU, jax 0.11.2, шов ``_fp8_matmul`` — деквантованный
f32-матмул):

* **(а) контракт гейта**: пусто/не задано = ``"0"``; ``"0"``/``"1"``;
  пробелы нормализуются; неизвестное — :class:`GemmFp8Error` (fail-closed);
  явный аргумент сильнее env; значение читается на трейсе (jaxpr под
  ``"1"`` содержит ``float8_e4m3fn`` и ``reduce_max`` скейлов);
* **(б) off — побитово ревизия до дельты** (``1c684ba``, ``baseline_commit``
  пакета): ``compute_dtype.gemm`` (включая ``fp8=True``) и callsite-ы
  ``kda._project``/``kda.apply_kda``/``kda.window_output``, ``mla._project``/
  ``_dense_apply``/``_sparse_apply``, ``mlp.apply`` совпадают с извлечённым
  ``net/`` baseline; CE-путь (``_chunk_sum_plain``) под гейтом ``"1"``
  побитово равен гейту ``"0"`` — блокирующий критерий цикла;
* **(в) on — численность**: forward против bf16-эталона и градиенты a/b
  против bf16-эталона в scale-relative ``max|d|/scale``; градиенты STE
  побитово равны градиентам обычного ``a @ b`` (математика backward не
  квантуется);
* **(г) композит**: KDA + MLA + MLP при гейте on — лосс с нормированным
  readout равен off в допуске; два шага SGD без nan;
* **(д) скейлы**: нулевой/вырожденный тензор (max = 0) не даёт nan; backward
  не содержит ``reduce_max`` — скейлы не текут в градиенты.

Фактическая численность (CPU, сид 0, ``a ~ N(0,1)``, ``b ~ N(0,0.02)``,
формы 32×64×32): forward ``max|d|/scale = 4.0e-2`` — это сетка e4m3
(3 мантиссных бита: шаг 1/8, максимум относительной ошибки округления
6.25 %, RMS ≈ 3.6 %), поэтому допуск forward — ``5e-2``, а не 2e-2:
per-tensor e4m3 физически не даёт 2e-2 на случайном GEMM.  Градиенты STE
идут через bf16/fp32 и укладываются в кампанийный ``2e-2`` (факт ≈ 3.2e-3).
Композитный лосс — ``1.7e-2`` (факт), допуск ``2e-2``.

Окружение: CPU-интерпретатор, ``NET_JAX_BACKEND=cpu``/``JAX_PLATFORMS=cpu``;
KDA-тело — ``jax`` (персистентный кернел кампании на CPU отказывает).
"""

from __future__ import annotations

import dataclasses
import io
import subprocess
import sys
import tarfile
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from conftest import small_config

from net import compute_dtype, kda, mla, mlp, model

#: Ревизия до дельты (``baseline_commit`` handoff-пакета): off обязан
#: воспроизводить её побитово.  ``git archive`` + skip, если недоступна
#: (форма test_29/test_47).
BASELINE_COMMIT = "1c684ba"

CASE_DIR = Path(__file__).resolve().parents[2]

#: Гейт FP8-проекций.
GATE = compute_dtype.GEMM_FP8_ENV

#: Допуск forward: сетка e4m3 (макс. относительная ошибка округления 6.25 %,
#: RMS 3.6 %); фактический ``max|d|/scale`` на случайном GEMM ≈ 3.5e-2.
FWD_TOL = 5e-2

#: Кампанийный допуск STE-градиентов (Am.45: ≤2e-2).
GRAD_TOL = 2e-2

#: Допуск композитного лосса (факт 1.7e-2 на нормированном readout).
COMPOSITE_TOL = 2e-2

#: Допуск композитных градиентов (fp8-ошибки проекций каскадируются;
#: факт ≈ 1.2e-1 — справочная величина, не блокирующий критерий).
COMPOSITE_GRAD_TOL = 2e-1

#: Геометрия композита: (T, hidden) маленькой модели conftest.
SEQ = 8


# --------------------------------------------------------------------------- #
# Общие помощники
# --------------------------------------------------------------------------- #


def _dequant_matmul(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    """Шов CPU: fp8-операнды деквантуются в f32 (значения e4m3 точны в f32)."""
    return jnp.matmul(a.astype(jnp.float32), b.astype(jnp.float32))


@pytest.fixture(autouse=True)
def _cpu_bodies(monkeypatch):
    """CPU-исполнение: KDA — ``jax``, прочие кампанийные гейты сняты; шов CPU."""
    monkeypatch.setenv("AXIOM_KDA_CC_IMPL", "jax")
    monkeypatch.delenv("AXIOM_MLA_PERSIST", raising=False)
    monkeypatch.delenv("AXIOM_MLA_DENSE_FLASH", raising=False)
    monkeypatch.delenv("AXIOM_MLA_BWD_BF16", raising=False)
    monkeypatch.delenv("AXIOM_CE_BF16", raising=False)
    monkeypatch.delenv("AXIOM_CE_FP8", raising=False)
    monkeypatch.delenv(compute_dtype.MODE_ENV, raising=False)
    monkeypatch.delenv(GATE, raising=False)
    monkeypatch.setattr(compute_dtype, "_fp8_matmul", _dequant_matmul)


def _bits(x: jnp.ndarray) -> np.ndarray:
    """Сырые fp32-биты массива (NaN-безопасное побитовое сравнение)."""
    arr = np.asarray(jnp.asarray(x))
    assert arr.dtype == np.float32, f"побитовое сравнение ждёт fp32, получено {arr.dtype}"
    return arr.view(np.uint32)


def _bitwise_equal(a: jnp.ndarray, b: jnp.ndarray) -> bool:
    return bool(np.array_equal(_bits(a), _bits(b)))


def _max_abs(x) -> float:
    return float(jnp.max(jnp.abs(x)))


def _scale_relative(on, off) -> tuple[float, float, float]:
    """``(max|d|, scale, max|d|/scale)``; scale — максимум модуля обоих."""
    delta = _max_abs(on - off)
    scale = max(_max_abs(on), _max_abs(off))
    return delta, scale, (delta / scale if scale else 0.0)


def _tree_relative(on, off) -> tuple[float, float, float]:
    """То же для деревьев параметров/градиентов (общий масштаб дерева)."""
    leaves_on = jax.tree_util.tree_leaves(on)
    leaves_off = jax.tree_util.tree_leaves(off)
    delta = max(_max_abs(x - y) for x, y in zip(leaves_on, leaves_off))
    scale = max(
        max(_max_abs(x) for x in leaves_on),
        max(_max_abs(y) for y in leaves_off),
    )
    return delta, scale, (delta / scale if scale else 0.0)


def _bf16_matmul(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    """bf16-эталон: bf16-операнды, fp32-аккумулятор (Beam-рецепт)."""
    return jnp.matmul(
        a.astype(jnp.bfloat16),
        b.astype(jnp.bfloat16),
        preferred_element_type=jnp.float32,
    )


def _set_gate(monkeypatch, value: str | None) -> None:
    if value is None:
        monkeypatch.delenv(GATE, raising=False)
    else:
        monkeypatch.setenv(GATE, value)


def _bcfg(base_cfg, bconfig):
    """Тот же конфиг в пространстве имён baseline-пакета."""
    return bconfig.ModelConfig(**dataclasses.asdict(base_cfg))


def _primitives(obj, out=None, depth=0) -> list[str]:
    """Имена примитивов jaxpr, включая вложенные (custom_vjp/remat)."""
    if out is None:
        out = []
    if depth > 25:
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


@pytest.fixture(scope="module")
def baseline():
    """``net/`` на ``BASELINE_COMMIT``, импортируемый как ``baseline_net``.

    ``git archive`` (не worktree), поэтому состояние чекаута не трогается, а
    пакет переименован — относительные импорты внутри ``net/`` продолжают
    работать (форма test_29/test_47).
    """
    root = Path(__file__).resolve().parent / ".baseline-net-test50"
    if root.exists():
        import shutil

        shutil.rmtree(root)
    root.mkdir()
    try:
        archive = subprocess.run(
            ["git", "-C", str(CASE_DIR), "archive", BASELINE_COMMIT, "net/"],
            check=True,
            capture_output=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:  # pragma: no cover
        pytest.skip(f"baseline {BASELINE_COMMIT} недоступен: {exc}")
    with tarfile.open(fileobj=io.BytesIO(archive.stdout)) as bundle:
        bundle.extractall(root)
    (root / "net").rename(root / "baseline_net")

    sys.path.insert(0, str(root))
    try:
        import baseline_net.compute_dtype as bcompute_dtype
        import baseline_net.config as bconfig
        import baseline_net.kda as bkda
        import baseline_net.mla as bmla
        import baseline_net.mlp as bmlp
        import baseline_net.model as bmodel

        yield SimpleNamespace(
            compute_dtype=bcompute_dtype,
            config=bconfig,
            kda=bkda,
            mla=bmla,
            mlp=bmlp,
            model=bmodel,
        )
    finally:
        sys.path.remove(str(root))
        for name in [m for m in list(sys.modules) if m.startswith("baseline_net")]:
            del sys.modules[name]
        import shutil

        shutil.rmtree(root, ignore_errors=True)


# --------------------------------------------------------------------------- #
# (а) контракт гейта
# --------------------------------------------------------------------------- #


def test_gate_default_is_off(monkeypatch):
    """Пусто/не задано = ``"0"``: дефолт не включает FP8 ни на одном callsite."""
    monkeypatch.delenv(GATE, raising=False)
    assert compute_dtype.gemm_fp8_mode() is False
    for raw in ("", "   ", "  "):
        monkeypatch.setenv(GATE, raw)
        assert compute_dtype.gemm_fp8_mode() is False, repr(raw)


def test_gate_values_and_whitespace(monkeypatch):
    """``"0"``/``"1"``; окружающие пробелы нормализуются."""
    for raw, expected in {
        "0": False,
        "1": True,
        " 0 ": False,
        " 1 ": True,
        "\t1\n": True,
    }.items():
        monkeypatch.setenv(GATE, raw)
        assert compute_dtype.gemm_fp8_mode() is expected, repr(raw)


@pytest.mark.parametrize("raw", ["2", "01", "-1", "true", "on", "yes", "fP8"])
def test_gate_unknown_fails_closed(monkeypatch, raw):
    """Неизвестное значение — :class:`GemmFp8Error` (тихий откат запрещён)."""
    monkeypatch.setenv(GATE, raw)
    with pytest.raises(compute_dtype.GemmFp8Error) as excinfo:
        compute_dtype.gemm_fp8_mode()
    assert GATE in str(excinfo.value)
    assert repr(raw.strip().lower()) in str(excinfo.value)


def test_gate_explicit_argument_beats_env(monkeypatch):
    """Явный аргумент сильнее env (включая неизвестное env-значение)."""
    monkeypatch.setenv(GATE, "0")
    assert compute_dtype.gemm_fp8_mode("1") is True
    monkeypatch.setenv(GATE, "1")
    assert compute_dtype.gemm_fp8_mode("0") is False
    monkeypatch.setenv(GATE, "2")
    assert compute_dtype.gemm_fp8_mode("0") is False
    assert compute_dtype.gemm_fp8_mode("1") is True


def test_gate_is_read_on_trace(monkeypatch):
    """Гейт разрешается на трейсе: jaxpr под ``"1"`` содержит fp8 и скейлы."""
    monkeypatch.setenv(GATE, "0")
    a = jr.normal(jr.PRNGKey(0), (4, 8))
    b = jr.normal(jr.PRNGKey(1), (8, 6))

    off = str(jax.make_jaxpr(lambda x, y: compute_dtype.gemm(x, y, fp8=True))(a, b))
    assert "float8_e4m3fn" not in off
    assert "reduce_max" not in off

    monkeypatch.setenv(GATE, "1")
    on = str(jax.make_jaxpr(lambda x, y: compute_dtype.gemm(x, y, fp8=True))(a, b))
    assert "float8_e4m3fn" in on, "FP8-ветка не попала в граф под гейтом on"
    assert "reduce_max" in on, "per-tensor скейлы не попали в граф под гейтом on"
    # CE-путь (без ``fp8=True``) под тем же гейтом остаётся прежним.
    ce = str(jax.make_jaxpr(lambda x, y: compute_dtype.gemm(x, y))(a, b))
    assert "float8_e4m3fn" not in ce and "reduce_max" not in ce


# --------------------------------------------------------------------------- #
# (б) off — побитово ревизия до дельты
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("compute", [None, "bf16"])
def test_gemm_off_is_bitwise_baseline(baseline, monkeypatch, compute):
    """``gemm`` (дефолт и ``fp8=True``) при гейте off — побитово baseline."""
    monkeypatch.setenv(GATE, "0")
    if compute is None:
        monkeypatch.delenv(compute_dtype.MODE_ENV, raising=False)
    else:
        monkeypatch.setenv(compute_dtype.MODE_ENV, compute)

    a = jr.normal(jr.PRNGKey(0), (13, 17))
    b = jr.normal(jr.PRNGKey(1), (17, 9))
    ref = baseline.compute_dtype.gemm(a, b)
    assert _bitwise_equal(compute_dtype.gemm(a, b), ref)
    assert _bitwise_equal(compute_dtype.gemm(a, b, fp8=True), ref)


def test_kda_callsites_off_are_bitwise_baseline(baseline, monkeypatch):
    """``kda._project``/``apply_kda``/``window_output`` при off — baseline."""
    monkeypatch.setenv(GATE, "0")
    cfg = small_config()
    bcfg = _bcfg(cfg, baseline.config)
    key = jr.PRNGKey(0)
    params = kda.init_kda(key, cfg)
    bparams = baseline.kda.init_kda(key, bcfg)
    x = jr.normal(jr.PRNGKey(1), (SEQ, cfg.hidden))

    got = kda._project(params, cfg, x)
    ref = baseline.kda._project(bparams, bcfg, x)
    assert set(got) == set(ref)
    for name in got:
        assert _bitwise_equal(got[name], ref[name]), name

    assert _bitwise_equal(
        kda.apply_kda(params, cfg, x, chunk_size=4),
        baseline.kda.apply_kda(bparams, bcfg, x, chunk_size=4),
    )

    cfg_win = dataclasses.replace(
        cfg, attn_dense_reference=False, swa_window=4,
        swa_share_kda_projections=False,
    )
    bcfg_win = dataclasses.replace(
        bcfg, attn_dense_reference=False, swa_window=4,
        swa_share_kda_projections=False,
    )
    params_win = kda.init_kda(key, cfg_win)
    bparams_win = baseline.kda.init_kda(key, bcfg_win)
    assert _bitwise_equal(
        kda.window_output(params_win, cfg_win, x),
        baseline.kda.window_output(bparams_win, bcfg_win, x),
    )


def test_mla_callsites_off_are_bitwise_baseline(baseline, monkeypatch):
    """Проекции MLA (dense и sparse) при off — побитово baseline."""
    monkeypatch.setenv(GATE, "0")
    cfg = small_config()
    bcfg = _bcfg(cfg, baseline.config)
    key = jr.PRNGKey(2)
    params = mla.init_mla(key, cfg)
    bparams = baseline.mla.init_mla(key, bcfg)
    x = jr.normal(jr.PRNGKey(3), (1, SEQ, cfg.hidden))
    x32 = x.astype(jnp.float32)

    c = compute_dtype.gemm(x32, params.W_c.astype(jnp.float32))
    bc = baseline.compute_dtype.gemm(x32, bparams.W_c.astype(jnp.float32))
    assert _bitwise_equal(c, bc)
    got = mla._project(params, cfg, x32, c)
    ref = baseline.mla._project(bparams, bcfg, x32, bc)
    for name, (g, r) in enumerate(zip(got, ref)):
        assert _bitwise_equal(g, r), name

    assert _bitwise_equal(
        mla._dense_apply(params, cfg, x),
        baseline.mla._dense_apply(bparams, bcfg, x),
    )

    cfg_sp = dataclasses.replace(
        cfg,
        attn_dense_reference=False,
        mla_top_k=4,
        swa_window=4,
        mla_layer_modes=("full",),
    )
    bcfg_sp = dataclasses.replace(
        bcfg,
        attn_dense_reference=False,
        mla_top_k=4,
        swa_window=4,
        mla_layer_modes=("full",),
    )
    assert _bitwise_equal(
        mla._sparse_apply(params, cfg_sp, x)[0],
        baseline.mla._sparse_apply(bparams, bcfg_sp, x)[0],
    )


def test_mlp_callsite_off_is_bitwise_baseline(baseline, monkeypatch):
    """``mlp.apply`` (gate/up/down) при off — побитово baseline."""
    monkeypatch.setenv(GATE, "0")
    cfg = small_config()
    bcfg = _bcfg(cfg, baseline.config)
    key = jr.PRNGKey(4)
    params = mlp.init_mlp(key, cfg)
    bparams = baseline.mlp.init_mlp(key, bcfg)
    x = jr.normal(jr.PRNGKey(5), (2, SEQ, cfg.hidden))
    assert _bitwise_equal(mlp.apply(params, cfg, x), baseline.mlp.apply(bparams, bcfg, x))


def test_ce_path_is_bitwise_unchanged_under_gate_on(monkeypatch):
    """CE не трогается: ``_chunk_sum_plain`` под ``"1"`` побитово равен ``"0"``.

    Блокирующий критерий цикла: FP8-гейт включает только объявленные
    проекционные callsite-ы; lm_head/CE (и его градиент) остаются прежними,
    иначе замер FP8-проекций смешался бы с измеренным минусом FP8-CE (Am.44).
    """
    f = jr.normal(jr.PRNGKey(6), (2, 7, 16))
    t = jr.randint(jr.PRNGKey(7), (2, 7), 1, 64)
    emb = jr.normal(jr.PRNGKey(8), (64, 16)) * 0.02
    mask = jnp.zeros((2, 7), jnp.float32).at[:, :5].set(1.0)

    _set_gate(monkeypatch, "0")
    v0, c0 = model._chunk_sum_plain(f, t, emb, mask)
    _set_gate(monkeypatch, "1")
    v1, c1 = model._chunk_sum_plain(f, t, emb, mask)
    assert _bitwise_equal(v0, v1)
    assert _bitwise_equal(c0, c1)


# --------------------------------------------------------------------------- #
# (в) on — forward и STE-градиенты
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "a_shape,b_shape",
    [
        ((32, 64), (64, 32)),
        ((2, 16, 64), (64, 32)),  # ведущие оси a (батч), b — 2-D вес
    ],
)
def test_forward_matches_bf16_reference(monkeypatch, a_shape, b_shape):
    """on: forward ``c`` против bf16-эталона в scale-relative.

    Фактический ``max|d|/scale`` печатается; допуск ``5e-2`` — сетка e4m3
    (см. docstring модуля).  ``scale`` — максимум модуля обоих выходов.
    """
    _set_gate(monkeypatch, "1")
    a = jr.normal(jr.PRNGKey(0), a_shape) * 1.0
    b = jr.normal(jr.PRNGKey(1), b_shape) * 0.02
    c = compute_dtype.gemm(a, b, fp8=True)
    ref = _bf16_matmul(a, b)
    delta, scale, rel = _scale_relative(c, ref)
    print(f"[цикл-26] forward shape={a.shape}: max|d|={delta:.3e} scale={scale:.3e} rel={rel:.3e}")
    assert c.dtype == jnp.float32
    assert rel <= FWD_TOL, f"forward rel={rel:.3e} > {FWD_TOL:.1e}"
    assert not _bitwise_equal(c, ref), "FP8-ветка не включилась (выход побитово bf16)"


def test_ste_gradients_match_bf16_reference(monkeypatch):
    """on: градиенты a/b против bf16-эталона в scale-relative ≤2e-2."""
    _set_gate(monkeypatch, "1")
    a = jr.normal(jr.PRNGKey(2), (32, 64))
    b = jr.normal(jr.PRNGKey(3), (64, 32)) * 0.02
    g = jr.normal(jr.PRNGKey(4), (32, 32))

    def loss_fp8(x, y):
        return jnp.sum(compute_dtype.gemm(x, y, fp8=True) * g)

    def loss_ref(x, y):
        return jnp.sum(_bf16_matmul(x, y) * g)

    ga, gb = jax.grad(loss_fp8, argnums=(0, 1))(a, b)
    ra, rb = jax.grad(loss_ref, argnums=(0, 1))(a, b)
    for name, got, ref in (("a", ga, ra), ("b", gb, rb)):
        delta, scale, rel = _scale_relative(got, ref)
        print(f"[цикл-26] grad {name}: max|d|={delta:.3e} scale={scale:.3e} rel={rel:.3e}")
        assert rel <= GRAD_TOL, f"grad {name} rel={rel:.3e} > {GRAD_TOL:.1e}"


def test_ste_gradients_are_the_plain_gemm_gradients(monkeypatch):
    """STE: backward не квантуется — градиенты побитово как у ``a @ b``.

    Это же доказывает, что скейлы не текут в котангенты: backward не читает
    ``max|a|``/``max|b|`` (их нет в остатке custom_vjp).
    """
    _set_gate(monkeypatch, "1")
    a = jr.normal(jr.PRNGKey(5), (9, 7))
    b = jr.normal(jr.PRNGKey(6), (7, 5))
    g = jr.normal(jr.PRNGKey(7), (9, 5))

    def loss_fp8(x, y):
        return jnp.sum(compute_dtype.gemm(x, y, fp8=True) * g)

    ga, gb = jax.grad(loss_fp8, argnums=(0, 1))(a, b)
    # Аналитическая форма VJP обычного ``a @ b`` (те же выражения, что в bwd).
    pa = g @ b.T
    pb = a.T @ g
    assert _bitwise_equal(ga, pa), "ga не совпал с аналитическим GEMM-градиентом"
    assert _bitwise_equal(gb, pb), "gb не совпал с аналитическим GEMM-градиентом"


def test_gate_off_and_fp8_false_are_the_legacy_path(monkeypatch):
    """``fp8=False`` (дефолт) под гейтом on — прежний путь; ``fp8=True`` — FP8."""
    a = jr.normal(jr.PRNGKey(8), (6, 4))
    b = jr.normal(jr.PRNGKey(9), (4, 3))
    _set_gate(monkeypatch, "1")
    assert _bitwise_equal(compute_dtype.gemm(a, b), a @ b)
    assert _bitwise_equal(compute_dtype.gemm(a, b, fp8=False), a @ b)
    assert not _bitwise_equal(compute_dtype.gemm(a, b, fp8=True), a @ b)


# --------------------------------------------------------------------------- #
# (г) композит KDA + MLA + MLP
# --------------------------------------------------------------------------- #


def _composite(cfg):
    """Параметры и вход композита; readout нормирует вклад каждого слоя."""
    x = jr.normal(jr.PRNGKey(1), (SEQ, cfg.hidden))
    params = (
        kda.init_kda(jr.PRNGKey(0), cfg),
        mla.init_mla(jr.PRNGKey(2), cfg),
        mlp.init_mlp(jr.PRNGKey(3), cfg),
    )
    tgt = (
        jr.normal(jr.PRNGKey(4), (SEQ, cfg.hidden)),
        jr.normal(jr.PRNGKey(5), (1, SEQ, cfg.hidden)),
        jr.normal(jr.PRNGKey(6), (1, SEQ, cfg.hidden)),
    )
    tgt = tuple(t / jnp.linalg.norm(t) for t in tgt)
    # Масштабы readout: выходы MLA/MLP на инициализации малы (латент → softmax,
    # затем квадратичный SiTU-GLU), поэтому каждый слой нормируется до O(1);
    # fp8-ошибка слоя при этом сохраняется относительной.
    weights = (1.0, 1e4, 1e10)

    def loss(pk, pm, pp):
        h = kda.apply_kda(pk, cfg, x, chunk_size=4)
        h2 = mla.apply(pm, cfg, h[None])
        o = mlp.apply(pp, cfg, h2)
        return (
            weights[0] * jnp.sum(h * tgt[0])
            + weights[1] * jnp.sum(h2 * tgt[1])
            + weights[2] * jnp.sum(o * tgt[2])
        )

    return params, loss


def test_composite_loss_matches_off(monkeypatch):
    """gate=on: лосс композита KDA+MLA+MLP равен off в допуске."""
    cfg = small_config()
    params, loss = _composite(cfg)
    _set_gate(monkeypatch, "0")
    v_off = loss(*params)
    _set_gate(monkeypatch, "1")
    v_on = loss(*params)
    rel = abs(float(v_on) - float(v_off)) / max(abs(float(v_off)), 1e-30)
    print(f"[цикл-26] composite loss: off={float(v_off):.6f} on={float(v_on):.6f} rel={rel:.3e}")
    assert rel <= COMPOSITE_TOL, f"composite loss rel={rel:.3e} > {COMPOSITE_TOL:.1e}"


def test_composite_two_steps_without_nan(monkeypatch):
    """gate=on: два шага SGD по композиту — без nan/inf.

    Градиенты сверяются с off в справочном допуске (fp8-ошибки проекций
    каскадируются; блокирующий критерий цикла — лосс и отсутствие nan).
    """
    cfg = small_config()
    params, loss = _composite(cfg)
    _set_gate(monkeypatch, "0")
    g_off = jax.grad(loss, argnums=(0, 1, 2))(*params)
    _set_gate(monkeypatch, "1")
    g_on = jax.grad(loss, argnums=(0, 1, 2))(*params)
    delta, scale, rel = _tree_relative(g_on, g_off)
    print(f"[цикл-26] composite grads: max|d|={delta:.3e} scale={scale:.3e} rel={rel:.3e}")
    assert rel <= COMPOSITE_GRAD_TOL, f"composite grads rel={rel:.3e}"

    p = params
    for step in range(2):
        value, grads = jax.value_and_grad(loss, argnums=(0, 1, 2))(*p)
        for tree in (value, grads):
            assert all(bool(jnp.isfinite(x).all()) for x in jax.tree_util.tree_leaves(tree)), (
                f"nan/inf на шаге {step}"
            )
        # Раздельное обновление деревьев: NamedTuple-типы сохраняются.
        p = tuple(
            jax.tree_util.tree_map(lambda a, b: a - 1e-3 * b, pt, gt)
            for pt, gt in zip(p, grads)
        )


# --------------------------------------------------------------------------- #
# (д) скейлы: нулевой/вырожденный тензор и отсутствие утечки в градиенты
# --------------------------------------------------------------------------- #


def test_zero_tensor_scales_do_not_produce_nan(monkeypatch):
    """max=0 → скейл ``1/448`` (не деление на ноль), forward/градиент конечны."""
    _set_gate(monkeypatch, "1")
    b = jr.normal(jr.PRNGKey(10), (8, 5))
    z = jnp.zeros((3, 8), jnp.float32)
    c = compute_dtype.gemm_fp8(z, b)
    gc = jax.grad(lambda a: jnp.sum(compute_dtype.gemm_fp8(a, b)), argnums=0)(z)
    assert bool(jnp.isfinite(c).all()) and float(jnp.max(jnp.abs(c))) == 0.0
    assert bool(jnp.isfinite(gc).all())

    zb = jnp.zeros((8, 5), jnp.float32)
    c2 = compute_dtype.gemm_fp8(z, zb)
    assert bool(jnp.isfinite(c2).all()) and float(jnp.max(jnp.abs(c2))) == 0.0


def test_degenerate_outlier_is_finite(monkeypatch):
    """Выброс 1e6 в операнде: скейл поглощает диапазон, nan не возникает."""
    _set_gate(monkeypatch, "1")
    b = jr.normal(jr.PRNGKey(11), (16, 8)) * 0.02
    a = jr.normal(jr.PRNGKey(12), (4, 16)) * 1e-6
    a = a.at[0, 0].set(1e6)
    c = compute_dtype.gemm_fp8(a, b)
    ga, gb = jax.grad(
        lambda x, y: jnp.sum(compute_dtype.gemm_fp8(x, y)), argnums=(0, 1)
    )(a, b)
    assert bool(jnp.isfinite(c).all())
    assert bool(jnp.isfinite(ga).all()) and bool(jnp.isfinite(gb).all())


def test_backward_has_no_reduce_max(monkeypatch):
    """Скейлы не текут в градиенты: в jaxpr backward нет ``reduce_max``.

    Forward-граф скейлы содержит (контроль, что шов действительно FP8), а
    правило backward — только два обычных матмула по ``g``; производная
    ``max|·|`` туда не попадает (STE).
    """
    a = jr.normal(jr.PRNGKey(13), (6, 8))
    b = jr.normal(jr.PRNGKey(14), (8, 4)) * 0.02
    g = jr.normal(jr.PRNGKey(15), (6, 4))

    fwd = jax.make_jaxpr(lambda x, y: compute_dtype.gemm_fp8(x, y))(a, b)
    assert "reduce_max" in _primitives(fwd), "forward не содержит скейлов"

    bwd = jax.make_jaxpr(
        lambda gg: compute_dtype._gemm_fp8_bwd((a, b), gg)
    )(g)
    names = _primitives(bwd)
    assert names, "backward-граф пуст — проверка потеряла зубы"
    assert "reduce_max" not in names, f"скейл потёк в backward: {names}"
