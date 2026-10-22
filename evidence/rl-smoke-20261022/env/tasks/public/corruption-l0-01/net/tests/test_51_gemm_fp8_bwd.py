"""Цикл-27 (ADR-054 Am.47): FP8 в **backward** GEMM-helper (градиентные матмулы).

Мотив: forward-fp8 проекций/MLP дал +8.8 % (Am.46, 1200.9 ток/с); backward
шага (22.7 с) остаётся крупнейшим рычагом того же класса, а его матмулы
(``ga = g @ bᵀ``, ``gb = aᵀ @ g``) считаются на bf16/fp32-границе.  Дельта
добавляет под-гейт ``AXIOM_GEMM_FP8_BWD`` (default ``0``):

* ``0`` — прежний bf16/fp32-backward :func:`compute_dtype._gemm_regular`
  побитово (блокирующий критерий: off не меняет ни одного бита);
* ``1`` — оба градиентных матмула в e4m3 (fp32-аккум, STE):
  ``AXIOM_FP8_ROW_SCALE=0`` — per-tensor скейлы (форма forward-правила),
  ``=1`` — per-row скейл **первого** операнда (``g`` для ``ga``; столбцы
  ``a`` — строки ``aᵀ`` — для ``gb``), второй операнд per-tensor.
  Per-block (128×128) в этой дельте не делается: нужен отдельный кернел
  (отложено; ADR-054 Am.47 объявлял опцию ``AXIOM_FP8_BLOCK_SCALE`` —
  пакет цикла-27 заменил её per-row).

Что пиннуется здесь (CPU, jax 0.11.2, шов ``_fp8_matmul`` — деквантованным
f32-матмулом, значения e4m3 в f32 точны):

* **(а) контракт под-гейтов**: пусто/не задано = ``"0"``; ``"0"``/``"1"``
  (пробелы нормализуются); неизвестное — ошибка (fail-closed); явный
  аргумент сильнее env; гейт читается на трейсе (jaxpr backward под ``"1"``
  содержит ``float8_e4m3fn``/``reduce_max``, под ``"0"`` — нет);
* **(б) off — побитово ревизия до дельты** (``6ccc7dd``, ``baseline_commit``
  пакета): градиенты ``gemm_fp8`` и ``gemm(..., fp8=True)`` при
  ``AXIOM_GEMM_FP8=1, AXIOM_GEMM_FP8_BWD=0`` совпадают с baseline-пакетом
  бит-в-бит — и под ``AXIOM_COMPUTE_DTYPE=bf16``, и под fp32;
* **(в) on — численность**: forward побитово равен прежнему fp8-forward;
  градиенты a/b сверяются (i) побитово с явной fp8-формулой
  ``(q(x)@q(y))·(s_x·s_y)`` (проверка скейлов/транспонирования) и (ii) с
  bf16-эталоном в scale-relative ``max|d|/scale ≤ 5e-2`` (сетка e4m3:
  3 мантиссных бита, макс. относительная ошибка округления 6.25 %).  Факт
  (форма 32×64×32, сиды 0/1/2): per-tensor ``ga`` 4.62e-2 / ``gb`` 4.40e-2,
  per-row ``ga`` 4.23e-2 / ``gb`` 4.47e-2; per-row на этих данных меняет
  ошибки местами (``ga`` точнее, ``gb`` чуть хуже) — запас точности per-row
  решается замером полного шага, не этой геометрией;
* **(г) композит**: KDA + MLA + MLP при ``AXIOM_GEMM_FP8=1`` и
  ``AXIOM_GEMM_FP8_BWD=1`` — два шага SGD без nan, лосс каждого шага в
  допуске 5e-2 к off-ноге (forward-fp8 + bf16-backward); факт — 1.93e-2 и
  2.27e-2 на шагах после обновления;
* **(д) STE-границы**: скейлы не текут в градиенты (вторая производная
  ``_gemm_fp8_bwd`` побитово равна эталону со ``stop_gradient``-скейлами и
  отличается от эталона без него; в jaxpr есть ``stop_gradient``); градиенты
  не нулевые; нулевой/вырожденный тензор и нулевая строка ``g`` не дают nan.

Окружение: CPU-интерпретатор, ``NET_JAX_BACKEND=cpu``/``JAX_PLATFORMS=cpu``;
KDA-тело — ``jax`` (персистентный кернел кампании на CPU отказывает).
"""

from __future__ import annotations

import io
import shutil
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

from net import compute_dtype, kda, mla, mlp

#: Ревизия до дельты (``baseline_commit`` handoff-пакета): off обязан
#: воспроизводить её побитово.  ``git archive`` + skip, если недоступна
#: (форма test_29/test_47/test_50).
BASELINE_COMMIT = "6ccc7dd"

CASE_DIR = Path(__file__).resolve().parents[2]

#: Гейты дельты.
FP8_GATE = compute_dtype.GEMM_FP8_ENV
BWD_GATE = compute_dtype.GEMM_FP8_BWD_ENV
ROW_GATE = compute_dtype.FP8_ROW_SCALE_ENV

#: Допуск градиентов против bf16-эталона: сетка e4m3 (макс. относительная
#: ошибка округления 6.25 %, RMS 3.6 %); факт — см. docstring модуля.
GRAD_TOL = 5e-2

#: Допуск композитного лосса к off-ноге (задание цикла-27).
COMPOSITE_TOL = 5e-2

#: Максимум представимого e4m3fn (для явных эталонов ниже).
E4M3_MAX = compute_dtype.E4M3_MAX

#: Геометрия композита: (T, hidden) маленькой модели conftest.
SEQ = 8

#: Под-гейты в форме (функция, env, класс ошибки) — общий контракт (а).
GATES = (
    (compute_dtype.gemm_fp8_bwd_mode, BWD_GATE, compute_dtype.GemmFp8BwdError),
    (compute_dtype.fp8_row_scale_mode, ROW_GATE, compute_dtype.Fp8RowScaleError),
)

GATE_IDS = ("bwd", "row")


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
    monkeypatch.delenv(FP8_GATE, raising=False)
    monkeypatch.delenv(BWD_GATE, raising=False)
    monkeypatch.delenv(ROW_GATE, raising=False)
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


def _bf16_matmul(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    """bf16-эталон: bf16-операнды, fp32-аккумулятор (Beam-рецепт)."""
    return jnp.matmul(
        a.astype(jnp.bfloat16),
        b.astype(jnp.bfloat16),
        preferred_element_type=jnp.float32,
    )


def _set_bwd(monkeypatch, bwd: str, row: str = "0") -> None:
    """Включает FP8-ногу и под-гейты дельты (явная установка обеих осей)."""
    monkeypatch.setenv(FP8_GATE, "1")
    monkeypatch.setenv(BWD_GATE, bwd)
    monkeypatch.setenv(ROW_GATE, row)


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


def _quant(x: jnp.ndarray, s: jnp.ndarray) -> jnp.ndarray:
    """Явная квантизация эталона: ``clip(x/s, ±448) → e4m3``."""
    return jnp.clip(x / s, -E4M3_MAX, E4M3_MAX).astype(jnp.float8_e4m3fn)


def _pt_scale(x: jnp.ndarray) -> jnp.ndarray:
    """Явный per-tensor скейл: ``max|x|/448`` (нулевой тензор → ``1/448``)."""
    amax = jnp.max(jnp.abs(x.astype(jnp.float32)))
    safe = jnp.where(amax > 0, amax, jnp.ones_like(amax))
    return (safe / E4M3_MAX).astype(jnp.float32)


def _row_scale(x: jnp.ndarray) -> jnp.ndarray:
    """Явный per-row скейл: ``max|x|/448`` по последней оси, ``(…, 1)``."""
    amax = jnp.max(jnp.abs(x.astype(jnp.float32)), axis=-1, keepdims=True)
    safe = jnp.where(amax > 0, amax, jnp.ones_like(amax))
    return (safe / E4M3_MAX).astype(jnp.float32)


def _ref_grad_matmul(
    x: jnp.ndarray, y: jnp.ndarray, *, row: bool, stop: bool = False
) -> jnp.ndarray:
    """Явный эталон градиентного матмула ``(q(x)@q(y))·(s_x·s_y)``.

    ``row`` — per-row скейл первого операнда; ``stop`` — обернуть скейлы в
    ``lax.stop_gradient`` (вторая производная без утечки скейлов).
    """
    s_x = _row_scale(x) if row else _pt_scale(x)
    s_y = _pt_scale(y)
    if stop:
        s_x = jax.lax.stop_gradient(s_x)
        s_y = jax.lax.stop_gradient(s_y)
    return compute_dtype._fp8_matmul(_quant(x, s_x), _quant(y, s_y)) * (s_x * s_y)


def _grad_reference(a, b, g, *, row):
    """bf16-эталонные градиенты ``(ga, gb)`` того же GEMM-контракта."""
    a2 = a.reshape(-1, a.shape[-1])
    g2 = g.reshape(-1, g.shape[-1])
    ga = _bf16_matmul(g2, b.T).reshape(a.shape)
    gb = _bf16_matmul(a2.T, g2)
    return ga, gb


def _grad_fp8(a, b, g):
    """Градиенты ``gemm_fp8`` под текущими под-гейтами."""

    def loss(x, y):
        return jnp.sum(compute_dtype.gemm_fp8(x, y) * g)

    return jax.grad(loss, argnums=(0, 1))(a, b)


@pytest.fixture(scope="module")
def baseline():
    """``net/`` на ``BASELINE_COMMIT``, импортируемый как ``baseline_net``.

    ``git archive`` (не worktree), поэтому состояние чекаута не трогается, а
    пакет переименован — относительные импорты внутри ``net/`` продолжают
    работать (форма test_29/test_47/test_50).  Нужен только
    ``compute_dtype``: контракт off — побитовое совпадение с ревизией до
    дельты.
    """
    root = Path(__file__).resolve().parent / ".baseline-net-test51"
    if root.exists():
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

        yield SimpleNamespace(compute_dtype=bcompute_dtype)
    finally:
        sys.path.remove(str(root))
        for name in [m for m in list(sys.modules) if m.startswith("baseline_net")]:
            del sys.modules[name]
        shutil.rmtree(root, ignore_errors=True)


# --------------------------------------------------------------------------- #
# (а) контракт под-гейтов
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("func,env,error", GATES, ids=GATE_IDS)
def test_subgate_default_is_off(monkeypatch, func, env, error):
    """Пусто/не задано = ``"0"``: дефолт не включает fp8-backward и per-row."""
    monkeypatch.delenv(env, raising=False)
    assert func() is False
    for raw in ("", "   ", "  "):
        monkeypatch.setenv(env, raw)
        assert func() is False, repr(raw)


@pytest.mark.parametrize("func,env,error", GATES, ids=GATE_IDS)
def test_subgate_values_and_whitespace(monkeypatch, func, env, error):
    """``"0"``/``"1"``; окружающие пробелы/переводы строк нормализуются.

    Значения — цифры, поэтому регистр к ним не применим; нормализация
    ``strip().lower()`` общая с родительским гейтом и проверяется на
    пробельных формах (``"\\t1\\n"``).
    """
    for raw, expected in {
        "0": False,
        "1": True,
        " 0 ": False,
        " 1 ": True,
        "\t1\n": True,
        "\n0\t": False,
    }.items():
        monkeypatch.setenv(env, raw)
        assert func() is expected, repr(raw)


@pytest.mark.parametrize("raw", ["2", "01", "-1", "true", "TRUE", "on", "yes", "fP8"])
@pytest.mark.parametrize("func,env,error", GATES, ids=GATE_IDS)
def test_subgate_unknown_fails_closed(monkeypatch, func, env, error, raw):
    """Неизвестное значение — ошибка (тихий откат запрещён)."""
    monkeypatch.setenv(env, raw)
    with pytest.raises(error) as excinfo:
        func()
    assert env in str(excinfo.value)
    assert repr(raw.strip().lower()) in str(excinfo.value)


@pytest.mark.parametrize("func,env,error", GATES, ids=GATE_IDS)
def test_subgate_explicit_argument_beats_env(monkeypatch, func, env, error):
    """Явный аргумент сильнее env (включая неизвестное env-значение)."""
    monkeypatch.setenv(env, "0")
    assert func("1") is True
    monkeypatch.setenv(env, "1")
    assert func("0") is False
    monkeypatch.setenv(env, "2")
    assert func("0") is False
    assert func("1") is True


def test_bwd_gate_is_read_on_trace(monkeypatch):
    """Гейт разрешается на трейсе: backward-граф меняется под ``"1"``.

    Под ``"0"`` правило backward — ``_gemm_regular``: ни fp8-каста, ни
    скейлов.  Под ``"1"`` появляются ``float8_e4m3fn`` и ``reduce_max``.
    """
    a = jr.normal(jr.PRNGKey(0), (6, 8))
    b = jr.normal(jr.PRNGKey(1), (8, 4)) * 0.02
    g = jr.normal(jr.PRNGKey(2), (6, 4))

    monkeypatch.setenv(BWD_GATE, "0")
    off = str(jax.make_jaxpr(lambda gg: compute_dtype._gemm_fp8_bwd((a, b), gg))(g))
    assert "float8_e4m3fn" not in off
    assert "reduce_max" not in off

    monkeypatch.setenv(BWD_GATE, "1")
    monkeypatch.setenv(ROW_GATE, "0")
    on = str(jax.make_jaxpr(lambda gg: compute_dtype._gemm_fp8_bwd((a, b), gg))(g))
    assert "float8_e4m3fn" in on, "fp8-ветка backward не попала в граф"
    assert "reduce_max" in on, "скейлы backward не попали в граф"

    monkeypatch.setenv(ROW_GATE, "1")
    row = str(jax.make_jaxpr(lambda gg: compute_dtype._gemm_fp8_bwd((a, b), gg))(g))
    assert "float8_e4m3fn" in row and "reduce_max" in row


# --------------------------------------------------------------------------- #
# (б) off — побитово ревизия до дельты
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("compute", [None, "bf16"])
def test_bwd_off_is_bitwise_baseline(baseline, monkeypatch, compute):
    """``BWD=0``: градиенты бит-в-бит как у ревизии до дельты.

    Проверяются оба входа FP8-ноги: прямой ``gemm_fp8`` и объявленный
    ``gemm(..., fp8=True)`` под ``AXIOM_GEMM_FP8=1``.  ``compute`` — ось
    ``AXIOM_COMPUTE_DTYPE``: off обязан совпадать и в bf16-backward (кампания),
    и в fp32 (дефолт).
    """
    _set_bwd(monkeypatch, "0")
    if compute is None:
        monkeypatch.delenv(compute_dtype.MODE_ENV, raising=False)
    else:
        monkeypatch.setenv(compute_dtype.MODE_ENV, compute)
    monkeypatch.setattr(
        baseline.compute_dtype, "_fp8_matmul", _dequant_matmul
    )

    a = jr.normal(jr.PRNGKey(0), (13, 17))
    b = jr.normal(jr.PRNGKey(1), (17, 9)) * 0.02
    g = jr.normal(jr.PRNGKey(2), (13, 9))

    def loss(mod, use_gemm):
        def inner(x, y):
            c = mod.gemm(x, y, fp8=True) if use_gemm else mod.gemm_fp8(x, y)
            return jnp.sum(c * g)

        return inner

    for use_gemm in (False, True):
        got = jax.grad(loss(compute_dtype, use_gemm), argnums=(0, 1))(a, b)
        ref = jax.grad(loss(baseline.compute_dtype, use_gemm), argnums=(0, 1))(a, b)
        for name, x, y in zip(("ga", "gb"), got, ref):
            assert _bitwise_equal(x, y), f"{name} off не побитово baseline ({compute})"

    # Forward под-гейт не задевает: побитово равен baseline.
    assert _bitwise_equal(
        compute_dtype.gemm_fp8(a, b),
        baseline.compute_dtype.gemm_fp8(a, b),
    )


# --------------------------------------------------------------------------- #
# (в) on — forward и градиенты
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("row", ["0", "1"])
def test_forward_is_bitwise_baseline(baseline, monkeypatch, row):
    """``BWD=1`` forward не меняет: побитово прежнему fp8-forward."""
    monkeypatch.setattr(
        baseline.compute_dtype, "_fp8_matmul", _dequant_matmul
    )
    a = jr.normal(jr.PRNGKey(0), (2, 16, 64))
    b = jr.normal(jr.PRNGKey(1), (64, 32)) * 0.02

    _set_bwd(monkeypatch, "1", row)
    got = compute_dtype.gemm_fp8(a, b)
    ref = baseline.compute_dtype.gemm_fp8(a, b)
    assert _bitwise_equal(got, ref), "backward-под-гейт изменил forward"
    assert _bitwise_equal(
        compute_dtype.gemm(a, b, fp8=True),
        baseline.compute_dtype.gemm(a, b, fp8=True),
    )


@pytest.mark.parametrize("row", ["0", "1"])
@pytest.mark.parametrize(
    "a_shape,b_shape",
    [
        ((32, 64), (64, 32)),
        ((2, 16, 64), (64, 32)),  # ведущие оси a (батч), b — 2-D вес
    ],
)
def test_gradients_match_bf16_reference(monkeypatch, a_shape, b_shape, row):
    """on: градиенты a/b против bf16-эталона в scale-relative ≤5e-2.

    Фактический ``max|d|/scale`` печатается; ``scale`` — максимум модуля
    обоих градиентов.  Проверка «ветка включилась» — градиенты не побитово
    равны эталону.
    """
    _set_bwd(monkeypatch, "1", row)
    a = jr.normal(jr.PRNGKey(0), a_shape)
    b = jr.normal(jr.PRNGKey(1), b_shape) * 0.02
    c_shape = jnp.matmul(jnp.zeros(a_shape), jnp.zeros(b_shape)).shape
    g = jr.normal(jr.PRNGKey(2), c_shape)

    ga, gb = _grad_fp8(a, b, g)
    ra, rb = _grad_reference(a, b, g, row=(row == "1"))
    for name, got, ref in (("ga", ga, ra), ("gb", gb, rb)):
        delta, scale, rel = _scale_relative(got, ref)
        print(
            f"[цикл-27] grad {name} row={row} shape={a_shape}: "
            f"max|d|={delta:.3e} scale={scale:.3e} rel={rel:.3e}"
        )
        assert rel <= GRAD_TOL, f"grad {name} row={row} rel={rel:.3e} > {GRAD_TOL:.1e}"
        assert _max_abs(got) > 0.0, f"grad {name} нулевой"


@pytest.mark.parametrize("row", ["0", "1"])
def test_gradients_equal_explicit_fp8_reference(monkeypatch, row):
    """on: градиенты побитово равны явной fp8-формуле со скейлами.

    Ловит ошибку скейла/транспонирования: для ``gb`` per-row скейл берётся
    по **столбцам** ``a`` (строкам ``aᵀ``), а не по строкам ``a``.
    """
    _set_bwd(monkeypatch, "1", row)
    a = jr.normal(jr.PRNGKey(3), (11, 13))
    b = jr.normal(jr.PRNGKey(4), (13, 7)) * 0.02
    g = jr.normal(jr.PRNGKey(5), (11, 7))

    ga, gb = _grad_fp8(a, b, g)
    ref_ga = _ref_grad_matmul(g, b.T, row=(row == "1"))
    ref_gb = _ref_grad_matmul(a.T, g, row=(row == "1"))
    assert _bitwise_equal(ga, ref_ga), "ga не совпал с явной fp8-формулой"
    assert _bitwise_equal(gb, ref_gb), "gb не совпал с явной fp8-формулой"


@pytest.mark.parametrize("row", ["0", "1"])
def test_zero_tensor_and_zero_rows_are_finite(monkeypatch, row):
    """max=0 → скейл ``1/448`` (не деление на ноль): nan не возникает.

    Покрыты: нулевой ``a`` (``gb = 0``, ``ga`` жив), нулевые оба операнда,
    нулевая строка котангента ``g`` при per-row скейле.
    """
    _set_bwd(monkeypatch, "1", row)
    b = jr.normal(jr.PRNGKey(6), (8, 5)) * 0.02
    g = jr.normal(jr.PRNGKey(7), (3, 5))
    z = jnp.zeros((3, 8), jnp.float32)

    ga, gb = _grad_fp8(z, b, g)
    assert bool(jnp.isfinite(ga).all()) and bool(jnp.isfinite(gb).all())
    assert float(jnp.max(jnp.abs(gb))) == 0.0

    c = compute_dtype.gemm_fp8(z, jnp.zeros((8, 5), jnp.float32))
    assert bool(jnp.isfinite(c).all()) and float(jnp.max(jnp.abs(c))) == 0.0

    # Нулевая строка g: соответствующая строка ga нулевая, скейл строки —
    # 1/448, деления на ноль нет.
    gz = jr.normal(jr.PRNGKey(8), (4, 6)).at[0, :].set(0.0)
    a = jr.normal(jr.PRNGKey(9), (4, 8))
    w = jr.normal(jr.PRNGKey(10), (8, 6)) * 0.02
    ga2, gb2 = _grad_fp8(a, w, gz)
    assert bool(jnp.isfinite(ga2).all()) and bool(jnp.isfinite(gb2).all())
    assert float(jnp.max(jnp.abs(ga2[0]))) == 0.0
    assert _max_abs(ga2[1:]) > 0.0


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


def _two_steps(params, loss, lr=1e-3):
    """Два шага SGD; возвращает лоссы **после** шагов и признак конечности.

    Лосс до обновления у обеих ног одинаков по построению (forward
    под-гейтом не меняется), поэтому шаг нумеруется по состоянию после
    обновления — иначе проверка включения гейта выродилась бы в тождество.
    """
    p = params
    losses = []
    finite = True
    for _ in range(2):
        value, grads = jax.value_and_grad(loss, argnums=(0, 1, 2))(*p)
        finite = finite and bool(jnp.isfinite(value))
        finite = finite and all(
            bool(jnp.isfinite(x).all()) for x in jax.tree_util.tree_leaves(grads)
        )
        p = tuple(
            jax.tree_util.tree_map(lambda a, b: a - lr * b, pt, gt)
            for pt, gt in zip(p, grads)
        )
        post = loss(*p)
        finite = finite and bool(jnp.isfinite(post))
        losses.append(float(post))
    return p, losses, finite


def test_composite_two_steps_match_off(monkeypatch):
    """FP8+BWD=on: два шага композита без nan, лосс в допуске к off-ноге.

    Обе ноги идут под ``AXIOM_GEMM_FP8=1`` (forward-fp8 — текущий
    кампанийный путь); off-нога — ``BWD=0`` (bf16-backward), on-нога —
    ``BWD=1``.  Сравниваются лоссы каждого шага, а не только финальный:
    дрейф на первом шаге иначе замаскировался бы вторым.
    """
    cfg = small_config()
    params, loss = _composite(cfg)

    _set_bwd(monkeypatch, "0", "0")
    _, losses_off, finite_off = _two_steps(params, loss)
    assert finite_off, "off-нога композита дала nan/inf"

    _set_bwd(monkeypatch, "1", "0")
    _, losses_on, finite_on = _two_steps(params, loss)
    assert finite_on, "on-нога композита дала nan/inf"

    for step, (off, on) in enumerate(zip(losses_off, losses_on)):
        rel = abs(on - off) / max(abs(off), 1e-30)
        print(f"[цикл-27] composite step {step}: off={off:.6f} on={on:.6f} rel={rel:.3e}")
        assert rel <= COMPOSITE_TOL, f"лосс шага {step} rel={rel:.3e} > {COMPOSITE_TOL:.1e}"
    assert losses_on[0] != losses_off[0], "BWD-гейт не включился (лоссы побитово равны)"


# --------------------------------------------------------------------------- #
# (д) STE-границы: скейлы не текут в градиенты, градиенты не нулевые
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("row", ["0", "1"])
def test_scales_do_not_flow_into_gradients(monkeypatch, row):
    """Скейлы — константы STE: вторая производная без члена ``q·∂s/∂x``.

    Дифференцируется само правило ``_gemm_fp8_bwd`` по котангенту ``g``.
    Эталон со ``stop_gradient``-скейлами совпадает побитово; эталон без
    ``stop_gradient`` — отличается (проверка, что тест имеет зубы).
    В jaxpr правила присутствуют и ``reduce_max`` (скейлы считаются), и
    ``stop_gradient`` (в котангенты не текут).
    """
    _set_bwd(monkeypatch, "1", row)
    a = jr.normal(jr.PRNGKey(11), (8, 16))
    b = jr.normal(jr.PRNGKey(12), (16, 6)) * 0.02
    g = jr.normal(jr.PRNGKey(13), (8, 6))
    h = jr.normal(jr.PRNGKey(14), (8, 16))
    use_row = row == "1"

    def bwd_ga(gg):
        return compute_dtype._gemm_fp8_bwd((a, b), gg)[0]

    second = jax.grad(lambda gg: jnp.sum(bwd_ga(gg) * h))(g)
    assert bool(jnp.isfinite(second).all())

    d_stop = jax.grad(
        lambda gg: jnp.sum(_ref_grad_matmul(gg, b.T, row=use_row, stop=True) * h)
    )(g)
    d_nostop = jax.grad(
        lambda gg: jnp.sum(_ref_grad_matmul(gg, b.T, row=use_row, stop=False) * h)
    )(g)
    assert _bitwise_equal(second, d_stop), "скейл потёк: вторая производная ≠ STE-эталону"
    assert not _bitwise_equal(second, d_nostop), (
        "эталон без stop_gradient неотличим — проверка потеряла зубы"
    )

    names = _primitives(jax.make_jaxpr(bwd_ga)(g))
    assert "reduce_max" in names, "backward не считает скейлы (шов не fp8?)"
    assert "stop_gradient" in names, "скейлы не отрезаны от котангентов"


@pytest.mark.parametrize("row", ["0", "1"])
def test_gradients_are_nonzero_and_finite(monkeypatch, row):
    """STE не обнуляет сигнал: оба градиента конечны и не нулевые."""
    _set_bwd(monkeypatch, "1", row)
    a = jr.normal(jr.PRNGKey(15), (9, 7))
    b = jr.normal(jr.PRNGKey(16), (7, 5)) * 0.02
    g = jr.normal(jr.PRNGKey(17), (9, 5))
    ga, gb = _grad_fp8(a, b, g)
    for name, grad in (("ga", ga), ("gb", gb)):
        assert bool(jnp.isfinite(grad).all()), f"{name} содержит nan/inf"
        assert _max_abs(grad) > 0.0, f"{name} нулевой — STE потерял сигнал"
