"""Гейт KDA-интервала MTP-слоя: ``AXIOM_MTP_KDA_IMPL`` (ADR-054 Am.32, цикл-20).

MTP-слой считал KDA старым jnp-путём ``apply_chunked`` — 256-графовый хвост
9.75 с GPU/шага (провенанс-карта цикла-19).  Цикл-20 вводит гейт:

* ``chunked`` (дефолт) — прежний прямой :func:`net.kda.apply_chunked`:
  **побитово** прежние выход и градиенты MTP-параметров и прежний jaxpr;
* ``campaign`` — :func:`net.kda.apply_kda` (диспетчеризация по
  ``cfg.kda_impl`` = ``chunked_cc``): кампанийный путь, снимающий хвост;
* пусто/не задано = ``chunked``; регистр и пробелы нормализуются; неизвестное
  значение — :class:`net.mtp.MtpKdaImplError` (fail-closed, форма
  ``net.mla.persist_mode``); явный аргумент гейта сильнее env; значение
  читается на трейсе (не кэшируется).

Допуск кампанийной ноги (тест c).  Сравнение идёт с off-ногой (побитовым
прежним путём) при ``AXIOM_KDA_CC_IMPL=jax`` — на CPU персистентный кернел
отказывает по построению.  Метрика — scale-relative ``max|d| / scale`` в
форме ``_max_rel`` тестов test_35/39/43: ``max|d|`` берётся по всем листьям,
``scale`` — максимум модуля референса (для выхода — по тензору, для
градиентов — общий масштаб дерева; иначе near-zero KDA-листья с градиентом
~1e-5 дают неограниченное отношение при абсолютной разнице того же порядка).
Фактически на сиде 0 (B=2, T=32, C=16): выход ``max|d|=4.77e-7`` при
``scale=3.51`` (``max|d|/scale=1.4e-7``); градиенты ``max|d|=1.73e-5`` при
``scale=2.03e2`` (``max|d|/scale=8.5e-8``).  Вторая страховка — построчный
кампанийный ``allclose(rtol=2e-2, atol=2e-3)`` (форма test_26): фактический
абсолютный ``max|d|`` по всем листьям ``1.73e-5 <= 2e-3``.  Порог приёмки —
``2e-2`` (кампанийный допуск bf16-операндов KDA).

Окружение прогона: CPU-интерпретатор jax 0.11.2,
``NET_JAX_BACKEND=cpu JAX_PLATFORMS=cpu``; тело ``chunked_cc`` — ``jax``.
"""

from __future__ import annotations

import ast
import dataclasses
import inspect
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

from conftest import small_config

from net import compute_dtype, kda, mlp, mtp
from net.config import ModelConfig
from net.norm import rms_norm

#: Геометрия прогона (маленькая модель conftest; B=2, T=32, C=16).
BATCH = 2
SEQ = 32
CHUNK = 16

#: Кампанийный допуск bf16-операндов KDA (ADR-054 Am.32).
ATOL_SCALE = 2e-2

#: Абсолютный пол кампанийного допуска (форма существующих тестов).
ATOL_ABS = 2e-3


@pytest.fixture(autouse=True)
def _campaign_bodies(monkeypatch):
    """CPU-исполнение: тело ``chunked_cc`` — ссылочный ``jax``, гейт MTP снят.

    Персистентный кернел кампании (``AXIOM_KDA_CC_IMPL=persistent`` по
    умолчанию) на CPU отказывает по построению, поэтому тесты нового пути
    выбирают ``jax`` явно — форма ``test_kda_chunked_cc.py``.  Гейт MTP по
    умолчанию снят; тесты, которым нужна кампанийная нога, ставят его сами.
    """
    monkeypatch.setenv("AXIOM_KDA_CC_IMPL", "jax")
    monkeypatch.delenv(mtp.MTP_KDA_IMPL_ENV, raising=False)


def _cfg(**overrides) -> ModelConfig:
    """Малый конфиг с кампанийным ``kda_impl`` (``apply_kda`` → ``chunked_cc``)."""
    base = dict(kda_impl="chunked_cc")
    base.update(overrides)
    return dataclasses.replace(small_config(), **base)


def _params(cfg: ModelConfig, *, seed: int = 0) -> mtp.MTPParams:
    return mtp.init_mtp(jr.PRNGKey(seed), cfg)


def _inputs(cfg: ModelConfig, *, seed: int = 1):
    """``(h, next_emb)`` формы (B, T, hidden)."""
    k1, k2 = jr.split(jr.PRNGKey(seed))
    return (
        jr.normal(k1, (BATCH, SEQ, cfg.hidden)),
        jr.normal(k2, (BATCH, SEQ, cfg.hidden)),
    )


def _reference_apply(
    params: mtp.MTPParams,
    cfg: ModelConfig,
    h: jnp.ndarray,
    next_emb: jnp.ndarray,
    chunk_size: int,
) -> jnp.ndarray:
    """Дословная прежняя ветка ``mtp.apply`` (KDA — прямой ``apply_chunked``).

    Повторяет тело :func:`net.mtp.apply` до гейта: тест b доказывает, что при
    ``chunked`` публичный путь даёт побитово те же выход, градиенты и jaxpr,
    что и этот прямой вызов.
    """
    fused = compute_dtype.gemm(
        jnp.concatenate([h, next_emb], axis=-1), params.W_f
    )
    z = rms_norm(fused, params.norm_in)
    attn = jax.vmap(
        lambda xb: kda.apply_chunked(params.kda, cfg, xb, chunk_size)
    )(z)
    z = z + attn
    z = z + mlp.apply(params.mlp, cfg, rms_norm(z, params.norm_mlp))
    return rms_norm(z, params.norm_out)


def _run(params, cfg, h, next_emb, chunk_size):
    """``(loss, выход)`` одной ветки: градиент — по ``sum(out**2)``."""
    out = mtp.apply(params, cfg, h, next_emb, chunk_size)
    return jnp.sum(out * out), out


def _value_grad_out(params, cfg, h, next_emb, chunk_size):
    """``(loss, выход, градиенты MTP-параметров)`` за один backward."""
    (loss, out), grads = jax.value_and_grad(
        lambda p: _run(p, cfg, h, next_emb, chunk_size),
        has_aux=True,
    )(params)
    return loss, out, grads


def _leaves(tree):
    return jax.tree_util.tree_leaves(tree)


def _max_abs(tree) -> float:
    return max(float(jnp.max(jnp.abs(leaf))) for leaf in _leaves(tree))


def _max_diff(a, b) -> float:
    return max(
        float(jnp.max(jnp.abs(x - y))) for x, y in zip(_leaves(a), _leaves(b))
    )


def _scale_relative(a, b) -> tuple[float, float, float]:
    """``(max|d|, scale, max|d|/scale)``; scale — максимум модуля обоих операндов."""
    d = _max_diff(a, b)
    scale = max(
        max(float(jnp.max(jnp.abs(x))) for x in _leaves(a)),
        max(float(jnp.max(jnp.abs(y))) for y in _leaves(b)),
    )
    return d, scale, (d / scale if scale else 0.0)


# ---------------------------------------------------------------------------
# a. гейт: значения, нормализация, fail-closed, приоритет явного аргумента
# ---------------------------------------------------------------------------


def test_gate_default_and_accepted_values(monkeypatch):
    """Пусто/не задано = ``chunked``; регистр и пробелы нормализуются."""
    monkeypatch.delenv(mtp.MTP_KDA_IMPL_ENV, raising=False)
    assert mtp.mtp_kda_impl() == "chunked"

    cases = {
        "": "chunked",
        "   ": "chunked",
        "chunked": "chunked",
        "Chunked": "chunked",
        " campaign ": "campaign",
        "CAMPAIGN": "campaign",
    }
    for raw, expected in cases.items():
        monkeypatch.setenv(mtp.MTP_KDA_IMPL_ENV, raw)
        assert mtp.mtp_kda_impl() == expected, raw


@pytest.mark.parametrize(
    "raw", ["campain", "campaign2", "chunky", "on", "1", "persistent", "jax"]
)
def test_gate_unknown_fails_closed(monkeypatch, raw):
    """Неизвестное значение — ``MtpKdaImplError`` (тихая подмена запрещена)."""
    monkeypatch.setenv(mtp.MTP_KDA_IMPL_ENV, raw)
    with pytest.raises(mtp.MtpKdaImplError) as excinfo:
        mtp.mtp_kda_impl()
    assert mtp.MTP_KDA_IMPL_ENV in str(excinfo.value)
    assert repr(raw.strip().lower()) in str(excinfo.value)


def test_unknown_gate_raises_before_any_trace(monkeypatch):
    """Ошибка гейта возникает до диспетчеризации и до какого-либо jax-трейса.

    Обе ветки-цели подменены «ловушками»: если бы гейт читался после входа в
    KDA, тест увидел бы ``AssertionError``, а не ``MtpKdaImplError``.  Аргументы
    ``None`` допустимы только потому, что гейт падает раньше обращения к ним.
    """

    def _dispatch_reached(*args, **kwargs):
        raise AssertionError("KDA dispatch reached after an unknown gate value")

    monkeypatch.setenv(mtp.MTP_KDA_IMPL_ENV, "campain")  # опечатка
    monkeypatch.setattr(mtp.kda_mod, "apply_kda", _dispatch_reached)
    monkeypatch.setattr(mtp.kda_mod, "apply_chunked", _dispatch_reached)
    with pytest.raises(mtp.MtpKdaImplError):
        mtp._kda_interval(None, None, None, CHUNK)


def test_public_apply_fails_closed_on_unknown_gate(monkeypatch):
    """``mtp.apply`` с неизвестным значением — ошибка гейта, не тихий путь."""
    cfg = _cfg()
    params = _params(cfg)
    h, next_emb = _inputs(cfg)
    monkeypatch.setenv(mtp.MTP_KDA_IMPL_ENV, "campaign ")  # хвостовой пробел ок
    assert mtp.mtp_kda_impl() == "campaign"
    monkeypatch.setenv(mtp.MTP_KDA_IMPL_ENV, "typo")
    with pytest.raises(mtp.MtpKdaImplError):
        mtp.apply(params, cfg, h, next_emb, CHUNK)


def test_explicit_argument_beats_env(monkeypatch):
    """Явный аргумент сильнее env, включая env с неизвестным значением."""
    monkeypatch.setenv(mtp.MTP_KDA_IMPL_ENV, "campaign")
    assert mtp.mtp_kda_impl("chunked") == "chunked"

    monkeypatch.setenv(mtp.MTP_KDA_IMPL_ENV, "chunked")
    assert mtp.mtp_kda_impl("campaign") == "campaign"

    monkeypatch.setenv(mtp.MTP_KDA_IMPL_ENV, "typo")
    assert mtp.mtp_kda_impl(" CAMPAIGN ") == "campaign"


def test_dispatch_routing_and_trace_time_reading(monkeypatch):
    """Маршрутизация по гейту и чтение env на трейсе (без кэша)."""
    calls: list[tuple[str, int]] = []

    def _fake_chunked(params, cfg, xb, chunk_size):
        calls.append(("chunked", chunk_size))
        return jnp.zeros((1,))

    def _fake_dispatch(params, cfg, xb, chunk_size):
        calls.append(("apply_kda", chunk_size))
        return jnp.zeros((1,))

    monkeypatch.setattr(mtp.kda_mod, "apply_chunked", _fake_chunked)
    monkeypatch.setattr(mtp.kda_mod, "apply_kda", _fake_dispatch)

    monkeypatch.setenv(mtp.MTP_KDA_IMPL_ENV, "campaign")
    mtp._kda_interval(None, None, None, CHUNK)
    assert calls == [("apply_kda", CHUNK)]

    monkeypatch.setenv(mtp.MTP_KDA_IMPL_ENV, "chunked")
    mtp._kda_interval(None, None, None, CHUNK)
    assert calls[-1] == ("chunked", CHUNK)

    # Гейт читается на трейсе: смена env между вызовами меняет ветку.
    monkeypatch.setenv(mtp.MTP_KDA_IMPL_ENV, "campaign")
    mtp._kda_interval(None, None, None, CHUNK)
    assert calls[-1] == ("apply_kda", CHUNK)

    # Явный аргумент сильнее env и на маршрутизации.
    mtp._kda_interval(None, None, None, CHUNK, impl="chunked")
    assert calls[-1] == ("chunked", CHUNK)


def test_apply_routes_through_kda_interval(monkeypatch):
    """``mtp.apply`` проводит KDA-интервал только через ``_kda_interval``."""
    cfg = _cfg()
    params = _params(cfg)
    h, next_emb = _inputs(cfg)
    seen: list[tuple[int, ...]] = []
    original = mtp._kda_interval

    def _spy(params_, cfg_, xb, chunk_size, impl=None):
        seen.append(tuple(xb.shape))
        return original(params_, cfg_, xb, chunk_size, impl)

    monkeypatch.setattr(mtp, "_kda_interval", _spy)
    out = mtp.apply(params, cfg, h, next_emb, CHUNK)
    assert out.shape == (BATCH, SEQ, cfg.hidden)
    assert seen == [(SEQ, cfg.hidden)]


# ---------------------------------------------------------------------------
# b. off: побитово прежняя ветка (выход, градиенты, jaxpr)
# ---------------------------------------------------------------------------


def test_off_is_bitwise_previous_branch(monkeypatch):
    """``chunked`` == прямой ``kda_mod.apply_chunked``: выход/градиенты/jaxpr."""
    monkeypatch.delenv(mtp.MTP_KDA_IMPL_ENV, raising=False)
    cfg = _cfg()
    params = _params(cfg)
    h, next_emb = _inputs(cfg)

    out_gate = mtp.apply(params, cfg, h, next_emb, CHUNK)
    out_ref = _reference_apply(params, cfg, h, next_emb, CHUNK)
    assert jnp.array_equal(out_gate, out_ref), "MTP output is not bit-for-bit"

    def _loss_gate(p):
        return jnp.sum(mtp.apply(p, cfg, h, next_emb, CHUNK) ** 2)

    def _loss_ref(p):
        return jnp.sum(_reference_apply(p, cfg, h, next_emb, CHUNK) ** 2)

    v_gate, g_gate = jax.value_and_grad(_loss_gate)(params)
    v_ref, g_ref = jax.value_and_grad(_loss_ref)(params)
    assert bool(v_gate == v_ref), "MTP loss is not bit-for-bit"
    for i, (a, b) in enumerate(zip(_leaves(g_gate), _leaves(g_ref))):
        assert jnp.array_equal(a, b), f"gradient leaf {i} is not bit-for-bit"

    # Блокирующий критерий цикла-20: при off граф — прежний.
    assert str(jax.make_jaxpr(_loss_gate)(params)) == str(
        jax.make_jaxpr(_loss_ref)(params)
    ), "jaxpr of the off path differs from the pre-gate branch"


# ---------------------------------------------------------------------------
# c. on: кампанийная нога в допуске 2e-2 (scale-relative)
# ---------------------------------------------------------------------------


def test_campaign_matches_off_within_campaign_tolerance(monkeypatch):
    """``campaign`` == ``chunked`` в допуске 2e-2 по выходу и градиентам.

    Фактические числа сида 0 — в docstring модуля: выход ``max|d|/scale =
    1.4e-7``, градиенты ``8.5e-8``; абсолютный ``max|d| = 1.73e-5 <= 2e-3``.
    """
    cfg = _cfg()
    params = _params(cfg)
    h, next_emb = _inputs(cfg)

    monkeypatch.setenv(mtp.MTP_KDA_IMPL_ENV, "chunked")
    _, out_off, g_off = _value_grad_out(params, cfg, h, next_emb, CHUNK)

    monkeypatch.setenv(mtp.MTP_KDA_IMPL_ENV, "campaign")
    _, out_on, g_on = _value_grad_out(params, cfg, h, next_emb, CHUNK)

    d_out, s_out, r_out = _scale_relative(out_off, out_on)
    assert r_out <= ATOL_SCALE, f"output max|d|={d_out:.3e} scale={s_out:.3e}"

    d_g, s_g, r_g = _scale_relative(g_off, g_on)
    assert r_g <= ATOL_SCALE, f"grads max|d|={d_g:.3e} scale={s_g:.3e}"

    # Вторая, абсолютная страховка: near-zero листья не должны «проскочить»
    # только за счёт общего масштаба дерева.
    for i, (a, b) in enumerate(zip(_leaves(g_off), _leaves(g_on))):
        assert jnp.allclose(a, b, rtol=ATOL_SCALE, atol=ATOL_ABS), (
            f"gradient leaf {i}: max|d|={_max_diff(a, b):.3e}"
        )


# ---------------------------------------------------------------------------
# d. гейт не меняет другие call-site (KDA backbone в model.py не тронут)
# ---------------------------------------------------------------------------


def _kda_calls(node: ast.AST) -> list[ast.Call]:
    """Вызовы ``kda_mod.apply_kda`` / ``kda_mod.apply_chunked`` в поддереве."""
    found = []
    for child in ast.walk(node):
        if (
            isinstance(child, ast.Call)
            and isinstance(child.func, ast.Attribute)
            and child.func.attr in ("apply_kda", "apply_chunked")
            and isinstance(child.func.value, ast.Name)
            and child.func.value.id == "kda_mod"
        ):
            found.append(child)
    return found


def test_apply_kda_is_called_only_from_kda_interval():
    """В ``net/mtp.py`` обе KDA-ветки вызываются только из ``_kda_interval``."""
    source = inspect.getsource(mtp)
    tree = ast.parse(source)
    intervals = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_kda_interval"
    ]
    assert len(intervals) == 1, "MTP gate must have exactly one _kda_interval"
    inside = _kda_calls(intervals[0])
    assert sorted(call.func.attr for call in inside) == ["apply_chunked", "apply_kda"]

    inside_ids = {id(call) for call in inside}
    outside = [call for call in _kda_calls(tree) if id(call) not in inside_ids]
    assert outside == [], (
        "KDA calls outside _kda_interval in net/mtp.py: "
        f"{[call.func.attr for call in outside]}"
    )

    # ``apply`` не обращается к kda_mod напрямую — только к гейту.
    apply_fn = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "apply"
    )
    assert _kda_calls(apply_fn) == []
    names = {
        call.func.id
        for call in ast.walk(apply_fn)
        if isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
    }
    assert "_kda_interval" in names


def test_model_kda_call_site_is_untouched_by_the_gate():
    """Backbone KDA в ``net/model.py`` не читает MTP-гейт и не переписан."""
    model_path = Path(__file__).resolve().parents[1] / "model.py"
    source = model_path.read_text(encoding="utf-8")
    assert source.count("kda_mod.apply_kda") == 1
    assert "MTP_KDA_IMPL" not in source
    assert "AXIOM_MTP_KDA_IMPL" not in source
    assert "apply_chunked" not in source


# ---------------------------------------------------------------------------
# e. nan/конечность кампанийной ноги на нескольких сидах
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_campaign_is_finite_on_several_seeds(monkeypatch, seed):
    """Выход, loss и градиенты кампанийной ноги конечны на сидах 0/1/2."""
    cfg = _cfg()
    params = _params(cfg, seed=seed)
    h, next_emb = _inputs(cfg, seed=seed + 11)
    monkeypatch.setenv(mtp.MTP_KDA_IMPL_ENV, "campaign")
    loss, out, grads = _value_grad_out(params, cfg, h, next_emb, CHUNK)
    assert bool(jnp.isfinite(loss))
    assert bool(jnp.all(jnp.isfinite(out)))
    assert all(bool(jnp.all(jnp.isfinite(leaf))) for leaf in _leaves(grads))
