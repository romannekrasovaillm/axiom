"""Цикл-23 (ADR-054 Am.35/36): fused backward чанкованного CE — гейт ``AXIOM_CE_BWD``.

Пофазовый профиль B=4 показал ``sec_bwd_ce`` 2.267 с при ~14 % пика: autodiff
чанкованного CE идёт через per-chunk ``jax.checkpoint`` и transpose-правила
``log_softmax``/gather.  Цикл-23 вводит явный :func:`jax.custom_vjp` вокруг
всей свёртки (:func:`net.model._chunked_ce_fused`): forward — дословно тот же
chunked-цикл, backward — пересчёт логитов по чанку и аналитический softmax-CE
градиент.

Что пиннуется здесь (CPU, jax 0.11.2):

* **(а) контракт гейта**: пусто/не задано = ``"jax"``; ``"fused"``; регистр и
  пробелы нормализуются; неизвестное — :class:`net.model.CeBwdError` до
  трейса; явный аргумент сильнее env; значение читается на трейсе;
* **(б) off — побитово ревизия до дельты** (``bd2d321``, ``baseline_commit``
  пакета): прямой вызов ``_chunked_cross_entropy`` (значение и градиенты
  features/embedding) и jaxpr графа совпадают с извлечённым ``net/``
  baseline — оба режима ``AXIOM_CE_BF16``; блокирующий критерий цикла;
* **(в) on — forward побитово равен off**, градиенты — scale-relative
  ``max|d|/scale ≤ 2e-2`` (кампанийный допуск CE bf16) на нескольких
  сидах/формах, включая хвостовой неполный чанк, ``loss_mask`` и bf16-операнды;
  фактический ``max|d|/scale`` печатается;
* **(г) память**: в backward-графе нет полного ``(B, T, vocab)`` буфера и нет
  ``remat``-границы; логиты живут только чанком ``(B, chunk, vocab)`` —
  пин на jaxpr pullback'а и ``lower().as_text()``;
* **(д) композит**: ``compute_loss`` (NTP + MTP) при ``fused`` — лосс побитово
  равен ``jax``, градиенты параметров в допуске.

Окружение: CPU-интерпретатор, ``NET_JAX_BACKEND=cpu``/``JAX_PLATFORMS=cpu``;
KDA-тело — ``jax`` (персистентный кернел кампании на CPU отказывает).
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
import numpy as np
import pytest

from conftest import tiny_config

from net import model

#: Ревизия до дельты (``baseline_commit`` handoff-пакета): off обязан
#: воспроизводить её побитово, включая jaxpr.  ``git archive`` + skip, если
#: ревизия недоступна (форма test_29/test_39).
BASELINE_COMMIT = "bd2d321"

CASE_DIR = Path(__file__).resolve().parents[2]

#: Кампанийный допуск CE bf16 (ADR-054 Am.35/36): scale-relative ``max|d|``.
BWD_TOL = 2e-2


# --------------------------------------------------------------------------- #
# Общие помощники
# --------------------------------------------------------------------------- #


@pytest.fixture(autouse=True)
def _cpu_bodies(monkeypatch):
    """CPU-исполнение: KDA — ``jax``; гейты цикла-23/вехи-9 сняты по умолчанию."""
    monkeypatch.setenv("AXIOM_KDA_CC_IMPL", "jax")
    monkeypatch.delenv(model.CE_BWD_ENV, raising=False)
    monkeypatch.delenv(model.CE_BF16_ENV, raising=False)


def _case(seed: int = 3, B: int = 2, T: int = 13, V: int = 64, H: int = 16):
    """Детерминированные ``(features, targets, embedding)`` для CE-ноги."""
    key = jr.PRNGKey(seed)
    feats = jr.normal(key, (B, T, H))
    targets = jr.randint(key, (B, T), 1, V)
    emb = jr.normal(jr.PRNGKey(seed + 1), (V, H))
    return feats, targets, emb


def _mask(B: int, T: int, real: int) -> jnp.ndarray:
    """Маска ``(B, T)``: первые ``real`` позиций — реальные, хвост — паддинг."""
    return jnp.zeros((B, T), jnp.float32).at[:, :real].set(1.0)


def _value_and_grads(feats, targets, emb, chunk, loss_mask=None):
    """``(loss, grad_features, grad_embedding)`` одной ноги."""

    def loss(f_, e_):
        return model._chunked_cross_entropy(
            f_, targets, e_, chunk, loss_mask=loss_mask
        )

    value, (grad_f, grad_e) = jax.value_and_grad(loss, argnums=(0, 1))(feats, emb)
    return value, grad_f, grad_e


def _max_abs(x) -> float:
    return float(jnp.max(jnp.abs(x)))


def _scale_relative(on, off) -> tuple[float, float, float]:
    """``(max|d|, scale, max|d|/scale)``; scale — максимум модуля обоих операндов."""
    delta = _max_abs(on - off)
    scale = max(_max_abs(on), _max_abs(off))
    return delta, scale, (delta / scale if scale else 0.0)


def _set_gate(monkeypatch, value: str | None) -> None:
    if value is None:
        monkeypatch.delenv(model.CE_BWD_ENV, raising=False)
    else:
        monkeypatch.setenv(model.CE_BWD_ENV, value)


# --------------------------------------------------------------------------- #
# (а) контракт гейта
# --------------------------------------------------------------------------- #


def test_gate_default_and_normalisation(monkeypatch):
    """Пусто/не задано = ``jax``; регистр и окружающие пробелы нормализуются."""
    monkeypatch.delenv(model.CE_BWD_ENV, raising=False)
    assert model.ce_bwd_mode() == model.CE_BWD_JAX

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
        monkeypatch.setenv(model.CE_BWD_ENV, raw)
        assert model.ce_bwd_mode() == expected, raw


@pytest.mark.parametrize("raw", ["fusion", "fuzed", "on", "1", "0", "auto", "jax2"])
def test_gate_unknown_fails_closed(monkeypatch, raw):
    """Неизвестное значение — :class:`CeBwdError` (тихий откат запрещён)."""
    monkeypatch.setenv(model.CE_BWD_ENV, raw)
    with pytest.raises(model.CeBwdError) as excinfo:
        model.ce_bwd_mode()
    assert model.CE_BWD_ENV in str(excinfo.value)
    assert repr(raw.strip().lower()) in str(excinfo.value)


def test_unknown_gate_raises_before_any_trace(monkeypatch):
    """Ошибка гейта — до чанкования и до какого-либо jax-трейса.

    ``_chunked_ce_reduce`` подменён ловушкой: если бы гейт читался после входа
    в цикл, тест увидел бы ``AssertionError``, а не ``CeBwdError``.  ``None``
    в аргументах допустим только потому, что гейт падает раньше их разбора.
    """

    def _reached(*_args, **_kwargs):
        raise AssertionError("chunk loop reached after an unknown gate value")

    monkeypatch.setattr(model, "_chunked_ce_reduce", _reached)
    monkeypatch.setenv(model.CE_BWD_ENV, "fuzed")
    with pytest.raises(model.CeBwdError):
        model._chunked_cross_entropy(None, None, None, 0)


def test_explicit_argument_beats_env(monkeypatch):
    """Явный аргумент сильнее env, включая env с неизвестным значением."""
    monkeypatch.setenv(model.CE_BWD_ENV, "fused")
    assert model.ce_bwd_mode("jax") == "jax"

    monkeypatch.setenv(model.CE_BWD_ENV, "jax")
    assert model.ce_bwd_mode("fused") == "fused"

    monkeypatch.setenv(model.CE_BWD_ENV, "typo")
    assert model.ce_bwd_mode(" FUSED ") == "fused"


def test_explicit_argument_selects_the_path(monkeypatch):
    """``ce_bwd`` управляет веткой: fused — custom_vjp, jax — remat-цикл.

    ``jax.grad`` инлайнит правило ``custom_vjp`` в джахпр (примитива
    ``custom_vjp_call`` в тексте нет), поэтому маршрут пинуется вызовом
    фабрики, а отсутствие remat-границы — по тексту графа.
    """
    feats, targets, emb = _case()
    calls: list[int] = []
    real_factory = model._chunked_ce_fused

    def _spy(chunk):
        calls.append(chunk)
        return real_factory(chunk)

    monkeypatch.setattr(model, "_chunked_ce_fused", _spy)

    def _graph(ce_bwd: str | None):
        def loss(f_, e_):
            return model._chunked_cross_entropy(
                f_, targets, e_, 5, ce_bwd=ce_bwd
            )

        return str(jax.make_jaxpr(jax.grad(loss, argnums=(0, 1)))(feats, emb))

    monkeypatch.setenv(model.CE_BWD_ENV, "jax")
    fused_text = _graph("fused")
    assert calls == [5], "явный ce_bwd=fused обязан вызвать fused-фабрику"
    assert "remat" not in fused_text, "fused-ветка не должна содержать remat"

    calls.clear()
    monkeypatch.setenv(model.CE_BWD_ENV, "fused")
    jax_text = _graph("jax")
    assert calls == [], "явный ce_bwd=jax не должен трогать fused-фабрику"
    assert "remat" in jax_text, "явный ce_bwd=jax обязан оставить remat-путь"


# --------------------------------------------------------------------------- #
# (б) off — побитово ревизия до дельты (блокирующий критерий)
# --------------------------------------------------------------------------- #

#: Пробник для ревизии до дельты: значения/градиенты в хешах + jaxpr.
_PROBE = '''
import hashlib
import json
import os

import numpy as np
import jax
import jax.numpy as jnp
from net import model

def digest(x):
    a = np.asarray(x)
    return {"shape": list(a.shape), "dtype": str(a.dtype),
            "sha": hashlib.sha256(a.tobytes()).hexdigest()}

rng = np.random.default_rng(7)
B, T, V, H = 2, 13, 64, 16
feats = jnp.asarray(rng.normal(size=(B, T, H)), dtype=jnp.float32)
targets = jnp.asarray(rng.integers(1, V, size=(B, T)), dtype=jnp.int32)
emb = jnp.asarray(rng.normal(size=(V, H)), dtype=jnp.float32)
mask = jnp.asarray((rng.random((B, T)) > 0.2).astype(np.float32))

out = {}
for name, kw in (("plain", {}), ("mask", {"loss_mask": mask})):
    def loss(f_, e_):
        return model._chunked_cross_entropy(f_, targets, e_, 5, **kw)
    value = loss(feats, emb)
    grad_f, grad_e = jax.grad(loss, argnums=(0, 1))(feats, emb)
    out[name] = {"v": digest(value), "gf": digest(grad_f), "ge": digest(grad_e)}

def loss_masked(f_, e_):
    return model._chunked_cross_entropy(f_, targets, e_, 5, loss_mask=mask)

out["jaxpr"] = str(jax.make_jaxpr(jax.grad(loss_masked, argnums=(0, 1)))(
    feats, emb))
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
        if key not in {
            model.CE_BWD_ENV,
            model.CE_BF16_ENV,
            "AXIOM_COMPUTE_DTYPE",
            "PYTHONPATH",
        }
    }
    env.update(
        {
            "NET_JAX_BACKEND": "cpu",
            "JAX_PLATFORMS": "cpu",
            "PYTHONPATH": str(root),
        }
    )
    env.update(extra_env or {})
    done = subprocess.run(
        [sys.executable, str(script)], cwd=root, env=env, capture_output=True, text=True
    )
    if done.returncode != 0:  # pragma: no cover — диагностика прогона
        raise AssertionError(f"проба упала ({root}):\n{done.stderr[-2000:]}")
    lines = [line for line in done.stdout.strip().splitlines() if line.strip()]
    return json.loads(lines[-1])


def test_off_path_is_bitwise_the_baseline(tmp_path):
    """``jax``-ветка — побитово ревизия до дельты: числа, градиенты и jaxpr.

    Сверяются оба режима ``AXIOM_CE_BF16`` (0 и 1): дельта касается только
    выбора backward, поэтому ни одна из прежних ног не имеет права сдвинуться.
    """
    baseline = _extract_baseline(tmp_path)
    script = tmp_path / "probe.py"
    script.write_text(_PROBE, encoding="utf-8")

    for extra in ({}, {model.CE_BF16_ENV: "1"}):
        base_out = _run_probe(baseline, script, extra_env=extra)
        current_out = _run_probe(CASE_DIR, script, extra_env=extra)
        assert current_out == base_out, (
            "off-путь разошёлся с ревизией до дельты "
            f"(AXIOM_CE_BF16={extra.get(model.CE_BF16_ENV, '<unset>')})"
        )


# --------------------------------------------------------------------------- #
# (в) on: forward побитово, градиенты в кампанийном допуске
# --------------------------------------------------------------------------- #

_CASES = [
    # (seed, B, T, chunk, real_rows_for_mask | None)
    (3, 2, 13, 8, None),   # хвостовой неполный чанк (13 = 8 + 5)
    (11, 3, 10, 4, None),  # неполный хвост 10 = 4 + 4 + 2
    (5, 2, 13, 5, 11),     # маска через границу чанка, хвост 3
    (7, 1, 9, 2, 7),       # батч 1, маска, много чанков
]


@pytest.mark.parametrize("seed,B,T,chunk,real", _CASES)
def test_fused_forward_is_bitwise_and_grads_within_tolerance(
    monkeypatch, seed, B, T, chunk, real
):
    """Forward ``fused`` == ``jax`` побитово; grads — ``max|d|/scale ≤ 2e-2``."""
    feats, targets, emb = _case(seed=seed, B=B, T=T)
    mask = None if real is None else _mask(B, T, real)

    _set_gate(monkeypatch, None)
    v_off, gf_off, ge_off = _value_and_grads(feats, targets, emb, chunk, mask)
    _set_gate(monkeypatch, "fused")
    v_on, gf_on, ge_on = _value_and_grads(feats, targets, emb, chunk, mask)

    assert jnp.array_equal(v_off, v_on), (
        f"forward fused разошёлся с jax: {v_off!r} vs {v_on!r}"
    )

    for name, on, off in (("features", gf_on, gf_off), ("embedding", ge_on, ge_off)):
        delta, scale, rel = _scale_relative(on, off)
        print(
            f"[цикл-23] seed={seed} B={B} T={T} chunk={chunk} mask={real}: "
            f"{name} max|d|={delta:.3e} scale={scale:.3e} rel={rel:.3e}"
        )
        assert rel <= BWD_TOL, (
            f"градиент {name} разошёлся сильнее {BWD_TOL:g} (rel={rel:.3e})"
        )


def test_fused_grads_close_under_bf16_operands(monkeypatch):
    """Тот же допуск для bf16-операндов (кампанийный режим CE bf16).

    Сравнение — с autodiff того же bf16-пути (не с fp32): допуск меряет
    пересчёт логитов, а не округление bf16-операндов.
    """
    feats, targets, emb = _case(seed=13, B=2, T=11)
    feats = feats.astype(jnp.bfloat16)
    emb = emb.astype(jnp.bfloat16)

    _set_gate(monkeypatch, None)
    v_off, gf_off, ge_off = _value_and_grads(feats, targets, emb, 4)
    _set_gate(monkeypatch, "fused")
    v_on, gf_on, ge_on = _value_and_grads(feats, targets, emb, 4)

    assert jnp.array_equal(v_off, v_on)
    for name, on, off in (("features", gf_on, gf_off), ("embedding", ge_on, ge_off)):
        assert on.dtype == off.dtype, f"{name}: dtype {on.dtype} != {off.dtype}"
        delta, scale, rel = _scale_relative(
            on.astype(jnp.float32), off.astype(jnp.float32)
        )
        print(
            f"[цикл-23] bf16-операнды: {name} max|d|={delta:.3e} "
            f"scale={scale:.3e} rel={rel:.3e}"
        )
        assert rel <= BWD_TOL, (
            f"bf16-градиент {name} разошёлся сильнее {BWD_TOL:g} (rel={rel:.3e})"
        )


# --------------------------------------------------------------------------- #
# (г) память: живой тензор — один чанк, полного (B, T, vocab) нет
# --------------------------------------------------------------------------- #


def _collect_shapes(obj, out=None, depth=0):
    """Все aval-формы дерева jaxpr, включая вложенные jaxpr'ы в params.

    Спуск обязателен: ``custom_vjp_call`` держит тело правила в ``params``, а
    ``remat`` — в теле вызова; верхний уровень их не видит.  Защита от
    самоссылки ``Jaxpr.jaxpr is Jaxpr`` (jax >= 0.11) — как в test_25.
    """
    if out is None:
        out = []
    if depth > 30 or obj is None:
        return out
    if isinstance(obj, (str, bytes, int, float, bool)):
        return out
    inner = getattr(obj, "jaxpr", None)
    if (
        inner is not None
        and inner is not obj
        and not isinstance(obj, (dict, list, tuple))
    ):
        _collect_shapes(inner, out, depth + 1)
    eqns = getattr(obj, "eqns", None)
    if eqns is not None:
        for var in list(getattr(obj, "invars", [])) + list(
            getattr(obj, "outvars", [])
        ):
            aval = getattr(var, "aval", None)
            if aval is not None:
                out.append(tuple(getattr(aval, "shape", ())))
        for eqn in eqns:
            for var in eqn.outvars:
                aval = getattr(var, "aval", None)
                if aval is not None:
                    out.append(tuple(getattr(aval, "shape", ())))
            for value in eqn.params.values():
                _collect_shapes(value, out, depth + 1)
        return out
    if isinstance(obj, dict):
        for value in obj.values():
            _collect_shapes(value, out, depth + 1)
    elif isinstance(obj, (list, tuple)):
        for value in obj:
            _collect_shapes(value, out, depth + 1)
    return out


def test_fused_backward_keeps_only_one_chunk(monkeypatch):
    """В backward нет полного ``(B, T, vocab)`` и нет ``remat``-границы.

    Пины: (1) pullback-джахпр — состав остатков и живых тензоров; ни одной
    формы ``(B, T, V)`` с ``T > chunk``; пик логитов ≤ ``B·chunk·V``;
    (2) ``lower().as_text()`` — полного буфера нет и в HLO, чанковые формы
    присутствуют; (3) в fused-backward нет ``remat``/``checkpoint``.
    """
    B, T, V, H, chunk = 2, 13, 64, 16, 5
    feats, targets, emb = _case(seed=3, B=B, T=T, V=V, H=H)
    mask = _mask(B, T, 11)
    monkeypatch.setenv(model.CE_BWD_ENV, "fused")

    def loss(f_, e_):
        return model._chunked_cross_entropy(
            f_, targets, e_, chunk, loss_mask=mask
        )

    _, pullback = jax.vjp(loss, feats, emb)
    closed = jax.make_jaxpr(pullback)(jnp.ones(()))
    text = str(closed)

    shapes = _collect_shapes(closed)
    full = [
        shape
        for shape in shapes
        if len(shape) == 3
        and shape[0] == B
        and shape[-1] == V
        and shape[1] > chunk
    ]
    assert not full, f"в backward сохранён полный логит-тензор: {full!r}"

    logits_shapes = [
        shape for shape in shapes if len(shape) == 3 and shape[0] == B and shape[-1] == V
    ]
    assert logits_shapes, "в backward нет ни одного чанкового логит-тензора"
    peak = max(math.prod(shape) for shape in logits_shapes)
    assert peak <= B * chunk * V, (
        f"живой логит-тензор {peak} больше чанка {B * chunk * V}"
    )
    assert "remat" not in text and "checkpoint" not in text, (
        "fused-ветка не должна содержать jax.checkpoint/remat"
    )

    lowered = jax.jit(jax.grad(loss, argnums=(0, 1))).lower(feats, emb).as_text()
    full_tag = f"tensor<{B}x{T}x{V}x"
    assert full_tag not in lowered, (
        f"HLO содержит полный (B, T, vocab)-буфер {full_tag}"
    )
    for width in (chunk, T % chunk or chunk):
        chunk_tag = f"tensor<{B}x{width}x{V}x"
        assert chunk_tag in lowered, f"HLO не содержит чанкового буфера {chunk_tag}"


# --------------------------------------------------------------------------- #
# (д) композит: compute_loss (NTP + MTP) при fused
# --------------------------------------------------------------------------- #


def test_composite_compute_loss_with_mtp(monkeypatch):
    """Маленькая модель: лосс ``fused`` == ``jax`` побитово, градиенты в допуске."""
    cfg = tiny_config(ce_chunk_tokens=8)
    params = model.init_params(jr.PRNGKey(0), cfg)
    ids = jr.randint(jr.PRNGKey(2), (2, 16), 1, cfg.vocab_size)

    def loss(p_):
        return model.compute_loss(p_, cfg, ids, chunk_size=16)

    _set_gate(monkeypatch, None)
    v_off, grads_off = jax.value_and_grad(loss)(params)
    _set_gate(monkeypatch, "fused")
    v_on, grads_on = jax.value_and_grad(loss)(params)

    assert jnp.array_equal(v_off, v_on), (
        f"композитный лосс разошёлся: {v_off!r} vs {v_on!r}"
    )

    leaves_off = jax.tree_util.tree_leaves(grads_off)
    leaves_on = jax.tree_util.tree_leaves(grads_on)
    assert len(leaves_off) == len(leaves_on)
    delta = max(_max_abs(on - off) for on, off in zip(leaves_on, leaves_off))
    scale = max([1e-30] + [_max_abs(leaf) for leaf in leaves_off + leaves_on])
    rel = delta / scale
    print(
        f"[цикл-23] композит NTP+MTP: max|d|={delta:.3e} scale={scale:.3e} "
        f"rel={rel:.3e}"
    )
    assert rel <= BWD_TOL, f"градиенты композита разошлись: rel={rel:.3e}"

    # Боевой путь кампании — jit: гейт читается на трейсе, поэтому каждая нога
    # получает СВЕЖУЮ функцию (jit кэширует граф по идентичности функции).
    def _jit_leg(gate: str | None):
        _set_gate(monkeypatch, gate)

        def _loss(p_):
            return model.compute_loss(p_, cfg, ids, chunk_size=16)

        return jax.jit(jax.value_and_grad(_loss))(params)

    v_off_j, grads_off_j = _jit_leg(None)
    v_on_j, grads_on_j = _jit_leg("fused")
    assert jnp.array_equal(v_off_j, v_on_j), "jit-лосс fused разошёлся с jax"
    leaves_off_j = jax.tree_util.tree_leaves(grads_off_j)
    leaves_on_j = jax.tree_util.tree_leaves(grads_on_j)
    delta_j = max(
        _max_abs(on - off) for on, off in zip(leaves_on_j, leaves_off_j)
    )
    scale_j = max(
        [1e-30] + [_max_abs(leaf) for leaf in leaves_off_j + leaves_on_j]
    )
    rel_j = delta_j / scale_j
    print(
        f"[цикл-23] jit-композит: max|d|={delta_j:.3e} scale={scale_j:.3e} "
        f"rel={rel_j:.3e}"
    )
    assert rel_j <= BWD_TOL, f"jit-градиенты композита разошлись: rel={rel_j:.3e}"
