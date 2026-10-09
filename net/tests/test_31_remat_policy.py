"""Политика рематериализации ``jax.checkpoint`` (ADR-049, ``remat_policy``).

Диагностика шага l3-full показала, что 73% времени уходит в backward, а
grad-checkpointing отключить нельзя (OOM 956 ГиБ по активациям).  Значит вопрос
не «пересчитывать ли», а **что** пересчитывать: ``jax.checkpoint`` без политики
(save-nothing) пересчитывает всё, а ``dots_saveable`` /
``dots_with_no_batch_dims_saveable`` оставляют сохранёнными выходы matmul.

Что пиннуют эти тесты:

* поле схемы ``remat_policy`` с дефолтом ``none`` и fail-closed валидацией
  значения (неизвестное имя — ошибка, а не молчаливый ``none``);
* политика доходит до remat-примитивов ``net/model.py`` (backbone-слой) и
  ``net/kda.py`` (тела scan) — объявление без механизма было бы AD-9-дефектом;
* политика не добавляет и не убирает remat-границы (их число задаётся
  ``grad_ckpt_policy``) — меняется только ``policy`` внутри границы;
* **численный паритет**: loss/градиенты совпадают между ``none`` и обеими
  политиками (политика меняет место активаций, не математику);
* ``none`` воспроизводит поведение **до дельты побитово**: граф строится
  вызовом ``jax.checkpoint(fn)`` без аргумента ``policy``, и сравнение с этой
  конструкцией (подменённой в модулях-потребителях) даёт побитово равные loss
  и градиенты;
* политика инертна там, где remat-границ нет (``grad_ckpt_policy="none"`` без
  ``kda_chunked_backward``) — выключенная механика не платит за поле.

Матрица «политика → время/память» на GB10 — прогоны архитектора; CPU-числа
снимает ``tools/remat_policy_smoke.py`` (``evidence/kda-rewrite/remat-policy-smoke.json``).
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

from conftest import small_config

from net import kda, model
from net.config import REMAT_POLICIES, load_config, validate_config
from net.remat import remat_checkpoint, remat_policy_fn

#: Число слоёв малой геометрии — это же число per-layer remat-границ.
NUM_LAYERS = 4

#: Декларация кейса (``net/config.json``) — файл-переключатель (spine AD-9).
CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.json"

#: Все объявленные политики; ``none`` — дефолт (проверяется отдельно).
POLICIES = ("none", "dots_saveable", "dots_with_no_batch_dims_saveable")

#: Значение до дельты: ``jax.checkpoint`` БЕЗ аргумента ``policy``.  Тест
#: подменяет этим модульную ссылку и требует побитового совпадения — то же
#: утверждение, что «дефолт не двигается», но против кода, а не против значения.
def _pre_delta_checkpoint(fn, policy: str = "none"):
    del policy  # политика рематериализации до ADR-049 не передавалась вовсе
    return jax.checkpoint(fn)


def _remat_config(**overrides):
    """Конфиг, при котором remat-границы **реально стоят**.

    ``per_layer`` ставит границу на каждый backbone-слой, ``chunked_cc`` +
    ``kda_chunked_backward`` — ещё и на тело scan внутри KDA.  Обе точки ADR-049
    (``net/model.py`` и ``net/kda.py``) оказываются в одном графе.
    """
    base = dict(
        grad_ckpt_policy="per_layer",
        kda_impl="chunked_cc",
        kda_chunked_backward=True,
    )
    base.update(overrides)
    return dataclasses.replace(small_config(), **base)


def _problem(cfg, remat_policy=None):
    """(params, loss-функция) — один сид, детерминированные входы."""
    key = jr.PRNGKey(0)
    params = model.init_params(key, cfg)
    ids = jr.randint(key, (2, 32), 0, cfg.vocab_size)

    def loss(p):
        return model.compute_loss(
            p, cfg, ids, chunk_size=8, remat_policy=remat_policy
        )

    return params, loss


def _remat_policy_params(cfg, remat_policy: str) -> list:
    """Параметр ``policy`` каждого remat-примитива в трассированном графе."""
    params, loss = _problem(cfg, remat_policy)
    closed = jax.make_jaxpr(loss)(params)
    return [
        eqn.params.get("policy")
        for eqn in closed.jaxpr.eqns
        # JAX нумерует версии примитива (``remat2`` на jax 0.10.x): матчим
        # семейство, а не точное имя (как в test_24).
        if str(eqn.primitive.name).startswith("remat")
    ]


def _bitwise_equal(a, b) -> bool:
    leaves_a = jax.tree_util.tree_leaves(a)
    leaves_b = jax.tree_util.tree_leaves(b)
    if len(leaves_a) != len(leaves_b):
        return False
    return all(bool(jnp.array_equal(x, y)) for x, y in zip(leaves_a, leaves_b))


def _tree_allclose(a, b, *, rtol=2e-2, atol=2e-3) -> bool:
    leaves_a = jax.tree_util.tree_leaves(a)
    leaves_b = jax.tree_util.tree_leaves(b)
    if len(leaves_a) != len(leaves_b):
        return False
    return all(
        bool(jnp.allclose(x, y, rtol=rtol, atol=atol, equal_nan=True))
        for x, y in zip(leaves_a, leaves_b)
    )


# ---------------------------------------------------------------------------
# имя → политика JAX: закрытый список, fail-closed
# ---------------------------------------------------------------------------


def test_only_none_means_no_policy_argument():
    assert remat_policy_fn("none") is None
    assert remat_policy_fn("dots_saveable") is jax.checkpoint_policies.dots_saveable
    assert (
        remat_policy_fn("dots_with_no_batch_dims_saveable")
        is jax.checkpoint_policies.dots_with_no_batch_dims_saveable
    )


def test_unknown_policy_name_raises_fail_closed():
    with pytest.raises(ValueError, match="неизвестная политика"):
        remat_policy_fn("dots_save_everything_please")
    # пустая строка и ``None`` — тоже не «дефолт», а неизвестное значение
    for bad in ("", "None", "NONE", "none "):
        with pytest.raises(ValueError):
            remat_policy_fn(bad)


def test_policy_list_is_closed_and_matches_schema():
    assert POLICIES == REMAT_POLICIES
    assert len(set(REMAT_POLICIES)) == len(REMAT_POLICIES)


def test_schema_default_is_none():
    """Конфиг, собранный в коде (тесты, смоуки), остаётся на прежнем графе."""
    assert small_config().remat_policy == "none"


def test_case_config_does_not_turn_the_policy_on():
    """Декларация кейса обязана остаться ``none`` — дефолт не двигается."""
    assert load_config(CONFIG_PATH).remat_policy == "none"


def test_unknown_policy_rejected_by_validate_config():
    cfg = dataclasses.replace(small_config(), remat_policy="sometimes")
    with pytest.raises(AssertionError, match="remat_policy"):
        validate_config(cfg)


def test_unknown_policy_raises_even_with_boundaries_off():
    """Fail-closed не зависит от того, включена ли механика сейчас.

    ``grad_ckpt_policy="none"`` не строит ни одной remat-границы — но
    объявленное и неизвестное значение обязано падать сразу, иначе оно оживёт
    молча в тот день, когда границы включат.
    """
    cfg = dataclasses.replace(small_config(), grad_ckpt_policy="none")
    params, loss = _problem(cfg, "nope")
    with pytest.raises(ValueError, match="неизвестная политика"):
        loss(params)


def test_unknown_policy_raises_in_kda_scan_body():
    cfg = _remat_config(remat_policy="nope")
    params = kda.init_kda(jr.PRNGKey(0), cfg)
    x = jr.normal(jr.PRNGKey(1), (16, cfg.hidden))
    with pytest.raises(ValueError, match="неизвестная политика"):
        kda.apply_chunked_cc(params, cfg, x, chunk_size=8)


# ---------------------------------------------------------------------------
# механизм: политика доходит до remat-примитивов и не меняет их число
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("scan_layers", [True, False])
def test_policy_reaches_every_remat_boundary(scan_layers):
    """Каждая граница несёт объявленную политику — и в scan-, и в unrolled-пути.

    Оба пути ``net/model.py`` (тело group-scan и развёрнутый цикл) обязаны
    применять политику согласованно: «объявлено, но не доходит» — это AD-9-дефект.
    """
    cfg = _remat_config(scan_layers=scan_layers)
    assert _remat_policy_params(cfg, "none") == [None] * NUM_LAYERS
    assert (
        _remat_policy_params(cfg, "dots_saveable")
        == [jax.checkpoint_policies.dots_saveable] * NUM_LAYERS
    )
    assert _remat_policy_params(cfg, "dots_with_no_batch_dims_saveable") == [
        jax.checkpoint_policies.dots_with_no_batch_dims_saveable
    ] * NUM_LAYERS


def test_policy_does_not_change_the_number_of_boundaries():
    """Гранулярность задаёт ``grad_ckpt_policy``; политика — только содержимое."""
    cfg = _remat_config()
    counts = {
        policy: len(_remat_policy_params(cfg, policy)) for policy in POLICIES
    }
    assert set(counts.values()) == {NUM_LAYERS}


def test_policy_is_inert_without_boundaries():
    """Нет границ — нечего настраивать: граф одинаков при любой политике."""
    cfg = dataclasses.replace(
        small_config(), grad_ckpt_policy="none", kda_chunked_backward=False
    )
    graphs = {}
    for policy in POLICIES:
        params, loss = _problem(cfg, policy)
        graphs[policy] = str(jax.make_jaxpr(loss)(params))
    assert len(set(graphs.values())) == 1


def test_remat_checkpoint_none_is_plain_checkpoint():
    """``none`` — это ровно ``jax.checkpoint(fn)``, а не «то же самое на глаз»."""
    f = lambda x: jnp.sin(x) @ jnp.cos(x)  # noqa: E731 — проба политики, не модель
    x = jr.normal(jr.PRNGKey(0), (4, 4))
    plain = str(jax.make_jaxpr(jax.checkpoint(f))(x))
    none = str(jax.make_jaxpr(remat_checkpoint(f, "none"))(x))
    dots = str(
        jax.make_jaxpr(remat_checkpoint(f, "dots_saveable"))(x)
    )
    assert plain == none
    assert dots != none  # политика действительно меняет remat-примитив


# ---------------------------------------------------------------------------
# паритет: политика меняет место активаций, не математику
# ---------------------------------------------------------------------------


def test_loss_parity_across_policies():
    cfg = _remat_config()
    values = {}
    for policy in POLICIES:
        params, loss = _problem(cfg, policy)
        values[policy] = loss(params)
    assert _tree_allclose(values["none"], values["dots_saveable"], rtol=1e-5, atol=1e-6)
    assert _tree_allclose(
        values["none"], values["dots_with_no_batch_dims_saveable"], rtol=1e-5, atol=1e-6
    )


def test_grad_parity_across_policies():
    cfg = _remat_config()
    grads = {}
    for policy in POLICIES:
        params, loss = _problem(cfg, policy)
        grads[policy] = jax.grad(loss)(params)
    assert _tree_allclose(grads["none"], grads["dots_saveable"])
    assert _tree_allclose(
        grads["none"], grads["dots_with_no_batch_dims_saveable"]
    )


def test_forward_parity_across_policies():
    cfg = _remat_config()
    key = jr.PRNGKey(2)
    params = model.init_params(key, cfg)
    ids = jr.randint(key, (2, 32), 0, cfg.vocab_size)
    outputs = {
        policy: model.forward(
            params, cfg, ids, chunk_size=8, remat_policy=policy
        )
        for policy in POLICIES
    }
    assert _tree_allclose(outputs["none"], outputs["dots_saveable"], rtol=1e-5, atol=1e-6)
    assert _tree_allclose(
        outputs["none"], outputs["dots_with_no_batch_dims_saveable"], rtol=1e-5, atol=1e-6
    )


def test_grads_survive_jit_for_every_policy():
    """Тренировочный путь: ``jit(value_and_grad)`` на каждой политике."""
    cfg = _remat_config()
    results = {}
    for policy in POLICIES:
        params, loss = _problem(cfg, policy)
        results[policy] = jax.jit(jax.value_and_grad(loss))(params)
    for policy in ("dots_saveable", "dots_with_no_batch_dims_saveable"):
        assert _tree_allclose(results["none"][0], results[policy][0], rtol=1e-5, atol=1e-6)
        assert _tree_allclose(results["none"][1], results[policy][1])


# ---------------------------------------------------------------------------
# «none» = поведение до дельты, побитово
# ---------------------------------------------------------------------------


def test_none_is_bitwise_the_pre_delta_graph(monkeypatch):
    """``none`` не двигает дефолт: сравнение с ``jax.checkpoint(fn)`` (до ADR-049).

    Подменяем модульные ссылки на ``remat_checkpoint`` конструкцией без
    аргумента ``policy`` — это буквально прежний код — и требуем побитового
    равенства loss и всех листьев градиента, а не ``allclose``.
    """
    cfg = _remat_config()
    params, loss = _problem(cfg, "none")
    reference_loss = loss(params)
    reference_grads = jax.grad(loss)(params)

    monkeypatch.setattr(model, "remat_checkpoint", _pre_delta_checkpoint)
    monkeypatch.setattr(kda, "remat_checkpoint", _pre_delta_checkpoint)
    _, pre_delta_loss = _problem(cfg, "none")
    assert _bitwise_equal(reference_loss, pre_delta_loss(params))
    assert _bitwise_equal(reference_grads, jax.grad(pre_delta_loss)(params))


def test_default_field_is_bitwise_the_explicit_none():
    """Отсутствие поля и явное ``none`` — один и тот же граф и те же биты."""
    cfg = _remat_config()
    key = jr.PRNGKey(3)
    params = model.init_params(key, cfg)
    ids = jr.randint(key, (2, 48), 0, cfg.vocab_size)
    implicit = model.compute_loss(params, cfg, ids, chunk_size=8)
    explicit = model.compute_loss(
        params, cfg, ids, chunk_size=8, remat_policy="none"
    )
    assert jnp.array_equal(implicit, explicit)


def test_policies_do_not_touch_the_chunked_ce_path():
    """Chunked CE — не граница ADR-049: политика его не настраивает.

    Причина: сохранять там matmul-выход (``dots_saveable``) значило бы держать
    ровно тот ``(chunk, vocab)`` тензор, ради отказа от которого чекинг и
    сделан.  Проверяем, что граф CE-пути одинаков при всех политиках.
    """
    cfg = dataclasses.replace(
        _remat_config(), ce_chunk_tokens=8, grad_ckpt_policy="none"
    )
    key = jr.PRNGKey(4)
    params = model.init_params(key, cfg)
    ids = jr.randint(key, (2, 32), 0, cfg.vocab_size)
    graphs = {
        policy: str(
            jax.make_jaxpr(
                lambda p: model.compute_loss(
                    p, cfg, ids, chunk_size=8, remat_policy=policy
                )
            )(params)
        )
        for policy in POLICIES
    }
    assert len(set(graphs.values())) == 1
