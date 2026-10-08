"""ADR-048 — классификация параметров по группам оптимизатора.

После переписывания KDA (ADR-047) шаг оптимизатора стал доминантой шага
(``sec_backopt`` = 20.40 с из ~31 с), и его узкое место — Muon/NS5, который
применялся к *любому* 2-D листу, включая связанные embeddings/LM head
(160000x1536).  ADR-048 выводит их из Muon в AdamW и требует, чтобы выбор
группы делался явным предикатом по именам листьев, а не по одной размерности.

Что пинуется здесь (CPU, без GPU):

* **полнота классификации на реальной модели** ``l3-full`` — ни один 2-D лист
  не остаётся неклассифицированным, а разбиение 32 имён на группы совпадает с
  зафиксированным (переименование листа в модели валит этот тест — цена,
  объявленная в ADR-048);
* **embeddings/LM head действительно AdamW** — по группе И по поведению
  (обновление совпадает с формулой AdamW и расходится с Muon);
* **скрытые матрицы остались Muon**, per-head и batched-ветки сохранены
  (per-head — по поведению ``per_head_newtonschulz5``);
* **fail-closed** на неизвестном 2-D имени (ошибка + запись в отчёт), при этом
  1-D и ndim>=3 листья по-прежнему классифицируются по умолчанию
  (AdamW-вектор / batched-Muon);
* **legacy-флаг** возвращает прежнюю классификацию («любой ndim==2 -> Muon»);
* ``ns_steps`` параметризован, дефолт 5 не сдвинут;
* оптимизатор jit-совместим и loss-шаг не падает.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

from net import model, optimizer

ROOT = Path(__file__).resolve().parents[2]
L3_CONFIG = ROOT / "net" / "config.json"

#: Разбиение 2-D листьев l3-full по группам — спецификация ADR-048 в виде
#: данных.  Ключ — группа, значение — множество имён листьев модели.
EXPECTED_2D_GROUPS = {
    optimizer.GROUP_MUON_PER_HEAD: {"W_q", "W_k", "W_v", "W_k_up", "W_v_up"},
    optimizer.GROUP_ADAMW_EMBED: {"embedding"},
    optimizer.GROUP_MUON_MATRIX: {
        "W_o", "W_g", "W_a_down", "W_a_up", "W_beta",
        "conv_q", "conv_k", "conv_v",
        "W_c", "W_idx_k", "W_idx_q", "W_swa_k", "W_swa_v",
        "W_up", "W_down", "W_u", "router_w", "W_f",
        "w", "patch_embed", "pos_embed", "projector", "qkv", "out", "fc1", "fc2",
    },
}

#: Число листьев в каждой группе на l3-full (измерено на eval_shape модели).
EXPECTED_LEAF_COUNTS = {
    optimizer.GROUP_MUON_MATRIX: 322,
    optimizer.GROUP_MUON_PER_HEAD: 75,
    optimizer.GROUP_MUON_BATCHED: 138,
    optimizer.GROUP_ADAMW_VECTOR: 187,
    optimizer.GROUP_ADAMW_EMBED: 1,
    optimizer.GROUP_UNCLASSIFIED: 0,
}

#: Связанный LM head — это сам ``embedding`` (``tie_embeddings``), поэтому
#: размер группы AdamW для emb/head равен словарю x hidden конфига.
L3_VOCAB, L3_HIDDEN = 160000, 1536


def _l3_full_shapes():
    """Абстрактное дерево параметров ``l3-full`` — без аллокации 1 ГБ."""
    from net.config import load_config

    cfg = load_config(L3_CONFIG)
    shapes = jax.eval_shape(
        lambda key: model.init_params(key, cfg),
        jax.ShapeDtypeStruct((2,), jnp.uint32),
    )
    return cfg, shapes


def _name_of(path) -> str:
    return optimizer.leaf_name(path[-1]) if path else ""


def _leaf_groups(tree) -> dict[str, list[str]]:
    """Группа -> имена листьев (walks the tree the same way ``init_state`` does)."""
    out: dict[str, list[str]] = {}

    def visit(path, leaf):
        group = optimizer.classify_leaf(_name_of(path), leaf.ndim)
        out.setdefault(group, []).append(_name_of(path))

    jax.tree_util.tree_map_with_path(visit, tree)
    return out


def _ones_like(tree):
    return jax.tree_util.tree_map(jnp.ones_like, tree)


def _adamw_reference(p, g, lr, wd, b1=0.9, b2=0.95):
    """Независимая формула AdamW — то, что обязана давать группа emb/head."""
    m = (1.0 - b1) * g
    v = (1.0 - b2) * (g * g)
    mh = m / (1.0 - b1)
    vh = v / (1.0 - b2)
    return p * (1.0 - lr * wd) - lr * mh / (jnp.sqrt(vh) + 1e-8)


def _muon_reference(p, g, lr, wd, momentum=0.95, steps=5, clip=1.0):
    """Независимая формула Muon (momentum + NS + weight clipping)."""
    m = momentum * 0 + (1.0 - momentum) * g
    upd = optimizer.newtonschulz5(m, steps)
    upd = jnp.clip(p * (1.0 - lr * wd) - lr * upd, -clip, clip)
    return upd


# ---------------------------------------------------------------------------
# 1. Полнота и состав классификации на реальной модели l3-full
# ---------------------------------------------------------------------------


def test_every_2d_leaf_of_l3_full_is_classified():
    """Ни один 2-D лист модели не остаётся вне групп (fail-closed гарантирует
    ``make_step``, а этот тест — что полный набор имён вообще покрыт)."""
    _, shapes = _l3_full_shapes()
    groups = _leaf_groups(shapes)
    assert groups.get(optimizer.GROUP_UNCLASSIFIED, []) == []
    assert sum(len(names) for names in groups.values()) == 723


def test_l3_full_2d_partition_matches_adr048():
    """Разбиение 2-D имён совпадает с ADR-048: emb/head — AdamW, остальное — Muon."""
    _, shapes = _l3_full_shapes()
    groups = _leaf_groups(shapes)
    got = {
        group: {name for name in names}
        for group, names in groups.items()
        if group in EXPECTED_2D_GROUPS
    }
    assert got == EXPECTED_2D_GROUPS


def test_l3_full_leaf_counts_per_group():
    """«Сколько чего» — числом, а не верой (ADR-048 п. 4)."""
    _, shapes = _l3_full_shapes()
    groups = _leaf_groups(shapes)
    counts = {group: len(groups.get(group, [])) for group in EXPECTED_LEAF_COUNTS}
    assert counts == EXPECTED_LEAF_COUNTS


def test_report_counts_embedding_as_adamw_and_names_examples():
    """Отчёт: emb/head выведены из Muon, скрытые матрицы остались (числом)."""
    _, shapes = _l3_full_shapes()
    report = optimizer.classification_report(shapes)
    groups = report["groups"]

    embed = groups[optimizer.GROUP_ADAMW_EMBED]
    assert embed["leaves"] == 1
    assert embed["params"] == L3_VOCAB * L3_HIDDEN
    assert embed["names"] == ["embedding"]

    assert "embedding" not in groups[optimizer.GROUP_MUON_MATRIX]["names"]
    assert "embedding" not in groups[optimizer.GROUP_MUON_PER_HEAD]["names"]
    assert "W_o" in groups[optimizer.GROUP_MUON_MATRIX]["names"]
    assert "W_q" in groups[optimizer.GROUP_MUON_PER_HEAD]["names"]
    assert "expert_d" in groups[optimizer.GROUP_MUON_BATCHED]["names"]

    assert report["unclassified"] == []
    assert report["total_leaves"] == 723
    assert report["total_params"] == sum(
        group["params"] for group in groups.values()
    )
    # Рендер отчёта — таблица «группа -> листьев -> параметров -> имена».
    table = optimizer.format_report(report)
    assert "embedding" in table
    assert optimizer.GROUP_ADAMW_EMBED in table


# ---------------------------------------------------------------------------
# 2. Поведение: emb/head -> AdamW, скрытые матрицы -> Muon / per-head / batched
# ---------------------------------------------------------------------------


def test_embedding_step_is_adamw_not_muon(tiny_cfg):
    params = model.init_params(jr.PRNGKey(0), tiny_cfg)
    grads = _ones_like(params)
    state = optimizer.init_state(params)
    step = optimizer.make_step(tiny_cfg)
    lr = 1e-3

    new_params, _ = step(params, grads, state, lr)
    expected = _adamw_reference(
        params.embedding, grads.embedding, lr, tiny_cfg.weight_decay
    )
    assert bool(jnp.allclose(new_params.embedding, expected, rtol=1e-5, atol=1e-7))
    assert not bool(
        jnp.allclose(
            new_params.embedding,
            _muon_reference(params.embedding, grads.embedding, lr, tiny_cfg.weight_decay),
            rtol=1e-3,
            atol=1e-5,
        )
    ), "embedding обновлён Muon — классификация ADR-048 не применилась"


def test_embedding_state_is_adamw_pair():
    """Состояние emb/head — пара (m, v), а не один momentum Muon."""
    tree = {"embedding": jnp.zeros((8, 4)), "W_o": jnp.zeros((4, 4))}
    state = optimizer.init_state(tree)
    m, v = state["embedding"]
    assert m.shape == (8, 4) and v.shape == (8, 4)
    assert state["W_o"].shape == (4, 4), "скрытая матрица потеряла momentum Muon"


def test_hidden_matrix_step_is_muon(tiny_cfg):
    params = model.init_params(jr.PRNGKey(0), tiny_cfg)
    grads = _ones_like(params)
    state = optimizer.init_state(params)
    step = optimizer.make_step(tiny_cfg)
    lr = 1e-3

    new_params, _ = step(params, grads, state, lr)
    leaf = new_params.layers[0].attn.W_o  # KDA output projection — скрытая матрица
    expected = _muon_reference(
        params.layers[0].attn.W_o, grads.layers[0].attn.W_o, lr, tiny_cfg.weight_decay
    )
    assert bool(jnp.allclose(leaf, expected, rtol=1e-5, atol=1e-7))


def test_per_head_matrix_uses_per_head_newtonschulz(tiny_cfg):
    """Ветка per-head (``_PER_HEAD_LEAVES``) сохранена: W_q ортогонализуется по
    головам, а не целиком."""
    params = model.init_params(jr.PRNGKey(0), tiny_cfg)
    grads = jax.tree_util.tree_map(
        lambda leaf: jr.normal(jr.PRNGKey(int(leaf.size) + 3), leaf.shape),
        params,
    )
    state = optimizer.init_state(params)
    step = optimizer.make_step(tiny_cfg)
    lr = 1e-3

    new_params, _ = step(params, grads, state, lr)
    p0 = params.layers[0].attn.W_q
    g0 = grads.layers[0].attn.W_q
    m = 0.95 * 0 + 0.05 * g0
    upd = optimizer.per_head_newtonschulz5(m, tiny_cfg.num_heads)
    expected = jnp.clip(p0 * (1.0 - lr * tiny_cfg.weight_decay) - lr * upd, -1.0, 1.0)
    assert bool(jnp.allclose(new_params.layers[0].attn.W_q, expected, rtol=1e-5, atol=1e-7))

    whole = jnp.clip(
        p0 * (1.0 - lr * tiny_cfg.weight_decay) - lr * optimizer.newtonschulz5(m),
        -1.0,
        1.0,
    )
    assert not bool(jnp.allclose(new_params.layers[0].attn.W_q, whole, rtol=1e-3, atol=1e-5))


def test_batched_moe_expert_branch_preserved(tiny_cfg):
    """ndim>=3 (эксперты LatentMoE) — прежняя batched-ветка vmap(NS)."""
    params = model.init_params(jr.PRNGKey(0), tiny_cfg)
    grads = jax.tree_util.tree_map(
        lambda leaf: jr.normal(jr.PRNGKey(int(leaf.size) + 5), leaf.shape), params
    )
    expert = params.layers[1].mlp.expert_g  # LatentMoE routed weights (n, d_in, d_out)
    assert expert.ndim >= 3

    state = optimizer.init_state(params)
    step = optimizer.make_step(tiny_cfg)
    lr = 1e-3
    new_params, _ = step(params, grads, state, lr)

    g0 = grads.layers[1].mlp.expert_g
    m = 0.95 * 0 + 0.05 * g0
    upd = jax.vmap(optimizer.newtonschulz5)(m.reshape(-1, *m.shape[-2:])).reshape(m.shape)
    expected = jnp.clip(
        expert * (1.0 - lr * tiny_cfg.weight_decay) - lr * upd, -1.0, 1.0
    )
    assert bool(
        jnp.allclose(new_params.layers[1].mlp.expert_g, expected, rtol=1e-5, atol=1e-7)
    )


# ---------------------------------------------------------------------------
# 3. Fail-closed на неизвестном 2-D имени
# ---------------------------------------------------------------------------


def test_unknown_2d_name_fails_closed_in_step_and_init(tiny_cfg):
    tree = {
        "embedding": jnp.zeros((8, 4)),
        "W_mystery": jnp.zeros((4, 4)),  # 2-D имя вне классификации
        "norm": jnp.zeros((4,)),
    }
    with pytest.raises(optimizer.UnclassifiedMatrixError):
        optimizer.init_state(tree)

    # Состояние собирается legacy-классификацией, падает именно шаг.
    state = optimizer.init_state(tree, legacy_muon_all_2d=True)
    step = optimizer.make_step(tiny_cfg)
    with pytest.raises(optimizer.UnclassifiedMatrixError):
        step(tree, tree, state, 1e-3)


def test_unknown_2d_name_is_recorded_in_report():
    tree = {"embedding": jnp.zeros((8, 4)), "W_mystery": jnp.zeros((4, 4))}
    report = optimizer.classification_report(tree)
    assert report["unclassified"] == ["W_mystery"]
    assert report["groups"][optimizer.GROUP_UNCLASSIFIED]["leaves"] == 1
    assert "W_mystery" in optimizer.format_report(report)


def test_unknown_names_of_other_ranks_do_not_fail():
    """Fail-closed — только про 2-D: векторы и стеки классифицируются по умолчанию."""
    assert optimizer.classify_leaf("b_mystery", 1) == optimizer.GROUP_ADAMW_VECTOR
    assert optimizer.classify_leaf("stack_mystery", 3) == optimizer.GROUP_MUON_BATCHED


# ---------------------------------------------------------------------------
# 4. Legacy-флаг: прежняя классификация «любой ndim==2 -> Muon»
# ---------------------------------------------------------------------------


def test_legacy_flag_restores_any_2d_is_muon(tiny_cfg):
    assert (
        optimizer.classify_leaf("embedding", 2, legacy=True)
        == optimizer.GROUP_MUON_MATRIX
    )
    assert (
        optimizer.classify_leaf("embedding", 2, legacy=False)
        == optimizer.GROUP_ADAMW_EMBED
    )

    params = model.init_params(jr.PRNGKey(0), tiny_cfg)
    grads = _ones_like(params)
    lr = 1e-3
    legacy_state = optimizer.init_state(params, legacy_muon_all_2d=True)
    legacy_step = optimizer.make_step(tiny_cfg, legacy_muon_all_2d=True)
    new_params, _ = legacy_step(params, grads, legacy_state, lr)
    assert bool(
        jnp.allclose(
            new_params.embedding,
            _muon_reference(
                params.embedding, grads.embedding, lr, tiny_cfg.weight_decay
            ),
            rtol=1e-5,
            atol=1e-7,
        )
    ), "legacy-флаг не вернул Muon для embeddings"

    # Состояние legacy — один momentum на 2-D лист, как было до ADR-048.
    assert legacy_state.embedding.shape == params.embedding.shape


def test_legacy_flag_partitions_l3_full_the_old_way():
    _, shapes = _l3_full_shapes()
    groups = _leaf_groups_legacy(shapes)
    assert groups.get(optimizer.GROUP_ADAMW_EMBED, []) == []
    assert groups.get(optimizer.GROUP_UNCLASSIFIED, []) == []
    # All 2-D leaves (except per-head ones) land in the plain Muon group.
    assert len(groups[optimizer.GROUP_MUON_MATRIX]) == 322 + 1


def _leaf_groups_legacy(tree) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}

    def visit(path, leaf):
        group = optimizer.classify_leaf(_name_of(path), leaf.ndim, legacy=True)
        out.setdefault(group, []).append(_name_of(path))

    jax.tree_util.tree_map_with_path(visit, tree)
    return out


# ---------------------------------------------------------------------------
# 5. ns_steps параметризован, дефолт 5 не сдвинут
# ---------------------------------------------------------------------------


def test_ns_steps_default_is_five():
    default = inspect.signature(optimizer.make_step).parameters["ns_steps"].default
    assert default == 5, "ADR-048: дефолт NS-итераций остаётся 5"


def test_ns_steps_changes_hidden_matrix_update(tiny_cfg):
    params = model.init_params(jr.PRNGKey(0), tiny_cfg)
    grads = jax.tree_util.tree_map(
        lambda leaf: jr.normal(jr.PRNGKey(int(leaf.size) + 7), leaf.shape), params
    )
    state = optimizer.init_state(params)
    lr = 1e-3

    p5, _ = optimizer.make_step(tiny_cfg, ns_steps=5)(params, grads, state, lr)
    p1, _ = optimizer.make_step(tiny_cfg, ns_steps=1)(params, grads, state, lr)
    p_default, _ = optimizer.make_step(tiny_cfg)(params, grads, state, lr)

    leaf5 = p5.layers[0].attn.W_o
    leaf1 = p1.layers[0].attn.W_o
    assert not bool(jnp.allclose(leaf5, leaf1, rtol=1e-4, atol=1e-6))
    assert bool(jnp.allclose(leaf5, p_default.layers[0].attn.W_o, rtol=0, atol=0))

    # per-head ветка тоже уважает ns_steps
    assert not bool(
        jnp.allclose(p5.layers[0].attn.W_q, p1.layers[0].attn.W_q, rtol=1e-4, atol=1e-6)
    )
    # batched-ветка тоже
    assert not bool(
        jnp.allclose(
            p5.layers[1].mlp.expert_g, p1.layers[1].mlp.expert_g, rtol=1e-4, atol=1e-6
        )
    )


def test_ns_steps_must_be_positive(tiny_cfg):
    with pytest.raises(ValueError):
        optimizer.make_step(tiny_cfg, ns_steps=0)


# ---------------------------------------------------------------------------
# 6. jit-совместимость и loss-шаг
# ---------------------------------------------------------------------------


def test_step_is_jittable_and_loss_step_survives(tiny_cfg):
    ids = jr.randint(jr.PRNGKey(0), (2, 16), 0, tiny_cfg.vocab_size)
    params = model.init_params(jr.PRNGKey(0), tiny_cfg)
    state = optimizer.init_state(params)
    step = jax.jit(optimizer.make_step(tiny_cfg))

    loss, grads = jax.jit(jax.value_and_grad(
        lambda p: model.compute_loss(p, tiny_cfg, ids, chunk_size=8)
    ))(params)
    assert bool(jnp.isfinite(loss))

    new_params, new_state = step(params, grads, state, 1e-3)
    assert all(
        bool(jnp.all(jnp.isfinite(leaf))) for leaf in jax.tree_util.tree_leaves(new_params)
    )
    # emb/head в состоянии — пара (m, v), остальное — momentum
    assert isinstance(new_state.embedding, tuple) and len(new_state.embedding) == 2
