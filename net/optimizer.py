"""Optimizer — Per-Head Muon + weight clipping, cosine LR + 1% warmup, wd 0.1.

Follows MODEL-L3-SKELETON.md section 3 and kda-formulas.md section 5:

* **Muon** orthogonalises the momentum of matrix parameters via Newton-Schulz
  (:661-671).  Attention projections (Q/K/V) use the **per-head** variant: the
  momentum is split along the head axis and orthogonalised per head.
* 1-D parameters (norms, biases) use AdamW.
* Stacked expert weights (n, d_in, d_out) are orthogonalised per matrix
  (batched Newton-Schulz over the leading axis).
* **Weight clipping** (MuonClip, our value since the exact clip is not
  disclosed) clamps matrix magnitudes after each step.
* Cosine schedule with a 1% linear warmup and weight decay 0.1 (:772-773).

Parameter groups (ADR-048)
--------------------------
The branch is **not** "any ``ndim == 2``": the group is decided by an explicit
predicate over leaf *names* (:func:`classify_leaf`), and the tree is walked in
one place.  After the KDA rewrite (ADR-047) the optimizer step became the
dominant cost of a step (``sec_backopt`` = 20.4 s of ~31 s), because Muon/NS5
was applied to every 2-D leaf — including the tied embeddings/LM head
(``vocab x hidden``) and the ViT ``patch_embed``/``pos_embed``
(ADR-048 Amendment, 08.10.2026 п.3), where orthogonalisation has no meaning.
ADR-048 routes them to AdamW:

============================  ============================================
group                         leaves
============================  ============================================
``muon_matrix``               hidden 2-D projections (attention/MLA/MoE
                              router/MLP/MTP/AttnRes/ViT projections)
``muon_per_head``             Q/K/V-family projections (``_PER_HEAD_LEAVES``)
``muon_batched``              stacked expert weights (``ndim >= 3``)
``adamw_vector``              norms, biases, scalars (``ndim <= 1``)
``adamw_embed``               the "embeddings": text embedding / tied LM head
                              and the ViT ``patch_embed``/``pos_embed``
============================  ============================================

An unknown **2-D** name is **fail-closed** (``UnclassifiedMatrixError``, and it
is listed in :func:`classification_report`) instead of silently taking the Muon
branch.  ``legacy_muon_all_2d=True`` restores the pre-ADR-048 classification
("any ``ndim == 2`` -> Muon") on the same code, for an honest before/after
comparison; the number of NS iterations is parameterised by ``ns_steps``
(default 5 — the ADR keeps that default until a measurement says otherwise).

Cost of the step (ADR-048 Amendment, 08.10.2026)
------------------------------------------------
The 20.4 s was **host overhead, not arithmetic**: the whole step ran *outside*
``jax.jit``, and ``_one`` walked the 723-leaf tree twice (two
``tree_map_with_path`` calls, so the Newton–Schulz work was materialised twice
over).  The step built by :func:`make_step` is therefore jitted (``lr`` is a
*dynamic* argument — a Python-float ``lr`` closed over would recompile on every
schedule tick) and walks the tree once.  The per-branch arithmetic is unchanged;
the CPU smoke in ``tools/optimizer_jit_smoke.py`` measures the ratio.

Batching of Newton–Schulz (ADR-054 Am.11, веха-5)
--------------------------------------------------
After the KDA victory the GPU front-end (not arithmetic) is the barrier: the
optimizer's NS5 was unrolled **per leaf** (``tree_map`` over ~533 Muon leaves on
l3-full), so the step carried hundreds of small fusion chains.  ``opt_batch =
"grouped"`` (``AXIOM_OPT_BATCH=grouped``) collects the Muon leaves of one
parameter group *and one shape* into a stack and runs NS5 with **one** ``vmap``
per group — one batched ``dot_general`` instead of N.  The arithmetic of a leaf
is unchanged (parity: ``net/tests/test_34_optimizer_ns5_batch.py``).  No padding:
groups are strictly equal-shape (``newtonschulz5(0) = 0`` exactly — verified —
so pad-to-max would be numerically safe, but it re-rounds the reductions and
burns FLOPs on the pad).  Default stays ``leaf`` — the rollback handle.

bf16 master tree (ADR-054 Am.13, цикл-5)
----------------------------------------

``master_dtype="bf16"`` (env ``AXIOM_MASTER_DTYPE``) keeps the parameter tree in
bf16 as its **only** representation: ``params is master``, grads arrive in bf16,
and fp32 survives only in the moments and in the arithmetic of the applied
update.  The leaf maths is therefore split in two:

* moment + Newton–Schulz — unchanged, but explicitly in fp32
  (``g.astype(jnp.float32)`` before the moment, so a bf16 grad does not round
  the momentum update in bf16);
* application — the fp32 target ``p32·(1 − lr·wd) − lr·upd`` (clip as before) is
  turned into ``update = target − p32`` and applied to the bf16 leaf by one of
  the :mod:`net.master_dtype` arms (``kahan`` | ``sr`` | ``none``).  The state
  of a bf16-mode leaf is ``(m, extra)`` (Muon) / ``(m, v, extra)`` (AdamW),
  where ``extra`` is the compensation buffer (bf16), the RNG key (uint32[2]) or
  ``None``.  Checkpoint/resume across modes goes through :func:`adapt_state`.

Связка с NS5-батчингом запрещена (фикс-раунд вехи-6)
----------------------------------------------------

``bf16``-мастер × ``AXIOM_OPT_BATCH=grouped`` **отвергается на сборке шага**
(:class:`IncompatibleModeError`).  Причина измерена: применение компенсации в
grouped-пути симметрично leaf (каждый лист применяется со своим ``comp``-
буфером — это проверено), но batched NS5 переупорядочивает редукции (Am.11;
на бакете (6,16,32) дельта обновления до ~1.1e-5 абс. ≈ 2e-5 отн.), а в
bf16-хранилище микро-сдвиг обновления (после ``lr`` — ~1.6e-8) перешагивает
середину округления: сохранённые листья расходятся ровно на шаг сетки bf16
(соседние представимые значения).  Критерий приёмки «не больше одного
округления хранилища» в формуле ``|y|·2^-8`` не выразим: у bf16 семь бит
мантиссы, и шаг сетки в 1..2 раза больше — «одно округление» становится
неотличимо от тихой расходимости.  До отдельного решения архитектора связка
не живёт молча (Am.13): шаг строится только как ``leaf`` × fp32-in-bf16-tree
или ``grouped`` × fp32-мастер.
"""

from __future__ import annotations

import os

import jax
import jax.numpy as jnp

from net import master_dtype as master_dtype_mod

# Newton-Schulz order-5 coefficients (Muon).
NS_COEFFS = (3.4445, -4.7750, 2.0315)

#: Default number of Newton-Schulz iterations (ADR-048: parameterised, this
#: default is the pre-existing value and is not to be changed without a
#: measurement — the architect decides).
NS_STEPS_DEFAULT = 5

# Attention-projection leaf names that get per-head orthogonalisation.
_PER_HEAD_LEAVES = frozenset({"W_q", "W_k", "W_v", "W_k_up", "W_v_up"})

# ---------------------------------------------------------------------------
# ADR-048 — parameter groups
# ---------------------------------------------------------------------------

#: Hidden 2-D projections orthogonalised by Muon (everything the model trains
#: that is not an embedding, a head, or a per-head attention projection).
#: Explicit on purpose: a new leaf name must be added here, otherwise
#: :func:`classify_leaf` fails closed.
MUON_MATRIX_LEAVES = frozenset({
    # KDA attention: output projection, output gate, decay/first-order
    # projections and the short convolutions.
    "W_o", "W_g", "W_a_down", "W_a_up", "W_beta", "conv_q", "conv_k", "conv_v",
    # MLA (latent attention): latent projection, indexer and the sliding-window
    # branch.
    "W_c", "W_idx_k", "W_idx_q", "W_swa_k", "W_swa_v",
    # MLP / LatentMoE router and the MTP block.
    "W_up", "W_down", "W_u", "router_w", "W_f",
    # AttnRes mixing matrix.
    "w",
    # ViT tower hidden matrices (image tokens only): the projections stay Muon,
    # its *embeddings* do not — see ADAMW_MATRIX_LEAVES.
    "projector", "qkv", "out", "fc1", "fc2",
})

#: 2-D leaves Muon must not touch — the "embeddings" group.  The LM head is
#: *tied* to ``embedding`` (``tie_embeddings``), so one named leaf covers both —
#: AdamW (ADR-048), as in Muon recipes.  The ViT tower's ``patch_embed``
#: (``patch*patch*3, vit_hidden``) and ``pos_embed`` (``num_patches,
#: vit_hidden``) are embeddings too: orthogonalising them has no meaning
#: (ADR-048 Amendment, 08.10.2026 п.3), so they live here, not in
#: :data:`MUON_MATRIX_LEAVES`.
ADAMW_MATRIX_LEAVES = frozenset({"embedding", "patch_embed", "pos_embed"})

#: Group ids, in report/table order.
GROUP_MUON_MATRIX = "muon_matrix"
GROUP_MUON_PER_HEAD = "muon_per_head"
GROUP_MUON_BATCHED = "muon_batched"
GROUP_ADAMW_VECTOR = "adamw_vector"
GROUP_ADAMW_EMBED = "adamw_embed"
GROUP_UNCLASSIFIED = "unclassified"

GROUP_NAMES = (
    GROUP_MUON_MATRIX,
    GROUP_MUON_PER_HEAD,
    GROUP_MUON_BATCHED,
    GROUP_ADAMW_VECTOR,
    GROUP_ADAMW_EMBED,
    GROUP_UNCLASSIFIED,
)

#: Groups whose state is the AdamW pair ``(m, v)``; all others keep one Muon
#: momentum buffer.
_ADAMW_GROUPS = frozenset({GROUP_ADAMW_VECTOR, GROUP_ADAMW_EMBED})

#: Muon groups — the ones whose NS5 is batched by shape under
#: ``AXIOM_OPT_BATCH=grouped``.  The AdamW branches are elementwise and stay
#: per-leaf: they are not touched by the batching (Am.11).
_MUON_GROUPS = frozenset({GROUP_MUON_MATRIX, GROUP_MUON_PER_HEAD, GROUP_MUON_BATCHED})

# ---------------------------------------------------------------------------
# ADR-054 (цикл-4, веха-5) — режим батчинга NS5
# ---------------------------------------------------------------------------

#: Flag values of ``AXIOM_OPT_BATCH``: ``leaf`` — the previous path (default,
#: the rollback handle), ``grouped`` — NS5 batched by shape-group.
OPT_BATCH_ENV = "AXIOM_OPT_BATCH"
OPT_BATCH_LEAF = "leaf"
OPT_BATCH_GROUPED = "grouped"
OPT_BATCH_MODES = (OPT_BATCH_LEAF, OPT_BATCH_GROUPED)


def opt_batch_mode(value: str | None = None) -> str:
    """NS5 batching mode: ``leaf`` (default) | ``grouped``.

    ``AXIOM_OPT_BATCH`` is read when the step is **built** (:func:`make_step`),
    not per call: the mode changes the *structure of the traced graph* (the
    grouping is built during tracing, where the shapes are static), so it cannot
    change in the middle of a run.  As with ``AXIOM_KDA_*``, an unknown value is
    an explicit error, never a silent fallback (fail-closed) — a typo must not
    pick a mode by accident.  An explicit ``value`` (the ``make_step(opt_batch=)``
    argument) wins over the environment.
    """
    raw = value if value is not None else os.environ.get(OPT_BATCH_ENV)
    mode = (raw or OPT_BATCH_LEAF).strip().lower()
    if mode not in OPT_BATCH_MODES:
        raise ValueError(
            f"{OPT_BATCH_ENV}={mode!r}: ожидается одно из {OPT_BATCH_MODES}")
    return mode


class UnclassifiedMatrixError(ValueError):
    """A 2-D leaf is outside the classification — fail-closed (ADR-048).

    Raised instead of falling back to Muon: silently orthogonalising a new
    matrix is exactly the defect ADR-048 removes (Muon on embeddings/LM head),
    and silently switching it to AdamW would hide a missing entry from the
    report.  Fix by adding the leaf name to :data:`MUON_MATRIX_LEAVES` or
    :data:`ADAMW_MATRIX_LEAVES` (or by running with ``legacy_muon_all_2d``).
    """


class IncompatibleModeError(ValueError):
    """Связка режимов шага отвергнута на сборке — fail-closed (ADR-054 Am.13).

    Первый случай — ``AXIOM_MASTER_DTYPE=bf16`` × ``AXIOM_OPT_BATCH=grouped``:
    batched NS5 переупорядочивает редукции (Am.11), а bf16-хранилище мастера
    превращает микро-сдвиг обновления в скачок округления у части элементов
    (фикс-раунд вехи-6: расхождение — ровно шаг сетки bf16, соседние
    представимые значения, при формально «одном округлении»).  Пока связка не
    доказана по критерию «не больше одного округления хранилища», она падает
    здесь, а не молча расходится.  Симптом лечится выбором ``leaf`` (дефолт)
    или fp32-мастера.
    """


def leaf_name(entry) -> str:
    """Name of a pytree key (``GetAttrKey``/``DictKey``/``SequenceKey``)."""
    return getattr(entry, "name", None) or getattr(entry, "key", None) or getattr(entry, "idx", None) or ""


def classify_leaf(name: str, ndim: int, *, legacy: bool = False) -> str:
    """Group of a parameter leaf — explicit predicate, not "any ``ndim == 2``".

    ``legacy=True`` reproduces the pre-ADR-048 classification (every 2-D leaf
    Muon, embedding included) so a before/after comparison can run on one code
    revision.
    """
    if legacy:
        if ndim >= 3:
            return GROUP_MUON_BATCHED
        if ndim == 2:
            return GROUP_MUON_PER_HEAD if name in _PER_HEAD_LEAVES else GROUP_MUON_MATRIX
        return GROUP_ADAMW_VECTOR
    if ndim >= 3:
        return GROUP_MUON_BATCHED
    if ndim == 2:
        if name in _PER_HEAD_LEAVES:
            return GROUP_MUON_PER_HEAD
        if name in ADAMW_MATRIX_LEAVES:
            return GROUP_ADAMW_EMBED
        if name in MUON_MATRIX_LEAVES:
            return GROUP_MUON_MATRIX
        return GROUP_UNCLASSIFIED
    return GROUP_ADAMW_VECTOR


def newtonschulz5(g: jnp.ndarray, steps: int = NS_STEPS_DEFAULT, eps: float = 1e-7) -> jnp.ndarray:
    """Newton-Schulz orthogonalisation of a 2-D matrix."""
    a, b, c = NS_COEFFS
    x = g / (jnp.linalg.norm(g) + eps)
    transposed = x.shape[0] > x.shape[1]
    if transposed:
        x = x.T
    for _ in range(steps):
        A = x @ x.T
        B = b * A + c * (A @ A)
        x = a * x + B @ x
    if transposed:
        x = x.T
    return x


def per_head_newtonschulz5(
    g: jnp.ndarray, heads: int, steps: int = NS_STEPS_DEFAULT
) -> jnp.ndarray:
    """Orthogonalise a Q/K/V momentum matrix ``(dim_in, heads*dim_head)`` per head."""
    dim_in = g.shape[0]
    dim_head = g.shape[-1] // heads
    g = g.reshape(dim_in, heads, dim_head).transpose(1, 0, 2)  # (H, dim_in, dim_head)
    out = jax.vmap(lambda x: newtonschulz5(x, steps))(g)  # (H, dim_in, dim_head)
    return out.transpose(1, 0, 2).reshape(dim_in, heads * dim_head)


def _state_arity(group: str, *, bf16: bool) -> int:
    """Число листьев состояния на один лист параметров этой группы.

    fp32-режим — прежняя раскладка: Muon ``m`` (1), AdamW ``(m, v)`` (2).
    bf16-режим добавляет каждому листу **слот компенсации** (Am.13): Muon
    ``(m, extra)`` (2), AdamW ``(m, v, extra)`` (3).  Слот есть и у плеча
    ``none`` (там он пустой, ``None``) — раскладка одна на все плечи, поэтому
    адаптер состояния и групповой обход не ветвятся по плечу ещё раз.
    """
    if group in _MUON_GROUPS:
        return 2 if bf16 else 1
    return 3 if bf16 else 2


def state_leaf_counts(params, *, master_dtype: str | None = None,
                      legacy_muon_all_2d: bool = False) -> list[int]:
    """Число листьев состояния на каждый лист параметров, в порядке дерева.

    Контракт раскладки состояния: сумма — полное число листьев состояния, а
    пара с :func:`adapt_state` позволяет перенести состояние между режимами
    мастера (``net/train_loop.py``, resume).  Классификация — та же
    (:func:`classify_leaf`, fail-closed на неизвестном 2-D имени).
    """
    dtype = master_dtype_mod.master_dtype_mode(master_dtype)
    bf16 = dtype == master_dtype_mod.BF16
    counts: list[int] = []
    for path, leaf in jax.tree_util.tree_flatten_with_path(params)[0]:
        group = classify_leaf(leaf_name(path[-1]) if path else "", leaf.ndim,
                              legacy=legacy_muon_all_2d)
        if group == GROUP_UNCLASSIFIED:
            raise UnclassifiedMatrixError(
                f"2-D параметр {leaf_name(path[-1]) if path else ''!r} "
                f"({jax.tree_util.keystr(path)}) вне классификации ADR-048: "
                "дополни MUON_MATRIX_LEAVES или ADAMW_MATRIX_LEAVES, либо "
                "включи legacy_muon_all_2d"
            )
        counts.append(_state_arity(group, bf16=bf16))
    return counts


def _fresh_extra(leaf, comp: str, base_key, counter: int):
    """Свежий слот компенсации листа: буфер (kahan), ключ (sr) или ``None``."""
    if comp == master_dtype_mod.KAHAN:
        return jnp.zeros(leaf.shape, dtype=jnp.bfloat16)
    if comp == master_dtype_mod.SR:
        return jax.random.fold_in(base_key, counter)
    return None


def _leaf_state(group: str, leaf, *, bf16: bool, comp: str, base_key, counter: int):
    """Начальное состояние одного листа в раскладке режима."""
    if bf16:
        # Моменты — всегда fp32 (Am.13): bf16 живёт только в мастере и в
        # буфере компенсации; dtype листа параметров состояние не задаёт.
        if group in _MUON_GROUPS:
            return (jnp.zeros(leaf.shape, dtype=jnp.float32),
                    _fresh_extra(leaf, comp, base_key, counter))
        return (jnp.zeros(leaf.shape, dtype=jnp.float32),
                jnp.zeros(leaf.shape, dtype=jnp.float32),
                _fresh_extra(leaf, comp, base_key, counter))
    if group in _ADAMW_GROUPS:
        return (jnp.zeros_like(leaf), jnp.zeros_like(leaf))
    return jnp.zeros_like(leaf)


def init_state(params, *, legacy_muon_all_2d: bool = False,
               master_dtype: str | None = None, master_comp: str | None = None,
               key_seed: int = 0):
    """Momentum (Muon groups) / AdamW m,v (Adam groups) state tree over ``params``.

    The group of every leaf comes from :func:`classify_leaf`, so the state
    mirrors the optimizer branch that will actually run: an embedding moved to
    AdamW gets an ``(m, v)`` pair instead of a single momentum buffer.

    ADR-054 Am.13 (цикл-5): при ``AXIOM_MASTER_DTYPE=bf16`` моментам задаётся
    fp32 **независимо от dtype листа параметров** (мастер bf16, моменты —
    хранилище точности), а каждому листу добавляется слот компенсации
    (:func:`_leaf_state`): буфер ``comp`` (bf16, kahan) или RNG-ключ (sr),
    у ``none`` — пустой.  ``key_seed`` заводит поток ключей sr: он же —
    «сид» воспроизводимости плеча (ключи едут в состоянии, поэтому resume
    продолжает поток, а не начинает его заново).
    """
    dtype = master_dtype_mod.master_dtype_mode(master_dtype)
    comp = master_dtype_mod.master_comp_mode(master_comp, master_dtype=dtype)
    bf16 = dtype == master_dtype_mod.BF16
    base_key = jax.random.PRNGKey(int(key_seed))
    counter = 0

    def _init(path, leaf):
        nonlocal counter
        index = counter
        counter += 1
        group = classify_leaf(leaf_name(path[-1]), leaf.ndim, legacy=legacy_muon_all_2d)
        if group == GROUP_UNCLASSIFIED:
            raise UnclassifiedMatrixError(
                f"2-D параметр {leaf_name(path[-1])!r} ({jax.tree_util.keystr(path)}) "
                "вне классификации ADR-048: дополни MUON_MATRIX_LEAVES или "
                "ADAMW_MATRIX_LEAVES, либо включи legacy_muon_all_2d"
            )
        return _leaf_state(group, leaf, bf16=bf16, comp=comp, base_key=base_key,
                           counter=index)

    return jax.tree_util.tree_map_with_path(_init, params)


def adapt_state(saved_state, params, *, saved_master_dtype: str | None = None,
                saved_master_comp: str | None = None,
                master_dtype: str | None = None, master_comp: str | None = None,
                legacy_muon_all_2d: bool = False, key_seed: int = 0):
    """Привести восстановленное состояние к раскладке текущего режима мастера.

    Контракт resume (ADR-054 Am.13 п.3): чекпойнт fp32-режима, продолженный в
    bf16-режиме, даёт «разовый каст + нулевой comp»: моменты у обоих режимов
    fp32, поэтому переносятся как есть, а слот компенсации заводится свежим
    (нулевой буфер / новый ключ).  Обратно (bf16 → fp32) слот компенсации
    отбрасывается — апкаст мастера делает вызывающий
    (``net/checkpoint.py``, читатель формата), а состояние здесь остаётся
    в fp32-раскладке.

    Раскладка восстановленного состояния определяется **режимом чекпойнта**
    (``saved_*``, из манифеста), а не текущим: число листьев на параметр —
    :func:`state_leaf_counts`.  Порядок листьев обоих деревьев — канонический
    (обход дерева параметров), поэтому листья сопоставляются позиционно.
    """
    target_dtype = master_dtype_mod.master_dtype_mode(master_dtype)
    target_comp = master_dtype_mod.master_comp_mode(master_comp, master_dtype=target_dtype)
    target_bf16 = target_dtype == master_dtype_mod.BF16
    saved_dtype = saved_master_dtype or master_dtype_mod.FP32
    saved_counts = state_leaf_counts(params, master_dtype=saved_dtype,
                                     legacy_muon_all_2d=legacy_muon_all_2d)
    path_leaves, param_treedef = jax.tree_util.tree_flatten_with_path(params)
    saved_leaves = jax.tree_util.tree_leaves(saved_state)
    if len(saved_leaves) != sum(saved_counts):
        raise ValueError(
            f"состояние не соответствует раскладке {saved_dtype}-режима: "
            f"листьев {len(saved_leaves)}, ожидалось {sum(saved_counts)} "
            f"({len(saved_counts)} листьев параметров)"
        )
    base_key = jax.random.PRNGKey(int(key_seed))
    cursor = 0
    new_leaves: list = []
    for index, ((path, leaf), count) in enumerate(zip(path_leaves, saved_counts)):
        chunk = saved_leaves[cursor:cursor + count]
        cursor += count
        group = classify_leaf(leaf_name(path[-1]) if path else "", leaf.ndim,
                              legacy=legacy_muon_all_2d)
        if group == GROUP_UNCLASSIFIED:
            raise UnclassifiedMatrixError(
                f"2-D параметр {leaf_name(path[-1]) if path else ''!r} "
                f"({jax.tree_util.keystr(path)}) вне классификации ADR-048"
            )
        if target_bf16:
            extra = _fresh_extra(leaf, target_comp, base_key, index)
            if group in _MUON_GROUPS:
                new_leaves.append((chunk[0].astype(jnp.float32), extra))
            else:
                new_leaves.append((chunk[0].astype(jnp.float32),
                                   chunk[1].astype(jnp.float32), extra))
        elif group in _MUON_GROUPS:
            new_leaves.append(chunk[0].astype(jnp.float32))
        else:
            new_leaves.append((chunk[0].astype(jnp.float32),
                               chunk[1].astype(jnp.float32)))
    return param_treedef.unflatten(new_leaves)


def _state_signature(state) -> tuple:
    """dtype-подпись листа состояния (ключ группировки NS5-батчинга)."""
    return tuple(str(leaf.dtype) for leaf in jax.tree_util.tree_leaves(state))


def make_step(
    cfg,
    muon_momentum: float = 0.95,
    adam_b1: float = 0.9,
    adam_b2: float = 0.95,
    weight_clip: float | None = 1.0,
    ns_steps: int = NS_STEPS_DEFAULT,
    legacy_muon_all_2d: bool = False,
    *,
    jit: bool = True,
    single_pass: bool = True,
    donate: bool = False,
    work_dtype: str | None = None,
    opt_batch: str | None = None,
    master_dtype: str | None = None,
    master_comp: str | None = None,
):
    """Build the optimizer step — jitted, one walk over the parameter tree.

    ``ns_steps`` — Newton-Schulz iterations for the three Muon groups
    (ADR-048 п. 2; the default is the pre-existing value 5).
    ``legacy_muon_all_2d`` — restore the pre-ADR-048 branch ("any ``ndim == 2``
    -> Muon", embeddings included) for a before/after comparison on one code
    revision.

    ADR-048 Amendment (08.10.2026): the measured ``sec_backopt`` = 20.4 s of a
    ~31 s step was **host** overhead, not arithmetic (the whole step's NS work is
    ~3.6e13 FLOP, tens of milliseconds).  Two causes are removed here:

    * the step ran *outside* ``jax.jit`` — now wrapped, with ``lr`` left
      **dynamic** (a Python float closed over instead would recompile the step
      on every schedule tick; the tree structure and the hyper-parameters —
      momenta, ``eps``, ``ns_steps`` — stay static, which is what jit wants);
    * ``_one`` walked the tree twice (one ``tree_map_with_path`` per output
      tree, so every leaf's update was computed twice) — now walked once by
      :func:`_walk_once`.

    The arithmetic of every branch is untouched.  ``jit=False`` returns the
    plain function (eager-vs-jit smoke, debugging); ``single_pass=False``
    restores the pre-Amendment two-walk traversal as the parity oracle for the
    one-walk refactor — **never enable it in training**, it doubles the
    Newton–Schulz work.

    ADR-054 (donate-аргументы): ``donate=True`` — **контракт владения
    буферами**, не математика: шаг получает ``donate_argnums=(0, 1, 2)`` и
    вправе переиспользовать буферы ``params`` (master) / ``grads`` / ``state``
    под свои выходы.  Вызывающий обязан не читать переданные деревья после
    вызова (они помечаются удалёнными: «Array has been deleted»); возврат шага
    — новые деревья, они живые.  Арифметика выхода идентична ``donate=False``
    (паритет — ``net/tests/test_donate_step.py``).  Дефолт ``False`` безопасен
    для всех вызывающих, которые читают входные деревья после шага
    (профайлер, SFT/RL-смоуки, тесты); тренировочный цикл включает donate
    явно там, где буферы мертвы после вызова (``net/train_loop.py``).

    ADR-054 (цикл-4, Am.10 п.1): ``work_dtype`` — dtype **рабочей копии**
    параметров, возвращаемой третьим выходом шага
    (``(new_params, new_state, new_work)``).  Каст ``new_params.astype(work_dtype)``
    идёт **внутри того же jitted-графа**, что и сам шаг: у цикла исчезает
    ``working_params(master)`` вне jit — на шаг это ~330 отдельных
    ``jit(convert_element_type)`` модулей (замер цикла-3, ADR-054 Am.10 «Очередь»).
    Премисса легальности при ``donate=True``: каст читает ``new_params`` — уже
    вычисленный возврат шага, а не донированный вход.  ``None`` (дефолт) —
    прежний контракт (два выхода); существующие вызывающие не меняются.

    ADR-054 (цикл-4, Am.11, веха-5): ``opt_batch`` — режим NS5-батчинга
    (:func:`opt_batch_mode`; ``None`` = ``AXIOM_OPT_BATCH`` или ``leaf``).
    ``"leaf"`` — прежний путь: NS5 разворачивается **по каждому** Muon-листу
    отдельно (на l3-full 533 Muon-листа ⇒ сотни мелких фьюзий в step_fn —
    доминирующая стоимость GPU front-end после победы над KDA, Am.11).
    ``"grouped"`` — Muon-листья собираются в стёк по **(группа, форма)** и NS5
    исполняется одним ``vmap`` на группу: одно batched ``dot_general`` вместо N,
    на l3-full 533 цепочки NS5 → 23 (17 групп ``muon_matrix`` + 2
    ``muon_per_head`` + 4 ``muon_batched``; из них 2 листа-одиночки своей формы
    идут прямым вызовом).  Арифметика на лист та же (момент,
    NS5, wd, clip; паритет — ``net/tests/test_34_optimizer_ns5_batch.py``);
    группы строго равной формы, паддинга нет.  Одиночный лист своей группы
    проходит прежним leaf-вызовом (побитово прежнее), AdamW-группы и их
    momentum/Adam-цепочки батчинг не трогает.  ``single_pass=False`` (оракул
    паритета «один проход ↔ два») остаётся leaf-обходом и старше ``opt_batch``.
    Связка ``"grouped"`` с bf16-мастером запрещена на сборке
    (:class:`IncompatibleModeError`; причина — см. блок Am.13 ниже и модульный
    докстринг): batched NS5 переупорядочивает редукции (Am.11), а в
    bf16-хранилище это даёт расхождение мастеров на шаг сетки, не отличимое от
    тихой расходимости.

    ADR-054 (цикл-5, Am.13): ``master_dtype`` (``None`` = ``AXIOM_MASTER_DTYPE``,
    дефолт ``fp32``) и ``master_comp`` (``None`` = ``AXIOM_MASTER_COMP``,
    дефолт ``kahan`` при bf16-мастере) — режим дерева параметров.  При
    ``master_dtype="fp32"`` шаг **побитово прежний** (эта ветка кода не
    менялась ни одним выражением: гейт-off — прежний граф, критерий Am.13).
    При ``"bf16"``:

    * ``params`` (мастер) — bf16, ``grads`` — bf16, моменты (Muon ``m``,
      AdamW ``m/v``) — fp32; в bf16-графе градиент поднимается в fp32 до
      арифметики листа (``g32``), чтобы момент считался в fp32, а не в bf16;
    * обновление считается в fp32 из fp32-моментов: целевое значение —
      ``target = p32·(1 − lr·wd) − lr·upd`` с прежним clip, ``update = target − p32``;
    * ``update`` применяется к bf16-мастеру плечом ``master_comp``
      (:mod:`net.master_dtype`): ``kahan`` — компенсационный буфер, ``sr`` —
      стохастическое округление с ключом, ``none`` — контрольное RN-плечо;
    * третий выход (``work_dtype``) упразднён: рабочая копия **есть** мастер
      (``params is master``); передать оба аргумента — ошибка сборки, а не
      тихий выбор одного из двух контрактов;
    * связка с ``opt_batch="grouped"`` — тоже ошибка сборки
      (:class:`IncompatibleModeError`, фикс-раунд вехи-6): batched NS5 меняет
      порядок редукций (Am.11; измеренная дельта обновления ~2e-5 отн.), и в
      bf16-хранилище микро-сдвиг перешагивает середину округления — листья
      расходятся ровно на шаг сетки bf16, а порог «одно округление» в
      ``|y|·2^-8`` (7 бит мантиссы у bf16) такой скачок не ловит; запрет — вместо
      тихой расходимости.
    """
    if int(ns_steps) < 1:
        raise ValueError(f"ns_steps must be >= 1, got {ns_steps!r}")
    # Опечатка в dtype падает здесь, на сборке шага, а не в первом шаге цикла.
    work_dtype = None if work_dtype is None else jnp.dtype(work_dtype)
    heads = cfg.num_heads
    wd = cfg.weight_decay
    # Веха-5: режим читается на сборке шага — группировка строится на трассировке
    # (формы статичны), тип графа от режима зависит.  Неизвестное значение флага
    # падает здесь, а не тихо выбирает путь.
    batch = opt_batch_mode(opt_batch)
    # Цикл-5: режим мастера — тоже на сборке (dtype дерева/состояния — часть
    # контракта графа).  fp32 — прежнее поведение по построению.
    master = master_dtype_mod.master_dtype_mode(master_dtype)
    comp = master_dtype_mod.master_comp_mode(master_comp, master_dtype=master)
    bf16_master = master == master_dtype_mod.BF16
    if bf16_master and work_dtype is not None:
        raise ValueError(
            "work_dtype задан при bf16-мастере: рабочая копия ЕСТЬ мастер "
            "(params is master), третьего выхода в этом режиме нет"
        )
    if bf16_master and batch == OPT_BATCH_GROUPED:
        # Фикс-раунд вехи-6: связка «bf16-мастер × grouped» отвергнута на сборке.
        # Причина измерена, а не унаследована: применение компенсации в grouped
        # симметрично leaf (каждый лист — со своим буфером), но batched NS5
        # переупорядочивает редукции (Am.11) — на бакете (6,16,32) дельта
        # обновления до 1.1e-5 абс.; после lr=1e-2 это ~1.6e-8, и элементы
        # вблизи середины округления хранилища перескакивают на соседнее bf16-
        # значение: разница листьев — ровно шаг сетки bf16 (7 бит мантиссы, шаг до
        # 2x больше |y|·2^-8).  Критерий «не больше одного округления хранилища»
        # по построению не выразим, поэтому связка падает здесь (Am.13: «тихо
        # расходится» — неприемлемо), а не тихо живёт.
        raise IncompatibleModeError(
            f"{master_dtype_mod.MASTER_DTYPE_ENV}=bf16 несовместим с "
            f"{OPT_BATCH_ENV}=grouped: batched NS5 переупорядочивает редукции "
            "(Am.11), а в bf16-хранилище микро-сдвиг обновления перешагивает "
            "середину округления — листья master расходятся на шаг сетки bf16 "
            "(7 бит манты: шаг до 2x больше |y|·2^-8), т.е. «одно округление "
            "хранилища» не отличимо от тихой расходимости. Выбери "
            f"{OPT_BATCH_ENV}=leaf (дефолт) или "
            f"{master_dtype_mod.MASTER_DTYPE_ENV}=fp32."
        )

    def _matrix_step(p, g, m, lr, per_head):
        m = muon_momentum * m + (1.0 - muon_momentum) * g
        upd = (
            per_head_newtonschulz5(m, heads, ns_steps)
            if per_head
            else newtonschulz5(m, ns_steps)
        )
        p = p * (1.0 - lr * wd) - lr * upd
        if weight_clip is not None:
            p = jnp.clip(p, -weight_clip, weight_clip)
        return p, m

    def _batched_matrix_step(p, g, m, lr):
        """Muon for stacked matrices (n, d_in, d_out) — LatentMoE expert weights."""
        m = muon_momentum * m + (1.0 - muon_momentum) * g
        flat = m.reshape(-1, *m.shape[-2:])
        upd = jax.vmap(lambda x: newtonschulz5(x, ns_steps))(flat).reshape(m.shape)
        p = p * (1.0 - lr * wd) - lr * upd
        if weight_clip is not None:
            p = jnp.clip(p, -weight_clip, weight_clip)
        return p, m

    def _adamw_step(p, g, st, lr, *, clip: bool):
        """AdamW, elementwise — vectors (norms/biases) and the 2-D embeddings."""
        m, v = st
        m = adam_b1 * m + (1.0 - adam_b1) * g
        v = adam_b2 * v + (1.0 - adam_b2) * (g * g)
        mh = m / (1.0 - adam_b1)
        vh = v / (1.0 - adam_b2)
        p = p * (1.0 - lr * wd) - lr * mh / (jnp.sqrt(vh) + 1e-8)
        # ADR-048 Amendment п.4: ``weight_clip`` (1.0) also covers ``adamw_embed``,
        # exactly as the Muon branch clipped it.  ``adamw_vector`` keeps the
        # pre-Amendment behaviour (no clip): two properties are not changed at once.
        if clip and weight_clip is not None:
            p = jnp.clip(p, -weight_clip, weight_clip)
        return p, (m, v)

    def _muon_leaf(group, p, g, m, lr):
        """Muon одного листа по его группе — тело ``_matrix_step``/``_batched``."""
        if group == GROUP_MUON_PER_HEAD:
            return _matrix_step(p, g, m, lr, True)  # (new_p, momentum)
        if group == GROUP_MUON_MATRIX:
            return _matrix_step(p, g, m, lr, False)  # (new_p, momentum)
        return _batched_matrix_step(p, g, m, lr)  # (new_p, momentum)

    # ------------------------------------------------------------------
    # ADR-054 Am.13 (цикл-5): bf16-мастер — те же момент и NS5 в fp32,
    # применение обновления — плечом компенсации к bf16-листу.
    # ------------------------------------------------------------------

    def _bf16_muon_leaf(group, p, g, st, lr):
        """Muon листа bf16-мастера: ``(m, extra)`` → ``(new_p, (m, new_extra))``.

        Момент и NS5 — прежняя арифметика (Am.13: «Muon-ортогонализация без
        изменений»), отличие одно: считается в fp32 из fp32-момента (``g32``),
        а результат применяется к bf16-мастеру компенсированно.
        """
        m, extra = st
        g32 = g.astype(jnp.float32)
        m = muon_momentum * m + (1.0 - muon_momentum) * g32
        if group == GROUP_MUON_PER_HEAD:
            upd = per_head_newtonschulz5(m, heads, ns_steps)
        elif group == GROUP_MUON_MATRIX:
            upd = newtonschulz5(m, ns_steps)
        else:
            flat = m.reshape(-1, *m.shape[-2:])
            upd = jax.vmap(lambda x: newtonschulz5(x, ns_steps))(flat).reshape(m.shape)
        p32 = p.astype(jnp.float32)
        target = p32 * (1.0 - lr * wd) - lr * upd
        if weight_clip is not None:
            target = jnp.clip(target, -weight_clip, weight_clip)
        new_p, new_extra = master_dtype_mod.apply_update(comp, p, target - p32, extra)
        return new_p, (m, new_extra)

    def _bf16_adamw_step(p, g, st, lr, *, clip: bool):
        """AdamW листа bf16-мастера: ``(m, v, extra)`` → ``(new_p, (m, v, extra))``.

        ``m/v`` — fp32 (в т.ч. ``g*g``: градиент поднимается в fp32 **до**
        квадрата, иначе квадрат округлился бы в bf16); clip — прежний
        (``adamw_embed`` — да, ``adamw_vector`` — нет), но применён к
        fp32-цели: bf16-мастер ведёт ту же траекторию, что fp32-мастер.
        """
        m, v, extra = st
        g32 = g.astype(jnp.float32)
        m = adam_b1 * m + (1.0 - adam_b1) * g32
        v = adam_b2 * v + (1.0 - adam_b2) * (g32 * g32)
        mh = m / (1.0 - adam_b1)
        vh = v / (1.0 - adam_b2)
        p32 = p.astype(jnp.float32)
        target = p32 * (1.0 - lr * wd) - lr * mh / (jnp.sqrt(vh) + 1e-8)
        # ADR-048 Amendment п.4: ``weight_clip`` (1.0) also covers ``adamw_embed``,
        # exactly as the Muon branch clipped it.  ``adamw_vector`` keeps the
        # pre-Amendment behaviour (no clip): two properties are not changed at once.
        if clip and weight_clip is not None:
            target = jnp.clip(target, -weight_clip, weight_clip)
        new_p, new_extra = master_dtype_mod.apply_update(comp, p, target - p32, extra)
        return new_p, (m, v, new_extra)

    def _muon_group_step(group, ps, gs, ss, lr):
        """Muon для стека листьев одной (группа, форма): ОДИН NS5 на группу.

        Тот же момент ``m``, тот же NS5 (vmap по стеку вместо разворачивания по
        листьям), тот же wd/clip: одно batched ``dot_general`` на итерацию вместо
        ``len(ps)`` отдельных.  Внутри — те же функции, что у leaf-пути
        (:func:`newtonschulz5` / :func:`per_head_newtonschulz5`).

        bf16-мастер сюда не доходит: связка «bf16 × grouped» отвергается на
        сборке шага (:class:`IncompatibleModeError`, фикс-раунд вехи-6 — batched
        NS5 переупорядочивает редукции, и в bf16-хранилище это перешагивает
        округление), поэтому здесь только fp32-ветка.
        """
        # ss — стёк моментов (fp32-режим; ветка bf16 сюда не доходит — связка
        # с grouped отвергнута на сборке).  Веха-6 переименовала параметр
        # ms -> ss (под пару (m, extra)); здесь — восстановленная привязка
        # fp32-пути (без неё grouped × fp32 падал NameError).
        m = muon_momentum * ss + (1.0 - muon_momentum) * gs
        if group == GROUP_MUON_PER_HEAD:
            upd = jax.vmap(lambda one: per_head_newtonschulz5(one, heads, ns_steps))(m)
        elif group == GROUP_MUON_BATCHED:
            # (N, n, d_in, d_out) -> (N*n, d_in, d_out): тот же разворот ведущей
            # оси, что у ``_batched_matrix_step`` на одном листе.
            flat = m.reshape(-1, *m.shape[-2:])
            upd = jax.vmap(lambda one: newtonschulz5(one, ns_steps))(flat).reshape(m.shape)
        else:
            upd = jax.vmap(lambda one: newtonschulz5(one, ns_steps))(m)
        p = ps * (1.0 - lr * wd) - lr * upd
        if weight_clip is not None:
            p = jnp.clip(p, -weight_clip, weight_clip)
        return p, m

    def _one(path, p, g, s, lr):
        """Update one leaf; returns ``(new_param, new_state_leaf)``."""
        name = leaf_name(path[-1]) if path else ""
        group = classify_leaf(name, p.ndim, legacy=legacy_muon_all_2d)
        if group == GROUP_UNCLASSIFIED:
            raise UnclassifiedMatrixError(
                f"2-D параметр {name!r} ({jax.tree_util.keystr(path)}) вне "
                "классификации ADR-048: дополни MUON_MATRIX_LEAVES или "
                "ADAMW_MATRIX_LEAVES, либо включи legacy_muon_all_2d"
            )
        if group in _MUON_GROUPS:
            if bf16_master:
                return _bf16_muon_leaf(group, p, g, s, lr)
            return _muon_leaf(group, p, g, s, lr)
        if bf16_master:
            # ``adamw_embed`` — с clip, ``adamw_vector`` — без (Am.4 не переоткрыт).
            return _bf16_adamw_step(p, g, s, lr, clip=group == GROUP_ADAMW_EMBED)
        if group == GROUP_ADAMW_EMBED:
            return _adamw_step(p, g, s, lr, clip=True)  # (new_p, (m, v))
        # ``adamw_vector`` — norms, biases, scalars.
        return _adamw_step(p, g, s, lr, clip=False)  # (new_p, (m, v))

    def _walk_once(params, grads, state, lr):
        """One walk: ``_one`` runs once per leaf, not once per output tree.

        ``jax.tree_util.tree_map`` cannot split the ``(new_param, new_state)``
        pair directly — ``tree_unflatten`` would read the tuple as *nested
        structure* (the defect that forced the two-pass version), and
        ``tree_transpose`` refuses it too, because the state leaf of an Adam
        group is itself an ``(m, v)`` tuple, so the pair's inner structure is
        not uniform.  Flattening the parameter tree once and unflattening each
        output against its own treedef does the split in a single traversal.
        """
        path_leaves, param_treedef = jax.tree_util.tree_flatten_with_path(params)
        grads_for = param_treedef.flatten_up_to(grads)
        state_for = param_treedef.flatten_up_to(state)
        new_params: list = []
        new_state: list = []
        for (path, p), g, s in zip(path_leaves, grads_for, state_for):
            updated, next_state = _one(path, p, g, s, lr)
            new_params.append(updated)
            new_state.append(next_state)
        return (
            param_treedef.unflatten(new_params),
            jax.tree_util.tree_structure(state).unflatten(
                [leaf for ns in new_state for leaf in jax.tree_util.tree_leaves(ns)]
            ),
        )

    def _walk_grouped(params, grads, state, lr):
        """Один проход, Muon-часть — батчированными shape-группами (веха-5).

        Классификация та же (:func:`classify_leaf`, fail-closed на неизвестном
        2-D имени — как ``_walk_once``); Muon-листья одной группы и одной формы
        собираются в стёк и обрабатываются одним ``vmap``-NS5 на группу
        (:func:`_muon_group_step`).  Ключ группы — ``(группа, форма, dtypes)``:
        в один ``jnp.stack`` попадают только строго совместимые листья, паддинга
        нет (Am.11 — группы строго equal-shape).  Одиночный лист группы и все
        AdamW-листья идут прежним leaf-путём (побитово теми же вызовами).

        Группировка строится **на трассировке** (формы статичны) — под ``jit``
        она выполняется один раз, в графе остаются только batched-операции.

        bf16-мастер (Am.13) этой дорогой не идёт: связка отвергается на сборке
        шага (:class:`IncompatibleModeError`) — batched NS5 переупорядочивает
        редукции (Am.11), а в bf16-хранилище это перешагивает середину округления
        (фикс-раунд вехи-6), поэтому здесь предполагается fp32-дерево.
        """
        path_leaves, param_treedef = jax.tree_util.tree_flatten_with_path(params)
        grads_for = param_treedef.flatten_up_to(grads)
        state_for = param_treedef.flatten_up_to(state)
        leaf_groups: list[str] = []
        for path, p in path_leaves:
            name = leaf_name(path[-1]) if path else ""
            group = classify_leaf(name, p.ndim, legacy=legacy_muon_all_2d)
            if group == GROUP_UNCLASSIFIED:
                raise UnclassifiedMatrixError(
                    f"2-D параметр {name!r} ({jax.tree_util.keystr(path)}) вне "
                    "классификации ADR-048: дополни MUON_MATRIX_LEAVES или "
                    "ADAMW_MATRIX_LEAVES, либо включи legacy_muon_all_2d"
                )
            leaf_groups.append(group)
        # (группа, форма, dtypes) -> индексы плоских листьев, в порядке дерева.
        buckets: dict[tuple, list[int]] = {}
        for index, group in enumerate(leaf_groups):
            if group not in _MUON_GROUPS:
                continue
            p = path_leaves[index][1]
            key = (group, p.shape, p.dtype, grads_for[index].dtype,
                   _state_signature(state_for[index]))
            buckets.setdefault(key, []).append(index)

        new_params: list = [None] * len(path_leaves)
        new_state: list = [None] * len(path_leaves)
        for key, indices in buckets.items():
            group = key[0]
            # bf16-мастер сюда не доходит (IncompatibleModeError на сборке шага):
            # режим дерева несовместим с переупорядочиванием редукций batched NS5.
            leaf_call = _muon_leaf
            if len(indices) == 1:
                # Одиночная группа (одна такая форма на всю модель): прежний
                # leaf-вызов — побитово прежняя арифметика.
                index = indices[0]
                new_params[index], new_state[index] = leaf_call(
                    group, path_leaves[index][1], grads_for[index], state_for[index], lr
                )
                continue
            p_stack, state_stack = _muon_group_step(
                group,
                jnp.stack([path_leaves[i][1] for i in indices]),
                jnp.stack([grads_for[i] for i in indices]),
                # Лист состояния fp32-режима — момент или пара (m, v); стёк
                # строится по дереву и разбирается обратно по индексам.
                jax.tree_util.tree_map(
                    lambda *xs: jnp.stack(list(xs)), *[state_for[i] for i in indices]
                ),
                lr,
            )
            for k, index in enumerate(indices):
                new_params[index] = p_stack[k]
                new_state[index] = jax.tree_util.tree_map(lambda x, k=k: x[k], state_stack)
        adam_call = _adamw_step
        for index, group in enumerate(leaf_groups):
            if group in _MUON_GROUPS:
                continue
            new_params[index], new_state[index] = adam_call(
                path_leaves[index][1], grads_for[index], state_for[index], lr,
                clip=group == GROUP_ADAMW_EMBED,
            )
        return (
            param_treedef.unflatten(new_params),
            jax.tree_util.tree_structure(state).unflatten(
                [leaf for ns in new_state for leaf in jax.tree_util.tree_leaves(ns)]
            ),
        )

    def _walk_twice(params, grads, state, lr):
        """Pre-Amendment traversal (the parity oracle): two walks, twice the work.

        ``tree_map`` takes the structure of the *first* tree (params) and pulls
        the state values ``flatten_up_to`` it, so an Adam leaf's ``(m, v)`` pair
        arrives as one value and the returned tuple is re-flattened as the output
        tree's leaf.
        """
        new_params = jax.tree_util.tree_map_with_path(
            lambda path, p, g, s: _one(path, p, g, s, lr)[0], params, grads, state
        )
        new_state = jax.tree_util.tree_map_with_path(
            lambda path, p, g, s: _one(path, p, g, s, lr)[1], params, grads, state
        )
        return new_params, new_state

    def step(params, grads, state, lr):
        if bf16_master:
            # Fail-closed на вход: режим объявлен bf16-мастером, и fp32-дерево на
            # входе — ошибка вызывающего, а не «тихо другой режим» (иначе
            # применение плеча округлило бы fp32-дерево в bf16 без объявления).
            leaves = jax.tree_util.tree_leaves(params)
            if leaves and jnp.asarray(leaves[0]).dtype != jnp.bfloat16:
                raise ValueError(
                    "bf16-мастер (AXIOM_MASTER_DTYPE=bf16) получил не-bf16 дерево "
                    f"параметров: {jnp.asarray(leaves[0]).dtype}"
                )
        if not single_pass:
            # Пре-Amendment обход — оракул паритета «один проход ↔ два»; он
            # старше режима батчинга: grouped-путь — это и есть новый один проход.
            walk = _walk_twice
        elif batch == OPT_BATCH_GROUPED:
            walk = _walk_grouped
        else:
            walk = _walk_once
        new_params, new_state = walk(params, grads, state, lr)
        if work_dtype is None or bf16_master:
            # bf16-мастер: рабочая копия ЕСТЬ мастер (``params is master``) —
            # третьего выхода нет по построению (Am.13; ``work_dtype`` при
            # bf16-мастере отвергается на сборке).
            return new_params, new_state
        # ADR-054 (цикл-4): рабочая копия — часть того же графа; при donate=True
        # new_params — возврат шага (не донированный вход), каст легален.
        new_work = jax.tree_util.tree_map(
            lambda leaf: leaf.astype(work_dtype), new_params
        )
        return new_params, new_state, new_work

    if not jit:
        return step
    if donate:
        # ADR-054: master/grads/state потребляются шагом (владелец — возврат).
        return jax.jit(step, donate_argnums=(0, 1, 2))
    return jax.jit(step)


def classification_report(
    params,
    *,
    legacy_muon_all_2d: bool = False,
    examples: int = 3,
) -> dict:
    """Group -> {leaves, params, names, examples} over a parameter tree.

    ADR-048 п. 4: the classification is fixed by a report, so "how much of what"
    is a number and not a belief.  Accepts any pytree with ``ndim``/``size`` on
    its leaves — including the abstract tree of ``jax.eval_shape``, which lets
    the real ``l3-full`` geometry be reported without allocating it.

    Unlike :func:`make_step`/:func:`init_state`, this never raises on an
    unknown 2-D name: the point is to *record* the gap (``unclassified``),
    while those two fail closed.
    """
    groups = {
        group: {"leaves": 0, "params": 0, "names": [], "examples": []}
        for group in GROUP_NAMES
    }
    unclassified: list[str] = []

    def visit(path, leaf):
        name = leaf_name(path[-1]) if path else ""
        group = classify_leaf(name, leaf.ndim, legacy=legacy_muon_all_2d)
        entry = groups[group]
        entry["leaves"] += 1
        entry["params"] += int(leaf.size)
        if name not in entry["names"]:
            entry["names"].append(name)
        if group == GROUP_UNCLASSIFIED and name not in unclassified:
            unclassified.append(name)
        if len(entry["examples"]) < examples:
            entry["examples"].append(jax.tree_util.keystr(path))

    jax.tree_util.tree_map_with_path(visit, params)
    for entry in groups.values():
        entry["names"].sort()
    return {
        "legacy_muon_all_2d": bool(legacy_muon_all_2d),
        "groups": groups,
        "group_order": list(GROUP_NAMES),
        "unclassified": sorted(unclassified),
        "total_leaves": sum(entry["leaves"] for entry in groups.values()),
        "total_params": sum(entry["params"] for entry in groups.values()),
    }


def format_report(report: dict) -> str:
    """Human-readable table of :func:`classification_report` (markdown)."""
    lines = [
        "| группа | листьев | параметров | имена |",
        "|---|---:|---:|---|",
    ]
    for group in report["group_order"]:
        entry = report["groups"][group]
        names = ", ".join(entry["names"][:8])
        if len(entry["names"]) > 8:
            names += f", … (+{len(entry['names']) - 8})"
        lines.append(
            f"| {group} | {entry['leaves']} | {entry['params']} | {names or '—'} |"
        )
    lines.append("")
    lines.append(
        f"всего: листьев {report['total_leaves']}, параметров {report['total_params']}"
        f", legacy_muon_all_2d={report['legacy_muon_all_2d']}"
    )
    if report["unclassified"]:
        lines.append(
            "НЕ КЛАССИФИЦИРОВАНО (fail-closed): " + ", ".join(report["unclassified"])
        )
    return "\n".join(lines)


def ns5_shape_groups(params, *, legacy_muon_all_2d: bool = False, examples: int = 2) -> dict:
    """Muon-листья по (группа, форма) — расчёт NS5-цепочек режима ``grouped``.

    Отчёт-инструмент вехи-5 (ADR-054 Am.11): под ``AXIOM_OPT_BATCH=grouped``
    каждый кортеж (группа, форма) исполняется **одним** ``vmap``-вызовом NS5
    (листья одной формы собираются в стёк), лист-одиночка своей формы идёт
    прежним прямым вызовом.  ``chains_leaf`` — столько NS5-цепочек
    разворачивает ``leaf``-режим (по вызову на лист), ``chains_grouped`` —
    столько остаётся в ``grouped``.  Принимает и абстрактное дерево
    ``jax.eval_shape``: геометрия l3-full считается без аллокации.  Ключ здесь —
    (группа, форма) без dtypes (отчёт о геометрии); сам ``_walk_grouped``
    страхует совместимость стека по dtypes.
    """
    buckets: dict[tuple, dict] = {}
    muon_leaves = 0

    def visit(path, leaf):
        nonlocal muon_leaves
        group = classify_leaf(
            leaf_name(path[-1]) if path else "", leaf.ndim, legacy=legacy_muon_all_2d
        )
        if group not in _MUON_GROUPS:
            return
        muon_leaves += 1
        entry = buckets.setdefault(
            (group, tuple(leaf.shape)), {"group": group, "shape": list(leaf.shape), "leaves": 0, "examples": []}
        )
        entry["leaves"] += 1
        if len(entry["examples"]) < examples:
            entry["examples"].append(jax.tree_util.keystr(path))

    jax.tree_util.tree_map_with_path(visit, params)
    groups = sorted(buckets.values(), key=lambda e: (-e["leaves"], e["group"], e["shape"]))
    return {
        "legacy_muon_all_2d": bool(legacy_muon_all_2d),
        "groups": groups,
        "muon_leaves": muon_leaves,
        "chains_leaf": muon_leaves,
        "chains_grouped": len(groups),
        "batched_chains": sum(1 for entry in groups if entry["leaves"] > 1),
    }


def cosine_schedule(peak_lr: float, total_steps: int, warmup_ratio: float = 0.01, min_ratio: float = 0.0):
    """Cosine decay with a linear warmup; returns ``lr(step)`` as a function."""
    warmup_steps = max(1, int(total_steps * warmup_ratio))

    def lr_at(step: int) -> jnp.ndarray:
        step = jnp.asarray(step, dtype=jnp.float32)
        warm = peak_lr * (step / warmup_steps)
        t = (step - warmup_steps) / jnp.maximum(total_steps - warmup_steps, 1.0)
        t = jnp.clip(t, 0.0, 1.0)
        cos = peak_lr * (min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + jnp.cos(jnp.pi * t)))
        return jnp.where(step < warmup_steps, warm, cos)

    return lr_at
