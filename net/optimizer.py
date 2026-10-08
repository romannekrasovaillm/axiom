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
(``vocab x hidden``), where orthogonalisation has no meaning.  ADR-048 routes
them to AdamW:

============================  ============================================
group                         leaves
============================  ============================================
``muon_matrix``               hidden 2-D projections (attention/MLA/MoE
                              router/MLP/MTP/AttnRes/ViT)
``muon_per_head``             Q/K/V-family projections (``_PER_HEAD_LEAVES``)
``muon_batched``              stacked expert weights (``ndim >= 3``)
``adamw_vector``              norms, biases, scalars (``ndim <= 1``)
``adamw_embed``               input embeddings and the (tied) LM head
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
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

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
    # ViT tower (image tokens only).
    "patch_embed", "pos_embed", "projector", "qkv", "out", "fc1", "fc2",
})

#: 2-D leaves Muon must not touch: the input embedding matrix.  The LM head is
#: *tied* to it (``tie_embeddings``), so one named leaf covers both — AdamW
#: (ADR-048), as in Muon recipes.
ADAMW_MATRIX_LEAVES = frozenset({"embedding"})

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


class UnclassifiedMatrixError(ValueError):
    """A 2-D leaf is outside the classification — fail-closed (ADR-048).

    Raised instead of falling back to Muon: silently orthogonalising a new
    matrix is exactly the defect ADR-048 removes (Muon on embeddings/LM head),
    and silently switching it to AdamW would hide a missing entry from the
    report.  Fix by adding the leaf name to :data:`MUON_MATRIX_LEAVES` or
    :data:`ADAMW_MATRIX_LEAVES` (or by running with ``legacy_muon_all_2d``).
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


def init_state(params, *, legacy_muon_all_2d: bool = False):
    """Momentum (Muon groups) / AdamW m,v (Adam groups) state tree over ``params``.

    The group of every leaf comes from :func:`classify_leaf`, so the state
    mirrors the optimizer branch that will actually run: an embedding moved to
    AdamW gets an ``(m, v)`` pair instead of a single momentum buffer.
    """
    def _init(path, leaf):
        group = classify_leaf(leaf_name(path[-1]), leaf.ndim, legacy=legacy_muon_all_2d)
        if group == GROUP_UNCLASSIFIED:
            raise UnclassifiedMatrixError(
                f"2-D параметр {leaf_name(path[-1])!r} ({jax.tree_util.keystr(path)}) "
                "вне классификации ADR-048: дополни MUON_MATRIX_LEAVES или "
                "ADAMW_MATRIX_LEAVES, либо включи legacy_muon_all_2d"
            )
        if group in _ADAMW_GROUPS:
            return (jnp.zeros_like(leaf), jnp.zeros_like(leaf))
        return jnp.zeros_like(leaf)
    return jax.tree_util.tree_map_with_path(_init, params)


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
    """
    if int(ns_steps) < 1:
        raise ValueError(f"ns_steps must be >= 1, got {ns_steps!r}")
    heads = cfg.num_heads
    wd = cfg.weight_decay

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
        if group == GROUP_MUON_PER_HEAD:
            return _matrix_step(p, g, s, lr, True)  # (new_p, momentum)
        if group == GROUP_MUON_MATRIX:
            return _matrix_step(p, g, s, lr, False)  # (new_p, momentum)
        if group == GROUP_MUON_BATCHED:
            return _batched_matrix_step(p, g, s, lr)  # (new_p, momentum)
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
        walk = _walk_once if single_pass else _walk_twice
        return walk(params, grads, state, lr)

    return jax.jit(step) if jit else step


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
