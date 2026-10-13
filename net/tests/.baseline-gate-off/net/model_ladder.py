"""The model ladder for cheap prototyping — M0 ``tiny`` → M3 ``l3-full`` (ADR-036).

ADR-036 canonises four *rungs* of scale so that a prototyping run pays the price
of the question it asks, not the price of the target model (an SFT smoke on
``l3-full`` costs ~8 min of XLA compilation and 100-500 s/step; the same smoke on
``small`` costs seconds).  **Selection rule, one line:** start an experiment on
the lowest rung whose ``purpose`` covers its question; climb to the next rung
only on a green gate of the previous one (tests + smoke), the sole exception
being the target runs on M3 after the A4 gates.  The same rule mirrors
``docs/RUNBOOK-GB10.ru.md`` §8.

The ladder invariant is **architectural parity**: every rung carries the *full*
feature set of M3 (KDA + MLA + MoE + MTP + SiTi gates + AttnRes), differing only
in dimensions and layer count, with the KDA:MLA composition pinned at 3:1 and
the MoE topology at top-2.  A rung that switched a feature off would make its
conclusions non-transferable to M3 *by mechanism* — the integrity test
``tools/tests/test_model_ladder.py`` (T-ML-1..T-ML-6) is what makes the parity a
checked property instead of a claim.

This module is the **single carrier** of the ladder: the rungs, their factories,
the parameter estimates and the corridors are declared here and nowhere else.
The low rungs ``tiny``/``small`` duplicate the frozen smoke presets of
``net/tests/conftest.py`` (production code must not import a test module — the
test-side canon of ADR-010 is pinned against these factories by T-ML-2, field by
field); ``l3-full`` is built lazily by
``tools/run_sft_smoke.load_l3_full_config`` so importing the registry never
pulls the heavy runner.  The dense-124m reference is deliberately *not* a rung:
it answers an orthogonal architectural A/B question (MoE against dense), not a
question of scale (ADR-036, alternative 3).
"""

from __future__ import annotations

import dataclasses
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from net.config import ModelConfig

__all__ = [
    "LadderRung",
    "MODEL_LADDER",
    "get_rung",
    "tiny_config",
    "small_config",
    "mid_config",
    "l3_full_config",
]


@dataclass(frozen=True)
class LadderRung:
    """One rung of the ladder (ADR-036): a scale level with a fixed purpose.

    ``params_estimate`` is the *measured* total parameter count of the factory's
    config (``net.model.param_count``, ``jax.eval_shape`` — shapes only, no
    allocation), pinned here so that a silent drift of a factory shows up as a
    red T-ML-5 instead of a quietly wrong budget.  ``param_corridor`` is the
    accepted band around it — ``None`` for the target rung, which is outside
    corridor control by design (M3 is the target, not a prototype scale).
    ``factory`` is any callable returning a :class:`~net.config.ModelConfig` and
    accepting keyword ``overrides`` (the stage runners use that to substitute
    ``vocab_size``/``qat_enabled``, as they do for the frozen presets).
    """

    name: str
    level: str
    budget_hint: str
    purpose: str
    factory: Callable[..., ModelConfig]
    params_estimate: int | None
    param_corridor: tuple[int, int] | None


def small_config(**overrides) -> ModelConfig:
    """M1 rung: hidden 64 / vocab 512 skeleton, 3 KDA + 1 MLA, 4 layers.

    Mirrors ``net/tests/conftest.small_config`` field for field — the frozen
    smoke preset of the stage runners (ADR-010) is the canon, this is the
    production-side copy that T-ML-2 pins against it.  Any edit here without the
    matching edit in ``conftest`` (or vice versa) fails the integrity test.
    """
    base = dict(
        vocab_size=512,
        hidden=64,
        num_layers=4,  # 3 KDA + 1 MLA
        num_kda_layers=3,
        num_mla_layers=1,
        num_heads=4,
        head_dim=16,
        kda_dk=16,
        kda_dv=16,
        kda_decay_rank=16,
        kda_short_conv_kernel=4,
        kda_g_min=-5.0,
        mla_latent_dim=32,
        mla_head_dim=16,
        mlp_intermediate=128,
        siti_beta_gate=4.0,
        siti_beta_up=25.0,
        mtp_layers=1,
        mtp_loss_weight=0.1,
        attnres_blocks=1,
        attnres_block_size=4,
        vit_patch=14,
        vit_hidden=32,
        vit_depth=2,
        vit_heads=2,
        vit_mlp=64,
        image_size=56,
        moe_dense_layers=1,
        moe_latent_dim=32,
        moe_num_routed=6,
        moe_num_shared=2,
        moe_top_k=2,
        moe_expert_intermediate=16,
        moe_shared_intermediate=32,
        qb_weight=0.01,
        routing_seed=0,
    )
    base.update(overrides)
    return ModelConfig(**base)


def tiny_config(**overrides) -> ModelConfig:
    """M0 rung: the wire-test skeleton, hidden 16 / vocab 64, ~0.06M params.

    Mirrors ``net/tests/conftest.tiny_config``: the even smaller preset used by
    the overfit / needle-recall smokes, obtained from :func:`small_config` by
    shrinking every dimension but keeping the same 3 KDA + 1 MLA composition and
    the same invariants — ``num_heads * head_dim == hidden`` (the KDA output
    gate requires ``H * dv == hidden``) and
    ``kda_dk == kda_dv == mla_head_dim == head_dim``.
    """
    return small_config(
        vocab_size=64,
        hidden=16,
        num_heads=2,
        head_dim=8,
        kda_dk=8,
        kda_dv=8,
        kda_decay_rank=8,
        mla_latent_dim=16,
        mla_head_dim=8,
        mlp_intermediate=32,
        moe_latent_dim=8,
        moe_num_routed=4,
        moe_top_k=2,
        moe_num_shared=1,
        moe_expert_intermediate=16,
        moe_shared_intermediate=32,
        **overrides,
    )


def mid_config(**overrides) -> ModelConfig:
    """M2 rung: hidden 640, 8 layers (6 KDA + 2 MLA), LatentMoE 12+2 top-2, ~51M.

    ``small_config`` is the template, scaled up along the *ratios the existing
    presets pin* — the rung differs from M3 only by size, never by mechanism
    (ADR-036, «архитектурная парность»):

    ============================  =============  ==========  ==============================
    dimension                     l3-full (M3)   mid (M2)    ratio rule (from the presets)
    ============================  =============  ==========  ==============================
    ``hidden``                    1536           640         --
    ``num_heads * head_dim``      12 * 128       10 * 64     ``== hidden`` (validate_config)
    ``kda_decay_rank``            128            64          ``== head_dim``
    ``mla_latent_dim``            512            320         ``hidden // 2`` (small/tiny)
    ``mlp_intermediate``          4096           1280        ``2 * hidden`` (small/tiny)
    ``moe_latent_dim``            768            320         ``hidden // 2`` (all presets)
    ``moe_num_routed/shared/k``   12 / 2 / 2     12 / 2 / 2  l3 topology verbatim, top-2
    ``moe_expert_intermediate``   384            160         ``hidden // 4`` (l3/small)
    ``moe_shared_intermediate``   384            320         ``hidden // 2`` (small)
    ``moe_dense_layers``          1              1           leading dense SiTU-GLU layer
    ``num_layers`` (KDA:MLA)      24 (18:6)      8 (6:2)     whole ``[K,K,K,M]`` blocks, 3:1
    ``attnres_blocks/size``       2 / 12         2 / 4       ``size == num_layers // blocks``
    ``mtp_layers``                1              1           MTP head kept on
    ============================  =============  ==========  ==============================

    The result is 51,147,395 parameters (``net.model.param_count``), inside the
    rung's corridor (25M..80M) and inside the ~30-60M band ADR-036 quotes for
    M2.  The composition is whole ``[K,K,K,M]`` blocks, so the shape the schema
    validates (``tail % 4 == 0``) is the shape the layers build.
    """
    base = dict(
        vocab_size=512,
        hidden=640,
        num_layers=8,  # 6 KDA + 2 MLA — two whole [K,K,K,M] blocks
        num_kda_layers=6,
        num_mla_layers=2,
        num_heads=10,
        head_dim=64,
        kda_dk=64,
        kda_dv=64,
        kda_decay_rank=64,
        kda_short_conv_kernel=4,
        kda_g_min=-5.0,
        mla_latent_dim=320,
        mla_head_dim=64,
        mlp_intermediate=1280,
        siti_beta_gate=4.0,
        siti_beta_up=25.0,
        mtp_layers=1,
        mtp_loss_weight=0.1,
        attnres_blocks=2,
        attnres_block_size=4,
        vit_patch=14,
        vit_hidden=32,
        vit_depth=2,
        vit_heads=2,
        vit_mlp=64,
        image_size=56,
        moe_dense_layers=1,
        moe_latent_dim=320,
        moe_num_routed=12,
        moe_num_shared=2,
        moe_top_k=2,
        moe_expert_intermediate=160,
        moe_shared_intermediate=320,
        qb_weight=0.01,
        routing_seed=0,
    )
    base.update(overrides)
    return ModelConfig(**base)


def l3_full_config(**overrides) -> ModelConfig:
    """M3 rung: the target skeleton, built lazily from ``net/config.json``.

    The runner's ``load_l3_full_config`` reads the declarative config through
    ``net.config.load_config`` (AD-9/C-035: the file is the switch), so the
    architecture numbers are not duplicated here.  The import lives *inside* the
    function — importing the registry must stay cheap, and the runner pulls the
    stage machinery (journal, tokenizer, checkpointing) that a rung lookup has
    no business loading.
    """
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from tools.run_sft_smoke import load_l3_full_config

    cfg = load_l3_full_config()
    if overrides:
        cfg = dataclasses.replace(cfg, **overrides)
    return cfg


#: The ladder, in climb order M0 → M3 (ADR-036).  ``name`` is the identity the
#: stage runners resolve (``--model-preset``), ``level`` the M-number the ADR
#: and RUNBOOK §8 quote.
MODEL_LADDER: tuple[LadderRung, ...] = (
    LadderRung(
        name="tiny",
        level="M0",
        budget_hint="CPU-секунды",
        purpose=(
            "wire-тесты конвейера, паритет-инварианты, CI, контракт записей"
        ),
        factory=tiny_config,
        params_estimate=64_737,
        param_corridor=(42_000, 78_000),
    ),
    LadderRung(
        name="small",
        level="M1",
        budget_hint="минуты (CPU/GPU)",
        purpose=(
            "смоуки стадий, RL-роллаут смоук, отладка фазового профиля, дески"
        ),
        factory=small_config,
        params_estimate=340_583,
        # ADR-036/RUNBOOK quote «~1-2M» for M1; the frozen conftest preset the
        # rung must equal counts 340,583 (T-ML-2 pins the equality), so the
        # corridor follows the factory, not the ADR's rounded figure.
        param_corridor=(238_000, 443_000),
    ),
    LadderRung(
        name="mid",
        level="M2",
        budget_hint="десятки минут GPU",
        purpose=(
            "калибровочные кривые, гиперпараметрические свипы, "
            "методологические пробы, BPB-кривая лесенки"
        ),
        factory=mid_config,
        params_estimate=51_147_395,
        param_corridor=(35_700_000, 66_300_000),
    ),
    LadderRung(
        name="l3-full",
        level="M3",
        budget_hint="часы-дни GB10, аренда H100",
        purpose=(
            "целевые прогоны; только после зелёных гейтов M1/M2"
        ),
        factory=l3_full_config,
        # Pinned from net/config.json (actual_param_count, the 1.01B MoE target).
        params_estimate=1_009_632_913,
        # The target rung is the reference, not a prototype scale: no corridor.
        param_corridor=None,
    ),
)

_BY_NAME: dict[str, LadderRung] = {rung.name: rung for rung in MODEL_LADDER}


def get_rung(name: str) -> LadderRung:
    """The rung called ``name`` (KeyError listing what the ladder carries)."""
    try:
        return _BY_NAME[name]
    except KeyError:
        available = ", ".join(rung.name for rung in MODEL_LADDER)
        raise KeyError(
            f"unknown ladder rung {name!r}; available rungs: {available}"
        ) from None
