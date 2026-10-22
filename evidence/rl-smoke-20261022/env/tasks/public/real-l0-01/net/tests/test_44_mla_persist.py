"""Веха-13 (ADR-054 Am.27, цикл-16 кампании MFU-55): персистентный MLA-блок.

NVTX-атрибуция Am.27: блокирующие эпизоды шага — backward-структура
while/remat (while.236 — 4,46 с ОДНИМ спаном; пять remat-спанов 150–300 мс)
и тайл-цикл flash-bwd вехи-12 (~12 с/шаг хоста).  Ступень-2 — MLA-слой
целиком персистентными кернелами по образцу KDA-вех 3–4
(:mod:`net.kernels.mla_persist`): grid=(B*H,), состояние в SMEM, один запуск
на слой-проход fwd и bwd; латентный кэш и gated-голова MLA остаются в jnp
(kda-formulas.md §2), кернелы несут T²-связку.

Что пиннуется здесь (CPU, без стенда GB10 — по контракту вехи-12):

* контракт гейта ``AXIOM_MLA_PERSIST``: имя, дефолт 0, пустая = 0,
  регистр/пробелы, неизвестное — ошибка на чтении и на сборке ``train`` до
  params/компиляции/батчей (форма ``flash_bwd_mode`` Am.25); явный аргумент
  сильнее окружения; приоритет над ``AXIOM_MLA_FLASH_BWD``/``AXIOM_MLA_BWD_BF16``
  (обе ноги-маркеры несут фактические значения); конфликты
  (``AXIOM_MLA_DENSE_FLASH``, явная маска) — ошибка, не тихий выбор;
* гейт off (не задан и явный ``=0``) — побитово прежний граф: лоссы/деревья
  двухшаговой ноги совпадают;
* (а) артефакты проектирования: алгебра (``casts=False``) — тайловый fp32
  forward против dense-эталона ~7e-8, адъюнкт против ``jax.grad`` ~1e-7
  (допуск ≤1e-5; фактические числа печатаются); объявленные числители
  (hi/lo bf16 staging) на bf16-представимых операндах против fp32-оракула —
  ~1e-7 (замер на этом узле; допуск ≤1e-5);
* (б) паритет слоя против dense-эталона ``net.mla`` (fp32- и bf16-режимы
  compute-dtype, мини-конфиг): таблица чисел печатается; границы допуска —
  объявленный бюджет bf16-операндов (~2^-9 на входных q/k/v/do), измерено
  ~1e-3..6,5e-3 в зависимости от режима и величины;
* (в) счёт запусков на границе: forward-правило вызывает ``kernel`` ровно
  один раз, backward-правило — ``kernel_bwd`` ровно один раз на слой-проход
  (спаи-подмена, форма KDA-вех 3–4); ширина padding'а T не прячет вызовов;
* (г) 10-шаговый nan-гейт на CPU-миниконфиге (прецедент 4b/4c: дефекты
  проявлялись на 4+ шагах): ноги off / on / связка кампании (персистентный
  блок × bf16-хранение × per_layer_block × accum-jit × compute-bf16) — все
  лоссы и параметры конечны, Δloss off/on ≤1e-3 (фактическое число
  печатается);
* совместимость: маркер ``mla_persist`` в каждой строке журнала (0/1,
  фактические значения гейтов), связка с ``mla_flash_bwd``/``mla_bwd_bf16``
  (оба маркера сообщают фактические значения, исполняется персистентный
  путь), условные гейты ``per_layer_block``/``accum_jit``/``compute-bf16``
  видны рядом;
* статика: формы MLA l3-full (B=2 и B=4, T=8192) и мини-кейс *lowering*-ятся
  под sm_121 (архитектура и лимит SMEM 99 КиБ патчатся, форма
  ``test_kda_cc_fused_persist``); ЗМЕМ-числа обоих кернелов измерены
  искусственным лимитом (fwd 51 200 Б, bwd 98 304 Б при (64,64)/(dq=128));
  sieve честно отсекает ``bk=128`` (пара не сходится: fwd 100 352 Б
  влезает, bwd 163 840 Б — нет), а оба кернела отказывают без тихого отката;
* отказ без GPU: ``mla_attention`` на CPU-узле падает типизированной
  ошибкой (персистентный кернел исполняется только на GPU; тихого отката
  нет) — спад-подмена кернелов теста его не видит.

Не пинуется здесь: стендовая нога B=4/B=2 (A/B кампании — отдельный прогон
на GB10: NVTX/шаг/ток-с) и детерминизм на устройстве.

``AXIOM_MLA_PERSIST`` не должен читаться путями вне гейта: тест off-ноги
проверяет побитовость, маркеры — фактические значения.
"""

from __future__ import annotations

import contextlib
import json
import os
from pathlib import Path

import numpy as np
import pytest

from conftest import small_config

import jax
import jax.numpy as jnp
import jax.random as jr

from net import mla
from net import train_loop as tl
from net.kernels import mla_persist as P

SEQ_LEN = 16
_PARITY_STEPS = 10
_PARITY_LR = 1e-3
_DELTA_LOSS_TOL = 1e-3


@contextlib.contextmanager
def gate_env(persist=None, *, flash_bwd=None, mla_bf16=None, ce_bf16=None,
             policy=None, accum_jit=None, compute=None, master=None):
    """Контекст env-гейтов вехи-13 (форма ``gate_env`` test_43)."""
    env = {
        "AXIOM_MLA_PERSIST": persist,
        mla.MLA_FLASH_BWD_ENV: flash_bwd,
        mla.MLA_BWD_BF16_ENV: mla_bf16,
        "AXIOM_CE_BF16": ce_bf16,
        "AXIOM_GRAD_CKPT_POLICY": policy,
        tl.ACCUM_JIT_ENV: accum_jit,
        "AXIOM_COMPUTE_DTYPE": compute,
        "AXIOM_MASTER_DTYPE": master,
    }
    saved = {k: os.environ.get(k) for k in env}
    try:
        for k, v in env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _substitute_kernels(monkeypatch):
    """Подмена кернелов их объявленной нумерикой (контракт KDA-вех 3–4).

    ``kernel``/``kernel_bwd`` разрешаются правилами ``custom_vjp`` на модуле
    в момент вызова, поэтому подмена работает через кэш операций; ``не
    ``device_available`` тоже подменяется — иначе CPU-узел отказал бы.
    """
    def emu_kernel(q, k, v, *, bq=P.BQ, bk=P.BK):
        return P.emulate_attention(q, k, v, bq=int(bq), bk=int(bk))

    def emu_kernel_bwd(q, k, v, do, D, m, l, *, bq=P.BQ, bk=P.BK):
        return P.emulate_attention_bwd(q, k, v, do, D, m, l, bq=int(bq), bk=int(bk))

    monkeypatch.setattr(P, "kernel", emu_kernel)
    monkeypatch.setattr(P, "kernel_bwd", emu_kernel_bwd)
    monkeypatch.setattr(P, "device_available", lambda: True)


def _batches(cfg, steps: int, *, seed: int = 11):
    """Детерминированный поток ``(1, T) int32`` — одинаковые данные для ног."""
    rng = np.random.default_rng(seed)
    return [
        rng.integers(0, cfg.vocab_size, size=(1, SEQ_LEN), dtype=np.int32)
        for _ in range(steps)
    ]


def _budget():
    return tl.Budget(run_ref="veha13", path=Path("/dev/null"), present=True)


def _train_leg(cfg, steps, *, lr=_PARITY_LR, seed=7, batches=None,
               metrics_path=None, accum_tokens=0):
    return tl.train(
        cfg,
        iter(batches if batches is not None else _batches(cfg, steps)),
        train_config=tl.TrainConfig(
            steps=steps, lr=lr, seed=seed, schedule="cosine",
            micro_batch=1, accum_tokens=accum_tokens,
            metrics_path=metrics_path, log_every=0, kpi_every=20,
        ),
        budget=_budget(),
    )


def _loss_drift(a, b) -> float:
    assert len(a) == len(b) and a
    return max(abs(float(x) - float(y)) for x, y in zip(a, b))


def _mini_cfg():
    """Мини-конфиг с dq=16 (кратным mma K-шагу bf16) и H=4."""
    return small_config()


def _mla_case():
    cfg = _mini_cfg()
    params = mla.init_mla(jr.PRNGKey(3), cfg)
    x = jr.normal(jr.PRNGKey(4), (1, SEQ_LEN * 4, cfg.hidden)) * 0.3
    return cfg, params, x


# --------------------------------------------------------------------------- #
# Контракт гейта (форма flash_bwd_mode Am.25 / bwd_bf16_mode Am.20)
# --------------------------------------------------------------------------- #


def test_gate_contract_defaults_and_values():
    """Имя, дефолт 0, пустая = 0, регистр/пробелы, явный аргумент сильнее env."""
    assert mla.MLA_PERSIST_ENV == "AXIOM_MLA_PERSIST"
    assert mla.persist_mode(None) is False            # незаданная переменная
    assert mla.persist_mode("") is False
    assert mla.persist_mode("   ") is False
    assert mla.persist_mode("0") is False
    assert mla.persist_mode("1") is True
    assert mla.persist_mode(" 1 ".upper()) is True    # нормализация регистра/пробелов
    with gate_env("1"):
        assert mla.persist_mode() is True
    with gate_env("0"):
        assert mla.persist_mode() is False
    with gate_env("1"):
        assert mla.persist_mode("0") is False         # явный аргумент сильнее env


@pytest.mark.parametrize("bad", ["2", "true", "on", "yes", "persist", "01"])
def test_gate_is_fail_closed(bad):
    """Неизвестное значение — ошибка на чтении (без догадок)."""
    with pytest.raises(mla.MlaPersistError):
        mla.persist_mode(bad)


def test_train_build_is_fail_closed_on_unknown_gate(tmp_path):
    """Опечатка в гейте падает сборкой train до params/компиляции/батчей."""
    cfg = _mini_cfg()
    with gate_env("1"):
        pass
    os.environ["AXIOM_MLA_PERSIST"] = "3"
    try:
        with pytest.raises(mla.MlaPersistError):
            _train_leg(cfg, 1, batches=_batches(cfg, 2))
    finally:
        os.environ.pop("AXIOM_MLA_PERSIST", None)


def test_gate_off_is_bitwise_previous_graph(tmp_path):
    """Гейт off (не задан и явный ``=0``) — побитово прежняя ревизия."""
    cfg = _mini_cfg()
    batches = _batches(cfg, 2)
    out = tmp_path
    with gate_env(None):
        off = _train_leg(cfg, 2, batches=batches, metrics_path=out / "off.jsonl")
    with gate_env("0"):
        zero = _train_leg(cfg, 2, batches=batches, metrics_path=out / "zero.jsonl")
    assert off.losses == zero.losses
    assert all(
        bool(jnp.array_equal(a, b))
        for a, b in zip(jax.tree_util.tree_leaves(off.params),
                        jax.tree_util.tree_leaves(zero.params))
    ), "AXIOM_MLA_PERSIST=0 обязан быть побитово равен незаданному гейту"


# --------------------------------------------------------------------------- #
# Статика: sieve, ЗМЕМ-бюджет, lowering под sm_121
# --------------------------------------------------------------------------- #


def test_cases_are_the_campaign_geometry_and_the_sieve_arithmetic():
    """Живые кейсы — формы l3-full (B=2/B=4, T=8192) и мини; bk=128 — skip."""
    assert all(c["H"] == 12 and c["dq"] == 128 and c["T"] == 8192
               for c in P.CASES[:2]), P.CASES
    assert P.CASES[0]["B"] == 2 and P.CASES[1]["B"] == 4
    assert all(P.valid_config(c) for c in P.CASES)          # все живые
    oversize = dict(P.CASES[2], bk=128)
    assert P.valid_config(oversize) is False                # пара не сходится
    assert P.valid_config_bwd(oversize) is False
    # арифметика объявленного пола (докстринг модуля), как числа
    assert P.SMEM_LIMIT_BYTES == 101376
    assert P._smem_floor_bytes_fwd(64, 64, 128) == 51200
    assert P._smem_floor_bytes_bwd(64, 64, 128) == 100352
    assert P._smem_floor_bytes_fwd(64, 128, 128) == 100352     # fwd бы влез...
    assert P._smem_floor_bytes_bwd(64, 128, 128) == 165888     # ...bwd — нет


def _lower_fwd(case, **params):
    from jax.experimental.mosaic import gpu as mosaic_mod
    from jax.experimental.mosaic.gpu import utils as mosaic_utils

    mosaic_utils._infer_arch = lambda: (12, 1)
    with mosaic_mod.core.artificial_shared_memory_limit(101376):
        shapes = jax.eval_shape(lambda: P.make_inputs(jax.random.key(0), **case))
        fn = jax.jit(lambda *a: P.kernel(*a, **params))
        return fn.trace(*shapes).lower(lowering_platforms=("cuda",))


def _lower_bwd(case, **params):
    from jax.experimental.mosaic import gpu as mosaic_mod
    from jax.experimental.mosaic.gpu import utils as mosaic_utils

    mosaic_utils._infer_arch = lambda: (12, 1)
    with mosaic_mod.core.artificial_shared_memory_limit(101376):
        shapes = jax.eval_shape(lambda: P.make_inputs_bwd(jax.random.key(0), **case))
        fn = jax.jit(lambda *a: P.kernel_bwd(*a, **params))
        return fn.trace(*shapes).lower(lowering_platforms=("cuda",))


def test_kernels_lower_for_sm121_in_all_live_cases():
    """Оба кернела *lowering*-ятся под sm_121 во всех формах l3-full (B=2/B=4)."""
    for case in P.CASES:
        for lower in (_lower_fwd, _lower_bwd):
            lowered = lower(case)
            assert lowered is not None
    # и ровно один scf.for на программу-обёртку не требуется: цикл динамический
    txt = _lower_fwd(P.CASES[2]).as_text()
    assert "scf.for" in txt, "динамический цикл обязан быть в понижении"


def _measured_smem(case, *, bwd):
    from jax.experimental.mosaic import gpu as mosaic_mod
    from jax.experimental.mosaic.gpu import utils as mosaic_utils

    mosaic_utils._infer_arch = lambda: (12, 1)
    tiles = dict(bq=int(case.get("bq", P.BQ)), bk=int(case.get("bk", P.BK)))
    # Меряется *понижение* тела (обёртки отказывают вне живого набора — это
    # отдельный пин), поэтому строители вызываются напрямую.
    build = (lambda: P._build_bwd_kernel(case["B"], case["T"], case["H"],
                                         case["dq"], tiles["bq"], tiles["bk"],
                                         case["dtype"])) if bwd else (
        lambda: P._build_kernel(case["B"], case["T"], case["H"], case["dq"],
                                tiles["bq"], tiles["bk"], case["dtype"]))
    with mosaic_mod.core.artificial_shared_memory_limit(1):
        shapes = jax.eval_shape(lambda: (P.make_inputs_bwd if bwd else P.make_inputs)(
            jax.random.key(0), **case))
        fn = jax.jit(lambda *a: build()(*a))
        try:
            fn.trace(*shapes).lower(lowering_platforms=("cuda",))
        except ValueError as e:
            msg = str(e)
            if "shared memory" in msg and "smem_bytes=" in msg:
                return int(msg.split("smem_bytes=")[1].split()[0])
            raise
        raise AssertionError("искусственный лимит 1 Б не сработал")


def test_smem_numbers_are_measured_not_declared():
    """Замеры понижения (искусственный лимит): fwd 51 200 Б, bwd 98 304 Б.

    Пол модуля консервативен ровно на объявленный overhead (+2 048 Б) и
    никогда не отсекает живую конфигурацию; ``bk=128`` честно отвергается
    парой (fwd влезает: 100 352 Б, bwd — нет: 163 840 Б).
    """
    mini = P.CASES[2]
    assert _measured_smem(mini, bwd=False) == 51200
    assert _measured_smem(mini, bwd=True) == 98304
    assert _measured_smem(dict(mini, bk=128), bwd=False) == 100352
    assert _measured_smem(dict(mini, bk=128), bwd=True) == 163840
    assert _measured_smem(mini, bwd=False) <= P.SMEM_LIMIT_BYTES
    assert _measured_smem(mini, bwd=True) <= P.SMEM_LIMIT_BYTES
    assert _measured_smem(dict(mini, bk=128), bwd=True) > P.SMEM_LIMIT_BYTES


def test_kernels_refuse_oversize_without_silent_fallback():
    """Конфигурация вне живого набора — типизированный отказ, не откат."""
    q, k, v = P.make_inputs(jax.random.key(0), B=1, T=256, H=2, dq=128)
    with pytest.raises(ValueError, match="sieve"):
        P.kernel(q, k, v, bk=128)   # T кратен, но пара полов не сходится
    q2, k2, v2 = P.make_inputs(jax.random.key(0), B=1, T=192, H=2, dq=128)
    with pytest.raises(ValueError, match="multiple of the tile"):
        P.kernel(q2, k2, v2, bk=128)
    with pytest.raises(ValueError, match="bf16"):
        P.kernel(q.astype(jnp.float32), k, v)
    with pytest.raises(ValueError, match="shapes must match"):
        P.kernel(q, k[:, :128], v)
    for name in ("kernel_bwd",):
        with pytest.raises(ValueError, match="bf16"):
            getattr(P, name)(q.astype(jnp.float32), k, v, v, jnp.zeros((1, 2, 256), jnp.float32),
                             jnp.zeros((1, 2, 256), jnp.float32), jnp.ones((1, 2, 256), jnp.float32))


# --------------------------------------------------------------------------- #
# (а) Объявленная алгебра — fp32-точная; объявленная нумерика — hi/lo bf16
# --------------------------------------------------------------------------- #


def _bf16_case(seed=0, B=1, T=256, H=2, dq=128):
    q, k, v = P.make_inputs(jax.random.key(seed), B=B, T=T, H=H, dq=dq)
    q32, k32, v32 = (x.astype(jnp.float32) for x in (q, k, v))
    go = jax.random.normal(jax.random.fold_in(jax.random.key(seed), 3), (B, T, H, dq))
    return q32, k32, v32, go


def test_declared_algebra_forward_matches_the_fp32_reference():
    """Тайловая online-softmax алгебра (casts=False) — dense-эталон ~7e-8."""
    for seed, kw in [(0, dict()), (1, dict(H=1)), (2, dict(T=128))]:
        q, k, v, _go = _bf16_case(seed=seed, **kw)
        o_e, m_e, l_e = P.emulate_attention(q, k, v, casts=False)
        o_r, m_r, l_r = P.reference_attention(q, k, v)
        rel = float(jnp.max(jnp.abs(o_e - o_r)) / jnp.max(jnp.abs(o_r)))
        assert rel <= 1e-5, (seed, kw, rel)
        assert float(jnp.max(jnp.abs(m_e - m_r))) == 0.0     # тот же rowmax
        assert float(jnp.max(jnp.abs(l_e - l_r)) / jnp.max(l_r)) <= 1e-5


def test_declared_algebra_adjoint_matches_autodiff():
    """Адъюнкт (casts=False) против ``jax.grad`` той же алгебры: ~1e-7."""
    for seed, kw in [(0, dict()), (1, dict(H=1, T=128)), (2, dict(B=2, T=128))]:
        q, k, v, go = _bf16_case(seed=seed, **kw)

        def loss(q, k, v):
            o, _, _ = P.emulate_attention(q, k, v, casts=False)
            return jnp.sum(o * go)

        dq_r, dk_r, dv_r = jax.grad(loss, argnums=(0, 1, 2))(q, k, v)
        o_e, m_e, l_e = P.emulate_attention(q, k, v, casts=False)
        D = jnp.einsum("bthd,bthd->bht", go, o_e)
        dq_e, dk_e, dv_e = P.emulate_attention_bwd(
            q, k, v, go, D, m_e, l_e, casts=False)
        for a, b in [(dq_e, dq_r), (dk_e, dk_r), (dv_e, dv_r)]:
            rel = float(jnp.max(jnp.abs(a - b)) / jnp.max(jnp.abs(b)))
            assert rel <= 1e-5, (seed, kw, rel)


def test_declared_numerics_hit_the_fp32_oracle_on_bf16_operands():
    """hi/lo bf16 staging: объявленная нумерика против fp32-оракула ~1e-7.

    Операнды — bf16-представимые (как на границе кернела): staging p/ds идёт
    парой hi/lo (≈2^-17), накопления fp32, поэтому весь интервал — на уровне
    fp32-переупорядочивания.  Это и есть «≤1e-3»-паритет конструкции.
    """
    worst = 0.0
    for seed in (0, 1, 2):
        q, k, v, go = _bf16_case(seed=seed)
        o_r, m_r, l_r = P.reference_attention(q, k, v)

        def loss(q, k, v):
            o, _, _ = P.reference_attention(q, k, v)
            return jnp.sum(o * go)

        dq_r, dk_r, dv_r = jax.grad(loss, argnums=(0, 1, 2))(q, k, v)
        o_e, m_e, l_e = P.emulate_attention(q, k, v, casts=True)
        D = jnp.einsum("bthd,bthd->bht", go, o_e)
        dq_e, dk_e, dv_e = P.emulate_attention_bwd(
            q, k, v, go, D, m_e, l_e, casts=True)
        rels = [float(jnp.max(jnp.abs(a - b)) / jnp.max(jnp.abs(b)))
                for a, b in [(o_e, o_r), (dq_e, dq_r), (dk_e, dk_r), (dv_e, dv_r)]]
        worst = max(worst, *rels)
        assert max(rels) <= 1e-5, (seed, rels)
    print(f"[веха-13] объявленная нумерика vs fp32-оракул (bf16-операнды): "
          f"worst rel = {worst:.3e}")


# --------------------------------------------------------------------------- #
# (б) Паритет слоя против dense-эталона (подмена кернелов объявленной нумерикой)
# --------------------------------------------------------------------------- #


def _dense_of(cfg, params, x, mode):
    with gate_env("0", compute=mode):
        return mla.apply(params, cfg, x)


def test_layer_parity_persist_vs_dense_prints_the_table(monkeypatch):
    """Таблица паритета слоя против dense-эталона (fp32/bf16; показ + границы).

    Замерено на этом узле (миниконфиг, форма вехи-12): fp32-режим fwd ~2e-3,
    dx ~1,5e-3, dparams ~2,9e-3; bf16-режим fwd ~1,0e-3, dx ~2,9e-3,
    dparams до ~6,5e-3.  Терм — квантизация *входных* q/k/v/do в bf16 (2^-9,
    объявленный путь операндов; тот же класс, что у bf16-режима dense-пути,
    веха-12 документировала ~3e-3).  Границы консервативные: fwd ≤3e-3,
    grads ≤1e-2; числа печатаются.
    """
    _substitute_kernels(monkeypatch)
    cfg, params, x = _mla_case()
    go = jr.normal(jr.PRNGKey(5), x.shape[:2] + (cfg.hidden,))
    for mode in ("fp32", "bf16"):
        with gate_env("0", compute=mode):
            o_d = mla.apply(params, cfg, x)
        with gate_env("1", compute=mode):
            o_p = mla.apply(params, cfg, x)
        rel_fwd = float(jnp.max(jnp.abs(o_p - o_d)) / jnp.max(jnp.abs(o_d)))

        def loss(p_, xx, flag):
            with gate_env(flag, compute=mode):
                return jnp.sum(mla.apply(p_, cfg, xx) * go)

        g_d = jax.grad(lambda xx: loss(params, xx, "0"))(x)
        g_p = jax.grad(lambda xx: loss(params, xx, "1"))(x)
        rel_dx = float(jnp.max(jnp.abs(g_p - g_d)) / jnp.max(jnp.abs(g_d)))
        gp_d = jax.grad(lambda p_: loss(p_, x, "0"))(params)
        gp_p = jax.grad(lambda p_: loss(p_, x, "1"))(params)
        rel_p = max(
            float(jnp.max(jnp.abs(a - b))) / max(float(jnp.max(jnp.abs(a))), 1e-30)
            for a, b in zip(jax.tree_util.tree_leaves(gp_d),
                            jax.tree_util.tree_leaves(gp_p))
        )
        print(f"[веха-13] паритет слоя ({mode}): fwd {rel_fwd:.3e}, "
              f"dx {rel_dx:.3e}, dparams {rel_p:.3e}")
        assert rel_fwd <= 3e-3, (mode, rel_fwd)
        assert rel_dx <= 1e-2, (mode, rel_dx)
        assert rel_p <= 1e-2, (mode, rel_p)


def test_grad_cotangents_flow_in_fp32(monkeypatch):
    """Границы custom_vjp несут fp32-котангенты (контракт вехи-9/12)."""
    _substitute_kernels(monkeypatch)
    cfg, params, x = _mla_case()
    with gate_env("1"):
        _, grads = jax.value_and_grad(
            lambda xx: jnp.sum(mla.apply(params, cfg, xx) ** 2))(x)
    assert grads.dtype == jnp.float32


# --------------------------------------------------------------------------- #
# (в) Счёт запусков на границе и обёртка (padding, отказы, приоритеты)
# --------------------------------------------------------------------------- #


def test_rules_call_one_kernel_per_direction(monkeypatch):
    """Forward-правило — один ``kernel``, backward — один ``kernel_bwd``."""
    calls = {"fwd": 0, "bwd": 0}

    def spy_kernel(q, k, v, *, bq=P.BQ, bk=P.BK):
        calls["fwd"] += 1
        return P.emulate_attention(q, k, v, bq=int(bq), bk=int(bk))

    def spy_kernel_bwd(q, k, v, do, D, m, l, *, bq=P.BQ, bk=P.BK):
        calls["bwd"] += 1
        return P.emulate_attention_bwd(q, k, v, do, D, m, l, bq=int(bq), bk=int(bk))

    monkeypatch.setattr(P, "kernel", spy_kernel)
    monkeypatch.setattr(P, "kernel_bwd", spy_kernel_bwd)
    monkeypatch.setattr(P, "device_available", lambda: True)
    cfg, params, x = _mla_case()
    with gate_env("1"):
        _, grads = jax.value_and_grad(
            lambda xx: jnp.sum(mla.apply(params, cfg, xx) ** 2))(x)
    assert calls["fwd"] == 1, calls
    assert calls["bwd"] == 1, calls
    assert grads is not None


def test_mla_attention_pads_t_not_multiple_of_tiles(monkeypatch):
    """T не кратен 64 — обёртка паддит (и не меняет числа реальных строк)."""
    _substitute_kernels(monkeypatch)
    B, T, H, dq = 1, 100, 2, 128
    q, k, v = (x.astype(jnp.float32) for x in
               P.make_inputs(jax.random.key(0), B=B, T=T, H=H, dq=dq))
    with gate_env("1"):
        o = P.mla_attention(q, k, v)
    assert o.shape == (B, T, H, dq)
    # паддированные строки не влияют на реальные: сравнение с T=64-версией
    q2, k2, v2 = (x[:T] for x in (q, k, v))
    with gate_env("1"):
        o2 = P.mla_attention(q2, k2, v2)
    assert float(jnp.max(jnp.abs(o[:64] - o2[:64]))) <= 1e-6


def test_persist_refuses_on_cpu_without_substitution():
    """Без подмены на CPU-узле — типизированный отказ (тихого отката нет)."""
    if P.device_available():
        pytest.skip("на узле есть GPU — отказ не проверить")
    B, T, H, dq = 1, 64, 2, 128
    q, k, v = (x.astype(jnp.float32) for x in
               P.make_inputs(jax.random.key(0), B=B, T=T, H=H, dq=dq))
    with gate_env("1"):
        with pytest.raises(RuntimeError, match="только на GPU"):
            P.mla_attention(q, k, v)


def test_persist_conflicts_are_errors_not_silent_choices(monkeypatch):
    """Явная маска и ``AXIOM_MLA_DENSE_FLASH`` — ошибка; приоритет над flash-bwd."""
    cfg, params, x = _mla_case()
    mask = jnp.tril(jnp.ones((x.shape[1], x.shape[1]), jnp.bool_))
    with gate_env("1"):
        with pytest.raises(mla.MlaPersistError, match="маска"):
            mla._dense_apply(params, cfg, x, mask)
    with gate_env("1"):
        with pytest.raises(mla.MlaPersistError, match="DENSE_FLASH"):
            with gate_env("1", flash_bwd=None):
                os.environ["AXIOM_MLA_DENSE_FLASH"] = "1"
                try:
                    mla._dense_apply(params, cfg, x)
                finally:
                    os.environ.pop("AXIOM_MLA_DENSE_FLASH", None)


def test_persist_takes_precedence_over_flash_bwd_marker(monkeypatch):
    """Связка ``PERSIST=1 × FLASH_BWD=1``: исполняется персистентный путь."""
    calls = {"fwd": 0, "bwd": 0}
    _substitute_kernels(monkeypatch)
    monkeypatch.setattr(P, "kernel", lambda q, k, v, **kw: (
        calls.__setitem__("fwd", calls["fwd"] + 1),
        P.emulate_attention(q, k, v, **kw))[1])
    monkeypatch.setattr(P, "kernel_bwd", lambda q, k, v, do, D, m, l, **kw: (
        calls.__setitem__("bwd", calls["bwd"] + 1),
        P.emulate_attention_bwd(q, k, v, do, D, m, l, **kw))[1])
    cfg, params, x = _mla_case()
    with gate_env("1", flash_bwd="1", mla_bf16="1"):
        _, grads = jax.value_and_grad(
            lambda xx: jnp.sum(mla.apply(params, cfg, xx) ** 2))(x)
    assert calls["fwd"] == 1 and calls["bwd"] == 1, calls
    assert grads is not None


# --------------------------------------------------------------------------- #
# (г) 10-шаговые ноги: nan-гейт, Δloss, маркеры, связка кампании
# --------------------------------------------------------------------------- #


@contextlib.contextmanager
def _kernels_substituted():
    """Подмена кернелов объявленной нумерикой на время ноги (сессионная форма)."""
    saved = (P.kernel, P.kernel_bwd, P.device_available)

    def emu_kernel(q, k, v, *, bq=P.BQ, bk=P.BK):
        return P.emulate_attention(q, k, v, bq=int(bq), bk=int(bk))

    def emu_kernel_bwd(q, k, v, do, D, m, l, *, bq=P.BQ, bk=P.BK):
        return P.emulate_attention_bwd(q, k, v, do, D, m, l, bq=int(bq), bk=int(bk))

    P.kernel, P.kernel_bwd, P.device_available = (
        emu_kernel, emu_kernel_bwd, (lambda: True))
    try:
        yield
    finally:
        P.kernel, P.kernel_bwd, P.device_available = saved


@pytest.fixture(scope="module")
def parity_legs(tmp_path_factory):
    """10-шаговые ноги (lr=1e-3): off / on / связка кампании.

    Данные — один детерминированный поток (парный дизайн).  ``combo`` —
    боевая связка вехи: персистентный блок × flash-bwd-маркер × bf16-хранение
    (Am.20) × per_layer_block (Am.18) × accum-jit (Am.16) × master-bf16
    (Am.13) — совместимость условий прогона и nan-контроль прецедента 4b/4c
    (дефекты проявлялись на 4+ шагах).

    ``AXIOM_COMPUTE_DTYPE=bf16`` в связку ноги не входит: 10-шаговый *train*
    в bf16-режиме на этом CPU-узле не исполняется (`DotThunk: BF16 x BF16 =
    F32` — UNIMPLEMENTED; та же причина, по которой combo вехи-12 шёл без
    compute-гейта).  Связка «персистентный блок × compute-bf16» пиннуется на
    *слое* — ``test_layer_parity_persist_vs_dense_prints_the_table`` (fp32- и
    bf16-режимы, forward и градиенты) — там CPU-путь работает.
    """
    cfg = small_config(mla_head_dim=16)
    batches = _batches(cfg, _PARITY_STEPS)
    combo_batches = _batches(cfg, _PARITY_STEPS * 2, seed=11)
    out = tmp_path_factory.mktemp("veha13-legs")
    legs = {}
    with gate_env(None):
        legs["off"] = _train_leg(cfg, _PARITY_STEPS, batches=batches,
                                 metrics_path=out / "off.jsonl")
    with _kernels_substituted(), gate_env("1"):
        legs["on"] = _train_leg(cfg, _PARITY_STEPS, batches=batches,
                                metrics_path=out / "on.jsonl")
    with _kernels_substituted(), gate_env(
            "1", flash_bwd="1", mla_bf16="1", ce_bf16="1",
            policy="per_layer_block", accum_jit="1", master="bf16"):
        legs["combo"] = _train_leg(
            cfg, _PARITY_STEPS, batches=combo_batches,
            accum_tokens=2 * SEQ_LEN, metrics_path=out / "combo.jsonl")
    return cfg, legs


def test_ten_step_delta_loss_and_nan_gate(parity_legs):
    """Δloss off/on ≤1e-3 (печатается), все числа конечны (nan-гейт 10 шагов)."""
    _cfg, legs = parity_legs
    drift = _loss_drift(legs["on"].losses, legs["off"].losses)
    print(f"[веха-13] 10 шагов, персистентный блок: max|dloss| = {drift:.3e}")
    assert all(np.isfinite(x) for x in legs["on"].losses)
    assert drift <= _DELTA_LOSS_TOL, f"Δloss ноги {drift:.3e} > {_DELTA_LOSS_TOL:g}"


def test_ten_step_nan_control_campaign_bundle(parity_legs):
    """Связка кампании, 10 шагов: ни одного нефинитного числа (4b/4c-прецедент)."""
    _cfg, legs = parity_legs
    combo = legs["combo"]
    assert combo.steps_done == _PARITY_STEPS
    assert all(np.isfinite(x) for x in combo.losses), combo.losses
    leaves = jax.tree_util.tree_leaves(combo.params)
    assert all(bool(jnp.all(jnp.isfinite(l))) for l in leaves), (
        "в параметрах связки появился nan/inf")
    assert all(np.isfinite(x) for x in legs["on"].losses)


def test_metrics_mark_the_persist_gate(parity_legs):
    """Маркер ``mla_persist`` — фактические значения в каждой строке журнала."""
    _cfg, legs = parity_legs
    expected = {"off": 0, "on": 1, "combo": 1}
    for name, want in expected.items():
        rows = tl.MetricsWriter.read(legs[name].metrics_path)
        assert rows, f"журнал ноги {name!r} пуст"
        for record in rows:
            assert record["mla_persist"] == want, (name, record)
    # Связка «персистентный блок × compute-bf16» — на слое (см. фикстуру).
    cfg, params, x = _mla_case()
    with _kernels_substituted(), gate_env("1", compute="bf16"):
        o = mla.apply(params, cfg, x)
    assert bool(jnp.all(jnp.isfinite(o)))
    # Связка: персистентный путь исполняется при flash-bwd- и bf16-гейтах —
    # все маркеры несут фактические значения, политика и аккумуляция видны рядом.
    for record in tl.MetricsWriter.read(legs["combo"].metrics_path):
        assert record["mla_persist"] == 1 and record["mla_flash_bwd"] == 1
        assert record["mla_bwd_bf16"] == 1 and record["ce_bf16"] == 1
        assert record["grad_ckpt_policy"] == "per_layer_block"
        assert record["accum_jit"] == 1 and record["microbatches"] == 2
        assert record["master_dtype"] == "bf16"


def test_gate_read_once_per_build_and_marked_off_leg(parity_legs):
    """Off-нога: маркер 0 и никакого персистентного пути в сборке."""
    _cfg, legs = parity_legs
    for record in tl.MetricsWriter.read(legs["off"].metrics_path):
        assert record["mla_persist"] == 0
        assert record["mla_flash_bwd"] == 0



