#!/usr/bin/env python3
"""Пофазовый профиль KDA-слоя — прибор диагностики ADR-047 п. 3.

ADR-047 запрещает оптимизировать *предполагаемое*: до переписывания реализации
время и память слоя раскладываются по фазам, и уже по этому отчёту выбирается,
что именно дорого. Прибор делает три вещи:

* **фазовые ноги** — цепочка отдельно jit-компилируемых стадий слоя
  (``projections`` → ``shortconv`` → ``intra`` → ``inter`` → ``output_gate`` →
  ``out_proj``), которую прогоняют по чанкам на хосте (128 чанков при T=8192,
  C=64), замеряя каждую стадию хост-таймером ``perf_counter`` +
  ``jax.block_until_ready``. Это та же дисциплина, что у уже принятого
  ``net/tests/test_phase_profile.py``. У каждой фазы дополнительно снимается
  пиковая память XLA (``memory_analysis``) её скомпилированного исполняемого
  объекта;
* **целые ноги** — fused forward и fused forward+backward всего слоя (то, что
  реально исполняется в шаге), с временем p50 и памятью XLA. Backward
  отчитывается как разность fused-ног: раздельных пофазовых backward-замеров
  прибор не заявляет;
* **сверка цепочки** — фазовые стадии обязаны воспроизвести выход fused-ноги
  (``phase_chain_max_abs_diff``). Если слой изменится, а прибор — нет, это
  видно сразу, а не как правдоподобные чужие числа.

ВНИМАНИЕ (границы прибора, честно): фазовые стадии компилируются раздельно,
поэтому XLA не фьюзит их между собой и хост платит за запуск на каждую стадию
на каждый чанк. Сумма фаз — это *атрибуция*, а не время шага; сверять её надо
с fused-ногой. Числа CPU-ног индикативны (рематериализация — механизм GPU).

Формы: ``chunked`` и ``wyut`` (обе существуют в ``net/kda.py``, ADR-047 п. 3)
и ``chunked_cc`` — новая форма той же дельты, для сравнения «до/после».

Запуск на GB10 (ADR-041: префлайт памяти вызывается до ``import jax``)::

    python tools/kda_phase_profile.py                     # l3 + small, все формы
    python tools/kda_phase_profile.py --geometry l3 --impls chunked,chunked_cc
    python tools/kda_phase_profile.py --smoke --out /tmp/p.json   # CPU-смоук

Без CUDA-устройства прибор **fail-closed**: числа фаз не имитируются, в
``evidence/kda-rewrite/phase-profile.json`` уходит ``EMPTY-PENDING`` с причиной
и ненулевой код возврата. Исключение — явный ``--smoke``: он честно меряет
малую геометрию на CPU и помечает отчёт ``SMOKE-CPU``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import median

# Запуск как ``python tools/kda_phase_profile.py``: ``net`` — в корне репозитория.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# ADR-041: дисциплина памяти JAX — префлайт ДО import jax (лимит XLA + гейт стенда).
import jax_preflight  # noqa: E402

jax_preflight.ensure_mem_fraction()

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import jax.random as jr  # noqa: E402

from net import kda  # noqa: E402
from net.config import ModelConfig, validate_config  # noqa: E402
from net.norm import headwise_rms_norm  # noqa: E402

#: Канонический путь отчёта (дельта ADR-047).
DEFAULT_OUT = _REPO_ROOT / "evidence" / "kda-rewrite" / "phase-profile.json"

#: Порядок фаз в отчёте (совпадает с ADR-047 п. 3).
PHASE_ORDER = (
    "projections",
    "shortconv",
    "intra",
    "inter",
    "output_gate",
    "out_proj",
)

#: Существующие формы (ADR-047 п. 3) и новая форма этой дельты.
KNOWN_IMPLS = ("chunked", "wyut", "chunked_cc")

#: Семейство реализации: моноид переходов против C x C-матрицы.
_IMPL_FAMILY = {"chunked": "monoid", "wyut": "cc", "chunked_cc": "cc"}

#: Геометрии: ранг L3a (ADR-047: T=8192, H=12, dk=dv=128, C=64) и малый профиль.
GEOMETRIES = {
    "l3": dict(seq_len=8192, heads=12, dk=128, chunk=64, rounds=5, warmup=2),
    "small": dict(seq_len=512, heads=4, dk=64, chunk=64, rounds=3, warmup=1),
}

#: Геометрия CPU-смоука (провод прибора, не число).
SMOKE_GEOMETRY = dict(seq_len=64, heads=2, dk=16, chunk=32, rounds=2, warmup=1)

STATUS_OK = "OK"
STATUS_SMOKE = "SMOKE-CPU"
STATUS_EMPTY = "EMPTY-PENDING"

_NO_GPU_REASON = (
    "нет CUDA-устройства: пофазовые время и память — свойство GPU-ядра, на CPU "
    "они не меряются (ADR-047 п. 3: числа не имитируются). Прогон выполняет "
    "архитектор на GB10; для проверки провода прибора на CPU — явный --smoke."
)


# ---------------------------------------------------------------------------
# конфиг и состояние стенда
# ---------------------------------------------------------------------------


def build_config(geom: dict, impl: str) -> ModelConfig:
    """Конфиг, несущий геометрию профиля (только KDA-значимые поля).

    Инварианты ``validate_config`` соблюдены; окно SWA выключено
    (``swa_window=0``), поэтому измеряется сама дельта-форма — та же граница,
    что у ``tools/bench_kda_wyut.py`` и у замера 2.17% MFU.
    """
    cfg = ModelConfig(
        vocab_size=1024,
        hidden=geom["heads"] * geom["dk"],
        num_layers=4,
        num_kda_layers=3,
        num_mla_layers=1,
        num_heads=geom["heads"],
        head_dim=geom["dk"],
        kda_dk=geom["dk"],
        kda_dv=geom["dk"],
        kda_decay_rank=geom["dk"],
        mla_head_dim=geom["dk"],
        mla_latent_dim=max(16, geom["dk"]),
        swa_window=0,
        attn_dense_reference=True,
        kda_wyut_chunk=geom["chunk"],
        kda_impl=impl,
    )
    validate_config(cfg)
    return cfg


def backend_is_gpu() -> bool:
    """True, если активное устройство JAX — GPU (иначе фазы не меряются)."""
    try:
        return any(device.platform == "gpu" for device in jax.devices())
    except Exception:  # pragma: no cover - защита от падения рантайма бэкенда
        return False


def resolve_status(*, smoke: bool, gpu: bool) -> str:
    """Статус отчёта по паре (--smoke, наличие GPU) — чистая функция."""
    if gpu:
        return STATUS_SMOKE if smoke else STATUS_OK
    return STATUS_SMOKE if smoke else STATUS_EMPTY


# ---------------------------------------------------------------------------
# фазовые стадии (зеркало net/kda.py; сверяются с fused-ногой по выходу)
# ---------------------------------------------------------------------------


def _phase_units(impl: str, cfg: ModelConfig, params: kda.KDAParams):
    """Упорядоченные jit-стадии слоя для одной формы.

    Стадии вызываются как ``fn(state, carry) -> (state, carry)``, где ``state``
    — плоский кортеж промежуточных тензоров, ``carry`` — ``kda.KDAState``.
    Используются приватные помощники ``net/kda.py`` намеренно: прибор обязан
    мерить *ту же* арифметику, и равенство его выхода fused-ноге это
    подтверждает (``phase_chain_max_abs_diff``).
    """
    family = _IMPL_FAMILY[impl]

    def projections(state, carry):
        (x,) = state
        proj = kda._project(params, cfg, x)
        return (
            proj["qp"],
            proj["kp"],
            proj["vp"],
            proj["beta"],
            proj["alpha"],
            proj["gate"],
        ), carry

    def shortconv(state, carry):
        qp, kp, vp, beta, alpha, gate = state
        qc, q_buf = kda._short_conv_chunk(qp, params.conv_q, carry.q_buf)
        kc, k_buf = kda._short_conv_chunk(kp, params.conv_k, carry.k_buf)
        vc, v_buf = kda._short_conv_chunk(vp, params.conv_v, carry.v_buf)
        q, k, v = kda._postprocess(qc, kc, vc, cfg)
        carry = carry._replace(q_buf=q_buf, k_buf=k_buf, v_buf=v_buf)
        return (q, k, v, beta, alpha, gate), carry

    if family == "monoid":

        def intra(state, carry):
            q, k, v, beta, alpha, gate = state
            m_mat, n_mat = kda._delta_transitions(k, v, beta, alpha)
            return (m_mat, n_mat, q, gate), carry

        def inter(state, carry):
            m_mat, n_mat, q, gate = state
            p_mat, q_mat = jax.lax.associative_scan(kda._combine, (m_mat, n_mat))
            pq = jnp.einsum("chab,cha->chb", p_mat, q)
            o = jnp.einsum("hdv,chd->chv", carry.S, pq) + jnp.einsum(
                "chuv,chu->chv", q_mat, q
            )
            s_new = p_mat[-1] @ carry.S + q_mat[-1]
            return (o, gate), carry._replace(S=s_new)

    else:

        def intra(state, carry):
            q, k, v, beta, alpha, gate = state
            C = q.shape[0]
            log_g = kda._log_cumulative_decay(alpha)
            q_t = q.transpose(1, 0, 2)
            k_t = k.transpose(1, 0, 2)
            v_t = v.transpose(1, 0, 2)
            beta_t = beta.transpose(1, 0)
            if impl == "wyut":
                e = kda._decay_ratio_exp(log_g)  # (H, C, C, dk) — тяжёлый тензор
                lower = jnp.tril(jnp.ones((C, C), dtype=bool))
                strict = jnp.tril(jnp.ones((C, C), dtype=bool), k=-1)
                aqk = jnp.where(
                    lower, jnp.einsum("hcid,hcd,hid->hci", e, q_t, k_t), 0.0
                )
                akk = jnp.where(
                    strict, jnp.einsum("hcid,hcd,hid->hci", e, k_t, k_t), 0.0
                )
            else:
                log_g_t = log_g.transpose(1, 0, 2)
                tile = kda._cc_tile(cfg)
                aqk = kda._cc_scores(q_t, k_t, log_g_t, tile, strict=False)
                akk = kda._cc_scores(k_t, k_t, log_g_t, tile, strict=True)
            l_mat = akk * beta_t[:, :, None]
            t_mat = jnp.linalg.inv(jnp.eye(C, dtype=l_mat.dtype) + l_mat)
            gamma = jnp.exp(log_g).transpose(1, 0, 2)
            xw = (gamma * k_t) * beta_t[:, :, None]
            vw = v_t * beta_t[:, :, None]
            w = jnp.einsum("hcs,hsd->hcd", t_mat, xw)
            u = jnp.einsum("hcs,hsd->hcd", t_mat, vw)
            gamma_q = gamma * q_t
            gamma_c = gamma[:, -1, :]
            lam = jnp.exp(jnp.minimum(log_g[-1][None] - log_g, 0.0)).transpose(1, 0, 2)
            return (aqk, w, u, gamma_q, gamma_c, lam * k_t, gate), carry

        def inter(state, carry):
            aqk, w, u, gamma_q, gamma_c, y, gate = state
            s_in = carry.S
            v_tilde = u - jnp.einsum("hcd,hde->hce", w, s_in)
            o = (
                jnp.einsum("hcd,hde->hce", gamma_q, s_in)
                + jnp.einsum("hci,hie->hce", aqk, v_tilde)
            ).transpose(1, 0, 2)
            s_new = gamma_c[:, :, None] * s_in + jnp.einsum("hcd,hce->hde", y, v_tilde)
            return (o, gate), carry._replace(S=s_new)

    def output_gate(state, carry):
        o, gate = state
        o = headwise_rms_norm(o)
        o = o.reshape(*o.shape[:-2], -1)
        return (gate * o,), carry

    def out_proj(state, carry):
        (gated,) = state
        return (gated @ params.W_o,), carry

    named = (
        ("projections", projections),
        ("shortconv", shortconv),
        ("intra", intra),
        ("inter", inter),
        ("output_gate", output_gate),
        ("out_proj", out_proj),
    )
    return [(name, jax.jit(fn)) for name, fn in named]


# ---------------------------------------------------------------------------
# замеры
# ---------------------------------------------------------------------------


def _memory_info(executable) -> dict:
    """Память скомпилированного исполняемого объекта (детерминированно, из XLA)."""
    analysis = executable.memory_analysis()
    if analysis is None:  # бэкенд не отдаёт разбор — честный null, не ноль
        return {"xla_scratch_bytes": None, "xla_peak_bytes": None}
    temp = int(analysis.temp_size_in_bytes)
    return {
        "xla_scratch_bytes": temp,
        "xla_peak_bytes": temp
        + int(analysis.argument_size_in_bytes)
        + int(analysis.output_size_in_bytes),
    }


def _time_jitted(jitted, args, *, rounds: int, warmup: int) -> dict:
    """Хост-замер jitted-функции: медиана, минимум, максимум (секунды)."""
    for _ in range(warmup):
        jax.block_until_ready(jitted(*args))
    times: list[float] = []
    for _ in range(rounds):
        tick = time.perf_counter()
        jax.block_until_ready(jitted(*args))
        times.append(time.perf_counter() - tick)
    return {
        "seconds_p50": median(times),
        "seconds_min": min(times),
        "seconds_max": max(times),
        "seconds": median(times),
    }


def _fused_legs(impl: str, cfg: ModelConfig, params, x, geom: dict) -> dict:
    """Fused-ноги слоя: forward и forward+backward, с памятью XLA."""
    chunk = geom["chunk"]

    def forward(p):
        return kda.apply_kda(p, cfg, x, chunk_size=chunk)

    fwd = jax.jit(forward)
    fwd_bwd = jax.jit(jax.value_and_grad(lambda p: jnp.sum(forward(p))))

    fwd_info = _time_jitted(fwd, (params,), rounds=geom["rounds"], warmup=geom["warmup"])
    fwd_exec = fwd.lower(params).compile()
    fwd_info.update(_memory_info(fwd_exec))

    bwd_info = _time_jitted(
        fwd_bwd, (params,), rounds=geom["rounds"], warmup=geom["warmup"]
    )
    bwd_exec = fwd_bwd.lower(params).compile()
    bwd_info.update(_memory_info(bwd_exec))

    return {
        "fused_fwd": fwd_info,
        "fused_fwd_bwd": bwd_info,
        "backward": {
            "seconds_p50": bwd_info["seconds_p50"] - fwd_info["seconds_p50"],
            "derived": True,
            "note": "fused_fwd_bwd - fused_fwd (раздельные пофазовые backward-замеры "
            "прибор не заявляет)",
        },
    }


def _phase_legs(impl: str, cfg: ModelConfig, params, x, geom: dict) -> dict:
    """Пофазовый прогон по чанкам с хост-таймером на каждую стадию."""
    C = geom["chunk"]
    T = geom["seq_len"]
    n_chunks = (T + C - 1) // C
    pad = n_chunks * C - T
    x_p = jnp.pad(x, ((0, pad), (0, 0))) if pad else x
    x_chunks = x_p.reshape(n_chunks, C, -1)

    units = _phase_units(impl, cfg, params)
    names = [name for name, _ in units]

    # Прогрев: он же даёт (а) входные тензоры каждой стадии для замера её памяти
    # и (б) выход цепочки для сверки с fused-ногой.
    carry = kda.init_state(cfg)
    outputs = []
    first_inputs: dict[str, tuple] = {}
    for index in range(n_chunks):
        state = (x_chunks[index],)
        for name, fn in units:
            if index == 0:
                first_inputs[name] = (state, carry)
            state, carry = fn(state, carry)
        outputs.append(state[0])
    chain_out = jnp.concatenate(outputs, axis=0)[:T]

    # Пиковая память стадии — по её собственному входу (первый чанк).
    memory: dict[str, dict] = {
        name: _memory_info(fn.lower(*first_inputs[name]).compile())
        for name, fn in units
    }

    totals = {name: 0.0 for name in names}
    calls = {name: 0 for name in names}
    for _ in range(geom["rounds"]):
        carry = kda.init_state(cfg)
        for index in range(n_chunks):
            state = (x_chunks[index],)
            for name, fn in units:
                tick = time.perf_counter()
                state, carry = fn(state, carry)
                jax.block_until_ready(state)
                totals[name] += time.perf_counter() - tick
                calls[name] += 1

    total_seconds = sum(totals.values())
    phases = {}
    for name in names:
        phases[name] = {
            "seconds": totals[name] / geom["rounds"],
            "share": (totals[name] / total_seconds) if total_seconds else None,
            "calls": calls[name],
            **memory[name],
        }
    return {
        "phases": phases,
        "phase_sum_seconds": total_seconds / geom["rounds"],
        "chunks": n_chunks,
        "chain_output": chain_out,
    }


def profile_impl(impl: str, geom: dict, *, seed: int = 0) -> dict:
    """Один прогон прибора для одной формы и одной геометрии."""
    cfg = build_config(geom, impl)
    params = kda.init_kda(jr.PRNGKey(seed), cfg)
    x = jr.normal(jr.PRNGKey(seed + 1), (geom["seq_len"], cfg.hidden))

    fused = _fused_legs(impl, cfg, params, x, geom)
    phases = _phase_legs(impl, cfg, params, x, geom)

    # Сверка цепочки с fused-ногой: прибор, разошедшийся с реализацией, обязан
    # быть виден, а не выдавать правдоподобные чужие числа.
    reference = kda.apply_kda(params, cfg, x, chunk_size=geom["chunk"])
    diff = float(jnp.max(jnp.abs(reference - phases.pop("chain_output"))))
    tolerance = 1e-3 * max(1.0, float(jnp.max(jnp.abs(reference))))
    return {
        "impl": impl,
        "geometry": dict(geom),
        "fused": fused,
        "phases": phases["phases"],
        "phase_sum_seconds": phases["phase_sum_seconds"],
        "chunks": phases["chunks"],
        "phase_chain_max_abs_diff": diff,
        "phase_chain_matches_fused": bool(diff <= tolerance),
        "phase_chain_tolerance": tolerance,
    }


# ---------------------------------------------------------------------------
# отчёт
# ---------------------------------------------------------------------------


def render_table(impl: str, report: dict) -> str:
    """Таблица «фаза, время, доля, пиковая память» для stdout."""
    geom = report["geometry"]
    lines = [
        f"[{impl}] backend={report['backend']} device={report['device']} "
        f"T={geom['seq_len']} C={geom['chunk']} H={geom['heads']} dk={geom['dk']} "
        f"rounds={geom['rounds']}",
        f"  {'фаза':<14}{'время, с':>12}{'доля':>9}{'вызовов':>9}{'XLA scratch, МиБ':>18}",
    ]
    for name in PHASE_ORDER:
        phase = report["phases"].get(name)
        if phase is None:
            continue
        scratch = phase.get("xla_scratch_bytes")
        scratch_mib = f"{scratch / (1 << 20):.2f}" if scratch is not None else "n/a"
        share = phase.get("share")
        lines.append(
            f"  {name:<14}{phase['seconds']:>12.4f}"
            f"{(f'{share * 100:.1f}%' if share is not None else 'n/a'):>9}"
            f"{phase['calls']:>9}{scratch_mib:>18}"
        )
    fused = report["fused"]
    lines.append(
        f"  {'СУММА ФАЗ':<14}{report['phase_sum_seconds']:>12.4f}"
        f"{'':>9}{'':>9}{'':>18}"
    )
    for leg in ("fused_fwd", "fused_fwd_bwd"):
        item = fused[leg]
        scratch = item.get("xla_scratch_bytes")
        lines.append(
            f"  {leg:<14}{item['seconds_p50']:>12.4f}{'':>9}{'':>9}"
            f"{(f'{scratch / (1 << 20):.2f}' if scratch is not None else 'n/a'):>18}"
        )
    lines.append(
        f"  {'backward*':<14}{fused['backward']['seconds_p50']:>12.4f}"
        f"{'':>9}{'':>9}{'':>18}   (* производная fused-ног, см. JSON)"
    )
    lines.append(
        f"  сверка цепочки с fused: max|Δ|={report['phase_chain_max_abs_diff']:.3e} "
        f"({'OK' if report['phase_chain_matches_fused'] else 'РАСХОЖДЕНИЕ'})"
    )
    return "\n".join(lines)


def empty_report(*, status: str, reason: str, out_path: Path) -> dict:
    """Отчёт без чисел: прибор не имитирует фазы, которых не измерял."""
    return {
        "instrument": "tools/kda_phase_profile.py",
        "adr": "ADR-047",
        "status": status,
        "reason": reason,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "out_path": str(out_path),
        "backend": jax.default_backend(),
        "devices": [str(device) for device in jax.devices()],
        "geometries": {},
        "note": (
            "Числа фаз не имитируются: без CUDA-устройства их измеряет архитектор "
            "на GB10 (ADR-047 п. 3); CPU-прогон с --smoke даёт индикативные числа "
            "и статус SMOKE-CPU."
        ),
    }


def run(geometry: str, impls: list[str], smoke: bool) -> dict:
    """Собирает отчёт: геометрии x формы (или EMPTY-PENDING без GPU)."""
    gpu = backend_is_gpu()
    status = resolve_status(smoke=smoke, gpu=gpu)

    if smoke:
        geometries = {"smoke": SMOKE_GEOMETRY}
    elif geometry == "both":
        geometries = {name: GEOMETRIES[name] for name in ("l3", "small")}
    else:
        geometries = {geometry: GEOMETRIES[geometry]}

    if status == STATUS_EMPTY:
        return empty_report(
            status=STATUS_EMPTY,
            reason=_NO_GPU_REASON,
            out_path=DEFAULT_OUT,
        )

    report = {
        "instrument": "tools/kda_phase_profile.py",
        "adr": "ADR-047",
        "status": status,
        "reason": (
            "CPU-смоук: числа индикативны, рематериализация — механизм GPU"
            if status == STATUS_SMOKE
            else "измерено на CUDA-устройстве"
        ),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "backend": jax.default_backend(),
        "devices": [str(device) for device in jax.devices()],
        "impls": list(impls),
        "geometries": {},
        "note": (
            "Фазовые стадии компилируются раздельно — их сумма есть атрибуция, а не "
            "время шага; ground truth — fused_fwd/fused_fwd_bwd. Каждая геометрия "
            "несёт phase_chain_max_abs_diff — сверку прибора с реализацией."
        ),
    }
    for geom_name, geom in geometries.items():
        entries = {}
        for impl in impls:
            entry = profile_impl(impl, geom)
            entry["backend"] = jax.default_backend()
            entry["device"] = str(jax.devices()[0]) if jax.devices() else None
            entries[impl] = entry
            print(render_table(impl, entry))
            print()
        report["geometries"][geom_name] = {"requested": dict(geom), "impls": entries}
    return report


def write_report(report: dict, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--geometry",
        choices=("l3", "small", "both"),
        default="both",
        help="геометрия профиля (по умолчанию обе: l3-full и малый профиль)",
    )
    parser.add_argument(
        "--impls", default=",".join(KNOWN_IMPLS), help="формы через запятую"
    )
    parser.add_argument("--out", default=None, help=f"путь отчёта (дефолт {DEFAULT_OUT})")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="малая геометрия на доступном бэкенде (провод прибора, не число)",
    )
    args = parser.parse_args(argv)

    impls = [item.strip() for item in args.impls.split(",") if item.strip()]
    unknown = [impl for impl in impls if impl not in KNOWN_IMPLS]
    if unknown:
        parser.error(f"неизвестные формы {unknown}; известные: {list(KNOWN_IMPLS)}")
    if not impls:
        parser.error("--impls не может быть пустым")

    out_path = Path(args.out) if args.out else DEFAULT_OUT

    jax_preflight.gate_or_exit()  # ADR-041: состояние стенда до реального прогона
    report = run(args.geometry, impls, args.smoke)
    report["out_path"] = str(out_path)
    write_report(report, out_path)

    if report["status"] == STATUS_EMPTY:
        print(f"[kda-phase-profile] {STATUS_EMPTY}: {report['reason']}", file=sys.stderr)
        print(f"[kda-phase-profile] отчёт: {out_path}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
