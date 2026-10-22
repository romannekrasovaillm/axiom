"""``lower_check``-адаптер fused backward-кернела экспертного блока MoE.

Скилл ``pallas-gb10-kernel`` опускает ``mod.kernel`` для каждого случая
``CASES``, который принимает ``valid_config``.  Боевой модуль
(:mod:`net.kernels.moe_experts_bwd`) держит типизированный отказ
``device_available()`` в :func:`net.kernels.moe_experts_bwd.kernel` (на
CPU-узле fused-режим обязан падать, а не молча откатываться), поэтому адаптер
зовёт построитель ``_build_kernel`` напрямую — lowering устройства не требует:

    python3 <skill>/scripts/lower_check.py net/tests/moe_experts_bwd_lower.py --configs all

Адаптер не является тест-модулем (нет префикса ``test_``: pytest его не
собирает); lowering живого случая пинуется и из CPU-набора
(``net/tests/test_54_moe_bwd_fused.py``).
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from net.kernels import moe_experts_bwd as _mod  # noqa: E402

#: Тот же грид случаев, что у боевого модуля: кампанийная геометрия
#: (latent=768/inter=384/n_exp=12), дешёвый случай для CPU-прогонов и
#: объявленный oversize, который сито честно скипает.
CASES = _mod.CASES

#: Статические ручки кернела (``bm``/``bt``/``bk``) — как в модуле.
TUNE_SPACE = dict(_mod.TUNE_SPACE)


def make_inputs(key, **case):
    """Один backward-вызов: сгруппированные строки, веса и ``group_sizes``."""
    return _mod.make_inputs(key, **case)


def valid_config(case, params):
    """Сито модуля: формы mma, bf16-операнды и SMEM-пол GB10."""
    return _mod.valid_config(case, params)


def _build(rows, latent, inter, n_exp, bm, bt, bk, dtype, beta_gate, beta_up):
    """Построитель без device-check (контракт lowering'а)."""
    return _mod._build_kernel(rows, latent, inter, n_exp, int(bm), int(bt),
                              int(bk), dtype, float(beta_gate), float(beta_up))


def kernel(z_grouped, h_g, h_u, a, de_grouped, wg, wu, wd, group_sizes, *,
           bm=_mod.BM_DEFAULT, bt=_mod.BT_DEFAULT, bk=_mod.BK_DEFAULT,
           beta_gate=_mod.BETA_GATE_DEFAULT, beta_up=_mod.BETA_UP_DEFAULT):
    """Fused backward-кернел без проверки устройства (только lowering)."""
    rows, latent = z_grouped.shape
    inter = h_g.shape[1]
    n_exp = wg.shape[0]
    case = dict(rows=int(rows), latent=int(latent), inter=int(inter),
                n_exp=int(n_exp), dtype=jnp.dtype(z_grouped.dtype).name)
    params = dict(bm=int(bm), bt=int(bt), bk=int(bk))
    if not valid_config(case, params):
        raise ValueError(f"конфигурация вне живых случаев сита: {case} {params}")
    run = jax.jit(_build(rows, latent, inter, n_exp, int(bm), int(bt), int(bk),
                         jnp.dtype(z_grouped.dtype).name,
                         float(beta_gate), float(beta_up)))
    dwg, dwu, dwd, dz, _dhg, _dhu = run(
        z_grouped, h_g, h_u, a, de_grouped, wg, wu, wd, group_sizes)
    return dwg, dwu, dwd, dz[:rows]


def reference(z_grouped, h_g, h_u, a, de_grouped, wg, wu, wd, group_sizes,
              **kwargs):
    """jnp-адъюнкт экспертного блока — численный пин кернела."""
    return _mod.reference_bwd(z_grouped, h_g, h_u, a, de_grouped, wg, wu, wd,
                              group_sizes, **kwargs)
