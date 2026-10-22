"""``lower_check``-адаптер fused dispatch/scatter MoE (ADR-054 Am.58).

Скилл ``pallas-gb10-kernel`` опускает ``mod.kernel`` для каждого случая
``CASES``, который принимает ``valid_config``.  Боевой модуль
(:mod:`net.kernels.moe_dispatch`) держит типизированный отказ
``device_available()`` в :func:`net.kernels.moe_dispatch.kernel` (на CPU-узле
fused-режим обязан падать, а не молча откатываться), поэтому адаптер зовёт
построитель ``_build_kernel`` напрямую — lowering устройства не требует:

    python3 <skill>/scripts/lower_check.py net/tests/moe_dispatch_lower.py --configs all

Адаптер не является тест-модулем (нет префикса ``test_``: pytest его не
собирает); lowering живого случая пинуется и из CPU-набора
(``net/tests/test_55_moe_dispatch.py``).
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from net.kernels import moe_dispatch as _mod  # noqa: E402

#: Тот же грид случаев, что у боевого модуля: кампанийная геометрия
#: (latent=768/k=2/n_exp=12), дешёвый случай для CPU-прогонов и объявленный
#: невалидный (``rows % bm != 0``), который сито честно скипает.
CASES = _mod.CASES

#: Статические ручки кернела (``bm``/``bc``) — как в модуле.
TUNE_SPACE = dict(_mod.TUNE_SPACE)


def make_inputs(key, **case):
    """Один dispatch-вызов: ``z``, ``topk_idx``, ``sel_p``."""
    return _mod.make_inputs(key, **case)


def valid_config(case, params):
    """Сито модуля: формы, операнд-путь и ``rows % bm == 0``."""
    return _mod.valid_config(case, params)


def _build(tokens, k, latent, n_exp, bm, bc, dtype):
    """Построитель без device-check (контракт lowering'а)."""
    return _mod._build_kernel(int(tokens), int(k), int(latent), int(n_exp),
                              int(bm), int(bc), dtype)


def kernel(z, topk_idx, sel_p, *, n_exp=12, bm=_mod.BM_DEFAULT,
           bc=_mod.BC_DEFAULT):
    """Fused dispatch без проверки устройства (только lowering).

    Обёртка повторяет префиксные суммы ``group_sizes`` боевого вызова: сито,
    ``bincount``/``cumsum``, затем кернел с уже готовыми ``sizes``/``offsets``.
    """
    tokens, k = int(topk_idx.shape[0]), int(topk_idx.shape[1])
    latent = int(z.shape[1])
    case = dict(tokens=tokens, k=k, latent=latent, n_exp=int(n_exp),
                dtype=jnp.dtype(z.dtype).name)
    params = dict(bm=int(bm), bc=int(bc))
    if not valid_config(case, params):
        raise ValueError(f"конфигурация вне живых случаев сита: {case} {params}")
    group_sizes = jnp.bincount(topk_idx.reshape(-1).astype(jnp.int32),
                               length=int(n_exp)).astype(jnp.int32)
    offsets = _mod.group_offsets(group_sizes)
    build = _build(tokens, k, latent, n_exp, bm, bc, case["dtype"])
    run = jax.jit(lambda *a: build(*a))
    zg, sizes, assign, pair = run(
        z, topk_idx.reshape(-1), group_sizes, offsets)
    rows = tokens * k
    return zg[:rows], sizes, assign[:rows], pair[:rows]


def reference(z, topk_idx, sel_p=None, *, n_exp, **kwargs):
    """jnp-эквивалент dispatch — численный пин канонической раскладки."""
    return _mod.reference_dispatch(z, topk_idx, sel_p, n_exp=n_exp)
