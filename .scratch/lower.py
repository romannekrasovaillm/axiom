"""Локальный harness: трассировка+lowering ядра под sm_121 без GPU (jax 0.11.2).

Запуск: /tmp/venv-jax0112/bin/python .scratch/lower.py [file.py]
Скрипт патчит arch (12,1) и SMEM-лимит (как lower_check.py скилла) и льёт
каждое ядро из файла через jax.jit(...).trace(...).lower(platforms=("cuda",)).
"""
from __future__ import annotations

import dataclasses
import importlib.util
import sys
import traceback

import jax
import jax.numpy as jnp


def patch_for_gb10(arch=(12, 1), smem=101376):
    has_gpu = any(d.platform == "gpu" for d in jax.devices())
    notes = []
    try:
        from jax.experimental.mosaic import gpu as mgpu
        from jax.experimental.mosaic.gpu import utils as mgu
        if not has_gpu:
            mgu._infer_arch = lambda: arch
            notes.append(f"mosaic arch patched to {arch}")
        ctx = mgpu.core.artificial_shared_memory_limit(smem)
        ctx.__enter__()
        notes.append(f"mosaic SMEM limit {smem} B")
    except Exception as e:  # noqa: BLE001
        notes.append(f"mosaic patch skipped: {e}")
    return has_gpu, notes


def lower(fn, *args):
    try:
        f = jax.jit(fn)
        txt = f.trace(*args).lower(lowering_platforms=("cuda",)).as_text()
        return "OK", txt
    except Exception as e:  # noqa: BLE001
        return "FAIL", f"{type(e).__name__}: {e}"


def report(name, fn, *args):
    status, txt = lower(fn, *args)
    if status == "OK":
        print(f"[{name}] OK  ({len(txt)} chars)")
    else:
        print(f"[{name}] FAIL: {txt[:800]}")
        if "-v" in sys.argv:
            traceback.print_exc()
    return status == "OK"


def load_module(path):
    spec = importlib.util.spec_from_file_location("proto", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["proto"] = mod
    spec.loader.exec_module(mod)
    return mod


if __name__ == "__main__":
    patch_for_gb10()
    path = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("-") else None
    if path:
        mod = load_module(path)
        names = [n for n in dir(mod) if n.startswith("case") and n != "case_inputs"]
        ok = fail = 0
        for n in names:
            fn = getattr(mod, n)
            inp = mod.case_inputs(n) if hasattr(mod, "case_inputs") else ()
            if inp is None:
                continue
            res = report(n, fn, *inp)
            ok += res
            fail += not res
        print(f"[summary] OK {ok}  FAIL {fail}")
    else:
        print("usage: lower.py <module.py>")
