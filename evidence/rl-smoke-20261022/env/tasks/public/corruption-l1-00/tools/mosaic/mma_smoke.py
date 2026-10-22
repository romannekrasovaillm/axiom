#!/usr/bin/env python3
"""Mosaic MMA smoke для JAX 0.11.2 / GB10 (sm_121) — компиляция и корректность.

Зачем отдельный скрипт
----------------------
Первый smoke падал на верификации:

    VerificationError: 'mosaic_gpu.mma' op operand #1 must be vector of A type
    supported by the 'a' and 'b' operands … got 'vector<128x128xf32>'

То есть операнд ``A`` приходил во float32: загрузка из GMEM шла без MMА-layout'ов, а
ядро вызывалось без явных ``compiler_params``. Эталонный апстрим-тест (вырезан в
``tools/mosaic/reference/``) делает иначе и ровно так, как нужно здесь:

* ``plgpu.kernel(..., compiler_params=dataclasses.replace(plgpu.CompilerParams(),
  lowering_semantics=plgpu.LoweringSemantics.Lane))`` — **Lane**-семантика явно, а не
  дефолтная (Warpgroup);
* ``a = plgpu.load(a_ref, layout=plgpu.Layout.MMA_LHS(dtype), optimized=False)``;
* ``b = plgpu.load(b_ref.T, layout=plgpu.Layout.MMA_RHS(dtype), optimized=False)`` —
  правая часть грузится **транспонированной** (форма ``(n, k)`` в памяти);
* аккумулятор — ``plgpu.layout_cast(jnp.zeros((m, n), acc_dtype), plgpu.Layout.MMA_ACC(dtype))``;
* входные массивы — numpy, приводятся через ``jnp.asarray(x, dtype=jnp.dtype("bfloat16"))``
  (не ``ndarray.astype(jnp.bfloat16)``).

Что скрипт доказывает и чего не доказывает
-----------------------------------------
Доказательство даёт только исполнение на GPU: ``blocked`` (нет GPU), импорт модуля и
``interpret=True`` — не доказательство GPU-компиляции (C-007: без подмены проверки).
Скрипт различает **compile**-стадию (сборка/нижний IR) и **run**-стадию (исполнение):
ошибка в каждой пишется своим ``error_type``.

Запуск на стенде (только там, где есть GPU; на исполнителе не запускается — AD-7):
интерпретатор стенда из окружения JAX 0.11.2, затем

    python tools/mosaic/mma_smoke.py --output /tmp/mma.json
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np

#: Комбинации (M, K, N), которые гоняет smoke: базовая из апстрим-теста + две добавки.
CASES: tuple[tuple[int, int, int], ...] = ((128, 128, 8), (128, 128, 16), (128, 64, 8), (256, 128, 8))

#: Дата-типы, которые проверяет апстрим-тест (в smoke по умолчанию — bf16).
DEFAULT_DTYPES: tuple[str, ...] = ("bfloat16",)


def _status_doc() -> dict:
    return {
        "status": "blocked",
        "shape": None,
        "dtype": None,
        "max_abs": None,
        "rel": None,
        "seconds": None,
        "error_type": None,
        "error": None,
        "stage": None,
        "cases": [],
    }


def _reasons_no_gpu(jax) -> str:
    try:
        devices = [d.platform for d in jax.devices()]
    except Exception as exc:  # noqa: BLE001 — сломанный плагин не читается как «GPU есть»
        return f"jax.devices() недоступен: {exc!r}"
    return f"GPU не найден: platforms={devices}"


def _build_kernel(jax, jnp, plgpu, m: int, k: int, n: int, dtype, acc_dtype):
    """Ядро ровно в форме эталона: MMA_ACC/MMA_LHS/MMA_RHS + Lane-семантика."""

    def body(a_ref, b_ref, o_ref):
        acc = plgpu.layout_cast(jnp.zeros((m, n), acc_dtype), plgpu.Layout.MMA_ACC(dtype))
        a = plgpu.load(a_ref, layout=plgpu.Layout.MMA_LHS(dtype), optimized=False)
        b = plgpu.load(b_ref.T, layout=plgpu.Layout.MMA_RHS(dtype), optimized=False)
        o_ref[...] = plgpu.mma(acc, a, b)

    compiler_params = dataclasses.replace(
        plgpu.CompilerParams(), lowering_semantics=plgpu.LoweringSemantics.Lane
    )
    return plgpu.kernel(
        body,
        out_type=jax.ShapeDtypeStruct((m, n), acc_dtype),
        compiler_params=compiler_params,
    )


def _case_inputs(m: int, k: int, n: int, dtype_str: str, jnp):
    """Входы как в эталоне: numpy (m,k) и (n,k), приведение — через jnp.asarray."""
    rng = np.random.default_rng(0)
    dtype = jnp.dtype(dtype_str)
    a_np = rng.uniform(-1.0, 1.0, (m, k))
    b_np = rng.uniform(-1.0, 1.0, (n, k))
    a = jnp.asarray(a_np, dtype=dtype)
    b = jnp.asarray(b_np, dtype=dtype)
    ref = jnp.asarray(a_np, dtype=jnp.float32) @ jnp.asarray(b_np, dtype=jnp.float32).T
    return a, b, ref


def _run_case(jax, jnp, plgpu, m: int, k: int, n: int, dtype_str: str) -> dict:
    dtype = jnp.dtype(dtype_str)
    acc_dtype = jnp.float32
    a, b, ref = _case_inputs(m, k, n, dtype_str, jnp)

    out = {
        "shape": [m, k, n],
        "dtype": dtype_str,
        "status": "blocked",
        "stage": None,
        "error_type": None,
        "error": None,
        "max_abs": None,
        "rel": None,
        "seconds": None,
    }

    kernel_fn = _build_kernel(jax, jnp, plgpu, m, k, n, dtype, acc_dtype)

    # --- стадия 1: компиляция (сборка + нижний IR) -------------------------------
    t0 = time.perf_counter()
    try:
        # У объекта plgpu.kernel НЕТ метода .lower — оборачиваем в jax.jit: это рабочий
        # путь (проверено архитектором на GB10). Иначе — ложный blocked/AttributeError.
        lowered = jax.jit(kernel_fn).lower(a, b)
        compiled = lowered.compile()
    except Exception as exc:  # noqa: BLE001 — любая ошибка компиляции идёт классом compile
        out.update(stage="compile", error_type=type(exc).__name__,
                   error=traceback.format_exc()[-2000:])
        return out

    # --- стадия 2: исполнение ---------------------------------------------------
    try:
        got = jax.block_until_ready(compiled(a, b))
    except Exception as exc:  # noqa: BLE001 — исполнение отделено от компиляции
        out.update(stage="run", error_type=type(exc).__name__,
                   error=traceback.format_exc()[-2000:], seconds=time.perf_counter() - t0)
        return out

    seconds = time.perf_counter() - t0
    diff = jnp.abs(got.astype(jnp.float32) - ref)
    max_abs = float(jnp.max(diff))
    denom = float(jnp.max(jnp.abs(ref))) + 1e-30
    rel = max_abs / denom
    # Допуск — как у апстрим-теста: bf16/fp16 считаются в fp32-аккумуляторе.
    ok = max_abs <= 1e-1 if dtype_str in ("bfloat16", "float16") else max_abs <= 1e-4
    out.update(status="correctness-pass" if ok else "correctness-fail",
               max_abs=max_abs, rel=rel, seconds=seconds, stage="run")
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Mosaic MMA smoke (JAX 0.11.2 / GB10)")
    ap.add_argument("--output", required=True, help="куда записать JSON-отчёт")
    ap.add_argument("--dtype", action="append", default=None,
                    help="dtype операндов (по умолчанию bfloat16; можно несколько)")
    ap.add_argument("--cases", action="store_true",
                    help="прогнать все комбинации CASES, а не только базовую M=K=128, N=8")
    args = ap.parse_args(argv)

    doc = _status_doc()
    try:
        import jax
        import jax.numpy as jnp
    except Exception as exc:  # noqa: BLE001
        doc.update(error_type=type(exc).__name__, error=f"jax недоступен: {exc}", stage="import")
        doc["cases"] = [{"shape": [m, k, n], "dtype": dt, "status": "blocked",
                        "stage": "import", "error_type": type(exc).__name__,
                        "error": doc["error"], "max_abs": None, "rel": None, "seconds": None}
                       for dt in (tuple(args.dtype) if args.dtype else DEFAULT_DTYPES)
                       for (m, k, n) in (CASES if args.cases else (CASES[0],))]
        _write(Path(args.output), doc)
        print(json.dumps(doc, ensure_ascii=False, indent=2))
        return 0

    try:
        from jax.experimental.pallas import mosaic_gpu as plgpu
    except Exception as exc:  # noqa: BLE001
        doc.update(error_type=type(exc).__name__, error=f"mosaic_gpu недоступен: {exc}", stage="import")
        doc["cases"] = [{"shape": [m, k, n], "dtype": dt, "status": "blocked",
                        "stage": "import", "error_type": type(exc).__name__,
                        "error": doc["error"], "max_abs": None, "rel": None, "seconds": None}
                       for dt in (tuple(args.dtype) if args.dtype else DEFAULT_DTYPES)
                       for (m, k, n) in (CASES if args.cases else (CASES[0],))]
        _write(Path(args.output), doc)
        print(json.dumps(doc, ensure_ascii=False, indent=2))
        return 0

    if not any(d.platform == "gpu" for d in jax.devices()):
        reason = _reasons_no_gpu(jax) + " — GPU-прогон делает архитектор на стенде (AD-7)"
        doc.update(error_type="no-gpu", stage="env", error=reason)
        dtypes_planned = tuple(args.dtype) if args.dtype else DEFAULT_DTYPES
        cases_planned = CASES if args.cases else (CASES[0],)
        doc["cases"] = [{"shape": [m, k, n], "dtype": dt, "status": "blocked",
                        "stage": "env", "error_type": "no-gpu", "error": reason,
                        "max_abs": None, "rel": None, "seconds": None}
                       for dt in dtypes_planned for (m, k, n) in cases_planned]
        _write(Path(args.output), doc)
        print(json.dumps(doc, ensure_ascii=False, indent=2))
        return 0

    dtypes = tuple(args.dtype) if args.dtype else DEFAULT_DTYPES
    cases = CASES if args.cases else (CASES[0],)
    results = []
    for dtype_str in dtypes:
        for (m, k, n) in cases:
            res = _run_case(jax, jnp, plgpu, m, k, n, dtype_str)
            results.append(res)
            print(f"[{dtype_str} M={m} K={k} N={n}] {res['status']} "
                  f"max_abs={res['max_abs']} err={res['error_type']}")

    doc["cases"] = results
    worst = results[0] if results else None
    if worst is not None:
        doc.update(status=worst["status"], shape=worst["shape"], dtype=worst["dtype"],
                   max_abs=worst["max_abs"], rel=worst["rel"], seconds=worst["seconds"],
                   error_type=worst["error_type"], error=worst["error"], stage=worst["stage"])
    _write(Path(args.output), doc)
    return 0 if doc["status"] == "correctness-pass" else 1


def _write(path: Path, doc: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
