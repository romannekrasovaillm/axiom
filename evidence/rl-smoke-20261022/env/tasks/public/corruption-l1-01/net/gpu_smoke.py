"""GPU smoke runner — writes ``net/gpu-smoke-report.json`` (spec §6.10, §7).

Measures the *actual* throughput (tokens/s) of the skeleton:

* tiny config — real forward + train step on the GPU (mirrors
  ``tests/test_10_gpu_smoke.py``, plus a pure-forward timing);
* full 1B config — ``jax.eval_shape`` parameter count (no allocation), then
  an optional real forward if it fits the card; an OOM is recorded honestly,
  not hidden.

Numbers are facts, not targets: no tokens/s threshold is asserted here — the
threshold N is fixed later against the recorded hardware (spec §7).

Run from the case root:  ``python -m net.gpu_smoke``  (writes the report next
to this file).  Exit code 0 when the tiny GPU smoke ran, 1 when no CUDA
device is available (the report is still written, status ``blocked``).
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

NET_DIR = Path(__file__).resolve().parent
REPORT_PATH = NET_DIR / "gpu-smoke-report.json"


def _load_tiny_config_factory():
    """Reuse the acceptance tests' tiny config (no config drift)."""
    conftest = NET_DIR / "tests" / "conftest.py"
    spec = importlib.util.spec_from_file_location("net_tests_conftest", conftest)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.tiny_config


def _jax_devices() -> list[str]:
    import jax

    try:
        return [f"{d.platform}:{d.device_kind}" for d in jax.devices()]
    except RuntimeError:
        return []


def _gpu_device():
    import jax

    try:
        for d in jax.devices():
            if d.platform == "gpu":
                return d
    except RuntimeError:
        pass
    return None


def _nvidia_smi() -> dict:
    """GPU name / VRAM / driver from nvidia-smi (empty dict when absent)."""
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True, text=True, timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired):
        return {}
    if proc.returncode != 0 or not proc.stdout.strip():
        return {}
    name, mem_mb, driver = [p.strip() for p in proc.stdout.strip().splitlines()[0].split(",")]
    return {"accelerator": name, "vram_total_mb": int(mem_mb), "driver": driver}


def _versions() -> dict:
    import jax
    import jaxlib

    def _pkg(name: str) -> str | None:
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            return None

    return {
        "python": platform.python_version(),
        "jax": jax.__version__,
        "jaxlib": jaxlib.__version__,
        "jax_cuda12_plugin": _pkg("jax-cuda12-plugin"),
        "jax_cuda12_pjrt": _pkg("jax-cuda12-pjrt"),
    }


def _vram_peak_mb(device) -> float | None:
    """Peak device memory in MiB (GPU-only stats; None when unavailable)."""
    try:
        stats = device.memory_stats()
    except Exception:  # noqa: BLE001 — статистика памяти опциональна
        return None
    if not stats or "peak_bytes_in_use" not in stats:
        return None
    return round(stats["peak_bytes_in_use"] / 2**20, 1)


def _measure_tiny(device) -> dict:
    """Real forward + train-step timing on the tiny config (mirror of test_10)."""
    import jax
    import jax.numpy as jnp
    import jax.random as jr

    from net import model, optimizer

    tiny_config = _load_tiny_config_factory()
    cfg = tiny_config()
    key = jr.PRNGKey(0)
    params = model.init_params(key, cfg)

    # --- train step (forward + backward + optimizer), as in test_10 ----------
    ids = jr.randint(key, (2, 128), 0, cfg.vocab_size)

    def loss_fn(p, x):
        return model.compute_loss(p, cfg, x, chunk_size=16)

    grad_fn = jax.jit(jax.value_and_grad(loss_fn))
    step = optimizer.make_step(cfg)
    state = optimizer.init_state(params)
    loss, grads = grad_fn(params, ids)  # warmup (compile)
    t0 = time.time()
    for _ in range(10):
        loss, grads = grad_fn(params, ids)
        params, state = step(params, grads, state, 1e-3)
    jax.block_until_ready(loss)
    train_dt = time.time() - t0
    train_tok_s = 10 * 2 * 128 / train_dt

    # --- pure forward (inference path) ---------------------------------------
    fwd_ids = jr.randint(key, (2, 2048), 0, cfg.vocab_size)
    fwd = jax.jit(lambda p, x: model.forward(p, cfg, x, chunk_size=64))
    out = fwd(params, fwd_ids)  # warmup (compile)
    t0 = time.time()
    for _ in range(10):
        out = fwd(params, fwd_ids)
    jax.block_until_ready(out)
    fwd_dt = time.time() - t0
    fwd_tok_s = 10 * 2 * 2048 / fwd_dt

    return {
        "params": model.param_count(cfg),
        "train": {"batch": 2, "seq_len": 128, "steps": 10, "tok_s": round(train_tok_s, 1)},
        "forward": {"batch": 2, "seq_len": 2048, "steps": 10, "tok_s": round(fwd_tok_s, 1)},
        "vram_peak_mb": _vram_peak_mb(device),
    }


def _measure_full(device) -> dict:
    """Full 1B config: shape-only count + optional forward (OOM recorded)."""
    import jax
    import jax.random as jr

    from net import model
    from net.config import ModelConfig

    cfg = ModelConfig()
    result: dict = {"params": model.param_count(cfg), "oom": False, "note": "", "attempts": []}

    def _forward(seq_len: int) -> float:
        key = jr.PRNGKey(0)
        params = model.init_params(key, cfg)
        ids = jr.randint(key, (1, seq_len), 0, cfg.vocab_size)
        fwd = jax.jit(lambda p, x: model.forward(p, cfg, x, chunk_size=64))
        out = fwd(params, ids)  # warmup (compile)
        t0 = time.time()
        for _ in range(3):
            out = fwd(params, ids)
        jax.block_until_ready(out)
        dt = time.time() - t0
        return 3 * seq_len / dt

    for seq_len in (8192, 2048):
        try:
            tok_s = _forward(seq_len)
        except Exception as exc:  # noqa: BLE001 — OOM честно фиксируется, не скрывается
            result["oom"] = True
            result["attempts"].append(
                {"seq_len": seq_len, "error": f"{type(exc).__name__}: {str(exc)[:300]}"}
            )
            continue
        result.update({
            "mode": "forward",
            "seq_len": seq_len,
            "batch": 1,
            "forward_tok_s": round(tok_s, 1),
            "vram_peak_mb": _vram_peak_mb(device),
            "oom": False,
            "note": "" if seq_len == 8192 else f"8K не влез, замер на {seq_len}",
        })
        return result
    result["mode"] = "shape-only"
    result["vram_peak_mb"] = _vram_peak_mb(device)
    return result


def main() -> int:
    generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    report: dict = {
        "report": "gpu-smoke",
        "spec": "MODEL-L3-SKELETON.md §6.10 / §7",
        "generated_at": generated_at,
        "hardware": _nvidia_smi(),
        "versions": _versions(),
        "devices": _jax_devices(),
        "results": {},
        "deviations": [],
    }

    device = _gpu_device()
    if device is None:
        report["status"] = "blocked"
        report["deviations"].append(
            "CUDA-enabled jaxlib не установлен: GPU-смоук невозможен, "
            "замеров нет (фаза B фиксируется как blocked)"
        )
        REPORT_PATH.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f"blocked: no CUDA-enabled JAX device; report -> {REPORT_PATH}")
        return 1

    report["status"] = "ok"
    report["results"]["tiny"] = _measure_tiny(device)
    report["results"]["full"] = _measure_full(device)
    if report["results"]["full"].get("oom"):
        report["status"] = "partial"
        report["deviations"].append(
            "полный 1B-конфиг: forward не влез в VRAM — зафиксирован OOM, "
            "замер только shape-only (честная фиксация по задаче)"
        )

    REPORT_PATH.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report["results"], ensure_ascii=False, indent=2))
    print(f"report -> {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
