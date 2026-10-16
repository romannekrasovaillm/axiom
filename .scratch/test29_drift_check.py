"""test_29-разбор: пилотный forward current (дельта) vs 97c6eb8 (база дельты) vs 773e389.

Если current == 97c6eb8 побитово, а 97c6eb8 != 773e389 — красный test_29
унаследованный дрейф, не дельта вехи-12.
"""
import io
import subprocess
import sys
import tarfile
from pathlib import Path

sys.path.insert(0, "net/tests")
sys.path.insert(0, ".")
import conftest  # noqa: F401 — пины ADR-010 до import jax

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

CASE = Path(".").resolve()
SCRATCH = CASE / ".scratch"


def extract(commit: str, tag: str) -> Path:
    root = SCRATCH / f"tree-{tag}"
    if not (root / tag).exists():
        root.mkdir(parents=True, exist_ok=True)
        arch = subprocess.run(
            ["git", "-C", str(CASE), "archive", commit, "net/"],
            check=True, capture_output=True,
        )
        with tarfile.open(fileobj=io.BytesIO(arch.stdout)) as bundle:
            bundle.extractall(root)
        (root / "net").rename(root / tag)
    return root


def load(commit: str, tag: str):
    root = extract(commit, tag)
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    import importlib
    cfg_mod = importlib.import_module(f"{tag}.config")
    model_mod = importlib.import_module(f"{tag}.model")
    return cfg_mod, model_mod


# PILOT_KWARGS из test_29 (импорт файла)
import importlib.util

spec = importlib.util.spec_from_file_location("t29", "net/tests/test_29_compute_dtype.py")
t29 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(t29)
PILOT = dict(t29.PILOT_KWARGS)

import net.config as cur_cfg  # noqa: E402
import net.model as cur_model  # noqa: E402


def forward_bits(cfg_mod, model_mod):
    cfg = cfg_mod.ModelConfig(**PILOT)
    params = model_mod.init_params(jr.PRNGKey(0), cfg)
    ids = jr.randint(jr.PRNGKey(2), (2, 16), 0, cfg.vocab_size)
    out = model_mod.forward(params, cfg, ids, grad_ckpt_policy="none")
    return np.asarray(out).view(np.uint32)


cur = forward_bits(cur_cfg, cur_model)
c97 = load("97c6eb8", "tree97")[0], load("97c6eb8", "tree97")[1]
out97 = forward_bits(*c97)
c773 = load("773e389", "tree773")[0], load("773e389", "tree773")[1]
out773 = forward_bits(*c773)

print("current == 97c6eb8 (база дельты):", bool(np.array_equal(cur, out97)))
print("97c6eb8 == 773e389 (baseline test_29):", bool(np.array_equal(out97, out773)))
print("current == 773e389:", bool(np.array_equal(cur, out773)))
f_cur = np.asarray(cur).view(np.float32)
f_773 = np.asarray(out773).view(np.float32)
print("max |f| diff current vs 773e389:", float(np.abs(f_cur - f_773).max()))
