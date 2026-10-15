"""Проверка counting-sort: паритет, перестановка, дегенеративные случаи, гейт."""
import os, sys
os.environ.setdefault("JAX_PLATFORMS", "cpu")
ROOT = "/home/roman/.arch-ml/worktrees/axiom-43e236d789188868/theseus-20261015183027"
sys.path.insert(0, ROOT + "/net/tests"); sys.path.insert(0, ROOT)
import numpy as np
import jax
jax.config.update("jax_default_matmul_precision", "highest")
import jax.numpy as jnp
import jax.random as jr
from conftest import tiny_config, small_config
from net import moe


def positions_ref(expert_of, n):
    pos = np.empty(expert_of.shape[0], dtype=np.int64)
    # стабильный порядок появления: для каждого эксперта — очередь позиций
    counts = np.zeros(n, dtype=np.int64)
    for i, e in enumerate(np.asarray(expert_of)):
        pos[i] = counts[e]
        counts[e] += 1
    offs = np.concatenate([[0], np.cumsum(counts)[:-1]])
    return pos + offs[np.asarray(expert_of)]


def check_positions(expert_of, n, name):
    group_sizes = jnp.bincount(expert_of, length=n).astype(jnp.int32)
    got = np.asarray(moe._counting_sort_positions(expert_of, group_sizes))
    ref = positions_ref(expert_of, n)
    ok = np.array_equal(got, ref)
    # и эквивалентность стабильному argsort
    order = np.asarray(jnp.argsort(expert_of, stable=True))
    ok2 = all(order[got[i]] == i for i in range(len(expert_of)))
    print(f"[pos] {name}: equal_ref={ok} pos_is_inverse_argsort={ok2}")
    assert ok and ok2


def main():
    n = 6
    rng = np.random.default_rng(0)
    check_positions(jnp.asarray(rng.integers(0, n, size=32)), n, "random")
    check_positions(jnp.asarray([3] * 10), n, "all-one")
    check_positions(jnp.asarray(list(range(n)) * 3), n, "round-robin")
    check_positions(jnp.asarray([0, 0, 5, 5, 5, 1, 1, 5, 0]), n, "duplicates")
    check_positions(jnp.asarray(rng.integers(0, n, size=1)), n, "single")
    check_positions(jnp.asarray(rng.integers(0, n, size=1)), n, "single2")
    check_positions(jnp.asarray([], dtype=jnp.int32), n, "empty")

    cfg = tiny_config()
    params = moe.init_moe(jr.PRNGKey(1), cfg)
    x = jr.normal(jr.PRNGKey(3), (9, cfg.hidden))
    os.environ.pop(moe.COUNTING_MODE_ENV, None)
    assert moe.counting_sort_mode() is False
    try:
        moe.counting_sort_mode("banana")
        raise AssertionError("опечатка прошла")
    except ValueError as e:
        print("[gate]", str(e)[:70])

    off = moe.apply(params, cfg, x)
    os.environ[moe.COUNTING_MODE_ENV] = "1"
    assert moe.counting_sort_mode() is True
    on = moe.apply(params, cfg, x)
    os.environ.pop(moe.COUNTING_MODE_ENV, None)

    same = bool(jnp.array_equal(off, on))
    print("[parity] full moe apply bitwise equal:", same,
          "max|d| =", float(jnp.max(jnp.abs(off - on))))
    assert same

    # bf16-режим compute
    import net.compute_dtype as cd
    os.environ["AXIOM_COMPUTE_DTYPE"] = "bf16"
    try:
        cd.reload_mode() if hasattr(cd, "reload_mode") else None
    except Exception:
        pass
    print("[dtype mode]", cd.is_bf16() if hasattr(cd, "is_bf16") else "?")
    os.environ.pop("AXIOM_COMPUTE_DTYPE", None)

    # средний конфиг с реальным числом экспертов
    cfg2 = small_config()
    params2 = moe.init_moe(jr.PRNGKey(5), cfg2)
    x2 = jr.normal(jr.PRNGKey(6), (37, cfg2.hidden))
    off2 = moe.apply(params2, cfg2, x2)
    os.environ[moe.COUNTING_MODE_ENV] = "1"
    on2 = moe.apply(params2, cfg2, x2)
    os.environ.pop(moe.COUNTING_MODE_ENV, None)
    same2 = bool(jnp.array_equal(off2, on2))
    print("[parity] small cfg (n_routed=%d) bitwise equal:" % cfg2.moe_num_routed, same2)
    assert same2
    print("[ok]")


if __name__ == "__main__":
    main()
