"""D-8 isolation: staged jit-compile of the full l3-full graph on GB10 (sm_121)."""
import sys, os, time
sys.path.insert(0, "/home/roman/axiom")
sys.path.insert(0, "/home/roman/axiom/tools")

# ADR-041: дисциплина памяти JAX — префлайт ДО import jax (лимит XLA + гейт стенда).
import jax_preflight

jax_preflight.ensure_mem_fraction()
jax_preflight.gate_or_exit()

os.environ.setdefault("JAX_PLATFORMS", "gpu")
import jax, jax.numpy as jnp
import net.model as model, net.config as config

cfg = config.load_config("/home/roman/axiom/net/config.json")
key = jax.random.PRNGKey(0)
T = 64
cfg = config.with_vocab(cfg, 1024) if hasattr(config, "with_vocab") else cfg
params = model.init_params(key, cfg)
ids = jnp.ones((1, T), dtype=jnp.int32)
stage = sys.argv[1]

if stage == "1-forward":
    f = jax.jit(lambda p, x: model.forward(p, cfg, x, return_hidden=True))
elif stage == "2-loss":
    f = jax.jit(lambda p, x: model.compute_loss(p, cfg, x, chunk_size=cfg.chunk_size if hasattr(cfg,'chunk_size') else 64))
elif stage == "3-grad":
    f = jax.jit(jax.value_and_grad(lambda p, x: model.compute_loss(p, cfg, x, chunk_size=64)))
t0 = time.time()
out = f(params, ids)
out[0].block_until_ready() if isinstance(out, tuple) else out.block_until_ready()
print(f"STAGE {stage}: OK in {time.time()-t0:.1f}s", flush=True)
