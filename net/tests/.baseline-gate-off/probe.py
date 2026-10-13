
import json, sys
sys.path.insert(0, "net/tests")
import conftest  # noqa: F401 — пины ADR-010 до import jax
import numpy as np
from conftest import tiny_config
from net import checkpoint, train_loop as tl

cfg = tiny_config()  # та же геометрия, что у ноги теста (общая для обеих сторон)
rng = np.random.default_rng(5)
batches = [rng.integers(0, cfg.vocab_size, size=(1, 16), dtype=np.int32) for _ in range(3)]
result = tl.train(
    cfg, iter(batches),
    train_config=tl.TrainConfig(steps=3, lr=1e-2, seed=7, schedule="cosine",
                                param_dtype="float32", log_every=0),
    budget=tl.Budget(run_ref="gate-off", path=__import__("pathlib").Path("/dev/null"),
                     present=True),
)
print(json.dumps({
    "losses": [repr(x) for x in result.losses],
    "params_hash": checkpoint.tree_hash(result.params),
    "master_hash": checkpoint.tree_hash(result.master_params),
    "param_dtype": str(__import__("jax").tree_util.tree_leaves(result.params)[0].dtype),
}))
