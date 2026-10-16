import sys, os
sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("net/tests"))
import numpy as np
from conftest import small_config
from net import train_loop as tl
from net.kernels import mla_persist as P

def emu_kernel(q, k, v, *, bq=P.BQ, bk=P.BK):
    return P.emulate_attention(q, k, v, bq=int(bq), bk=int(bk))
def emu_kernel_bwd(q, k, v, do, D, m, l, *, bq=P.BQ, bk=P.BK):
    return P.emulate_attention_bwd(q, k, v, do, D, m, l, bq=int(bq), bk=int(bk))

def run(tag, env):
    P.kernel, P.kernel_bwd, P.device_available = emu_kernel, emu_kernel_bwd, (lambda: True)
    saved = {k: os.environ.get(k) for k in env}
    os.environ.update({k: v for k, v in env.items()})
    cfg = small_config(mla_head_dim=16)
    rng = np.random.default_rng(11)
    batches = [rng.integers(0, cfg.vocab_size, size=(1, 16), dtype=np.int32) for _ in range(2)]
    try:
        res = tl.train(cfg, iter(batches), train_config=tl.TrainConfig(
            steps=2, lr=1e-3, seed=7, schedule="cosine", micro_batch=1,
            log_every=0, kpi_every=20), budget=tl.Budget(run_ref="probe", path=os.devnull, present=True))
        print(tag, "OK losses", [float(x) for x in res.losses])
    except Exception as e:
        print(tag, "FAIL", type(e).__name__, str(e).strip().splitlines()[-1][:120])
    finally:
        for k, v in saved.items():
            if v is None: os.environ.pop(k, None)
            else: os.environ[k] = v

run("persist only", {"AXIOM_MLA_PERSIST": "1"})
run("compute=bf16", {"AXIOM_MLA_PERSIST": "1", "AXIOM_COMPUTE_DTYPE": "bf16"})
run("master=bf16", {"AXIOM_MLA_PERSIST": "1", "AXIOM_MASTER_DTYPE": "bf16"})
run("both", {"AXIOM_MLA_PERSIST": "1", "AXIOM_COMPUTE_DTYPE": "bf16", "AXIOM_MASTER_DTYPE": "bf16"})
