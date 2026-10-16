import sys, os
sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, "/home/roman/experiments/agents/0710-ariadna/plugins/gb10/skills/pallas-gb10-kernel/scripts")
import lower_check
lower_check.patch_for_gb10()
import jax
from jax.experimental.mosaic import gpu as mosaic_mod
from jax.experimental.mosaic.gpu import utils as mosaic_utils
import jax.numpy as jnp
from net.kernels import mla_persist as P

mosaic_utils._infer_arch = lambda: (12, 1)

def measured(build, case, limit=1, bwd=False):
    with mosaic_mod.core.artificial_shared_memory_limit(limit):
        mk = P.make_inputs_bwd if bwd else P.make_inputs
        shapes = jax.eval_shape(lambda: mk(jax.random.key(0), **case))
        fn = jax.jit(lambda *a: build(**case)(*a))
        try:
            fn.trace(*shapes).lower(lowering_platforms=("cuda",))
            return None
        except Exception as e:
            msg = str(e)
            if "shared memory" in msg and "smem_bytes=" in msg:
                return int(msg.split("smem_bytes=")[1].split()[0])
            raise

for name, build in [("fwd", lambda **c: P._build_kernel(c["B"], c["T"], c["H"], c["dq"], c["bq"], c["bk"], c["dtype"])),
                    ("bwd", lambda **c: P._build_bwd_kernel(c["B"], c["T"], c["H"], c["dq"], c["bq"], c["bk"], c["dtype"]))]:
    for case in P.CASES:
        try:
            n = measured(build, case, bwd=(name == "bwd"))
            print(f"{name} {case['B']}x{case['T']}x{case['H']} dq={case['dq']} bq={case['bq']} bk={case['bk']}: smem={n}")
        except Exception as e:
            print(f"{name} {case}: FAIL {type(e).__name__} {str(e).strip().splitlines()[-1][:150]}")
    # oversize bk=128 on the mini case
    case = dict(P.CASES[2]); case["bk"] = 128
    try:
        n = measured(build, case, bwd=(name == "bwd"))
        print(f"{name} mini bk=128: smem={n}")
    except Exception as e:
        print(f"{name} mini bk=128: FAIL {type(e).__name__} {str(e).strip().splitlines()[-1][:150]}")
