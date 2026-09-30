# ADR-018 64K capture — why the scan form is not bit-equal to an *eager* `model.forward`

Date: 2026-09-30. Instrument: `tools/bench_block_merge.py` (capture of the MLA
layer inputs), tests `tools/tests/test_bench_block_merge.py` (T-b4, T-b4b, T-b4c).

## The question the 64K bench needed answered

`capture_forward` must return the activations `net/model.py`'s forward actually
produces.  The reference form (the layer loop) is bit-equal to `model.forward`
— but it does **not fit** a 16 GB card at 64K:

```
hlo_rematerialization.cc: Can't reduce memory use below 9.98GiB (10721451131 bytes)
  by rematerialization; only reduced to 20.27GiB (21763894324 bytes), down from 20.27GiB
bfc_allocator.cc: ran out of memory trying to allocate 15.39GiB   # the loop form
```

The scan form (repeated four-layer units in one `lax.scan`) does fit
(`only reduced to 10.86GiB, down from 11.25GiB`; forward 88–91 s at T=65536),
so it is the form the 64K numbers come from.  Its parity is therefore the thing
that had to be established.

## What was measured

`tiny_cfg` fixture, `T = 512`, `chunk_size = 16`, both forms over the same
parameters (reproduce with the snippets below):

| comparison | bit-equal | max abs diff |
|---|---|---|
| eager loop form vs `model.forward` | **yes** | 0 |
| eager **scan** form vs `model.forward` | no | 3.53e-05 |
| `jit` loop form vs `jit model.forward` | **yes** | 0 |
| `jit` **scan** form vs `jit model.forward` | **yes** | 0 |
| `jit` loop form vs `jit` scan form | **yes** | 0 |
| `jax.jit(rms_norm)` vs `rms_norm` | no | 4.77e-07 |
| `jax.jit(model.forward)` vs `model.forward` | no | 4.58e-05 |
| matmul: eager vs inside `lax.scan` | **yes** | 0 |
| the scan form with the units unrolled in python (stacked params, sliced back) vs `model.forward` | **yes** | 0 |

## Diagnosis

The scan form's **arithmetic is exact** — unrolled, with the very same stacked
parameter tree, it reproduces `model.forward` bit for bit at every layer
boundary.  The residual is a *lowering* artifact, and it is not specific to
`lax.scan`:

* `jax.jit(rms_norm) != rms_norm` by one ULP, with no scan anywhere.  XLA
  lowers a reduction inside a compiled body in a different association order
  than op by op; matmuls are unaffected (0.0 difference), reductions are not.
* RMSNorm sits on every layer, so ~1 ULP per layer compounds: 8.9e-08 at the
  first sub-layer of the scanned unit, 3.9e-06 after four, 3.53e-05 for the
  whole fixture — the same order as `jax.jit(model.forward)` against
  `model.forward` itself (4.58e-05).
* `lax.scan` compiles its body by construction.  **No** scan-based capture can
  reproduce an eagerly evaluated `model.forward` bit for bit; conversely, the
  scan form is bit-equal to the model as soon as the reference is compiled too,
  which is how the bench consumes the capture (`capture_leg` jits it).

## What the instrument does with this

* The form is declared, not inferred: `--capture-form {auto,loop,scan}` /
  `capture_form()`, recorded in the report at `bounds.capture_form` (with the
  estimate and the device budget that decided it, `bounds.capture_form_why`).
* `auto` keeps the **reference (eager-exact) form** while its estimated live
  set fits the device budget and takes the scan form when it does not — so
  small fixtures, including the whole existing test suite, run on the form whose
  parity is unconditional.
* Parity is pinned for **both** forms, each in the environment it runs in:
  eagerly for the reference form (T-b4, unchanged), compiled for the scan form
  (T-b4b, new).  The scan form cannot be held to the eager assertion, and this
  note is the measurement behind that statement.

## Reproduction

```bash
~/venv-axiom/bin/python - <<'PY'
import sys; sys.path[:0] = ['/home/roman/axiom', '/home/roman/axiom/tools',
                           '/home/roman/axiom/tools/tests']
import jax, jax.numpy as jnp
import bench_block_merge as bench
from net import model
from test_bench_block_merge import tiny_cfg, _tiny_params, T_TINY
cfg, params = tiny_cfg(), _tiny_params(tiny_cfg())
ids = jnp.arange(T_TINY, dtype=jnp.int32)[None, :] % cfg.vocab_size

def fwd(p, x):
    return model.forward(p, cfg, x, chunk_size=16, use_attnres=False,
                         return_hidden=True)[1]

def cap(p, x, form):
    return bench.capture_forward(p, cfg, x, use_attnres=False, chunk_size=16,
                                 form=form)[0]

for name, a, b in (
    ("eager loop vs eager fwd", cap(params, ids, "loop"), fwd(params, ids)),
    ("eager scan vs eager fwd", cap(params, ids, "scan"), fwd(params, ids)),
    ("jit scan  vs jit  fwd", jax.jit(lambda p, x: cap(p, x, "scan"))(params, ids),
                              jax.jit(fwd)(params, ids)),
):
    print(f"{name}: equal={bool(jnp.array_equal(a, b))} "
          f"max={float(jnp.max(jnp.abs(a - b))):.3e}")
PY
```

Memory (the other half of the story, both measured on the RTX 4080 SUPER,
16376 MiB): the loop form needs a 15.39 GiB allocation (and XLA reports a
20.27 GiB program); the scan form needs 10.86 GiB and runs.  On top of that,
jax caps itself at `XLA_PYTHON_CLIENT_MEM_FRACTION` (75% by default ≈ 11.7 GiB
here, *not* the card), which is why the 64K run is invoked with
`XLA_PYTHON_CLIENT_MEM_FRACTION=0.95` (and `XLA_PYTHON_CLIENT_PREALLOCATE=false`,
or the merged leg's CUBIN fails to load).
