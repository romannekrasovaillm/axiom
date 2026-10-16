"""Shared fixtures for the L3 skeleton acceptance tests.

The functional tests run on a *small* model config so they complete on a CPU
in reasonable time; the full 1B config is exercised by the parameter-budget
test (shape-only, no allocation).  ``small_config`` keeps the 3:1 KDA:MLA layer
ratio of the full model.

ADR-010 — this module is the single pinning point of the acceptance contour:

* the JAX backend is selected by the suite, not by the caller's shell: the
  ``nvidia-*`` wheel libraries are preloaded before ``import jax`` so the CUDA
  plugin initialises without a hand-exported ``LD_LIBRARY_PATH`` (otherwise the
  same commit yields ``[CpuDevice(id=0)]`` in one terminal and
  ``[CudaDevice(id=0)]`` in another), and an explicit CPU run stays on CPU
  (``NET_JAX_BACKEND=cpu`` → ``JAX_PLATFORMS=cpu``);
* ``jax_default_matmul_precision`` is pinned to ``highest`` so the oracles
  measure the algorithm rather than the hardware's precision policy (criterion 2
  is red on a GPU with the default policy — 3.37e-04 against a 1e-4 threshold —
  and green under ``highest``/``high``: 1.2e-07);
* both parameters are declared in the run header (``pytest_sessionstart``), so
  the verdict carries the platform it was measured on.

ADR-013 — the same point pins the *determinism* of the computation:

* in the gate profile ``--xla_gpu_deterministic_ops`` is added to ``XLA_FLAGS``
  before ``import jax`` (XLA parses the variable once, at client creation), so
  a rerun of a gate run is bit-reproducible on the GPU and A5 — "a rerun yields
  the same manifest", ``workspace_sha256`` included — holds by construction
  rather than by luck;
* the local profile does not set it (iteration speed) and does not pretend to:
  the header reports the mode and the raw ``XLA_FLAGS`` it was produced with.

The pins are exercised end to end from a fresh process by
``tools/check_precision_pinning.py`` (behavioural guard C-042).
"""

from __future__ import annotations

import ctypes
import os
import sys
import sysconfig
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# --------------------------------------------------------------------------
# ADR-010: backend and matmul-precision pinning (the single place, see above).
# --------------------------------------------------------------------------

#: Precision policy every acceptance oracle runs under (ADR-010, decision 1).
#: Not configurable on purpose: the perturbation of criterion 2 lives here.
PINNED_MATMUL_PRECISION = "highest"

#: Backend selection: ``auto`` (default) | ``gpu`` | ``cpu``.
BACKEND_ENV = "NET_JAX_BACKEND"

#: Gate profile marker (ADR-010, decision 3).  Truthy value → the run claims A4
#: gate status: a missing CUDA device is a FAIL, not a SKIP.  Unset/falsy →
#: local run, where the same absence is an explicit, preliminary SKIP.
GATE_ENV = "NET_GATE_PROFILE"

_LOCAL_VALUES = {"0", "false", "no", "off", "local"}

#: Determinism pin (ADR-013, decision 1).  XLA does not guarantee bit-identical
#: GPU kernels by default, so two formally identical gate runs could produce
#: different artifacts and therefore different ``workspace_sha256`` — A5
#: ("a rerun yields the same manifest") then holds by luck, not by construction.
#: Set in the gate profile only: local iterations keep the fast kernels.
DETERMINISM_FLAG = "--xla_gpu_deterministic_ops"

#: The variable XLA parses — once, when its client initialises.  Read back by
#: the run header and by the C-042 guard so both report what the process that
#: actually computed had, not what some file says it should have.
XLA_FLAGS_ENV = "XLA_FLAGS"

#: CUDA libraries shipped by the local ``nvidia-*`` wheels, in load order
#: (dependencies first: cusparse needs nvJitLink, cudnn needs cublas, ...).
#: Matching is case-insensitive because the wheel file names are camel-cased
#: (``nvjitlink/lib/libnvJitLink.so.12``).  Preloading them into the global
#: scope also keeps the system CUDA in ``/usr/local/cuda-12.4`` from winning
#: the SONAME race (its libnvJitLink 12.4 cannot satisfy cusparse 12.9 —
#: see net/README.md).
_CUDA_LIB_ORDER = (
    "libnvjitlink",
    "libcudart",
    "libcublaslt",
    "libcublas",
    "libcusparse",
    "libcusolver",
    "libcufft",
    "libcudnn",
    "libnccl",
)


def backend_mode() -> str:
    """Requested backend: ``auto`` (default), ``gpu`` or ``cpu``.

    An unrecognised value falls back to ``auto`` (and is echoed in the header),
    so a typo cannot silently force a CPU gate run.
    """
    value = (os.environ.get(BACKEND_ENV) or "auto").strip().lower()
    return value if value in ("auto", "gpu", "cpu") else "auto"


def gate_profile() -> bool:
    """True when the run claims A4 gate status (ADR-010, decision 3).

    Fail-safe: any non-empty value that is not an explicit local marker counts
    as the gate profile, so a typo makes the run stricter, never looser.
    """
    raw = os.environ.get(GATE_ENV)
    if raw is None or not raw.strip():
        return False
    return raw.strip().lower() not in _LOCAL_VALUES


def cuda_library_paths() -> list[Path]:
    """Absolute paths of the CUDA libraries shipped by the local wheels.

    ``NET_NVIDIA_LIB_ROOT`` overrides the search root (pointing it at a
    ``nvidia``-layout directory); by default the running interpreter's
    ``site-packages/nvidia`` is used.
    """
    override = (os.environ.get("NET_NVIDIA_LIB_ROOT") or "").strip()
    site_packages = Path(sysconfig.get_paths()["purelib"])
    roots = [Path(override)] if override else [site_packages / "nvidia"]

    buckets: dict[str, list[Path]] = {stem: [] for stem in _CUDA_LIB_ORDER}
    for root in roots:
        if not root.is_dir():
            continue
        for path in sorted(root.glob("*/lib/*.so*")):
            stem = path.name.split(".so")[0].lower()
            if stem in buckets:
                buckets[stem].append(path)
    return [path for stem in _CUDA_LIB_ORDER for path in buckets[stem]]


def select_backend() -> list[str]:
    """Select the backend explicitly, before ``import jax`` (ADR-010, decision 2).

    * ``NET_JAX_BACKEND=cpu`` pins the run to CPU via ``JAX_PLATFORMS=cpu`` — an
      explicit choice must hold regardless of what the shell's ``LD_LIBRARY_PATH``
      happens to make JAX discover (a local preliminary run has to be
      reproducible too).
    * ``auto`` (default) and ``gpu`` preload the wheels' CUDA libraries so the
      CUDA plugin initialises without a hand-exported environment: with the
      system CUDA in ``LD_LIBRARY_PATH`` the plugin picks a libcusparse whose
      nvJitLink version does not match and silently falls back to CPU.

    Best effort by design: a library that cannot be loaded is skipped and the
    run header then declares the backend the suite actually got.  Returns the
    library file names preloaded.
    """
    if backend_mode() == "cpu":
        os.environ["JAX_PLATFORMS"] = "cpu"
        return []
    loaded: list[str] = []
    for path in cuda_library_paths():
        try:
            ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)
        except OSError:
            continue
        loaded.append(path.name)
    return loaded


def xla_flags_value() -> str:
    """Raw ``XLA_FLAGS`` of this process (empty string when unset)."""
    return os.environ.get(XLA_FLAGS_ENV, "")


def xla_flags_tokens() -> list[str]:
    """``XLA_FLAGS`` as tokens (``--flag`` / ``--flag=value``)."""
    return [token for token in xla_flags_value().split() if token]


def determinism_pinned() -> bool:
    """True when ``XLA_FLAGS`` carries the deterministic-ops flag (ADR-013).

    The flag *name* is matched, not the whole token: XLA accepts both
    ``--xla_gpu_deterministic_ops`` and ``--xla_gpu_deterministic_ops=true``.
    """
    return any(
        token.split("=", 1)[0] == DETERMINISM_FLAG for token in xla_flags_tokens()
    )


def apply_determinism_pinning() -> bool:
    """Set ``--xla_gpu_deterministic_ops`` in the gate profile (ADR-013, 1).

    Must run **before** XLA initialises: the flags are parsed once, when the
    client is created, so a flag added later would leave a run that *declares*
    determinism without having it — precisely the failure mode ADR-013 exists
    to prevent.  An ``XLA_FLAGS`` already exported by the caller is extended,
    never replaced.  Returns whether the flag is in effect afterwards (a local
    run that inherits the flag from its shell reports ``True`` too — the header
    says where it came from).
    """
    if gate_profile() and not determinism_pinned():
        os.environ[XLA_FLAGS_ENV] = f"{xla_flags_value()} {DETERMINISM_FLAG}".strip()
    return determinism_pinned()


#: ADR-013 pin, applied before ``import jax`` (reported by the run header).
DETERMINISM_ENABLED = apply_determinism_pinning()

#: Libraries preloaded before ``import jax`` (reported by the run header).
PRELOADED_CUDA_LIBS = select_backend()

import jax  # noqa: E402  — must follow select_backend()
import pytest  # noqa: E402

from net.config import ModelConfig  # noqa: E402


def apply_precision_pinning() -> str:
    """Pin the matmul precision policy and verify the setting held (ADR-010).

    Called at import time by this module, and again from a fresh process by
    ``tools/check_precision_pinning.py``: the guard needs the *actual* runtime
    configuration, not the presence of a line in this file.  Raises if the
    setting did not survive (e.g. the config key was renamed by a JAX upgrade),
    because a silent loss of the pin is exactly what ADR-010 warns about.
    """
    jax.config.update("jax_default_matmul_precision", PINNED_MATMUL_PRECISION)
    effective = jax.config.jax_default_matmul_precision
    if effective != PINNED_MATMUL_PRECISION:
        raise RuntimeError(
            "ADR-010 pinning lost: jax_default_matmul_precision="
            f"{effective!r}, expected {PINNED_MATMUL_PRECISION!r}"
        )
    return effective


def describe_devices() -> list[str]:
    """``jax.devices()`` rendered for the run header (never raises).

    ``repr`` rather than ``str``: ``CudaDevice(id=0)`` names the platform, which
    is what the verdict has to carry (``str`` gives just ``cuda:0``).
    """
    try:
        return [repr(device) for device in jax.devices()]
    except Exception as exc:  # broken plugin: report it, do not mask collection
        return [f"<unavailable: {exc}>"]


def gpu_devices() -> list[str]:
    """GPU devices of the current backend (empty when running on CPU)."""
    try:
        return [repr(device) for device in jax.devices() if device.platform == "gpu"]
    except Exception:
        return []


def determinism_mode() -> str:
    """How the run header declares the A5 determinism mode (ADR-013, 2).

    Reports the fact (is the flag in this process's ``XLA_FLAGS``) and where it
    came from, so a verdict never claims a determinism mode it was not produced
    under.
    """
    if not determinism_pinned():
        return "off (local profile: not pinned)"
    if gate_profile():
        return "on (pinned by the gate profile)"
    return f"on (inherited from the caller's {XLA_FLAGS_ENV}, not pinned here)"


def session_banner() -> list[str]:
    """Run header: precision policy, devices, profile (ADR-010, decision 2)."""
    mode = backend_mode()
    devices = describe_devices()
    gpus = gpu_devices()
    if mode == "cpu":
        backend = "cpu (selected explicitly)"
    elif gpus:
        backend = "gpu"
    else:
        backend = "cpu (no CUDA device available)"
    profile = "gate" if gate_profile() else "local"
    lines = [
        "[ADR-010] policy: jax_default_matmul_precision="
        f"{jax.config.jax_default_matmul_precision}",
        f"[ADR-010] backend: {backend} ({BACKEND_ENV}={mode}) "
        f"jax.devices()={devices}",
        f"[ADR-010] profile: {profile} "
        f"({GATE_ENV}={os.environ.get(GATE_ENV, '<unset>')})",
        f"[ADR-013] determinism: {DETERMINISM_FLAG}={determinism_mode()} "
        f"{XLA_FLAGS_ENV}={xla_flags_value() or '<unset>'!r}",
    ]
    if not gate_profile() and not gpus:
        lines.append(
            "[ADR-010] preliminary run: no CUDA device and the local profile "
            "does not require one (the A4 gate profile does)"
        )
    return lines


apply_precision_pinning()


# --------------------------------------------------------------------------
# Run header + gate/local profile switch
# --------------------------------------------------------------------------


def _write_line(config, line: str) -> None:
    reporter = config.pluginmanager.getplugin("terminalreporter")
    if reporter is not None and hasattr(reporter, "write_line"):
        reporter.write_line(line)
    else:  # pragma: no cover — no terminal plugin (e.g. -p no:terminal)
        print(line)


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--net-gate",
        action="store_true",
        default=False,
        help="run the net acceptance suite in the A4 gate profile (ADR-010): "
        "a missing CUDA device is a FAIL, not a SKIP",
    )


def pytest_configure(config: pytest.Config) -> None:
    # The CLI flag and the environment variable resolve to the same marker, so
    # tests (and the guard) see one source of truth.
    if config.getoption("--net-gate"):
        os.environ[GATE_ENV] = "1"


def pytest_sessionstart(session: pytest.Session) -> None:
    for line in session_banner():
        _write_line(session.config, line)


# --------------------------------------------------------------------------
# Model configs
# --------------------------------------------------------------------------

FULL_CONFIG = ModelConfig()


def small_config(**overrides) -> ModelConfig:
    base = dict(
        vocab_size=512,
        hidden=64,
        num_layers=4,  # 3 KDA + 1 MLA
        num_kda_layers=3,
        num_mla_layers=1,
        num_heads=4,
        head_dim=16,
        kda_dk=16,
        kda_dv=16,
        kda_decay_rank=16,
        kda_short_conv_kernel=4,
        kda_g_min=-5.0,
        mla_latent_dim=32,
        mla_head_dim=16,
        mlp_intermediate=128,
        siti_beta_gate=4.0,
        siti_beta_up=25.0,
        mtp_layers=1,
        mtp_loss_weight=0.1,
        attnres_blocks=1,
        attnres_block_size=4,
        vit_patch=14,
        vit_hidden=32,
        vit_depth=2,
        vit_heads=2,
        vit_mlp=64,
        image_size=56,
        moe_dense_layers=1,
        moe_latent_dim=32,
        moe_num_routed=6,
        moe_num_shared=2,
        moe_top_k=2,
        moe_expert_intermediate=16,
        moe_shared_intermediate=32,
        qb_weight=0.01,
        routing_seed=0,
    )
    base.update(overrides)
    return ModelConfig(**base)


def tiny_config(**overrides) -> ModelConfig:
    """Even smaller config for the overfit / needle-recall smokes.

    Invariant: ``num_heads * head_dim == hidden`` (the KDA output gate requires
    ``H * dv == hidden``), and ``kda_dk == kda_dv == mla_head_dim == head_dim``.
    """
    return small_config(vocab_size=64, hidden=16, num_heads=2, head_dim=8,
                        kda_dk=8, kda_dv=8, kda_decay_rank=8, mla_latent_dim=16,
                        mla_head_dim=8, mlp_intermediate=32,
                        moe_latent_dim=8, moe_num_routed=4, moe_top_k=2,
                        moe_num_shared=1, moe_expert_intermediate=16,
                        moe_shared_intermediate=32, **overrides)


@pytest.fixture
def cfg() -> ModelConfig:
    return small_config()


@pytest.fixture
def tiny_cfg() -> ModelConfig:
    return tiny_config()
