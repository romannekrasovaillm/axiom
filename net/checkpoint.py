"""Orbax checkpointing + content hash manifest.

MODEL-L3-SKELETON.md section 6.7: Orbax save/load with bitwise identity after
round-trip, and the checkpoint content hash written to a manifest.  The hash is
over the serialised leaf bytes in a deterministic (path-sorted) order so it is
stable across runs.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp

DEFAULT_MANIFEST = "manifest.json"


def _checkpointer() -> ocp.PyTreeCheckpointer:
    return ocp.PyTreeCheckpointer()


def tree_hash(params) -> str:
    """SHA-256 over the serialised leaf bytes, path-sorted for determinism."""
    leaves, treedef = jax.tree_util.tree_flatten_with_path(params)
    h = hashlib.sha256()
    for path, leaf in sorted(leaves, key=lambda pl: str(pl[0])):
        arr = jnp.asarray(leaf)
        h.update(str(path).encode("utf-8"))
        h.update(arr.astype(arr.dtype).tobytes())
    return h.hexdigest()


def save_checkpoint(params, directory: str | Path, manifest: str | Path | None = None) -> str:
    """Save ``params`` with Orbax and write a manifest containing the hash.

    Returns the content hash.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    _checkpointer().save(str(directory), params, force=True)
    digest = tree_hash(params)
    if manifest is None:
        manifest = directory / DEFAULT_MANIFEST
    with open(manifest, "w", encoding="utf-8") as f:
        json.dump({"checkpoint_hash": digest, "format": "orbax"}, f, indent=2)
    return digest


def load_checkpoint(directory: str | Path, target=None):
    """Restore a checkpoint.  ``target`` supplies the pytree structure (NamedTuples,
    bools) so the restored tree matches the saved one; pass the original params
    tree to recover bitwise-identical leaves.
    """
    if target is None:
        return _checkpointer().restore(str(directory))
    return _checkpointer().restore(str(directory), item=target)


def read_manifest(manifest: str | Path) -> dict:
    with open(manifest, "r", encoding="utf-8") as f:
        return json.load(f)
