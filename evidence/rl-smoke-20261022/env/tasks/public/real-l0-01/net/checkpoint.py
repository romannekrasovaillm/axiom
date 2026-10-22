"""Orbax checkpointing + content hash manifest.

MODEL-L3-SKELETON.md section 6.7: Orbax save/load with bitwise identity after
round-trip, and the checkpoint content hash written to a manifest.  The hash is
over the serialised leaf bytes in a deterministic (path-sorted) order so it is
stable across runs.

ADR-054 Am.13 (цикл-5) — формат мастера в схеме
-----------------------------------------------

Схема манифеста получает поле ``master_dtype`` (``fp32`` | ``bf16``): какой
dtype у **хранимого** дерева мастера.  Хеш чекпойнта от этого не меняется (он
всегда считается по хранимым байтам: ``master_tree_hash`` — хеш хранимого
дерева, а bf16-дерево хешируется своими 2-байтовыми листьями), но читатель по
полю знает, что именно он получит, и не угадывает по инструментам.

Читатель обязан работать с **обоими** форматами (infer, evidence-инструменты):
``load_checkpoint(...)`` восстанавливает байты как есть, а
:func:`checkpoint_master_dtype` (:func:`read_manifest_master_dtype`) сообщает
хранимый dtype; отсутствие поля = ``fp32`` (чекпойнты прежних ревизий не
отвергаются).  Явный разовый каст — :func:`cast_tree`:

* ``fp32``-чекпойнт в bf16-режиме — разовый каст (нулевой comp заводит
  оптимизатор, ``net/optimizer.adapt_state``);
* ``bf16``-чекпойнт в fp32-режиме — апкаст **с предупреждением**: продолжение
  fp32-ноги с bf16-весов — осознанное решение, а не деталь загрузки (мелкие
  обновления уже потеряны хранилищем и апкаст их не вернёт).
"""

from __future__ import annotations

import hashlib
import json
import warnings
from pathlib import Path

import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp

DEFAULT_MANIFEST = "manifest.json"

#: Значения поля ``master_dtype`` схемы чекпойнта.
MASTER_DTYPE_FP32 = "fp32"
MASTER_DTYPE_BF16 = "bf16"
MASTER_DTYPES = (MASTER_DTYPE_FP32, MASTER_DTYPE_BF16)


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


def infer_master_dtype(tree) -> str:
    """dtype хранимого дерева по его листьям: ``bf16``, если все листья bf16.

    Смешанное дерево (например, состояние оптимизатора: fp32-моменты + bf16-
    буфер) даёт ``fp32`` — «не все bf16», а не догадку; для состояния режим
    прогона передаёт вызывающий явно.
    """
    leaves = jax.tree_util.tree_leaves(tree)
    if leaves and all(jnp.asarray(leaf).dtype == jnp.bfloat16 for leaf in leaves):
        return MASTER_DTYPE_BF16
    return MASTER_DTYPE_FP32


def checkpoint_master_dtype(meta) -> str:
    """Хранимый dtype мастера из манифеста/meta: отсутствует — ``fp32``.

    Читатель не падает на чекпойнтах прежних ревизий (поля в них нет) и не
    угадывает: значение может прийти из ``net/checkpoint``-манифеста или из
    ``meta.json`` записи ``CheckpointManager``.
    """
    value = meta.get("master_dtype") if isinstance(meta, dict) else None
    if value is None:
        return MASTER_DTYPE_FP32
    value = str(value).strip().lower()
    if value not in MASTER_DTYPES:
        raise ValueError(
            f"master_dtype={value!r}: ожидается одно из {list(MASTER_DTYPES)}"
        )
    return value


def read_manifest_master_dtype(directory: str | Path) -> str:
    """dtype мастера по манифесту каталога чекпойнта (нет манифеста/поля — fp32)."""
    path = Path(directory) / DEFAULT_MANIFEST
    if not path.is_file():
        return MASTER_DTYPE_FP32
    try:
        return checkpoint_master_dtype(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError, ValueError):
        return MASTER_DTYPE_FP32


def dtype_of(master_dtype: str) -> jnp.dtype:
    """jnp-dtype по имени режима мастера."""
    if master_dtype == MASTER_DTYPE_BF16:
        return jnp.bfloat16
    if master_dtype == MASTER_DTYPE_FP32:
        return jnp.float32
    raise ValueError(f"master_dtype={master_dtype!r}: ожидается одно из {list(MASTER_DTYPES)}")


def cast_tree(tree, master_dtype: str, *, warn_on_upcast: bool = False):
    """Разовый каст дерева к dtype мастера: fp32→bf16 или bf16→fp32 (апкаст).

    Каст идемпотентен: дерево нужного dtype возвращается как есть (побитово, без
    копии и без предупреждения).  Апкаст bf16→fp32 предупреждает (при
    ``warn_on_upcast``): продолжать fp32-траекторию с bf16-весов — осознанное
    действие, а потерянные хранилищем мелкие обновления апкаст не возвращает.
    """
    target = dtype_of(master_dtype)
    leaves = jax.tree_util.tree_leaves(tree)
    if leaves and all(jnp.asarray(leaf).dtype == target for leaf in leaves):
        return tree
    if target == jnp.float32 and warn_on_upcast:
        warnings.warn(
            "resume: bf16-мастер апкастится в fp32 (bf16-хранилище уже потеряло "
            "мелкие обновления; comp-буфер не восстанавливается — эпоха bf16 "
            "продолжается как fp32-нога с тем же деревом)",
            stacklevel=2,
        )
    return jax.tree_util.tree_map(
        lambda leaf: jnp.asarray(leaf).astype(target), tree
    )


def save_checkpoint(params, directory: str | Path, manifest: str | Path | None = None,
                    *, master_dtype: str | None = None) -> str:
    """Save ``params`` with Orbax and write a manifest containing the hash.

    ``master_dtype`` — dtype хранимого дерева (схема Am.13); ``None`` —
    определить по листьям (:func:`infer_master_dtype`).  Возврат — контент-хеш.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    _checkpointer().save(str(directory), params, force=True)
    digest = tree_hash(params)
    if manifest is None:
        manifest = directory / DEFAULT_MANIFEST
    stored_dtype = (
        infer_master_dtype(params) if master_dtype is None
        else checkpoint_master_dtype({"master_dtype": master_dtype})
    )
    with open(manifest, "w", encoding="utf-8") as f:
        json.dump(
            {"checkpoint_hash": digest, "format": "orbax",
             "master_dtype": stored_dtype},
            f, indent=2,
        )
    return digest



def load_checkpoint(directory: str | Path, target=None, *, master_dtype: str | None = None):
    """Restore a checkpoint.  ``target`` supplies the pytree structure (NamedTuples,
    bools) so the restored tree matches the saved one; pass the original params
    tree to recover bitwise-identical leaves.

    ``master_dtype`` — явное требование к dtype возврата (Am.13): orbax
    восстанавливает **хранимые** dtype (на bf16-чекпойнте bf16-таргет не
    апкастит и наоборот — молча вернёт хранимое), поэтому смена режима делается
    разовым кастом поверх: ``load_checkpoint(dir, target=..., master_dtype="fp32")``
    вернёт fp32-дерево и предупредит, если апкастит (см. :func:`cast_tree`).
    Без аргумента поведение прежнее — байты как они лежат (оба формата).
    """
    if target is None:
        restored = _checkpointer().restore(str(directory))
    else:
        restored = _checkpointer().restore(str(directory), item=target)
    if master_dtype is None:
        return restored
    return cast_tree(restored, master_dtype, warn_on_upcast=True)


def read_manifest(manifest: str | Path) -> dict:
    with open(manifest, "r", encoding="utf-8") as f:
        return json.load(f)
