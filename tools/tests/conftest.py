"""Коллекция tools/tests в окружении без ML-стека.

Шесть модулей сюиты импортируют ``net/...`` (а через него — ``jax``) на этапе
коллекции.  Без ``jax`` pytest обрывает весь прогон ошибками коллекции, и
быстрые стражи (в том числе ``tools/check_manifest_v2.py`` и
``tools/check_rl_precondition.py`` по ADR-058) не исполняются даже под ``-k``.
Здесь такие модули исключаются из коллекции ТОЛЬКО когда ``jax`` недоступен;
в окружении с ``jax`` поведение коллекции не меняется.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

#: Модули, импортирующие net/... → jax на этапе коллекции.
_JAX_ONLY_MODULES = frozenset({
    "test_bench_block_merge.py",
    "test_kda_phase_profile.py",
    "test_pretrain_loop.py",
    "test_sensors_replica_bytes.py",
    "test_sensors_runtime_config.py",
    "test_tokenizer.py",
})


def pytest_ignore_collect(collection_path: Path, config):  # noqa: ARG001
    if importlib.util.find_spec("jax") is not None:
        return None
    if Path(collection_path).name in _JAX_ONLY_MODULES:
        return True
    return None
