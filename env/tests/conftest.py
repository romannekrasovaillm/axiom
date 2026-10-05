"""Фикстуры тестов environment v1.

Бинарь arch-ml ищется в ENV_ARCH_ML_BIN → target/release/arch-ml репозитория →
PATH. Тесты verifier/calibrate пропускаются с явной причиной, если бинаря нет;
schemas/corruption/reward/manifest проходят всегда.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

CASE_DIR = Path(__file__).resolve().parents[2]
if str(CASE_DIR) not in sys.path:
    sys.path.insert(0, str(CASE_DIR))

from env.util import copy_case_snapshot  # noqa: E402
from env.verifier import arch_ml_available  # noqa: E402


def _find_bin() -> str | None:
    env = os.environ.get("ENV_ARCH_ML_BIN")
    if env and Path(env).is_file():
        return env
    repo_bin = CASE_DIR.parent.parent / "target" / "release" / "arch-ml"
    if repo_bin.is_file():
        return str(repo_bin)
    if arch_ml_available("arch-ml"):
        return "arch-ml"
    return None


ARCH_ML_BIN = _find_bin()


#: Файлы, добавленные посторонним коммитом ПОСЛЕ baseline задачи E-2.6
#: (ADR-030, коммит 9a38cdb): нарушают C-001/C-002 (секции ADR неканоничны) —
#: вне зоны задачи (`docs/`, `model/` запрещены исполнителю). Тестовый кейс
#: приводится к границам baseline; если файлы приведены к канону/удалены — no-op.
POST_BASELINE_CASE_FILES = (
    "docs/adr/ADR-030-hf-kanal-publikacii-artefaktov-axiom.md",
    "model/ADR-030-hf-publication-channel.md",
)


@pytest.fixture(scope="session")
def case_dir(tmp_path_factory) -> Path:
    """Кейс в границах baseline задачи: снапшот без пост-baseline файлов.

    Именно на этом состоянии достижим зелёный вердикт R-1'
    (fitness после фильтра ∧ spine ∧ trace) — состояние, верифицированное
    архитектором; файлы пост-baseline коммита ослабляли бы кейс вне зоны задачи.
    """
    dst = tmp_path_factory.mktemp("case-baseline") / "case"
    copy_case_snapshot(CASE_DIR, dst)
    for rel in POST_BASELINE_CASE_FILES:
        p = dst / rel
        if p.is_file():
            p.unlink()
    return dst


@pytest.fixture(scope="session")
def arch_ml():
    if ARCH_ML_BIN is None:
        pytest.skip("arch-ml бинарь недоступен (нет ENV_ARCH_ML_BIN/target/release/arch-ml)")
    return ARCH_ML_BIN


@pytest.fixture(scope="session")
def generated(tmp_path_factory, case_dir):
    """Генерирует полный набор один раз на сессию (20 публичных + 50 holdout)."""
    from env import generate  # ленивый импорт: generate требует pyyaml

    out = tmp_path_factory.mktemp("env-data")
    summary = generate.generate(case_dir, out, seed=0)
    return {"out": out, "case": case_dir, "summary": summary}


@pytest.fixture()
def clean_snapshot(tmp_path, case_dir) -> Path:
    """Чистый снапшот кейса (границы baseline) в свежем tmp-каталоге."""
    dst = tmp_path / "clean"
    copy_case_snapshot(case_dir, dst)
    return dst
