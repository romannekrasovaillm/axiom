"""``--legacy-optimizer-classification`` — ручка 50-шагового A/B классификации.

Долг ADR-048: эффект классификации параметров (embeddings / tied LM head / ViT ->
AdamW против прежней «любой ``ndim == 2`` -> Muon») обязан проверяться
50-шаговым сравнением кривой loss.  Раннер отдаёт архитектору ручку, а журнал —
метку режима, чтобы ноги A/B сшивались по артефактам, а не по памяти.

Что пиннится здесь (структурно, без GPU):

* дефолт — классификация ADR-048 (ручка молчит, поведение прежнее);
* ``--legacy-optimizer-classification`` включает прежнюю классификацию, и
  ``--legacy-muon-all-2d`` остаётся рабочим синонимом;
* метка ``optimizer_classification`` однозначно отображает флаг
  (``adr-048`` | ``legacy``);
* строка ``pretrain-metrics/v1`` несёт метку, поэтому журнал самодостаточен для
  A/B; jax-путь пропускается с причиной, если jax нет, а не падает.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pytest

TOOLS_DIR = Path(__file__).resolve().parents[1]
CASE_DIR = TOOLS_DIR.parent

# Пиннинг бэкенда ДО первого импорта jax (ADR-010): тест остаётся файловым.
os.environ.setdefault("NET_JAX_BACKEND", "cpu")

for _path in (str(CASE_DIR), str(TOOLS_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import pretrain_run  # noqa: E402


def _train_loop():
    """``net.train_loop`` под пином бэкенда тянет jax — без него тест пропускаем.

    Сам :mod:`net.train_loop` при импорте применяет пиннинг бэкенда (ADR-010) и
    с выставленным ``NET_JAX_BACKEND`` грузит ``net/tests/conftest.py`` -> jax.
    Поэтому импорт ленивый и за ``importorskip``: на машине без jax проверка
    честно пропускается с причиной, а не падает на сборе.
    """
    pytest.importorskip("jax", reason="net.train_loop применяет пин бэкенда и тянет jax")
    from net import train_loop as tl

    return tl


def test_flag_default_keeps_adr048_classification():
    """Без флага — действующая классификация ADR-048, а не legacy молча."""
    assert pretrain_run.parse_args([]).legacy_muon_all_2d is False


def test_new_flag_name_enables_legacy_classification():
    """Ручка долга ADR-048 включается под своим именем."""
    assert pretrain_run.parse_args(["--legacy-optimizer-classification"]).legacy_muon_all_2d is True


def test_old_flag_name_stays_a_working_alias():
    """Прежнее имя не ломается: то же поле, то же поведение."""
    assert pretrain_run.parse_args(["--legacy-muon-all-2d"]).legacy_muon_all_2d is True


def test_mode_label_maps_the_flag_one_to_one():
    """Метка журнала однозначна — именно её читает сшиватель A/B."""
    tl = _train_loop()
    assert tl.optimizer_classification(False) == "adr-048"
    assert tl.optimizer_classification(True) == "legacy"
    assert tl.CLASSIFICATION_ADR048 == "adr-048"
    assert tl.CLASSIFICATION_LEGACY == "legacy"


def _acceptance_conftest():
    """Численные смоук-конфиги приёмки сети (``net/tests/conftest.py``)."""
    path = CASE_DIR / "net" / "tests" / "conftest.py"
    spec = importlib.util.spec_from_file_location("net_acceptance_conftest", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _tiny_metrics(tmp_path: Path, *, legacy: bool) -> list[dict]:
    """Две строки ``pretrain-metrics/v1`` на крошечной модели скелета."""
    import numpy as np
    import jax.numpy as jnp
    import jax.random as jr

    tl = _train_loop()
    cfg = _acceptance_conftest().tiny_config()
    key = jr.PRNGKey(11)
    pool = [
        jr.randint(jr.fold_in(key, index), (16,), 0, cfg.vocab_size, dtype=jnp.int32)
        for index in range(2)
    ]
    metrics = tmp_path / f"metrics-{int(legacy)}.jsonl"
    tl.train(
        cfg,
        iter([np.asarray(sequence)[None, :] for sequence in pool]),
        train_config=tl.TrainConfig(
            steps=2, lr=1e-2, seed=7, metrics_path=metrics,
            legacy_muon_all_2d=legacy,
        ),
        budget=tl.Budget(run_ref="test", path=Path("/dev/null"), present=True),
    )
    return tl.MetricsWriter.read(metrics)


@pytest.mark.parametrize("legacy, expected", [(False, "adr-048"), (True, "legacy")])
def test_metrics_journal_carries_the_classification_label(tmp_path, legacy, expected):
    """Строка метрик несёт режим — A/B сшивается по журналу, без догадок."""
    tl = _train_loop()
    rows = _tiny_metrics(tmp_path, legacy=legacy)
    assert rows, "прогон не записал метрик"
    assert all(row.get("optimizer_classification") == expected for row in rows)
    # Поле опциональное: имя схемы строки не сменилось.
    assert all(row.get("schema") == tl.METRICS_SCHEMA for row in rows)
