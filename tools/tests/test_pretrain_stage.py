"""CLI стадии претрейна: ``--stage`` / ``--lr`` / блок ``policy`` журнала (ADR-055).

Политика мидтрейна (ADR-055) исполняема, а не только объявлена: стадия задаётся
флагом, дефолтный LR следует стадии (midtrain — 1e-3, pretrain — прежний 1e-2),
явный ``--lr`` сильнее дефолта, а журнал несёт стадию из argv и блок ``policy``,
по которому приёмка отличает мидтрейн от претрейна.

Тесты файловые и CPU-пиннутые (``NET_JAX_BACKEND=cpu`` ДО первого импорта jax).
"""

from __future__ import annotations

import json
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


def _cli():
    import pretrain_run

    return pretrain_run


# ---------------------------------------------------------------------------
# --stage и резолвинг --lr
# ---------------------------------------------------------------------------


def test_default_stage_is_pretrain_with_the_legacy_lr():
    """Без флагов — прежнее поведение: stage pretrain, LR 1e-2 (побитово прежний)."""
    args = _cli().parse_args([])
    assert args.stage == "pretrain"
    assert args.lr == 1e-2


def test_midtrain_stage_defaults_to_the_adr055_lr():
    """``--stage midtrain`` без ``--lr`` даёт LR политики мидтрейна (1e-3)."""
    args = _cli().parse_args(["--stage", "midtrain"])
    assert args.stage == "midtrain"
    assert args.lr == 1e-3


def test_explicit_lr_beats_the_midtrain_default():
    """Явно переданный ``--lr`` всегда сильнее дефолта стадии."""
    args = _cli().parse_args(["--stage", "midtrain", "--lr", "2e-3"])
    assert args.lr == 2e-3


def test_explicit_lr_beats_the_pretrain_default():
    """И на стадии pretrain явный LR не подменяется дефолтом."""
    args = _cli().parse_args(["--stage", "pretrain", "--lr", "3e-3"])
    assert args.lr == 3e-3


def test_unknown_stage_is_rejected(capsys):
    """Неизвестная стадия — SystemExit от argparse, а не молчаливый дефолт."""
    with pytest.raises(SystemExit):
        _cli().parse_args(["--stage", "sft"])
    # Отказ именно по значению стадии (choices), а не «флаг неизвестен»:
    # сообщение обязано назвать допустимые стадии.
    err = capsys.readouterr().err
    assert "invalid choice" in err and "midtrain" in err


def test_stage_flag_is_visible_in_help(capsys):
    """``--stage`` виден в ``--help`` (контракт CLI, а не скрытая ручка)."""
    with pytest.raises(SystemExit):
        _cli().parse_args(["--help"])
    assert "--stage" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# stage_journal_fields — чистая функция полей журнала
# ---------------------------------------------------------------------------


def test_stage_journal_fields_midtrain():
    """Мидтрейн: политика ADR-055 — holdout-отбор и обязательный leak-check."""
    fields = _cli().stage_journal_fields(_cli().parse_args(["--stage", "midtrain"]))
    assert fields["stage"] == "midtrain"
    policy = fields["policy"]
    assert policy["name"] == "ADR-055"
    assert policy["checkpoint_selection"] == "holdout"
    assert policy["leak_check_required"] is True
    assert policy["lr"] == 1e-3


def test_stage_journal_fields_pretrain():
    """Претрейн: политики нет (name null), отбор по train, leak-check не нужен."""
    fields = _cli().stage_journal_fields(_cli().parse_args([]))
    assert fields["stage"] == "pretrain"
    policy = fields["policy"]
    assert policy["name"] is None
    assert policy["checkpoint_selection"] == "train"
    assert policy["leak_check_required"] is False
    assert policy["lr"] == 1e-2


def test_stage_journal_fields_carries_the_resolved_lr():
    """В блок policy едет фактический LR прогона, включая явное перекрытие."""
    args = _cli().parse_args(["--stage", "midtrain", "--lr", "5e-4"])
    assert _cli().stage_journal_fields(args)["policy"]["lr"] == 5e-4


@pytest.mark.parametrize(
    ("argv", "expected"),
    [
        (
            [],
            {
                "stage": "pretrain",
                "policy": {
                    "name": None,
                    "checkpoint_selection": "train",
                    "leak_check_required": False,
                    "lr": 1e-2,
                },
            },
        ),
        (
            ["--stage", "midtrain"],
            {
                "stage": "midtrain",
                "policy": {
                    "name": "ADR-055",
                    "checkpoint_selection": "holdout",
                    "leak_check_required": True,
                    "lr": 1e-3,
                },
            },
        ),
    ],
)
def test_stage_journal_block_is_exactly_the_contract(argv, expected):
    """Блок policy — ровно контракт ADR-055: без лишних и без пропущенных ключей."""
    cli = _cli()
    assert cli.stage_journal_fields(cli.parse_args(argv)) == expected


# ---------------------------------------------------------------------------
# Проводка: журнал прогона несёт stage из argv и блок policy
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("stage_argv", "expected_stage", "expected_policy"),
    [
        (
            [],
            "pretrain",
            {
                "name": None,
                "checkpoint_selection": "train",
                "leak_check_required": False,
                "lr": 1e-2,
            },
        ),
        (
            ["--stage", "midtrain"],
            "midtrain",
            {
                "name": "ADR-055",
                "checkpoint_selection": "holdout",
                "leak_check_required": True,
                "lr": 1e-3,
            },
        ),
    ],
)
def test_execute_journal_carries_stage_and_policy(
    tmp_path: Path, monkeypatch, stage_argv, expected_stage, expected_policy
):
    """Отказ по смете пишется в журнал уже со stage=argv и блоком policy.

    Сметы нет — прогон не стартует (AD-8), но заголовок журнала собран раньше
    отказа: именно по нему приёмка отличает мидтрейн-прогон от претрейн-прогона.
    Претрейн-ветка заодно пинит прежние дефолты (stage/lr) побитово.

    Preflight стенда мокается только здесь: предмет теста — отказ по смете, а на
    стенде с живым vLLM ``gate_or_exit`` отказал бы по чужому compute-процессу
    раньше. Production-гейт остаётся fail-closed (ADR-041).
    """
    cli = _cli()
    monkeypatch.setattr(cli.jax_preflight, "gate_or_exit", lambda *a, **k: None)
    out_dir = tmp_path / "out"
    code = cli.main(
        [
            "--run-ref", f"нет-такой-сметы-{expected_stage}",
            *stage_argv,
            "--shard-root", str(tmp_path / "ds"),
            "--no-packed",
            "--out", str(out_dir),
            "--steps", "1",
            "--budget-file", str(tmp_path / "нет-сметы.json"),
        ]
    )
    assert code == 1
    journal = json.loads((out_dir / "journal.json").read_text(encoding="utf-8"))
    assert journal["stage"] == expected_stage
    assert journal["policy"] == expected_policy
