"""Тесты детектора changepoint кривой loss (C-058/C-059, ADR-060).

Проверяется контракт дельты ``changes/loss-changepoint-guard/DELTA.md``:
блочные границы и отбрасывание хвоста, схлопывание дублей шага, коды возврата
2 на битый вход (JSON/step/NaN/Inf/пустой набор/``sigma_b==0``), подавление
реестром и его граница допуска, слияние событий, схема и детерминизм
``--json``, разбор YAML-реестра обоими путями (PyYAML и встроенный парсер).
Живой прогон GB10 не требуется: все кейсы синтетические, сеть не используется.
"""

from __future__ import annotations

import json
import random
import statistics
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import check_loss_changepoint as clc  # noqa: E402  (путь добавляется выше)

REPO_ROOT = Path(__file__).resolve().parents[2]
REAL_LEDGER = REPO_ROOT / "evidence" / "monitoring" / "changepoints.yaml"


# ---------------------------------------------------------------------------
# Вспомогательные функции
# ---------------------------------------------------------------------------


def _write_rows(path: Path, rows: list[dict]) -> Path:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    return path


def _metric_row(step: int, loss: float) -> dict:
    return {"schema": clc.METRICS_SCHEMA, "step": step, "loss": loss}


def _write_ledger(path: Path, records: list[dict], *, header: bool = True) -> Path:
    """Собрать YAML-реестр; reason — блочная строка >- (как в живом реестре)."""

    lines: list[str] = []
    if header:
        lines.append("version: axiom-changepoints/1")
        lines.append("changepoints:")
    for record in records:
        prefix = "  - " if header else "- "
        lines.append(f"{prefix}step: {record['step']}")
        indent = "    " if header else "  "
        if "direction" in record:
            lines.append(f"{indent}direction: {record['direction']}")
        for name in ("owner", "ref", "review_by"):
            if name in record:
                lines.append(f"{indent}{name}: {json.dumps(record[name], ensure_ascii=False)}")
        if "reason" in record:
            lines.append(f"{indent}reason: >-")
            for chunk in str(record["reason"]).splitlines() or [""]:
                lines.append(f"{indent}  {chunk}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _valid_record(step: int, **overrides: object) -> dict:
    record = {
        "step": step,
        "direction": "up",
        "owner": "владелец прогона",
        "reason": "известный сдвиг, разобран",
        "ref": "runs/example/STATUS.md",
        "review_by": "2026-12-31",
    }
    record.update(overrides)
    return record


@pytest.fixture(scope="module")
def step_up_metrics(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """30 000 шагов, ступень +0.17 на шаге 10 000 — ровно одно событие.

    Ряд асимметричен (1/3 до сдвига, 2/3 после): при равных половинах
    глобальная MAD впитывает ступень и z сверху ограничен 2.70 — по
    определению ALGORITHM такая ступень недетектируема (см. selftest).
    """

    directory = tmp_path_factory.mktemp("metrics-step-up")
    rng = random.Random(7)
    rows = [
        _metric_row(
            step,
            5.0 + (0.17 if step >= 10_000 else 0.0) + rng.gauss(0.0, 0.25),
        )
        for step in range(30_000)
    ]
    return _write_rows(directory / "metrics-seg-1.jsonl", rows)


def _run_json(capsys: pytest.CaptureFixture[str], argv: list[str]):
    code = clc.main(argv)
    captured = capsys.readouterr()
    return code, json.loads(captured.out), captured.err


# ---------------------------------------------------------------------------
# Блочные границы и хвост
# ---------------------------------------------------------------------------


def test_block_boundaries_and_tail_dropped() -> None:
    series = [(step, float(step)) for step in range(52)]
    blocks = clc.build_blocks(series, clc.DEFAULT_BLOCK)

    assert len(blocks) == 2
    assert blocks[0].first_step == 0
    assert blocks[0].last_step == 24
    assert blocks[0].median_step == 12
    assert blocks[1].first_step == 25
    assert blocks[1].last_step == 49
    # Шаги 50–51 — хвост короче B=25: отброшены.
    assert blocks[-1].last_step == 49


def test_block_median_is_median_not_mean() -> None:
    losses = [((step * 13) % 29) for step in range(25)] + [1e9] * 27
    series = [(step, float(loss)) for step, loss in enumerate(losses)]
    blocks = clc.build_blocks(series, 25)

    assert len(blocks) == 2
    assert blocks[0].median_loss == float(statistics.median(losses[:25]))
    # 27 шагов второго блока — хвост, второй блок неполный и отброшен.
    assert [block.last_step for block in blocks] == [24, 49]


def test_no_full_block_yields_no_blocks() -> None:
    assert clc.build_blocks([(step, 1.0) for step in range(24)], 25) == []
    assert clc.build_blocks([], 25) == []


# ---------------------------------------------------------------------------
# Разрешение путей, сортировка сегментов, дедуп шагов и разрыв
# ---------------------------------------------------------------------------


def test_segment_numeric_sort_and_last_duplicate_wins(tmp_path: Path) -> None:
    directory = tmp_path / "metrics"
    directory.mkdir()
    _write_rows(directory / "metrics-seg-2.jsonl", [_metric_row(100, 1.0)])
    _write_rows(directory / "metrics-seg-10.jsonl", [_metric_row(100, 2.0)])
    _write_rows(
        directory / "metrics-seg-1.jsonl",
        [_metric_row(7, 1.0), _metric_row(7, 9.0)],  # дубль внутри файла
    )
    # Не метрики: не должны попасть в набор.
    _write_rows(directory / "journal-100.json", [_metric_row(100, 99.0)])
    _write_rows(directory / "metrics-seg-abc.jsonl", [_metric_row(100, 99.0)])

    files = clc.resolve_metrics_files([directory])
    assert [path.name for path in files] == [
        "metrics-seg-1.jsonl",
        "metrics-seg-2.jsonl",
        "metrics-seg-10.jsonl",
    ]

    loaded = clc.load_metrics([directory])
    values = dict(loaded.series)
    # Числовая сортировка: seg-10 прочитан последним, его loss побеждает.
    assert values[100] == 2.0
    # Внутри файла побеждает последнее вхождение дубля.
    assert values[7] == 9.0
    # Разрыв в нумерации шагов сохраняется как разрыв, а не ошибка.
    assert [step for step, _ in loaded.series] == [7, 100]
    assert loaded.rows_read == 4  # обе строки дубля 7 прочитаны


def test_duplicates_and_gap_do_not_create_event(tmp_path: Path) -> None:
    rng = random.Random(2)
    rows: list[dict] = []
    for step in range(20_000):
        if 7_000 <= step < 7_300:
            continue  # разрыв в нумерации шагов
        loss = 5.0 + rng.gauss(0.0, 0.25)
        rows.append(_metric_row(step, loss))
        if step % 500 == 0:
            rows.append(_metric_row(step, loss + 0.001))  # дубль шага
    _write_rows(tmp_path / "metrics-seg-1.jsonl", rows)

    loaded = clc.load_metrics([tmp_path])

    assert len(loaded.series) == 20_000 - 300
    assert clc.detect_changepoints(loaded.series).events == []


# ---------------------------------------------------------------------------
# Некорректный вход → exit 2
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("payload", "needle"),
    [
        ('{"schema": "pretrain-metrics/v1", "step": 0, "loss": NaN}\n', "NaN/Inf"),
        ('{"schema": "pretrain-metrics/v1", "step": 0, "loss": Infinity}\n', "NaN/Inf"),
        ('{"schema": "pretrain-metrics/v1", "step": "zero", "loss": 1.0}\n', "нечисловой step"),
        ('{"schema": "pretrain-metrics/v1", "step": 1.5, "loss": 1.0}\n', "нечисловой step"),
        ('{"schema": "pretrain-metrics/v1", "step": 0, "loss": "oops"}\n', "нечисловой loss"),
        ('{"schema": "pretrain-metrics/v1", "step": 0, "loss": 1.0\n', "битая JSON-строка"),
        ("[1, 2]\n", "JSON-объектом"),
    ],
)
def test_invalid_row_exit2(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], payload: str, needle: str
) -> None:
    path = tmp_path / "metrics-seg-1.jsonl"
    path.write_text(payload, encoding="utf-8")

    code = clc.main(["--metrics", str(path), "--quiet"])

    assert code == clc.EXIT_INPUT
    assert needle in capsys.readouterr().err


@pytest.mark.parametrize("payload", ["", "\n\n   \n"])
def test_empty_row_set_exit2(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], payload: str
) -> None:
    path = tmp_path / "metrics-seg-1.jsonl"
    path.write_text(payload, encoding="utf-8")

    code = clc.main(["--metrics", str(path), "--quiet"])

    assert code == clc.EXIT_INPUT
    assert "пустой набор строк" in capsys.readouterr().err


def test_sigma_zero_exit2(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = _write_rows(
        tmp_path / "metrics-seg-1.jsonl",
        [_metric_row(step, 4.0) for step in range(25 * 30)],
    )

    code = clc.main(["--metrics", str(path), "--quiet"])

    assert code == clc.EXIT_INPUT
    assert "sigma_b == 0" in capsys.readouterr().err


def test_invalid_params_exit2(capsys: pytest.CaptureFixture[str]) -> None:
    assert clc.main(["--metrics", "x", "--block", "0"]) == clc.EXIT_INPUT
    assert "--block" in capsys.readouterr().err
    assert clc.main(["--metrics", "x", "--persist", "0"]) == clc.EXIT_INPUT
    assert "--persist" in capsys.readouterr().err


def test_missing_metrics_argument_exit2(capsys: pytest.CaptureFixture[str]) -> None:
    assert clc.main([]) == clc.EXIT_INPUT
    assert "--metrics" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# Нет данных → neutral/exit 0 (C-059: отсутствие метрик не красный гейт)
# ---------------------------------------------------------------------------


def test_missing_path_is_neutral(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code, data, _ = _run_json(
        capsys, ["--metrics", str(tmp_path / "does-not-exist"), "--json"]
    )

    assert code == clc.EXIT_OK
    assert data["status"] == clc.STATUS_NEUTRAL
    assert data["rows"] == 0 and data["blocks"] == 0


def test_directory_without_metrics_files_is_neutral(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    directory = tmp_path / "empty"
    directory.mkdir()
    _write_rows(directory / "journal-1.json", [_metric_row(0, 1.0)])

    code, data, _ = _run_json(capsys, ["--metrics", str(directory), "--json"])

    assert code == clc.EXIT_OK
    assert data["status"] == clc.STATUS_NEUTRAL


# ---------------------------------------------------------------------------
# Подавление реестром: валидная запись, граница tol, недействительные записи
# ---------------------------------------------------------------------------


def _single_event(capsys, metrics_path: Path) -> dict:
    code, data, _ = _run_json(
        capsys, ["--metrics", str(metrics_path), "--json"]
    )
    assert code == clc.EXIT_ALERT
    assert len(data["events"]) == 1
    return data["events"][0]


def test_valid_ledger_suppresses_event(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], step_up_metrics: Path
) -> None:
    event = _single_event(capsys, step_up_metrics)
    ledger = _write_ledger(
        tmp_path / "changepoints.yaml", [_valid_record(event["step"])]
    )

    code, data, err = _run_json(
        capsys,
        ["--metrics", str(step_up_metrics), "--ledger", str(ledger), "--json"],
    )

    assert code == clc.EXIT_OK
    assert data["status"] == clc.STATUS_OK
    assert data["events"] == []
    assert len(data["suppressed"]) == 1
    assert data["suppressed"][0]["step"] == event["step"]
    assert data["suppressed"][0]["owner"] == "владелец прогона"
    assert err == ""


def test_tolerance_boundary_inclusive(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], step_up_metrics: Path
) -> None:
    event = _single_event(capsys, step_up_metrics)
    at_limit = _write_ledger(
        tmp_path / "at-limit.yaml",
        [_valid_record(event["step"] + clc.DEFAULT_TOL)],
    )
    over_limit = _write_ledger(
        tmp_path / "over-limit.yaml",
        [_valid_record(event["step"] + clc.DEFAULT_TOL + 1)],
    )

    code_at, data_at, _ = _run_json(
        capsys,
        ["--metrics", str(step_up_metrics), "--ledger", str(at_limit), "--json"],
    )
    code_over, data_over, _ = _run_json(
        capsys,
        ["--metrics", str(step_up_metrics), "--ledger", str(over_limit), "--json"],
    )

    assert code_at == clc.EXIT_OK and data_at["suppressed"]
    assert code_over == clc.EXIT_ALERT and data_over["suppressed"] == []


@pytest.mark.parametrize("missing", ["owner", "reason", "ref", "review_by"])
def test_ledger_without_required_field_does_not_suppress(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    step_up_metrics: Path,
    missing: str,
) -> None:
    event = _single_event(capsys, step_up_metrics)
    record = _valid_record(event["step"])
    record.pop(missing)
    ledger = _write_ledger(tmp_path / "changepoints.yaml", [record])

    code, data, err = _run_json(
        capsys,
        ["--metrics", str(step_up_metrics), "--ledger", str(ledger), "--json"],
    )

    assert code == clc.EXIT_ALERT
    assert data["suppressed"] == []
    assert "недействительна" in err
    assert missing in err


def test_ledger_without_direction_still_suppresses(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], step_up_metrics: Path
) -> None:
    """TASK перечисляет недействительными только owner/reason/ref/review_by.

    DELTA A6 требует ещё и direction; взято более точное операционное
    указание TASK: direction проверяется, если присутствует.
    """

    event = _single_event(capsys, step_up_metrics)
    record = _valid_record(event["step"])
    record.pop("direction")
    ledger = _write_ledger(tmp_path / "changepoints.yaml", [record])

    code, data, _ = _run_json(
        capsys,
        ["--metrics", str(step_up_metrics), "--ledger", str(ledger), "--json"],
    )

    assert code == clc.EXIT_OK
    assert len(data["suppressed"]) == 1


def test_ledger_bad_direction_is_invalid(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], step_up_metrics: Path
) -> None:
    event = _single_event(capsys, step_up_metrics)
    ledger = _write_ledger(
        tmp_path / "changepoints.yaml",
        [_valid_record(event["step"], direction="sideways")],
    )

    code, data, err = _run_json(
        capsys,
        ["--metrics", str(step_up_metrics), "--ledger", str(ledger), "--json"],
    )

    assert code == clc.EXIT_ALERT
    assert data["suppressed"] == []
    assert "direction" in err


def test_missing_ledger_file_exit2(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], step_up_metrics: Path
) -> None:
    code = clc.main(
        [
            "--metrics",
            str(step_up_metrics),
            "--ledger",
            str(tmp_path / "nope.yaml"),
            "--quiet",
        ]
    )

    assert code == clc.EXIT_INPUT
    assert "ledger не найден" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# Слияние событий
# ---------------------------------------------------------------------------


def _event(start_block: int, step: int) -> clc.Event:
    return clc.Event(
        step=step,
        direction="up",
        delta=0.2,
        z=4.0,
        blocks=3,
        start_block=start_block,
    )


def test_merge_keeps_earlier_event_within_three_blocks() -> None:
    merged = clc.merge_events([_event(10, 1_000), _event(12, 1_050)])

    assert len(merged) == 1
    assert merged[0].step == 1_000


def test_merge_keeps_events_three_blocks_apart() -> None:
    merged = clc.merge_events([_event(10, 1_000), _event(13, 1_075)])

    assert [event.step for event in merged] == [1_000, 1_075]


def test_detection_merges_close_events_end_to_end() -> None:
    # Порог обнулён и persist=1: событий много, знак delta часто меняется.
    # После слияния начала событий обязаны отстоять не менее чем на 3 блока.
    params = clc.Params(persist=1, min_abs=0.0, z_threshold=0.0)
    analysis = clc.detect_changepoints(
        clc._series_stationary(2_000, 0.25, seed=2), params
    )

    assert analysis.events
    starts = [event.start_block for event in analysis.events]
    gaps = [right - left for left, right in zip(starts, starts[1:])]
    assert all(gap >= clc.MERGE_DISTANCE_BLOCKS for gap in gaps)


# ---------------------------------------------------------------------------
# --json: схема, отсутствие лишних строк, детерминизм
# ---------------------------------------------------------------------------


def test_json_schema_and_determinism(
    capsys: pytest.CaptureFixture[str], step_up_metrics: Path
) -> None:
    argv = ["--metrics", str(step_up_metrics), "--json"]

    code_first = clc.main(argv)
    first = capsys.readouterr().out
    code_second = clc.main(argv)
    second = capsys.readouterr().out

    assert code_first == code_second == clc.EXIT_ALERT
    assert first == second  # байт-в-байт
    assert first.count("\n") == 1  # ровно один JSON-объект, без прочих строк

    data = json.loads(first)
    assert sorted(data) == [
        "blocks",
        "events",
        "params",
        "rows",
        "status",
        "suppressed",
    ]
    assert data["status"] == clc.STATUS_ALERT
    assert data["rows"] == 30_000
    assert data["blocks"] == 1_200
    assert data["suppressed"] == []
    params = data["params"]
    assert {
        key: params[key]
        for key in (
            "block",
            "recent",
            "reference",
            "gap",
            "min_abs",
            "z_threshold",
            "persist",
            "tol",
        )
    } == {
        "block": 25,
        "recent": 6,
        "reference": 12,
        "gap": 2,
        "min_abs": 0.1,
        "z_threshold": 3.5,
        "persist": 3,
        "tol": 400,
    }
    assert isinstance(params["sigma_b"], float)
    assert sorted(data["events"][0]) == ["blocks", "delta", "direction", "step", "z"]
    assert data["events"][0]["direction"] == "up"


def test_json_has_no_extra_lines_on_neutral(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = clc.main(["--metrics", str(tmp_path / "nope"), "--json"])
    out = capsys.readouterr().out

    assert code == clc.EXIT_OK
    assert out.count("\n") == 1
    assert json.loads(out)["status"] == clc.STATUS_NEUTRAL


# ---------------------------------------------------------------------------
# --quiet
# ---------------------------------------------------------------------------


def test_quiet_alert_prints_single_summary_line(
    capsys: pytest.CaptureFixture[str], step_up_metrics: Path
) -> None:
    code = clc.main(["--metrics", str(step_up_metrics), "--quiet"])
    out = capsys.readouterr().out

    assert code == clc.EXIT_ALERT
    assert out == "НЕПОДАВЛЕННЫХ CHANGEPOINT: 1\n"


def test_quiet_clean_non_tty_prints_nothing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], step_up_metrics: Path
) -> None:
    event = _single_event(capsys, step_up_metrics)
    ledger = _write_ledger(
        tmp_path / "changepoints.yaml", [_valid_record(event["step"])]
    )

    code = clc.main(
        ["--metrics", str(step_up_metrics), "--ledger", str(ledger), "--quiet"]
    )

    assert code == clc.EXIT_OK
    assert capsys.readouterr().out == ""


# ---------------------------------------------------------------------------
# YAML-реестр: оба парсера, блочные строки, недействительные записи
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not REAL_LEDGER.is_file(), reason="живой реестр отсутствует")
@pytest.mark.parametrize("prefer_pyyaml", [False, True])
def test_real_ledger_parses_in_both_modes(prefer_pyyaml: bool) -> None:
    if prefer_pyyaml and not clc.HAVE_PYYAML:
        pytest.skip("PyYAML недоступен")
    text = REAL_LEDGER.read_text(encoding="utf-8")

    ledger = clc.parse_ledger_text(text, prefer_pyyaml=prefer_pyyaml)

    assert ledger.invalid == []
    assert [record.step for record in ledger.records] == [14501, 28436, 42375]
    assert all(record.direction == "up" for record in ledger.records)
    assert all(
        record.owner and record.reason and record.ref and record.review_by
        for record in ledger.records
    )
    # Блочная строка reason развёрнута, комментарии не приклеились.
    assert "Исторический сдвиг" in ledger.records[0].reason


@pytest.mark.parametrize("prefer_pyyaml", [False, True])
def test_minimal_ledger_subset(prefer_pyyaml: bool) -> None:
    if prefer_pyyaml and not clc.HAVE_PYYAML:
        pytest.skip("PyYAML недоступен")
    text = """
# реестр известных сдвигов
version: test/1
defaults:
  tol: 400
changepoints:
  - step: 100  # граница допуска
    direction: down
    owner: "тест"
    reason: >-
      Причина в две
      строки без переноса.
    ref: "runs/example/STATUS.md"
    review_by: "2026-12-31"
"""
    ledger = clc.parse_ledger_text(text, prefer_pyyaml=prefer_pyyaml)

    assert ledger.invalid == []
    (record,) = ledger.records
    assert record.step == 100
    assert record.direction == "down"
    assert record.owner == "тест"
    assert record.reason == "Причина в две строки без переноса."
    assert record.ref == "runs/example/STATUS.md"


@pytest.mark.parametrize("prefer_pyyaml", [False, True])
def test_bare_list_ledger(prefer_pyyaml: bool) -> None:
    if prefer_pyyaml and not clc.HAVE_PYYAML:
        pytest.skip("PyYAML недоступен")
    text = (
        "- step: 5\n"
        "  direction: up\n"
        "  owner: a\n"
        "  reason: b\n"
        "  ref: c\n"
        "  review_by: d\n"
    )
    ledger = clc.parse_ledger_text(text, prefer_pyyaml=prefer_pyyaml)

    assert [(record.step, record.owner) for record in ledger.records] == [(5, "a")]


def test_invalid_ledger_entries_are_reported() -> None:
    text = (
        "changepoints:\n"
        "  - step: 10\n"
        "    direction: up\n"
        "    reason: без владельца\n"
        "    ref: x\n"
        "    review_by: y\n"
        "  - step: not-a-number\n"
        "    owner: a\n"
        "    reason: b\n"
        "    ref: c\n"
        "    review_by: d\n"
    )
    ledger = clc.parse_ledger_text(text, prefer_pyyaml=False)

    assert ledger.records == []
    assert [item.index for item in ledger.invalid] == [0, 1]
    assert "owner" in ledger.invalid[0].reason
    assert "step" in ledger.invalid[1].reason


def test_broken_ledger_yaml_exit2(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], step_up_metrics: Path
) -> None:
    ledger = tmp_path / "broken.yaml"
    ledger.write_text("changepoints: [unclosed\n", encoding="utf-8")

    code = clc.main(
        ["--metrics", str(step_up_metrics), "--ledger", str(ledger), "--quiet"]
    )

    assert code == clc.EXIT_INPUT
    assert "ledger" in capsys.readouterr().err.lower()


# ---------------------------------------------------------------------------
# Selftest (C-058) и производительность (T14)
# ---------------------------------------------------------------------------


def test_selftest_green(capsys: pytest.CaptureFixture[str]) -> None:
    code = clc.main(["--selftest"])
    out = capsys.readouterr().out

    assert code == clc.EXIT_OK
    assert "SELFTEST: PASS" in out
    assert out.count("PASS ") >= 9


def test_30k_rows_under_60_seconds(
    capsys: pytest.CaptureFixture[str], step_up_metrics: Path
) -> None:
    started = time.monotonic()
    code = clc.main(["--metrics", str(step_up_metrics), "--quiet"])
    elapsed = time.monotonic() - started
    capsys.readouterr()

    assert code in (clc.EXIT_OK, clc.EXIT_ALERT)
    assert elapsed < 60.0
