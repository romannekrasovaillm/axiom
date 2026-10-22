"""Тесты прибора ``tools/profile_step.py`` (профиль train-шага, ADR-047 п. 3).

Проверяется контракт прибора на **синтетическом трейсе** — без jax, без CUDA и
без сети.  Предмет тестов:

* **арифметика раскладки** — покрытие интервалов, отсутствие двойного счёта при
  вложенности, перекрытие категорий, gap как время ВНЕ операций;
* **границы ошибок (fail-closed)** — пустой/битый трейс без X-событий и трейс без
  device-операций дают ``ProfileError``, а не правдоподобные нули;
* **сверка** — внутреннее тождество (покрытие + gap == окно) и внешний порог 95 %
  против стенного времени шага; без метрик порог НЕ выдумывается;
* **пофазовые поля** — доля первого шага (компиляция) отдельным полем, границы
  шагов не выдумываются при несовпадении числа маркеров;
* **категоризация** — правила на фикстурах имён и происхождение (device/host);
* **дисциплина ADR-041** — маркер ``ensure_mem_fraction()`` и его вызов на
  прогонном пути до старта окна.

Модуль под тестом не импортирует jax на уровне модуля (jax тянется только внутри
прогонных функций), поэтому сьют проходит на машине без jax.
"""

from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

import pytest

TOOLS_DIR = Path(__file__).resolve().parents[1]
CASE_DIR = TOOLS_DIR.parent
for _path in (str(TOOLS_DIR), str(CASE_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import profile_step as ps  # noqa: E402


# ---------------------------------------------------------------------------
# фикстуры
# ---------------------------------------------------------------------------


def write_trace(directory: Path, events: list[dict], *, name: str = "trace.json") -> Path:
    """Каталог трейса с одним файлом ``traceEvents``."""
    directory.mkdir(parents=True, exist_ok=True)
    (directory / name).write_text(json.dumps({"traceEvents": events}), encoding="utf-8")
    return directory


def device_op(name: str, ts: int, dur: int, *, pid: int = 0, tid: int = 0, **extra) -> dict:
    """Device-операция: имя канонического вида HLO + дорожка устройства."""
    event = {"ph": "X", "name": name, "ts": ts, "dur": dur, "pid": pid, "tid": tid}
    event.update(extra)
    return event


def host_frame(name: str, ts: int, dur: int, *, pid: int = 1, tid: int = 1) -> dict:
    """Хостовый кадр (jit-диспетчер шага) — время вне раскладки device-операций."""
    return {"ph": "X", "name": name, "ts": ts, "dur": dur, "pid": pid, "tid": tid, "cat": "jit"}


def step_trace(steps: int = 2, *, scale_second: float = 0.85) -> list[dict]:
    """Правдоподобный макет шага: маркер шага + скан-тело + именованные операции.

    Второй шаг короче — на макете видно поле доли первого шага (компиляция).
    """
    events: list[dict] = []
    for index in range(steps):
        scale = 1.0 if index == 0 else scale_second
        base = index * 1_000_000
        events.append(host_frame("jit_loss_fn", base, int(900_000 * scale)))
        scan_start = base + 5_000
        events.append(device_op("while.0", scan_start, int(700_000 * scale)))
        cursor = scan_start + 1_000
        for layer in range(4):
            events.append(device_op(f"fusion.{layer}", cursor, int(3_000 * scale)))
            cursor += int(4_000 * scale)
        events.append(device_op("kda_cc_scores.1", cursor, int(60_000 * scale)))
        cursor += int(61_000 * scale)
        events.append(device_op("mla_flash.1", cursor, int(40_000 * scale)))
        cursor += int(41_000 * scale)
        events.append(device_op("moe_dispatch.1", cursor, int(50_000 * scale)))
        cursor += int(51_000 * scale)
        events.append(device_op("einsum.9", cursor, int(120_000 * scale)))
        events.append(device_op("cross_entropy.1", base + 750_000, int(80_000 * scale)))
        events.append(device_op("adam_apply.1", base + 840_000, int(60_000 * scale)))
    return events


def write_metrics(path: Path, records: list[dict]) -> Path:
    path.write_text(
        "\n".join(json.dumps(record) for record in records), encoding="utf-8"
    )
    return path


# ---------------------------------------------------------------------------
# категоризация и происхождение
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("einsum.42", "gemm_einsum"),
        ("dot.7", "gemm_einsum"),
        ("kda_cc_scores.1", "kda"),
        ("associative_scan.2", "kda"),
        ("shortconv.3", "kda"),
        ("mla_flash.1", "mla_attention"),
        ("flash_attn.9", "mla_attention"),
        ("moe_dispatch.1", "moe_ffn"),
        ("expert_gate.4", "moe_ffn"),
        ("cross_entropy.1", "lm_head"),
        ("logits.5", "lm_head"),
        ("adam_apply.1", "optimizer_apply"),
        ("scale_by_lr.2", "optimizer_apply"),
        ("reduce-scatter.0", "collective"),
        ("all-gather.1", "collective"),
        ("memcpy-d2d.1", "memcpy_memset"),
        ("memset.2", "memcpy_memset"),
        ("fusion.9", "elementwise"),
        ("totally-unknown-op.1", "other_ops"),
    ],
)
def test_category_rules_on_name_fixtures(name: str, expected: str) -> None:
    assert ps.classify_category(name)[0] == expected


def test_collective_wins_over_reduce_and_memcpy() -> None:
    """Порядок правил: ``reduce-scatter`` — коллектив, а не elementwise/reduce."""
    assert ps.classify_category("reduce-scatter.3")[0] == "collective"
    assert ps.classify_category("all-reduce.3")[0] == "collective"


def test_module_evidence_outranks_name_heuristic() -> None:
    """Структурное свидетельство (``args.hlo_module``) идёт раньше имени."""
    assert ps.classify_category("fusion.1", module="kda_block") == ("kda", "by_module")
    assert ps.classify_category("fusion.1", module="moe_expert") == ("moe_ffn", "by_module")


@pytest.mark.parametrize(
    ("event", "expected"),
    [
        ({"name": "train", "cat": "python_function"}, "host"),
        ({"name": "jit_loss_fn", "cat": "jit"}, "host"),
        ({"name": "cudaLaunchKernel", "cat": "cuda_runtime"}, "host"),
        ({"name": "fusion.12"}, "device"),
        ({"name": "opaque", "args": {"hlo_op": "dot"}}, "device"),
        ({"name": "kernel_x", "cat": "XLA Ops"}, "device"),
        ({"name": "jit_loss_fn"}, "unknown"),
    ],
)
def test_origin_classification(event: dict, expected: str) -> None:
    assert ps.classify_origin(event) == expected


def test_host_frames_do_not_enter_device_partition() -> None:
    """Хостовый кадр шага накрывает устройство, но операцией не считается."""
    events = ps.complete_events(
        [
            host_frame("jit_loss_fn", 0, 1000),
            device_op("einsum.1", 10, 100),
            device_op("fusion.2", 200, 100),
        ]
    )
    buckets = ps.split_by_origin(events)
    assert len(buckets["device"]) == 2
    assert len(buckets["host"]) == 1
    window = ps.trace_window(buckets["device"])
    assert window["window_seconds"] == pytest.approx(1e-6 * (300 - 10))


# ---------------------------------------------------------------------------
# арифметика раскладки
# ---------------------------------------------------------------------------


def test_complete_events_keeps_only_timed_x_events() -> None:
    events = ps.complete_events(
        [
            {"ph": "X", "name": "einsum.1", "ts": 0, "dur": 10},
            {"ph": "X", "name": "einsum.1", "ts": 10, "dur": 10},
            {"ph": "B", "name": "einsum.1", "ts": 0},
            {"ph": "M", "name": "meta", "ts": 0},
            {"ph": "X", "name": "negative.1", "ts": 0, "dur": -5},
            {"ph": "X", "name": "no-ts.1", "dur": 5},
            {"ph": "X", "name": "bool-dur.1", "ts": 0, "dur": True},
        ]
    )
    assert [event["name"] for event in events] == ["einsum.1", "einsum.1"]
    assert ps.trace_window(events)["window_seconds"] == pytest.approx(20e-6)


def test_nesting_is_not_double_counted() -> None:
    events = ps.complete_events(
        [
            device_op("while.0", 0, 100),
            device_op("einsum.1", 10, 20),
            device_op("reduce.2", 40, 10),
        ]
    )
    leaves = ps.leaf_events(events)
    assert len(leaves) == 2
    gap = ps.compute_gap(events, leaves=leaves)
    assert gap["covered_seconds"] == pytest.approx(30e-6)
    assert gap["gap_seconds"] == pytest.approx(70e-6)
    assert gap["gap_share"] == pytest.approx(0.7)


def test_container_is_reported_not_silently_dropped() -> None:
    events = ps.complete_events(
        [
            device_op("while.0", 0, 100),
            device_op("einsum.1", 10, 20),
            device_op("reduce.2", 40, 10),
        ]
    )
    summary = ps.container_summary(ps.mark_leaves(events))
    assert summary["container_events"] == 1
    assert summary["container_names"] == ["while.0"]
    assert summary["container_seconds"] == pytest.approx(100e-6)


def test_overlap_across_tracks_is_not_nesting() -> None:
    """Наложение интервалов разных дорожек — перекрытие, а не вложение.

    Без учёта дорожки крупная KDA-операция стала бы «контейнером» и молча
    исчезла бы из раскладки.
    """
    events = ps.complete_events(
        [
            device_op("kda_cc_scores.1", 0, 100, pid=1, tid=1),
            device_op("fusion.0", 10, 5, pid=2, tid=1),
            device_op("fusion.1", 40, 5, pid=2, tid=1),
        ]
    )
    leaves = ps.leaf_events(events)
    assert len(leaves) == 3
    categories = {row["category"]: row for row in ps.aggregate_categories(events, leaves)["categories"]}
    assert categories["kda"]["seconds"] == pytest.approx(100e-6)
    assert categories["elementwise"]["seconds"] == pytest.approx(10e-6)


def test_categories_partition_device_time() -> None:
    events = ps.complete_events(step_trace(steps=1))
    device = ps.split_by_origin(events)["device"]
    leaves = ps.leaf_events(device)
    categories = ps.aggregate_categories(device, leaves)
    gap = ps.compute_gap(device, leaves=leaves)
    # Категории не перекрываются → сумма равна покрытию, а покрытие + gap = окно.
    assert categories["overlap_seconds"] == pytest.approx(0.0, abs=1e-12)
    assert categories["categories_seconds"] == pytest.approx(categories["covered_seconds"])
    assert categories["covered_seconds"] + gap["gap_seconds"] == pytest.approx(
        gap["window_seconds"]
    )


def test_category_overlap_is_reported_separately() -> None:
    """Перекрытие категорий даёт сумму больше покрытия — и это видно полем."""
    events = ps.complete_events(
        [
            device_op("einsum.1", 0, 20),
            device_op("memcpy-d2d.1", 10, 15, pid=2),
        ]
    )
    leaves = ps.leaf_events(events)
    categories = ps.aggregate_categories(events, leaves)
    assert categories["covered_seconds"] == pytest.approx(25e-6)
    assert categories["categories_seconds"] == pytest.approx(35e-6)
    assert categories["overlap_seconds"] == pytest.approx(10e-6)


def test_duplicate_events_do_not_swallow_each_other() -> None:
    events = ps.complete_events([device_op("einsum.1", 0, 20), device_op("einsum.1", 0, 20)])
    assert len(ps.leaf_events(events)) == 2


def test_top_ops_are_leaves_not_containers() -> None:
    events = ps.complete_events(
        [
            device_op("while.0", 0, 1000),
            device_op("einsum.1", 10, 20),
            device_op("reduce.2", 40, 10),
        ]
    )
    ops = ps.aggregate_ops(ps.leaf_events(events))
    names = [row["name"] for row in ops["top_ops"]]
    assert "while.0" not in names
    assert names[0] == "einsum.1"
    assert ops["top_ops"][0]["total_seconds"] == pytest.approx(20e-6)


def test_structural_share_counts_module_evidence() -> None:
    events = ps.complete_events(
        [
            device_op("fusion.1", 0, 50, args={"hlo_module": "kda_block"}),
            device_op("fusion.2", 60, 50),
        ]
    )
    categories = ps.aggregate_categories(events, ps.leaf_events(events))
    assert categories["structural_share"] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# сверка и находки
# ---------------------------------------------------------------------------


def test_reconcile_internal_identity_and_external_threshold() -> None:
    fail = ps.reconcile(
        window_seconds=40e-6,
        categories_seconds=30e-6,
        covered_seconds=30e-6,
        overlap_seconds=0.0,
        gap_seconds=10e-6,
        wall_step_seconds=11e-6,
        steps=4,
    )
    assert fail["internal_ok"] is True
    assert fail["external_share"] == pytest.approx(40e-6 / 44e-6)
    assert fail["external_ok"] is False

    ok = ps.reconcile(
        window_seconds=40e-6,
        categories_seconds=30e-6,
        covered_seconds=30e-6,
        overlap_seconds=0.0,
        gap_seconds=10e-6,
        wall_step_seconds=10.5e-6,
        steps=4,
    )
    assert ok["external_ok"] is True


def test_reconcile_without_metrics_does_not_fabricate_threshold() -> None:
    result = ps.reconcile(
        window_seconds=40e-6,
        categories_seconds=30e-6,
        covered_seconds=30e-6,
        overlap_seconds=0.0,
        gap_seconds=10e-6,
        wall_step_seconds=None,
        steps=4,
    )
    assert result["basis"] == "none"
    assert result["external_share"] is None
    assert result["external_ok"] is False  # None → не «зелено по умолчанию»


def test_internal_identity_violation_is_critical() -> None:
    """Расхождение раскладки с окном — дефект прибора, а не свойство шага."""
    findings = ps.build_findings(
        gap={"gap_share": 0.1},
        categories={"structural_share": 0.9, "unclassified_share_of_covered": 0.0},
        reconciliation={
            "internal_ok": False,
            "internal_residual_seconds": 1e-3,
            "basis": "none",
            "external_ok": None,
            "external_share": None,
            "threshold": ps.RECONCILE_THRESHOLD,
        },
        step_windows={"resolved": True},
    )
    assert [finding["code"] for finding in findings] == ["partition_residual"]
    assert findings[0]["severity"] == "critical"


@pytest.mark.parametrize(
    ("overrides", "expected_code"),
    [
        ({"gap": {"gap_share": 0.8}}, "gap_dominates"),
        ({"categories": {"structural_share": 0.1, "unclassified_share_of_covered": 0.0}},
         "attribution_name_only"),
        ({"categories": {"structural_share": 0.9, "unclassified_share_of_covered": 0.3}},
         "unclassified_high"),
        ({"origins": {"unknown_share_of_window": 0.3}}, "unknown_origin_high"),
        ({"step_windows": {"resolved": False, "reason": "нет маркеров"}},
         "step_boundaries_unresolved"),
    ],
)
def test_findings_fire_on_their_triggers(overrides: dict, expected_code: str) -> None:
    payload = {
        "gap": {"gap_share": 0.1},
        "categories": {"structural_share": 0.9, "unclassified_share_of_covered": 0.0},
        "reconciliation": {"basis": "none", "internal_ok": True},
        "step_windows": {"resolved": True},
        "origins": {"unknown_share_of_window": 0.0},
    }
    payload.update(overrides)
    codes = [finding["code"] for finding in ps.build_findings(**payload)]
    assert expected_code in codes


def test_clean_profile_has_no_findings() -> None:
    findings = ps.build_findings(
        gap={"gap_share": 0.1},
        categories={"structural_share": 0.9, "unclassified_share_of_covered": 0.01},
        reconciliation={"basis": "none", "internal_ok": True},
        step_windows={"resolved": True},
        origins={"unknown_share_of_window": 0.0},
    )
    assert findings == []


def test_gap_share_boundary_is_exclusive() -> None:
    """Ровно 40 % — ещё не находка: порог строгий (``>``), иначе шум флейкует."""
    findings = ps.build_findings(
        gap={"gap_share": ps.GAP_FINDING_THRESHOLD},
        categories={"structural_share": 0.9, "unclassified_share_of_covered": 0.0},
        reconciliation={"basis": "none", "internal_ok": True},
        step_windows={"resolved": True},
        origins={"unknown_share_of_window": 0.0},
    )
    assert findings == []


# ---------------------------------------------------------------------------
# пошаговая раскладка
# ---------------------------------------------------------------------------


def test_detect_steps_resolves_windows_and_first_step_share() -> None:
    events = ps.complete_events(step_trace(steps=2))
    detected = ps.detect_steps(events, device_events=ps.split_by_origin(events)["device"], expected=2)
    assert detected["resolved"] is True
    assert len(detected["windows"]) == 2
    device_seconds = [row["device_seconds"] for row in detected["windows"]]
    assert device_seconds[0] > device_seconds[1]


def test_detect_steps_refuses_to_guess_on_marker_mismatch() -> None:
    events = ps.complete_events(step_trace(steps=2))
    detected = ps.detect_steps(events, device_events=ps.split_by_origin(events)["device"], expected=4)
    assert detected["resolved"] is False
    assert detected["matches"] == 2
    assert detected["windows"] == []


def test_detect_steps_without_markers_is_unresolved() -> None:
    events = ps.complete_events([device_op("einsum.1", 0, 10)])
    assert ps.detect_steps(events, expected=1)["resolved"] is False


def test_per_step_summary_reports_first_step_share_and_compile_share() -> None:
    events = ps.complete_events(step_trace(steps=2))
    detected = ps.detect_steps(events, device_events=ps.split_by_origin(events)["device"], expected=2)
    summary = ps.per_step_summary(
        detected,
        [{"step_seconds": 33.0}, {"step_seconds": 31.0}, {"step_seconds": 31.0}],
    )
    assert summary["wall_first_step_share"] == pytest.approx(33.0 / 95.0)
    assert summary["wall_median_step_seconds"] == pytest.approx(31.0)
    assert summary["compile_share"] == pytest.approx(2.0 / 31.0)
    # И device-времена шагов, и стенные — оба поля на месте, а не одно вместо другого.
    assert summary["device_first_step_share"] is not None


def test_per_step_summary_without_metrics_leaves_fields_empty() -> None:
    summary = ps.per_step_summary({"resolved": False, "windows": []}, None)
    assert summary["wall_step_seconds"] is None
    assert summary["wall_median_step_seconds"] is None
    assert summary["compile_share"] is None


def test_compile_share_is_not_negative_when_first_step_is_not_slower() -> None:
    summary = ps.per_step_summary(
        {"resolved": False, "windows": []},
        [{"step_seconds": 30.0}, {"step_seconds": 31.0}],
    )
    assert summary["compile_share"] == 0.0


# ---------------------------------------------------------------------------
# чтение трейса: fail-closed
# ---------------------------------------------------------------------------


def test_load_trace_events_picks_the_richest_source(tmp_path: Path) -> None:
    directory = tmp_path / "dup"
    write_trace(directory, [device_op("copy.1", 0, 1)], name="a.json")
    write_trace(directory, [device_op("einsum.1", 0, 2), device_op("reduce.1", 2, 3)], name="b.json")
    events, meta = ps.load_trace_events(directory)
    assert len(events) == 2
    assert meta["file_used"].endswith("b.json")
    assert len(meta["duplicate_trace_files"]) == 1


def test_load_trace_events_reads_gzip(tmp_path: Path) -> None:
    directory = tmp_path / "gz"
    directory.mkdir()
    with gzip.open(directory / "trace.json.gz", "wt", encoding="utf-8") as handle:
        json.dump({"traceEvents": [device_op("einsum.1", 0, 5)]}, handle)
    events, _ = ps.load_trace_events(directory)
    assert len(events) == 1


def test_load_trace_events_fail_closed_on_empty_and_broken(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    write_trace(empty, [])
    with pytest.raises(ps.ProfileError):
        ps.load_trace_events(empty)

    no_trace_events = tmp_path / "no-events"
    no_trace_events.mkdir()
    (no_trace_events / "summary.json").write_text(json.dumps({"other": 1}), encoding="utf-8")
    with pytest.raises(ps.ProfileError):
        ps.load_trace_events(no_trace_events)

    broken = tmp_path / "broken"
    broken.mkdir()
    (broken / "trace.json").write_text("{не json", encoding="utf-8")
    with pytest.raises(ps.ProfileError):
        ps.load_trace_events(broken)

    with pytest.raises(ps.ProfileError):
        ps.load_trace_events(tmp_path / "does-not-exist")


def test_analyze_trace_fail_closed_without_events_or_device_ops(tmp_path: Path) -> None:
    no_x = write_trace(tmp_path / "no-x", [{"ph": "M", "name": "meta"}])
    with pytest.raises(ps.ProfileError):
        ps.analyze_trace(no_x, expected_steps=1)

    host_only = write_trace(tmp_path / "host-only", [host_frame("jit_loss_fn", 0, 1000)])
    with pytest.raises(ps.ProfileError):
        ps.analyze_trace(host_only, expected_steps=1)


def test_analyze_trace_end_to_end_on_fixture(tmp_path: Path) -> None:
    directory = write_trace(tmp_path / "trace", step_trace(steps=2))
    metrics = write_metrics(
        tmp_path / "metrics.jsonl",
        [
            {"step": 1, "step_seconds": 33.0, "phase_profile": True, "sec_kda": 6.0},
            {"step": 2, "step_seconds": 31.0, "phase_profile": True, "sec_kda": 5.5},
        ],
    )
    analysis = ps.analyze_trace(directory, expected_steps=2)
    # 11 device-операций на шаг × 2 шага; маркеры шага — хостовые кадры.
    assert analysis["trace"]["device_event_count"] == 22
    assert analysis["origins"]["host_events"] == 2
    assert analysis["origins"]["unknown_events"] == 0
    assert analysis["containers"]["container_events"] == 2

    report = ps.build_report(
        cell=ps.cell_meta("net/config.json", 1, 8192, impl="chunked_cc", name="fixture", steps=2, warmup=2),
        status="COMPLETE",
        analysis=analysis,
        phase_legs=ps.phase_legs_from_metrics(metrics),
    )
    codes = {finding["code"] for finding in report["findings"]}
    assert "partition_residual" not in codes
    assert report["reconciliation"]["internal_ok"] is True
    assert {row["category"] for row in report["categories"]} >= {"kda", "mla_attention", "moe_ffn"}
    assert report["phase_legs"]["legs_seconds"]["sec_kda"] == pytest.approx(6.0)


# ---------------------------------------------------------------------------
# опорная нога из метрик
# ---------------------------------------------------------------------------


def test_phase_legs_read_metrics_and_count_broken_lines(tmp_path: Path) -> None:
    metrics = tmp_path / "metrics.jsonl"
    metrics.write_text(
        "\n".join(
            [
                json.dumps({"step": 1, "step_seconds": 33.0, "phase_profile": True,
                            "sec_kda": 6.0, "sec_mla": 2.0, "sec_moe": 9.0,
                            "sec_ce": 1.0, "sec_backopt": 8.0}),
                "не json",
                json.dumps({"step": 2, "phase_profile": True, "sec_kda": 5.5}),
            ]
        ),
        encoding="utf-8",
    )
    legs = ps.phase_legs_from_metrics(metrics)
    assert legs["available"] is True
    assert legs["profiled_records"] == 2
    assert legs["broken_lines"] == 1
    assert legs["legs_seconds"]["sec_kda"] == pytest.approx(6.0)
    assert "НЕ равна" in legs["caveat"]


def test_phase_legs_absent_is_not_an_error(tmp_path: Path) -> None:
    assert ps.phase_legs_from_metrics(tmp_path / "nope.jsonl")["available"] is False
    assert ps.phase_legs_from_metrics(None)["available"] is False


# ---------------------------------------------------------------------------
# отчёт и CLI
# ---------------------------------------------------------------------------


def test_empty_pending_report_carries_no_numbers() -> None:
    report = ps.build_report(
        cell=ps.cell_meta("net/config.json", 1, 8192, impl="chunked_cc", name="l3full", steps=4, warmup=2),
        status="EMPTY-PENDING",
    )
    assert report["gap"] is None
    assert report["categories"] == []
    assert report["top_ops"] == []
    assert report["steps"] is None
    assert report["reconciliation"] is None
    assert report["findings"] == []
    assert report["schema"] == ps.REPORT_SCHEMA
    assert any("эвристич" in caveat for caveat in report["caveats"])
    # Таблица не выводит выдуманных сходимостей/раскладок: чисел нет — строк нет.
    table = ps.format_table(report)
    assert "сходимость" not in table
    assert "структурных свидетельств" not in table
    assert "чисел нет" in table


def test_format_table_renders_categories_and_findings(tmp_path: Path) -> None:
    directory = write_trace(tmp_path / "trace", step_trace(steps=2))
    metrics = write_metrics(
        tmp_path / "metrics.jsonl",
        [{"step": 1, "step_seconds": 1900e-3, "phase_profile": True, "sec_kda": 6.0}],
    )
    analysis = ps.analyze_trace(directory, expected_steps=2)
    report = ps.build_report(
        cell=ps.cell_meta("net/config.json", 1, 8192, impl="chunked_cc", name="fixture", steps=2, warmup=2),
        status="COMPLETE",
        analysis=analysis,
        phase_legs=ps.phase_legs_from_metrics(metrics),
    )
    table = ps.format_table(report)
    assert "категория" in table
    assert "kda" in table
    assert "gap" in table
    assert "операция" in table
    assert "контейнеров" in table
    assert "опорные ноги" in table
    # Со стенным временем шага внешняя сходимость выводится — иначе порог не выдуман.
    assert "сходимость" in table
    assert report["reconciliation"]["basis"] == "wall_clock_steps"


def test_table_omits_reconciliation_line_without_metrics(tmp_path: Path) -> None:
    """Без стенного времени шага строка сходимости не печатается — порог не выдуман."""
    directory = write_trace(tmp_path / "trace", step_trace(steps=2))
    report = ps.build_report(
        cell=ps.cell_meta("net/config.json", 1, 8192, impl="chunked_cc", name="fixture", steps=2, warmup=2),
        status="COMPLETE",
        analysis=ps.analyze_trace(directory, expected_steps=2),
    )
    table = ps.format_table(report)
    assert "сходимость" not in table
    assert report["reconciliation"]["basis"] == "none"


def test_selftest_passes() -> None:
    assert ps.selftest() == 0


def test_cli_plan_lists_the_chunked_cc_cell(capsys: pytest.CaptureFixture[str]) -> None:
    assert ps.main(["--plan"]) == 0
    out = capsys.readouterr().out
    assert "l3full-chunked-cc" in out
    assert "chunked_cc" in out
    assert "--impl chunked_cc" in out


def test_cli_parse_trace_end_to_end(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    directory = write_trace(tmp_path / "trace", step_trace(steps=2))
    out_path = tmp_path / "step-profile.json"
    code = ps.main(
        ["--parse-trace", str(directory), "--steps", "2", "--name", "fixture", "--out", str(out_path)]
    )
    assert code == 0
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["status"] == "COMPLETE"
    assert payload["cell"]["impl"] == "chunked_cc"
    assert payload["reconciliation"]["internal_ok"] is True
    assert "категория" in capsys.readouterr().out


def test_cli_parse_trace_is_fail_closed(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    out_path = tmp_path / "report.json"
    code = ps.main(["--parse-trace", str(empty), "--out", str(out_path)])
    assert code == 2
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["status"] == "TRACE-ERROR"
    assert payload["categories"] == []
    assert payload["error"]


def test_cli_without_gpu_writes_empty_pending(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(ps, "gpu_available", lambda: False)
    out_path = tmp_path / "step-profile.json"
    code = ps.main(["--out", str(out_path), "--name", "l3full-chunked-cc"])
    assert code == 0
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["status"] == "EMPTY-PENDING"
    assert payload["categories"] == []
    assert payload["note"]


def test_cli_run_path_failure_is_fail_closed_not_a_traceback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Отказ прогона (нет jax/устройства/ошибка компиляции) → TRACE-ERROR, не стек."""
    monkeypatch.setattr(ps, "gpu_available", lambda: True)
    monkeypatch.setattr(ps, "_preflight_memory", lambda: None)

    def boom(*args, **kwargs):
        raise RuntimeError("нет jax на этой машине")

    monkeypatch.setattr(ps, "run_window", boom)
    out_path = tmp_path / "step-profile.json"
    code = ps.main(["--out", str(out_path), "--trace-dir", str(tmp_path / "trace")])
    assert code == 2
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["status"] == "TRACE-ERROR"
    assert "RuntimeError" in payload["error"]
    assert payload["categories"] == []


def test_cli_parses_step_marker_override(tmp_path: Path) -> None:
    events = [
        device_op("while.0", 0, 1000, cat="XLA Ops"),
        device_op("einsum.1", 10, 100, cat="XLA Ops"),
        device_op("einsum.1", 1010, 90, cat="XLA Ops"),
    ]
    for index, event in enumerate(events):
        event["name"] = f"{event['name']}" if index else event["name"]
    events.append({"ph": "X", "name": "my_own_step", "ts": 0, "dur": 1000, "pid": 9, "tid": 9, "cat": "jit"})
    events.append({"ph": "X", "name": "my_own_step", "ts": 1000, "dur": 900, "pid": 9, "tid": 9, "cat": "jit"})
    directory = write_trace(tmp_path / "trace", events)
    out_path = tmp_path / "report.json"
    code = ps.main(
        ["--parse-trace", str(directory), "--steps", "2",
         "--step-marker", "my_own_step", "--out", str(out_path)]
    )
    assert code == 0
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["steps"]["resolved"] is True
    assert len(payload["steps"]["windows"]) == 2


# ---------------------------------------------------------------------------
# дисциплина ADR-041
# ---------------------------------------------------------------------------


def test_adr041_preflight_marker_is_present_and_precedes_the_window() -> None:
    """ADR-041: лимит памяти XLA — маркер в файле и вызов ДО старта окна."""
    source = (TOOLS_DIR / "profile_step.py").read_text(encoding="utf-8")
    assert "jax_preflight.ensure_mem_fraction()" in source
    main_body = source[source.index("def main(") :]
    assert main_body.index("_preflight_memory()") < main_body.index("run_window(")
    # Лимит выставляется ДО import jax: в теле `_preflight_memory` импорта jax нет.
    preflight_body = source[source.index("def _preflight_memory") : source.index("def main(")]
    assert "import jax\n" not in preflight_body


def test_module_does_not_import_jax_at_module_scope() -> None:
    """Сьют и разбор трейса работают без jax: импорт — только внутри прогонных ног."""
    source = (TOOLS_DIR / "profile_step.py").read_text(encoding="utf-8")
    assert "\nimport jax\n" not in source
    assert "import jax" in source  # но внутри функций он есть — иначе прогон невозможен
