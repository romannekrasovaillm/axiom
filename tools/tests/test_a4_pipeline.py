"""Контрактные тесты оркестратора прогона A4 (tools/run_a4_pipeline.py).

Источник истины — спека `docs/specs/A4-RUN.delta.md` (§5.2 CLI-контракт
оркестратора, §4 п. 6–7 портируемость путей, §6 критерии приёмки K3, K8 и
K12), а не реализация. Код оркестратора написан другим исполнителем (узел
n2); расхождения кода со спекой здесь не подгоняются, а фиксируются
провалом теста.

Сценарии (все прогоны — CLI подпроцессом, каталог evidence/ кейса не
затрагивается — манифест направляется в tmp через --manifest-out).

Фикстуры двух видов (§5.2 + §4 п. 6–7: `--out` обязан лежать внутри
репозитория — иначе относительного пути для run_ref в манифесте не
существует, и оркестратор отклоняет прогон до исполнения стадий):

* каталоги артефактов прогона (`--out`) — во временных каталогах ВНУТРИ
  репозитория (tools/tests/.a4-*-out-*, удаляются после тестов). До дельты
  K11/K12 они лежали в tmp_path — после введения охраны «--out внутри
  репозитория» фикстуры переведены внутрь репо (сценарии и утверждения
  тестов не изменены);
* манифесты (`--manifest-out`) — в tmp_path: этот путь в манифест не
  попадает, ограничение на него не распространяется.

* K3 — wire-прогон с частичным покрытием: в --out непустой журнал прогона
  (путь к нему виден полем `run_journal` манифеста), манифест создан,
  `pipeline_complete=false`, `--verify` даёт код 1 и поимённый список
  непокрытых стадий (список выводится из самого манифеста: после дельты
  «статус из следа» `sft`/`rl_base_scheme` могут быть `executed` — их следы
  исполнены на стенде и лежат в evidence/a4-run-wire/, — а `spark_inference`
  непокрыта в прогоне этой сюиты: сюита идёт на CPU (`JAX_PLATFORMS=cpu`),
  `device_kind` журнала — не стенд gb10/DGX Spark, значит стадия обязана
  остаться `skipped`; стендовый прогон эта сюита не исполняет). `run_ref`
  манифеста — slug без «/» (C-041): имя прогона, по которому страж ищет смету;
* K8 — детерминизм: два прогона с одним --seed дают одинаковые
  `model_weights_sha256` и вердикты среды;
* §5.2 — stdout содержит JSON-сводку {run_ref, model_weights_sha256,
  dataset_sha256, stages, pipeline_complete}; в CPU-прогоне стадия
  spark_inference НЕ имеет статуса executed (журнал несёт `device_kind=cpu`,
  а не стенд gb10/DGX Spark: подмена стенда запрещена); код 0 при частичном
  покрытии. Разбор «стенд / не стенд» по журналу — T-stand1–T-stand3;
* K12 — сквозная портируемость: прогон из НЕ-корневого cwd (tmp-каталог) с
  минимальными `--steps 1 --tasks 1` и относительным `--out` внутри
  репозитория -> код 0; в манифесте рекурсивно нет ни одной строки,
  начинающейся с '/', а `run_journal` резолвится от корня репозитория (не от
  cwd запуска), существует и непуст; `run_ref` — slug без «/». Каталог
  артефактов прогона (tools/tests/.a4-k12-out-*) удаляется после модуля.

  Разрешение прежнего расхождения (дельта C-041, спека §4 п. 5): путь журнала
  живёт в `run_journal`, `run_ref` — имя прогона; барьер существования журнала
  сохранён, но проверяется по `run_journal`, а не по `run_ref`.

* C-041 — идентификатор прогона: run_ref по умолчанию — slug из последней
  компоненты `--out`, при невозможности `a4-run-<хеш>`; вызов генератора
  передаёт его отдельным `--run-id`, поэтому в манифесте run_ref — имя, а путь
  журнала виден полем `run_journal` (T-r1, T-r2);
* §2 — статус `failed` из следа стадии эмитится как есть, в `absent` не
  превращается (T4).

Бюджет: прогонов оркестратора ровно три на всю сюиту — общий модульный
фикстурный прогон (K3 + §5.2), один повторный для K8 и один прогон K12 из
чужого cwd. Прогоны минимальные (--steps 1, --tasks 1–2). Ручные замеры на
этой машине (CPU, JAX_PLATFORMS=cpu): 2026-09-13 узел n4 — 169,6 с за прогон;
2026-09-13 узел n3 — 122,9 с (`--seed 0 --steps 1 --tasks 1` из чужого cwd,
exit 0). Одиночный прогон > 120 с, поэтому сюита помечена slow и по
умолчанию скипается; запуск: `A4_SLOW=1 ~/venv-axiom/bin/python -m pytest
tools/tests -q` (~10 мин; интерпретатор обязан быть с jax — оркестратор
наследует sys.executable).
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import uuid
from collections.abc import Iterator
from pathlib import Path

import pytest

TOOLS_DIR = Path(__file__).resolve().parents[1]
CASE_DIR = TOOLS_DIR.parent
TESTS_DIR = Path(__file__).resolve().parent
ORCHESTRATOR = TOOLS_DIR / "run_a4_pipeline.py"
GENERATOR = TOOLS_DIR / "a4_manifest.py"


def _detect_repo_root() -> Path:
    """Корень репозитория (`git rev-parse --show-toplevel`) — якорь
    относительных путей манифеста (§4 п. 6, ADR-014 п. 8)."""
    proc = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        capture_output=True,
        text=True,
        timeout=30,
        cwd=str(CASE_DIR),
    )
    assert proc.returncode == 0, (
        f"git rev-parse --show-toplevel failed: {proc.stderr}"
    )
    return Path(os.path.realpath(proc.stdout.strip()))


REPO_ROOT = _detect_repo_root()

# Эталонный набор стадий `stage_set v1` (спека §2, ADR-014): ровно пять имён.
STAGE_SET_V1 = (
    "pretrain_checkpoint",
    "spark_inference",
    "rl_environment",
    "sft",
    "rl_base_scheme",
)

# Стадии, непокрытые в прогоне ЭТОЙ сюиты: она идёт на CPU (`JAX_PLATFORMS=cpu`
# в `_run_orchestrator`), `device_kind` журнала инференса — не стенд gb10/DGX
# Spark, поэтому `spark_inference` обязана остаться `skipped` (стенд даёт
# `executed` только по журналу стенда — T-stand1; подмена стенда запрещена,
# ADR-010). Статусы `sft` и `rl_base_scheme` оркестратор берёт из следов
# стадий (SFT-STAGE.delta §4 п. 4) и в конкретном окружении они могут быть
# `executed` — поэтому ожидаемый список непокрытых стадий K3 выводится из
# собранного манифеста, а не из константы, зафиксированной до появления следов.
CPU_RUN_UNCOVERED_STAGES = ("spark_inference",)

# Верхний предел одного прогона в тесте (ручной замер 2026-09-13 — 169,6 с на
# CPU; запас под jit-компиляцию на медленной машине).
RUN_TIMEOUT_SEC = 600

# Ручные замеры на этой машине (CPU, JAX_PLATFORMS=cpu):
# * узел n4 (2026-09-13): прогон `--seed 0 --steps 1 --tasks 2` — 169,6 с
#   (2:49.61 wall, exit 0);
# * узел n3 (2026-09-13): прогон `--seed 0 --steps 1 --tasks 1` из чужого cwd
#   (K12) — 122,9 с wall, exit 0.
# Одиночный прогон БОЛЬШЕ бюджета 120 с, поэтому прогонные тесты помечены
# `WIRE_RUN` (slow + skip без A4_SLOW=1) — ровно три прогона оркестратора на
# сюиту: общий фикстурный + повторный для K8 + прогон K12 (~10 мин).
# Метки стоят на самих прогонных тестах, а не на модуле: файловые тесты следа
# стадии (T1–T3, `assemble_stages`/`read_stage_trace` на tmp-фикстурах) быстрые,
# GPU не требуют и обязаны исполняться в обычном прогоне.
def WIRE_RUN(test):
    """Метки прогонного теста: slow + пропуск без A4_SLOW=1 (см. выше)."""
    test = pytest.mark.skipif(
        os.environ.get("A4_SLOW") != "1",
        reason="прогон оркестратора ~123–170 с (>120 с); включить: A4_SLOW=1",
    )(test)
    return pytest.mark.slow(test)


def _run_orchestrator(out_dir: Path, manifest_out: Path) -> subprocess.CompletedProcess[str]:
    """Минимальный wire-прогон оркестратора (§5.2): --seed 0 --steps 1
    --tasks 2, манифест направлен в tmp (evidence/ кейса не затрагивается).
    JAX принудительно на CPU — вердикт гейта не зависит от окружения (ADR-010).
    """
    env = dict(os.environ)
    env["JAX_PLATFORMS"] = "cpu"
    return subprocess.run(
        [
            sys.executable, str(ORCHESTRATOR),
            "--seed", "0", "--steps", "1", "--tasks", "2",
            "--out", str(out_dir),
            "--manifest-out", str(manifest_out),
        ],
        capture_output=True,
        text=True,
        timeout=RUN_TIMEOUT_SEC,
        cwd=str(CASE_DIR),
        env=env,
    )


def _summary(proc: subprocess.CompletedProcess[str]) -> dict:
    """JSON-сводка оркестратора из stdout (§5.2: печатается всегда при коде 0)."""
    assert proc.returncode == 0, f"оркестратор упал: {proc.stderr[-2000:]}"
    return json.loads(proc.stdout)


def _repo_out_dir(prefix: str) -> Path:
    """Каталог артефактов прогона ВНУТРИ репозитория (tools/tests/, не
    evidence/): по контракту §5.2 + §4 п. 6–7 `--out` обязан лежать внутри
    репозитория, иначе относительного пути для run_ref в манифесте не
    существует и оркестратор отклоняет прогон до исполнения стадий."""
    return CASE_DIR / "tools" / "tests" / f"{prefix}-{uuid.uuid4().hex[:8]}"


@pytest.fixture(scope="module")
def wire_run(tmp_path_factory: pytest.TempPathFactory) -> Iterator[dict]:
    """Общий wire-прогон для K3 и контракта §5.2 (один прогон на модуль).
    Каталог прогона — внутри репозитория (см. `_repo_out_dir`), удаляется
    после модуля; манифест — в tmp (в манифест этот путь не попадает)."""
    base = tmp_path_factory.mktemp("wire")
    out_dir = _repo_out_dir(".a4-wire-out")
    manifest = base / "manifest.json"
    proc = _run_orchestrator(out_dir, manifest)
    try:
        yield {
            "proc": proc,
            "out_dir": out_dir,
            "manifest_path": manifest,
            "summary": _summary(proc),
        }
    finally:
        shutil.rmtree(out_dir, ignore_errors=True)


def _verify(manifest_path: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(GENERATOR), "--verify", "--manifest", str(manifest_path)],
        capture_output=True,
        text=True,
        timeout=30,
        cwd=str(CASE_DIR),
    )


def _env_verdicts(out_dir: Path) -> list[dict]:
    """Механические вердикты среды прогона (env/summary.json). Поле `manifest`
    — путь прогона (относительный от корня репозитория), у двух прогонов
    разный по построению; в детерминизм (K8) входят сами вердикты: задача,
    seed, passed, reward."""
    summary_path = out_dir / "env" / "summary.json"
    assert summary_path.is_file(), f"нет сводки среды: {summary_path}"
    verdicts = json.loads(summary_path.read_text(encoding="utf-8"))["verdicts"]
    return [{k: v for k, v in verdict.items() if k != "manifest"} for verdict in verdicts]


# --- K3: wire-прогон с частичным покрытием (§6, критерий K3) ---------------


@WIRE_RUN
def test_k3_wire_run_leaves_nonempty_journal_and_manifest(wire_run: dict) -> None:
    """K3 / C-041: в --out есть непустой журнал прогона, а манифест несёт
    `run_journal` — относительный от корня репозитория путь к нему (§4 п. 5,
    барьер «манифест из ниоткуда»), и `run_ref` — slug без «/» (имя прогона,
    по которому страж стоимости ищет `evidence/budget/<run_ref>.json`).

    Прежнее утверждение «в run_ref лежит путь журнала» снято дельтой: путь
    несёт `run_journal`, `run_ref` стал именем. Инвариант существования и
    непустоты журнала сохранён — проверяется по `run_journal`."""
    manifest_path = wire_run["manifest_path"]
    assert manifest_path.is_file(), (
        "манифест не создан оркестратором; stderr генератора в журнале прогона"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["pipeline_complete"] is False

    journal_rel = manifest["run_journal"]
    assert not journal_rel.startswith("/"), (
        f"run_journal манифеста обязан быть относительным путём: {journal_rel!r}"
    )
    journal = REPO_ROOT / journal_rel
    assert journal.is_file(), f"журнал прогона не создан: {journal}"
    assert journal.stat().st_size > 0, f"журнал прогона пуст: {journal}"
    # журнал — артефакт самого прогона: он лежит в его каталоге --out
    assert journal.resolve().is_relative_to(wire_run["out_dir"].resolve()), (
        f"журнал {journal} вне каталога прогона {wire_run['out_dir']}"
    )

    run_ref = manifest["run_ref"]
    assert run_ref == wire_run["summary"]["run_ref"]
    assert run_ref and "/" not in run_ref, (
        f"run_ref обязан быть именем прогона (slug без '/', C-041): {run_ref!r}"
    )


@WIRE_RUN
def test_k3_verify_reports_partial_coverage_with_stage_names(wire_run: dict) -> None:
    """K3: `--verify` на частичном манифесте — код 1 и поимённый список
    непокрытых стадий: красный гейт читается как план работ (§4 п. 3).

    Состав списка выводится из собранного манифеста (правило §4 п. 2: не
    `executed` или пустой `evidence`), а не из константы: статусы `sft` и
    `rl_base_scheme` приходят из следов стадий и зависят от их наличия на
    диске. Непокрытость `spark_inference` — инвариант для CPU-прогона сюиты
    (журнал несёт `device_kind=cpu`, а не стенд gb10/DGX Spark, ADR-010)."""
    result = _verify(wire_run["manifest_path"])
    assert result.returncode == 1, result.stderr
    assert "покрытие частично" in result.stderr

    manifest = json.loads(wire_run["manifest_path"].read_text(encoding="utf-8"))
    uncovered = [
        stage["name"]
        for stage in manifest["stages"]
        if stage["status"] != "executed" or not stage["evidence"]
    ]
    assert "spark_inference" in uncovered, (
        f"spark_inference обязана быть непокрытой (ADR-010): {uncovered}"
    )
    for stage in CPU_RUN_UNCOVERED_STAGES:
        assert stage in result.stderr, f"стадия {stage} не поименована в: {result.stderr}"
    for stage in uncovered:
        assert stage in result.stderr, f"стадия {stage} не поименована в: {result.stderr}"


@WIRE_RUN
def test_sft_rl_statuses_come_from_stage_journals(wire_run: dict) -> None:
    """Канон SFT-STAGE.delta §4 п. 4: статус `sft`/`rl_base_scheme` — из следа
    стадии, а не из копии в оркестраторе. Проверяется на реальном прогоне:
    `executed` допустим только со ссылкой на след `stage-journal.json`; иначе
    стадия обязана быть `absent` с причиной (фабрикация статуса запрещена)."""
    stages = {stage["name"]: stage for stage in wire_run["summary"]["stages"]}
    for name, subdir in (("sft", "sft"), ("rl_base_scheme", "rl")):
        stage = stages[name]
        if stage["status"] == "executed":
            expected = f"stage_journal=evidence/a4-run-wire/{subdir}/stage-journal.json"
            assert expected in stage["evidence"], (
                f"{name}: executed без ссылки на след стадии: {stage['evidence']}"
            )
        else:
            assert stage["status"] in ("absent", "skipped"), (
                f"{name}: не executed и не skipped — статус вне словаря §2: "
                f"{stage['status']}"
            )
            assert any(item.startswith("причина:") for item in stage["evidence"]), (
                f"{name}: неисполненная стадия обязана нести причину: {stage['evidence']}"
            )


# --- K8: детерминизм при одном seed (§6, критерий K8) -----------------------


@WIRE_RUN
def test_k8_same_seed_same_weights_hash_and_verdicts(wire_run: dict, tmp_path: Path) -> None:
    """K8: повторный прогон с тем же --seed 0 даёт тот же
    `model_weights_sha256` и те же вердикты среды (репетиция A5)."""
    repeat_out = _repo_out_dir(".a4-repeat-out")
    repeat_manifest = tmp_path / "repeat-manifest.json"
    try:
        repeat = _summary(_run_orchestrator(repeat_out, repeat_manifest))

        first = wire_run["summary"]
        assert repeat["model_weights_sha256"] == first["model_weights_sha256"]
        assert repeat["dataset_sha256"] == first["dataset_sha256"]
        assert _env_verdicts(repeat_out) == _env_verdicts(wire_run["out_dir"])
    finally:
        shutil.rmtree(repeat_out, ignore_errors=True)


# --- Контракт §5.2: stdout JSON-сводка и честность статусов -----------------


@WIRE_RUN
def test_cli_exit_zero_on_partial_coverage(wire_run: dict) -> None:
    """§5.2: код 0 — доступные стадии исполнены, независимо от полноты
    конвейера (частичный прогон легален)."""
    assert wire_run["proc"].returncode == 0, wire_run["proc"].stderr[-2000:]


@WIRE_RUN
def test_cli_stdout_json_summary_contract(wire_run: dict) -> None:
    """§5.2: stdout — JSON-сводка {run_ref, model_weights_sha256,
    dataset_sha256, stages, pipeline_complete}; хеши — sha256 (64 hex)."""
    summary = wire_run["summary"]
    for key in ("run_ref", "model_weights_sha256", "dataset_sha256", "stages", "pipeline_complete"):
        assert key in summary, f"в сводке нет поля {key}"
    for key in ("model_weights_sha256", "dataset_sha256"):
        value = summary[key]
        assert isinstance(value, str) and len(value) == 64
        int(value, 16)  # hex
    assert summary["pipeline_complete"] is False
    names = [stage["name"] for stage in summary["stages"]]
    assert sorted(names) == sorted(STAGE_SET_V1)


@WIRE_RUN
def test_spark_inference_not_executed(wire_run: dict) -> None:
    """§5.2/§2: в CPU-прогоне (`JAX_PLATFORMS=cpu`) стадия spark_inference НЕ
    имеет статуса executed — журнал несёт `device_kind=cpu`, а не стенд
    gb10/DGX Spark; подмена стенда запрещена (ADR-010). Статус честный:
    skipped или absent. Разбор «стенд / не стенд» — T-stand1–T-stand3."""
    statuses = {stage["name"]: stage["status"] for stage in wire_run["summary"]["stages"]}
    assert statuses["spark_inference"] != "executed"
    assert statuses["spark_inference"] in ("skipped", "absent")


# --- K12 / §5.2 + §4 п. 6–7: сквозная портируемость путей --------------------


def _iter_strings(obj: object) -> Iterator[str]:
    """Все строковые значения JSON рекурсивно (dict/list/str)."""
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for value in obj.values():
            yield from _iter_strings(value)
    elif isinstance(obj, list):
        for item in obj:
            yield from _iter_strings(item)


@pytest.fixture(scope="module")
def wire_run_foreign_cwd(tmp_path_factory: pytest.TempPathFactory) -> Iterator[dict]:
    """K12: wire-прогон оркестратора из НЕ-корневого cwd (tmp-каталог) с
    минимальными `--steps 1 --tasks 1` и ОТНОСИТЕЛЬНЫМ `--out` внутри
    репозитория (по §5.2 относительный --out якорится к корню кейса, а не к
    cwd вызова). Манифест направлен в tmp (evidence/ кейса не затрагивается).
    Каталог артефактов прогона внутри репозитория удаляется после модуля."""
    base = tmp_path_factory.mktemp("k12")
    foreign_cwd = base / "cwd"
    foreign_cwd.mkdir()
    out_abs = _repo_out_dir(".a4-k12-out")
    out_rel = out_abs.relative_to(CASE_DIR).as_posix()  # --out относительным
    manifest = base / "manifest.json"
    env = dict(os.environ)
    env["JAX_PLATFORMS"] = "cpu"  # вердикт не зависит от окружения (ADR-010)
    proc = subprocess.run(
        [
            sys.executable, str(ORCHESTRATOR),
            "--seed", "0", "--steps", "1", "--tasks", "1",
            "--out", out_rel,
            "--manifest-out", str(manifest),
        ],
        capture_output=True,
        text=True,
        timeout=RUN_TIMEOUT_SEC,
        cwd=str(foreign_cwd),
        env=env,
    )
    try:
        yield {
            "proc": proc,
            "manifest_path": manifest,
            "out_abs": out_abs,
            "foreign_cwd": foreign_cwd,
        }
    finally:
        shutil.rmtree(out_abs, ignore_errors=True)


@WIRE_RUN
def test_k12_run_from_foreign_cwd_manifest_has_no_absolute_paths(
    wire_run_foreign_cwd: dict,
) -> None:
    """K12: прогон из произвольного cwd -> код 0, манифест создан, и в нём
    нет ни одной строки, начинающейся с '/' (рекурсивно по всем строковым
    значениям JSON); дополнительно ни один токен строки (в т.ч. значение
    после '=' формата `key=<путь>`) не является абсолютным путём (§4 п. 6)."""
    proc = wire_run_foreign_cwd["proc"]
    assert proc.returncode == 0, (
        f"оркестратор из чужого cwd упал: {proc.stderr[-2000:]}"
    )
    manifest_path = wire_run_foreign_cwd["manifest_path"]
    assert manifest_path.is_file(), "манифест не создан оркестратором"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for text in _iter_strings(manifest):
        assert not text.startswith("/"), f"абсолютный путь в манифесте: {text!r}"
        for token in text.split():
            candidate = token.split("=", 1)[1] if "=" in token else token
            assert not candidate.startswith("/"), (
                f"абсолютный путь-токен в строке манифеста: {text!r}"
            )


@WIRE_RUN
def test_k12_run_ref_resolves_from_repo_root(wire_run_foreign_cwd: dict) -> None:
    """K12 / C-041 (повторный вывод): `run_journal` манифеста — относительный
    путь, резолвится от КОРНЯ РЕПОЗИТОРИЯ (а не от cwd запуска), существует и
    непуст; `run_ref` — slug без «/», совпадающий со сводкой stdout: прогон
    из чужого cwd находит свою смету и не теряет журнал."""
    proc = wire_run_foreign_cwd["proc"]
    assert proc.returncode == 0, proc.stderr[-2000:]
    summary = json.loads(proc.stdout)
    manifest = json.loads(
        wire_run_foreign_cwd["manifest_path"].read_text(encoding="utf-8")
    )
    assert manifest["run_ref"] == summary["run_ref"]
    run_ref = summary["run_ref"]
    assert run_ref and "/" not in run_ref, (
        f"run_ref обязан быть slug без '/': {run_ref!r}"
    )

    journal_rel = manifest["run_journal"]
    assert not journal_rel.startswith("/"), (
        f"run_journal манифеста обязан быть относительным путём: {journal_rel!r}"
    )
    resolved = REPO_ROOT / journal_rel
    assert resolved.is_file(), (
        f"run_journal не резолвится от корня репозитория: {resolved}"
    )
    assert resolved.stat().st_size > 0, f"run_journal пуст: {resolved}"


# --- T1–T3: статус стадии sft/rl_base_scheme — из следа стадии --------------
#
# SFT-STAGE.delta §4 п. 4: «оркестратор обязан эмитить статус стадии из следа
# (stage-journal.json), а не хранить свою копию». Проверяется файловыми
# фикстурами на явном repo_root (cwd и каталог evidence/ кейса не
# затрагиваются; GPU не нужен): оркестратор след только читает и не
# переигрывает стадию.

# Следы стадий лежат в repo_root по этому префиксу (см. TRACE_ROOT оркестратора).
TRACE_ROOT = "evidence/a4-run-wire"
STAGE_TRACE_SUBDIR = {"sft": "sft", "rl_base_scheme": "rl"}

# Прежние причины absent — при отсутствии/битом следе они сохраняются дословно.
ABSENT_REASON = {
    "sft": "причина: кода SFT нет (ADR-005 п. 1 не реализован)",
    "rl_base_scheme": "причина: кода RL по схеме базы нет (ADR-005 п. 2 не реализован)",
}

TREE_HASH = "ab" * 32


@pytest.fixture(scope="module")
def pipeline():
    """Модуль оркестратора (импорт требует jax-интерпретатора, как и прогонные
    тесты; сами T1–T3 работают только с файловой системой)."""
    sys.path.insert(0, str(TOOLS_DIR))
    import run_a4_pipeline

    return run_a4_pipeline


def _write_trace(repo_root: Path, name: str, payload: object) -> Path:
    """След стадии в фикстурном репозитории: <repo_root>/evidence/a4-run-wire/
    <sft|rl>/stage-journal.json (payload — то, что окажется в файле)."""
    path = repo_root / TRACE_ROOT / STAGE_TRACE_SUBDIR[name] / "stage-journal.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False),
        encoding="utf-8",
    )
    return path


def _valid_trace(name: str, steps: int) -> dict:
    """Валидный след стадии: status=executed, steps, checkpoint.tree_hash."""
    return {
        "schema": f"{STAGE_TRACE_SUBDIR[name]}-stage-journal/v1",
        "stage": name,
        "status": "executed",
        "steps": steps,
        "checkpoint": {
            "path": f"{TRACE_ROOT}/{STAGE_TRACE_SUBDIR[name]}/checkpoint",
            "format": "orbax",
            "tree_hash": TREE_HASH,
        },
    }


def _assemble(pipeline, repo_root: Path) -> list[dict]:
    """assemble_stages() с явным repo_root (cwd не подменяется). Входные пути
    стадий оркестратора — фиктивные файлы внутри фикстурного репозитория:
    проверяются только стадии sft/rl_base_scheme, читаемые из следов."""
    return pipeline.assemble_stages(
        model_weights_sha256="0" * 64,
        checkpoint_dir=repo_root / "run" / "checkpoints" / "pretrain",
        executed_steps=1,
        env_summary=repo_root / "run" / "env" / "summary.json",
        env_verdicts=[{"passed": True}],
        inference_journal=repo_root / "run" / "inference" / "journal.json",
        dataset_sha256="1" * 64,
        repo_root=repo_root,
    )


def _stage(stages: list[dict], name: str) -> dict:
    return next(stage for stage in stages if stage["name"] == name)


def test_t1_stages_emitted_from_valid_traces(pipeline, tmp_path: Path) -> None:
    """T1: валидные следы (status=executed, steps, checkpoint.tree_hash) ->
    sft и rl_base_scheme эмитятся как executed, evidence несёт ссылку на след
    и собран только из присутствующих полей (steps/tree_hash/checkpoint)."""
    repo_root = tmp_path.resolve()
    _write_trace(repo_root, "sft", _valid_trace("sft", steps=200))
    _write_trace(repo_root, "rl_base_scheme", _valid_trace("rl_base_scheme", steps=10))

    stages = _assemble(pipeline, repo_root)

    for name, subdir, steps in (("sft", "sft", 200), ("rl_base_scheme", "rl", 10)):
        stage = _stage(stages, name)
        assert stage["status"] == "executed", stage
        assert stage["evidence"] == [
            f"stage_journal={TRACE_ROOT}/{subdir}/stage-journal.json",
            f"steps={steps}",
            f"tree_hash={TREE_HASH}",
            f"checkpoint={TRACE_ROOT}/{subdir}/checkpoint",
        ], stage["evidence"]
        for item in stage["evidence"]:
            assert not item.split("=", 1)[-1].startswith("/"), (
                f"абсолютный путь в evidence (ADR-014 п. 8): {item!r}"
            )


def test_t2_missing_trace_gives_absent_with_prior_reason(pipeline, tmp_path: Path) -> None:
    """T2: следа нет -> прежний absent с прежней причиной (executed не
    выдумывается)."""
    repo_root = tmp_path.resolve()  # следов не создаём

    stages = _assemble(pipeline, repo_root)

    for name in ("sft", "rl_base_scheme"):
        stage = _stage(stages, name)
        assert stage["status"] == "absent", stage
        assert stage["evidence"] == [ABSENT_REASON[name]], stage["evidence"]


@pytest.mark.parametrize(
    ("name", "payload"),
    [
        ("sft", "{ это не JSON"),  # файл не парсится
        ("rl_base_scheme", {"schema": "rl-stage-journal/v1", "steps": 10}),  # нет status
        ("sft", {"status": ""}),  # status пуст
        # Статус вне словаря §2; регистр значим: FAILED ≠ failed (после дельты
        # «статус failed» сам failed — легальный статус и эмитится как есть,
        # см. T4, поэтому битым следом проверяется именно FAILED).
        ("rl_base_scheme", {"status": "FAILED"}),
    ],
)
def test_t3_broken_trace_gives_absent_never_executed(
    pipeline, tmp_path: Path, name: str, payload: object
) -> None:
    """T3: след битый (невалидный JSON, нет/пуст `status`, статус вне словаря
    манифеста §2) -> absent с прежней причиной, НЕ executed: фабрикация
    статуса запрещена (гейт C-038 не должен ложно зеленеть)."""
    repo_root = tmp_path.resolve()
    _write_trace(repo_root, name, payload)

    stages = _assemble(pipeline, repo_root)

    stage = _stage(stages, name)
    assert stage["status"] != "executed", stage
    assert stage["status"] == "absent", stage
    assert stage["evidence"] == [ABSENT_REASON[name]], stage["evidence"]
    assert pipeline.read_stage_trace(repo_root, name) is None


# --- T4: статус failed из следа эмитится как есть (§2) ---------------------
#
# §2 (дельта «статус failed»): «след стадии stage-journal.json с status=failed
# эмитится как есть — превращать failed в absent запрещено: искажает факт
# исполнения». След `sft` со `status=failed` пишет tools/run_sft_smoke.py
# (loss не упал / отказ чекпойнта) — оркестратор обязан перенести его в
# манифест без подмены. GPU не нужен: след только читается.


def test_t4_failed_trace_emitted_as_failed_not_absent(
    pipeline, tmp_path: Path
) -> None:
    """T4 / §2: валидный след со `status=failed` -> стадия `failed` (НЕ absent
    и не executed), evidence честный — путь следа и реально присутствующие
    поля, без домысленных значений (фабрикация запрещена)."""
    repo_root = tmp_path.resolve()
    failed = _valid_trace("sft", steps=200)
    failed["status"] = "failed"
    failed["error"] = "лосс не упал на смоук-окне (loss_first <= loss_last)"
    _write_trace(repo_root, "sft", failed)
    _write_trace(repo_root, "rl_base_scheme", _valid_trace("rl_base_scheme", steps=10))

    stages = _assemble(pipeline, repo_root)

    sft = _stage(stages, "sft")
    assert sft["status"] == "failed", sft
    assert sft["evidence"] == [
        f"stage_journal={TRACE_ROOT}/sft/stage-journal.json",
        "steps=200",
        f"tree_hash={TREE_HASH}",
        f"checkpoint={TRACE_ROOT}/sft/checkpoint",
    ], sft["evidence"]
    assert pipeline.read_stage_trace(repo_root, "sft") is not None
    # Соседняя стадия не заражена: статус берётся из её собственного следа.
    assert _stage(stages, "rl_base_scheme")["status"] == "executed"


# --- T-stand1…T-stand4: стадия spark_inference — по журналу инференса -------
#
# Спека A4-RUN.delta §2: `spark_inference` доказывается журналом инференса с
# `device_kind` стенда gb10 (DGX Spark); «замер есть на RTX 4080 → skipped».
# Оркестратор различает стенд и локальный прогон по `device_kind` САМОГО
# журнала (`detect_spark_stage`), а не по факту запуска: до этой дельты подпись
# устройства не читалась вовсе, и боевой прогон на GB10 (29.09.2026) записал
# честно исполненную стадию как `skipped` с причиной «инференс исполнен
# локально на CPU». Проверки файловые (журнал-фикстура): прогон инференса и GPU
# не нужны, cwd и каталог evidence/ кейса не затрагиваются.

# Журнал, который `_assemble` передаёт в assemble_stages как inference_journal.
INFERENCE_JOURNAL_REL = "run/inference/journal.json"

# Прежняя пометка журнала для локального прогона — при не-стенде обязана
# сохраниться ДОСЛОВНО (дельты прозы в стадии не вносит).
LOCAL_JOURNAL_NOTE = (
    "Локальный инференс (CPU JAX) — провод «чекпойнт → генерация»; "
    "это НЕ стадия spark_inference (стенд gb10/DGX Spark)."
)

STAND_JOURNAL_NOTE = "инференс на стенде (spark_inference)"


def _journal_path(repo_root: Path) -> Path:
    return repo_root / INFERENCE_JOURNAL_REL


def _write_journal(repo_root: Path, payload: object) -> Path:
    """Журнал инференса на пути, который видит `_assemble` (payload — то, что
    окажется в файле)."""
    path = _journal_path(repo_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False),
        encoding="utf-8",
    )
    return path


def _rel(repo_root: Path, path: Path) -> str:
    """Путь от корня так, как его считает `repo_rel` оркестратора (через
    resolve): сверяются именно те строки, что уйдут в манифест."""
    return (
        Path(os.path.realpath(path))
        .relative_to(Path(os.path.realpath(repo_root)))
        .as_posix()
    )


def _spark_skipped_evidence(repo_root: Path) -> list[str]:
    """Прежняя причина skipped — дословно, с путём локального журнала."""
    return [
        "причина: инференс исполнен локально на CPU JAX, а не на стенде "
        "gb10/DGX Spark; подмена стенда запрещена (ADR-010). "
        f"Локальный журнал: {_rel(repo_root, _journal_path(repo_root))}"
    ]


@pytest.mark.parametrize("device_kind", ["NVIDIA GB10", "gb10", "NVIDIA DGX Spark"])
def test_t_stand1_stand_journal_gives_executed(
    pipeline, tmp_path: Path, device_kind: str
) -> None:
    """T-stand1 / §2: журнал с `device_kind` стенда (GB10 или DGX; регистр не
    значим — JAX отдаёт «NVIDIA GB10», спека пишет `device_kind=gb10`) даёт
    стадии `executed`; evidence — относительный путь журнала (ADR-014 п. 8) и
    `device_kind` из журнала, и ничего домысленного."""
    repo_root = tmp_path.resolve()
    _write_journal(repo_root, {"device_kind": device_kind, "generated_ids": [1, 2]})

    stage = _stage(_assemble(pipeline, repo_root), "spark_inference")

    assert stage["status"] == "executed", stage
    assert stage["evidence"] == [
        f"journal={INFERENCE_JOURNAL_REL}",
        f"device_kind={device_kind}",
    ], stage["evidence"]
    for item in stage["evidence"]:
        assert not item.split("=", 1)[-1].startswith("/"), (
            f"абсолютный путь в evidence (ADR-014 п. 8): {item!r}"
        )
    # Тот же вердикт даёт и чистая функция детекции (её вызывает assemble).
    assert pipeline.detect_spark_stage(_journal_path(repo_root), repo_root) == stage


def test_t_stand2_rtx4080_journal_stays_skipped(pipeline, tmp_path: Path) -> None:
    """T-stand2 (anti-weakening) / §2: журнал с `device_kind` другой GPU —
    «NVIDIA GeForce RTX 4080 SUPER» (реальная подпись замеров кейса,
    evidence/a4-run-wire/*/stage-journal.json) — `executed` НЕ даёт: статус
    прежний `skipped` с прежней причиной. Замер на 4080 стендом не является,
    подмена стенда запрещена (ADR-010)."""
    repo_root = tmp_path.resolve()
    _write_journal(repo_root, {"device_kind": "NVIDIA GeForce RTX 4080 SUPER"})

    stage = _stage(_assemble(pipeline, repo_root), "spark_inference")

    assert stage["status"] != "executed", stage
    assert stage["status"] == "skipped", stage
    assert stage["evidence"] == _spark_skipped_evidence(repo_root), stage["evidence"]


@pytest.mark.parametrize(
    "payload",
    [
        None,  # журнала нет вовсе
        "{ это не JSON",  # журнал не парсится
        [1, 2, 3],  # не JSON-объект
        {"note": "журнал без device_kind"},  # поля device_kind нет
        {"device_kind": ""},  # поле пусто
        {"device_kind": 42},  # поле не строка
        {"device_kind": "cpu"},  # CPU JAX — локальный прогон
        # Подпись с разделителем следов: генератор §5.1 делит evidence по «;»,
        # такая строка исказила бы состав стадии — стендом не признаётся.
        {"device_kind": "NVIDIA GB10; sft=executed"},
    ],
)
def test_t_stand3_missing_or_broken_journal_keeps_prior_behaviour(
    pipeline, tmp_path: Path, payload: object
) -> None:
    """T-stand3 / §2: журнала нет или он бит (не JSON, не объект, нет/пусто/
    не строка `device_kind`, чужая подпись устройства) -> прежнее поведение:
    `skipped` с прежней причиной, не `executed`. Без следа стенда `executed` не
    выдумывается (фабрикация статуса запрещена)."""
    repo_root = tmp_path.resolve()
    if payload is not None:
        _write_journal(repo_root, payload)

    stage = _stage(_assemble(pipeline, repo_root), "spark_inference")

    assert stage["status"] != "executed", stage
    assert stage["status"] == "skipped", stage
    assert stage["evidence"] == _spark_skipped_evidence(repo_root), stage["evidence"]


def test_t_stand4_journal_note_names_stand_as_stand(pipeline) -> None:
    """T-stand4 / §2 + честность журнала: `note` журнала инференса называет
    стенд стендом («инференс на стенде (spark_inference)»), а локальный
    прогон — прежним текстом ДОСЛОВНО. Проверяется на сборщике тела журнала:
    сам инференс на CPU требует jit-компиляции и в быстрый прогон не входит,
    поэтому честность пометки держится на чистой функции."""
    for device_kind in ("NVIDIA GB10", "gb10", "NVIDIA DGX Spark"):
        payload = pipeline.inference_journal_payload(device_kind, [1, 2], 0)
        assert payload["device_kind"] == device_kind
        assert payload["note"] == STAND_JOURNAL_NOTE, payload["note"]

    for device_kind in ("cpu", "NVIDIA GeForce RTX 4080 SUPER"):
        payload = pipeline.inference_journal_payload(device_kind, [1, 2], 0)
        assert payload["note"] == LOCAL_JOURNAL_NOTE, payload["note"]


# --- C-041: run_ref по умолчанию — slug, а не путь --------------------------
#
# Страж стоимости (tools/check_budget_gate.py, C-041) читает поле run_ref
# манифеста и требует имени прогона (RUN_REF_RE: буквы/цифры/'.'/'_'/'-');
# по нему ищется смета evidence/budget/<run_ref>.json. Прежнее поведение
# оркестратора записывало в run_ref путь журнала
# («evidence/a4-run-wire/run-journal.json/run-journal.json» — наблюдение
# приёмки 26.09.2026), из-за чего прогон блокировался стражем.


def test_r1_default_run_id_is_slug_for_nested_out(pipeline) -> None:
    """T-r1 / C-041: run_ref по умолчанию — slug без «/», выведенный из
    последней компоненты --out (в т.ч. при вложенном пути и «грязном» имени);
    когда слага не вывести — детерминированный `a4-run-<хеш>` от
    (seed, steps, tasks), а не пустое имя."""
    nested = CASE_DIR / "tools" / "tests" / ".a4-wire-out-1234abcd"
    run_id = pipeline.default_run_id(nested, seed=0, steps=1, tasks=2)
    assert run_id and "/" not in run_id
    assert pipeline.RUN_ID_RE.match(run_id)
    assert run_id == "a4-wire-out-1234abcd"

    # Наблюдённый дефект: --out указывал на «.../run-journal.json», и run_ref
    # становился путём «.../run-journal.json/run-journal.json».
    observed = CASE_DIR / "evidence" / "a4-run-wire" / "run-journal.json"
    slug = pipeline.default_run_id(observed, seed=0, steps=1, tasks=2)
    assert slug == "run-journal"
    assert "/" not in slug

    # Компонента без допустимых символов: слага не вывести -> fallback.
    hopeless = Path("/tmp/---")
    fallback = pipeline.default_run_id(hopeless, seed=0, steps=1, tasks=2)
    assert fallback.startswith("a4-run-")
    assert pipeline.RUN_ID_RE.match(fallback)
    assert fallback == pipeline.default_run_id(hopeless, seed=0, steps=1, tasks=2)
    assert fallback != pipeline.default_run_id(hopeless, seed=1, steps=1, tasks=2)


def test_r2_generator_call_records_slug_run_ref(pipeline, tmp_path: Path) -> None:
    """T-r2 / C-041 (сквозной, без GPU): вызов генератора оркестратором даёт
    манифест, поле run_ref которого — slug (проходит RUN_REF_RE стража
    стоимости), а путь журнала прогона виден полем run_journal: прогон находит
    свою смету, манифест не теряет источник доказательства (§4 п. 5)."""
    scratch = Path(tempfile.mkdtemp(prefix=".a4-r2-", dir=str(TESTS_DIR)))
    try:
        journal = scratch / "run-journal.json"
        journal.write_text('{"schema": "a4-run-journal/v1"}\n', encoding="utf-8")
        manifest_out = tmp_path / "manifest.json"
        result = pipeline.call_generator(
            weights_hash="a" * 64,
            dataset_hash="b" * 64,
            run_ref=journal,
            stages=[],
            manifest_out=manifest_out,
            repo_root=REPO_ROOT,
            run_id="a4-wire-out-1234abcd",
        )
        assert result["returncode"] == 0, result["stderr"]
        manifest = json.loads(manifest_out.read_text(encoding="utf-8"))
        assert manifest["run_ref"] == "a4-wire-out-1234abcd"
        assert pipeline.RUN_ID_RE.match(manifest["run_ref"])
        assert manifest["run_journal"] == (
            journal.resolve().relative_to(REPO_ROOT).as_posix()
        )
        assert not manifest["run_journal"].startswith("/")
        assert manifest["pipeline_complete"] is False
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
