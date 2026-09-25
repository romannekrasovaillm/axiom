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
  (run_ref), манифест создан, `pipeline_complete=false`, `--verify` даёт код 1
  и поимённый список непокрытых стадий (sft, rl_base_scheme, spark_inference);
* K8 — детерминизм: два прогона с одним --seed дают одинаковые
  `model_weights_sha256` и вердикты среды;
* §5.2 — stdout содержит JSON-сводку {run_ref, model_weights_sha256,
  dataset_sha256, stages, pipeline_complete}; стадия spark_inference НЕ имеет
  статуса executed (подмена стенда запрещена); код 0 при частичном покрытии;
* K12 — сквозная портируемость: прогон из НЕ-корневого cwd (tmp-каталог) с
  минимальными `--steps 1 --tasks 1` и относительным `--out` внутри
  репозитория -> код 0; в манифесте рекурсивно нет ни одной строки,
  начинающейся с '/'; `run_ref` резолвится от корня репозитория, существует
  и непуст. Каталог артефактов прогона (tools/tests/.a4-k12-out-*) удаляется
  после модуля.

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
import shutil
import subprocess
import sys
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

# Стадии, которые на 13.09.2026 не могут быть executed (спека §2): стенда gb10
# у оркестратора нет, кода SFT/RL нет. Их подмена запрещена (ADR-010).
UNCOVERED_STAGES = ("sft", "rl_base_scheme", "spark_inference")

# Верхний предел одного прогона в тесте (ручной замер 2026-09-13 — 169,6 с на
# CPU; запас под jit-компиляцию на медленной машине).
RUN_TIMEOUT_SEC = 600

# Ручные замеры на этой машине (CPU, JAX_PLATFORMS=cpu):
# * узел n4 (2026-09-13): прогон `--seed 0 --steps 1 --tasks 2` — 169,6 с
#   (2:49.61 wall, exit 0);
# * узел n3 (2026-09-13): прогон `--seed 0 --steps 1 --tasks 1` из чужого cwd
#   (K12) — 122,9 с wall, exit 0.
# Одиночный прогон БОЛЬШЕ бюджета 120 с. Поэтому все прогонные тесты ниже —
# slow и по умолчанию скипаются; запуск: `A4_SLOW=1 ~/venv-axiom/bin/python
# -m pytest tools/tests -q` (сюите нужно ~10 мин: ровно три прогона
# оркестратора — общий фикстурный + повторный для K8 + прогон K12).
pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        os.environ.get("A4_SLOW") != "1",
        reason="прогон оркестратора ~123–170 с (>120 с); включить: A4_SLOW=1",
    ),
]


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


def test_k3_wire_run_leaves_nonempty_journal_and_manifest(wire_run: dict) -> None:
    """K3: в --out есть непустой журнал прогона (run_ref из сводки §5.2 —
    относительный от корня репозитория путь) и созданный манифест с
    вычисленным pipeline_complete=false."""
    run_ref = REPO_ROOT / wire_run["summary"]["run_ref"]
    assert run_ref.is_file(), f"журнал прогона не создан: {run_ref}"
    assert run_ref.stat().st_size > 0, f"журнал прогона пуст: {run_ref}"

    manifest_path = wire_run["manifest_path"]
    assert manifest_path.is_file(), (
        "манифест не создан оркестратором; stderr генератора в журнале прогона"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["pipeline_complete"] is False


def test_k3_verify_reports_partial_coverage_with_stage_names(wire_run: dict) -> None:
    """K3: `--verify` на частичном манифесте — код 1 и поимённый список
    непокрытых стадий (sft, rl_base_scheme, spark_inference): красный гейт
    читается как план работ (§4 п. 3)."""
    result = _verify(wire_run["manifest_path"])
    assert result.returncode == 1, result.stderr
    assert "покрытие частично" in result.stderr
    for stage in UNCOVERED_STAGES:
        assert stage in result.stderr, f"стадия {stage} не поименована в: {result.stderr}"


# --- K8: детерминизм при одном seed (§6, критерий K8) -----------------------


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


def test_cli_exit_zero_on_partial_coverage(wire_run: dict) -> None:
    """§5.2: код 0 — доступные стадии исполнены, независимо от полноты
    конвейера (частичный прогон легален)."""
    assert wire_run["proc"].returncode == 0, wire_run["proc"].stderr[-2000:]


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


def test_spark_inference_not_executed(wire_run: dict) -> None:
    """§5.2/§2: стадия spark_inference НЕ имеет статуса executed — подмена
    стенда gb10 запрещена (ADR-010); статус честный: skipped или absent."""
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


def test_k12_run_ref_resolves_from_repo_root(wire_run_foreign_cwd: dict) -> None:
    """K12 (повторный вывод): `run_ref` из сводки stdout и из манифеста —
    относительный путь, резолвится от КОРНЯ РЕПОЗИТОРИЯ (а не от cwd
    запуска), существует и непуст."""
    proc = wire_run_foreign_cwd["proc"]
    assert proc.returncode == 0, proc.stderr[-2000:]
    summary = json.loads(proc.stdout)
    manifest = json.loads(
        wire_run_foreign_cwd["manifest_path"].read_text(encoding="utf-8")
    )
    assert manifest["run_ref"] == summary["run_ref"]
    run_ref = summary["run_ref"]
    assert not run_ref.startswith("/")
    resolved = REPO_ROOT / run_ref
    assert resolved.is_file(), (
        f"run_ref не резолвится от корня репозитория: {resolved}"
    )
    assert resolved.stat().st_size > 0, f"run_ref пуст: {resolved}"
