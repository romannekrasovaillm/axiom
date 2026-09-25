"""Контрактные тесты генератора/верификатора манифеста прогона A4
(tools/a4_manifest.py), схема `a4-skeleton-run-manifest/v2`.

Источник истины — спека `docs/specs/A4-RUN.delta.md` (§3 схема v2, §4 правила
полноты и ретенции, §5.1 CLI-контракт генератора, §6 критерии приёмки K1–K7,
K11) и ADR-014, а не реализация. Все сценарии — вызов CLI подпроцессом;
каталог evidence/ кейса не затрагивается.

Фикстуры двух видов (§4 п. 6–7: пути манифеста обязаны быть относительными
от корня репозитория, абсолютный путь вне репозитория — отказ генерации):

* артефакты, не попадающие в манифест (проверяемые манифесты, выходной файл
  генерации), — в tmp_path;
* ЖУРНАЛ ПРОГОНА для успешной генерации — во временном каталоге ВНУТРИ
  репозитория (фикстура `repo_scratch`, mkdtemp под tools/tests/, удаляется
  после теста): журнал обязан существовать внутри репо (§4 п. 5–7). До дельты
  K11 журналы лежали в tmp_path — после введения §4 п. 7 абсолютный --run-ref
  вне репозитория законно отклоняется, поэтому фикстуры успешной генерации
  переведены внутрь репозитория (сценарии и утверждения тестов не изменены).

Сценарии:
* манифест отсутствует -> --verify код 1, «манифест не найден» (K1);
* пустой хеш -> FAIL; мусорный хеш -> FAIL (поле поименовано в stderr);
* валидный манифест v2 полного покрытия (фикстура) -> --verify PASS (K4);
* генерация без --weights-hash / --dataset-hash / --run-ref -> код 1,
  выходной файл не создан (K2);
* частичное покрытие -> манифест создан с pipeline_complete=false,
  --verify код 1 с «покрытие частично: <поимённый список>» (K3);
* pipeline_complete вычисляется генератором, переданное извне значение
  игнорируется (K5, §4 п. 1);
* манифест схемы v1 -> код 1, сообщение про ожидаемую v2 (K6);
* --run-ref на несуществующий или пустой журнал -> код 1, файл не создан
  (K7, §4 п. 5);
* контракт §3/§5.1: в записанном манифесте ровно пять имён stage_set v1,
  непереданные стадии получают статус absent, неизвестное имя стадии или
  недопустимый статус -> код 1 без записи;
* портируемость путей (K11, §4 п. 6–7): --verify отклоняет манифест с
  абсолютным путём в run_ref и с абсолютным путём внутри строки
  stages[].evidence («абсолютный путь запрещён: …»); генерация с абсолютным
  --run-ref ВНЕ репозитория -> код 1, файл не создан; генерация с абсолютным
  --run-ref ВНУТРИ репозитория -> в записанном манифесте путь относительный.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Iterator
from pathlib import Path

import pytest

TOOLS_DIR = Path(__file__).resolve().parents[1]
CASE_DIR = TOOLS_DIR.parent
TESTS_DIR = Path(__file__).resolve().parent
SCRIPT = TOOLS_DIR / "a4_manifest.py"


def _detect_repo_root() -> Path:
    """Корень репозитория (`git rev-parse --show-toplevel`) — якорь
    относительных путей манифеста (§4 п. 6), как и у генератора."""
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

# Эталонный набор стадий `stage_set v1` (спека §2, ADR-014 п. 2): ровно пять
# имён. Расширение — только новой версией набора, молча дорисовывать запрещено.
STAGE_SET_V1 = (
    "pretrain_checkpoint",
    "spark_inference",
    "rl_environment",
    "sft",
    "rl_base_scheme",
)

# Фикстура схемы v1: после дельты ADR-014 невалидна (§4 п. 2, ADR-014 п. 7).
MANIFEST_V1 = {
    "schema": "a4-skeleton-run-manifest/v1",
    "model_weights_sha256": "a" * 64,
    "dataset_sha256": "b" * 64,
    "environment_version": "v1.1",
    "harness_version": "0.1.4",
    "git_commit": "c" * 40,
    "run_date": "2026-09-13",
    "run_ref": "https://example.invalid/runs/42",
}


def _stage(name: str, status: str = "executed", evidence: list[str] | None = None) -> dict:
    """Стадия манифеста v2; по умолчанию `executed` с непустым evidence."""
    if evidence is None:
        evidence = [f"{name}: след исполнения"]
    return {"name": name, "status": status, "evidence": evidence}


def _v2_manifest(**overrides: object) -> dict:
    """Валидный манифест схемы v2 ПОЛНОГО покрытия (§3): обязательные поля v1
    сохранены, backend заполнен (ADR-010), все пять стадий stage_set v1
    `executed` с непустым `evidence`, `pipeline_complete=true`."""
    manifest: dict = {
        "schema": "a4-skeleton-run-manifest/v2",
        "stage_set_version": "v1",
        "model_weights_sha256": "a" * 64,
        "dataset_sha256": "b" * 64,
        "environment_version": "v1.1",
        "harness_version": "0.1.4",
        "git_commit": "c" * 40,
        "run_date": "2026-09-13",
        "run_ref": "https://example.invalid/runs/42",
        "backend": {
            "platform": "cpu",
            "device_kind": "cpu",
            "jax_version": "0.4.30",
            "matmul_precision": "default",
        },
        "stages": [_stage(name) for name in STAGE_SET_V1],
        "pipeline_complete": True,
    }
    manifest.update(overrides)
    return manifest


def _run(*args: str, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    # A4_* переменные окружения — тоже вход генератора; вычищаем их, чтобы
    # результат теста не зависел от окружения раннера.
    env = {key: value for key, value in os.environ.items() if not key.startswith("A4_")}
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        capture_output=True,
        text=True,
        timeout=30,
        cwd=str(cwd) if cwd else None,
        env=env,
    )


def _write(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


@pytest.fixture
def repo_scratch() -> Iterator[Path]:
    """Временный каталог ВНУТРИ репозитория (mkdtemp под tools/tests/,
    удаляется после теста). Журнал прогона для генерации обязан лежать
    внутри репозитория: абсолютный --run-ref вне репо отклоняется,
    а в манифест попадает путь, относительный от его корня (§4 п. 5–7).
    Каталог evidence/ кейса не используется."""
    scratch = Path(tempfile.mkdtemp(prefix=".a4-test-", dir=str(TESTS_DIR)))
    try:
        yield scratch
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def _repo_rel(path: Path) -> str:
    """Путь относительно корня репозитория (posix-форма), как в манифесте."""
    return Path(
        os.path.relpath(os.path.realpath(path), os.path.realpath(REPO_ROOT))
    ).as_posix()


def _generation_args(
    tmp_path: Path, scratch: Path, journal: Path | None = None
) -> tuple[Path, list[str]]:
    """Аргументы валидной генерации (§5.1): хеши прогона, существующий и
    непустой журнал прогона ВНУТРИ репозитория (`scratch`), явные версии
    (детекторы окружения не нужны). `--run-ref` передаётся относительным от
    корня репозитория (§4 п. 6); `--output` — в tmp_path (этот путь в
    манифест не попадает, ограничения §4 п. 6 на него не распространяются).
    Возвращает (путь выходного файла, argv)."""
    if journal is None:
        journal = scratch / "run-journal.jsonl"
        journal.write_text('{"event": "run"}\n', encoding="utf-8")
    output = tmp_path / "out" / "a4-skeleton-run-manifest.json"
    args = [
        "--weights-hash", "a" * 64,
        "--dataset-hash", "b" * 64,
        "--run-ref", _repo_rel(journal),
        "--environment-version", "v1.1",
        "--harness-version", "0.1.4",
        "--git-commit", "c" * 40,
        "--run-date", "2026-09-13",
        "--output", str(output),
    ]
    return output, args


# --- Существующие сценарии (смысл сохранён), переведённые на схему v2 ---


def test_verify_missing_manifest_fails(tmp_path: Path) -> None:
    missing = tmp_path / "no-manifest.json"
    result = _run("--verify", "--manifest", str(missing))
    assert result.returncode != 0
    assert "не найден" in result.stderr or "FAIL" in result.stderr


def test_verify_empty_hash_fails(tmp_path: Path) -> None:
    """Пустой хеш весов -> FAIL: поле обязано быть непустым (контракт пиннинга
    сохраняется из v1, §3 «Обязательные поля v1 сохраняются»)."""
    manifest = _v2_manifest(model_weights_sha256="")
    path = tmp_path / "manifest.json"
    _write(path, manifest)
    result = _run("--verify", "--manifest", str(path))
    assert result.returncode != 0
    assert "model_weights_sha256" in result.stderr


def test_verify_garbage_hash_fails(tmp_path: Path) -> None:
    """Мусор вместо хеша датасета -> FAIL: значение обязано быть в формате
    хеша (пиннинг снапшотов, AD-4)."""
    manifest = _v2_manifest(dataset_sha256="not-a-hash")
    path = tmp_path / "manifest.json"
    _write(path, manifest)
    result = _run("--verify", "--manifest", str(path))
    assert result.returncode != 0
    assert "dataset_sha256" in result.stderr


def test_verify_valid_manifest_passes(tmp_path: Path) -> None:
    """Валидный манифест -> PASS. По v2 это фикстура ПОЛНОГО покрытия: все
    пять стадий stage_set v1 executed с непустым evidence (§4 п. 2)."""
    path = tmp_path / "manifest.json"
    _write(path, _v2_manifest())
    result = _run("--verify", "--manifest", str(path))
    assert result.returncode == 0, result.stderr


def test_generate_without_run_data_fails_and_writes_nothing(tmp_path: Path) -> None:
    output = tmp_path / "out" / "a4-skeleton-run-manifest.json"
    # Без --weights-hash/--dataset-hash/--run-ref: данных прогона нет.
    result = _run("--output", str(output))
    assert result.returncode != 0
    assert not output.exists()
    assert not output.parent.exists() or not any(output.parent.iterdir())


# --- Контрактные тесты дельты ADR-014 (§6, критерии K1–K7) ---


def test_k1_verify_missing_manifest_reports_not_found(tmp_path: Path) -> None:
    """K1: манифеста нет -> --verify код 1 и сообщение «манифест не найден»
    (§4 п. 3: красный гейт читается как план работ)."""
    result = _run("--verify", "--manifest", str(tmp_path / "absent.json"))
    assert result.returncode == 1
    assert "манифест не найден" in result.stderr


@pytest.mark.parametrize("missing", ["weights", "dataset", "run_ref"])
def test_k2_generate_without_required_input_fails_and_writes_nothing(
    tmp_path: Path, repo_scratch: Path, missing: str
) -> None:
    """K2 / §4 п. 4 (запрет фабрикации): генерация без --weights-hash /
    --dataset-hash / --run-ref -> код 1, выходной файл не создан."""
    journal = repo_scratch / "run-journal.jsonl"
    journal.write_text('{"event": "run"}\n', encoding="utf-8")
    output = tmp_path / "out" / "a4-skeleton-run-manifest.json"
    args: list[str] = []
    if missing != "weights":
        args += ["--weights-hash", "a" * 64]
    if missing != "dataset":
        args += ["--dataset-hash", "b" * 64]
    if missing != "run_ref":
        args += ["--run-ref", _repo_rel(journal)]
    args += ["--output", str(output)]
    result = _run(*args)
    assert result.returncode == 1
    assert not output.exists()


def test_k3_partial_coverage_writes_manifest_and_verify_names_uncovered(
    tmp_path: Path, repo_scratch: Path
) -> None:
    """K3: частичное покрытие -> манифест создан, pipeline_complete=false,
    --verify код 1 с «покрытие частично:» и поимённым списком непокрытых."""
    output, args = _generation_args(tmp_path, repo_scratch)
    args += [
        "--stage", "pretrain_checkpoint=executed:checkpoint/tree_hash=ab12cd;steps=1",
        "--stage", "rl_environment=executed:env-run-manifest.json;tasks=10",
        "--stage", "spark_inference=skipped:стенд gb10 занят — операторская стадия",
    ]
    result = _run(*args)
    assert result.returncode == 0, result.stderr
    manifest = json.loads(output.read_text(encoding="utf-8"))
    assert manifest["pipeline_complete"] is False

    verify = _run("--verify", "--manifest", str(output))
    assert verify.returncode == 1
    assert "покрытие частично:" in verify.stderr
    # Поимённый список: skipped-стадия и обе absent-стадии названы,
    # исполненные стадии в список непокрытых не попадают.
    for name in ("spark_inference", "sft", "rl_base_scheme"):
        assert name in verify.stderr
    assert "pretrain_checkpoint" not in verify.stderr
    assert "rl_environment" not in verify.stderr


def test_k4_full_coverage_fixture_verify_passes(tmp_path: Path) -> None:
    """K4: фикстура полного покрытия (пять стадий executed с непустым
    evidence, pipeline_complete=true, схема v2) -> --verify код 0."""
    manifest = _v2_manifest()
    # Контроль самой фикстуры по §3/§4: пять стадий stage_set v1,
    # все executed с evidence, полнота выставлена.
    assert [stage["name"] for stage in manifest["stages"]] == list(STAGE_SET_V1)
    assert all(
        stage["status"] == "executed" and stage["evidence"]
        for stage in manifest["stages"]
    )
    assert manifest["pipeline_complete"] is True
    path = tmp_path / "manifest.json"
    _write(path, manifest)
    result = _run("--verify", "--manifest", str(path))
    assert result.returncode == 0, result.stderr


def test_k5_pipeline_complete_input_is_ignored_and_computed(
    tmp_path: Path, repo_scratch: Path
) -> None:
    """K5 / §4 п. 1: в генерацию передан pipeline_complete=true при неполных
    стадиях -> в ЗАПИСАННОМ файле pipeline_complete=false (поле вычисляется)."""
    output, args = _generation_args(tmp_path, repo_scratch)
    args += [
        "--pipeline-complete",
        "--stage", "pretrain_checkpoint=executed:checkpoint/tree_hash=ab12cd",
    ]
    result = _run(*args)
    assert result.returncode == 0, result.stderr
    manifest = json.loads(output.read_text(encoding="utf-8"))
    assert manifest["pipeline_complete"] is False


def test_k6_v1_schema_rejected_with_expected_v2_message(tmp_path: Path) -> None:
    """K6 / §4 п. 2–3: манифест схемы v1 -> --verify код 1, сообщение про
    ожидаемую v2 (совместимость: v1 невалиден, ADR-014 п. 7)."""
    path = tmp_path / "manifest-v1.json"
    _write(path, MANIFEST_V1)
    result = _run("--verify", "--manifest", str(path))
    assert result.returncode == 1
    assert "схема v1 не поддерживается" in result.stderr
    assert "ожидается v2" in result.stderr


def test_k7_run_ref_nonexistent_or_empty_rejected(
    tmp_path: Path, repo_scratch: Path
) -> None:
    """K7 / §4 п. 5: --run-ref на несуществующий путь И на пустой файл ->
    код 1, манифест не создан (второй барьер против фабрикации). Пути —
    относительные от корня репозитория внутри него: проверяется именно
    барьер существования/непустоты, а не отклонение абсолютного пути
    вне репозитория (это сценарий K11)."""
    missing_journal = repo_scratch / "no-such-journal.jsonl"
    output, args = _generation_args(tmp_path, repo_scratch, journal=missing_journal)
    result = _run(*args)
    assert result.returncode == 1
    assert "run_ref" in result.stderr
    assert not output.exists()

    empty_journal = repo_scratch / "empty-journal.jsonl"
    empty_journal.write_text("", encoding="utf-8")
    output2, args2 = _generation_args(tmp_path, repo_scratch, journal=empty_journal)
    result2 = _run(*args2)
    assert result2.returncode == 1
    assert "run_ref" in result2.stderr
    assert not output2.exists()


# --- Контракт §3 / §5.1: состав стадий stage_set v1 ---


def test_contract_stage_set_v1_exact_names_and_absent_default(
    tmp_path: Path, repo_scratch: Path
) -> None:
    """§3/§5.1: в записанном манифесте ровно пять имён stage_set v1;
    отсутствие --stage даёт всем стадиям статус absent (а не «пропущены
    молча»)."""
    output, args = _generation_args(tmp_path, repo_scratch)
    result = _run(*args)
    assert result.returncode == 0, result.stderr
    manifest = json.loads(output.read_text(encoding="utf-8"))
    stages = manifest["stages"]
    assert len(stages) == 5
    assert [stage["name"] for stage in stages] == list(STAGE_SET_V1)
    for stage in stages:
        assert stage["status"] == "absent"
        assert stage["evidence"] == []
    assert manifest["pipeline_complete"] is False


def test_contract_unknown_stage_name_rejected_without_write(
    tmp_path: Path, repo_scratch: Path
) -> None:
    """§5.1: неизвестное имя стадии -> код 1, диагностика в stderr, файл не
    создан и не изменён."""
    output, args = _generation_args(tmp_path, repo_scratch)
    output.parent.mkdir(parents=True)
    output.write_text("SENTINEL\n", encoding="utf-8")
    args += ["--stage", "quantum_leap=executed:whatever"]
    result = _run(*args)
    assert result.returncode == 1
    assert "quantum_leap" in result.stderr
    assert output.read_text(encoding="utf-8") == "SENTINEL\n"


def test_contract_invalid_stage_status_rejected_without_write(
    tmp_path: Path, repo_scratch: Path
) -> None:
    """§5.1: недопустимый статус стадии (вне executed|skipped|absent) ->
    код 1, файл не создан."""
    output, args = _generation_args(tmp_path, repo_scratch)
    args += ["--stage", "sft=done:sft-log"]
    result = _run(*args)
    assert result.returncode == 1
    assert "sft" in result.stderr
    assert not output.exists()


# --- K11 / §4 п. 6–7: портируемость путей манифеста --------------------------


def test_k11_verify_rejects_absolute_run_ref(tmp_path: Path) -> None:
    """K11 / §4 п. 6–7: манифест с абсолютным путём в `run_ref` -> --verify
    код 1 и сообщение «абсолютный путь запрещён: …». Остальная часть
    манифеста валидна (полное покрытие), так что вердикт — именно о пути,
    в том числе если файл вручную отредактирован после генерации."""
    manifest = _v2_manifest(run_ref="/home/builder/runs/a4/run-journal.json")
    path = tmp_path / "manifest.json"
    _write(path, manifest)
    result = _run("--verify", "--manifest", str(path))
    assert result.returncode == 1
    assert "абсолютный путь запрещён" in result.stderr
    assert "run_ref" in result.stderr
    assert "/home/builder/runs/a4/run-journal.json" in result.stderr


def test_k11_verify_rejects_absolute_path_inside_stage_evidence(tmp_path: Path) -> None:
    """K11 / §4 п. 6–7: абсолютный путь ВНУТРИ строки `stages[].evidence`
    (формат оркестратора `checkpoint=<путь>`) -> --verify код 1. Остальная
    часть манифеста валидна (полное покрытие): вердикт — именно о пути."""
    stages = [_stage(name) for name in STAGE_SET_V1]
    stages[0]["evidence"] = [
        "checkpoint=/var/builds/run-42/checkpoints/pretrain",
        "tree_hash=ab12cd",
        "steps=1",
    ]
    manifest = _v2_manifest(stages=stages)
    path = tmp_path / "manifest.json"
    _write(path, manifest)
    result = _run("--verify", "--manifest", str(path))
    assert result.returncode == 1
    assert "абсолютный путь запрещён" in result.stderr
    assert "/var/builds/run-42/checkpoints/pretrain" in result.stderr


def test_k11_generate_run_ref_absolute_outside_repo_rejected(tmp_path: Path) -> None:
    """K11 / §4 п. 7: генерация с --run-ref абсолютным путём ВНЕ репозитория
    (журнал в tmp_path) -> код 1, манифест не создан."""
    journal = tmp_path / "run-journal.jsonl"
    journal.write_text('{"event": "run"}\n', encoding="utf-8")
    assert journal.is_absolute() and not journal.is_relative_to(REPO_ROOT)
    output = tmp_path / "out" / "a4-skeleton-run-manifest.json"
    args = [
        "--weights-hash", "a" * 64,
        "--dataset-hash", "b" * 64,
        "--run-ref", str(journal),
        "--environment-version", "v1.1",
        "--harness-version", "0.1.4",
        "--git-commit", "c" * 40,
        "--run-date", "2026-09-13",
        "--output", str(output),
    ]
    result = _run(*args)
    assert result.returncode == 1
    assert "run_ref" in result.stderr
    assert not output.exists()


def test_k11_generate_run_ref_absolute_inside_repo_normalized(
    tmp_path: Path, repo_scratch: Path
) -> None:
    """K11 / §4 п. 7: генерация с --run-ref абсолютным путём ВНУТРИ
    репозитория -> код 0, а в записанном манифесте путь относительный
    (не начинается с '/') и резолвится от корня репозитория."""
    journal = repo_scratch / "run-journal.jsonl"
    journal.write_text('{"event": "run"}\n', encoding="utf-8")
    assert journal.is_absolute() and journal.is_relative_to(REPO_ROOT)
    output = tmp_path / "out" / "a4-skeleton-run-manifest.json"
    args = [
        "--weights-hash", "a" * 64,
        "--dataset-hash", "b" * 64,
        "--run-ref", str(journal),
        "--environment-version", "v1.1",
        "--harness-version", "0.1.4",
        "--git-commit", "c" * 40,
        "--run-date", "2026-09-13",
        "--output", str(output),
    ]
    result = _run(*args)
    assert result.returncode == 0, result.stderr
    manifest = json.loads(output.read_text(encoding="utf-8"))
    run_ref = manifest["run_ref"]
    assert not run_ref.startswith("/")
    assert run_ref == _repo_rel(journal)
    resolved = REPO_ROOT / run_ref
    assert resolved.is_file() and resolved.stat().st_size > 0
