#!/usr/bin/env python3
"""Генератор и верификатор манифеста прогона A4 (evidence/a4-skeleton-run-manifest.json).

Манифест — механическое доказательство гейта A4 (conformance): walking skeleton
L3 прошёл конвейер целиком, и снапшоты запиннены (AD-4). Поведенческий страж
C-038 вызывает `python3 tools/a4_manifest.py --verify`.

Схема `a4-skeleton-run-manifest/v2` (ADR-014, docs/specs/A4-RUN.delta.md §3):
манифест несёт состав стадий конвейера (эталонный набор `stage_set v1`),
пиннинг бэкенда и политики точности (ADR-010) и ВЫЧИСЛЯЕМЫЙ признак полноты
`pipeline_complete`. Частичный прогон гейт не закрывает: `--verify` зелёный
только при всех стадиях `executed` с непустым `evidence` и
`pipeline_complete=true`.

Два режима:

* генерация (по умолчанию) — собирает поля манифеста из реального прогона и
  репозитория и атомарно записывает JSON (temp + os.replace). Хеши весов и
  датасета и журнал прогона не берутся «из воздуха»: без них (или при пустом
  `--run-ref`) скрипт обязан завершиться с ненулевым кодом и НИЧЕГО не
  записать. Манифест без прогона — фабрикация доказательства (AD-4).

* `--verify` — проверяет существующий манифест по правилам §4 спеки: схема v2,
  состав стадий совпадает со `stage_set_version`, все стадии `executed` с
  непустым `evidence`, `pipeline_complete=true`. Нулевой код = гейт закрыт;
  ненулевой код сопровождается классифицированным сообщением («манифест не
  найден» / «покрытие частично: …» / «схема v1 не поддерживается» / «абсолютный
  путь запрещён: <поле>: <значение>» / список полей с нарушениями).

Портируемость путей (§4 п. 6–7, K11–K12, ADR-014 п. 8): все пути в манифесте
(`run_ref`, пути внутри `stages[].evidence`) — относительные от корня
репозитория (`git rev-parse --show-toplevel`). Абсолютные пути непортируемы
(worktree флота удаляется после мержа) и выдают личные пути сборочной машины,
поэтому:

* генерация НОРМАЛИЗУЕТ абсолютный путь внутри репозитория в относительный
  (например, `кейсы/axiom/evidence/...`) и ОТКЛОНЯЕТ абсолютный путь вне
  репозитория (код 1, диагностика в stderr, файл не создан); `run_ref` после
  нормализации проверяется на существование/непустоту резолвом от корня
  репозитория, а не от cwd;
* в evidence-строках абсолютный путь — это токен (разделитель — пробел),
  начинающийся с `/`, либо значение после `=` в токене вида `key=/abs/path`
  (формат оркестратора `checkpoint=<путь>`). Выбранная политика: токен внутри
  репозитория нормализуется в относительный путь (префикс `key=` и текст
  вокруг сохраняются), токен вне репозитория — отказ генерации. URL
  (`https://…`) и относительные обозначения с `/` (`gb10/DGX`) абсолютными
  путями не считаются;
* `--verify` отклоняет манифест с ЛЮБЫМ абсолютным путём (рекурсивно по всем
  строкам, включая вручную отредактированные после генерации); ошибки путей
  собираются вместе с вердиктом покрытия, а не подавляют его.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import date
from pathlib import Path
from typing import Any, Optional

MANIFEST_SCHEMA = "a4-skeleton-run-manifest/v2"
MANIFEST_SCHEMA_V1 = "a4-skeleton-run-manifest/v1"
DEFAULT_MANIFEST_PATH = "evidence/a4-skeleton-run-manifest.json"

# Эталонные наборы стадий конвейера A4 (A4-RUN.delta §2). Расширение — только
# новой версией набора со ссылкой на решение; молча дорисовывать запрещено.
STAGE_SETS: dict[str, tuple[str, ...]] = {
    "v1": (
        "pretrain_checkpoint",
        "spark_inference",
        "rl_environment",
        "sft",
        "rl_base_scheme",
    ),
}
DEFAULT_STAGE_SET_VERSION = "v1"
STAGE_STATUSES = ("executed", "skipped", "absent")

# Поля, обязанные быть непустыми и в формате хеша (AD-4: пиннинг снапшотов).
HASH_FIELDS = ("model_weights_sha256", "dataset_sha256")
# Поля пиннинга манифеста прогона, обязанные быть непустыми (контракт A4).
REQUIRED_FIELDS = (
    "model_weights_sha256",
    "dataset_sha256",
    "environment_version",
    "harness_version",
    "git_commit",
    "run_date",
    "run_ref",
    "stage_set_version",
)
# Пиннинг бэкенда (ADR-010): без него вердикт не интерпретируем.
BACKEND_FIELDS = ("platform", "device_kind", "jax_version", "matmul_precision")

# Хеш: md5 (32) / sha1 (40) / sha256 (64) hex. «Формат хеша» — не пустая строка.
HASH_RE = re.compile(r"^[0-9a-fA-F]{32,64}$")


class ManifestError(ValueError):
    """Манифест отсутствует, не JSON, не проходит проверку полей или
    параметры генерации недопустимы."""


def compute_pipeline_complete(stages: list[dict[str, Any]]) -> bool:
    """Полнота ВЫЧИСЛЯЕТСЯ (§4 п.1): true ⇔ все стадии `executed` и у каждой
    непустой `evidence`. Переданное извне значение игнорируется."""
    return all(
        stage.get("status") == "executed" and bool(stage.get("evidence"))
        for stage in stages
    )


def validate_manifest_structure(manifest: Any) -> list[str]:
    """Структурная проверка манифеста (схема v2): поля, типы, состав стадий.
    Возвращает список ошибок (пусто = структура валидна)."""
    if not isinstance(manifest, dict):
        return ["манифест не является JSON-объектом"]

    schema = manifest.get("schema")
    if schema == MANIFEST_SCHEMA_V1:
        return ["схема v1 не поддерживается (ожидается v2)"]
    if schema != MANIFEST_SCHEMA:
        return [f"schema: '{schema}' не поддерживается (ожидается {MANIFEST_SCHEMA})"]

    errs: list[str] = []
    for field in REQUIRED_FIELDS:
        value = manifest.get(field)
        if not isinstance(value, str) or not value.strip():
            errs.append(f"{field}: поле отсутствует или пусто")

    for field in HASH_FIELDS:
        value = manifest.get(field)
        if isinstance(value, str) and value and not HASH_RE.match(value):
            errs.append(f"{field}: '{value}' не в формате хеша (ожидается 32–64 hex)")

    backend = manifest.get("backend")
    if not isinstance(backend, dict):
        errs.append("backend: поле отсутствует или не является объектом")
    else:
        for field in BACKEND_FIELDS:
            value = backend.get(field)
            if not isinstance(value, str) or not value.strip():
                errs.append(f"backend.{field}: поле отсутствует или пусто")

    pipeline_complete = manifest.get("pipeline_complete")
    if not isinstance(pipeline_complete, bool):
        errs.append("pipeline_complete: поле отсутствует или не является bool")

    stage_set_version = manifest.get("stage_set_version")
    stage_set: Optional[tuple[str, ...]] = None
    if isinstance(stage_set_version, str) and stage_set_version.strip():
        stage_set = STAGE_SETS.get(stage_set_version)
        if stage_set is None:
            known = ", ".join(sorted(STAGE_SETS))
            errs.append(
                f"stage_set_version: '{stage_set_version}' неизвестен "
                f"(поддерживаются: {known})"
            )

    stages = manifest.get("stages")
    if stage_set is not None:
        if not isinstance(stages, list) or not all(
            isinstance(stage, dict) for stage in stages
        ):
            errs.append("stages: поле отсутствует или не является списком объектов")
        else:
            names = [stage.get("name") for stage in stages]
            for expected_name in stage_set:
                if expected_name not in names:
                    errs.append(f"stages: стадия '{expected_name}' отсутствует")
            for index, stage in enumerate(stages):
                label = stage.get("name") or f"#{index}"
                name = stage.get("name")
                if name not in stage_set:
                    errs.append(
                        f"stages[{label}]: неизвестная стадия для "
                        f"stage_set {stage_set_version}"
                    )
                if stage.get("status") not in STAGE_STATUSES:
                    errs.append(
                        f"stages[{label}].status: недопустимый статус "
                        f"'{stage.get('status')}' (допустимы: {'|'.join(STAGE_STATUSES)})"
                    )
                evidence = stage.get("evidence")
                if not isinstance(evidence, list) or not all(
                    isinstance(item, str) for item in evidence
                ):
                    errs.append(
                        f"stages[{label}].evidence: обязан быть списком строк"
                    )
            if len(set(names)) != len(names):
                errs.append("stages: дубликаты имён стадий")

    return errs


def verify_manifest_dict(manifest: Any) -> list[str]:
    """Полная проверка манифеста по правилам гейта A4 (§4 п.2): структура v2,
    полное покрытие стадий и `pipeline_complete=true`. Дополнительно (§4 п. 7)
    отклоняется любой абсолютный путь; ошибки путей собираются ВМЕСТЕ с
    вердиктом покрытия, а не вместо него — красный гейт читается как план
    работ целиком. Пустой список = PASS."""
    errs = validate_manifest_structure(manifest)
    if errs:
        return errs

    errs = find_forbidden_absolute_paths(manifest)

    stages = manifest["stages"]
    uncovered = [
        stage["name"]
        for stage in stages
        if stage["status"] != "executed" or not stage["evidence"]
    ]
    if uncovered:
        detail = ", ".join(
            f"{stage['name']} ({stage['status']})"
            for stage in stages
            if stage["name"] in uncovered
        )
        errs.append(f"покрытие частично: {detail}")
        if manifest.get("pipeline_complete") is True:
            errs.append(
                "pipeline_complete: заявлено true при неполном покрытии — "
                "значение недоверенно (полнота вычисляется, а не декларируется)"
            )
        return errs

    if manifest.get("pipeline_complete") is not True:
        errs.append(
            "pipeline_complete: значение не true при полном покрытии "
            "(ожидается вычисленное true)"
        )
    return errs


def verify_manifest_file(path: Path) -> list[str]:
    """Проверяет манифест на диске. Возвращает список ошибок (пусто = валидно)."""
    if not path.is_file():
        return [f"манифест не найден: {path}"]
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return [f"манифест не является валидным JSON: {path}: {exc}"]
    except OSError as exc:
        return [f"не удалось прочитать манифест: {path}: {exc}"]
    return verify_manifest_dict(obj)


def parse_stage_spec(spec: str) -> tuple[str, str, list[str]]:
    """Разбирает `--stage <name>=<status>[:<evidence>]`.

    evidence — строка; несколько следов разделяются `;`.
    Неизвестное имя стадии или недопустимый статус — ManifestError.
    """
    name, sep, rest = spec.partition("=")
    name = name.strip()
    if not sep or not name:
        raise ManifestError(
            f"неверный формат --stage '{spec}': "
            f"ожидается <name>=<status>[:<evidence>]"
        )
    stage_set = STAGE_SETS[DEFAULT_STAGE_SET_VERSION]
    if name not in stage_set:
        raise ManifestError(
            f"неизвестное имя стадии '{name}' "
            f"(stage_set {DEFAULT_STAGE_SET_VERSION}: {', '.join(stage_set)})"
        )
    status, _, evidence_raw = rest.partition(":")
    status = status.strip()
    if status not in STAGE_STATUSES:
        raise ManifestError(
            f"недопустимый статус стадии '{name}': '{status}' "
            f"(допустимы: {'|'.join(STAGE_STATUSES)})"
        )
    evidence = (
        [item.strip() for item in evidence_raw.split(";") if item.strip()]
        if evidence_raw
        else []
    )
    return name, status, evidence


def detect_repo_root(case_dir: Path) -> Optional[Path]:
    """Корень репозитория (`git rev-parse --show-toplevel`) — якорь
    относительных путей манифеста (§4 п. 6). None, если git недоступен
    или cwd вне репозитория."""
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            timeout=30,
            cwd=str(case_dir),
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    root = (proc.stdout or "").strip()
    return Path(root) if root else None


def find_absolute_path_tokens(text: str) -> list[str]:
    """Абсолютные пути в строке манифеста (§4 п. 6).

    Токены разделяются пробелами; абсолютным путём считается токен,
    начинающийся с `/`, либо значение после первого `=` в токене вида
    `key=/abs/path` (формат evidence оркестратора: `checkpoint=<путь>`).
    URL (`https://…`) и относительные обозначения с `/` (`gb10/DGX`)
    не считаются абсолютными путями.
    """
    tokens: list[str] = []
    for token in text.split():
        candidate = token
        if not candidate.startswith("/") and "=" in candidate:
            candidate = candidate.split("=", 1)[1]
        if candidate.startswith("/"):
            tokens.append(candidate)
    return tokens


def normalize_repo_path(value: str, repo_root: Optional[Path], *, field: str) -> str:
    """Приводит путь к относительному от корня репозитория (§4 п. 7).

    Относительный путь возвращается как есть (по контракту он уже от корня
    репозитория, K12). Абсолютный путь внутри репозитория нормализуется в
    относительный (posix-форма); абсолютный путь вне репозитория —
    ManifestError (отказ генерации). Если корень репозитория определить не
    удалось, абсолютный путь отклоняется (fail-closed: портируемость не
    доказана).
    """
    if not value.startswith("/"):
        return value
    if repo_root is None:
        raise ManifestError(
            f"{field}: не удалось определить корень репозитория "
            f"(git rev-parse --show-toplevel), абсолютный путь отклонён: "
            f"'{value}' — пути манифеста обязаны быть относительными "
            f"от корня репозитория (§4 п. 6–7)"
        )
    root_resolved = Path(os.path.realpath(str(repo_root)))
    resolved = Path(os.path.realpath(value))
    try:
        relative = resolved.relative_to(root_resolved)
    except ValueError:
        raise ManifestError(
            f"{field}: абсолютный путь вне репозитория: '{value}' — "
            f"пути манифеста обязаны быть относительными от корня "
            f"репозитория ({root_resolved}), §4 п. 6–7"
        ) from None
    return relative.as_posix()


def normalize_evidence_paths(
    text: str, repo_root: Optional[Path], *, stage_name: str
) -> str:
    """Нормализует абсолютные пути в evidence-строке стадии (§4 п. 7).

    Политика (выбрана и зафиксирована здесь): токен абсолютного пути внутри
    репозитория заменяется на относительный от корня (префикс `key=` и текст
    вокруг сохраняются); токен вне репозитория — ManifestError (отказ
    генерации, код 1, файл не создаётся). Строка без абсолютных путей
    возвращается без изменений.
    """
    tokens = find_absolute_path_tokens(text)
    if not tokens:
        return text
    normalized_tokens: dict[str, str] = {}
    for token in tokens:
        normalized = normalize_repo_path(
            token, repo_root, field=f"stages[{stage_name}].evidence"
        )
        normalized_tokens[token] = normalized
    parts: list[str] = []
    for token in text.split():
        candidate = token
        prefix = ""
        if not candidate.startswith("/") and "=" in candidate:
            prefix, _, candidate = candidate.partition("=")
            prefix += "="
        replacement = normalized_tokens.get(candidate)
        parts.append(prefix + replacement if replacement is not None else token)
    return " ".join(parts)


def find_forbidden_absolute_paths(manifest: Any) -> list[str]:
    """Рекурсивно ищет абсолютные пути во всех строках манифеста (§4 п. 7):
    `run_ref`, `stages[].evidence` и любые другие строковые поля, включая
    вручную отредактированные после генерации. Полей, где ведущий `/`
    легален по формату, в схеме v2 нет."""

    def iter_strings(obj: Any, path: str) -> Any:
        if isinstance(obj, str):
            yield path, obj
        elif isinstance(obj, dict):
            for key, value in obj.items():
                yield from iter_strings(value, f"{path}.{key}" if path else key)
        elif isinstance(obj, list):
            for index, value in enumerate(obj):
                yield from iter_strings(value, f"{path}[{index}]")

    errs: list[str] = []
    for field, text in iter_strings(manifest, ""):
        for token in find_absolute_path_tokens(text):
            errs.append(f"абсолютный путь запрещён: {field}: {token}")
    return errs


def check_run_ref(run_ref: str, base_dir: Path) -> Optional[str]:
    """Журнал прогона обязан существовать и быть непустым (§4 п.5):
    файл нулевого размера и каталог без файлов отклоняются.
    Относительный `run_ref` резолвится от корня репозитория (`base_dir`),
    а не от cwd (K12). Возвращает текст ошибки или None."""
    path = Path(run_ref)
    if not path.is_absolute():
        path = base_dir / path
    if not path.exists():
        return f"run_ref указывает на несуществующий путь: {run_ref}"
    if path.is_file():
        try:
            if path.stat().st_size == 0:
                return f"run_ref указывает на пустой файл (нулевой размер): {run_ref}"
        except OSError as exc:
            return f"не удалось прочитать run_ref: {run_ref}: {exc}"
        return None
    if path.is_dir():
        try:
            has_files = any(entry.is_file() for entry in path.rglob("*"))
        except OSError as exc:
            return f"не удалось прочитать run_ref: {run_ref}: {exc}"
        if not has_files:
            return f"run_ref указывает на каталог без файлов: {run_ref}"
        return None
    return f"run_ref указывает на неподдерживаемый тип пути: {run_ref}"


def parse_environment_version(case_dir: Path) -> Optional[str]:
    """Извлекает версию среды из docs/specs/ENVIRONMENT-V1.md.

    Ожидаемый вид строки: `- **Версия:** v1.1 (12.09.2026, ...)`.
    """
    spec = case_dir / "docs" / "specs" / "ENVIRONMENT-V1.md"
    if not spec.is_file():
        return None
    text = spec.read_text(encoding="utf-8")
    match = re.search(r"\*\*Версия:\*\*\s*([^\s()]+)", text)
    if not match:
        return None
    return match.group(1).strip()


def detect_harness_version() -> Optional[str]:
    """Версия агентного харнесса из `arch-ml --version` (вывод `arch-ml 0.1.4`)."""
    try:
        proc = subprocess.run(
            ["arch-ml", "--version"],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    match = re.search(r"\d+\.\d+\.\d+(?:[-+][0-9A-Za-z.-]+)?", proc.stdout or "")
    return match.group(0) if match else None


def detect_git_commit(case_dir: Path) -> Optional[str]:
    """git-коммит репозитория на момент прогона (`git rev-parse HEAD`)."""
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=30,
            cwd=str(case_dir),
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    commit = (proc.stdout or "").strip()
    return commit or None


def detect_backend() -> tuple[dict[str, str], Optional[str]]:
    """Пиннинг бэкенда и политики точности (ADR-010), детект из окружения.

    JAX необязателен: при его отсутствии (или сбое инициализации) генерация
    НЕ падает — поля заполняются по `sys.platform` с пометкой в значениях,
    а пометка возвращается вторым значением для вывода в stderr/evidence.
    """
    try:
        import jax  # type: ignore[import-not-found]
    except Exception:
        note = f"jax не установлен; backend заполнен по sys.platform={sys.platform}"
        return (
            {
                "platform": sys.platform,
                "device_kind": f"unknown ({note})",
                "jax_version": f"unavailable ({note})",
                "matmul_precision": f"unknown ({note})",
            },
            note,
        )
    try:
        devices = jax.devices()
    except Exception as exc:
        note = (
            f"jax {getattr(jax, '__version__', '?')} без рабочего бэкенда "
            f"({exc}); backend заполнен по sys.platform={sys.platform}"
        )
        return (
            {
                "platform": sys.platform,
                "device_kind": f"unknown ({note})",
                "jax_version": str(getattr(jax, "__version__", "unknown")),
                "matmul_precision": f"unknown ({note})",
            },
            note,
        )

    if devices:
        device = devices[0]
        platform = str(getattr(device, "platform", "unknown"))
        device_kind = str(getattr(device, "device_kind", None) or device)
    else:
        platform = "unknown"
        device_kind = "unknown (jax.devices() пуст)"
    try:
        precision = getattr(jax.config, "jax_default_matmul_precision", None)
    except Exception:
        precision = None
    return (
        {
            "platform": platform,
            "device_kind": device_kind,
            "jax_version": str(getattr(jax, "__version__", "unknown")),
            "matmul_precision": str(precision) if precision else "default",
        },
        None,
    )


def first_nonempty(*values: Optional[str]) -> Optional[str]:
    """Первое непустое значение (аргумент → окружение → детектор)."""
    for value in values:
        if value and value.strip():
            return value.strip()
    return None


def write_manifest_atomic(path: Path, manifest: dict) -> None:
    """Атомарно пишет JSON (temp + os.replace); не оставляет частичных файлов."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    try:
        tmp.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass


def cmd_generate(args: argparse.Namespace) -> int:
    """Генерация манифеста. При отсутствии данных прогона или недопустимых
    параметрах — ненулевой код, файл не создаётся и не изменяется."""
    case_dir = Path.cwd()
    output = Path(args.output or DEFAULT_MANIFEST_PATH)
    # Якорь относительных путей манифеста (§4 п. 6): корень репозитория.
    repo_root = detect_repo_root(case_dir)

    # Стадии конвейера: непереданные получают статус absent (не «пропущены молча»).
    stages_by_name: dict[str, dict[str, Any]] = {}
    for spec in args.stage or []:
        try:
            name, status, evidence = parse_stage_spec(spec)
            evidence = [
                normalize_evidence_paths(item, repo_root, stage_name=name)
                for item in evidence
            ]
        except ManifestError as exc:
            print(f"ошибка: {exc}. Манифест не записан.", file=sys.stderr)
            return 1
        if status == "executed" and not evidence:
            print(
                f"предупреждение: стадия '{name}' executed без evidence — "
                f"в покрытие не засчитывается.",
                file=sys.stderr,
            )
        stages_by_name[name] = {"name": name, "status": status, "evidence": evidence}
    stages = [
        stages_by_name.get(
            name, {"name": name, "status": "absent", "evidence": []}
        )
        for name in STAGE_SETS[DEFAULT_STAGE_SET_VERSION]
    ]

    # Неизвлекаемые из репозитория данные прогона (AD-4): хеши весов и датасета.
    weights_hash = first_nonempty(args.weights_hash, os.environ.get("A4_WEIGHTS_SHA256"))
    dataset_hash = first_nonempty(args.dataset_hash, os.environ.get("A4_DATASET_SHA256"))
    run_ref = first_nonempty(args.run_ref, os.environ.get("A4_RUN_REF"))

    if not weights_hash:
        print(
            "ошибка: нет хеша весов модели — данные прогона отсутствуют, "
            "манифест не создан (фабрикация доказательства запрещена, AD-4). "
            "Передайте --weights-hash или A4_WEIGHTS_SHA256.",
            file=sys.stderr,
        )
        return 1
    if not dataset_hash:
        print(
            "ошибка: нет хеша датасета — данные прогона отсутствуют, "
            "манифест не создан (фабрикация доказательства запрещена, AD-4). "
            "Передайте --dataset-hash или A4_DATASET_SHA256.",
            file=sys.stderr,
        )
        return 1
    if not run_ref:
        print(
            "ошибка: нет ссылки на прогон — данные прогона отсутствуют, "
            "манифест не создан. Передайте --run-ref или A4_RUN_REF.",
            file=sys.stderr,
        )
        return 1

    # Портируемость пути (§4 п. 6–7): абсолютный run_ref внутри репозитория
    # нормализуется в относительный от корня, вне репозитория — отказ.
    try:
        run_ref = normalize_repo_path(run_ref, repo_root, field="run_ref")
    except ManifestError as exc:
        print(f"ошибка: {exc}. Манифест не записан.", file=sys.stderr)
        return 1

    # Второй барьер фабрикации: журнал прогона существует и непуст (§4 п.5).
    # Относительный путь резолвится от корня репозитория, а не от cwd (K12).
    run_ref_error = check_run_ref(run_ref, repo_root or case_dir)
    if run_ref_error:
        print(f"ошибка: {run_ref_error}. Манифест не записан.", file=sys.stderr)
        return 1

    environment_version = first_nonempty(
        args.environment_version,
        os.environ.get("A4_ENVIRONMENT_VERSION"),
        parse_environment_version(case_dir),
    )
    harness_version = first_nonempty(
        args.harness_version,
        os.environ.get("A4_HARNESS_VERSION"),
        detect_harness_version(),
    )
    git_commit = first_nonempty(
        args.git_commit,
        os.environ.get("A4_GIT_COMMIT"),
        detect_git_commit(case_dir),
    )
    run_date = first_nonempty(
        args.run_date,
        os.environ.get("A4_RUN_DATE"),
        date.today().isoformat(),
    )

    if not environment_version:
        print(
            "ошибка: не удалось извлечь версию среды из docs/specs/ENVIRONMENT-V1.md "
            "и она не передана явно (--environment-version).",
            file=sys.stderr,
        )
        return 1
    if not harness_version:
        print(
            "ошибка: не удалось определить версию агентного харнесса "
            "(arch-ml --version) и она не передана явно (--harness-version).",
            file=sys.stderr,
        )
        return 1
    if not git_commit:
        print(
            "ошибка: не удалось определить git-коммит репозитория (git rev-parse HEAD) "
            "и он не передан явно (--git-commit).",
            file=sys.stderr,
        )
        return 1

    backend, backend_note = detect_backend()
    if backend_note:
        print(f"предупреждение: {backend_note}.", file=sys.stderr)

    if args.pipeline_complete or os.environ.get("A4_PIPELINE_COMPLETE"):
        print(
            "примечание: pipeline_complete вычисляется генератором по стадиям; "
            "переданное значение игнорируется (§4 п.1).",
            file=sys.stderr,
        )

    manifest = {
        "schema": MANIFEST_SCHEMA,
        "stage_set_version": DEFAULT_STAGE_SET_VERSION,
        "model_weights_sha256": weights_hash,
        "dataset_sha256": dataset_hash,
        "environment_version": environment_version,
        "harness_version": harness_version,
        "git_commit": git_commit,
        "run_date": run_date,
        "run_ref": run_ref,
        "backend": backend,
        "stages": stages,
        "pipeline_complete": compute_pipeline_complete(stages),
        "generated_by": "tools/a4_manifest.py",
    }

    # Самопроверка структуры и портируемости путей перед записью (полнота
    # гейтом проверяется отдельно: частичный прогон легален и фиксируется
    # с pipeline_complete=false). Абсолютных путей здесь быть не должно по
    # построению (нормализация выше) — проверка страхует от регресса.
    errs = validate_manifest_structure(manifest) + find_forbidden_absolute_paths(manifest)
    if errs:
        print("ошибка: собранный манифест структурно не валиден:", file=sys.stderr)
        for err in errs:
            print(f"  - {err}", file=sys.stderr)
        print("манифест не записан.", file=sys.stderr)
        return 1

    write_manifest_atomic(output, manifest)
    print(f"манифест записан: {output}")
    if not manifest["pipeline_complete"]:
        uncovered = ", ".join(
            stage["name"]
            for stage in stages
            if stage["status"] != "executed" or not stage["evidence"]
        )
        print(
            f"покрытие частично ({uncovered}): pipeline_complete=false, "
            f"гейт A4 остаётся красным (штатно, ADR-014).",
            file=sys.stderr,
        )
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    """Верификация существующего манифеста. Нулевой код = полное покрытие."""
    path = Path(args.manifest or DEFAULT_MANIFEST_PATH)
    errs = verify_manifest_file(path)
    if errs:
        print("верификация манифеста A4: FAIL", file=sys.stderr)
        for err in errs:
            print(f"  - {err}", file=sys.stderr)
        return 1
    print(f"верификация манифеста A4: PASS ({path})")
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Генератор и верификатор манифеста прогона A4 (AD-4, C-038, схема v2).",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="проверить существующий манифест (по умолчанию — генерация)",
    )
    parser.add_argument(
        "--manifest",
        default=None,
        help="путь к проверяемому манифесту (режим --verify)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="путь для записи манифеста (режим генерации)",
    )
    parser.add_argument("--weights-hash", default=None, help="хеш весов модели (sha256)")
    parser.add_argument("--dataset-hash", default=None, help="хеш датасета (sha256)")
    parser.add_argument("--environment-version", default=None, help="версия среды (по умолчанию из docs/specs)")
    parser.add_argument("--harness-version", default=None, help="версия агентного харнесса (по умолчанию arch-ml --version)")
    parser.add_argument("--git-commit", default=None, help="git-коммит (по умолчанию git rev-parse HEAD)")
    parser.add_argument("--run-date", default=None, help="дата прогона (по умолчанию сегодня)")
    parser.add_argument(
        "--run-ref",
        default=None,
        help=(
            "путь к журналу прогона (обязателен, существующий и непустой; "
            "абсолютный путь внутри репозитория нормализуется в относительный "
            "от его корня, вне репозитория — отказ, §4 п. 6–7)"
        ),
    )
    parser.add_argument(
        "--stage",
        action="append",
        default=None,
        metavar="NAME=STATUS[:EVIDENCE]",
        help=(
            "стадия конвейера (повторяемый): NAME из stage_set "
            f"{DEFAULT_STAGE_SET_VERSION} ({', '.join(STAGE_SETS[DEFAULT_STAGE_SET_VERSION])}); "
            "STATUS = executed|skipped|absent; следы EVIDENCE через ';'. "
            "Непереданные стадии получают статус absent. Абсолютные пути в "
            "EVIDENCE нормализуются/отклоняются по §4 п. 7."
        ),
    )
    parser.add_argument(
        "--pipeline-complete",
        action="store_true",
        help="ИГНОРИРУЕТСЯ: pipeline_complete вычисляется генератором (§4 п.1).",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.verify:
        return cmd_verify(args)
    return cmd_generate(args)


if __name__ == "__main__":
    sys.exit(main())
