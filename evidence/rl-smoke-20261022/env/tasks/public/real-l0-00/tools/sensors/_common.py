"""Общие помощники датчиков (ADR-037, дельты C2–C6).

Разбор аргументов, безопасная проверка недоступных источников (сетевые
монтирования гб10-shared могут подвиснуть — проверка идёт с таймаутом) и
единая точка записи факта. Импорт ``net``/``env`` подтягивает корень репозитория
в ``sys.path``, чтобы датчик работал при запуске ``python3 -m tools.sensors.<x>``.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Optional

from .fact import DEFAULT_FACTS_DIR, write_fact
from .subject import REPO_ROOT, build_subject, sha256_file

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

#: Канонический путь корпуса претрейна tokens-v2 (ADR-004-амендмент).
TOKENS_V2_ROOT = Path(
    "/home/roman/gb10-shared/datasets/axiom-pretrain-l3/tokens-v2"
)
#: Артефакт канонического токенизатора внутри tokens-v2.
TOKENIZER_ARTIFACT = TOKENS_V2_ROOT / "tokenizer" / "tokenizer.model"


def emit(
    sensor: str,
    fact: str,
    value: Any,
    *,
    unit: str,
    quality: str,
    method: str,
    subject: Any,
    out_dir: Optional[str | Path] = None,
    inputs: Optional[list[dict[str, Any]]] = None,
    status: str = "ok",
    note: str = "",
    raw_ref: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Пишет одну запись факта (тонкая обёртка над :func:`tools.sensors.fact.write_fact`)."""
    return write_fact(
        sensor, fact, value, unit=unit, quality=quality, method=method,
        subject=subject, out_dir=out_dir, inputs=inputs, status=status, note=note,
        raw_ref=raw_ref,
    )


def safe_exists(path: str | Path, timeout: int = 5) -> bool:
    """Существует ли путь; недоступное/подвисшее монтирование → ``False``.

    Проверка идёт отдельным процессом с таймаутом: ``Path.exists()`` на мёртвом
    сетевом монтировании может блокироваться неограниченно.
    """
    try:
        proc = subprocess.run(["test", "-e", str(path)], timeout=timeout)
    except (OSError, subprocess.SubprocessError):
        return False
    return proc.returncode == 0


def config_subject(
    *,
    config_path: Optional[str | Path] = None,
    tokenizer_path: Optional[str | Path] = None,
    checkpoint_path: Optional[str | Path] = None,
    dataset_ref: Optional[str] = None,
    run_ref: Optional[str] = None,
    device: Optional[str] = None,
) -> dict[str, Any]:
    return build_subject(
        config_path=config_path,
        tokenizer_path=tokenizer_path,
        checkpoint_path=checkpoint_path,
        dataset_ref=dataset_ref,
        run_ref=run_ref,
        device=device,
    )


def load_config_json(config_path: str | Path) -> dict[str, Any]:
    """Сырой ``net/config.json`` (метаданные вне схемы ``ModelConfig``).

    ``load_config`` отфильтровывает поля, не входящие в ``ModelConfig``
    (``actual_param_count``, ``active_params_per_token``, ``tokenizer_vocab`` и
    т.п.), поэтому объявленные значения для сверки берутся из JSON напрямую.
    """
    with open(config_path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    return data if isinstance(data, dict) else {}


__all__ = [
    "DEFAULT_FACTS_DIR",
    "REPO_ROOT",
    "TOKENS_V2_ROOT",
    "TOKENIZER_ARTIFACT",
    "emit",
    "safe_exists",
    "sha256_file",
    "config_subject",
    "load_config_json",
]
