"""Пин предмета измерения (ADR-036).

Запись факта без пина предмета не привязывается к утверждению и считается
``unverified``. Здесь собирается то, **на чём** измерено: git sha (и флаг
грязного дерева), sha256 файлов предмета (конфиг, токенизатор, чекпойнт),
ссылки на датасет/прогон, хост и устройство.

Правило: ничего не угадывать. Неизвестное поле — ``None`` (``null`` в JSON),
а не правдоподобная подстановка.
"""

from __future__ import annotations

import hashlib
import os
import socket
import subprocess
from pathlib import Path
from typing import Any, Optional

#: Корень репозитория (файл лежит в ``tools/sensors/``).
REPO_ROOT = Path(__file__).resolve().parents[2]

#: Канонический порядок ключей пина (контракт ADR-036).
SUBJECT_KEYS: tuple[str, ...] = (
    "git_sha",
    "git_dirty",
    "config_sha256",
    "tokenizer_sha256",
    "checkpoint_sha256",
    "dataset_ref",
    "run_ref",
    "host",
    "device_kind",
)


def sha256_file(path: str | os.PathLike[str] | None) -> Optional[str]:
    """sha256 файла по байтам; ``None``, если путь не задан или файла нет."""
    if path is None:
        return None
    try:
        p = Path(path)
        if not p.is_file():
            return None
        digest = hashlib.sha256()
        with p.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return None


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def git_head(repo_root: str | os.PathLike[str] = REPO_ROOT) -> Optional[str]:
    """``git rev-parse HEAD``; ``None``, если git недоступен или это не репозиторий."""
    try:
        proc = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    sha = proc.stdout.strip()
    return sha or None


def git_dirty(repo_root: str | os.PathLike[str] = REPO_ROOT) -> Optional[bool]:
    """Есть ли незакоммиченные изменения (tracked+untracked); ``None`` — неизвестно."""
    try:
        proc = subprocess.run(
            ["git", "-C", str(repo_root), "status", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    return bool(proc.stdout.strip())


def host_name() -> Optional[str]:
    """Имя хоста измерения; ``None``, если не определилось."""
    try:
        name = socket.gethostname()
    except OSError:
        return None
    return name or None


_DEVICE_KIND_CACHE: Any = None


def device_kind(override: Optional[str] = None) -> Optional[str]:
    """Вид устройства измерения: ``cpu`` | ``gpu-модель`` | ``None`` (неизвестно).

    Переопределяется переменной окружения ``AXIOM_DEVICE_KIND`` (для фикстур и
    CPU-прогонов) либо явным аргументом. Иначе спрашивается JAX; недоступный
    JAX — ``None``, а не догадка «cpu».
    """
    global _DEVICE_KIND_CACHE
    if override is not None:
        return override
    env = os.environ.get("AXIOM_DEVICE_KIND")
    if env:
        return env
    if _DEVICE_KIND_CACHE is not None:
        return _DEVICE_KIND_CACHE
    try:  # тяжёлый импорт — только по требованию
        import jax

        kinds = sorted({d.device_kind for d in jax.devices()})
    except Exception:  # noqa: BLE001 — любая ошибка окружения = «неизвестно»
        return None
    _DEVICE_KIND_CACHE = "; ".join(kinds) if kinds else None
    return _DEVICE_KIND_CACHE


def build_subject(
    *,
    repo_root: str | os.PathLike[str] = REPO_ROOT,
    config_path: str | os.PathLike[str] | None = None,
    tokenizer_path: str | os.PathLike[str] | None = None,
    checkpoint_path: str | os.PathLike[str] | None = None,
    dataset_ref: Optional[str] = None,
    run_ref: Optional[str] = None,
    device: Optional[str] = None,
    git_sha: Optional[str] = None,
    dirty: Optional[bool] = None,
) -> dict[str, Any]:
    """Собирает пин предмета. Неизвестные поля — ``None`` (никаких подстановок)."""
    return {
        "git_sha": git_sha if git_sha is not None else git_head(repo_root),
        "git_dirty": dirty if dirty is not None else git_dirty(repo_root),
        "config_sha256": sha256_file(config_path),
        "tokenizer_sha256": sha256_file(tokenizer_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "dataset_ref": dataset_ref,
        "run_ref": run_ref,
        "host": host_name(),
        "device_kind": device_kind(device),
    }


def subject_is_pinned(subject: Any) -> bool:
    """Есть ли у предмета хоть один непустой пин-идентификатор.

    Пустой предмет (все поля ``None``) — запись без предмета: она не
    привязывается к утверждению (контракт ADR-036).
    """
    if not isinstance(subject, dict):
        return False
    return any(subject.get(key) not in (None, "", [], {}) for key in SUBJECT_KEYS)
