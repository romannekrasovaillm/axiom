"""Общие помощники среды: хеширование, дерево, JSON, запуск процессов.

Всё детерминировано: хеши считаются по байтам, обход дерева — отсортированный.
Никаких тяжёлых зависимостей, только стандартная библиотека.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from typing import Iterable, Optional

# sha256 пустой строки — канонический хеш «пустого» holdout-набора (H=0).
EMPTY_HIDDEN_SHA256 = hashlib.sha256(b"").hexdigest()

# Расширения файлов весов, запрещённые в workspace (C-032).
WEIGHT_SUFFIXES = (".safetensors", ".gguf", ".pt", ".pth", ".ckpt")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def is_sha256_hex(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(c in "0123456789abcdef" for c in value)
    )


def tree_sha256(root: Path) -> str:
    """Content-addressed хеш дерева файлов: sha256 над отсортированными
    (относительный путь, sha256 содержимого). Детерминирован от содержимого.
    """
    entries: list[tuple[str, str]] = []
    for p in sorted(root.rglob("*")):
        if p.is_file():
            rel = p.relative_to(root).as_posix()
            entries.append((rel, sha256_file(p)))
    h = hashlib.sha256()
    for rel, ch in entries:
        h.update(rel.encode("utf-8"))
        h.update(b"\0")
        h.update(ch.encode("ascii"))
        h.update(b"\0")
    return h.hexdigest()


def find_weight_files(root: Path) -> list[Path]:
    """Реальные файлы весов в дереве (симлинки не считаются копиями)."""
    out: list[Path] = []
    for p in root.rglob("*"):
        if p.is_file() and not p.is_symlink() and p.suffix.lower() in WEIGHT_SUFFIXES:
            out.append(p)
    return sorted(out)


def read_json(path: Path):
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(obj, ensure_ascii=False, indent=2) + "\n"
    path.write_text(text, encoding="utf-8")


def run_cmd(
    cmd: Iterable[str],
    cwd: Optional[Path] = None,
    timeout: int = 180,
) -> subprocess.CompletedProcess:
    """Запуск процесса с захватом stdout/stderr как текста."""
    return subprocess.run(
        list(cmd),
        cwd=str(cwd) if cwd else None,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def stable_coin(task_seed: int, model_seed: int, salt: str) -> float:
    """Детерминированная монета в [0,1) от (task_seed, model_seed, salt).

    Не зависит от PYTHONHASHSEED и версии интерпретатора — хеш от строки.
    """
    digest = hashlib.sha256(f"{task_seed}:{model_seed}:{salt}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(2**64)


# Каталоги/файлы, исключаемые из снапшота чистого кейса (не архитектура, а
# рантайм среды): сам env/, артефакты evidence/, тяжёлый Archify-HTML, кэши,
# скрытые файлы.
_EXCLUDED_DIRS = {"env", "evidence", "__pycache__"}


def _snapshot_ignored(path: str) -> bool:
    name = Path(path).name
    if name in _EXCLUDED_DIRS:
        return True
    if name.startswith("."):
        return True
    if name.endswith(".pyc"):
        return True
    if name.endswith(".html"):
        return True
    return False


def copy_case_snapshot(src: Path, dst: Path) -> None:
    """Копирует чистый кейс ``src`` в ``dst``, исключая рантайм-каталоги."""
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst, ignore=lambda d, names: [n for n in names if _snapshot_ignored(n)])
