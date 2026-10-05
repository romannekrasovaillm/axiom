"""Общие помощники среды: хеширование, дерево, JSON, запуск процессов.

Всё детерминировано: хеши считаются по байтам, обход дерева — отсортированный.
Никаких тяжёлых зависимостей, только стандартная библиотека.
"""

from __future__ import annotations

import hashlib
import json
import os
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


# Каталоги, исключаемые из снапшота чистого кейса на ЛЮБОМ уровне (не
# архитектура, а рантайм среды и тяжёлые артефакты прогонов): сам env/,
# evidence/, тяжёлый Archify-HTML/кэши (см. суффиксы), скрытые файлы и каталоги
# (правило «имя начинается с точки» отсекает .git, .arch-handoff, .pytest_cache).
_SNAPSHOT_EXCLUDED_DIRS = frozenset({"benchmarks", "evidence", "env", "__pycache__"})

# Файлы, исключаемые из снапшота по расширению на любом уровне: веса (C-032),
# бинарные данные (zst/parquet) и кэши/тяжёлый HTML.
_SNAPSHOT_EXCLUDED_SUFFIXES = WEIGHT_SUFFIXES + (".zst", ".parquet", ".pyc", ".html")

# Глобальный кап объёма снапшота воркспейса (§7, §11(8)). Эталонный воркспейс
# ~4 МБ; превышение — дефект генерации (типовой случай — неисключённый data/).
WORKSPACE_CAP_BYTES = 64 * 1024 * 1024


def _snapshot_ignored(src_root: str, current_dir: str, name: str) -> bool:
    """True — запись ``name`` в ``current_dir`` не попадает в снапшот кейса.

    ``src_root`` — нормализованный корень кейса: верхнеуровневый ``data/``
    исключается, вложенные ``data/`` (если появятся) — нет (§7).
    """
    if name.startswith("."):
        return True
    if name in _SNAPSHOT_EXCLUDED_DIRS:
        return True
    if name == "runs" or name.startswith("runs-"):
        return True
    if name == "data" and os.path.normpath(current_dir) == src_root:
        return True
    if Path(name).suffix.lower() in _SNAPSHOT_EXCLUDED_SUFFIXES:
        return True
    return False


def copy_case_snapshot(src: Path, dst: Path) -> None:
    """Копирует чистый кейс ``src`` в ``dst``, исключая тяжёлые/нерантайм-пути.

    Сигнатура и возврат (void) стабильны. Набор копируемых файлов детерминирован
    от содержимого и не зависит от порядка обхода: ``tree_sha256`` снапшота
    воспроизводим (§7, §11(8)).
    """
    if dst.exists():
        shutil.rmtree(dst)
    root = os.path.normpath(os.fspath(src))

    def ignore(directory: str, names: list[str]) -> list[str]:
        d = os.path.normpath(directory)
        return [n for n in names if _snapshot_ignored(root, d, n)]

    shutil.copytree(src, dst, ignore=ignore)


def dir_total_bytes(root: Path) -> int:
    """Суммарный размер обычных файлов дерева (симлинки считаются по ссылке)."""
    total = 0
    for p in root.rglob("*"):
        if p.is_file() and not p.is_symlink():
            total += p.stat().st_size
    return total


def human_bytes(n: int) -> str:
    """Человекочитаемый размер (для диагностических сообщений)."""
    value = float(n)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or unit == "TiB":
            return f"{int(value)} B" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{int(value)} B"


def _top_path_sizes(root: Path, top: int = 5) -> tuple[int, list[tuple[str, int]]]:
    """Объём ``root`` и ``top`` крупнейших непосредственных детей (по имени)."""
    total = 0
    sized: list[tuple[str, int]] = []
    for child in sorted(root.iterdir()):
        if child.is_dir() and not child.is_symlink():
            size = dir_total_bytes(child)
        else:
            try:
                size = child.stat().st_size
            except OSError:
                size = 0
        total += size
        sized.append((child.name, size))
    sized.sort(key=lambda kv: (-kv[1], kv[0]))
    return total, sized[:top]


def workspace_size_cap(ws_dir: Path, cap_bytes: int = WORKSPACE_CAP_BYTES) -> None:
    """Гейт объёма снапшота воркспейса (§7, §11(8)).

    Превышение ``cap_bytes`` → ``ValueError`` с задачей, объёмом и топ-5
    крупнейших путей: это дефект генерации (обычно неисключённый каталог данных),
    а не сбой агента.
    """
    total, top = _top_path_sizes(ws_dir)
    if total <= cap_bytes:
        return
    detail = ", ".join(f"{name} ({human_bytes(size)})" for name, size in top) or "нет"
    raise ValueError(
        f"workspace {Path(ws_dir).as_posix()} (задача {Path(ws_dir).name}) превышает "
        f"кап {human_bytes(cap_bytes)}: {human_bytes(total)}; "
        f"топ-5 крупнейших путей: {detail}"
    )
