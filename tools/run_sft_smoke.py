#!/usr/bin/env python3
"""Стадия ``sft`` конвейера A4 — цикл SFT-обучения и машинный след стадии.

Реализация ADR-005 п. 1 (SFT → RL по схеме базы → MOPD) в объёме дельты
``docs/specs/SFT-STAGE.delta.md``.  Прогон по умолчанию — **смоук**: короткое
окно (``--steps`` <= 200), доказательство провода «данные → токенизация →
упаковка → шаг оптимизатора → чекпойнт → журнал», а не обучение модели.
Масштаб (шаги, токены, часы, размер пула документов) пишется в журнал явно —
подмена смоука закрытием стадии запрещена (спека §5).

Что делает прогон (нумерация — по спеке §3):

1. **Данные.** Потоково читает выбранный jsonl: sha256 содержимого
   (``net.data.shard_hash`` — блоками, не в память целиком), число строк,
   байты, состав по ``source``.  Путь обязан быть **симлинком** на канонический
   сетевой диск ``~/gb10-shared`` (C-032/C-033, решение владельца 12.09);
   копия в рабочем каталоге — отказ до всякого обучения.
2. **Карточка датасета** (ADR-004, ADR-005 п. 1): имя, домен, источник, лицензия
   каждого блока, дата, sha256 — в форме, читаемой ``net.data.DatasetCard``
   (``card_to_datacard``).
3. **Подготовка.** Токенизация пиннутым каноническим BPE (``net/tokenizer.py``,
   хеш сверяется с ``net/config.json:tokenizer_hash``), упаковка в
   последовательности 8K (``net.data.pack_sequence``), детерминированный шаффл
   пула документов пиннутым сидом (AD-11: сид и версия в журнале).
4. **Цикл.** N шагов на скелете L3 (``net/model.py``), bf16, существующий
   оптимизатор (``net/optimizer.py``: Per-Head Muon + AdamW, cosine + 1%
   warmup).  QAT: fake-quant MXFP4 весов включается со стадии SFT (ADR-005 п. 7,
   ``net/quant.py``) — по умолчанию в смоуке ``on``.
   Конфиг — ``--model-preset``: ``small``/``tiny`` (численные смоук-конфиги
   ``net/tests/conftest.py``, заморожены) либо ``l3-full`` — полный скелет,
   прочитанный из декларативного ``net/config.json`` (AD-9/C-035: конфиг
   первичен), с печатью и записью в журнал оценки параметров и памяти
   состояния до старта обучения.
5. **Чекпойнт.** Orbax (``net/checkpoint.py``) + ``tree_hash``; канонически —
   на ``~/gb10-shared`` (C-032), в рабочем каталоге симлинк; round-trip
   проверяется сразу (хеш после восстановления совпадает).
6. **След стадии.** ``evidence/a4-run-wire/sft/stage-journal.json``: пути ТОЛЬКО
   относительные от корня репозитория (ADR-014 п. 8) либо в форме ``~/…`` для
   канонического диска — абсолютных путей в артефакте нет.
7. **Стоимость.** Печать фактических GPU-часов и сравнение со сметой
   (AD-8/C-041), если смета ``evidence/budget/<run-ref>.json`` есть; её
   отсутствие для смоука фиксируется как факт, а не замалчивается.

Запуск (гейтовый профиль, GPU обязателен; см. net/README.md):

    export LD_LIBRARY_PATH=$(ls -d ~/venv-axiom/lib/python3.11/site-packages/nvidia/*/lib | tr '\n' ':')
    NET_GATE_PROFILE=1 ~/venv-axiom/bin/python tools/run_sft_smoke.py \\
        --data data/datasets/sft_train_v12.jsonl \\
        --out evidence/a4-run-wire/sft --steps 200

Выход: ``0`` — стадия исполнена и лосс упал (``loss_first > loss_last``);
``1`` — стадия не исполнена (отказ данных, нет падения лосса, ошибка) —
журнал при этом пишется честно, с фактическим статусом и причиной.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.util
import json
import os
import platform
import random
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

CASE_DIR = Path(__file__).resolve().parent.parent
if str(CASE_DIR) not in sys.path:
    sys.path.insert(0, str(CASE_DIR))

#: Схема журнала стадии.
JOURNAL_SCHEMA = "sft-stage-journal/v1"

#: Каталог журнала стадии по умолчанию (относительно корня репозитория).
DEFAULT_OUT = "evidence/a4-run-wire/sft"

#: Симлинк на выбранный набор по умолчанию (симлинки — C-032/C-033).
DEFAULT_DATA = "data/datasets/sft_train_v12.jsonl"

#: Канонический сетевой диск (net/data.py:GB10_SHARED — та же переменная).
SHARED_ROOT_ENV = "GB10_SHARED"

#: Каталог канонического хранения чекпойнтов стадии на сетевом диске.
CHECKPOINT_SUBDIR = "checkpoints/sft-smoke"

#: Численные пресеты смоук-конфигов приёмки сети (``net/tests/conftest.py``).
#: Заморожены: их поля сверяются тестами дельты ``l3-full`` и не меняются.
SMOKE_PRESETS = ("small", "tiny")

#: Пресет полного конфига скелета L3: не уменьшенная копия, а сам скелет,
#: прочитанный из декларативного ``net/config.json``.
L3_FULL_PRESET = "l3-full"

#: Декларативный конфиг скелета L3 — источник истины архитектуры сети
#: (AD-9/C-035: конфиг первичен, меняется параметром файла, а не кодом).
L3_CONFIG_PATH = CASE_DIR / "net" / "config.json"

#: Оценка памяти состояния обучения на ОДИН параметр (шаг 3 дельты l3-full).
#: Оценка, не замер: bf16-веса (2 Б) + fp32-мастер (4) + Muon-momentum/Adam m
#: (4) + Adam v (4) + bf16-градиенты (2).  Замер потребовал бы аллокации —
#: ровно той, чью влезаемость пресет и проверяет.
STATE_BYTES_PER_PARAM = 16

#: Гибибайт — единица оценок памяти в журнале.
GIB = 1 << 30

#: Лицензии блоков по ``source`` (ADR-004: лицензия каждого блока).
#: Объявлены владельцем данных; источник сведений — ``~/gb10-shared/datasets/
#: OXALPHA_HANDOFF.md`` (24.08.2026) и ``~/gb10-shared/build_sft_v*.py``.
#: Блок, которого здесь нет, получает «не установлена» — карточка не выдумывает
#: лицензию (спека §5: не выдумывать датасет).
BLOCK_LICENSES: dict[str, tuple[str, str]] = {
    # source-prefix: (license, provenance)
    "teacher_oxalpha_reasoning": (
        "не заявлена (синтетический вывод сторонней модели-учителя; "
        "внутреннее экспериментальное использование)",
        "teacher traces stealth/ox-alpha через OpenRouter, OXALPHA_HANDOFF.md 24.08.2026",
    ),
    "teacher_oxalpha_multiturn": (
        "не заявлена (синтетический вывод сторонней модели-учителя; "
        "внутреннее экспериментальное использование)",
        "teacher traces stealth/ox-alpha (мультитёрн), OXALPHA_HANDOFF.md 24.08.2026",
    ),
    "oxalpha_reasoning": (
        "не заявлена (синтетика того же пула, отфильтрованная RL-промптами)",
        "фильтр утечки build_sft_v10.py по RL-пулу",
    ),
    "oxalpha_multiturn": (
        "не заявлена (синтетика того же пула, отфильтрованная RL-промптами)",
        "фильтр утечки build_sft_v10.py по RL-пулу",
    ),
    "teacher_native_xhigh": (
        "не заявлена (синтетический вывод модели-учителя Qwen3.8-27B)",
        "teacher_reasoning_v10 (нативный xhigh), build_sft_v10.py",
    ),
    "teacher_multiturn_xhigh": (
        "не заявлена (синтетический вывод модели-учителя Qwen3.8-27B)",
        "teacher_multiturn_v10 (LLM-based), build_sft_v10.py",
    ),
    "v4_": (
        "generated (собственные прогоны, self-distillation)",
        "self-distill Qwen2.5-15B (s1337/s2024/s42/s777/s31415), build_sft_v4.py",
    ),
    "v10_base": (
        "не установлена (в записи о сборке происхождение не объявлено)",
        "базовые примеры sft_train_v3, сохранённые build_sft_v10.py",
    ),
}

#: Какой набор брать по умолчанию — решено карточкой (спека §6), фиксируется
#: в журнале.  v12 = v10.1 + v11 (build_sft_v12.sh): свежее по дате сборки и
#: полнее по составу (траекторный мультитёрн-блок присутствует, 7.4k+2.3k
#: примеров против отсутствующего в v10.1 блока v11), поэтому по умолчанию v12.
DATASET_CHOICE_NOTE = (
    "sft_train_v12.jsonl = sft_train_v10.1.jsonl + sft_train_v11.jsonl "
    "(build_sft_v12.sh, 28.08.2026): свежее v10.1 (25.08) и полнее по составу "
    "(мультитёрн-траектории агентного цикла). Выбор зафиксирован карточкой."
)

#: Максимальное число строк, читаемых для карточки состава (полный проход —
#: 44 949 строк; ограничение снимается ``--card-scan-lines 0``).
DEFAULT_CARD_SCAN_LINES = 200_000


class StageError(RuntimeError):
    """Стадия не исполнена: данные недопустимы или инвариант нарушен."""


# ---------------------------------------------------------------------------
# Окружение и пути
# ---------------------------------------------------------------------------


def detect_repo_root(cwd: Optional[Path] = None) -> Optional[Path]:
    """Корень репозитория (``git rev-parse --show-toplevel``), иначе None.

    Пути журнала относительны от него (ADR-014 п. 8); в git-worktree это
    корень worktree.  Пути аргументов командной строки, наоборот, резолвятся
    от каталога кейса (``CASE_DIR``) — как в остальных инструментах кейса.
    """
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            capture_output=True, text=True, timeout=30,
            cwd=str(cwd or CASE_DIR),
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    root = (proc.stdout or "").strip()
    return Path(root).resolve() if root else None


def case_path(value: str) -> Path:
    """Путь аргумента: абсолютный как есть, относительный — от каталога кейса."""
    path = Path(value)
    return path if path.is_absolute() else (CASE_DIR / path)


def repo_rel(path: Path, repo_root: Optional[Path]) -> str:
    """Путь относительно корня репозитория; вне репозитория — ``~/…`` форма.

    Абсолютный путь в журнал не попадает никогда (ADR-014 п. 8, спека §4.5):
    путь вне репозитория, лежащий на каноническом диске, записывается от
    домашнего каталога (``~/gb10-shared/…``) — это не абсолютный путь.
    """
    resolved = Path(path)
    try:
        resolved = resolved.resolve()
    except OSError:
        pass
    if repo_root is not None:
        try:
            return resolved.relative_to(repo_root).as_posix()
        except ValueError:
            pass
    home = Path.home()
    try:
        return "~/" + resolved.relative_to(home).as_posix()
    except ValueError:
        return str(resolved)


def repo_rel_link(path: Path, repo_root: Optional[Path]) -> str:
    """Путь симлинка БЕЗ разыменования: рабочая точка монтирования (C-032).

    ``Path.resolve()`` увёл бы симлинк на канонический диск и потерял бы адрес
    монтирования в рабочем каталоге; журналу нужны оба (монтирование и цель).
    """
    absolute = Path(os.path.abspath(str(path)))
    if repo_root is not None:
        try:
            return absolute.relative_to(repo_root).as_posix()
        except ValueError:
            pass
    try:
        return "~/" + absolute.relative_to(Path.home()).as_posix()
    except ValueError:
        return absolute.as_posix()


def shared_root() -> Path:
    """Канонический сетевой диск (``GB10_SHARED`` или ``~/gb10-shared``)."""
    return Path(os.environ.get(SHARED_ROOT_ENV) or str(Path.home() / "gb10-shared"))


def _load_acceptance_conftest():
    """Загрузить ``net/tests/conftest.py`` (ADR-010: единая точка пиннинга).

    Пиннинг бэкенда и политики точности обязан применяться **до** импорта
    ``net.*`` (тот тянет jax): ``conftest`` предзагружает ``nvidia-*``
    библиотеки, ставит ``XLA_FLAGS`` детерминизма в гейтовом профиле (ADR-013)
    и пинует ``jax_default_matmul_precision``.  Дублировать это здесь значило
    бы завести вторую точку пиннинга.
    """
    spec = importlib.util.spec_from_file_location(
        "net_tests_conftest_for_sft", CASE_DIR / "net" / "tests" / "conftest.py"
    )
    if spec is None or spec.loader is None:  # pragma: no cover — сломанная установка
        raise StageError("conftest приёмки не найден: нет net/tests/conftest.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# 1. Данные: проверка пути и карточка
# ---------------------------------------------------------------------------


def validate_data_path(
    path: Path, shared: Optional[Path] = None, repo_root: Optional[Path] = None
) -> dict[str, Any]:
    """Путь к набору — симлинк на канонический сетевой диск.

    Копия (обычный файл) в рабочем каталоге — отказ до всякого обучения:
    C-032/C-033, решение владельца 12.09.  Возвращает описание симлинка для
    журнала (все значения — без абсолютных путей).
    """
    path = Path(path)
    repo_root = repo_root if repo_root is not None else detect_repo_root()
    if not path.exists():
        raise StageError(f"набор не найден: {repo_rel(path, repo_root)}")
    if not path.is_symlink():
        raise StageError(
            "путь к датасету не симлинк, а копия в рабочем каталоге: "
            "копии запрещены (C-032/C-033), набор монтируется симлинком на "
            "~/gb10-shared"
        )
    shared = Path(shared) if shared is not None else shared_root()
    try:
        target = Path(os.readlink(path))
    except OSError as exc:  # pragma: no cover — гонка с удалением
        raise StageError(f"симлинк не читается: {exc}") from exc
    resolved = target if target.is_absolute() else (path.parent / target)
    try:
        resolved = resolved.resolve()
    except OSError as exc:
        raise StageError(f"симлинк не разрешается: {exc}") from exc
    try:
        shared_resolved = shared.resolve()
    except OSError:
        shared_resolved = shared
    if not resolved.exists():
        raise StageError(
            "симлинк ведёт на несуществующий файл: цель не найдена "
            f"({repo_rel(resolved, repo_root)})"
        )
    if shared_resolved not in resolved.parents and resolved != shared_resolved:
        raise StageError(
            "симлинк ведёт вне канонического диска ~/gb10-shared: "
            f"{repo_rel(resolved, repo_root)} (C-032/C-033)"
        )
    return {
        "path": repo_rel_link(path, repo_root),
        "is_symlink": True,
        "target": repo_rel(resolved, repo_root),
        "shared_root": repo_rel(shared_resolved, repo_root),
        "kind": "dir" if resolved.is_dir() else "file",
        "size_bytes": resolved.stat().st_size if resolved.is_file() else None,
    }


def _source_license(source: str) -> tuple[str, str]:
    """Лицензия и происхождение блока по имени ``source`` (иначе — честно нет)."""
    for prefix, (license_, provenance) in BLOCK_LICENSES.items():
        if source.startswith(prefix):
            return license_, provenance
    return (
        "не установлена (источник не объявлен в реестре лицензий карточки)",
        "происхождение в записи о сборке не объявлено",
    )


def build_dataset_card(
    path: Path,
    *,
    scan_lines: int = DEFAULT_CARD_SCAN_LINES,
    provenance_note: str = "",
) -> dict[str, Any]:
    """Карточка набора: sha256 содержимого (потоково), состав, лицензии, дата.

    Форма совместима с ``net.data.DatasetCard`` (``card_to_datacard`` отбирает
    поля dataclass'а): пять полей карточки плюс машинные подробности
    (``blocks``, ``lines``, ``bytes``, ``path``, ``date``).
    """
    from net.data import shard_hash

    path = Path(path)
    resolved = path.resolve()
    digest = shard_hash(resolved)  # потоковый sha256, без чтения в память
    blocks: dict[str, int] = {}
    lines = 0
    total_bytes = 0
    scanned = 0
    with open(resolved, "rb") as raw:
        for line in raw:
            lines += 1
            total_bytes += len(line)
            if scan_lines and lines > scan_lines:
                continue
            scanned = lines
            stripped = line.strip()
            if not stripped:
                continue
            try:
                doc = json.loads(stripped)
            except json.JSONDecodeError:
                continue
            source = doc.get("source") or "<не объявлен>"
            blocks[source] = blocks.get(source, 0) + 1
    if not scan_lines:
        scanned = lines
    truncated = scanned < lines

    mtime = datetime.fromtimestamp(resolved.stat().st_mtime, tz=timezone.utc)
    block_entries = []
    for source, count in sorted(blocks.items(), key=lambda kv: (-kv[1], kv[0])):
        license_, provenance = _source_license(source)
        block_entries.append({
            "source": source,
            "examples": count,
            "license": license_,
            "provenance": provenance,
        })
    licenses = sorted({entry["license"] for entry in block_entries})
    summary = (
        "смешанный: " + "; ".join(licenses)
        if len(licenses) > 1
        else (licenses[0] if licenses else "не установлена")
    )
    if truncated:
        summary += f" [состав по первым {scanned} строкам из {lines}]"
    return {
        # --- поля net.data.DatasetCard ------------------------------------
        "name": resolved.stem,
        "domain": "sft_messages",
        "source": provenance_note or (
            "gb10-shared/datasets/" + resolved.name
        ),
        "license": summary,
        "hash": digest,
        # --- машинные подробности ------------------------------------------
        "sha256": digest,
        "blocks": block_entries,
        "lines": lines,
        "bytes": total_bytes,
        "path": repo_rel_link(path, detect_repo_root()),
        "canonical_path": repo_rel(resolved, detect_repo_root()),
        "mtime_utc": mtime.isoformat(),
        "date": datetime.now(timezone.utc).date().isoformat(),
        "card_scan_lines": scanned,
    }


def card_to_datacard(card: dict[str, Any]):
    """``DatasetCard`` из карточки журнала (совместимость формы, ADR-004)."""
    from net.data import DatasetCard

    known = {field.name for field in dataclasses.fields(DatasetCard)}
    return DatasetCard(**{key: card[key] for key in known if key in card})


# ---------------------------------------------------------------------------
# 2. Подготовка: токенизация, упаковка, шаффл
# ---------------------------------------------------------------------------


def canonical_tokenizer():
    """Канонический BPE скелета: детерминированная заглушка стадии (ADR-4).

    Стадия строит BPE на синтетическом корпусе скелета; его хеш — собственный
    **объявленный хеш заглушки**, и он уезжает в журнал как есть.  Соответствие
    ``net/config.json:tokenizer_hash`` здесь НЕ проверяется: пин описывает
    токенизатор претрейн-корпуса (BPE 160K, обученный на реальных данных), а
    стадии скелета (SFT/RL/смоук) легитимно работают на заглушке — их следы
    (``evidence/a4-run-wire/*/stage-journal.json``) несут именно хеш заглушки.
    Сверку пина выполняет тот пресет, для которого она обязательна
    (``l3-full``, ``tools/pretrain_run.py``), и только против токенизатора,
    которым размечены данные — см. :func:`config_tokenizer_pin` и
    :func:`tokenizer_hash_matches`.
    """
    from net.data import synthetic_corpus_texts
    from net.tokenizer import BPETokenizer

    tok = BPETokenizer(vocab_size=160_000).train(
        synthetic_corpus_texts(seed=0, n_docs=20, words_per_doc=32), seed=0
    )
    return tok, tok.vocab_hash()


def config_tokenizer_pin(config_path: Optional[Path] = None) -> str:
    """Пин токенизатора из конфига прогона (по умолчанию ``net/config.json``).

    Пин объявлен как 16-hex префикс sha256 файла-артефакта корпусного BPE
    (``tools/bpe_train.py``), поэтому сравнение обязано быть префиксным, а не
    на равенство: полный хеш манифеста ``tokens/`` длиннее пина.

    ``config_path`` — явный конфиг прогона (``--config-path``, ревизия
    VERIFICATION-LEG): пин обязан читаться из того же файла, чью архитектуру
    строит лоадер, иначе сверка данных проверяла бы чужой файл.  Без аргумента
    поведение прежнее (пин ``net/config.json``) — аргумент добавлен с дефолтом,
    поэтому вызовы и подмены без него не меняются.
    """
    path = Path(config_path) if config_path is not None else (CASE_DIR / "net" / "config.json")
    config = json.loads(path.read_text(encoding="utf-8"))
    pinned = config.get("tokenizer_hash")
    return str(pinned).strip() if isinstance(pinned, str) else ""


def tokenizer_hash_matches(actual: str, pinned: str) -> bool:
    """Совпадает ли хеш токенизатора с пином: пин — префикс (короткий или равный)."""
    actual = str(actual or "").strip()
    pinned = str(pinned or "").strip()
    if not pinned:
        return True  # пин не объявлен — сверять нечего
    if len(pinned) <= len(actual):
        return actual.startswith(pinned)
    return pinned.startswith(actual)


def document_text(doc: dict[str, Any]) -> str:
    """Детерминированная сериализация messages-документа в текст SFT."""
    messages = doc.get("messages") or []
    parts = []
    for message in messages:
        role = message.get("role", "?")
        content = message.get("content") or ""
        parts.append(f"<|{role}|>{content}")
    return "\n".join(parts)


def load_pool(
    path: Path,
    tokenizer,
    *,
    docs: int,
    seq_len: int,
    doc_chars: int,
) -> tuple[list, dict[str, Any]]:
    """Пул упакованных 8K-последовательностей из головы набора.

    Смоук-окно ограничено ``--pool-docs`` документами: токенизация в
    ``net/tokenizer.py`` — чистый Python, и полный проход по 44 949 документам
    не помещается в смоук-бюджет.  Ограничение записывается в журнал вместе с
    составом окна (какие блоки набора оно реально покрыло — спека §6.2:
    траекторный блок либо покрыт, либо честно помечен как непокрытый).
    """
    from net.data import pack_sequence

    pool: list[Any] = []
    sources: dict[str, int] = {}
    trajectory_docs = 0
    read = 0
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if read >= docs:
                break
            stripped = line.strip()
            if not stripped:
                continue
            doc = json.loads(stripped)
            read += 1
            source = doc.get("source") or "<не объявлен>"
            sources[source] = sources.get(source, 0) + 1
            if int(doc.get("n_tool_calls") or 0) >= 2 or "multiturn" in source:
                trajectory_docs += 1
            text = document_text(doc)[:doc_chars]
            tokens = tokenizer.encode(text)
            inputs, _labels = pack_sequence(tokens, seq_len)
            pool.append(inputs)
    provenance = {
        "documents": read,
        "sources": dict(sorted(sources.items(), key=lambda kv: (-kv[1], kv[0]))),
        "trajectory_documents": trajectory_docs,
        "trajectory_block_covered": trajectory_docs > 0,
    }
    return pool, provenance


def shuffle_pool(pool: list, seed: int) -> list:
    """Детерминированный шаффл пула пиннутым сидом (AD-11)."""
    order = list(range(len(pool)))
    random.Random(seed).shuffle(order)
    return [pool[i] for i in order]


# ---------------------------------------------------------------------------
# 3. Цикл обучения
# ---------------------------------------------------------------------------


# Конфиг стадии: численные смоук-пресеты приёмки (``net/tests/conftest.py``,
# заморожены) и полный скелет L3 из декларативного ``net/config.json``
# (пресет ``l3-full``, AD-9/C-035) вместе с оценкой ресурсов до старта
# обучения (шаг 3 дельты).


def load_l3_full_config():
    """Пресет ``l3-full``: полный конфиг скелета L3 из ``net/config.json``.

    Читается существующим загрузчиком ``net.config.load_config`` — числа
    архитектуры (24 слоя 18 KDA : 6 MLA, hidden 1536, LatentMoE 12+2 top-2, MTP,
    AttnRes) не дублируются в коде пресета, поэтому расхождение пресета с
    декларативным конфигом невозможно по построению (AD-9/C-035: конфиг
    первичен).  ``vocab_size`` и ``qat_enabled`` подставляет стадия — как и для
    численных пресетов ``small``/``tiny``.
    """
    from net.config import load_config

    return load_config(L3_CONFIG_PATH)


def load_config_path(config_path) -> object:
    """Конфиг из явного ``--config-path`` (ревизия VERIFICATION-LEG).

    Читается тем же загрузчиком ``net.config.load_config``, что и пресет
    ``l3-full``: декларативное поле ``layer_composition`` (в т.ч.
    ``dense-standard``) и валидация схемы — одни и те же, отдельного пути для
    «другого» конфига нет.  Подмена ``net/config.json`` при этом запрещена
    (пин скелета), поэтому переопределение пути — единственная точка выбора.
    """
    from net.config import load_config

    return load_config(Path(config_path))


def config_path_report(config_path) -> dict[str, Any]:
    """След явного конфига в журнал: относительный путь и канонический sha256."""
    path = Path(config_path)
    return {
        "path": repo_rel(path, detect_repo_root()),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def l3_config_report() -> dict[str, Any]:
    """Отчёт о переносе декларативного конфига в смоук-конфиг (шаг 1 дельты).

    ``unmapped`` — поля ``net/config.json``, у которых нет аналога в
    dataclass'е конфига: метаданные прогона и бюджета (счётчики параметров,
    состав слоёв, хеш токенизатора, ``deviations``), а не архитектура сети.
    Они не выдумываются и не переносятся — список идёт в журнал стадии, чтобы
    «неперенесённое» было видно явно, а не потеряно молча.
    """
    from net.config import ModelConfig

    declared = json.loads(L3_CONFIG_PATH.read_text(encoding="utf-8"))
    known = {fld.name for fld in dataclasses.fields(ModelConfig)}
    return {
        "path": repo_rel(L3_CONFIG_PATH, detect_repo_root()),
        # Тот же канонический дайджест, что net.data.shard_hash (потоковый
        # sha256), но без импорта jax ради чтения конфига.
        "sha256": hashlib.sha256(L3_CONFIG_PATH.read_bytes()).hexdigest(),
        "unmapped": sorted(key for key in declared if key not in known),
        "declared_vocab_size": declared.get("vocab_size"),
        "declared_param_count": declared.get("actual_param_count"),
        "declared_active_params": declared.get("active_params_per_token"),
    }


def state_estimate_bytes(params_total: int) -> int:
    """Оценка памяти состояния (веса + оптимизатор) для ``params_total``."""
    return STATE_BYTES_PER_PARAM * int(params_total)


def gib(value: Optional[int]) -> Optional[float]:
    """Байты в ГиБ для журнала (``None`` проходит насквозь — «не посчитано»)."""
    return None if value is None else round(int(value) / GIB, 3)


def device_memory_free_bytes() -> tuple[Optional[int], str]:
    """Свободная память GPU по данным JAX — без nvidia-smi/pynvml/новых env.

    Единственный источник — ``jax.devices()[i].memory_stats()``: ``bytes_limit``
    (пул аллокатора) минус ``peak_bytes_in_use`` (пик процесса) даёт оценку
    доступного сверху.  Проверка не имитируется: нет GPU-устройства, нет
    статистики или ключей — вернётся ``None`` и причина, а вердикт оценки станет
    «не проверено» (честная фиксация вместо зелёного по умолчанию).
    """
    try:
        import jax

        gpus = [device for device in jax.devices() if device.platform == "gpu"]
    except Exception as exc:  # noqa: BLE001 — сломанный плагин не роняет стадию
        return None, f"jax.devices() недоступны: {type(exc).__name__}: {exc}"
    if not gpus:
        return None, "GPU-устройства нет (jax.devices(): cpu) — сверка невозможна"
    device = gpus[0]
    try:
        stats = device.memory_stats()
    except Exception as exc:  # noqa: BLE001 — статистика может быть недоступна
        return None, f"memory_stats недоступны: {type(exc).__name__}: {exc}"
    if not isinstance(stats, dict):
        return None, "memory_stats вернули None (статистика недоступна)"
    limit = stats.get("bytes_limit")
    used = stats.get("peak_bytes_in_use")
    if used is None:
        used = stats.get("bytes_in_use")
    if limit is None or used is None:
        return None, f"memory_stats без bytes_limit/peak_bytes_in_use: {sorted(stats)}"
    return max(int(limit) - int(used), 0), (
        f"{device.device_kind}: jax memory_stats (bytes_limit - peak_bytes_in_use)"
    )


def assess_l3_resources(cfg, report: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    """Оценка ресурсов пресета ``l3-full`` до старта обучения (шаг 3 дельты).

    Параметры считаются по формам (``net.model.param_count`` /
    ``active_param_count`` — ``jax.eval_shape`` без аллокации), а не замером
    памяти: замер потребовал бы аллокации, влезаемость которой и проверяется.
    Состояние оценивается как ``STATE_BYTES_PER_PARAM`` байт на параметр.

    ``*_declared`` — декларативные числа ``net/config.json``: они описывают
    скелет с полным vocab 160K и больше построенных, потому что смоук
    подставляет свой vocab (покрывающий id канонического токенизатора).
    Сверка с железом идёт по консервативной (большей) из двух оценок — иначе
    WARN мог бы оказаться ложно-зелёным именно на полном масштабе.

    Функция не поднимает исключений: «не посчитано» — это вердикт
    ``not_verified`` с причиной, а не падение стадии.
    """
    report = report or {}
    try:
        from net.model import active_param_count, param_count

        params_total = int(param_count(cfg))
        params_active = int(active_param_count(cfg))
        counting = (
            "net.model.param_count/active_param_count (jax.eval_shape, без аллокации)"
        )
    except Exception as exc:  # noqa: BLE001 — оценка не роняет стадию
        params_total = params_active = None
        counting = f"не посчитано: {type(exc).__name__}: {exc}"

    declared_total = report.get("declared_param_count")
    declared_active = report.get("declared_active_params")
    built_bytes = (
        state_estimate_bytes(params_total) if params_total is not None else None
    )
    declared_bytes = (
        state_estimate_bytes(declared_total)
        if isinstance(declared_total, int) and declared_total > 0
        else None
    )
    free_bytes, source = device_memory_free_bytes()

    if built_bytes is None:
        verdict = "not_verified"
        note = f"число параметров не посчитано ({counting}) — оценка состояния не сделана"
    else:
        worst = max(built_bytes, declared_bytes or 0)
        if free_bytes is None:
            verdict = "not_verified"
            note = (
                f"оценка состояния {gib(worst)} ГиБ не сверена с железом: {source} "
                "(AD-7: одна нагрузка за раз — пре-флайт за владельцем)"
            )
        elif worst > free_bytes:
            verdict = "warn"
            note = (
                f"оценка состояния {gib(worst)} ГиБ > свободной памяти "
                f"{gib(free_bytes)} ГиБ ({source}) — прогону пресета l3-full "
                "может не хватить памяти (AD-7)"
            )
        else:
            verdict = "ok"
            note = (
                f"оценка состояния {gib(worst)} ГиБ <= свободной памяти "
                f"{gib(free_bytes)} ГиБ ({source})"
            )
    return {
        "preset": L3_FULL_PRESET,
        "built_vocab_size": getattr(cfg, "vocab_size", None),
        "params_total": params_total,
        "params_active": params_active,
        "params_total_declared": declared_total,
        "params_active_declared": declared_active,
        "bytes_per_param": STATE_BYTES_PER_PARAM,
        "state_estimate_bytes": built_bytes,
        "state_estimate_gb": gib(built_bytes),
        "state_estimate_gb_declared": gib(declared_bytes),
        "free_device_bytes": free_bytes,
        "free_device_gb": gib(free_bytes),
        "device_memory_source": source,
        "param_counting": counting,
        "verdict": verdict,
        "note": note,
    }


def l3_full_notes(estimate: dict[str, Any], report: dict[str, Any]) -> list[str]:
    """Строки ``notes`` журнала стадии для пресета ``l3-full`` (шаги 1 и 3).

    Отдельная функция — чтобы контракт журнала (``preset=l3-full,
    params_total=…, state_estimate_gb=…`` плюс список неперенесённых полей
    конфига) проверялся тестом без прогона стадии.
    """
    total = estimate.get("params_total")
    active = estimate.get("params_active")
    state_gb = estimate.get("state_estimate_gb")
    unmapped = report.get("unmapped") or []
    declared_vocab = report.get("declared_vocab_size")
    built_vocab = estimate.get("built_vocab_size")
    verdict = estimate.get("verdict")
    notes = [
        f"preset={L3_FULL_PRESET}, "
        f"params_total={total if total is not None else 'не посчитано'}, "
        f"state_estimate_gb={state_gb if state_gb is not None else 'не посчитано'}",
        (
            f"{L3_FULL_PRESET}: конфиг — полный скелет L3 из декларативного "
            f"{report.get('path', 'net/config.json')} (AD-9/C-035: конфиг первичен), "
            f"sha256={str(report.get('sha256'))[:16]}…; "
            f"vocab {declared_vocab} → {built_vocab} при построении (vocab смоука "
            "покрывает испускаемые id канонического токенизатора)"
        ),
        (
            f"{L3_FULL_PRESET}: поля net/config.json без аналога в конфиге стадии "
            "не переносятся (метаданные прогона и бюджета, не архитектура): "
            + (", ".join(unmapped) if unmapped else "нет")
        ),
        (
            f"{L3_FULL_PRESET}: оценка ресурсов — params_active="
            f"{active if active is not None else 'не посчитано'}, "
            f"состояние ≈ {state_gb if state_gb is not None else 'не посчитано'} ГиБ "
            f"({STATE_BYTES_PER_PARAM} Б/параметр: bf16-веса + fp32-мастер + m/v + "
            f"градиенты; декларативно по net/config.json — "
            f"{estimate.get('state_estimate_gb_declared')} ГиБ), "
            f"свободно {estimate.get('free_device_gb')} ГиБ "
            f"[{estimate.get('device_memory_source')}], вердикт={verdict}"
        ),
        (
            f"{L3_FULL_PRESET}: "
            + ("WARN — " if verdict == "warn" else "")
            + str(estimate.get("note", ""))
        ),
    ]
    return notes


def build_model_config(
    vocab_size: int, preset: str, qat_weights: bool = True, config_path=None
):
    """Конфиг скелета L3 для смоука.

    ``small``/``tiny`` — численные смоук-конфиги ``net/tests/conftest.py``
    (заморожены: их поля стоят приёмочные тесты сети); ``l3-full`` — полный
    конфиг скелета из декларативного ``net/config.json``, чтобы смоук-провод
    шёл по той же архитектуре, что пиннута в конфиге, а не по её уменьшенной
    копии.

    ``config_path`` — явный конфиг (ревизия VERIFICATION-LEG, флаг
    ``--config-path``): путь имеет приоритет над пресетом, потому что пресеты
    ``small``/``tiny``/``l3-full`` заморожены и не трогаются, а прогон
    A/B (dense-124M vs arch-124M) обязан идти по своему файлу.  ``None`` —
    прежнее поведение (выбор по пресету), поэтому старые вызовы не меняются;
    проверяется механически (``tools/tests/test_run_sft_smoke_config_path.py``).

    ``qat_enabled`` выставляется вместе с флагом прогона: QAT включается со
    стадии SFT (ADR-005 п. 7), поэтому конфиг стадии обязан нести включённый
    признак — иначе журнал и конфиг противоречили бы друг другу.
    """
    conftest = _load_acceptance_conftest()  # ADR-010: пиннинг до импорта net.*
    if config_path is not None:
        base = load_config_path(config_path)
    elif preset == "tiny":
        base = conftest.tiny_config()
    elif preset == L3_FULL_PRESET:
        base = load_l3_full_config()
    else:
        base = conftest.small_config()
    return dataclasses.replace(
        base, vocab_size=vocab_size, qat_enabled=bool(qat_weights)
    )


def run_training(
    pool: list,
    cfg,
    *,
    steps: int,
    seed: int,
    lr: float,
    warmup_ratio: float,
    qat_weights: bool,
    chunk_size: int = 64,
):
    """N шагов SFT на скелете L3; возвращает (losses, steps_done, params, tree_hash).

    QAT: при ``qat_weights`` веса fake-quant'ятся MXFP4 в прямом проходе
    (straight-through, ``net/quant.py``) — ADR-005 п. 7; мастер-веса остаются
    полной точности, как и требует QAT.
    """
    import jax
    import jax.numpy as jnp
    import jax.random as jr

    from net import model, optimizer, quant
    from net import checkpoint as checkpoint_mod

    if not pool:
        raise StageError("пул пуст: нечего обучать")

    params = model.init_params(jr.PRNGKey(seed), cfg)
    state = optimizer.init_state(params)
    lr_at = optimizer.cosine_schedule(lr, steps, warmup_ratio)
    step = optimizer.make_step(cfg)

    def loss_fn(p, x):
        qp = quant.apply_fake_quant_tree(p) if qat_weights else p
        return model.compute_loss(qp, cfg, x, chunk_size=chunk_size)

    grad_fn = jax.jit(jax.value_and_grad(loss_fn))

    losses: list[float] = []
    step_seconds: list[float] = []
    for index in range(steps):
        ids = jnp.asarray(pool[index % len(pool)])[None, :]
        tick = time.time()
        loss, grads = grad_fn(params, ids)
        params, state = step(params, grads, state, lr_at(index))
        losses.append(float(loss))
        step_seconds.append(time.time() - tick)
        if index == 0 or (index + 1) % 50 == 0 or index + 1 == steps:
            print(
                f"[sft] шаг {index + 1}/{steps}: loss={losses[-1]:.4f} "
                f"lr={float(lr_at(index)):.3e} {step_seconds[-1]:.2f} с/шаг",
                flush=True,
            )
    timings = {
        "first_step_seconds": round(step_seconds[0], 3),
        "step_seconds_mean_tail": round(
            sum(step_seconds[1:]) / max(len(step_seconds) - 1, 1), 4
        ),
    }
    return losses, len(losses), params, checkpoint_mod.tree_hash(params), timings


# ---------------------------------------------------------------------------
# 4. Чекпойнт
# ---------------------------------------------------------------------------


def save_stage_checkpoint(params, ckpt_dir: Path, shared: Path, tree_digest: str):
    """Orbax-чекпойнт: канонически на сетевом диске, в репозитории — симлинк.

    Копия весов в рабочем каталоге запрещена (C-032), поэтому запись идёт в
    канонический каталог на ``~/gb10-shared``, а ``ckpt_dir`` — симлинк на него
    (не наоборот: orbax при ``force=True`` сам очищает целевой каталог, а
    ``shutil.rmtree`` отказывается работать с симлинком).  Round-trip
    проверяется здесь же.  Возвращает описание чекпойнта для журнала.
    """
    from net import checkpoint as checkpoint_mod

    shared = Path(shared)
    canonical = shared / CHECKPOINT_SUBDIR
    canonical.mkdir(parents=True, exist_ok=True)
    ckpt_dir = Path(ckpt_dir)
    if ckpt_dir.is_symlink():
        pass  # повторный прогон: симлинк уже смонтирован
    elif ckpt_dir.exists():
        raise StageError(
            f"каталог чекпойнта {repo_rel(ckpt_dir, detect_repo_root())} — не "
            "симлинк: копия весов в рабочем каталоге запрещена (C-032)"
        )
    else:
        ckpt_dir.parent.mkdir(parents=True, exist_ok=True)
        ckpt_dir.symlink_to(canonical)

    digest = checkpoint_mod.save_checkpoint(params, canonical)
    if digest != tree_digest:
        raise StageError(
            f"tree_hash до сохранения {tree_digest} != после {digest}"
        )
    restored = checkpoint_mod.load_checkpoint(canonical, target=params)
    if checkpoint_mod.tree_hash(restored) != digest:
        raise StageError("round-trip чекпойнта не совпал по tree_hash")
    return {
        "path": repo_rel_link(ckpt_dir, detect_repo_root()),
        "canonical": repo_rel(canonical, detect_repo_root()),
        "symlink": True,
        "format": "orbax",
        "tree_hash": digest,
        "roundtrip_ok": True,
    }


# ---------------------------------------------------------------------------
# 5. Смета (AD-8 / C-041)
# ---------------------------------------------------------------------------


def budget_report(run_ref: str, gpu_hours_actual: float, explicit_limit: Optional[float]):
    """Смета стадии: факт часов против лимита; отсутствие сметы — факт, не тишина."""
    estimate_path = CASE_DIR / "evidence" / "budget" / f"{run_ref}.json"
    report: dict[str, Any] = {
        "run_ref": run_ref,
        "estimate_path": repo_rel(estimate_path, detect_repo_root()),
        "estimate_present": estimate_path.is_file(),
        "gpu_hours_actual": round(gpu_hours_actual, 6),
        "limit_usd": explicit_limit,
        "usd_estimate": None,
        "verdict": "смета отсутствует: запуск «на длинную дистанцию» блокирован (AD-8)",
    }
    if estimate_path.is_file():
        try:
            estimate = json.loads(estimate_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            report["verdict"] = f"смета не читается: {exc}"
            return report
        report["usd_estimate"] = estimate.get("usd_estimate")
        if report["limit_usd"] is None:
            report["limit_usd"] = estimate.get("limit_usd")
        estimate_hours = estimate.get("gpu_hours_estimate")
        report["gpu_hours_estimate"] = estimate_hours
        if isinstance(estimate_hours, (int, float)) and estimate_hours > 0:
            report["share_of_estimate"] = round(gpu_hours_actual / estimate_hours, 8)
        report["verdict"] = (
            "факт в пределах сметы"
            if isinstance(report["limit_usd"], (int, float))
            and isinstance(report["usd_estimate"], (int, float))
            else "смета найдена, лимит не сопоставлен"
        )
    elif explicit_limit is not None:
        report["verdict"] = (
            "сметы нет; объявлен явный лимит смоука "
            f"{explicit_limit} USD — прогон смоука в пределах лимита"
        )
    return report


# ---------------------------------------------------------------------------
# Прогон стадии
# ---------------------------------------------------------------------------


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Стадия sft конвейера A4: смоук-цикл SFT + журнал стадии",
    )
    parser.add_argument("--data", default=DEFAULT_DATA,
                        help="jsonl набора; обязан быть симлинком на ~/gb10-shared")
    parser.add_argument("--out", default=DEFAULT_OUT,
                        help="каталог следа стадии (внутри репозитория)")
    parser.add_argument("--steps", type=int, default=200, help="число шагов (смоук ≤ 200)")
    parser.add_argument("--seq-len", type=int, default=8192, help="длина упаковки (8K)")
    parser.add_argument("--pool-docs", type=int, default=64,
                        help="документов в смоук-пуле (голова набора)")
    parser.add_argument("--doc-chars", type=int, default=8192,
                        help="символов документа в пул (ограничение смоук-окна)")
    parser.add_argument("--seed", type=int, default=1337, help="пиннутый сид (AD-11)")
    parser.add_argument("--lr", type=float, default=1e-2, help="пиковый LR (cosine + warmup)")
    parser.add_argument("--warmup-ratio", type=float, default=0.01)
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--model-preset", choices=SMOKE_PRESETS + (L3_FULL_PRESET,),
                        default="small",
                        help="small/tiny — смоук-конфиги приёмки сети "
                             "(net/tests/conftest.py); l3-full — полный конфиг "
                             "скелета L3 из net/config.json (AD-9/C-035)")
    parser.add_argument("--config-path", type=Path, default=None,
                        help="явный путь конфига модели (ревизия VERIFICATION-LEG): "
                             "имеет приоритет над --model-preset; подмена "
                             "net/config.json запрещена, поэтому путь задаётся "
                             "здесь. По умолчанию (не задан) — прежнее поведение "
                             "пресетов small/tiny/l3-full")
    parser.add_argument("--qat-weights", dest="qat_weights", action="store_true",
                        default=True, help="QAT fake-quant MXFP4 весов (ADR-005 п.7)")
    parser.add_argument("--no-qat-weights", dest="qat_weights", action="store_false")
    parser.add_argument("--ckpt-dir", default=None,
                        help="симлинк каталога чекпойнта (по умолчанию <out>/checkpoint)")
    parser.add_argument("--budget-limit-usd", type=float, default=None,
                        help="явный лимит смоука, если сметы evidence/budget/<run-ref>.json нет")
    parser.add_argument("--card-scan-lines", type=int, default=DEFAULT_CARD_SCAN_LINES)
    parser.add_argument("--json", action="store_true", help="печать журнала в stdout")
    parser.add_argument("--preflight-gate", default="sft-start",
                        help="открывающий гейт: preflight перед стартом стадии (ADR-038, K3)")
    parser.add_argument("--override-preflight", default=None,
                        help="аварийный обход preflight гейта с причиной (пишется фактом)")
    parser.add_argument("--no-preflight", action="store_true",
                        help="не вызывать preflight (осознанно; по умолчанию вызывается)")
    return parser.parse_args(list(argv) if argv is not None else None)


def run_stage(args: argparse.Namespace) -> tuple[dict[str, Any], bool]:
    """Исполнить стадию; вернуть (журнал, успех).  Журнал пишется всегда."""
    started = time.time()
    started_wall = datetime.now(timezone.utc)
    repo_root = detect_repo_root()
    out_dir = case_path(args.out)
    data_path = case_path(args.data)
    ckpt_dir = case_path(args.ckpt_dir) if args.ckpt_dir else out_dir / "checkpoint"

    shared = shared_root()
    conftest = _load_acceptance_conftest()
    backend = _backend_block(conftest)

    journal: dict[str, Any] = {
        "schema": JOURNAL_SCHEMA,
        "stage": "sft",
        "status": "absent",
        "scale": "smoke",
        "run_ref": repo_rel(out_dir / "stage-journal.json", repo_root),
        "started_at": started_wall.isoformat(),
        "seed": args.seed,
        "steps": args.steps,
        "qat_weights": "on" if args.qat_weights else "off",
        "backend": backend,
        "notes": [
            "Смоук-масштаб: доказательство провода стадии, не обучение модели "
            "(спека §5).",
            DATASET_CHOICE_NOTE,
        ],
    }

    # --- 1. данные и карточка -------------------------------------------
    try:
        print(f"[sft] данные: {repo_rel_link(data_path, repo_root)}", flush=True)
        journal["dataset_symlink"] = validate_data_path(data_path, shared, repo_root)
        print(f"[sft] симлинк ок: -> {journal['dataset_symlink']['target']}", flush=True)
        print("[sft] карточка: sha256 потоково + состав по source ...", flush=True)
        card = build_dataset_card(
            data_path,
            scan_lines=args.card_scan_lines,
            provenance_note=(
                "gb10-shared/datasets/" + Path(data_path).resolve().name
                + " (симлинк; сборка build_sft_v12.sh = v10.1 + v11)"
            ),
        )
        card_to_datacard(card)  # совместимость формы с net.data.DatasetCard
        journal["dataset_card"] = card
        print(
            f"[sft] карточка: {card['lines']} строк, {card['bytes']} Б, "
            f"sha256={card['sha256'][:16]}…, блоков={len(card['blocks'])}",
            flush=True,
        )
    except StageError as exc:
        journal["status"] = "absent"
        journal["error"] = str(exc)
        journal["wall_clock_s"] = round(time.time() - started, 3)
        _write_journal(journal, out_dir)
        print(f"[sft] ОТКАЗ данных: {exc}", file=sys.stderr, flush=True)
        return journal, False

    # --- 2. подготовка ---------------------------------------------------
    tokenizer, tokenizer_hash = canonical_tokenizer()
    max_id = max(tokenizer._merge_id.values()) if tokenizer._merge_id else 0
    max_emitted = max(3 + 256 - 1, max_id)
    vocab_size = 1 << max(10, int(max_emitted).bit_length())
    journal["tokenizer"] = {
        "hash": tokenizer_hash,
        "source": "net/tokenizer.py canonical (net/config.json:tokenizer_hash)",
        "vocab_size": tokenizer.vocab_size,
        "merges": len(tokenizer.merges),
        "max_emitted_id": int(max_emitted),
        "model_vocab_size": int(vocab_size),
        "note": (
            "модельный vocab смоука покрывает все испускаемые id канонического "
            "токенизатора; полный vocab 160K — масштаб вне смоука"
        ),
    }
    print(
        f"[sft] токенизатор: hash={tokenizer_hash[:16]}…, merges={len(tokenizer.merges)}, "
        f"модельный vocab={vocab_size}",
        flush=True,
    )

    started_pool = time.time()
    pool, pool_provenance = load_pool(
        data_path, tokenizer,
        docs=args.pool_docs, seq_len=args.seq_len, doc_chars=args.doc_chars,
    )
    pool = shuffle_pool(pool, args.seed)
    tokens_in_pool = int(sum(int(seq.shape[0]) for seq in pool))
    journal["pool"] = {
        "docs": pool_provenance["documents"],
        "sources": pool_provenance["sources"],
        "trajectory_documents": pool_provenance["trajectory_documents"],
        "trajectory_block_covered": pool_provenance["trajectory_block_covered"],
        "sequences": len(pool),
        "seq_len": args.seq_len,
        "tokens": tokens_in_pool,
        "tokens_per_step": args.seq_len - 1,
        "epochs_over_pool": round(args.steps / max(len(pool), 1), 3),
        "tokenize_seconds": round(time.time() - started_pool, 3),
        "shuffle_seed": args.seed,
        "note": (
            "смоук-окно: голова набора, ограничено --pool-docs/--doc-chars; "
            "полный проход по набору не входит в смоук-бюджет (чистый Python BPE)"
        ),
    }
    journal["notes"].append(
        "Траекторный блок (ADR-005 п. 1): смоук-окно "
        + (
            f"покрыло {pool_provenance['trajectory_documents']} мультитёрн-документов "
            "агентного цикла — блок присутствует в окне."
            if pool_provenance["trajectory_block_covered"]
            else "не покрыло ни одного мультитёрн-документа — покрыт только "
            "доменный блок (честная пометка, спека §6.2)."
        )
    )
    journal["scope"] = {
        "dataset_coverage": (
            f"{pool_provenance['documents']} документов из {journal['dataset_card']['lines']} "
            f"в наборе ({journal['dataset_card']['lines']} строк, состав по всей голове)"
        ),
        "trajectory_block_covered": pool_provenance["trajectory_block_covered"],
        "not_full_stage": (
            "СМОУК: стадия исполнена в смоук-масштабе; закрытием стадии "
            "«в полном объёме» не является (спека §5)"
        ),
    }
    print(
        f"[sft] пул: {len(pool)} последовательностей × {args.seq_len} ток. "
        f"= {tokens_in_pool} токенов ({journal['pool']['tokenize_seconds']} с)",
        flush=True,
    )
    if not pool:
        journal["status"] = "absent"
        journal["error"] = "пул пуст: набор не дал ни одного документа"
        journal["wall_clock_s"] = round(time.time() - started, 3)
        _write_journal(journal, out_dir)
        return journal, False

    # --- 3. цикл ---------------------------------------------------------
    config_path = getattr(args, "config_path", None)
    cfg = build_model_config(
        vocab_size, args.model_preset, args.qat_weights, config_path=config_path
    )
    journal["model_config"] = {
        "preset": args.model_preset,
        "qat_enabled": bool(cfg.qat_enabled),
        "qat_kv_enabled": bool(cfg.qat_kv_enabled),
        "vocab_size": cfg.vocab_size,
        "hidden": cfg.hidden,
        "num_layers": cfg.num_layers,
        "num_kda_layers": cfg.num_kda_layers,
        "num_mla_layers": cfg.num_mla_layers,
        "dense_standard_layers": int(cfg.dense_standard_layers),
        "num_heads": cfg.num_heads,
        "head_dim": cfg.head_dim,
        "dtype": "bf16",
        "note": (
            "скелет L3 той же архитектуры, что net/config.json (3 KDA + 1 MLA, "
            "MTP, LatentMoE, AttnRes), в смоук-масштабе net/tests/conftest.py"
        ),
    }
    # Явный конфиг (--config-path, VERIFICATION-LEG): источник заменяет пресет,
    # и это видно в журнале — путь относительный (ADR-014 п. 8) + sha256.
    if config_path is not None:
        report = config_path_report(config_path)
        journal["model_config"]["config_source"] = report["path"]
        journal["model_config"]["config_sha256"] = report["sha256"]
        journal["model_config"]["note"] = (
            "конфиг модели задан явным --config-path (ревизия VERIFICATION-LEG): "
            f"{report['path']}; пресет {args.model_preset} для выбора конфига не "
            "использован (замороженные пресеты small/tiny/l3-full не тронуты)"
        )
    # Пресет l3-full: полный конфиг из декларативного net/config.json + оценка
    # ресурсов до старта обучения (шаги 1 и 3 дельты).  Численные пресеты
    # small/tiny этой ветки не касаются — их поведение заморожено.  Явный
    # --config-path эту ветку выключает: иначе она перезаписала бы источник
    # конфига ссылкой на net/config.json, по которому прогон НЕ шёл.
    if args.model_preset == L3_FULL_PRESET and config_path is None:
        report = l3_config_report()
        estimate = assess_l3_resources(cfg, report)
        journal["model_config"]["config_source"] = report["path"]
        journal["model_config"]["config_sha256"] = report["sha256"]
        journal["model_config"]["moe"] = {
            "dense_layers": cfg.moe_dense_layers,
            "num_routed": cfg.moe_num_routed,
            "num_shared": cfg.moe_num_shared,
            "top_k": cfg.moe_top_k,
            "latent_dim": cfg.moe_latent_dim,
        }
        journal["model_config"]["mla_pool"] = {
            "block": cfg.mla_pool_block,
            "size": cfg.mla_pool_size,
            "modes": list(cfg.mla_layer_modes),
        }
        journal["model_config"]["note"] = (
            "полный конфиг скелета L3, прочитанный из декларативного "
            "net/config.json (AD-9/C-035: конфиг первичен): 24 слоя (18 KDA + 6 "
            "MLA), hidden 1536, LatentMoE 12+2 top-2, MTP, AttnRes — пресет не "
            "переписывает архитектуру числами"
        )
        journal["resource_estimate"] = estimate
        journal["notes"].extend(l3_full_notes(estimate, report))
        print(
            f"[sft] ресурсы ({L3_FULL_PRESET}): params_total={estimate['params_total']} "
            f"(декларативно {estimate['params_total_declared']}), "
            f"params_active={estimate['params_active']}, состояние ≈ "
            f"{estimate['state_estimate_gb']} ГиБ "
            f"({STATE_BYTES_PER_PARAM} Б/параметр — оценка, не замер), свободно "
            f"{estimate['free_device_gb']} ГиБ [{estimate['device_memory_source']}] "
            f"→ {estimate['verdict']}",
            flush=True,
        )
        if estimate["verdict"] == "warn":
            print(
                f"[sft] WARN ({L3_FULL_PRESET}): {estimate['note']}",
                file=sys.stderr, flush=True,
            )
    journal["optimizer"] = {
        "name": "per-head-muon+adamw",
        "lr": args.lr,
        "schedule": "cosine",
        "warmup_ratio": args.warmup_ratio,
        "source": "net/optimizer.py",
    }
    print(
        f"[sft] цикл: {args.steps} шагов, пресет {args.model_preset}, "
        f"QAT={journal['qat_weights']}, LR={args.lr}",
        flush=True,
    )
    started_train = time.time()
    try:
        losses, steps_done, params, tree_digest, timings = run_training(
            pool, cfg,
            steps=args.steps, seed=args.seed, lr=args.lr,
            warmup_ratio=args.warmup_ratio, qat_weights=args.qat_weights,
            chunk_size=args.chunk_size,
        )
    except Exception as exc:  # noqa: BLE001 — журнал обязан быть записан честно
        journal["status"] = "absent"
        journal["error"] = f"{type(exc).__name__}: {exc}"
        journal["wall_clock_s"] = round(time.time() - started, 3)
        _write_journal(journal, out_dir)
        print(f"[sft] ОТКАЗ цикла: {exc}", file=sys.stderr, flush=True)
        return journal, False
    train_seconds = time.time() - started_train
    journal["train_seconds"] = round(train_seconds, 3)
    journal["timing"] = timings
    journal["loss_curve"] = [round(value, 6) for value in losses]
    journal["steps"] = steps_done
    journal["tokens_seen"] = steps_done * (args.seq_len - 1)
    journal["loss_first"] = round(losses[0], 6)
    journal["loss_last"] = round(losses[-1], 6)
    journal["loss_fell"] = bool(losses[0] > losses[-1])
    print(
        f"[sft] лосс: первый {losses[0]:.4f} -> последний {losses[-1]:.4f} "
        f"({'падает' if journal['loss_fell'] else 'НЕ падает'})",
        flush=True,
    )

    # --- 4. чекпойнт -----------------------------------------------------
    try:
        journal["checkpoint"] = save_stage_checkpoint(
            params, ckpt_dir, shared, tree_digest
        )
    except Exception as exc:  # noqa: BLE001 — журнал обязан быть записан честно
        journal["status"] = "failed"
        journal["error"] = f"{type(exc).__name__}: {exc}"
        journal["wall_clock_s"] = round(time.time() - started, 3)
        _write_journal(journal, out_dir)
        print(f"[sft] ОТКАЗ чекпойнта: {exc}", file=sys.stderr, flush=True)
        return journal, False
    print(
        f"[sft] чекпойнт: tree_hash={tree_digest[:16]}…, "
        f"round-trip ok -> {journal['checkpoint']['path']}",
        flush=True,
    )

    # --- 5. стоимость и статус -------------------------------------------
    wall_clock = time.time() - started
    journal["wall_clock_s"] = round(wall_clock, 3)
    journal["gpu_hours_actual"] = round(wall_clock / 3600.0, 6)
    journal["budget"] = budget_report(
        f"sft-smoke-{started_wall.date().isoformat()}",
        wall_clock / 3600.0,
        args.budget_limit_usd,
    )
    journal["status"] = "executed" if journal["loss_fell"] else "failed"
    if not journal["loss_fell"]:
        journal["error"] = (
            "лосс не упал на смоук-окне (loss_first <= loss_last) — "
            "честная фиксация, подгонка запрещена (спека §4.2)"
        )
    journal["finished_at"] = datetime.now(timezone.utc).isoformat()
    _write_journal(journal, out_dir)

    print(
        f"[sft] стадия: {journal['status']}; {steps_done} шагов, "
        f"{journal['tokens_seen']} токенов, {wall_clock:.1f} с "
        f"({journal['gpu_hours_actual']:.6f} GPU-ч) — СМОУК",
        flush=True,
    )
    print(f"[sft] смета (AD-8/C-041): {journal['budget']['verdict']}", flush=True)
    print(f"[sft] журнал: {journal['run_ref']}", flush=True)
    return journal, journal["status"] == "executed"


def _backend_block(conftest) -> dict[str, Any]:
    """Пиннинг бэкенда в журнал (ADR-010): платформа, устройства, точность."""
    try:
        import jax

        devices = [
            {"platform": device.platform, "kind": device.device_kind}
            for device in jax.devices()
        ]
    except Exception as exc:  # pragma: no cover — сломанный плагин
        devices = [{"error": str(exc)}]
        jax = None
    platform_name = devices[0].get("platform") if devices else "unknown"
    return {
        "platform": platform_name,
        "devices": devices,
        "device_kind": devices[0].get("kind") if devices else "unknown",
        "jax_version": getattr(jax, "__version__", "unknown"),
        "matmul_precision": (
            jax.config.jax_default_matmul_precision if jax is not None else "unknown"
        ),
        "gate_profile": bool(conftest.gate_profile()),
        "determinism_pinned": bool(conftest.determinism_pinned()),
        "python": platform.python_version(),
    }


def _write_journal(journal: dict[str, Any], out_dir: Path) -> None:
    """Атомарная запись журнала стадии (temp + os.replace)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "stage-journal.json"
    tmp = path.with_name(path.name + ".tmp")
    try:
        tmp.write_text(
            json.dumps(journal, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)
    try:
        from tools.stage_preflight import enforce as _enforce_preflight
    except ImportError:  # запуск как скрипт из tools/
        from stage_preflight import enforce as _enforce_preflight

    gate_code = _enforce_preflight(
        args.preflight_gate, args.override_preflight, no_preflight=args.no_preflight
    )
    if gate_code != 0:
        return gate_code
    journal, ok = run_stage(args)
    if args.json:
        print(json.dumps(journal, ensure_ascii=False, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
