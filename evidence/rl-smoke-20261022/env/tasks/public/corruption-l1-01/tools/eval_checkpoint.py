#!/usr/bin/env python3
"""ADR-055 — NTP-PPL оценка чекпоинтов Axiom на двух holdout.

Инструмент постоянный (заменяет транзиентный зонд
``~/axiom-run/demo-20261022/eval_checkpoint.py``) и делает ровно одну вещь:
считает next-token loss на уже **чистом** доменном наборе и на неиспользованном
срезе общего корпуса (W-shard).  Leak-check сюда не входит — это отдельный
``tools/check_eval_leak.py`` (AD-12/C-043); инструмент принимает набор, который
уже прошёл проверку.

Протокол (ADR-055 п. 3): seq 2048, единый токенизатор, NTP loss, отчёт —
дельты, не абсолюты.  Формула — только NTP-голова связанного эмбеддинга::

    hidden = model.forward(params, cfg, ids[None, :], ...,
                           emit_logits=False, return_hidden=True)
    loss   = model._chunked_cross_entropy(
        hidden[:, :-1], ids[None, 1:], params.embedding, ce_chunk_tokens)
    ppl    = exp(loss)

Итог по набору — среднее, взвешенное числом предсказанных токенов
(``len(ids) - 1`` на окно), а не среднее средних: короткое хвостовое окно не
должно весить столько же, сколько полное (та же ловушка, что и в chunked CE).

Входы
-----
``--domain-ids PATH.npy``
    uint32, 1-D, уже токенизированный holdout (BOS/EOS включены).  Fail-closed:
    не 1-D, пусто, dtype не uint32, ``max(id) >= cfg.vocab_size``.
``--w-shard PATH.bin``
    packed uint32 little-endian; записи по ``seq_len``.  Окна нарезаются по
    ``--seq``; неполный хвост отбрасывается и фиксируется в ``meta``.
``--max-tokens``
    потолок числа токенов **на источник** (и для ``.npy``, и для ``.bin``):
    читается/оценивается не больше указанного, чтобы прогон оставался
    ограниченным по времени и памяти.

Режимы
------
``--plan-only``
    CPU/CI-режим: **не импортирует jax** и не загружает чекпоинты.  Проверяет
    существование входов и каталогов чекпоинтов, размеры, уникальность имён,
    dtype/диапазон домена, печатает JSON-план и возвращает 0.  Наличие
    ``<ckpt>/params`` фиксируется флагом ``params_present``: пустой dummy-
    каталог годится для плана, но прогон без ``params/`` откажет.
Без ``--plan-only``
    Ленивый импорт jax строго после ``ensure_mem_fraction()`` (ADR-041) и
    ``XLA_PYTHON_CLIENT_PREALLOCATE=false``; ``load_config('net/config.json')``
    с политиками ``grad_ckpt_policy='none'`` / ``remat_policy='none'``;
    ``init_params`` даёт структуру для ``load_checkpoint(<ckpt>/params)``;
    окна идут последовательно, JIT — один раз на форму.

JSON-контракт (schema ``axiom-eval-checkpoint/1``)::

    {"schema": "axiom-eval-checkpoint/1", "seq": 2048, "max_tokens": 131072,
     "checkpoints": {"<name>": {"path": "...",
        "domain":    {"loss": ..., "ppl": ..., "windows": N, "pred_tokens": M},
        "w_holdout": {"loss": ..., "ppl": ..., "windows": N, "pred_tokens": M}}},
     "meta": {"w_shard": "<basename|none>", "w_shard_bytes": N,
              "domain_ids_sha256": "...", "git_sha": "...",
              "tokenizer_hash": "...", "config_sha256": "...",
              "domain_tokens": N, "domain_tail_tokens": T,
              "w_tokens": N, "w_tail_tokens": T}}

``meta`` заполняется только из доступных артефактов: ``tokenizer_hash`` — пин
``net/config.json``, ``config_sha256`` — sha256 файла конфига,
``domain_ids_sha256`` — sha256 **сырых id** (uint32 LE, без заголовка ``.npy``;
та же конвенция, что ``dataset_digest`` в ``tools/run_a4_pipeline.py``),
``git_sha`` — ``git rev-parse HEAD``.  Чего нет — ``null``, а не догадка.
При отсутствии ``--w-shard`` источник ``w_holdout`` равен ``null``.

Коды возврата: ``0`` — успех; ``1`` — сбой оценки (включая NaN/Inf loss);
``2`` — отвергнутый вход/аргументы (fail-closed).

Запуск::

    python3 tools/eval_checkpoint.py --plan-only \
        --ckpt pretrain=.../step-00000010 --ckpt midtrain=.../step-00000020 \
        --domain-ids domain_eval_v3_clean.npy --w-shard W-00051.bin --out plan.json

    NET_JAX_BACKEND=cpu python3 tools/eval_checkpoint.py \
        --ckpt pretrain=.../step-00000010 \
        --domain-ids domain_eval_v3_clean.npy --w-shard W-00051.bin \
        --out eval_ppl.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

#: Схема отчёта (контракт JSON).
REPORT_SCHEMA = "axiom-eval-checkpoint/1"

#: Корень репозитория: конфиг и ``net/`` читаются от него, а не от cwd.
REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "net" / "config.json"

#: Код возврата: успех.
EXIT_OK = 0
#: Код возврата: сбой оценки (NaN/Inf loss, нечитаемый чекпоинт и т. п.).
EXIT_EVAL_ERROR = 1
#: Код возврата: отвергнутый вход/аргументы (fail-closed).
EXIT_INPUT_ERROR = 2

#: Каталог с параметрами внутри каталога чекпоинта (контракт ``--ckpt``).
PARAMS_DIRNAME = "params"

#: Размер записи packed-шарда: uint32 little-endian.
TOKEN_BYTES = 4

#: ``git rev-parse`` — локальная команда без сети; таймаут держит fail-closed.
_GIT_TIMEOUT_SECS = 10

#: Импорт ``tools.jax_preflight``/``net`` — от корня репозитория, даже если
#: инструмент запущен как ``python3 tools/eval_checkpoint.py`` (тогда в
#: ``sys.path[0]`` лежит ``tools/``, а не корень).
for _path in (str(REPO_ROOT), str(REPO_ROOT / "tools")):
    if _path not in sys.path:
        sys.path.insert(0, _path)


class InputError(ValueError):
    """Вход/аргументы не годятся: fail-closed, код 2."""


class EvalError(RuntimeError):
    """Оценка невозможна/испорчена (например, NaN loss): код 1."""


# --------------------------------------------------------------------------- #
# Чистые функции (тестируются без jax/GPU)
# --------------------------------------------------------------------------- #


def parse_ckpt_spec(specs: Sequence[str]) -> list[tuple[str, Path]]:
    """Разобрать повторяемый ``--ckpt NAME=PATH``.

    Имя непустое и уникальное (пересечение имён — отказ, а не «последний
    победил»); путь непустой.  Разделитель — **первый** ``=``, поэтому ``=``
    внутри пути допустим.  Порядок спецификаций сохраняется (порядок ключей в
    JSON-отчёте детерминирован входом).
    """
    parsed: list[tuple[str, Path]] = []
    seen: set[str] = set()
    for raw in specs:
        if not isinstance(raw, str) or "=" not in raw:
            raise InputError(
                f"--ckpt {raw!r}: ожидается NAME=PATH (нет разделителя '=')"
            )
        name, path = raw.split("=", 1)
        name = name.strip()
        path = path.strip()
        if not name:
            raise InputError(f"--ckpt {raw!r}: пустое имя чекпоинта")
        if not path:
            raise InputError(f"--ckpt {raw!r}: пустой путь чекпоинта")
        if name in seen:
            raise InputError(f"--ckpt: имя {name!r} повторяется — имена обязаны быть уникальны")
        seen.add(name)
        parsed.append((name, Path(path)))
    if not parsed:
        raise InputError("нужен хотя бы один --ckpt NAME=PATH")
    return parsed


def window_ids(ids: Any, seq: int) -> tuple[list[Any], int]:
    """Нарезать поток на полные окна длины ``seq``; вернуть ``(окна, хвост)``.

    Хвост — число токенов, не составивших полное окно (``len(ids) % seq``,
    если ``seq`` не больше длины).  Неполное окно не оценивается: NTP-окно
    фиксированной формы — единица JIT-компиляции и сравнимости.  Работает и с
    ``numpy.ndarray``, и со списком (срезы сохраняют тип входа).
    """
    if seq <= 0:
        raise ValueError(f"seq должен быть > 0, получено {seq!r}")
    total = len(ids)
    count = total // seq
    windows = [ids[index * seq : (index + 1) * seq] for index in range(count)]
    tail = total - count * seq
    return windows, tail


def plan_stream_counts(total_tokens: int, seq: int, max_tokens: int) -> dict[str, int]:
    """План одного источника: сколько токенов/окон/предсказаний будет оценено.

    ``max_tokens`` — потолок **входных** токенов на источник; хвост, не
    составивший окно ``seq``, отбрасывается.  Предсказанных токенов в окне
    ``seq - 1`` (последний входной токен окна не имеет цели внутри окна).
    """
    if seq <= 0:
        raise InputError(f"--seq должен быть > 0, получено {seq!r}")
    if max_tokens <= 0:
        raise InputError(f"--max-tokens должен быть > 0, получено {max_tokens!r}")
    used = min(int(total_tokens), int(max_tokens))
    windows = used // seq
    tail = used - windows * seq
    return {
        "tokens": used,
        "windows": windows,
        "tail_tokens": tail,
        "pred_tokens": windows * (seq - 1),
    }


def summarize(window_losses: Sequence[float], pred_tokens: Sequence[int]) -> dict[str, Any]:
    """Среднее loss по набору, взвешенное числом предсказанных токенов.

    Не среднее средних: вес окна — ``len(ids) - 1``.  ``ppl = exp(loss)``.
    Fail-closed на пустой набор, несовпадение длин, неположительные веса и
    нефинитный loss/ppl (NaN/Inf не должны попасть в JSON).
    """
    losses = list(window_losses)
    weights = [int(value) for value in pred_tokens]
    if not losses:
        raise EvalError("summarize: пустой набор окон — нечего усреднять")
    if len(losses) != len(weights):
        raise EvalError(
            f"summarize: {len(losses)} окон против {len(weights)} весов — "
            "длины обязаны совпадать"
        )
    total = sum(weights)
    if total <= 0:
        raise EvalError(f"summarize: суммарный вес {total} — ожидалось > 0")
    for index, (loss, weight) in enumerate(zip(losses, weights)):
        if weight <= 0:
            raise EvalError(
                f"summarize: окно {index}: вес {weight} — ожидалось > 0"
            )
        if not math.isfinite(float(loss)):
            raise EvalError(f"summarize: окно {index}: loss={loss!r} не конечен")
    weighted = math.fsum(float(loss) * weight for loss, weight in zip(losses, weights))
    loss = weighted / total
    if not math.isfinite(loss):
        raise EvalError(f"summarize: средневзвешенный loss={loss!r} не конечен")
    ppl = math.exp(loss)
    if not math.isfinite(ppl):
        raise EvalError(f"summarize: ppl=exp({loss!r}) не конечен")
    return {
        "loss": float(loss),
        "ppl": float(ppl),
        "windows": len(losses),
        "pred_tokens": total,
    }


def build_report(
    seq: int,
    max_tokens: int,
    checkpoints: Mapping[str, Mapping[str, Any]],
    meta: Mapping[str, Any],
) -> dict[str, Any]:
    """Собрать документ ровно по контракту ``axiom-eval-checkpoint/1``."""
    return {
        "schema": REPORT_SCHEMA,
        "seq": int(seq),
        "max_tokens": int(max_tokens),
        "checkpoints": {
            str(name): {
                "path": str(entry["path"]),
                "domain": dict(entry["domain"]),
                "w_holdout": (
                    None if entry.get("w_holdout") is None else dict(entry["w_holdout"])
                ),
            }
            for name, entry in checkpoints.items()
        },
        "meta": dict(meta),
    }


# --------------------------------------------------------------------------- #
# Входы (stdlib + numpy только в plan-only/чтении данных)
# --------------------------------------------------------------------------- #


def sha256_file(path: str | Path) -> str:
    """sha256 файла потоково (1 МиБ блоками); OSError пробрасывается."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_config_document(path: str | Path = CONFIG_PATH) -> dict[str, Any]:
    """Прочитать ``net/config.json`` как словарь (без импорта ``net``).

    В plan-only это единственный источник ``vocab_size``/``tokenizer_hash``;
    импорт ``net.config`` тянет JAX и запрещён режимом.  Нечитаемый конфиг —
    отказ: без vocab нельзя проверить диапазон id.
    """
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise InputError(f"{path}: конфиг не читается ({type(exc).__name__}): {exc}") from exc
    if not isinstance(data, dict):
        raise InputError(f"{path}: конфиг обязан быть JSON-объектом")
    return data


def domain_ids_sha256(ids: Any) -> str:
    """sha256 сырых id (uint32 little-endian), без заголовка ``.npy``.

    Конвенция репозитория (``run_a4_pipeline.dataset_digest``): хешируются
    байты значений, а не контейнер — два одинаковых набора в разных ``.npy``
    дают один хеш.  ``memoryview`` не копирует массив.
    """
    import numpy as np

    arr = np.ascontiguousarray(ids)
    if arr.dtype != np.dtype("uint32"):
        raise InputError(
            f"domain_ids_sha256: dtype={arr.dtype} — ожидался uint32"
        )
    raw = arr.astype("<u4", copy=False)
    return hashlib.sha256(memoryview(np.ascontiguousarray(raw)).cast("B")).hexdigest()


def config_vocab_size(config: Mapping[str, Any]) -> int:
    """``vocab_size`` из конфига; отсутствие/мусор — fail-closed."""
    vocab = config.get("vocab_size")
    if not isinstance(vocab, int) or isinstance(vocab, bool) or vocab <= 0:
        raise InputError(f"config: vocab_size={vocab!r} — ожидалось целое > 0")
    return int(vocab)


def load_domain_ids(path: str | Path, vocab_size: int):
    """Загрузить и провалидировать доменный holdout ``.npy`` (uint32, 1-D).

    Проверки fail-closed: файл существует/читается как ``.npy`` без pickle;
    ndim == 1; непусто; dtype ровно ``uint32``; ``max(id) < vocab_size``.
    """
    import numpy as np

    path = Path(path)
    if not path.is_file():
        raise InputError(f"--domain-ids: файл не найден: {path}")
    try:
        ids = np.load(path, allow_pickle=False)
    except (OSError, ValueError, EOFError) as exc:
        raise InputError(
            f"--domain-ids: {path} не читается как .npy ({type(exc).__name__}): {exc}"
        ) from exc
    if not isinstance(ids, np.ndarray):
        raise InputError(f"--domain-ids: {path} дал не массив: {type(ids).__name__}")
    if ids.ndim != 1:
        raise InputError(
            f"--domain-ids: {path}: ndim={ids.ndim} — ожидался 1-D поток id"
        )
    if ids.size == 0:
        raise InputError(f"--domain-ids: {path}: пустой набор")
    if ids.dtype != np.dtype("uint32"):
        raise InputError(
            f"--domain-ids: {path}: dtype={ids.dtype} — ожидался uint32"
        )
    max_id = int(ids.max())
    if max_id >= int(vocab_size):
        raise InputError(
            f"--domain-ids: {path}: max id {max_id} >= vocab_size {vocab_size}"
        )
    return ids


def inspect_w_shard(path: str | Path) -> tuple[int, int]:
    """Проверить packed-шард по размеру; вернуть ``(байты, токены)``.

    Запись — uint32 LE, поэтому размер обязан быть положительным и кратным 4;
    содержимое в plan-only не читается (проверяются размеры, не данные).
    """
    path = Path(path)
    if not path.is_file():
        raise InputError(f"--w-shard: файл не найден: {path}")
    size = path.stat().st_size
    if size == 0:
        raise InputError(f"--w-shard: {path}: пустой файл")
    if size % TOKEN_BYTES:
        raise InputError(
            f"--w-shard: {path}: размер {size} не кратен {TOKEN_BYTES} "
            "(uint32 little-endian)"
        )
    return size, size // TOKEN_BYTES


def load_w_tokens(path: str | Path, max_tokens: int):
    """Прочитать первые ``min(всего, max_tokens)`` uint32 LE токенов шарда."""
    import numpy as np

    _size, total = inspect_w_shard(path)
    take = min(total, int(max_tokens))
    try:
        ids = np.fromfile(path, dtype="<u4", count=take)
    except (OSError, ValueError) as exc:
        raise InputError(
            f"--w-shard: {path} не читается ({type(exc).__name__}): {exc}"
        ) from exc
    if ids.size != take:
        raise InputError(
            f"--w-shard: {path}: прочитано {ids.size} токенов вместо {take} "
            "(файл изменился во время чтения?)"
        )
    return ids


# --------------------------------------------------------------------------- #
# Метаданные и план
# --------------------------------------------------------------------------- #


def git_sha(repo_root: str | Path) -> str | None:
    """``git rev-parse HEAD``; нет git/репозитория — ``None`` (не выдумывать)."""
    try:
        proc = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=_GIT_TIMEOUT_SECS,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    sha = proc.stdout.strip()
    return sha or None


def build_meta(
    *,
    w_shard_path: str | Path | None,
    w_shard_bytes: int,
    domain_ids: Any,
    config_path: str | Path,
    config: Mapping[str, Any],
    repo_root: str | Path,
    domain_counts: Mapping[str, int],
    w_counts: Mapping[str, int] | None,
) -> dict[str, Any]:
    """Собрать ``meta``: только из доступных артефактов, иначе ``null``.

    ``domain_ids_sha256`` — sha256 сырых значений id (uint32 LE), не байтов
    ``.npy``-контейнера: хеш описывает набор токенов, а не способ упаковки.
    ``tokenizer_hash`` — пин ``net/config.json``.  Хвосты окон фиксируются
    здесь же (``*_tail_tokens``).
    """
    tokenizer_hash = config.get("tokenizer_hash")
    if not isinstance(tokenizer_hash, str) or not tokenizer_hash.strip():
        tokenizer_hash = None

    try:
        config_sha256: str | None = sha256_file(config_path)
    except OSError:
        config_sha256 = None

    domain_sha256 = domain_ids_sha256(domain_ids)

    w_tail = int(w_counts["tail_tokens"]) if w_counts is not None else 0
    w_tokens = int(w_counts["tokens"]) if w_counts is not None else 0

    return {
        "w_shard": Path(w_shard_path).name if w_shard_path is not None else "none",
        "w_shard_bytes": int(w_shard_bytes),
        "domain_ids_sha256": domain_sha256,
        "git_sha": git_sha(repo_root),
        "tokenizer_hash": tokenizer_hash,
        "config_sha256": config_sha256,
        "domain_tokens": int(domain_counts["tokens"]),
        "domain_tail_tokens": int(domain_counts["tail_tokens"]),
        "w_tokens": w_tokens,
        "w_tail_tokens": w_tail,
    }


def build_plan(
    *,
    seq: int,
    max_tokens: int,
    checkpoints: Mapping[str, Mapping[str, str]],
    domain_path: str | Path,
    domain_counts: Mapping[str, int],
    w_shard_path: str | Path | None,
    w_counts: Mapping[str, int] | None,
    meta: Mapping[str, Any],
) -> dict[str, Any]:
    """JSON-план ``--plan-only``: те же верхнеуровневые ключи, без метрик.

    План не притворяется отчётом: ``mode="plan"``, у чекпоинтов есть только
    пути, а счётчики окон лежат в ``datasets``.  Схема та же
    (``axiom-eval-checkpoint/1``), чтобы потребитель мог отличить план от
    результата по ``mode``, не заводя второй контракт.
    """
    return {
        "schema": REPORT_SCHEMA,
        "mode": "plan",
        "seq": int(seq),
        "max_tokens": int(max_tokens),
        "checkpoints": {
            str(name): {
                "path": str(entry["path"]),
                "params": str(entry["params"]),
                "params_present": bool(entry.get("params_present", True)),
            }
            for name, entry in checkpoints.items()
        },
        "datasets": {
            "domain": {"path": str(domain_path), **dict(domain_counts)},
            "w_holdout": (
                None
                if w_counts is None
                else {"path": str(w_shard_path), **dict(w_counts)}
            ),
        },
        "meta": dict(meta),
    }


def _validate_numbers(args: argparse.Namespace) -> None:
    """Fail-closed проверка числовых аргументов до чтения данных."""
    if args.seq < 2:
        raise InputError(f"--seq={args.seq}: нужно >= 2 (хотя бы один предсказанный токен)")
    if args.max_tokens <= 0:
        raise InputError(f"--max-tokens={args.max_tokens}: нужно > 0")
    if args.max_tokens < args.seq:
        raise InputError(
            f"--max-tokens={args.max_tokens} < --seq={args.seq}: "
            "не составится ни одного полного окна"
        )
    if args.ce_chunk_tokens <= 0:
        raise InputError(
            f"--ce-chunk-tokens={args.ce_chunk_tokens}: нужно > 0 "
            "(0 выключил бы chunked CE, а формула NTP требует его)"
        )
    if args.chunk_size <= 0:
        raise InputError(f"--chunk-size={args.chunk_size}: нужно > 0")


def _require_stream(counts: Mapping[str, int], label: str, path: str | Path) -> None:
    """Источник обязан дать хотя бы одно полное окно."""
    if int(counts["windows"]) <= 0:
        raise InputError(
            f"{label}: {path}: {counts['tokens']} токенов не дают ни одного "
            "полного окна оценки"
        )


# --------------------------------------------------------------------------- #
# Оценка (ленивый JAX)
# --------------------------------------------------------------------------- #


def _evaluate_windows(loss_fn, params, windows: Sequence[Any], label: str) -> dict[str, Any]:
    """Последовательно оценить окна; NaN/Inf loss — немедленный отказ."""
    import jax.numpy as jnp

    losses: list[float] = []
    counts: list[int] = []
    for index, window in enumerate(windows):
        ids = jnp.asarray(window, dtype=jnp.int32)
        loss = float(loss_fn(params, ids))
        if not math.isfinite(loss):
            raise EvalError(
                f"{label}: окно {index}/{len(windows)}: loss={loss!r} — "
                "NaN/Inf запрещён (fail-closed)"
            )
        losses.append(loss)
        counts.append(int(len(window)) - 1)
    return summarize(losses, counts)


def evaluate_checkpoints(
    *,
    specs: Sequence[tuple[str, Path]],
    domain_ids,
    w_ids,
    seq: int,
    max_tokens: int,
    ce_chunk_tokens: int,
    chunk_size: int,
) -> dict[str, dict[str, Any]]:
    """Посчитать domain + W для каждого чекпоинта (JAX импортируется здесь).

    Порядок обязателен: ``ensure_mem_fraction()`` до ``import jax`` (ADR-041),
    затем ``XLA_PYTHON_CLIENT_PREALLOCATE=false``.  JIT-функция принимает
    ``params`` аргументом (а не замыканием), поэтому один и тот же скомпилиро-
    ванный граф переиспользуется всеми чекпоинтами той же формы/dtype.
    """
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

    # ADR-041: лимит XLA выставляется ДО инициализации рантайма JAX.
    from tools.jax_preflight import ensure_mem_fraction

    ensure_mem_fraction()

    import dataclasses

    import jax
    import jax.numpy as jnp

    from net import model
    from net.checkpoint import load_checkpoint
    from net.config import load_config

    cfg = load_config(CONFIG_PATH)
    cfg = dataclasses.replace(
        cfg,
        grad_ckpt_policy="none",
        remat_policy="none",
        ce_chunk_tokens=int(ce_chunk_tokens),
    )

    domain_used = int(min(len(domain_ids), max_tokens))
    domain_windows, domain_tail = window_ids(domain_ids[:domain_used], seq)
    expected_domain = plan_stream_counts(len(domain_ids), seq, max_tokens)
    if expected_domain["windows"] != len(domain_windows) or expected_domain["tail_tokens"] != domain_tail:
        raise EvalError("внутренняя ошибка: окна домена разошлись с планом")

    w_windows: list[Any] | None = None
    if w_ids is not None:
        w_used = int(min(len(w_ids), max_tokens))
        w_windows, w_tail = window_ids(w_ids[:w_used], seq)
        expected_w = plan_stream_counts(len(w_ids), seq, max_tokens)
        if expected_w["windows"] != len(w_windows) or expected_w["tail_tokens"] != w_tail:
            raise EvalError("внутренняя ошибка: окна W разошлись с планом")

    def _loss(params, ids):
        hidden = model.forward(
            params,
            cfg,
            ids[None, :],
            chunk_size=int(chunk_size),
            grad_ckpt_policy="none",
            remat_policy="none",
            emit_logits=False,
            return_hidden=True,
        )
        return model._chunked_cross_entropy(
            hidden[:, :-1],
            ids[None, 1:],
            params.embedding,
            int(ce_chunk_tokens),
        )

    # JIT — один раз на форму; все окна имеют форму (1, seq).
    window_loss = jax.jit(_loss)

    # Структура параметров — из схемы; значения не важны, их заменит restore.
    structure = model.init_params(jax.random.PRNGKey(0), cfg)

    results: dict[str, dict[str, Any]] = {}
    for name, path in specs:
        params_dir = path / PARAMS_DIRNAME
        if not path.is_dir():
            raise InputError(f"чекпоинт {name!r}: каталог не найден: {path}")
        if not params_dir.is_dir():
            raise InputError(f"чекпоинт {name!r}: нет каталога params: {params_dir}")
        params = load_checkpoint(params_dir, target=structure)
        domain_stats = _evaluate_windows(
            window_loss, params, domain_windows, f"{name}/domain"
        )
        w_stats = (
            None
            if w_windows is None
            else _evaluate_windows(window_loss, params, w_windows, f"{name}/w_holdout")
        )
        results[name] = {
            "path": str(path),
            "domain": domain_stats,
            "w_holdout": w_stats,
        }
    return results


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    """CLI по контракту задачи (argparse, без top-level импорта jax)."""
    parser = argparse.ArgumentParser(
        description=(
            "ADR-055: NTP-PPL оценка чекпоинтов Axiom на доменном holdout "
            "и неиспользованном W-shard (схема axiom-eval-checkpoint/1)."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--ckpt",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="каталог чекпоинта (внутри params/); можно повторять",
    )
    parser.add_argument(
        "--domain-ids",
        type=Path,
        default=None,
        metavar="PATH.npy",
        help="чистый доменный holdout: uint32 .npy, 1-D, BOS/EOS включены",
    )
    parser.add_argument(
        "--w-shard",
        type=Path,
        default=None,
        metavar="PATH.bin",
        help="packed uint32 LE shard общего корпуса (необязателен)",
    )
    parser.add_argument("--seq", type=int, default=2048, help="длина окна NTP")
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=131072,
        help="потолок токенов на источник",
    )
    parser.add_argument(
        "--ce-chunk-tokens",
        type=int,
        default=2048,
        help="ширина chunked cross-entropy (строк на чанк)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=32,
        help="chunk_size forward (KDA/MLA)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        metavar="PATH.json",
        help="записать отчёт/план в файл (JSON всегда печатается в stdout)",
    )
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="проверить входы и напечатать план без импорта jax/загрузки чекпоинтов",
    )
    return parser


def _emit(document: Mapping[str, Any], out: str | Path | None) -> None:
    """Напечатать JSON в stdout и (опционально) записать файл."""
    text = json.dumps(document, ensure_ascii=False, indent=2)
    print(text)
    if out is not None:
        Path(out).write_text(text + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    """Точка входа; коды: 0 — успех, 1 — сбой оценки, 2 — отвергнутый вход."""
    args = build_parser().parse_args(argv)

    try:
        _validate_numbers(args)
        if args.domain_ids is None:
            raise InputError("--domain-ids обязателен (чистый доменный holdout)")
        specs = parse_ckpt_spec(args.ckpt)

        config = read_config_document(CONFIG_PATH)
        vocab_size = config_vocab_size(config)
        domain_ids = load_domain_ids(args.domain_ids, vocab_size)

        domain_counts = plan_stream_counts(len(domain_ids), args.seq, args.max_tokens)
        _require_stream(domain_counts, "--domain-ids", args.domain_ids)

        w_shard_bytes = 0
        w_counts: dict[str, int] | None = None
        w_ids = None
        if args.w_shard is not None:
            w_shard_bytes, w_total = inspect_w_shard(args.w_shard)
            w_counts = plan_stream_counts(w_total, args.seq, args.max_tokens)
            _require_stream(w_counts, "--w-shard", args.w_shard)

        meta = build_meta(
            w_shard_path=args.w_shard,
            w_shard_bytes=w_shard_bytes,
            domain_ids=domain_ids,
            config_path=CONFIG_PATH,
            config=config,
            repo_root=REPO_ROOT,
            domain_counts=domain_counts,
            w_counts=w_counts,
        )

        if args.plan_only:
            checkpoints: dict[str, dict[str, Any]] = {}
            for name, path in specs:
                params_dir = path / PARAMS_DIRNAME
                if not path.is_dir():
                    raise InputError(
                        f"чекпоинт {name!r}: каталог не найден: {path}"
                    )
                # План проверяет существование чекпоинта, а не его грузимость:
                # сам прогон потребует ``params/`` (fail-closed ниже), но
                # dummy-каталог CPU/CI-плана не обязан содержать Orbax-дерево.
                params_present = params_dir.is_dir()
                if not params_present:
                    print(
                        f"[eval-checkpoint] предупреждение: чекпоинт {name!r}: "
                        f"нет каталога params ({params_dir}) — план возможен, "
                        "прогон без него откажет",
                        file=sys.stderr,
                    )
                checkpoints[name] = {
                    "path": str(path),
                    "params": str(params_dir),
                    "params_present": params_present,
                }
            document = build_plan(
                seq=args.seq,
                max_tokens=args.max_tokens,
                checkpoints=checkpoints,
                domain_path=args.domain_ids,
                domain_counts=domain_counts,
                w_shard_path=args.w_shard,
                w_counts=w_counts,
                meta=meta,
            )
        else:
            if args.w_shard is not None:
                w_ids = load_w_tokens(args.w_shard, args.max_tokens)
                max_w_id = int(w_ids.max())
                if max_w_id >= vocab_size:
                    raise InputError(
                        f"--w-shard: max id {max_w_id} >= vocab_size {vocab_size}"
                    )
            results = evaluate_checkpoints(
                specs=specs,
                domain_ids=domain_ids,
                w_ids=w_ids,
                seq=args.seq,
                max_tokens=args.max_tokens,
                ce_chunk_tokens=args.ce_chunk_tokens,
                chunk_size=args.chunk_size,
            )
            document = build_report(
                seq=args.seq,
                max_tokens=args.max_tokens,
                checkpoints=results,
                meta=meta,
            )

        _emit(document, args.out)
        return EXIT_OK

    except InputError as exc:
        print(f"[eval-checkpoint] вход отвергнут: {exc}", file=sys.stderr)
        return EXIT_INPUT_ERROR
    except EvalError as exc:
        print(f"[eval-checkpoint] оценка остановлена: {exc}", file=sys.stderr)
        return EXIT_EVAL_ERROR
    except Exception as exc:  # noqa: BLE001 — fail-closed: любой сбой = ненулевой код
        import traceback

        print(
            f"[eval-checkpoint] сбой ({type(exc).__name__}): {exc}",
            file=sys.stderr,
        )
        traceback.print_exc(file=sys.stderr)
        return EXIT_EVAL_ERROR


if __name__ == "__main__":
    raise SystemExit(main())
