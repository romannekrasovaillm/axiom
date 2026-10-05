"""PPL-прибор претрейн-чекпойнта (SFT-STAGE §1.1 п.2).

Перплексия модели на пиннутом holdout-корпусе: teacher-forcing, суммарный NLL /
число предсказанных токенов, ``exp``.  Число — база сравнения с пост-SFT
(различимость > SE; урок «PPL ×0.99» — артефакт утечки, поэтому holdout собирает
:mod:`env.eval_holdout` с фильтром 12-граммового пересечения).

Ядро (:func:`perplexity`, :func:`nll_from_logits`) считает только числовую часть
и не зависит от ML-стека — на нём проверяется формула на mock-модели (равномерное
распределение над словарём 100 даёт PPL ≈ 100).  Модельная часть
(:func:`run_ppl`) лениво поднимает ``jax``/``net`` и при их отсутствии даёт
понятную :class:`~env.jaxlm_adapter.JaxUnavailableError`, а не падает на импорте.

Батч-обработка: документы дополняются справа pad-токеном, лосс считается только
на реальных токенах.  Для причинной LM правое дополнение безопасно — реальные
токены не видят padding впереди, поэтому PPL не зависит от размера батча.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from .jaxlm_adapter import (  # noqa: E402
    JaxUnavailableError,
    load_pretrain_model,
)

#: Идентификатор pad-токена словаря сети (``net.tokenizer.SPECIAL_TOKENS[0]``).
DEFAULT_PAD_ID = 0
#: Дефолтный размер батча (1 — самый совместимый; GPU-прогон может поднять).
DEFAULT_BATCH_SIZE = 1


class PplError(ValueError):
    """Вход непригоден для замера."""


# --------------------------------------------------------------------------- #
# Ядро: числовая формула PPL (без ML-стека)
# --------------------------------------------------------------------------- #


def _log_softmax(logits: Any):
    """``log_softmax`` по последней оси.  numpy — если есть, иначе чистый Python.

    Возвращает ``(values, numpy_array_or_None)``: numpy-ветка предпочтительна
    на реальных прогонах (словарь 160K), чистый Python покрывает минимальное
    окружение и mock-тесты.
    """
    try:
        import numpy as np
    except ImportError:  # pragma: no cover — окружение без numpy
        np = None
    if np is not None:
        arr = np.asarray(logits, dtype=np.float64)
        m = arr.max(axis=-1, keepdims=True)
        lse = m + np.log(np.exp(arr - m).sum(axis=-1, keepdims=True))
        return arr - lse, np
    rows = []
    for batch in logits:
        out_batch = []
        for row in batch:
            m = max(row)
            s = sum(math.exp(x - m) for x in row)
            logsum = m + math.log(s) if s > 0 else 0.0
            out_batch.append([x - logsum for x in row])
        rows.append(out_batch)
    return rows, None


def nll_from_logits(
    logits: Any, ids: Any, mask: Any | None = None
) -> tuple[float, int]:
    """Суммарный NLL и число предсказанных токенов (teacher forcing).

    ``logits`` — (B, T, V); ``ids`` — (B, T); ``mask`` — (B, T), 1 на реальном
    токене.  Позиция ``t`` предсказывается логитами ``t-1``; позиция 0 не
    предсказывается (для неё нет предыдущего контекста).
    """
    values, np = _log_softmax(logits)
    nll = 0.0
    n_tokens = 0
    if np is not None:
        ids_arr = np.asarray(ids, dtype=np.int64)
        targets = ids_arr[:, 1:]
        preds = values[:, :-1]
        gathered = np.take_along_axis(preds, targets[:, :, None], axis=-1)[:, :, 0]
        if mask is None:
            valid = np.ones_like(targets, dtype=bool)
        else:
            valid = np.asarray(mask)[:, 1:].astype(bool)
        nll = float(-gathered[valid].sum())
        n_tokens = int(valid.sum())
        return nll, n_tokens
    for b, row_ids in enumerate(ids):  # pragma: no cover — fallback без numpy
        row_mask = mask[b] if mask is not None else [1] * len(row_ids)
        for t in range(1, len(row_ids)):
            if not row_mask[t]:
                continue
            nll -= values[b][t - 1][int(row_ids[t])]
            n_tokens += 1
    return nll, n_tokens


def compute_ppl(nll_sum: float, n_tokens: int) -> float:
    """``exp(NLL / N)``; пустой набор — ошибка (делить не на что)."""
    if n_tokens <= 0:
        raise PplError("n_tokens = 0: перплексия не определена (пустой holdout)")
    return math.exp(float(nll_sum) / int(n_tokens))


def _batches(
    sequences: Sequence[Sequence[int]], batch_size: int
) -> Iterator[list[Sequence[int]]]:
    """Батчи документов (по возрастанию длины: меньше padding)."""
    ordered = sorted(sequences, key=lambda s: (len(s), list(s)))
    for start in range(0, len(ordered), max(1, batch_size)):
        yield list(ordered[start : start + max(1, batch_size)])


def perplexity(
    logits_fn: Callable[[Any], Any],
    sequences: Sequence[Sequence[int]],
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
    pad_id: int = DEFAULT_PAD_ID,
) -> dict[str, Any]:
    """PPL набора токенизированных документов.

    ``logits_fn`` принимает 2D-батч id (список списков) и возвращает логиты
    (B, T, V) — numpy-массив или вложенный список.  Пропуски справа маскируются.
    """
    if not sequences:
        raise PplError("пустой набор последовательностей")
    total_nll = 0.0
    total_tokens = 0
    batch_count = 0
    for batch in _batches(sequences, batch_size):
        width = max(len(seq) for seq in batch)
        padded = [list(seq) + [int(pad_id)] * (width - len(seq)) for seq in batch]
        mask = [[1] * len(seq) + [0] * (width - len(seq)) for seq in batch]
        logits = logits_fn(padded)
        batch_nll, batch_tokens = nll_from_logits(logits, padded, mask)
        total_nll += batch_nll
        total_tokens += batch_tokens
        batch_count += 1
    return {
        "ppl": compute_ppl(total_nll, total_tokens),
        "nll_sum": total_nll,
        "n_tokens": total_tokens,
        "n_texts": len(sequences),
        "batches": batch_count,
    }


# --------------------------------------------------------------------------- #
# Модельная часть
# --------------------------------------------------------------------------- #


def read_holdout_texts(path: str | Path) -> list[str]:
    """Тексты holdout-jsonl ``{text, source_path}`` (контракт eval_holdout)."""
    p = Path(path)
    if not p.is_file():
        raise PplError(f"holdout не найден: {p}")
    texts: list[str] = []
    for lineno, line in enumerate(p.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise PplError(f"{p}:{lineno}: строка не разбирается как JSON: {exc}") from exc
        text = record.get("text") if isinstance(record, dict) else None
        if not isinstance(text, str) or not text:
            raise PplError(f"{p}:{lineno}: нет непустого строкового поля text")
        texts.append(text)
    if not texts:
        raise PplError(f"{p}: holdout пуст (fail-closed)")
    return texts


def run_ppl(
    checkpoint: str | Path,
    holdout: str | Path,
    tokenizer_path: str | Path | None,
    *,
    config: Any,
    batch_size: int = DEFAULT_BATCH_SIZE,
    seed: int = 0,
    max_texts: int | None = None,
    add_bos: bool = True,
    chunk_size: int = 64,
) -> dict[str, Any]:
    """Перплексия чекпойнта на holdout.  Требует ``jax``/``net`` (иначе — отказ)."""
    try:
        import jax.numpy as jnp
        import numpy as np
        from net import infer
    except ImportError as exc:
        raise JaxUnavailableError(
            "PPL-прибору нужен jax/net (venv-axiom с jax). "
            f"Импорт упал: {exc}"
        ) from exc

    model = load_pretrain_model(
        checkpoint, config, tokenizer_path=tokenizer_path, seed=seed
    )
    if model.tokenizer is None:
        raise PplError("нужен --tokenizer: без токенизатора holdout не закодировать")

    texts = read_holdout_texts(holdout)
    if max_texts is not None:
        texts = texts[: int(max_texts)]
    bos_id = int(getattr(infer, "BOS_ID", 1))
    sequences: list[list[int]] = []
    for text in texts:
        ids = [int(t) for t in model.tokenizer.encode(text)]
        if add_bos:
            ids = [bos_id] + ids
        if len(ids) >= 2:
            sequences.append(ids)
    if not sequences:
        raise PplError("после токенизации нет последовательностей длиной >= 2")

    params = model.params
    cfg = model.config

    def logits_fn(batch: Any) -> Any:
        x = jnp.asarray(batch, dtype=jnp.int32)
        logits = infer.model_mod.forward(params, cfg, x, chunk_size=chunk_size)
        return np.asarray(logits)

    report = perplexity(logits_fn, sequences, batch_size=batch_size)
    report["checkpoint_sha256"] = model.checkpoint_sha256
    report["add_bos"] = bool(add_bos)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="PPL претрейн-чекпойнта на holdout")
    parser.add_argument("--checkpoint", required=True, help="каталог чекпойнта")
    parser.add_argument("--holdout", required=True, help="eval_holdout.jsonl")
    parser.add_argument("--tokenizer", default=None, help="путь к токенизатору (tokens-v2)")
    parser.add_argument(
        "--config", default=None,
        help="ModelConfig: путь к JSON или пресет (по умолчанию config.json репо)",
    )
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-texts", type=int, default=None)
    parser.add_argument("--no-bos", dest="add_bos", action="store_false", default=True)
    parser.add_argument("--chunk-size", type=int, default=64)
    args = parser.parse_args(argv)

    config = args.config or str(_REPO_ROOT / "net" / "config.json")
    try:
        report = run_ppl(
            args.checkpoint,
            args.holdout,
            args.tokenizer,
            config=config,
            batch_size=args.batch_size,
            seed=args.seed,
            max_texts=args.max_texts,
            add_bos=args.add_bos,
            chunk_size=args.chunk_size,
        )
    except (JaxUnavailableError, PplError, FileNotFoundError, ValueError) as exc:
        print(json.dumps({"error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 2
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
