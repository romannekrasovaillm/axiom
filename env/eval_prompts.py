"""Проба генерации претрейн-чекпойнта: 20 фиксированных промптов (§1.1 п.3).

Справочный (не вердиктный) прибор: список констант :data:`PROMPTS` прогоняется
через JaxLM-адаптер (:mod:`env.jaxlm_adapter`, §13) в штатном режиме декодирования
— greedy + запрет повтора 4-грамм + eos на конце хода (§8.6 SFT-STAGE,
LAG-ADR-040: петли — свойство декодера, не модели: greedy давал 54% петель,
запрет 4-грамм — 0.0%).

Выход — jsonl ``{prompt, generation}``; пиннинг протокола (``protocol.decoding``)
печатается в stdout вместе с sha256 чекпойнта и файла.  Без ``jax``/``net`` —
понятный отказ, а не ImportError.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Sequence

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from .jaxlm_adapter import JaxLMAdapter, JaxUnavailableError  # noqa: E402

#: Протокол декодирования проб (§8.6): фиксируется в отчёте.
DECODING_PROTOCOL: dict[str, Any] = {
    "mode": "greedy",
    "temperature": 0.0,
    "no_repeat_ngram": 4,
    "eos_at_turn_end": True,
}

DEFAULT_MAX_NEW_TOKENS = 128
DEFAULT_OUT = "eval_prompts.jsonl"

#: 20 фиксированных промптов — ML / архитектура / код (русскоязычные).
#: Список — константа модуля: смена набора = смена прибора (пересъёмка базы).
PROMPTS: tuple[str, ...] = (
    "Объясни, чем отличается обучение с учителем от обучения с подкреплением.",
    "Что такое перплексия языковой модели и как её интерпретировать?",
    "Опиши, как работает механизм внимания в трансформере.",
    "В чём разница между dense- и MoE-архитектурами нейросетей?",
    "Что такое градиентный чекпоинтинг и зачем он нужен при обучении больших моделей?",
    "Объясни идею KV-кэша при авторегрессионной генерации.",
    "Что такое архитектурное решение (ADR) и зачем его фиксировать до реализации?",
    "Опиши принцип CQRS и приведи пример применения в банковской системе.",
    "Чем отличается идемпотентный потребитель сообщений от обычного?",
    "Что такое circuit breaker и как он защищает сервис от каскадного отказа?",
    "Объясни разницу между control plane и data plane на примере платёжной системы.",
    "Что такое fitness-функция архитектуры и как её автоматизировать?",
    "Опиши, как организовать трассировку требования до компонента реализации.",
    "В чём смысл паттерна transaction outbox и какую проблему он решает?",
    "Что такое event sourcing и когда его стоит выбирать?",
    "Напиши на Python функцию, которая считает n-граммы в списке токенов.",
    "Напиши на Python функцию бинарного поиска по отсортированному массиву.",
    "Как в Python прочитать JSONL-файл построчно, не загружая его целиком в память?",
    "Напиши SQL-запрос, который находит дубликаты по ключу в таблице пользователей.",
    "Как написать юнит-тест на функцию, которая зависит от текущего времени?",
)


class PromptsError(ValueError):
    """Вход непригоден для пробы генерации."""


def run_prompts(
    checkpoint: str | Path,
    *,
    config: Any,
    tokenizer_path: str | Path | None = None,
    out: str | Path = DEFAULT_OUT,
    seed: int = 0,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    prompts: Sequence[str] = PROMPTS,
) -> dict[str, Any]:
    """Прогнать промпты через адаптер и записать jsonl ``{prompt, generation}``."""
    adapter = JaxLMAdapter(
        checkpoint,
        config,
        tokenizer_path=tokenizer_path,
        seed=seed,
        temperature=DECODING_PROTOCOL["temperature"],
        no_repeat_ngram=DECODING_PROTOCOL["no_repeat_ngram"],
    )
    records: list[dict[str, str]] = []
    for prompt in prompts:
        result = adapter.generate(
            [{"role": "user", "content": prompt}],
            seed=seed,
            max_tokens=max_new_tokens,
        )
        records.append({"prompt": prompt, "generation": result.text})

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(
        json.dumps(record, ensure_ascii=False) + "\n" for record in records
    )
    out_path.write_text(payload, encoding="utf-8")
    return {
        "n_prompts": len(records),
        "seed": int(seed),
        "max_new_tokens": int(max_new_tokens),
        "protocol": {"decoding": dict(DECODING_PROTOCOL)},
        "checkpoint_sha256": adapter.checkpoint_sha256,
        "policy_version": adapter.policy_version,
        "out": str(out_path),
        "sha256": hashlib.sha256(out_path.read_bytes()).hexdigest(),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Проба генерации по 20 фиксированным промптам (SFT-STAGE §1.1)"
    )
    parser.add_argument("--checkpoint", required=True, help="каталог чекпойнта")
    parser.add_argument("--tokenizer", default=None, help="путь к токенизатору (tokens-v2)")
    parser.add_argument(
        "--config", default=None,
        help="ModelConfig: путь к JSON (по умолчанию config.json репо)",
    )
    parser.add_argument("--out", default=DEFAULT_OUT, help="выходной jsonl")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    args = parser.parse_args(argv)

    config = args.config or str(_REPO_ROOT / "net" / "config.json")
    try:
        report = run_prompts(
            args.checkpoint,
            config=config,
            tokenizer_path=args.tokenizer,
            out=args.out,
            seed=args.seed,
            max_new_tokens=args.max_new_tokens,
        )
    except (JaxUnavailableError, PromptsError, FileNotFoundError, ValueError) as exc:
        print(json.dumps({"error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 2
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
