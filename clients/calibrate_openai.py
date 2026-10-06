"""CLI калибровки лесенки §10 на внешнем endpoint (track-2 Stage A, ADR-033).

Запуск из корня репозитория::

    python3 -m clients.calibrate_openai \
        --tasks env/data --out evidence/track2-stage-a \
        --base-url http://127.0.0.1:8080/v1 --served-model qwen3-4b-instruct

Калибровка идёт ровно тем же путём среды, что и RL-роллауты: ``env.calibrate``
зовёт пиннутый harness-луп §13 (``tools/rollout_harness``), вердикт —
``evaluate_run``, награда — arm-конфиг.  Отличие от ``env.main calibrate``
только в источнике эпизодов: здесь ``model_factory`` строит адаптер внешнего
endpoint'а.

Почему отдельный вход, а не ``--adapter openai`` в ``env.main``: пакет ``env/``
— контур награды RL (C-039/AD-2), транзитный импорт HTTP-клиента внутрь него
запрещён гейтом.  Поэтому CLI, знающий про httpx, живёт вне контура, а ``env/``
принимает готовую фабрику через существующий шов ``model_factory``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from clients.openai_http import build_openai_factory  # noqa: E402
from env.openai_adapter import DEFAULT_BASE_URL, DEFAULT_SERVED_MODEL  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="clients.calibrate_openai",
        description="Калибровка лесенки §10 на OpenAI-совместимом endpoint",
    )
    parser.add_argument("--tasks", type=Path, required=True, help="каталог с public/ и holdout/")
    parser.add_argument("--out", type=Path, required=True, help="каталог evidence/")
    parser.add_argument("--case", type=Path, default=REPO_ROOT, help="каталог чистого кейса")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="base URL endpoint")
    parser.add_argument("--served-model", default=DEFAULT_SERVED_MODEL, help="имя модели в запросе")
    parser.add_argument("--api-key", default=None, help="ключ endpoint, если требуется")
    parser.add_argument("--model-name", default="openai", help="имя модели в отчёте")
    parser.add_argument("--model-seed", type=int, default=7)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    from env import calibrate as calibrate_mod
    from env.verifier import arch_ml_bin

    factory = build_openai_factory(
        base_url=args.base_url,
        model=args.served_model,
        api_key=args.api_key,
        seed=args.model_seed,
    )
    report = calibrate_mod.calibrate(
        args.tasks,
        args.case,
        args.out,
        model_name=args.model_name,
        model_seed=args.model_seed,
        bin=arch_ml_bin(),
        model_factory=factory,
    )
    print(json.dumps({
        "out": str(args.out / f"calibration-{args.model_name}.json"),
        "pass_rate": report["aggregate"]["pass_rate"],
        "ready": report["readiness"]["ready"],
        "tasks": len(report["matrix"][args.model_name]),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
