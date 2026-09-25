"""Стыковочный адаптер net/ ↔ environment v1: сеть как исполнитель задачи.

Интерфейс — по образцу stub_model.py (``run_*`` → ``*Run`` с final_ws и
токенами): токенизация промпта задачи (net/tokenizer.py, byte-level BPE) →
генерация (net/infer.py; seed — из task spec, temperature/top_p — из
decoding-параметров манифеста) → детокенизация → ответ исполнителя в среду.

Модель необучена (REQ-001: веса инициализируются с нуля), содержательное
решение задачи не ожидается: валиден сам провод «генерация → ответ →
verifier → вердикт → манифест». Пиннинг (AD-4): снапшот модели — sha256
JSON-конфигурации, routing seed — ``cfg.routing_seed``, decoding-параметры —
полностью в манифесте §8.

jax и пакет net импортируются лениво внутри функций: остальная среда
(calibrate, verifier, CLI) обязана работать без ML-стека.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from . import generate as generate_mod
from . import manifest as manifest_mod
from . import run as run_mod
from .util import copy_case_snapshot, sha256_text, tree_sha256
from .verifier import arch_ml_bin, arch_ml_build_hash

# Ответ исполнителя материализуется файлом в workspace (для restore-gates;
# для keep-gates-implement ответ пишется в IMPLEMENTATION.md задачи).
RESPONSE_FILE = "MODEL-RESPONSE.md"


@dataclass
class NetRun:
    final_ws: Path
    tokens_in: int
    tokens_out: int
    spent_tokens: int
    response_text: str
    generated_ids: list[int]


def model_snapshot_sha256(cfg) -> str:
    """Хеш снапшота модели: sha256 канонического JSON конфигурации (AD-4)."""
    return sha256_text(json.dumps(cfg.as_dict(), sort_keys=True))


def run_net_model(
    task_spec: dict,
    base_ws: Path,
    out_dir: Path,
    *,
    params,
    cfg,
    tokenizer,
    decoding: dict,
    max_new_tokens: int = 64,
) -> NetRun:
    """Прогон сети-исполнителя: base_ws → final_ws (детерминированно).

    ``decoding`` — decoding-параметры манифеста §8: temperature, top_p, seed
    (seed совпадает с seed task spec — выставляет вызывающая сторона).
    """
    from net import infer  # ленивый импорт: среда без ML-стека остаётся лёгкой

    copy_case_snapshot(base_ws, out_dir)

    prompt = str(task_spec.get("prompt", ""))
    token_ids = tokenizer.encode(prompt)
    response_text, out_ids = infer.generate_text(
        params, cfg, tokenizer, prompt, max_new_tokens,
        float(decoding["temperature"]), int(decoding["seed"]),
        top_p=float(decoding.get("top_p", 1.0)),
    )

    kind = task_spec.get("objective", {}).get("kind", "restore-gates")
    target = out_dir / (generate_mod.REAL_IMPL_FILE if kind == "keep-gates-implement" else RESPONSE_FILE)
    target.write_text(response_text or "\n", encoding="utf-8")

    tokens_in, tokens_out = len(token_ids), len(out_ids)
    return NetRun(
        final_ws=out_dir,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        spent_tokens=tokens_in + tokens_out,
        response_text=response_text,
        generated_ids=out_ids,
    )


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def run_net_task(
    task_spec: dict,
    base_ws: Path,
    out_dir: Path,
    *,
    params,
    cfg,
    tokenizer,
    decoding: dict,
    max_new_tokens: int = 64,
    bin: Optional[str] = None,
    hidden_constraints: Optional[Path] = None,
    run_id: Optional[str] = None,
    started: Optional[str] = None,
    finished: Optional[str] = None,
    model_base: str = "net-l3-skeleton-untrained",
    host: str = "net-executor",
) -> tuple[run_mod.RunResult, dict]:
    """Сквозной прогон: исполнитель → вердикт/награда → Run Manifest §8.

    ``run_id``/``started``/``finished`` инъецируемы — детерминированные
    прогоны (A5: повтор с тем же seed → идентичный манифест) передают
    фиксированные значения.
    """
    b = bin or arch_ml_bin()
    net_run = run_net_model(
        task_spec, base_ws, out_dir,
        params=params, cfg=cfg, tokenizer=tokenizer,
        decoding=decoding, max_new_tokens=max_new_tokens,
    )
    rr = run_mod.evaluate_run(
        task_spec, base_ws, out_dir, net_run.spent_tokens,
        bin=b, hidden_constraints=hidden_constraints,
    )
    now = _now_iso()
    dec = {
        "temperature": float(decoding["temperature"]),
        "top_p": float(decoding.get("top_p", 1.0)),
        "seed": int(decoding["seed"]),
        "max_new_tokens": int(max_new_tokens),
    }
    m = manifest_mod.build_manifest(
        run_id=run_id or str(uuid.uuid4()),
        task_spec=task_spec,
        arch_ml_build=arch_ml_build_hash(b),
        constraints_sha256=manifest_mod_sha256(out_dir, task_spec),
        hidden_constraints_sha256=task_spec["verifier"]["hidden_constraints_sha256"],
        workspace_sha256=tree_sha256(out_dir),
        model_base=model_base,
        model_snapshot_sha256=model_snapshot_sha256(cfg),
        decoding=dec,
        attempts_used=1,
        usage={
            "tokens_in": net_run.tokens_in,
            "tokens_out": net_run.tokens_out,
            "cost_usd": 0.0,
            "host": host,
        },
        timing={
            "started": started or now,
            "finished": finished or now,
            "resume_count": 0,
        },
        verdict={
            "pass": rr.verdict.passed,
            "reward": rr.reward.to_manifest_dict(),
            "issues_warn": rr.verdict.warn_issues,
        },
    )
    m["model"]["routing_seed"] = int(cfg.routing_seed)
    return rr, m


def manifest_mod_sha256(ws_dir: Path, task_spec: dict) -> str:
    """sha256 файла CONSTRAINTS финального workspace (пиннинг gates_version)."""
    from .util import sha256_file

    return sha256_file(ws_dir / task_spec["verifier"]["constraints"])
