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

A5: снапшот workspace не несёт байткод-кеш (``__pycache__``/``*.pyc``) — он
маршалит абсолютный путь исходника и сделал бы ``workspace_sha256`` функцией
пути снапшота (``snapshot_workspace``, ``purge_bytecode``).

jax и пакет net импортируются лениво внутри функций: остальная среда
(calibrate, verifier, CLI) обязана работать без ML-стека.
"""

from __future__ import annotations

import json
import os
import shutil
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

# --- A5: снапшот workspace не несёт байткод ---------------------------------
#
# Правила ``command_succeeds`` гейт исполняет **внутри проверяемого workspace**
# (``arch-ml control check``): импорт модулей кейса пишет рядом с исходником
# ``__pycache__/*.pyc``, а байткод маршалит абсолютный путь исходника
# (``co_filename``). Тогда ``workspace_sha256`` = ``tree_sha256(out_dir)``
# становится функцией пути снапшота: два одинаковых прогона в разные каталоги
# (ws-a/ws-b) дают разные манифесты — A5 («повтор прогона → тот же манифест»)
# нарушается. Тот же инвариант в своём процессе держит страж C-042
# (``tools/check_precision_pinning.py``); здесь он обеспечивается для снапшота
# целиком: запрет записи байткода подпроцессам (``PYTHONDONTWRITEBYTECODE``)
# плюс вычистка кеша из снапшота — на копировании и перед хешированием.

BYTECODE_DIR = "__pycache__"
BYTECODE_SUFFIX = ".pyc"
DONT_WRITE_BYTECODE_ENV = "PYTHONDONTWRITEBYTECODE"


def purge_bytecode(root: Path) -> int:
    """Удаляет из дерева ``root`` байткод-кеш (``__pycache__``, ``*.pyc``).

    Возвращает число удалённых файлов. Байткод — артефакт рантайма, а не
    состояние кейса: он несёт mtime исходника и его абсолютный путь, из-за
    чего одно и то же содержимое кейса даёт разные байты в разных каталогах.
    """
    removed = 0
    for cache in sorted(root.rglob(BYTECODE_DIR)):
        if cache.is_dir() and not cache.is_symlink():
            removed += sum(1 for p in cache.rglob("*") if p.is_file())
            shutil.rmtree(cache)
        else:  # симлинк на каталог кеша: снимаем ссылку, цель не трогаем
            cache.unlink()
            removed += 1
    for stray in sorted(root.rglob("*" + BYTECODE_SUFFIX)):
        if stray.is_file() or stray.is_symlink():  # старый layout: .pyc рядом с исходником
            stray.unlink()
            removed += 1
    return removed


def snapshot_workspace(src: Path, dst: Path) -> None:
    """Снапшот workspace прогона: копия кейса без рантайм-мусора (A5).

    Копирование и вычистка байткода — в одном месте: снапшот, по которому
    считается ``workspace_sha256``, не несёт ``__pycache__``/``*.pyc`` даже
    если они были в источнике (гейт исполняется и на базовом кейсе).
    """
    copy_case_snapshot(src, dst)
    purge_bytecode(dst)


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

    snapshot_workspace(base_ws, out_dir)

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
    # A5: до первого подпроцесса (гейты исполняют command_succeeds внутри
    # workspace) запрещаем запись байткода — иначе в снапшоте появится
    # __pycache__ с абсолютным путём каталога прогона (см. блок A5 выше).
    os.environ.setdefault(DONT_WRITE_BYTECODE_ENV, "1")
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
    # Вторая линия A5: снапшот хешируется и отдаётся вызывающему без байткода
    # (переменная выше — первая; подпроцесс мог её не унаследовать).
    purge_bytecode(out_dir)
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
