"""Детерминированный фейковый исполнитель (НЕ LLM) для кальбровки и тестов.

Модель-заглушка «решает» задачу по монетам от (task_seed, model_seed, salt):
- restore-gates: откатывает подмножество повреждений (копированием исходных
  файлов из чистого кейса);
- keep-gates-implement: пишет IMPLEMENTATION.md с маркерами задачных тестов.

Используется в calibrate.py и в тестах приёмки §11(5) — детерминированный
аналог агента, без обращения к LLM.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from . import corruption
from .util import copy_case_snapshot, stable_coin


@dataclass
class StubRun:
    final_ws: Path
    tokens_in: int
    tokens_out: int
    spent_tokens: int


def _tokens(task_spec: dict, model_seed: int, damages: int) -> tuple[int, int]:
    """Детерминированные токены прогона (вход/выход)."""
    seed = int(task_spec.get("seed", 0))
    tokens_in = len(str(task_spec.get("prompt", ""))) + 64 + (seed % 512)
    tokens_out = 256 + (model_seed % 4096) + 32 * damages
    return tokens_in, tokens_out


def run_stub_model(
    task_spec: dict,
    base_ws: Path,
    clean_dir: Path,
    out_dir: Path,
    model_seed: int,
    restore_probability: float = 0.5,
    solve_probability: float = 0.5,
    introduce_violation_prob: float = 0.0,
) -> StubRun:
    """Прогон заглушки: base_ws → final_ws (детерминированно)."""
    task_seed = int(task_spec.get("seed", 0))
    kind = task_spec.get("objective", {}).get("kind", "restore-gates")
    level = task_spec.get("difficulty", {}).get("level", "L0")

    copy_case_snapshot(base_ws, out_dir)
    damages: list[corruption.Damage] = []

    if kind == "keep-gates-implement":
        # Реализация: записываем IMPLEMENTATION.md (маркеры — из generate.py).
        from . import generate

        impl = out_dir / generate.REAL_IMPL_FILE
        if stable_coin(task_seed, model_seed, "solve") < solve_probability:
            impl.write_text(generate.real_impl_content(task_spec["id"]), encoding="utf-8")
        else:
            impl.write_text("# Заглушка без маркеров\n", encoding="utf-8")
        n_actions = 1
    else:
        damages = corruption.plan_damages(clean_dir, task_seed, level)
        for i, d in enumerate(damages):
            if stable_coin(task_seed, model_seed, f"restore:{i}") < restore_probability:
                corruption.revert_damage(out_dir, clean_dir, d)
        n_actions = len(damages)

    # Опционально: вносим новое нарушение (для ветки new_violations).
    if introduce_violation_prob > 0.0 and stable_coin(task_seed, model_seed, "new-viol") < introduce_violation_prob:
        _introduce_extra_violation(out_dir, clean_dir, damages)

    tokens_in, tokens_out = _tokens(task_spec, model_seed, n_actions)
    return StubRun(final_ws=out_dir, tokens_in=tokens_in, tokens_out=tokens_out,
                   spent_tokens=tokens_in + tokens_out)


def _introduce_extra_violation(
    ws_dir: Path,
    clean_dir: Path,
    damages: list[corruption.Damage],
) -> None:
    """Ломает ещё одно правило, не входящее в базовые повреждения."""
    damaged_files = {d.file for d in damages}
    adr = corruption._adr_files(clean_dir)
    candidate = next((f for f in adr if f not in damaged_files), None)
    if candidate is None:
        return
    extra = corruption.Damage("remove_adr_section", candidate, "## Reversibility")
    corruption.apply_damage(ws_dir, extra)
