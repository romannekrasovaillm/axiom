"""Вызов единого preflight из стадий конвейера (ADR-038, дельта K3).

Стадия перед стартом обязана пройти preflight своего открывающего гейта
(``model/opening-gates.yaml``): ``unverified``/``fail`` — отказ, пока не дан
аварийный обход ``--override-preflight <причина>`` (обход виден фактом S-033).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[1]


def run_preflight(gate: str, override: Optional[str] = None, root: Optional[str] = None,
                  timeout: int = 120) -> tuple[int, str]:
    cmd = [sys.executable, str(REPO_ROOT / "tools" / "preflight.py"), gate,
           "--root", str(root or REPO_ROOT)]
    if override:
        cmd += ["--override-preflight", override]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.SubprocessError) as exc:
        return 2, f"preflight недоступен: {exc}\n"
    return proc.returncode, (proc.stdout or "") + (proc.stderr or "")


def enforce(gate: str, override: Optional[str] = None, *, no_preflight: bool = False,
            root: Optional[str] = None) -> int:
    """Прогоняет preflight гейта. ``0`` — можно продолжать, иначе — отказ.

    Обход (``override``) разрешает продолжить, но выводит предупреждение; сам
    обход пишется фактом S-033 внутри preflight.
    """
    if no_preflight or not gate:
        return 0
    code, text = run_preflight(gate, override, root=root)
    sys.stdout.write(text)
    if code == 0:
        return 0
    if override:
        print(
            f"ВНИМАНИЕ: preflight гейта «{gate}» обойдён: {override} "
            "(обход зафиксирован фактом S-033)",
            file=sys.stderr,
        )
        return 0
    print(
        f"стадия не стартует: preflight гейта «{gate}» отказал; "
        "аварийный обход — --override-preflight <причина>",
        file=sys.stderr,
    )
    return code or 1


__all__ = ["REPO_ROOT", "run_preflight", "enforce"]
