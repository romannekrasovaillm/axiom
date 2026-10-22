"""Экспортёр S-034 fail_open_audit: аудит fail-open стражей (ADR-038, дельта K4).

Для каждого ``tools/check_*.py`` — проба «отсутствующий вход» (несуществующий
путь, пустой каталог) во временном каталоге, без GPU и сети. Исход: ``exit 0``
при отсутствующем входе = **fail-open** (недоказанное выдано за зелёное);
ненулевой код = fail-closed. Таблица проб — в коде датчика, по одной строке на
стража. Известное: performance-roofline без ``--require-verified`` нейтрален
(fail-open по решению E-3.5); check_gb10_single_load — fail-closed.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Optional

from tools.sensors.protocol import BaseExporter, Fact, SensorSpec

#: Таблица проб: логическое имя → (скрипт, аргументы с ``{tmp}``, таймаут с).
#: Логическое имя совпадает с полем ``guard`` в model/opening-gates.yaml.
PROBES: tuple[tuple[str, str, tuple[str, ...], int], ...] = (
    ("check_adr_config_consistency", "check_adr_config_consistency.py", ("--case", "{tmp}"), 60),
    ("rental-budget-gate", "check_budget_gate.py", ("--verify", "--case-dir", "{tmp}"), 60),
    ("check_claims", "check_claims.py", ("--verify", "--root", "{tmp}"), 60),
    ("check_constraints_meta", "check_constraints_meta.py", ("--case-dir", "{tmp}"), 60),
    ("check_declarative_context", "check_declarative_context.py", ("--net-dir", "{tmp}"), 60),
    ("check_eval_leak", "check_eval_leak.py", ("--eval", "{tmp}/e.jsonl", "--source", "{tmp}/s.jsonl"), 60),
    ("gb10-single-load", "check_gb10_single_load.py", ("--lock-dir", "{tmp}"), 60),
    ("check_handoff_ruleset", "check_handoff_ruleset.py", ("--case", "{tmp}"), 60),
    ("performance-roofline", "check_performance_roofline.py", ("--run", "__absent__", "--metrics", "{tmp}/m.jsonl"), 60),
    ("check_precision_pinning", "check_precision_pinning.py", ("--case-dir", "{tmp}", "--conftest", "{tmp}/conftest.py"), 120),
    ("check_rental_block", "check_rental_block.py", ("--case", "{tmp}"), 60),
    ("check_reward_isolation", "check_reward_isolation.py", ("--root", "{tmp}"), 60),
    ("check_rl_health", "check_rl_health.py", ("--input", "{tmp}/missing.jsonl"), 60),
    ("check_sft_structure", "check_sft_structure.py", ("--input", "{tmp}/missing.jsonl"), 60),
)


def _run_probe(repo_root: Path, tmp: Path, script: str, args: tuple[str, ...], timeout: int) -> dict[str, Any]:
    path = repo_root / "tools" / script
    if not path.is_file():
        return {"probe": "нет файла стража", "exit_code": None, "outcome": "absent"}
    cmd = [sys.executable, str(path)] + [a.replace("{tmp}", str(tmp)) for a in args]
    try:
        proc = subprocess.run(
            cmd, cwd=str(repo_root), capture_output=True, text=True, timeout=timeout,
            env=_clean_env(),
        )
    except subprocess.TimeoutExpired:
        return {"probe": " ".join(cmd[1:]), "exit_code": None, "outcome": "timeout"}
    except OSError as exc:
        return {"probe": " ".join(cmd[1:]), "exit_code": None, "outcome": f"oserror:{exc}"}
    return {"probe": " ".join(cmd[1:]), "exit_code": proc.returncode, "outcome": "ran"}


def _clean_env() -> dict[str, str]:
    import os

    env = dict(os.environ)
    env.pop("NET_GATE_PROFILE", None)
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["AXIOM_DEVICE_KIND"] = "cpu"
    return env


def measure(repo_root: str | Path) -> dict[str, Any]:
    repo_root = Path(repo_root)
    fail_open: dict[str, bool] = {}
    outcomes: dict[str, dict[str, Any]] = {}
    not_probed: list[str] = []
    with tempfile.TemporaryDirectory(prefix="fail-open-audit-") as tmpdir:
        tmp = Path(tmpdir)
        for name, script, args, timeout in PROBES:
            outcome = _run_probe(repo_root, tmp, script, args, timeout)
            outcomes[name] = outcome
            if outcome["exit_code"] is None:
                not_probed.append(name)
                continue
            fail_open[name] = outcome["exit_code"] == 0
    return {
        "fail_open": fail_open,
        "outcomes": outcomes,
        "guards_count": len(PROBES),
        "not_probed": not_probed,
    }


class FailOpenAuditExporter(BaseExporter):
    SPEC = SensorSpec(
        id="S-034",
        facts=("fail_open", "outcomes", "guards_count", "not_probed"),
        schema={
            "fail_open": {"unit": "", "quality": "measured", "level": "diagnostic"},
            "outcomes": {"unit": "", "quality": "measured", "level": "diagnostic"},
            "guards_count": {"unit": "count", "quality": "measured", "level": "diagnostic"},
            "not_probed": {"unit": "", "quality": "measured", "level": "diagnostic"},
        },
        level="diagnostic",
        raw={"note": "пробы во временном каталоге: запуск стражей без входа (K4, ADR-038)"},
        context=("repo",),
        pack="spine",
    )

    def collect(self, subject: dict[str, Any], *, root: Any = None, **_: Any) -> list[Fact]:
        repo_root = Path(root) if root is not None else Path(__file__).resolve().parents[4]
        data = measure(repo_root)
        method = "прогон tools/check_*.py с отсутствующим входом (ADR-038, дельта K4)"
        return [
            self.fact("fail_open", data["fail_open"], subject=subject, method=method),
            self.fact("outcomes", data["outcomes"], subject=subject, method=method),
            self.fact("guards_count", data["guards_count"], subject=subject, method=method),
            self.fact("not_probed", data["not_probed"], subject=subject, method=method),
        ]


EXPORTER = FailOpenAuditExporter()
