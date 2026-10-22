"""S-010 — доли коммитов по кучкам и коллизии нумерации ADR (дельта C2).

Что меряет: классификацию коммитов окна анализа «две кучки» (25.09–06.10) по
путям изменений на три кучки — **документы** (docs/adr, docs/specs, model/ —
кроме правки ``**Rule**``, evidence/, spine, README), **поведение**
(CONSTRAINTS.yaml, tools/check_*.py, H-слой env/, бюджетные гейты, правка
``**Rule**`` в model/AD-*.md), **работа** (остальное). Смешанный коммит
относится к большинству путей с пометкой ``mixed``. Дополнительно — число
коллизий нумерации ADR и число коммитов-разрешений коллизий.

Запуск::

    python3 -m tools.sensors.process_piles [--since 2026-09-25] [--until 2026-10-07] [--out-dir DIR]
    python3 -m tools.sensors.process_piles --selftest
"""

from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path
from typing import Any, Optional

from ._common import REPO_ROOT, config_subject, emit

DOCS = "docs"
BEHAVIOUR = "behaviour"
WORK = "work"

_BEHAVIOUR_FILES = {
    "CONSTRAINTS.yaml",
    "env/verifier.py",
    "env/corruption.py",
    "env/schemas.py",
    "env/generate.py",
    "env/sft_block.py",
}
_BEHAVIOUR_PREFIXES = ("tools/sensors/", "evidence/budget/")
_COLLISION_RE = re.compile(r"коллиз|collision|нумерац|numbering", re.IGNORECASE)
_ADR_NUM_RE = re.compile(r"^ADR-(\d+)-")


def _is_behaviour(path: str, rule_touched: bool) -> bool:
    if path in _BEHAVIOUR_FILES:
        return True
    if path.startswith(_BEHAVIOUR_PREFIXES):
        return True
    if re.match(r"tools/check_.*\.py$", path) or path == "tools/check_claims.py":
        return True
    if path.startswith("model/AD-") and path.endswith(".md") and rule_touched:
        return True
    return False


def _is_docs(path: str) -> bool:
    if path.startswith(("docs/adr/", "docs/specs/", "evidence/")):
        return True
    if path in ("ARCHITECTURE-SPINE.md", "AGENTS.md", "README.md"):
        return True
    if path.endswith(".md"):
        return True
    if path.startswith("model/"):
        return True
    return False


def _classify_paths(paths: list[str], rule_touched: bool) -> str:
    """Класс коммита по большинству путей."""
    counts = {DOCS: 0, BEHAVIOUR: 0, WORK: 0}
    for path in paths:
        if _is_behaviour(path, rule_touched):
            counts[BEHAVIOUR] += 1
        elif _is_docs(path):
            counts[DOCS] += 1
        else:
            counts[WORK] += 1
    return max(counts, key=lambda k: (counts[k], -list(counts).index(k)))


def _git(repo: Path, args: list[str]) -> str:
    proc = subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, text=True, timeout=120
    )
    return proc.stdout if proc.returncode == 0 else ""


def _parse_log(text: str) -> list[dict[str, Any]]:
    commits: list[dict[str, Any]] = []
    current: Optional[dict[str, Any]] = None
    # Внимание: ``str.splitlines()`` режет по \x1e (record separator), поэтому
    # делим только по \n — иначе маркер начала коммита теряется.
    for line in text.split("\n"):
        if line.startswith("\x1e"):
            if current is not None:
                commits.append(current)
            sha, _, subject = line[1:].partition("\x1f")
            current = {"sha": sha.strip(), "subject": subject.strip(), "files": []}
        elif current is not None and line.strip():
            parts = line.split("\t")
            if len(parts) >= 2:
                current["files"].append(parts[-1])
    if current is not None:
        commits.append(current)
    return commits


def _rule_touched(repo: Path, sha: str) -> bool:
    diff = _git(repo, ["show", "--unified=0", "--format=", sha, "--", "model/AD-*.md"])
    return any("**Rule**" in ln for ln in diff.split("\n") if ln[:1] in "+-")


def measure(
    repo: str | Path = REPO_ROOT,
    *,
    since: str = "2026-09-25",
    until: Optional[str] = None,
    out_dir: Optional[str | Path] = None,
    device: Optional[str] = None,
) -> dict[str, Any]:
    repo = Path(repo)
    args = ["log", "--no-merges", "--name-status", "--format=%x1e%H%x1f%s"]
    args += [f"--since={since}"]
    if until:
        args += [f"--until={until}"]
    raw = _git(repo, args)
    commits = _parse_log(raw)

    counts = {DOCS: 0, BEHAVIOUR: 0, WORK: 0}
    mixed = 0
    collision_fixes = 0
    for commit in commits:
        if _COLLISION_RE.search(commit["subject"]):
            collision_fixes += 1
        paths = commit["files"]
        if not paths:
            counts[WORK] += 1
            continue
        rule_touched = _rule_touched(repo, commit["sha"]) if any(
            p.startswith("model/AD-") for p in paths
        ) else False
        categories = {
            (BEHAVIOUR if _is_behaviour(p, rule_touched) else DOCS if _is_docs(p) else WORK)
            for p in paths
        }
        if len(categories) > 1:
            mixed += 1
        counts[_classify_paths(paths, rule_touched)] += 1

    total = len(commits)
    adr_collisions = _adr_collisions(repo)
    subject = config_subject(device=device)
    method = (
        f"git log --no-merges --name-status --since={since}"
        + (f" --until={until}" if until else "")
        + "; классификатор путей (docs|behaviour|work), смешанные — по большинству с пометкой mixed"
    )
    written: dict[str, Any] = {}
    for key, bucket in (("docs_share", DOCS), ("behaviour_share", BEHAVIOUR), ("work_share", WORK)):
        share = (counts[bucket] / total) if total else None
        written[key] = emit(
            "S-010", key, round(share, 6) if share is not None else None, unit="fraction",
            quality="measured", method=method, subject=subject, out_dir=out_dir,
            status="ok" if share is not None else "unverified",
            note=(
                f"коммитов в окне: {total}; mixed: {mixed}"
                if total else "в окне нет коммитов"
            ),
        )
    written["adr_numbering_collisions"] = emit(
        "S-010", "adr_numbering_collisions", adr_collisions, unit="count",
        quality="measured", method="группировка docs/adr/ADR-<N>-* по номеру", subject=subject,
        out_dir=out_dir,
    )
    written["collision_fix_commits"] = emit(
        "S-010", "collision_fix_commits", collision_fixes, unit="count", quality="measured",
        method=method, subject=subject, out_dir=out_dir,
    )
    return written


def _adr_collisions(repo: Path) -> int:
    adr_dir = repo / "docs" / "adr"
    if not adr_dir.is_dir():
        return 0
    by_num: dict[str, set[str]] = {}
    for path in adr_dir.glob("ADR-*.md"):
        m = _ADR_NUM_RE.match(path.name)
        if m:
            by_num.setdefault(m.group(1), set()).add(path.name)
    return sum(1 for names in by_num.values() if len(names) > 1)


def run_selftest() -> int:
    checks: list[tuple[str, bool]] = [
        ("ADR-файл → docs", _classify_paths(["docs/adr/ADR-001-x.md"], False) == DOCS),
        ("CONSTRAINTS → behaviour", _classify_paths(["CONSTRAINTS.yaml"], False) == BEHAVIOUR),
        ("check_*.py → behaviour", _classify_paths(["tools/check_claims.py"], False) == BEHAVIOUR),
        ("net/ → work", _classify_paths(["net/model.py"], False) == WORK),
        ("model/AD Rule → behaviour", _classify_paths(["model/AD-8-x.md"], True) == BEHAVIOUR),
        ("model/AD без Rule → docs", _classify_paths(["model/AD-8-x.md"], False) == DOCS),
        ("большинство wins", _classify_paths(["net/a.py", "net/b.py", "docs/x.md"], False) == WORK),
    ]
    ok = all(p for _, p in checks)
    for label, passed in checks:
        print(f"[selftest] {'PASS' if passed else 'FAIL'}: {label}")
    print(f"[selftest] {'PASS' if ok else 'FAIL'}: process_piles")
    return 0 if ok else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="S-010: доли коммитов по кучкам (ADR-037)")
    parser.add_argument("--repo", default=str(REPO_ROOT))
    parser.add_argument("--since", default="2026-09-25")
    parser.add_argument("--until", default=None)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--selftest", action="store_true")
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.selftest:
        return run_selftest()
    written = measure(args.repo, since=args.since, until=args.until, out_dir=args.out_dir, device=args.device)
    for name, rec in written.items():
        print(f"S-010 {name} = {rec['value']} [{rec['status']}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
