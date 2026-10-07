"""Теневое сравнение стражей и шаблонов (ADR-039, дельта R2).

Это шаблон ``differential``, применённый к самим стражам: для каждого правила с
``migration: full`` запускаются исходный страж (на том же состоянии) и экземпляр
шаблона; сравниваются классы вердикта. Совпадение (или расхождение) классов
пишется фактом S-038 ``guard_shadow_agreement`` — переключение стража на шаблон
остаётся решением владельца (R3).

Состояния: текущий HEAD плюс исторические refs через ``git worktree`` во
временном каталоге. Временный worktree удаляется после прогона (RUNBOOK §6);
основное дерево не трогается.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Optional

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.properties import Facts, get_property  # noqa: E402
from tools.properties.base import load_properties_registry  # noqa: E402

RULES_FILE = Path("model/rule-properties.yaml")
CONSTRAINTS_FILE = "CONSTRAINTS.yaml"

#: Исторические состояния для теневого прогона (R2): последний зелёный коммит до
#: C-046, коммит с WY/UT (ed1268f), приёмка ворктри t2.
HISTORICAL_REFS: tuple[str, ...] = ("92d2985~1", "ed1268f", "c119bbd")

_VERDICT_RE = re.compile(r'"(verdict_class|verdict|class)"\s*:\s*"([A-Za-z_]+)"')
_CLASS_MAP = {"pass": "pass", "ok": "pass", "fail": "fail", "failed": "fail",
              "unverified": "unverified", "neutral": "unverified", "unknown": "unverified"}


def load_rule_properties(root: str | Path) -> list[dict]:
    from tools.miniyaml import load_file

    path = Path(root) / RULES_FILE
    if not path.is_file():
        return []
    data = load_file(path)
    if isinstance(data, dict):
        data = data.get("rules") or data.get("rule-properties") or []
    return [e for e in data if isinstance(e, dict)] if isinstance(data, list) else []


def _rule_commands(root: Path) -> dict[str, str]:
    text = (root / CONSTRAINTS_FILE).read_text(encoding="utf-8")
    out: dict[str, str] = {}
    current: Optional[str] = None
    for line in text.splitlines():
        m = re.match(r"^\s*-\s*id:\s*(C-\d+)", line)
        if m:
            current = m.group(1)
            continue
        if current:
            mc = re.match(r"^\s+command:\s*'(.+)'\s*$", line)
            if mc:
                out[current] = mc.group(1)
                current = None
    return out


def guard_class(root: Path, command: str) -> tuple[str, str]:
    """Класс вердикта стража: JSON-поле, иначе код возврата. ``(класс, деталь)``."""
    cmd = command
    if cmd.startswith("python3 "):
        cmd = sys.executable + " " + cmd[len("python3 "):]
    try:
        proc = subprocess.run(cmd, shell=True, cwd=str(root), capture_output=True, text=True, timeout=600)
    except subprocess.TimeoutExpired:
        return "unverified", "таймаут стража"
    out = (proc.stdout or "") + "\n" + (proc.stderr or "")
    for key, value in reversed(_VERDICT_RE.findall(out)):
        cls = _CLASS_MAP.get(value.lower())
        if cls:
            return cls, f"json:{key}={value}"
    code = proc.returncode
    if code == 0:
        return "pass", "exit 0"
    if code == 1:
        return "fail", "exit 1"
    return "unverified", f"exit {code}"


def template_class(root: Path, entry: dict) -> tuple[str, str]:
    template = get_property(str(entry.get("property")))
    if template is None:
        return "unverified", f"шаблон {entry.get('property')!r} не найден"
    facts = Facts(root, out_dir=root / "evidence" / "facts", subject_match=[], subject={})
    verdict = template.evaluate(entry.get("params") or {}, facts)
    return verdict.cls, verdict.reason


def compare_state(root: Path, entries: list[dict], commands: dict[str, str]) -> list[dict]:
    rows = []
    for entry in entries:
        rule = str(entry.get("rule"))
        command = str(entry.get("probe") or commands.get(rule, ""))
        if not command:
            guard_cls, guard_detail = "unverified", "команда стража не найдена"
        else:
            guard_cls, guard_detail = guard_class(root, command)
        tpl_cls, tpl_detail = template_class(root, entry)
        rows.append({
            "rule": rule, "state": str(root), "guard": guard_cls, "template": tpl_cls,
            "agreement": guard_cls == tpl_cls,
            "guard_detail": guard_detail, "template_detail": tpl_detail,
        })
    return rows


def _add_worktree(ref: str, dest: Path) -> None:
    subprocess.run(["git", "worktree", "add", "--detach", "--force", str(dest), ref],
                   cwd=str(_REPO_ROOT), check=True, capture_output=True, text=True)


def _remove_worktree(dest: Path) -> None:
    subprocess.run(["git", "worktree", "remove", "--force", str(dest)],
                   cwd=str(_REPO_ROOT), capture_output=True, text=True)
    subprocess.run(["git", "worktree", "prune"], cwd=str(_REPO_ROOT), capture_output=True, text=True)


def shadow(root: str | Path = _REPO_ROOT, refs: Optional[list[str]] = None) -> dict:
    root = Path(root).resolve()
    entries = [e for e in load_rule_properties(root) if e.get("migration") == "full"]
    commands = _rule_commands(root)
    states = [(str(root), root)]
    for ref in (refs if refs is not None else list(HISTORICAL_REFS)):
        tmp = Path(tempfile.mkdtemp(prefix="shadow-wt-"))
        try:
            _add_worktree(ref, tmp)
            states.append((ref, tmp))
        except subprocess.CalledProcessError as exc:
            states.append((ref, None))
            _remove_worktree(tmp)
    rows: list[dict] = []
    for label, path in states:
        if path is None:
            for entry in entries:
                rows.append({"rule": str(entry.get("rule")), "state": label, "guard": "unverified",
                             "template": "unverified", "agreement": True,
                             "guard_detail": "состояние недоступно", "template_detail": ""})
            continue
        for row in compare_state(path, entries, commands):
            row["state"] = label
            rows.append(row)
    for label, path in states:
        if path is not None and str(path) != str(root):
            _remove_worktree(path)
    agreement = all(r["agreement"] for r in rows) if rows else True
    return {"full_rules": [str(e.get("rule")) for e in entries], "rows": rows, "agreement": agreement}


def write_fact(root: Path, payload: dict) -> None:
    from tools.sensors.fact import write_fact
    from tools.sensors.subject import build_subject

    subject = build_subject(repo_root=root, config_path=root / "net" / "config.json", device="cpu")
    write_fact("S-038", "guard_shadow_agreement", payload, unit="", quality="measured",
               method="tools.properties.shadow (ADR-039, дельта R2)", subject=subject)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Теневое сравнение стражей и шаблонов (R2)")
    parser.add_argument("--root", default=".", help="корень кейса")
    parser.add_argument("--ref", action="append", default=None, help="исторический ref (можно несколько)")
    parser.add_argument("--no-fact", action="store_true", help="не писать факт S-038")
    parser.add_argument("--json", dest="json_path", default=None)
    args = parser.parse_args(argv)
    root = Path(args.root).resolve()
    payload = shadow(root, args.ref)
    if not args.no_fact:
        write_fact(root, payload)
    if args.json_path:
        Path(args.json_path).write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"full-стражи: {payload['full_rules']}; согласие: {payload['agreement']}")
    for row in payload["rows"]:
        mark = "OK" if row["agreement"] else "DIVERGE"
        print(f"  [{mark}] {row['rule']} @ {row['state']}: страж={row['guard']}, шаблон={row['template']}"
              f" ({row['guard_detail']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
