"""S-006 — объём снапшота чистого кейса (дельта C2).

Что меряет: реальный размер снапшота, который строит среда
(``env.util.copy_case_snapshot`` + ``dir_total_bytes``) — основание для
утверждения «кап снапшота 64 МБ» (E-1.5) и инцидента «35 ГБ воркспейсов вместо
снапшотов». Снапшот собирается в временный каталог и удаляется.

Запуск::

    python3 -m tools.sensors.snapshot_size [--case DIR] [--out-dir DIR]
    python3 -m tools.sensors.snapshot_size --selftest
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path
from typing import Any, Optional

from ._common import REPO_ROOT, config_subject, emit

METHOD = "env.util.copy_case_snapshot(чистый кейс) + dir_total_bytes + tree_sha256"


def measure(
    case_dir: str | Path = REPO_ROOT,
    *,
    out_dir: Optional[str | Path] = None,
    device: Optional[str] = None,
) -> dict[str, Any]:
    from env.util import WORKSPACE_CAP_BYTES, copy_case_snapshot, dir_total_bytes, tree_sha256

    case = Path(case_dir)
    subject = config_subject(device=device)
    with tempfile.TemporaryDirectory(prefix="s006-snapshot-") as tmp:
        dst = Path(tmp) / "snapshot"
        copy_case_snapshot(case, dst)
        total = dir_total_bytes(dst)
        tree = tree_sha256(dst)

    within = total <= WORKSPACE_CAP_BYTES
    note = "" if within else (
        f"объём {total} Б превышает кап {WORKSPACE_CAP_BYTES} Б (E-1.5)"
    )
    rec = emit(
        "S-006", "snapshot_bytes", total, unit="bytes", quality="measured",
        method=f"{METHOD}; tree_sha256={tree[:16]}…", subject=subject,
        out_dir=out_dir, note=note,
    )
    rec["_tree_sha256"] = tree
    rec["_within_cap"] = within
    return {"snapshot_bytes": rec}


def run_selftest() -> int:
    import tempfile

    from env.util import WORKSPACE_CAP_BYTES

    with tempfile.TemporaryDirectory(prefix="s006-selftest-") as tmp:
        first = measure(REPO_ROOT, out_dir=tmp)
        second = measure(REPO_ROOT, out_dir=tmp)
    a = first["snapshot_bytes"]
    b = second["snapshot_bytes"]
    checks: list[tuple[str, bool]] = [
        ("snapshot_bytes > 0", a["value"] > 0),
        ("снапшот в пределах капа 64 МБ", a["_within_cap"] is True),
        ("объём воспроизводим", a["value"] == b["value"]),
        ("tree_sha256 воспроизводим", a["_tree_sha256"] == b["_tree_sha256"]),
        ("кап равен 64 МиБ", WORKSPACE_CAP_BYTES == 64 * 1024 * 1024),
    ]
    ok = all(p for _, p in checks)
    for label, passed in checks:
        print(f"[selftest] {'PASS' if passed else 'FAIL'}: {label}")
    print(f"[selftest] {'PASS' if ok else 'FAIL'}: snapshot_size")
    return 0 if ok else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="S-006: объём снапшота кейса (ADR-036)")
    parser.add_argument("--case", default=str(REPO_ROOT))
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--selftest", action="store_true")
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.selftest:
        return run_selftest()
    written = measure(args.case, out_dir=args.out_dir, device=args.device)
    rec = written["snapshot_bytes"]
    print(f"S-006 snapshot_bytes = {rec['value']} bytes [{rec['status']}] note={rec['note']!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
