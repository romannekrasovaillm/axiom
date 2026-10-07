"""S-014 — обёртка бенчмарков block-merge и KDA WY/UT (дельта C3).

Читает ``evidence/block-merge-bench-*.json`` (block-merge: отношение стоимости и
recall) и ``evidence/kda-wyut/*`` (ускорение/память компонента KDA). Инструменты
не меняются — только их выход. Отсутствующий источник → ``unverified``.

Запуск::

    python3 -m tools.sensors.wrap_bench [--out-dir DIR]
    python3 -m tools.sensors.wrap_bench --selftest
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Optional

from ._common import REPO_ROOT, config_subject, emit
from ._wrap import latest_glob, read_json

METHOD = "обёртка: evidence/block-merge-bench-*.json и evidence/kda-wyut/* (bench_block_merge, bench_kda_wyut)"


def _find(obj: Any, keys: tuple[str, ...]) -> Any:
    """Первое значение по любому из ключей (рекурсивно, в порядке дерева)."""
    if isinstance(obj, dict):
        for key in keys:
            if key in obj:
                return obj[key]
        for value in obj.values():
            found = _find(value, keys)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for item in obj:
            found = _find(item, keys)
            if found is not None:
                return found
    return None


def measure(*, out_dir: Optional[str | Path] = None, device: Optional[str] = None) -> dict[str, Any]:
    subject = config_subject(device=device)
    written: dict[str, Any] = {}

    bench = latest_glob("evidence/block-merge-bench-*.json")
    bench_data = read_json(bench) if bench else None

    def _dig(obj: Any, *path: str) -> Any:
        cur = obj
        for key in path:
            if not isinstance(cur, dict) or key not in cur:
                return None
            cur = cur[key]
        return cur

    cost = _dig(bench_data, "cost", "ratio_of_medians")
    if cost is None:
        cost = _find(bench_data, ("ratio_of_medians",))
    recall = _dig(bench_data, "recall", "overall", "recall_on")
    if recall is None:
        recall = _dig(bench_data, "recall", "overall", "delta")
    written["block_merge_cost_ratio"] = emit(
        "S-014", "block_merge_cost_ratio", cost, unit="ratio", quality="wrapped",
        method=METHOD, subject=subject, out_dir=out_dir,
        status="ok" if cost is not None else "unverified",
        note="" if cost is not None else "evidence/block-merge-bench-*.json без cost-поля",
    )
    written["block_merge_recall"] = emit(
        "S-014", "block_merge_recall", recall, unit="recall", quality="wrapped",
        method=METHOD, subject=subject, out_dir=out_dir,
        status="ok" if recall is not None else "unverified",
        note="" if recall is not None else "evidence/block-merge-bench-*.json без recall-поля",
    )

    kda_dir = REPO_ROOT / "evidence" / "kda-wyut"
    speedup = mem_ratio = None
    kda_manifest = None
    if kda_dir.is_dir():
        files = sorted(kda_dir.glob("*.json"))
        if files:
            kda_manifest = files[-1]
            data = read_json(kda_manifest)
            speedup = _find(data, ("speedup", "component_speedup"))
            mem_ratio = _find(data, ("mem_ratio", "memory_ratio"))
    note = "" if kda_manifest is not None else "evidence/kda-wyut/* отсутствует (прогон WY/UT-дельты не снят)"
    written["kda_component_speedup"] = emit(
        "S-014", "kda_component_speedup", speedup, unit="speedup", quality="wrapped",
        method=METHOD, subject=subject, out_dir=out_dir,
        status="ok" if speedup is not None else "unverified", note=note or "нет поля speedup",
    )
    written["kda_component_mem_ratio"] = emit(
        "S-014", "kda_component_mem_ratio", mem_ratio, unit="ratio", quality="wrapped",
        method=METHOD, subject=subject, out_dir=out_dir,
        status="ok" if mem_ratio is not None else "unverified", note=note or "нет поля mem_ratio",
    )
    return written


def run_selftest() -> int:
    import json
    import tempfile

    checks: list[tuple[str, bool]] = []
    with tempfile.TemporaryDirectory(prefix="s014-selftest-") as tmp:
        root = Path(tmp)
        (root / "evidence").mkdir()
        (root / "evidence" / "block-merge-bench-x.json").write_text(
            json.dumps({"metrics": {"ratio_of_medians": 2.49, "recall_on": 0.97, "recall_off": 0.96}}),
            encoding="utf-8",
        )
        bench = latest_glob("evidence/block-merge-bench-*.json", root=root)
        data = read_json(bench)
        checks.append(("cost найден", _find(data, ("ratio_of_medians",)) == 2.49))
        checks.append(("recall найден", _find(data, ("recall_on",)) == 0.97))
        written = measure(out_dir=tmp)
        checks.append(("кда-источник отсутствует → unverified",
                       written["kda_component_speedup"]["status"] == "unverified"))
    ok = all(p for _, p in checks)
    for label, passed in checks:
        print(f"[selftest] {'PASS' if passed else 'FAIL'}: {label}")
    print(f"[selftest] {'PASS' if ok else 'FAIL'}: wrap_bench")
    return 0 if ok else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="S-014: обёртка бенчмарков (ADR-037)")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--selftest", action="store_true")
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.selftest:
        return run_selftest()
    written = measure(out_dir=args.out_dir, device=args.device)
    for name, rec in written.items():
        print(f"S-014 {name} = {rec['value']} [{rec['status']}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
