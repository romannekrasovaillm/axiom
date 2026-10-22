"""S-011 — объём временных каталогов pytest/воркспейсов (дельта C2).

Что меряет: размер pytest tmp, временных воркспейсов ``env/`` (прогоны
``pytest-of-<user>/pytest-*``) и staging-каталогов по RUNBOOK §6 — основание для
O-5 (73 ГБ: ретенция pytest держит 3 прогона по ~24 ГБ). Измерение — ``du -sb``
с таймаутом; отсутствующий каталог — ноль с пометкой, а не ошибка.

Запуск::

    python3 -m tools.sensors.disk_usage [--out-dir DIR]
    python3 -m tools.sensors.disk_usage --selftest
"""

from __future__ import annotations

import argparse
import getpass
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Optional

from ._common import config_subject, emit

METHOD = "du -sb по путям RUNBOOK §6 (pytest tmp, воркспейсы env/, staging)"


def du_bytes(path: str | Path, timeout: int = 60) -> tuple[int, bool]:
    """Размер каталога в байтах и признак «путь существовал». ``du -sb``."""
    p = Path(path)
    try:
        proc = subprocess.run(
            ["du", "-sb", str(p)], capture_output=True, text=True, timeout=timeout
        )
    except (OSError, subprocess.SubprocessError):
        return 0, False
    if proc.returncode != 0:
        return 0, False
    line = proc.stdout.strip().split("\t", 1)[0]
    try:
        return int(line), True
    except ValueError:
        return 0, False


def _pytest_root() -> Path:
    return Path(tempfile.gettempdir()) / f"pytest-of-{getpass.getuser()}"


def _staging_dirs() -> list[Path]:
    tmp = Path(tempfile.gettempdir())
    out: list[Path] = []
    for pattern in ("ax-base", "ax-*", "axiom-run", "axiom-*"):
        out.extend(sorted(p for p in tmp.glob(pattern) if p.is_dir()))
    # уникализировать
    seen: set[str] = set()
    unique: list[Path] = []
    for p in out:
        if str(p) not in seen:
            seen.add(str(p))
            unique.append(p)
    return unique


def measure(*, out_dir: Optional[str | Path] = None, device: Optional[str] = None) -> dict[str, Any]:
    subject = config_subject(device=device)
    pytest_root = _pytest_root()
    pytest_bytes, pytest_exist = du_bytes(pytest_root)

    env_bytes = 0
    env_count = 0
    if pytest_exist:
        for run_dir in sorted(pytest_root.glob("pytest-*")):
            if run_dir.is_dir():
                size, _ = du_bytes(run_dir)
                env_bytes += size
                env_count += 1

    staging_bytes = 0
    staging_list: list[str] = []
    for d in _staging_dirs():
        size, _ = du_bytes(d)
        staging_bytes += size
        staging_list.append(d.name)

    written: dict[str, Any] = {}
    written["pytest_tmp_bytes"] = emit(
        "S-011", "pytest_tmp_bytes", pytest_bytes, unit="bytes", quality="measured",
        method=METHOD, subject=subject, out_dir=out_dir,
        note="" if pytest_exist else f"каталог {pytest_root} отсутствует",
    )
    written["env_workspace_bytes"] = emit(
        "S-011", "env_workspace_bytes", env_bytes, unit="bytes", quality="measured",
        method=METHOD, subject=subject, out_dir=out_dir,
        note=f"прогонов pytest-*: {env_count}",
    )
    written["staging_bytes"] = emit(
        "S-011", "staging_bytes", staging_bytes, unit="bytes", quality="measured",
        method=METHOD, subject=subject, out_dir=out_dir,
        note=f"каталоги: {', '.join(staging_list) if staging_list else 'нет'}",
    )
    return written


def run_selftest() -> int:
    import tempfile as _tf

    checks: list[tuple[str, bool]] = []
    with _tf.TemporaryDirectory(prefix="s011-selftest-") as tmp:
        d = Path(tmp) / "nested"
        d.mkdir()
        (d / "f.bin").write_bytes(b"x" * 4096)
        size, exists = du_bytes(Path(tmp))
        checks.append(("du видит каталог", exists))
        checks.append(("du считает байты", size >= 4096))
        missing, m_exists = du_bytes(Path(tmp) / "nope")
        checks.append(("нет каталога → 0/False", missing == 0 and not m_exists))
    written = measure(out_dir=tmp)
    checks.append(("pytest_tmp_bytes ≥ 0", written["pytest_tmp_bytes"]["value"] >= 0))
    checks.append(("staging_bytes ≥ 0", written["staging_bytes"]["value"] >= 0))
    ok = all(p for _, p in checks)
    for label, passed in checks:
        print(f"[selftest] {'PASS' if passed else 'FAIL'}: {label}")
    print(f"[selftest] {'PASS' if ok else 'FAIL'}: disk_usage")
    return 0 if ok else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="S-011: объём временных каталогов (ADR-037)")
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
        print(f"S-011 {name} = {rec['value']} bytes [{rec['status']}] note={rec['note']!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
