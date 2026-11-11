#!/usr/bin/env python3
"""C-055 — страж Run Manifest v2: заглушка и чужая ось не доказательство (ADR-058).

Проверяет контракт ``env.manifest`` поведением, а не текстом:

* валидный v2, собранный :func:`env.manifest.build_manifest`, проходит
  ``full_validate`` и сверку ``victory_axis`` с пакетом решения;
* каждый мутант краснеет: нет ``victory_axis``; нет
  ``baseline.snapshot_sha256``; ``model.is_stub = true``; ось не совпала с
  пакетом;
* исторический манифест v1 без ``allow_legacy`` — красный, с
  ``allow_legacy=True`` — зелёный по legacy-набору (доказательством не
  является, но аудит не блокирует).

``--check PATH`` валидирует готовый манифест против пакета решения кейса:
код 0 — PASS, 1 — FAIL.  ``--selftest`` — фикстуры во временном каталоге
(сеть и jax не нужны).

Запуск (из каталога кейса)::

    python3 tools/check_manifest_v2.py --selftest
    python3 tools/check_manifest_v2.py --check evidence/run-manifest.json
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import tempfile
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from env import manifest as manifest_mod  # noqa: E402

DEFAULT_PACKAGE = _REPO_ROOT / "docs" / "DECISION-PACKAGE.json"
SELFTEST_AXIS = "selftest-ось: победа по фитнес-гейту, не по сырому бенчмарку"

EXIT_PASS = 0
EXIT_FAIL = 1


def _fixture_manifest(axis_value: str) -> dict:
    """Валидный v2-манифест фикстуры (все компоненты — константы теста)."""
    sha = "a" * 64
    return manifest_mod.build_manifest(
        run_id="selftest-run",
        task_spec={"id": "corruption-l0-00"},
        arch_ml_build="selftest-build",
        constraints_sha256=sha,
        hidden_constraints_sha256=sha,
        workspace_sha256=sha,
        model_base="net-selftest",
        model_snapshot_sha256=sha,
        baseline_model="kimi-k3",
        baseline_snapshot_sha256="b" * 64,
        victory_axis=axis_value,
        decoding={"temperature": 0.0, "top_p": 0.95, "seed": 42},
        attempts_used=1,
        usage={"tokens_in": 1, "tokens_out": 1, "cost_usd": 0.0, "host": "selftest"},
        timing={
            "started": "2026-11-11T00:00:00+00:00",
            "finished": "2026-11-11T00:00:01+00:00",
            "resume_count": 0,
        },
        verdict={
            "pass": False,
            "reward": {
                "pass_component": 0.0, "soft": 0.0,
                "new_violations": 0.0, "effort_penalty": 0.0,
            },
            "issues_warn": [],
        },
    )


def _legacy_v1(manifest: dict) -> dict:
    """Тот же манифест в исторической форме v1 (ADR-058)."""
    legacy = copy.deepcopy(manifest)
    for key in ("schema_version", "victory_axis", "baseline"):
        legacy.pop(key, None)
    legacy["model"].pop("is_stub", None)
    return legacy


def run_selftest() -> int:
    checks: list[tuple[str, bool]] = []
    with tempfile.TemporaryDirectory(prefix="manifest-v2-selftest-") as tmp:
        package = Path(tmp) / "DECISION-PACKAGE.json"
        package.write_text(
            json.dumps({"choice": {"victory_axis": SELFTEST_AXIS}}, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        good = _fixture_manifest(SELFTEST_AXIS)

        checks.append((
            "валидный v2 — full_validate пусто",
            manifest_mod.full_validate(good, package_path=package) == [],
        ))

        no_axis = copy.deepcopy(good)
        del no_axis["victory_axis"]
        checks.append((
            "нет victory_axis — FAIL",
            manifest_mod.full_validate(no_axis, package_path=package) != [],
        ))

        no_baseline = copy.deepcopy(good)
        del no_baseline["baseline"]["snapshot_sha256"]
        checks.append((
            "нет baseline.snapshot_sha256 — FAIL",
            manifest_mod.full_validate(no_baseline, package_path=package) != [],
        ))

        stub = copy.deepcopy(good)
        stub["model"]["is_stub"] = True
        checks.append((
            "model.is_stub=true — FAIL",
            manifest_mod.full_validate(stub, package_path=package) != [],
        ))

        wrong_axis = copy.deepcopy(good)
        wrong_axis["victory_axis"] = "другая ось: обогнать K3 везде"
        checks.append((
            "ось не совпала с пакетом — FAIL",
            manifest_mod.full_validate(wrong_axis, package_path=package) != [],
        ))

        legacy = _legacy_v1(good)
        checks.append((
            "v1 без allow_legacy — FAIL",
            manifest_mod.full_validate(legacy, package_path=package) != [],
        ))
        checks.append((
            "v1 с allow_legacy — PASS по legacy-набору",
            manifest_mod.full_validate(legacy, package_path=package, allow_legacy=True) == [],
        ))

    failed = [name for name, ok in checks if not ok]
    for name, ok in checks:
        print(f"[manifest-v2:selftest] {'OK' if ok else 'FAIL'}: {name}")
    if failed:
        print(f"manifest-v2:selftest: {len(failed)} из {len(checks)} сценариев не совпали")
        return EXIT_FAIL
    print(f"manifest-v2:selftest: OK ({len(checks)} сценариев)")
    return EXIT_PASS


def check_file(path: Path, package: Path) -> list[str]:
    """Ошибки готового манифеста (пусто = PASS)."""
    if not path.is_file():
        return [f"манифест не найден: {path}"]
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [f"манифест не читается как JSON: {path}: {exc}"]
    return manifest_mod.full_validate(manifest, package_path=package)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="C-055: Run Manifest v2 — ось, baseline, запрет заглушки (ADR-058)",
    )
    parser.add_argument(
        "--check", type=Path, default=None,
        help="путь к манифесту для валидации (v2, без legacy-послаблений)",
    )
    parser.add_argument(
        "--package", type=Path, default=DEFAULT_PACKAGE,
        help="пакет решения для сверки victory_axis (по умолчанию docs/DECISION-PACKAGE.json)",
    )
    parser.add_argument(
        "--selftest", action="store_true",
        help="прогнать фикстуры-мутанты во временном каталоге",
    )
    args = parser.parse_args(argv)

    if args.selftest:
        return run_selftest()
    if args.check is None:
        parser.error("нужен --check PATH или --selftest")

    errors = check_file(args.check, args.package)
    for err in errors:
        print(f"[manifest-v2] FAIL: {err}")
    print(f"manifest-v2: {'PASS' if not errors else f'{len(errors)} нарушений'} ({args.check})")
    return EXIT_PASS if not errors else EXIT_FAIL


if __name__ == "__main__":
    raise SystemExit(main())
