#!/usr/bin/env python3
"""ADR-048 — отчёт классификации параметров по группам оптимизатора.

Зачем
-----
ADR-048 вывел embeddings/LM head из Muon в AdamW и потребовал, чтобы группа
каждого листа определялась явным предикатом по именам, а не размерностью
(«любой ndim==2 -> Muon»).  Пункт 4 решения: классификация фиксируется
**отчётом** — «сколько чего» должно быть числом, а не верой.  Этот прибор даёт
такое число для реальной геометрии пресета ``l3-full``.

Как
---
Дерево параметров берётся в абстрактном виде (``jax.eval_shape``): нужны только
имена листьев и их размерности, а не 1 ГБ весов, поэтому отчёт снимается на
любой машине, включая CPU, без аллокации модели.  Источник классификации —
``net.optimizer.classify_leaf`` (единственный предикат: тот же, что читают
``init_state`` и ``make_step``), а не копия списка имён в приборе.

Что пишет
---------
* ``evidence/kda-rewrite/optimizer-param-groups.json`` — машиночитаемый отчёт
  (группа -> листьев, параметров, имена, примеры путей);
* ``evidence/kda-rewrite/optimizer-param-groups.md`` — та же таблица прозой;
* таблицу в stdout.

``--legacy`` снимает отчёт по прежней классификации (все 2-D листья Muon) — та
же пара «до/после» на одной ревизии кода, что даёт флаг раннера.

Использование::

    python3 tools/optimizer_groups.py                 # решение ADR-048
    python3 tools/optimizer_groups.py --legacy        # прежняя классификация
    python3 tools/optimizer_groups.py --config net/config.json --print-only
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
for _path in (str(_REPO_ROOT), str(_REPO_ROOT / "tools")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

#: Префикс отчёта (общая с другими приборами площадка доказательств).
DEFAULT_OUT_DIR = _REPO_ROOT / "evidence" / "kda-rewrite"
DEFAULT_CONFIG = _REPO_ROOT / "net" / "config.json"
JSON_NAME = "optimizer-param-groups.json"
MD_NAME = "optimizer-param-groups.md"


def _preflight_memory() -> None:
    """ADR-041: лимит памяти XLA — ДО импорта jax (маркер ``ensure_mem_fraction()``).

    Прибор ничего не считает на устройстве (только абстрактные формы), но
    импорт jax создаёт клиента и берёт резерв: без лимита он равен дефолтным
    ~75 % устройства, и совмещённый стенд GB10 уходит в global OOM (инцидент
    08.10).  Дисциплина одна для всех JAX-инструментов ``tools/``.
    """
    import jax_preflight

    jax_preflight.ensure_mem_fraction()


def _abstract_params(config_path: Path):
    """Абстрактное дерево параметров пресета — только имена и размерности."""
    import jax
    import jax.numpy as jnp

    from net import model, optimizer
    from net.config import load_config

    cfg = load_config(config_path)
    shapes = jax.eval_shape(
        lambda key: model.init_params(key, cfg),
        jax.ShapeDtypeStruct((2,), jnp.uint32),
    )
    return cfg, shapes, optimizer


def build_artifact(config_path: Path, *, legacy: bool) -> dict:
    """Отчёт + провенанс (конфиг, режим классификации, источник предиката)."""
    cfg, shapes, optimizer = _abstract_params(config_path)
    report = optimizer.classification_report(shapes, legacy_muon_all_2d=legacy)
    report = dict(report)
    report["schema"] = "axiom/optimizer-param-groups/1"
    report["adr"] = "ADR-048"
    report["source"] = "net.optimizer.classify_leaf"
    report["config"] = str(config_path.relative_to(_REPO_ROOT)) if config_path.is_relative_to(_REPO_ROOT) else str(config_path)
    report["config_geometry"] = {
        "vocab_size": cfg.vocab_size,
        "hidden": cfg.hidden,
        "num_layers": cfg.num_layers,
        "num_heads": cfg.num_heads,
        "preset_note": "l3-full (net/config.json)",
    }
    return report


def write_artifact(report: dict, out_dir: Path) -> tuple[Path, Path]:
    import warnings

    from net import optimizer

    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "-legacy" if report["legacy_muon_all_2d"] else ""
    json_path = out_dir / f"{JSON_NAME[:-5]}{suffix}.json"
    md_path = out_dir / f"{MD_NAME[:-3]}{suffix}.md"
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )
    table = optimizer.format_report(report)
    # Провенанс добавляет ``build_artifact``; ``write_artifact`` обязан пережить
    # отчёт без него (например, собранный тестом из голого ``classification_report``).
    geom = report.get("config_geometry") or {}
    config_label = report.get("config", "—")
    geometry = (
        f" (vocab {geom['vocab_size']}, hidden {geom['hidden']}, {geom['num_layers']} слоёв)"
        if geom
        else ""
    )
    header = (
        f"# Классификация параметров по группам оптимизатора (ADR-048)\n\n"
        f"- конфиг: `{config_label}`{geometry}\n"
        f"- режим: `legacy_muon_all_2d={report['legacy_muon_all_2d']}`"
        + (" (прежняя классификация: любой ndim==2 -> Muon)" if report["legacy_muon_all_2d"] else "")
        + "\n"
        f"- источник предиката: `{report.get('source', 'net.optimizer.classify_leaf')}` "
        f"(тот же, что читают `init_state` и `make_step`)\n"
        f"- дерево: абстрактные формы (`jax.eval_shape`), модель не аллоцируется\n\n"
    )
    caveat = (
        "\n\nЧестная граница: отчёт описывает **геометрию дерева и маршрутизацию**,\n"
        "а не измеренное время или память шага.  Время шага и loss-динамика дают\n"
        "прогоны на GB10 (`--ns-steps` / `--legacy-muon-all-2d` у раннера).\n"
    )
    md_path.write_text(header + table + caveat, encoding="utf-8")
    if report["unclassified"]:
        warnings.warn(
            "неклассифицированные 2-D листья: "
            + ", ".join(report["unclassified"])
            + " — шаг упадёт fail-closed (UnclassifiedMatrixError)",
            stacklevel=2,
        )
    return json_path, md_path


def main(argv: list[str] | None = None) -> int:
    _preflight_memory()

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG,
                        help="конфиг модели (по умолчанию net/config.json — пресет l3-full)")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR,
                        help="каталог отчётов (по умолчанию evidence/kda-rewrite)")
    parser.add_argument("--legacy", action="store_true",
                        help="снять отчёт по прежней классификации (все 2-D листья Muon)")
    parser.add_argument("--print-only", action="store_true",
                        help="только напечатать таблицу, ничего не писать на диск")
    args = parser.parse_args(argv)

    from net import optimizer

    report = build_artifact(args.config, legacy=bool(args.legacy))
    print(optimizer.format_report(report))
    if not args.print_only:
        json_path, md_path = write_artifact(report, args.out_dir)
        print(f"\n[optimizer-groups] отчёт: {json_path}")
        print(f"[optimizer-groups] отчёт: {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
