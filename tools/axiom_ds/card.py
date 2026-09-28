"""Карточка датасета ``axiom-domain-ds-v1`` (ADR-020, дельта-3; ADR-004).

Карточка собирается **из артефактов прогонов**, а не из памяти о них:

* **E** — отчёт сборки эпизодов (:mod:`axiom_ds.build`): записи, классы
  верификации, объём, sha256 выхода;
* **K/D/S** — манифест CPT-корпуса (:mod:`axiom_ds.cpt_serialize`): записи и
  объём по компонентам и плагинам, sha256 шардов, фильтры источников, счётчики
  скраба и дедупа.

Доли считаются по ``approx_tokens`` (``chars // 4``, ADR-021) — той же мерой,
что CPT-корпус, иначе доли компонент несравнимы. Хеши шардов карточка
**пересчитывает с диска** и сверяет с манифестом: карточка без сверки — это
пересказ манифеста, а не свидетельство.

Приватность (AD-6, C-032/C-033): ни содержимого, ни абсолютных приватных путей —
источники записываются в форме ``<HOME>/…``, в карточку попадают только числа,
хеши, пути-от-дома и решения. Карточка живёт в репозитории, корпуса — на
``~/gb10-shared``.

Статус карточки — механический: ``v1`` только если сошлись все проверки
(артефакты на месте, хеши сверены, allowlist корпуса совпадает с константой,
объёмы измерены); иначе ``v1-draft`` со списком причин.

Примеры::

    python -m axiom_ds.card build-card
    python -m axiom_ds.card build-card --episodes /tmp/axiom-ds-full-report.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Sequence

if __package__ in (None, ""):  # запуск файлом: python tools/axiom_ds/card.py
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from axiom_ds import cpt_serialize as cpt_mod
    from axiom_ds import verify as verify_mod
    from prep_pretrain import common as pp_common
else:
    from . import cpt_serialize as cpt_mod
    from . import verify as verify_mod
    from prep_pretrain import common as pp_common

CARD_NAME = "axiom-domain-ds-v1"
CARD_VERSION = "v1"
STATUS_DRAFT = "v1-draft"
STATUS_FINAL = "v1"

#: Канонический корень датасета (C-032/C-033) — тот же, что у CPT-сериализатора.
DEFAULT_ROOT = cpt_mod.DATASET_ROOT

#: Артефакты компоненты E: манифест (если появится) либо отчёт сборки.
EPISODES_MANIFEST_NAME = "episodes-v1-manifest.json"
EPISODES_REPORT_NAME = "episodes-v1-report.json"

#: Куда ложится карточка: в репозиторий (docs/datasets), а не на gb10-shared.
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CARD_MD = REPO_ROOT / "docs" / "datasets" / f"{CARD_NAME}-card.md"
DEFAULT_CARD_JSON = REPO_ROOT / "docs" / "datasets" / f"{CARD_NAME}-card.json"

#: Микс-декларация: гипотеза долей при упаковке — отдельный документ, не карточка.
MIX_DECLARATION = f"docs/datasets/{CARD_NAME}-mix.md"

#: Legacy-источник CPT (txt-склейка прошлых корпусов): доля в v1 — 0.
LEGACY_CORPUS = "~/gb10-shared/datasets/cpt_corpus_full.txt"

#: Замер пересечения D↔K на реальных парах «одна статья» (диагностика 28.09.2026,
#: зафиксирована в ADR-020): ступень контейнмента на боевом корпусе не срабатывает —
#: ни одна пара не достигает порога окна. Числа — измерение, а не оценка; карточка
#: несёт их рядом со счётчиками прогона, потому что сам прогон даёт только нули.
CONTAINMENT_MEASURED = {
    "pairs_measured": 294,
    "max_window_jaccard": 0.3524,
    "threshold": 0.55,
    "pairs_at_or_above_threshold": 0,
    "source": (
        "ADR-020 (дельта-2/3): диагностика контейнмента D↔K 28.09.2026 — 294 реальные "
        "пары «одна статья»; боевой прогон дельты-2 (96 217 K × 5 658 D) пар ≥ порога не дал"
    ),
}

COMPONENT_ORDER = ("E", "K", "D", "S")
COMPONENT_TITLES = {
    "E": "агентные эпизоды сессий (класс исхода — механический)",
    "K": "концепты из статей (карточки библиотеки)",
    "D": "дистилляты статей",
    "S": "процедурные скиллы (SKILL.md плагинов домена)",
}

#: Имя шард-файла с номером шарда: ``<prefix>-00000.jsonl[.zst|.gz]``.
SHARD_INDEX_RE = re.compile(r"-\d+\.jsonl(?:\.(?:zst|gz))?$")

#: Что за единица в столбце «файлов» у компоненты.
COMPONENT_FILE_UNITS = {
    "E": "файлов сессий",
    "K": "файлов карточек",
    "D": "файлов дистиллятов",
    "S": "файлов SKILL.md",
}


# --------------------------------------------------------------------------- #
# Мелкие помощники
# --------------------------------------------------------------------------- #


def home_rel(path: str | os.PathLike[str]) -> str:
    """Путь в форме ``<HOME>/…`` — абсолютные приватные пути в карточку не идут."""
    return cpt_mod.replace_home(str(path))[0]


def load_json(path: str | os.PathLike[str] | None) -> dict | None:
    """Прочитать json-артефакт; отсутствие/битый файл — None (не исключение)."""
    if path is None:
        return None
    try:
        with open(os.path.expanduser(os.fspath(path)), "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def file_digest(path: str | os.PathLike[str]) -> dict | None:
    """sha256 и размер файла потоково (содержимое не читается целиком)."""
    target = Path(os.path.expanduser(os.fspath(path)))
    if not target.is_file():
        return None
    digest = hashlib.sha256()
    size = 0
    try:
        with target.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
                size += len(chunk)
    except OSError:
        return None
    return {"sha256": digest.hexdigest(), "bytes": size}


def _num(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:,.4f}".rstrip("0").rstrip(".")
    if isinstance(value, int):
        return f"{value:,}".replace(",", " ")
    return str(value)


def _pct(value: float | None) -> str:
    return "—" if value is None else f"{value * 100:.1f} %"


def _counts(mapping: dict) -> str:
    """Числовой состав словаря одной строкой: ``ключ N``, по убыванию числа."""
    items = sorted(mapping.items(), key=lambda item: (-int(item[1]), str(item[0])))
    return ", ".join(f"`{key}` {_num(value)}" for key, value in items) or "—"


def _table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    width = len(headers)
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in range(width)) + "|",
    ]
    for row in rows:
        cells = [str(cell) for cell in row]
        if len(cells) < width:
            cells.extend(["—"] * (width - len(cells)))
        lines.append("| " + " | ".join(cells[:width]) + " |")
    return "\n".join(lines)


def _first_existing(candidates: Sequence[Any]) -> Path | None:
    for candidate in candidates:
        if candidate is None:
            continue
        path = Path(os.path.expanduser(os.fspath(candidate)))
        if path.is_file():
            return path
    return None


def _report_for(manifest: dict | None, manifest_path: Path | None) -> Path | None:
    """Отчёт CPT-прогона рядом с манифестом: ``<prefix>-report.json``.

    Префикс берётся из имени шард-файла **с отсечением номера шарда**
    (``cpt-kds-v0.1-00000.jsonl.zst`` → ``cpt-kds-v0.1``): отчёт именуется по
    префиксу корпуса, а не по шарду.
    """
    if manifest is None or manifest_path is None:
        return None
    shards = manifest.get("shards") or []
    if not shards:
        return None
    name = str(shards[0].get("file", ""))
    prefix = SHARD_INDEX_RE.sub("", name) or cpt_mod.DATASET_NAME
    return _first_existing([manifest_path.parent / f"{prefix}-report.json"])


# --------------------------------------------------------------------------- #
# Компонента E
# --------------------------------------------------------------------------- #


def component_e(artifact: Path | None, data: dict | None) -> dict:
    """Компонента E: отчёт сборки эпизодов + сверка sha256 выхода с диском."""
    entry: dict[str, Any] = {
        "component": "E",
        "title": COMPONENT_TITLES["E"],
        "present": data is not None,
        "artifact": home_rel(artifact) if artifact else None,
        "generated_at": None,
        "records": None,
        "files": None,
        "file_unit": COMPONENT_FILE_UNITS["E"],
        "chars": None,
        "approx_tokens": None,
        "approx_tokens_basis": None,
        "by_class": {},
        "sft_ready": None,
        "sft_partial_ready": None,
        "negative_ready": None,
        "sources": [],
        "rules": {
            "classes": list(verify_mod.CLASSES),
            "sft_classes": list(verify_mod.SFT_CLASSES),
            "evidence": [
                verify_mod.EVIDENCE_CONTRACT,
                verify_mod.EVIDENCE_HARNESS,
                verify_mod.EVIDENCE_PARTIAL_GREEN,
                verify_mod.EVIDENCE_NONE,
            ],
            "green_suite_window": verify_mod.GREEN_SUITE_WINDOW,
            "green_suite_pattern": verify_mod.SUITE_GREEN_RE.pattern,
            "red_suite_pattern": verify_mod.SUITE_RED_RE.pattern,
        },
        "evidence": {},
        "notes": [],
    }
    if data is None:
        entry["notes"].append(
            "артефакта сборки эпизодов нет: компонента E не измерена "
            "(пересборка не завершена)"
        )
        return entry

    entry["generated_at"] = data.get("generated_at")
    entry["records"] = data.get("episodes_written")
    entry["files"] = data.get("sessions_processed")
    entry["sources"] = [home_rel(item) for item in data.get("sources", [])]
    entry["by_class"] = data.get("by_class_written") or data.get("by_class") or {}
    entry["sft_ready"] = data.get("sft_ready")
    entry["sft_partial_ready"] = data.get("sft_partial_ready")
    entry["negative_ready"] = data.get("negative_ready")
    entry["evidence"] = data.get("by_evidence") or {}
    # Доля verified-partial (флаг ``partial-green``) — в SFT-компоненте: это
    # допуск по снисхождению, и в карточке он виден числом, а не только флагом.
    complete = entry["sft_ready"]
    partial = entry["sft_partial_ready"]
    sft_total = (complete + partial) if isinstance(complete, int) and isinstance(partial, int) else 0
    entry["sft_partial_share"] = round(partial / sft_total, 4) if sft_total else None
    entry["sft_partial_share_of_records"] = (
        round(partial / entry["records"], 4)
        if isinstance(partial, int) and isinstance(entry["records"], int) and entry["records"]
        else None
    )

    chars = data.get("chars")
    tokens = data.get("approx_tokens")
    if isinstance(tokens, int):
        entry["approx_tokens"] = tokens
        entry["approx_tokens_basis"] = "manifest"
    elif isinstance(chars, int):
        entry["approx_tokens"] = chars // 4
        entry["approx_tokens_basis"] = "chars//4 (отчёт без пофайловых токенов)"
    if isinstance(chars, int):
        entry["chars"] = chars

    out = data.get("out")
    if out:
        digest = file_digest(out)
        entry["out"] = home_rel(out)
        entry["sha256_declared"] = data.get("out_sha256")
        entry["bytes_declared"] = data.get("out_bytes")
        if digest is None:
            entry["notes"].append("файл эпизодов по пути отчёта не найден: хеш не сверен")
        else:
            entry["sha256"] = digest["sha256"]
            entry["bytes"] = digest["bytes"]
    if entry["approx_tokens"] is None:
        entry["notes"].append(
            "отчёт без объёма (chars/approx_tokens): доля компоненты E не измерена"
        )
    return entry


# --------------------------------------------------------------------------- #
# Компоненты K/D/S
# --------------------------------------------------------------------------- #


def components_cpt(manifest: dict | None, report: dict | None) -> dict[str, dict]:
    """Компоненты K/D/S: записи, объём и счётчики из манифеста CPT-корпуса."""
    by_component = (manifest or {}).get("by_component") or {}
    report_components = (report or {}).get("components") or {}
    filters = (manifest or {}).get("filters") or {}
    out: dict[str, dict] = {}

    for name in ("K", "D", "S"):
        row = by_component.get(name) or {}
        extra = report_components.get(name) or {}
        entry: dict[str, Any] = {
            "component": name,
            "title": COMPONENT_TITLES[name],
            "present": bool(row),
            "artifact": None,
            "generated_at": (report or {}).get("generated_at"),
            "records": row.get("records"),
            "files": row.get("files_total"),
            "file_unit": COMPONENT_FILE_UNITS[name],
            "chars": row.get("chars"),
            "approx_tokens": None,
            "approx_tokens_basis": None,
            "dropped_dedup": row.get("dropped_dedup"),
            "dropped_containment": row.get("dropped_containment"),
            "sources": [],
            "filters": {
                "k_types": filters.get("k_types") if name == "K" else None,
                "k_levels": filters.get("k_levels") if name == "K" else None,
                "k_title_prefix": filters.get("k_title_prefix") if name == "K" else None,
                "d_subdirs": filters.get("d_subdirs") if name == "D" else None,
                "s_plugins": filters.get("s_plugins") if name == "S" else None,
                "skill_filename": filters.get("skill_filename") if name == "S" else None,
                "min_text_chars": filters.get("min_text_chars"),
                "containment": filters.get("containment"),
                "containment_threshold": filters.get("containment_threshold"),
            },
            "plugins": extra.get("plugins") or {},
            "plugin_tokens": extra.get("plugin_tokens") or {},
            "by_type": extra.get("by_type") or {},
            "by_level": extra.get("by_level") or {},
            "notes": [],
        }
        entry["sources"] = _component_roots(manifest, name)

        tokens = row.get("approx_tokens")
        if isinstance(tokens, int):
            entry["approx_tokens"] = tokens
            entry["approx_tokens_basis"] = "manifest"
        elif isinstance(row.get("chars"), int):
            entry["approx_tokens"] = row["chars"] // 4
            entry["approx_tokens_basis"] = "chars//4 (манифест без пофайловых токенов)"
        if not row:
            entry["notes"].append("компоненты нет в манифесте CPT-корпуса")
        out[name] = entry
    return out


def _component_roots(manifest: dict | None, name: str) -> list[str]:
    root = ((manifest or {}).get("roots") or {}).get(name)
    return [root] if root else []


# --------------------------------------------------------------------------- #
# Шарды: хеши с диска против манифеста
# --------------------------------------------------------------------------- #


def shards_section(manifest: dict | None, manifest_path: Path | None,
                   e_component: dict) -> list[dict]:
    """Шард-файлы датасета: объявленный хеш манифеста против пересчёта с диска."""
    rows: list[dict] = []
    base = manifest_path.parent if manifest_path else None
    for entry in (manifest or {}).get("shards", []) or []:
        name = str(entry.get("file", ""))
        path = (base / name) if base else Path(name)
        digest = file_digest(path)
        declared = entry.get("sha256")
        rows.append(
            {
                "component": "+".join(sorted((entry.get("by_component") or {}).keys())) or "K/D/S",
                "file": name,
                "records": entry.get("records"),
                "bytes_declared": entry.get("bytes"),
                "bytes": digest["bytes"] if digest else None,
                "sha256_declared": declared,
                "sha256": digest["sha256"] if digest else None,
                "match": bool(digest and declared and digest["sha256"] == declared),
                "present": digest is not None,
            }
        )
    if e_component.get("out"):
        declared = e_component.get("sha256_declared")
        measured = e_component.get("sha256")
        rows.append(
            {
                "component": "E",
                "file": e_component["out"],
                "records": e_component.get("records"),
                "bytes_declared": e_component.get("bytes_declared"),
                "bytes": e_component.get("bytes"),
                "sha256_declared": declared,
                "sha256": measured,
                "match": bool(measured and declared and measured == declared),
                "present": measured is not None,
            }
        )
    return rows


# --------------------------------------------------------------------------- #
# Плагины S: allowlist
# --------------------------------------------------------------------------- #


def plugin_rows(manifest: dict | None) -> tuple[list[dict], list[dict]]:
    """Таблица allowlist (в карточку) и компактный список плагинов вне него."""
    table = {row.get("plugin"): row for row in (manifest or {}).get("plugin_table", []) or []}
    inside: list[dict] = []
    outside: list[dict] = []
    for name in sorted(set(table) | set(cpt_mod.PLUGIN_ALLOWLIST)):
        row = table.get(name, {})
        allowlisted = name in cpt_mod.PLUGIN_ALLOWLIST
        entry = {
            "plugin": name,
            "in_allowlist": allowlisted,
            "in_corpus_run": bool(row.get("in", False)),
            "added_in_delta3": name in cpt_mod.PLUGIN_ALLOWLIST_DELTA3,
            "files": row.get("skills"),
            "records": row.get("records"),
            "approx_tokens": row.get("approx_tokens"),
        }
        if allowlisted:
            if not row:
                entry["reason"] = "в allowlist константы, прогона корпуса с ним ещё не было"
            else:
                entry["reason"] = row.get("reason")
            inside.append(entry)
        else:
            outside.append({"plugin": name, "files": row.get("skills")})
    return inside, outside


# --------------------------------------------------------------------------- #
# Решения и счётчики
# --------------------------------------------------------------------------- #


def decisions_section(manifest: dict | None) -> dict:
    """Решения дельты-3 по составу (то, что нельзя вывести из чисел прогона)."""
    filters = (manifest or {}).get("filters") or {}
    containment = (manifest or {}).get("containment") or {}
    return {
        "plugin_allowlist": {
            "decided": "2026-09-28 (ADR-020, дельта-3)",
            "base": sorted(cpt_mod.PLUGIN_ALLOWLIST_BASE),
            "added_delta3": sorted(cpt_mod.PLUGIN_ALLOWLIST_DELTA3),
            "total": len(cpt_mod.PLUGIN_ALLOWLIST),
            "corpus_run_allowlist": sorted(filters.get("s_plugins") or []),
        },
        "legacy_corpus": {
            "file": LEGACY_CORPUS,
            "share_in_v1": 0.0,
            "reason": (
                "txt-склейка прошлых корпусов (30.8 % пересечения типов с K/D/S): "
                "в v1 не входит — состав v1 целиком собирается сериализатором K/D/S "
                "(ADR-020, дельта-3)"
            ),
        },
        "containment": {
            "role": "монитор пересечения D↔K (не удаляющая ступень)",
            "threshold": containment.get("threshold"),
            "chunk_jaccard": containment.get("chunk_jaccard"),
            # Числа этого прогона — из манифеста, а не из текста решения: после
            # пересборки они меняются, и вердикт не должен пересказывать старые.
            "run": {
                "containers": containment.get("containers"),
                "checked": containment.get("checked"),
                "checked_pairs": containment.get("checked_pairs"),
                "dropped": containment.get("dropped"),
                "seconds": containment.get("seconds"),
            },
            "verdict": (
                "премиса вложения «карточка K внутри дистиллята D» опровергнута "
                "измерением 28.09.2026: на боевом корпусе при пороге окна 0.55 "
                "ступень сняла 0 записей — она работает наблюдателем (no-op) "
                "и оставлена в коде как непрерывный монитор пересечения"
            ),
            "measured_overlap": dict(CONTAINMENT_MEASURED),
        },
        "mixing": {
            "declaration": MIX_DECLARATION,
            "note": (
                "гипотеза долей при упаковке CPT/SFT — в микс-декларации; "
                "в карточке только измеренные доли компонент"
            ),
        },
    }


def counters_section(manifest: dict | None, e_data: dict | None) -> dict:
    """Счётчики скраба и дедупа: по компонентам и суммарно (числа, без значений)."""
    m_redactions = (manifest or {}).get("redactions") or {}
    e_redactions = (e_data or {}).get("redactions") or {}
    e_dedup = (e_data or {}).get("dedup") or {}
    m_dedup = (manifest or {}).get("dedup") or {}
    by_pattern: dict[str, int] = {}
    for source in (e_redactions.get("by_pattern") or {}, m_redactions.get("by_pattern") or {}):
        for pattern, count in source.items():
            by_pattern[pattern] = by_pattern.get(pattern, 0) + int(count)
    return {
        "redactions": {
            "total": int(e_redactions.get("total", 0)) + int(m_redactions.get("total", 0)),
            "by_component": {
                "E": int(e_redactions.get("total", 0)),
                "K/D/S": int(m_redactions.get("total", 0)),
            },
            "by_pattern": dict(sorted(by_pattern.items())),
        },
        "dedup": {
            "E": {
                "exact": e_dedup.get("exact"),
                "near": e_dedup.get("near"),
                "kept": e_dedup.get("kept"),
            },
            "K/D/S": {
                "exact": m_dedup.get("exact"),
                "near": m_dedup.get("near"),
                "kept": m_dedup.get("kept"),
                "candidates": m_dedup.get("candidates"),
            },
            "cross_component": "общий пул дедупа K∪D∪S; E дедуплицируется отдельно",
        },
        # Near-dup MinHash-LSH на боевом объёме даёт коллизии (нечёткие совпадения
        # похожих текстов) — это известное поведение ступени, а не ошибка прогона.
        # Повторный дедуп выхода даёт 0/0 по решению архитектора дельты-3b: числа
        # источника названы явно, чтобы карточка не выдавала их за перезамер.
        "near_dup": {
            "lsh_collisions": {
                "K/D/S": m_dedup.get("near"),
                "E": e_dedup.get("near"),
            },
            "known_behavior": True,
            "re_dedup": "0/0",
            "source": (
                "решение архитектора (дельта-3b): near-dup LSH-коллизии — известное "
                "поведение ступени; повторный дедуп выхода корпуса — 0/0 (точных и "
                "near-дублей не остаётся)"
            ),
        },
    }


# --------------------------------------------------------------------------- #
# Доли и статус
# --------------------------------------------------------------------------- #


def shares_section(volumes: dict[str, int | None]) -> dict:
    """Доли компонент по approx-токенам: только измеренные, остальные помечены."""
    measured = {
        name: int(value)
        for name, value in volumes.items()
        if isinstance(value, int) and value > 0
    }
    total = sum(measured.values())
    shares = {name: round(value / total, 4) for name, value in measured.items()} if total else {}
    return {
        "basis": "approx-tokens",
        "total_approx_tokens": total,
        "shares": shares,
        "measured": [name for name in COMPONENT_ORDER if name in measured],
        "unmeasured": [name for name in COMPONENT_ORDER if name not in measured],
    }


def draft_reasons(
    components: dict[str, dict], shards: list[dict], plugins: list[dict],
    manifest: dict | None, shares: dict,
) -> list[str]:
    """Механические причины статуса ``v1-draft``: пусто — карточка финальна."""
    reasons: list[str] = []
    if manifest is None:
        reasons.append("нет манифеста CPT-корпуса K/D/S: компоненты K/D/S не измерены")
    else:
        run_allowlist = sorted(((manifest.get("filters") or {}).get("s_plugins")) or [])
        if run_allowlist and run_allowlist != sorted(cpt_mod.PLUGIN_ALLOWLIST):
            reasons.append(
                f"allowlist корпуса ({len(run_allowlist)}) не совпадает с константой "
                f"({len(cpt_mod.PLUGIN_ALLOWLIST)}): корпус собран до расширения дельты-3 — "
                "нужна пересборка K/D/S"
            )
    if not components.get("E", {}).get("present"):
        reasons.append("нет артефакта сборки эпизодов (E): пересборка не завершена")
    elif components["E"].get("approx_tokens") is None:
        reasons.append("E-артефакт без объёма: доля компоненты E не измерена")
    if components.get("E", {}).get("sha256") is None and components.get("E", {}).get("out"):
        reasons.append("E: файл эпизодов не найден по пути отчёта — хеш не сверен")
    for row in shards:
        if not row.get("present"):
            reasons.append(f"шард {row.get('file')} не найден на диске: хеш не сверен")
        elif not row.get("match"):
            reasons.append(
                f"шард {row.get('file')}: sha256 не сошёлся с объявленным "
                "(пересчёт с диска)"
            )
    pending = [row["plugin"] for row in plugins
               if row.get("in_allowlist") and not row.get("in_corpus_run")]
    if pending:
        reasons.append(
            "в allowlist константы, но не в прогоне корпуса: "
            + ", ".join(pending)
            + " — нужна пересборка K/D/S"
        )
    if shares.get("unmeasured"):
        reasons.append(
            "доли не полны: не измерены " + ", ".join(shares["unmeasured"])
        )
    return reasons


# --------------------------------------------------------------------------- #
# Сборка карточки
# --------------------------------------------------------------------------- #


def build_card(
    *,
    root: str | os.PathLike[str] = DEFAULT_ROOT,
    episodes: str | os.PathLike[str] | None = None,
    cpt_manifest: str | os.PathLike[str] | None = None,
    cpt_report: str | os.PathLike[str] | None = None,
) -> dict:
    """Собрать карточку датасета (машиночитаемый вид) из артефактов компонент."""
    root_path = Path(os.path.expanduser(os.fspath(root)))
    manifest_path = _first_existing([cpt_manifest, root_path / cpt_mod.MANIFEST_NAME])
    manifest = load_json(manifest_path)
    report_path = _first_existing([cpt_report, _report_for(manifest, manifest_path)])
    report = load_json(report_path)
    episodes_path = _first_existing(
        [episodes, root_path / EPISODES_MANIFEST_NAME, root_path / EPISODES_REPORT_NAME]
    )
    e_data = load_json(episodes_path)

    components: dict[str, dict] = {"E": component_e(episodes_path, e_data)}
    components.update(components_cpt(manifest, report))

    shards = shards_section(manifest, manifest_path, components["E"])
    plugins, plugins_outside = plugin_rows(manifest)
    shares = shares_section(
        {name: components[name].get("approx_tokens") for name in COMPONENT_ORDER}
    )
    reasons = draft_reasons(components, shards, plugins, manifest, shares)

    card = {
        "card": CARD_NAME,
        "card_version": CARD_VERSION,
        "status": STATUS_DRAFT if reasons else STATUS_FINAL,
        "status_reasons": reasons,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dataset_root": home_rel(root_path),
        "repos_symlink": f"data/datasets/{CARD_NAME}",
        "decision": "ADR-020 (дельта-1..3), ADR-004 (карточка + хеш), ADR-005 (SFT-стадия)",
        "components": components,
        "shares": shares,
        "shards": shards,
        "plugin_allowlist": plugins,
        "plugin_allowlist_outside": {
            "plugins": plugins_outside,
            "files": sum(int(item.get("files") or 0) for item in plugins_outside),
        },
        "decisions": decisions_section(manifest),
        "counters": counters_section(manifest, e_data),
        "artifacts": {
            "cpt_manifest": home_rel(manifest_path) if manifest_path else None,
            "cpt_report": home_rel(report_path) if report_path else None,
            "episodes": home_rel(episodes_path) if episodes_path else None,
            "cpt_manifest_bytes": file_digest(manifest_path)["bytes"] if manifest_path else None,
            "episodes_bytes": file_digest(episodes_path)["bytes"] if episodes_path else None,
        },
        "notes": [
            "доли компонент — по approx_tokens (chars // 4, ADR-021), измеренные, "
            "не гипотеза: гипотеза упаковки — в микс-декларации",
            "приватные корпуса в репозиторий не попадают: в git — карточка и симлинк "
            "(C-032/C-033, AD-6)",
        ],
    }
    return card


# --------------------------------------------------------------------------- #
# Человекочитаемая карточка
# --------------------------------------------------------------------------- #


def render_markdown(card: dict) -> str:
    """Карточка датасета в человекочитаемом виде (по образцу карточки претрейна)."""
    draft = card["status"] == STATUS_DRAFT
    lines: list[str] = [
        f"# Карточка датасета `{card['card']}` — " + ("ЧЕРНОВИК" if draft else "финал"),
        "",
        f"- **Статус:** `{card['status']}`, собрана {card['generated_at']}",
        f"- **Решение:** {card['decision']}",
        f"- **Расположение:** `{card['dataset_root']}/` (C-032/C-033: корпуса на gb10-shared); "
        f"в репозитории — симлинк `{card['repos_symlink']}` на этот каталог",
        "- **Инструменты сборки:** `tools/axiom_ds/build.py` (E), "
        "`tools/axiom_ds/cpt_serialize.py` (K/D/S), `tools/axiom_ds/card.py` (эта карточка)",
        "",
    ]
    if draft:
        lines += [
            "**Почему черновик** (проверки карточки, а не оценка):",
            "",
        ]
        lines += [f"- {reason}" for reason in card["status_reasons"]]
        lines.append("")

    components = card["components"]
    shares = card["shares"]
    lines += [
        "## 1. Состав и измеренные доли",
        "",
        _table(
            ["Компонент", "Что это", "Записей", "Файлов", "approx-токенов", "Доля",
             "Основа доли"],
            [
                [
                    name,
                    components[name]["title"],
                    _num(components[name].get("records")),
                    _num(components[name].get("files")),
                    _num(components[name].get("approx_tokens")),
                    _pct(shares["shares"].get(name)),
                    components[name].get("approx_tokens_basis") or "—",
                ]
                for name in COMPONENT_ORDER
            ]
            + [["**Итого**", "", "", "", _num(shares["total_approx_tokens"]), "100.0 %", ""]],
        ),
        "",
        f"Доли считаются по измеренным компонентам ({', '.join(shares['measured']) or '—'}); "
        + (
            f"не измерены: {', '.join(shares['unmeasured'])}."
            if shares["unmeasured"]
            else "измерены все компоненты, доли в сумме 100 %."
        ),
        "Мера объёма — `approx_tokens = max(1, len(text) // 4)` (ADR-021), общая для E и K/D/S; "
        "упаковка в 8K-последовательности — при CPT-лупе.",
        "",
        "## 2. Шард-файлы и хеши",
        "",
        _table(
            ["Компонент", "Файл", "Записей", "Байт", "sha256 (пересчёт с диска)",
             "Сверка с манифестом"],
            [
                [
                    row["component"],
                    row["file"],
                    _num(row.get("records")),
                    _num(row.get("bytes")),
                    (row.get("sha256") or "—")[:16] + "…" if row.get("sha256") else "—",
                    "совпал" if row.get("match") else ("не сверен" if not row.get("present")
                                                      else "расхождение"),
                ]
                for row in card["shards"]
            ],
        ),
        "",
        "Полные хеши — в машинной карточке (`" + DEFAULT_CARD_JSON.name + "`). "
        "Сверка пересчитывает sha256 по файлам на диске: расхождение — сигнал "
        "подмены/дрейфа шарда, а не «шум отчёта».",
        "",
        "## 3. Происхождение: источники и правила отбора",
        "",
    ]
    artifacts = card["artifacts"]
    lines += [
        "- артефакты, из которых собрана карточка: манифест CPT "
        f"`{artifacts['cpt_manifest'] or '—'}`, отчёт CPT `{artifacts['cpt_report'] or '—'}`, "
        f"артефакт эпизодов `{artifacts['episodes'] or '—'}`",
        "",
    ]

    for name in COMPONENT_ORDER:
        entry = components[name]
        lines.append(f"**{name}** — {entry['title']}")
        lines.append("")
        if entry.get("sources"):
            lines += ["- источники: " + ", ".join(f"`{src}`" for src in entry["sources"])]
        else:
            lines.append("- источники: — (артефакта прогона нет)")
        if name == "E" and entry.get("artifact"):
            stamp = entry.get("generated_at") or "без отметки времени"
            lines.append(f"- артефакт прогона: `{entry['artifact']}` ({stamp})")
        if name == "E":
            rules = entry["rules"]
            lines.append(
                f"- класс исхода — механический: контракт сессии либо парный harness-отчёт; "
                f"классы {', '.join(f'`{item}`' for item in rules['classes'])}"
            )
            lines.append(
                f"- в SFT-компонент: {', '.join(f'`{item}`' for item in rules['sft_classes'])}; "
                f"`verified-partial` — статус `partial` плюс зелёный сьют в последних "
                f"{rules['green_suite_window']} tool-результатах "
                f"(`{rules['green_suite_pattern']}` при отсутствии `{rules['red_suite_pattern']}`)"
            )
            lines.append(
                f"- скраб и deny-list — до эпизодизации: значения секретов до карточки "
                f"не доходят, в отчёте только счётчики"
            )
            if entry.get("sft_partial_ready") is not None:
                share = entry.get("sft_partial_share")
                of_records = entry.get("sft_partial_share_of_records")
                lines.append(
                    f"- SFT-компонент: `verified-complete` {_num(entry.get('sft_ready'))} + "
                    f"`verified-partial` (флаг `partial-green`) "
                    f"{_num(entry['sft_partial_ready'])}"
                    + (f" — {_pct(share)} компоненты" if share is not None else "")
                    + (f" ({_pct(of_records)} всех записанных E)" if of_records is not None else "")
                    + f"; negative-пул `verified-failed` {_num(entry.get('negative_ready'))}"
                )
            classes = entry.get("by_class") or {}
            if classes:
                lines.append(
                    "- состав по классам (записано): "
                    + ", ".join(f"`{key}` {_num(value)}" for key, value in sorted(classes.items()))
                )
        else:
            filters = {key: value for key, value in (entry.get("filters") or {}).items()
                       if value not in (None, [], {})}
            for key, value in filters.items():
                rendered = ", ".join(f"`{item}`" for item in value) if isinstance(value, list) else value
                lines.append(f"- {key}: {rendered}")
            if entry.get("by_type"):
                lines.append("- состав источника по типам карточек: " + _counts(entry["by_type"]))
            if entry.get("by_level"):
                lines.append("- уровни карточек: " + _counts(entry["by_level"]))
            if entry.get("plugins"):
                lines.append("- записано по плагинам: " + _counts(entry["plugins"]))
        lines.append("")

    lines += [
        "## 4. Решения по составу",
        "",
    ]
    allowlist = card["decisions"]["plugin_allowlist"]
    measured_overlap = card["decisions"]["containment"]["measured_overlap"]
    run = card["decisions"]["containment"]["run"]
    lines += [
        f"- **Allowlist плагинов:** решение владельца {allowlist['decided']}; "
        f"база {len(allowlist['base'])} + расширение дельты-3 {len(allowlist['added_delta3'])} "
        f"= {allowlist['total']} плагинов.",
        f"- **Legacy-корпус** `{card['decisions']['legacy_corpus']['file']}` — доля в v1 "
        f"{card['decisions']['legacy_corpus']['share_in_v1']:.0%}: "
        f"{card['decisions']['legacy_corpus']['reason']}",
        f"- **Контейнмент D→K** — {card['decisions']['containment']['role']}; "
        f"{card['decisions']['containment']['verdict']}. "
        f"Прогон: контейнеров {_num(run['containers'])}, проверок {_num(run['checked'])}, "
        f"снято {_num(run['dropped'])} (порог {_num(card['decisions']['containment']['threshold'])}, "
        f"окно {_num(card['decisions']['containment']['chunk_jaccard'])}). "
        f"Замер пересечения по парам «одна статья»: max оконный Jaccard "
        f"{_num(measured_overlap['max_window_jaccard'])} при пороге "
        f"{_num(measured_overlap['threshold'])}, пар ≥ порога — "
        f"{_num(measured_overlap['pairs_at_or_above_threshold'])} "
        f"({_num(measured_overlap['pairs_measured'])} пар; {measured_overlap['source']})",
        f"- **Микс при упаковке** — {card['decisions']['mixing']['note']} "
        f"(`{card['decisions']['mixing']['declaration']}`).",
        "",
        "## 5. Плагины S: таблица allowlist",
        "",
        _table(
            ["Плагин", "В allowlist", "В прогоне корпуса", "Файлов", "Записей",
             "approx-токенов"],
            [
                [
                    row["plugin"] + (" **+Δ3**" if row["added_in_delta3"] else ""),
                    "да" if row["in_allowlist"] else "нет",
                    "да" if row["in_corpus_run"] else "нет",
                    _num(row.get("files")),
                    _num(row.get("records")),
                    _num(row.get("approx_tokens")),
                ]
                for row in card["plugin_allowlist"]
            ],
        ),
        "",
        f"Вне allowlist: {len(card['plugin_allowlist_outside']['plugins'])} плагинов / "
        f"{_num(card['plugin_allowlist_outside']['files'])} файлов SKILL.md — "
        "в корпус не входят (чужой домен контура).",
        "",
    ]
    if not any(row.get("approx_tokens") for row in card["plugin_allowlist"]):
        lines += [
            "Столбец approx-токенов заполняется для шардов, собранных после дельты-3: "
            f"в артефакте прогона `{artifacts['cpt_manifest'] or '—'}` пофайловых токенов "
            "ещё нет — стоят прочерки, а не нули.",
            "",
        ]
    near_dup = card["counters"]["near_dup"]
    lines += [
        "## 6. Скраб и дедуп (счётчики)",
        "",
        _table(
            ["Контур", "Замен скраба", "Точных дублей", "Near-дублей", "Оставлено"],
            [
                ["E (эпизоды)",
                 _num(card["counters"]["redactions"]["by_component"]["E"]),
                 _num(card["counters"]["dedup"]["E"]["exact"]),
                 _num(card["counters"]["dedup"]["E"]["near"]),
                 _num(card["counters"]["dedup"]["E"]["kept"])],
                ["K/D/S (CPT-корпус)",
                 _num(card["counters"]["redactions"]["by_component"]["K/D/S"]),
                 _num(card["counters"]["dedup"]["K/D/S"]["exact"]),
                 _num(card["counters"]["dedup"]["K/D/S"]["near"]),
                 _num(card["counters"]["dedup"]["K/D/S"]["kept"])],
            ],
        ),
        "",
        "Правила скраба и их счётчики (по шаблонам):",
        "",
        _table(
            ["Правило", "Замен"],
            [[name, _num(count)] for name, count in
             card["counters"]["redactions"]["by_pattern"].items()] or [["—", "—"]],
        ),
        "",
        f"Дедуп K/D/S — {card['counters']['dedup']['cross_component']} "
        f"(пересечения D↔K, S↔D↔K ожидаемы); шум E-контура в пул K/D/S не попадает.",
        "",
        f"Near-dup LSH-коллизии — известное поведение ступени "
        f"(K/D/S: {_num(near_dup['lsh_collisions']['K/D/S'])}, "
        f"E: {_num(near_dup['lsh_collisions']['E'])}); повторный дедуп выхода — "
        f"{near_dup['re_dedup']}. Источник числа: {near_dup['source']}.",
        "",
        "## 7. Что не сделано и открытые вопросы",
        "",
    ]
    open_items = list(card["status_reasons"]) or [
        "проверки карточки пройдены; состав зафиксирован"
    ]
    lines += [f"- {item}" for item in open_items]
    lines += [
        "- финальные доли микса (K+D+S домен / E-эпизоды / публичный претрейн-текст) "
        "определяются при CPT-лупе — не карточкой",
        "- токенизация и упаковка 8K — вне этого пайплайна (`net/data.py`)",
        "",
        "## 8. Воспроизведение",
        "",
        "```bash",
        "# тесты пайплайнов (синтетика, без сети)",
        "python -m pytest tools/tests/test_axiom_ds.py tools/tests/test_cpt_serialize.py "
        "tools/tests/test_axiom_ds_card.py -q",
        "",
        "# карточка из артефактов прогонов",
        "python -m axiom_ds.card build-card",
        "",
        "# пересборка компоненты E (сессии → эпизоды)",
        "python -m axiom_ds.build --source ~/.claude/projects \\",
        f"    --out {card['dataset_root']}/episodes-v1.jsonl \\",
        f"    --report {card['dataset_root']}/episodes-v1-report.json \\",
        f"    --sft-out {card['dataset_root']}/episodes-v1-sft.jsonl",
        "",
        "# пересборка CPT-корпуса K/D/S (полный прогон, ~40 мин)",
        "python -m axiom_ds.cpt_serialize build-cpt --restart",
        "```",
        "",
        "Приватные корпуса в git не попадают: в репозитории — эта карточка и симлинк "
        "`data/datasets/axiom-domain-ds-v1` на `~/gb10-shared` (C-032/C-033, AD-6).",
        "",
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def run_build_card(
    *,
    root: str | os.PathLike[str] = DEFAULT_ROOT,
    episodes: str | os.PathLike[str] | None = None,
    cpt_manifest: str | os.PathLike[str] | None = None,
    cpt_report: str | os.PathLike[str] | None = None,
    out_md: str | os.PathLike[str] | None = DEFAULT_CARD_MD,
    out_json: str | os.PathLike[str] | None = DEFAULT_CARD_JSON,
) -> dict:
    """Собрать карточку и записать её (md — человекочитаемая, json — машинная)."""
    card = build_card(root=root, episodes=episodes, cpt_manifest=cpt_manifest,
                      cpt_report=cpt_report)
    markdown = render_markdown(card)
    card["outputs"] = {
        "markdown": str(out_md) if out_md else None,
        "json": str(out_json) if out_json else None,
    }
    if out_json:
        pp_common.write_report(out_json, card)
    if out_md:
        target = Path(os.path.expanduser(os.fspath(out_md)))
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(markdown, encoding="utf-8")
    return card


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="axiom_ds.card",
        description="Карточка датасета axiom-domain-ds-v1 из артефактов компонент "
                    "(ADR-020, дельта-3)",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build-card", help="собрать карточку датасета")
    build.add_argument("--root", default=DEFAULT_ROOT,
                       help=f"корень датасета (по умолчанию {DEFAULT_ROOT})")
    build.add_argument("--episodes", default=None,
                       help="артефакт компоненты E (манифест или отчёт сборки эпизодов)")
    build.add_argument("--cpt-manifest", default=None, help=f"манифест CPT ({cpt_mod.MANIFEST_NAME})")
    build.add_argument("--cpt-report", default=None, help="отчёт прогона CPT")
    build.add_argument("--out-md", default=str(DEFAULT_CARD_MD),
                       help=f"человекочитаемая карточка (по умолчанию {DEFAULT_CARD_MD})")
    build.add_argument("--out-json", default=str(DEFAULT_CARD_JSON),
                       help=f"машинная карточка (по умолчанию {DEFAULT_CARD_JSON})")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command != "build-card":  # pragma: no cover - argparse не пропустит
        return 2
    card = run_build_card(
        root=args.root,
        episodes=args.episodes,
        cpt_manifest=args.cpt_manifest,
        cpt_report=args.cpt_report,
        out_md=args.out_md,
        out_json=args.out_json,
    )
    shares = card["shares"]["shares"]
    print(
        "[axiom-card] статус {status}, доли: {shares}, md {md}, json {js}".format(
            status=card["status"],
            shares=", ".join(f"{name} {value:.1%}" for name, value in shares.items()) or "—",
            md=args.out_md,
            js=args.out_json,
        ),
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
