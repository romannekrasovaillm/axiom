"""T-card1..T-card6 — карточка датасета ``axiom-domain-ds-v1`` (ADR-020, дельта-3).

Карточка — артефакт ADR-004: состав, доли, sha256 шард-файлов, происхождение,
решения. Проверяется не «текст красиво отрендерился», а что **числа карточки
сходятся с артефактами прогонов** и что хеши сверены с диском:

* **T-card1** — доли по approx-токенам сходятся с манифестом CPT-корпуса и
  отчётом сборки эпизодов; неполные доли помечены, а не досчитаны догадкой;
* **T-card2** — sha256 каждого шарда совпадает с пересчётом файла с диска;
  подмена файла делает карточку черновиком с причиной;
* **T-card3** — таблица allowlist присутствует: 22 базовых + 9 плагинов дельты-3
  с пометкой, столбцы «файлов» и «approx-токенов» заполнены из манифеста;
* **T-card4** — карточка пишется двумя файлами (md + json), CLI возвращает 0;
* **T-card5** — происхождение: пути источников в форме ``<HOME>/…``, приватные
  абсолютные пути в карточку не попадают;
* **T-card6** — статус механический: полный набор артефактов — ``v1``, отсутствие
  компоненты E — ``v1-draft`` с причиной;
* **T-card7** (дельта-3b) — финальные поля: контейнмент с числами прогона **и**
  замером пересечения, near-dup LSH-коллизии как известное поведение с названным
  источником числа, доля ``verified-partial`` (флаг ``partial-green``) числом.

Фикстуры синтетические: корпус K/D/S собирается реальным сериализатором, эпизоды
E — реальным сборщиком на синтетических сессиях; приватная библиотека в тесты не
попадает.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

TOOLS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS_DIR))

from axiom_ds import build as build_mod  # noqa: E402
from axiom_ds import card as card_mod  # noqa: E402
from axiom_ds import cpt_serialize as cpt  # noqa: E402
from axiom_ds import dedup as dedup_mod  # noqa: E402
from test_cpt_serialize import make_roots  # noqa: E402  (фикстуры K/D/S — те же)

CONTRACT_COMPLETE = (
    "Готово.\n```json\n"
    '{"status": "complete", "assumptions": [], "open_questions": [], '
    '"conflicts_with_prior_decisions": []}'
    "\n```\n"
)


# --------------------------------------------------------------------------- #
# Фикстуры: корень датасета как на gb10-shared
# --------------------------------------------------------------------------- #


def write_session(path: Path, sid: str, salt: str, contract: str | None) -> Path:
    """Синтетическая сессия агента: один эпизод с tool-циклом и контрактом."""
    stamp = "2026-09-%02dT10:%02d:00.000Z"
    body = " ".join(f"{salt}{index:03d}" for index in range(80))

    def event(kind: str, **kwargs) -> dict:
        base = {"sessionId": sid, "uuid": kwargs.pop("uuid"), "parentUuid": None}
        base.update(kwargs)
        base["type"] = kind
        return base

    events = [
        event("user", uuid="u1", timestamp=stamp % (10, 0),
              message={"role": "user", "content": f"Задача {salt}: разбери {body}"}),
        event("assistant", uuid="a1", timestamp=stamp % (10, 1), message={"role": "assistant",
              "content": [{"type": "tool_use", "id": "t1", "name": "Bash",
                           "input": {"command": f"pytest -q -k {salt}"}}]}),
        event("user", uuid="r1", timestamp=stamp % (10, 2), message={"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "t1",
             "content": f"{salt}: проверки пройдены, 4 passed in 0.31s {body[:40]}"}]}),
    ]
    if contract:
        events.append(
            event("assistant", uuid="a2", timestamp=stamp % (10, 3),
                  message={"role": "assistant", "content": [{"type": "text",
                                                             "text": contract}]})
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for item in events:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    return path


def make_dataset_root(tmp_path: Path) -> Path:
    """Корень датасета: CPT-корпус K/D/S (реальный прогон) + эпизоды E (реальный прогон)."""
    root = tmp_path / "gb10-shared" / "datasets" / "axiom-domain-ds-v1"
    root.mkdir(parents=True, exist_ok=True)
    cpt.run_build_cpt(
        out=root / "cpt-kds-v0.1.jsonl.gz",
        report=root / "cpt-kds-v0.1-report.json",
        manifest=root / "manifest-cpt.json",
        codec="gzip",
        **make_roots(tmp_path),
    )
    sessions = tmp_path / "sessions"
    write_session(sessions / "p1" / "s-1.jsonl", "sess-1", "альфа", CONTRACT_COMPLETE)
    write_session(sessions / "p2" / "s-2.jsonl", "sess-2", "бета", CONTRACT_COMPLETE)
    build_mod.run_build(
        source=[sessions],
        out=root / "episodes-v1.jsonl",
        report=root / "episodes-v1-report.json",
    )
    return root


# --------------------------------------------------------------------------- #
# T-card1 — доли сходятся с манифестами
# --------------------------------------------------------------------------- #


def test_t_card1_shares_match_manifest_and_report(tmp_path: Path):
    """Доли компонент — измеренные: сходятся с манифестом CPT и отчётом E."""
    root = make_dataset_root(tmp_path)
    card = card_mod.build_card(root=root)
    components = card["components"]
    manifest = json.loads((root / "manifest-cpt.json").read_text(encoding="utf-8"))
    report = json.loads((root / "episodes-v1-report.json").read_text(encoding="utf-8"))

    for name in ("K", "D", "S"):
        row = manifest["by_component"][name]
        assert components[name]["records"] == row["records"], name
        # дельта-3: пофайловые approx-токены есть в манифесте, доля не выводится
        assert row["approx_tokens"] > 0
        assert components[name]["approx_tokens"] == row["approx_tokens"], name
        assert components[name]["approx_tokens_basis"] == "manifest"

    assert components["E"]["records"] == report["episodes_written"]
    assert components["E"]["approx_tokens"] == report["approx_tokens"]
    assert components["E"]["present"] is True

    # отчёт CPT-прогона найден рядом с манифестом (префикс — из имени шарда)
    assert card["artifacts"]["cpt_report"].endswith("cpt-kds-v0.1-report.json")
    assert components["K"]["by_type"], "счётчики отчёта CPT не подхвачены"

    total = sum(components[name]["approx_tokens"] for name in card_mod.COMPONENT_ORDER)
    assert card["shares"]["total_approx_tokens"] == total
    assert card["shares"]["unmeasured"] == []
    assert set(card["shares"]["shares"]) == set(card_mod.COMPONENT_ORDER)
    for name, share in card["shares"]["shares"].items():
        assert share == round(components[name]["approx_tokens"] / total, 4), name
    assert abs(sum(card["shares"]["shares"].values()) - 1.0) < 0.001

    markdown = card_mod.render_markdown(card)
    for share in card["shares"]["shares"].values():
        assert f"{share * 100:.1f} %" in markdown


def test_t_card1_unmeasured_component_is_marked_not_guessed(tmp_path: Path):
    """Отчёт E без объёма — компонента помечена неполной, доли не досчитаны.

    Так выглядит старый отчёт (до дельты-3: без chars/approx_tokens): карточка
    обязана сказать «не измерено», а не подставить ноль или долю по числу записей.
    """
    root = make_dataset_root(tmp_path)
    old = json.loads((root / "episodes-v1-report.json").read_text(encoding="utf-8"))
    for key in ("chars", "approx_tokens", "generated_at", "out_sha256", "out_bytes"):
        old.pop(key, None)
    old_path = tmp_path / "episodes-v1-report-old.json"
    old_path.write_text(json.dumps(old, ensure_ascii=False), encoding="utf-8")

    card = card_mod.build_card(root=root, episodes=old_path)
    assert card["components"]["E"]["approx_tokens"] is None
    assert "E" in card["shares"]["unmeasured"]
    assert card["status"] == card_mod.STATUS_DRAFT
    assert any("E-артефакт без объёма" in reason for reason in card["status_reasons"])
    # доли остальных компонент при этом посчитаны и не «раздуты» до 100 %
    assert abs(sum(card["shares"]["shares"].values()) - 1.0) < 0.001
    markdown = card_mod.render_markdown(card)
    assert "не измерены: E" in markdown


# --------------------------------------------------------------------------- #
# T-card2 — sha256 совпадают с диска
# --------------------------------------------------------------------------- #


def test_t_card2_shard_hashes_match_disk(tmp_path: Path):
    """Хеш каждого шарда карточки — пересчёт файла на диске, а не пересказ манифеста."""
    root = make_dataset_root(tmp_path)
    card = card_mod.build_card(root=root)
    assert card["shards"], "шарды не попали в карточку"

    for row in card["shards"]:
        path = root / row["file"]
        measured = hashlib.sha256(path.read_bytes()).hexdigest()
        assert row["sha256"] == measured, row["file"]
        assert row["bytes"] == path.stat().st_size
        assert row["sha256"] == row["sha256_declared"]
        assert row["match"] is True, row["file"]
    assert len(card["shards"][0]["sha256"]) == 64


def test_t_card2_tampered_shard_makes_card_draft(tmp_path: Path):
    """Подмена шарда: хеш с диска расходится с манифестом — черновик с причиной."""
    root = make_dataset_root(tmp_path)
    shard = root / "cpt-kds-v0.1-00000.jsonl.gz"
    shard.write_bytes(shard.read_bytes() + "\nподмена\n".encode("utf-8"))

    card = card_mod.build_card(root=root)
    row = next(item for item in card["shards"] if item["file"].endswith(".jsonl.gz"))
    assert row["match"] is False
    assert row["sha256"] != row["sha256_declared"]
    assert card["status"] == card_mod.STATUS_DRAFT
    assert any("sha256 не сошёлся" in reason for reason in card["status_reasons"])


def test_t_card2_missing_shard_is_reported(tmp_path: Path):
    """Пропавший шард — причина черновика, а не молчаливая пустая строка."""
    root = make_dataset_root(tmp_path)
    (root / "cpt-kds-v0.1-00000.jsonl.gz").unlink()

    card = card_mod.build_card(root=root)
    row = next(item for item in card["shards"] if item["file"].endswith(".jsonl.gz"))
    assert row["present"] is False and row["sha256"] is None
    assert card["status"] == card_mod.STATUS_DRAFT
    assert any("не найден на диске" in reason for reason in card["status_reasons"])


# --------------------------------------------------------------------------- #
# T-card3 — таблица allowlist
# --------------------------------------------------------------------------- #


def test_t_card3_allowlist_table_present(tmp_path: Path):
    """Таблица allowlist: 31 плагин, 9 помечены дельтой-3, столбцы из манифеста."""
    root = make_dataset_root(tmp_path)
    card = card_mod.build_card(root=root)

    table = card["plugin_allowlist"]
    assert {row["plugin"] for row in table} == set(cpt.PLUGIN_ALLOWLIST)
    added = {row["plugin"] for row in table if row["added_in_delta3"]}
    assert added == set(cpt.PLUGIN_ALLOWLIST_DELTA3)
    assert len(added) == 9
    for row in table:
        assert row["in_allowlist"] is True
        assert row["in_corpus_run"] is True, row["plugin"]
        assert isinstance(row["files"], int)          # файлов в источнике
        assert isinstance(row["approx_tokens"], int)  # записано в корпус
    assert card["plugin_allowlist_outside"]["plugins"], "вне allowlist пусто?"

    markdown = card_mod.render_markdown(card)
    for name in cpt.PLUGIN_ALLOWLIST_DELTA3:
        assert name in markdown, name
    assert "| Файлов |" in markdown and "approx-токенов" in markdown
    assert "banking" not in {row["plugin"] for row in table}  # чужой домен — вне таблицы


def test_t_card3_delta3_decision_is_recorded(tmp_path: Path):
    """Решение об allowlist зафиксировано в карточке (база + расширение)."""
    root = make_dataset_root(tmp_path)
    card = card_mod.build_card(root=root)
    allowlist = card["decisions"]["plugin_allowlist"]

    assert set(allowlist["base"]) == set(cpt.PLUGIN_ALLOWLIST_BASE)
    assert set(allowlist["added_delta3"]) == set(cpt.PLUGIN_ALLOWLIST_DELTA3)
    assert allowlist["total"] == len(cpt.PLUGIN_ALLOWLIST) == 31
    assert allowlist["corpus_run_allowlist"] == sorted(cpt.PLUGIN_ALLOWLIST)
    assert card["decisions"]["legacy_corpus"]["share_in_v1"] == 0.0


# --------------------------------------------------------------------------- #
# T-card4 — запись карточки
# --------------------------------------------------------------------------- #


def test_t_card4_run_build_card_writes_markdown_and_json(tmp_path: Path):
    """Карточка пишется двумя файлами: md для человека, json для машин."""
    root = make_dataset_root(tmp_path)
    md_path = tmp_path / "card.md"
    json_path = tmp_path / "card.json"

    card = card_mod.run_build_card(root=root, out_md=md_path, out_json=json_path)
    assert md_path.is_file() and json_path.is_file()

    stored = json.loads(json_path.read_text(encoding="utf-8"))
    assert stored["card"] == "axiom-domain-ds-v1"
    assert stored["card_version"] == card_mod.CARD_VERSION
    assert stored["status"] == card["status"]
    assert stored["shares"] == card["shares"]
    assert stored["generated_at"]

    markdown = md_path.read_text(encoding="utf-8")
    assert markdown.startswith("# Карточка датасета `axiom-domain-ds-v1`")
    for section in ("## 1. Состав", "## 2. Шард-файлы", "## 3. Происхождение",
                    "## 5. Плагины S", "## 6. Скраб и дедуп", "## 8. Воспроизведение"):
        assert section in markdown, section
    assert card["outputs"]["markdown"] == str(md_path)


def test_t_card4_cli_build_card(tmp_path: Path, capsys: pytest.CaptureFixture):
    """CLI `build-card`: код возврата 0 и карточка на диске."""
    root = make_dataset_root(tmp_path)
    md_path = tmp_path / "cli-card.md"
    json_path = tmp_path / "cli-card.json"

    code = card_mod.main([
        "build-card",
        "--root", str(root),
        "--out-md", str(md_path),
        "--out-json", str(json_path),
    ])
    assert code == 0
    assert md_path.is_file() and json_path.is_file()
    assert "axiom-card" in capsys.readouterr().err


# --------------------------------------------------------------------------- #
# T-card5 — происхождение без приватных путей
# --------------------------------------------------------------------------- #


def test_t_card5_sources_are_home_relative(tmp_path: Path):
    """Источники — в форме <HOME>/…: абсолютные приватные пути в карточку не идут."""
    root = make_dataset_root(tmp_path)
    card = card_mod.build_card(root=root)
    assert card_mod.home_rel(Path.home() / "library" / "concepts") == "<HOME>/library/concepts"

    for name in ("K", "D", "S"):
        for source in card["components"][name]["sources"]:
            # корни фикстур лежат вне домашнего каталога (/tmp) и остаются как есть,
            # но домашний префикс в карточку не попадает ни в одном виде
            assert str(Path.home()) not in source, source
    # корень датасета в домашнем каталоге записывается как <HOME>/…
    absent = card_mod.build_card(root=Path.home() / "нет-такого-датасета")
    assert absent["dataset_root"] == "<HOME>/нет-такого-датасета"
    assert absent["status"] == card_mod.STATUS_DRAFT
    assert absent["shards"] == [] and "нет манифеста" in " ".join(absent["status_reasons"])
    # фильтры источников зафиксированы в карточке (происхождение, не только объём)
    assert card["components"]["K"]["filters"]["k_types"]
    assert card["components"]["D"]["filters"]["d_subdirs"]
    assert card["components"]["S"]["filters"]["skill_filename"] == "SKILL.md"


def test_t_card5_scrub_and_dedup_counters_present(tmp_path: Path):
    """Счётчики скраба и дедупа — в карточке (числами, без значений секретов)."""
    root = make_dataset_root(tmp_path)
    card = card_mod.build_card(root=root)
    assert "total" in card["counters"]["redactions"]
    assert "by_pattern" in card["counters"]["redactions"]
    assert card["counters"]["dedup"]["E"]["kept"] is not None
    assert card["counters"]["dedup"]["K/D/S"]["kept"] is not None


# --------------------------------------------------------------------------- #
# T-card6 — статус карточки
# --------------------------------------------------------------------------- #


def test_t_card6_full_set_of_artifacts_gives_final_status(tmp_path: Path):
    """Все проверки сошлись (свежие артефакты, хеши, allowlist) — статус v1."""
    root = make_dataset_root(tmp_path)
    card = card_mod.build_card(root=root)
    assert card["status"] == card_mod.STATUS_FINAL
    assert card["status_reasons"] == []
    assert "финал" in card_mod.render_markdown(card).splitlines()[0]


def test_t_card6_missing_episodes_artifact_is_draft(tmp_path: Path):
    """Нет артефакта E (пересборка идёт) — карточка черновик с причиной."""
    root = make_dataset_root(tmp_path)
    (root / "episodes-v1-report.json").unlink()

    card = card_mod.build_card(root=root)
    assert card["components"]["E"]["present"] is False
    assert card["status"] == card_mod.STATUS_DRAFT
    assert any("нет артефакта сборки эпизодов" in reason for reason in card["status_reasons"])
    assert "ЧЕРНОВИК" in card_mod.render_markdown(card).splitlines()[0]


def test_t_card6_allowlist_mismatch_is_draft(tmp_path: Path):
    """Корпус собран до расширения allowlist — черновик: нужна пересборка K/D/S."""
    root = make_dataset_root(tmp_path)
    manifest_path = root / "manifest-cpt.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["filters"]["s_plugins"] = sorted(cpt.PLUGIN_ALLOWLIST_BASE)
    for row in manifest["plugin_table"]:
        if row["plugin"] in cpt.PLUGIN_ALLOWLIST_DELTA3:
            row["in"] = False
            row["records"] = 0
            row["approx_tokens"] = 0
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")

    card = card_mod.build_card(root=root)
    assert card["status"] == card_mod.STATUS_DRAFT
    reasons = " ".join(card["status_reasons"])
    assert "не совпадает с константой" in reasons
    assert "не в прогоне корпуса" in reasons


# --------------------------------------------------------------------------- #
# T-card7 — финальные поля дельты-3b (контейнмент, near-dup, verified-partial)
# --------------------------------------------------------------------------- #


def test_t_card7_containment_has_run_numbers_and_measurement(tmp_path: Path):
    """Контейнмент — монитор no-op: числа прогона из манифеста + замер пересечения.

    Числа прогона карточка обязана читать из манифеста (после пересборки они
    меняются), а замер «max оконный Jaccard 0.3524» — нести отдельно, с источником:
    сам прогон даёт нули и о величине пересечения не говорит.
    """
    root = make_dataset_root(tmp_path)
    card = card_mod.build_card(root=root)
    containment = card["decisions"]["containment"]
    manifest = json.loads((root / "manifest-cpt.json").read_text(encoding="utf-8"))

    run = containment["run"]
    assert run["containers"] == manifest["containment"]["containers"]
    assert run["checked"] == manifest["containment"]["checked"]
    assert run["dropped"] == 0                      # монитор: на боевом корпусе no-op
    assert containment["role"].startswith("монитор")

    measured = containment["measured_overlap"]
    assert measured["max_window_jaccard"] == 0.3524
    assert measured["pairs_at_or_above_threshold"] == 0
    assert measured["pairs_measured"] > 0
    assert "ADR-020" in measured["source"]
    # порог замера — порог ступени (дрейф одного без другого ломает тест)
    assert measured["threshold"] == dedup_mod.CHUNK_JACCARD
    assert "0.3524" in card_mod.render_markdown(card)


def test_t_card7_near_dup_known_behavior_is_attributed(tmp_path: Path):
    """near-dup LSH-коллизии — известное поведение; источник числа назван явно."""
    root = make_dataset_root(tmp_path)
    card = card_mod.build_card(root=root)
    near_dup = card["counters"]["near_dup"]
    assert near_dup["known_behavior"] is True
    assert near_dup["re_dedup"] == "0/0"
    assert "дельта-3b" in near_dup["source"]
    assert set(near_dup["lsh_collisions"]) == {"K/D/S", "E"}
    # числа коллизий — из прогона, а не из текста решения
    assert near_dup["lsh_collisions"]["K/D/S"] == card["counters"]["dedup"]["K/D/S"]["near"]
    assert "известное поведение" in card_mod.render_markdown(card)


def test_t_card7_verified_partial_share_is_measured(tmp_path: Path):
    """Доля verified-partial (флаг partial-green) — числом, а не только флагом."""
    root = make_dataset_root(tmp_path)
    card = card_mod.build_card(root=root)
    entry = card["components"]["E"]
    complete = entry["sft_ready"]
    partial = entry["sft_partial_ready"]
    total = complete + partial
    assert entry["sft_partial_share"] == (round(partial / total, 4) if total else None)
    assert entry["sft_partial_share_of_records"] == round(partial / entry["records"], 4)
    assert "`verified-partial`" in card_mod.render_markdown(card)
