"""T-k1..T-p1 — сериализатор K/D/S → доменный CPT-корпус (ADR-020, дельта-2).

Проверяются свойства, от которых зависит боевой прогон 310k файлов библиотеки:

* **T-k1** — фильтр типов K: только объявленные типы, прочие (`behavioral_*`,
  `attack_strategy`, `interpretive_framework`, `boundary_*`) не читаются;
* **T-d1** — фильтр каталогов D: `2_статьи` и `3_блоги` в корпус, `1_методология`
  (про процесс дистилляции) — вне;
* **T-s2** — фильтр плагинов S по ``PLUGIN_ALLOWLIST``; вложенные пути
  ``<плагин>/…/SKILL.md`` и не-SKILL.md-файлы учтены; архивы (`_archive`,
  `_inbox_*`) — вне;
* **T-scr1** — скраб той же первой ступенью: секрет-приманка не доживает до jsonl;
* **T-hom1** — абсолютные пути `/home/roman/…` заменяются на `<HOME>/…`;
* **T-e1** — карточка с пустым телом (только frontmatter) не пишется;
* **T-ttl1** — тело K получает заголовок `# <title>` из frontmatter (H1 в телах нет);
* **T-dd1** — дедуп МЕЖКОМПОНЕНТНЫЙ: один пул хешей на K∪D∪S, дубль K↔D остаётся один;
* **T-f1** — манифест шарда: `file/bytes/sha256/records/by_component`, sha256
  совпадает с файлом на диске (проверяется `prep_pretrain.common.verify_manifest`);
* **T-l1** — `--limit-files N` — предел НА КОМПОНЕНТУ (проба покрывает все три);
* **T-r1** — resume-курсор: продолжение прогона не дублирует и не теряет записи;
* **T-c1** — CLI: jsonl + отчёт построены, обязательные ключи записи на месте,
  `/home/roman` в выходе нет, примеры id — без содержимого;
* **T-pl1** — таблица «плагин → в/вне» в отчёте;
* **T-p1** — каталоги по умолчанию: проба → `/tmp/axiom-cpt-probe`, бой → gb10-shared.

Все фикстуры синтетические: реальная библиотека, реальные скиллы и реальные
секреты в тесты не попадают (приманки собраны из очевидно тестовых значений).
"""

from __future__ import annotations

import gzip
import hashlib
import json
import sys
from pathlib import Path

import pytest

TOOLS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS_DIR))

from axiom_ds import cpt_serialize as cpt  # noqa: E402
from prep_pretrain import common as pp_common  # noqa: E402

# --------------------------------------------------------------------------- #
# Секреты-приманки (синтетические, не боевые)
# --------------------------------------------------------------------------- #

OPENAI_KEY = "sk-proj-A1b2C3d4E5f6G7h8I9j0K1l2M3n4O5p6Q7r8S9t0"
ENV_ASSIGNMENT = "AXIOM_SERVICE_TOKEN=Zx9Qw8Er7Ty6Ui5Op4As3Df2Gh1Jk0"
PRIVATE_KEY_BLOCK = (
    "-----BEGIN RSA PRIVATE KEY-----\n"
    "MIIEowIBAAKCAQEAxTESTONLYxNOTAREALKEYxMIIEowIBAAKCAQEA\n"
    "-----END RSA PRIVATE KEY-----"
)

CARD_BODY = """## Определение
Тестовый приём контроля стоимости внимания: понижение размерности проекций QKV.

## Мотивация
Полная размерность дорога; тестовое тело карточки достаточно длинное для записи.
"""


def unique_filler(tag: str, words: int = 120) -> str:
    """Уникальный длинный текст фикстуры: соседние документы не near-dup друг другу."""
    return " ".join(f"{tag}_слово{i:03d}" for i in range(words))


# --------------------------------------------------------------------------- #
# Фикстуры
# --------------------------------------------------------------------------- #


def write_card(path: Path, slug: str, ctype: str, body: str, title: str | None = None,
               level: str = "α", frontmatter: bool = True) -> Path:
    """Синтетическая карточка K: frontmatter объявленного вида + тело."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if frontmatter:
        head = (
            "---\n"
            f"slug: {slug}\n"
            f"type: {ctype}\n"
            f"level: {level}\n"
            "formality: B\n"
            f"title: {title or slug.replace('_', ' ').title()}\n"
            "family: Efficiency Techniques\n"
            "sources: [arxiv.2601.00001]\n"
            "---\n\n"
        )
        path.write_text(head + body, encoding="utf-8")
    else:
        path.write_text(body, encoding="utf-8")
    return path


def write_md(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def make_k_root(root: Path) -> Path:
    """K-корень: по карточке в каждом объявленном типе плюс отсекаемые типы."""
    for ctype in sorted(cpt.K_TYPES):
        body = CARD_BODY + "\n## Секции\n" + unique_filler(ctype) + "\n"
        write_card(root / ctype / f"{ctype}_card.md", f"{ctype}_card", ctype, body)
    for ctype in ("behavioral_capability", "attack_strategy", "interpretive_framework",
                  "boundary_condition"):
        write_card(root / ctype / "x.md", "x", ctype, CARD_BODY + unique_filler(ctype) + "\n")
    write_md(root / "README.md", "# Корень типов\n\nСлужебный файл, не карточка.\n")
    return root


def make_d_root(root: Path) -> Path:
    write_md(root / "2_статьи" / "digest-1.md",
             "# Дайджест 1\n\n" + CARD_BODY + unique_filler("дайджест") + "\n")
    write_md(root / "2_статьи" / "вложенный" / "digest-2.md",
             "# Дайджест 2\n\nтекст дистиллята\n" + unique_filler("вложенный") + "\n")
    write_md(root / "3_блоги" / "blog-1.md",
             "# Блог\n\nразбор блога\n" + unique_filler("блог") + "\n")
    write_md(root / "1_методология" / "process.md", "# Методология дистилляции\n\nпро процесс\n")
    return root


def make_s_root(root: Path) -> Path:
    write_md(root / "laguna" / "skills" / "skill-a" / "SKILL.md",
             "---\nname: skill-a\n---\n\nтело\n" + unique_filler("скилл-а") + "\n")
    write_md(
        root / "laguna" / "skills" / "group" / "skill-b" / "SKILL.md",
        "---\nname: skill-b\n---\n\nвложенный уровень\n" + unique_filler("скилл-б") + "\n",
    )
    write_md(root / "agent-harness" / "skills" / "h" / "SKILL.md",
             "---\nname: h\n---\n\nтело\n" + unique_filler("харнесс") + "\n")
    write_md(root / "banking" / "skills" / "c" / "SKILL.md", "---\nname: c\n---\n\nчужой домен\n")
    write_md(root / "_archive" / "laguna" / "old" / "SKILL.md", "---\nname: old\n---\n\nархив\n")
    write_md(root / "skills" / "skills" / "tmpl" / "assets" / "t" / "SKILL.md", "шаблон\n")
    write_md(root / "laguna" / "notes" / "readme.md", "не SKILL.md\n")
    return root


def make_roots(tmp_path: Path) -> dict[str, Path]:
    return {
        "k_root": make_k_root(tmp_path / "concepts"),
        "d_root": make_d_root(tmp_path / "distillate"),
        "s_root": make_s_root(tmp_path / "plugins"),
    }


def run(tmp_path: Path, **kwargs) -> dict:
    """Прогон на синтетике: вывод в tmp_path, gzip (stdlib, без zstd)."""
    roots = make_roots(tmp_path)
    out = kwargs.pop("out", tmp_path / "out" / "cpt-kds-v0.1.jsonl.gz")
    params = dict(
        out=out,
        report=tmp_path / "out" / "report.json",
        manifest=tmp_path / "out" / "manifest-cpt.json",
        codec="gzip",
        **roots,
    )
    params.update(kwargs)
    return cpt.run_build_cpt(**params)


def read_records(out_dir: Path, prefix: str = "cpt-kds-v0.1") -> list[dict]:
    """Записи всех шардов каталога вывода (в порядке номеров шардов)."""
    records: list[dict] = []
    for path in sorted(Path(out_dir).glob(f"{prefix}-*")):
        if path.name.endswith(".part"):
            continue
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rt", encoding="utf-8") as handle:
            records.extend(json.loads(line) for line in handle if line.strip())
    return records


# --------------------------------------------------------------------------- #
# T-k1 / T-d1 / T-s2 — фильтры источников
# --------------------------------------------------------------------------- #


def test_t_k1_type_filter_keeps_only_declared_types(tmp_path: Path):
    """K: читаются только объявленные типы каталогов; прочие — не открываются."""
    root = make_k_root(tmp_path / "concepts")
    files = cpt.discover_k(root, limit=None)

    types_seen = {f.path.parent.name for f in files}
    assert types_seen == set(cpt.K_TYPES)
    assert len(files) == len(cpt.K_TYPES)
    # каталоги объявленных типов, которых нет на диске, не выдумывают файлов
    assert not any("behavioral" in str(f.path) for f in files)
    assert not any(f.path.parent.name == "attack_strategy" for f in files)


def test_t_d1_distillate_subdir_filter(tmp_path: Path):
    """D: 2_статьи и 3_блоги в корпус, 1_методология — вне (это про процесс)."""
    root = make_d_root(tmp_path / "distillate")
    files = cpt.discover_d(root, limit=None)
    names = sorted(f.path.name for f in files)

    assert names == ["blog-1.md", "digest-1.md", "digest-2.md"]
    assert all(f.path.parts[-3] != "1_методология" for f in files)


def test_t_s2_plugin_allowlist(tmp_path: Path):
    """S: плагин берётся по первому компоненту пути; архивы и чужие домены — вне."""
    root = make_s_root(tmp_path / "plugins")
    files = cpt.discover_s(root, limit=None)
    rel = sorted(str(f.path.relative_to(root)) for f in files)

    assert rel == [
        "agent-harness/skills/h/SKILL.md",
        "laguna/skills/group/skill-b/SKILL.md",
        "laguna/skills/skill-a/SKILL.md",
    ]
    assert cpt.plugin_of(Path("laguna/skills/group/skill-b/SKILL.md")) == "laguna"
    assert "banking" not in cpt.PLUGIN_ALLOWLIST
    assert "laguna" in cpt.PLUGIN_ALLOWLIST


def test_t_s2_plugin_inventory_counts_all_plugins(tmp_path: Path):
    """Инвентарь плагинов — по всем SKILL.md источника, с признаком allowlist."""
    root = make_s_root(tmp_path / "plugins")
    inventory = cpt.plugin_inventory(root)

    assert inventory["laguna"] == 2
    assert inventory["agent-harness"] == 1
    assert inventory["banking"] == 1
    assert inventory["_archive"] == 1


# --------------------------------------------------------------------------- #
# T-scr1 / T-hom1 — скраб и замена абсолютных путей
# --------------------------------------------------------------------------- #


def test_t_scr1_secrets_scrubbed_before_write(tmp_path: Path):
    """Скраб — первой ступенью: приманки не доживают до jsonl."""
    body = (
        "## Определение\n"
        f"Ключ доступа: {ENV_ASSIGNMENT}\n"
        f"Резервный: AKIAIOSFODNN7EXAMPLE и {OPENAI_KEY}\n"
        f"{PRIVATE_KEY_BLOCK}\n"
        "## Секции\n"
        "Тестовое тело карточки длиной больше порога записи.\n"
        + unique_filler("утечка") + "\n"
    )
    write_card(tmp_path / "concepts" / "algorithmic_primitive" / "leaky.md", "leaky",
               "algorithmic_primitive", body)
    write_md(tmp_path / "distillate" / "2_статьи" / "leak.md",
             f"# Дайджест\n\n{ENV_ASSIGNMENT}\n" + unique_filler("дистиллят-утечка") + "\n")
    out_dir = tmp_path / "out"
    report = cpt.run_build_cpt(
        out=out_dir / "cpt-kds-v0.1.jsonl.gz",
        report=out_dir / "report.json",
        codec="gzip",
        k_root=tmp_path / "concepts",
        d_root=tmp_path / "distillate",
        s_root=tmp_path / "nonexistent",
    )
    records = read_records(out_dir)
    assert len(records) == 2

    raw = gzip.open(out_dir / "cpt-kds-v0.1-00000.jsonl.gz", "rt", encoding="utf-8").read()
    for leak in (OPENAI_KEY, "Zx9Qw8Er7Ty6Ui5Op4As3Df2Gh1Jk0", "AKIAIOSFODNN7EXAMPLE",
                 "MIIEowIBAAKCAQEAxTESTONLY"):
        assert leak not in raw
    assert "<REDACTED>" in raw
    assert report["redactions"]["total"] >= 2
    assert report["components"]["K"]["redactions"]["total"] >= 1


def test_t_hom1_home_paths_replaced(tmp_path: Path):
    """Абсолютные пути владельца в тексте и в source_path — под <HOME>."""
    body = ("## Определение\nФайл лежит в /home/roman/library/concepts/x.md и читается кодом.\n"
            "## Секции\nТело карточки достаточно длинное для записи в корпус.\n"
            + unique_filler("домашний-путь") + "\n")
    write_card(tmp_path / "concepts" / "benchmark" / "h.md", "h", "benchmark", body)
    out_dir = tmp_path / "out"
    cpt.run_build_cpt(
        out=out_dir / "cpt-kds-v0.1.jsonl.gz",
        report=out_dir / "report.json",
        codec="gzip",
        k_root=tmp_path / "concepts",
        d_root=tmp_path / "none",
        s_root=tmp_path / "none",
    )
    record = read_records(out_dir)[0]

    assert "<HOME>/library/concepts/x.md" in record["text"]
    assert "/home/roman" not in record["text"]
    assert "/home/roman" not in record["source_path"]


# --------------------------------------------------------------------------- #
# T-e1 / T-ttl1 — форма записи K
# --------------------------------------------------------------------------- #


def test_t_e1_empty_body_card_skipped(tmp_path: Path):
    """Карточка из одного frontmatter не пишется: тело пустое, учить нечему."""
    root = tmp_path / "concepts"
    write_card(root / "hypothesis" / "empty.md", "empty", "hypothesis", "")
    write_card(root / "hypothesis" / "full.md", "full", "hypothesis",
               CARD_BODY + unique_filler("полный") + "\n")
    out_dir = tmp_path / "out"
    report = cpt.run_build_cpt(
        out=out_dir / "cpt-kds-v0.1.jsonl.gz",
        report=out_dir / "report.json",
        codec="gzip",
        k_root=root,
        d_root=tmp_path / "none",
        s_root=tmp_path / "none",
    )
    records = read_records(out_dir)

    assert len(records) == 1
    assert report["components"]["K"]["skipped"]["empty_body"] == 1
    assert report["components"]["K"]["files_read"] == 2


def test_t_e2_unterminated_frontmatter_skipped(tmp_path: Path):
    """Оборванный frontmatter (без закрывающего ``---``) — не запись корпуса.

    В библиотеке есть обрезанные карточки-заготовки: весь файл — YAML-строки
    ``slug/type/title``. Как проза это мусор, и в CPT-корпус он не идёт.
    """
    root = tmp_path / "concepts"
    (root / "algorithmic").mkdir(parents=True)
    (root / "algorithmic" / "broken.md").write_text(
        "---\nslug: broken\ntype: algorithmic\n", encoding="utf-8"
    )
    write_card(root / "algorithmic_primitive" / "ok.md", "ok", "algorithmic_primitive",
               CARD_BODY + unique_filler("целая") + "\n")
    out_dir = tmp_path / "out"
    report = cpt.run_build_cpt(
        out=out_dir / "cpt-kds-v0.1.jsonl.gz",
        report=out_dir / "report.json",
        codec="gzip",
        k_root=root,
        d_root=tmp_path / "none",
        s_root=tmp_path / "none",
    )
    records = read_records(out_dir)

    assert len(records) == 1
    assert report["components"]["K"]["skipped"]["unterminated_frontmatter"] == 1
    assert "slug:" not in json.dumps(records, ensure_ascii=False)


def test_t_ttl1_card_title_becomes_heading(tmp_path: Path):
    """Заголовок карточки (из frontmatter) не теряется: тело получает H1."""
    root = tmp_path / "concepts"
    write_card(root / "task" / "c.md", "c", "task", CARD_BODY, title="Attention Downcasting")
    out_dir = tmp_path / "out"
    cpt.run_build_cpt(
        out=out_dir / "cpt-kds-v0.1.jsonl.gz",
        report=out_dir / "report.json",
        codec="gzip",
        k_root=root,
        d_root=tmp_path / "none",
        s_root=tmp_path / "none",
    )
    record = read_records(out_dir)[0]

    assert record["text"].startswith("# Attention Downcasting\n")
    assert "slug:" not in record["text"]  # frontmatter в корпус не уходит
    assert "## Определение" in record["text"]


# --------------------------------------------------------------------------- #
# T-dd1 — межкомпонентный дедуп
# --------------------------------------------------------------------------- #


def test_t_dd1_cross_component_dedup_keeps_one(tmp_path: Path):
    """Один и тот же текст в K и D → остаётся одна запись (первый компонент)."""
    same = "## Определение\nОдин и тот же текст карточки и дистиллята в двух компонентах.\n"
    write_card(tmp_path / "concepts" / "algorithmic_primitive" / "same.md", "same",
               "algorithmic_primitive", same)
    write_md(tmp_path / "distillate" / "2_статьи" / "same.md", same)
    out_dir = tmp_path / "out"
    report = cpt.run_build_cpt(
        out=out_dir / "cpt-kds-v0.1.jsonl.gz",
        report=out_dir / "report.json",
        codec="gzip",
        k_root=tmp_path / "concepts",
        d_root=tmp_path / "distillate",
        s_root=tmp_path / "none",
    )
    records = read_records(out_dir)

    assert len(records) == 1
    assert records[0]["component"] == "K"
    assert report["dedup"]["exact"] == 1
    assert report["components"]["D"]["dropped_dedup"] == 1
    assert report["by_component"]["K"]["records"] == 1


def test_t_dd1_cross_component_dedup_is_near_dup(tmp_path: Path):
    """Near-dup (не точный дубль) между S и D тоже снимается общим пулом.

    Фикстура короткая осознанно: у ``Deduper`` MinHash-оценка достоверна, пока
    уникальных шинглов не больше ``MAX_SHINGLES`` (иначе выборка по значениям
    хешей расходится — см. ``cpt.near_dup_calibration`` и T-cal1).
    """
    head = "# Скилл\n\n" + " ".join(f"шаг{i}" for i in range(120))
    write_md(tmp_path / "plugins" / "laguna" / "skills" / "s" / "SKILL.md", head + "\nфинал скилла\n")
    write_md(tmp_path / "distillate" / "2_статьи" / "d.md", head + "\nдругой финал дистиллята\n")
    out_dir = tmp_path / "out"
    report = cpt.run_build_cpt(
        out=out_dir / "cpt-kds-v0.1.jsonl.gz",
        report=out_dir / "report.json",
        codec="gzip",
        k_root=tmp_path / "none",
        d_root=tmp_path / "distillate",
        s_root=tmp_path / "plugins",
    )
    records = read_records(out_dir)

    assert len(records) == 1
    assert records[0]["component"] == "D"
    assert report["dedup"]["near"] == 1


# --------------------------------------------------------------------------- #
# T-cal1 — калибровка near-dup: где оценка Deduper ещё работает
# --------------------------------------------------------------------------- #


def test_t_cal1_near_dup_calibration_documents_estimator_domain():
    """Калибровка: MinHash-оценка верна до MAX_SHINGLES, выше — теряет чутьё.

    Характеризационный тест зависимости (``dedup.py``): если дельта-3 поднимет
    ``MAX_SHINGLES`` или сменит схему выборки шинглов, ожидания калибровки
    обновляются вместе с ней — молчаливая деградация дедупа недопустима.
    """
    calibration = cpt.near_dup_calibration(sizes=(40, 120, 600))
    pairs = {row["words"]: row for row in calibration["pairs"]}

    assert calibration["max_shingles"] == cpt.dedup_mod.MAX_SHINGLES
    assert pairs[120]["detected"] is True
    assert abs(pairs[120]["true_jaccard"] - pairs[120]["estimated_jaccard"]) < 0.15
    assert pairs[600]["unique_shingles"] > calibration["max_shingles"]
    assert pairs[600]["true_jaccard"] > 0.8
    assert pairs[600]["detected"] is False  # оценка обрушена сэмплированием шинглов


# --------------------------------------------------------------------------- #
# T-f1 — манифест со sha256
# --------------------------------------------------------------------------- #


def test_t_f1_manifest_hashes_match_files(tmp_path: Path):
    """Манифест: file/bytes/sha256/records/by_component; хеш — по файлу на диске."""
    out_dir = tmp_path / "out"
    cpt.run_build_cpt(
        out=out_dir / "cpt-kds-v0.1.jsonl.gz",
        report=out_dir / "report.json",
        manifest=out_dir / "manifest-cpt.json",
        codec="gzip",
        **make_roots(tmp_path),
    )
    manifest = json.loads((out_dir / "manifest-cpt.json").read_text(encoding="utf-8"))
    entry = manifest["shards"][0]
    blob = (out_dir / entry["file"]).read_bytes()

    assert entry["sha256"] == hashlib.sha256(blob).hexdigest()
    assert entry["bytes"] == len(blob)
    assert set(entry["by_component"]) <= {"K", "D", "S"}
    assert entry["records"] == sum(entry["by_component"].values())

    check = pp_common.verify_manifest(manifest, out_dir)
    assert check["bad"] == []
    assert check["ok"] == len(manifest["shards"])


# --------------------------------------------------------------------------- #
# T-l1 / T-r1 — предел на компоненту и resume-курсор
# --------------------------------------------------------------------------- #


def test_t_l1_limit_files_is_per_component(tmp_path: Path):
    """--limit-files N — предел на компоненту: проба видит все три источника."""
    out_dir = tmp_path / "out"
    report = cpt.run_build_cpt(
        out=out_dir / "cpt-kds-v0.1.jsonl.gz",
        report=out_dir / "report.json",
        codec="gzip",
        limit_files=2,
        **make_roots(tmp_path),
    )
    by_component = report["by_component"]

    assert by_component["K"]["files_read"] == 2
    assert by_component["D"]["files_read"] == 2
    assert by_component["S"]["files_read"] == 2
    # в K типов восемь: предел режет и внутри компоненты
    assert by_component["K"]["files_total"] == len(cpt.K_TYPES)


def test_t_r1_resume_cursor_no_duplicates(tmp_path: Path):
    """Resume: второй прогон продолжает с курсора, записи не дублируются."""
    roots = make_roots(tmp_path)
    out_dir = tmp_path / "out"
    first = cpt.run_build_cpt(
        out=out_dir / "cpt-kds-v0.1.jsonl.gz",
        report=out_dir / "r1.json",
        manifest=out_dir / "manifest-cpt.json",
        codec="gzip",
        limit_files=1,
        **roots,
    )
    second = cpt.run_build_cpt(
        out=out_dir / "cpt-kds-v0.1.jsonl.gz",
        report=out_dir / "r2.json",
        manifest=out_dir / "manifest-cpt.json",
        codec="gzip",
        limit_files=2,
        **roots,
    )
    records = read_records(out_dir)
    ids = [r["id"] for r in records]

    assert first["by_component"]["K"]["records"] == 1
    assert first["totals"]["records"] == 3  # окно limit=1 по каждой компоненте
    assert second["resume"]["cursor_by_component"] == {"K": 1, "D": 1, "S": 1}
    assert second["by_component"]["K"]["files_read"] == 1  # не 2: один уже прочитан
    assert len(ids) == len(set(ids))
    assert len(records) == 3 + 3  # состав = первые два файла каждой компоненты


def test_t_r2_restart_drops_previous_shards(tmp_path: Path):
    """--restart: прежние шарды снимаются, корпус собирается заново (не смешивается)."""
    roots = make_roots(tmp_path)
    out_dir = tmp_path / "out"
    for limit in (1, 2):
        assert cpt.main([
            "build-cpt",
            "--out", str(out_dir / "cpt-kds-v0.1.jsonl.gz"),
            "--report", str(out_dir / "report.json"),
            "--manifest", str(out_dir / "manifest-cpt.json"),
            "--codec", "gzip",
            "--limit-files", str(limit),
            "--k-root", str(roots["k_root"]),
            "--d-root", str(roots["d_root"]),
            "--s-root", str(roots["s_root"]),
        ]) == 0
    assert len(read_records(out_dir)) == 6  # без --restart прогоны дописываются

    assert cpt.main([
        "build-cpt",
        "--out", str(out_dir / "cpt-kds-v0.1.jsonl.gz"),
        "--report", str(out_dir / "report.json"),
        "--manifest", str(out_dir / "manifest-cpt.json"),
        "--codec", "gzip",
        "--limit-files", "1",
        "--restart",
        "--k-root", str(roots["k_root"]),
        "--d-root", str(roots["d_root"]),
        "--s-root", str(roots["s_root"]),
    ]) == 0
    records = read_records(out_dir)

    assert len(records) == 3  # окно limit=1 по трём компонентам, состав пересобран
    manifest = json.loads((out_dir / "manifest-cpt.json").read_text(encoding="utf-8"))
    assert manifest["resume_cursor"] == {"K": 0, "D": 0, "S": 0}
    assert len(manifest["shards"]) == 1


def test_t_opt1_min_chars_and_title_prefix_knobs(tmp_path: Path):
    """Порог длины и заголовок карточки — управляемые ручки, обе видны в отчёте."""
    roots = make_roots(tmp_path)
    out_dir = tmp_path / "out"
    report = cpt.run_build_cpt(
        out=out_dir / "cpt-kds-v0.1.jsonl.gz",
        report=out_dir / "report.json",
        codec="gzip",
        k_root=roots["k_root"],
        d_root=roots["d_root"],
        s_root=roots["s_root"],
        min_chars=100_000,
        title_prefix=False,
    )

    assert report["totals"]["records"] == 0
    assert report["filters"]["min_text_chars"] == 100_000
    assert report["filters"]["k_title_prefix"] is False
    assert report["components"]["K"]["skipped"]["too_short"] == len(cpt.K_TYPES)

    report = cpt.run_build_cpt(
        out=out_dir / "cpt-kds-v0.1.jsonl.gz",
        report=out_dir / "report2.json",
        manifest=out_dir / "manifest2.json",
        codec="gzip",
        k_root=roots["k_root"],
        d_root=tmp_path / "none",
        s_root=tmp_path / "none",
        title_prefix=False,
    )
    k_record = read_records(out_dir)[0]
    assert k_record["text"].startswith("## Определение")


# --------------------------------------------------------------------------- #
# T-c1 / T-pl1 — CLI и отчёт
# --------------------------------------------------------------------------- #


def test_t_c1_cli_writes_jsonl_and_report(tmp_path: Path):
    """CLI: jsonl + числовой отчёт; в записи — ключи контракта, без абсолютных путей."""
    roots = make_roots(tmp_path)
    out_dir = tmp_path / "out"
    code = cpt.main([
        "build-cpt",
        "--out", str(out_dir / "cpt-kds-v0.1.jsonl.gz"),
        "--report", str(out_dir / "report.json"),
        "--codec", "gzip",
        "--k-root", str(roots["k_root"]),
        "--d-root", str(roots["d_root"]),
        "--s-root", str(roots["s_root"]),
    ])
    assert code == 0

    records = read_records(out_dir)
    assert records
    for record in records:
        assert set(record) == {"id", "component", "source_path", "text"}
        assert record["component"] in {"K", "D", "S"}
        assert record["id"]
        assert record["text"].strip()
        assert "/home/roman" not in json.dumps(record, ensure_ascii=False)

    report = json.loads((out_dir / "report.json").read_text(encoding="utf-8"))
    assert report["status"] == "ok"
    assert len(report["sample_ids"]) == 3
    assert all(isinstance(item, str) for item in report["sample_ids"])
    blob = json.dumps(report, ensure_ascii=False)
    assert "## Определение" not in blob  # примеры id — без содержимого
    assert report["components"]["K"]["records_written"] == len(cpt.K_TYPES)
    # распределение длин — свидетельство о составе корпуса (для карточки датасета);
    # фикстурные карточки длиннее 1000 символов, тонких среди них быть не может
    buckets_k = report["components"]["K"]["length_buckets"]
    assert buckets_k["1000-3000"] + buckets_k["3000+"] == len(cpt.K_TYPES)
    for component in ("K", "D", "S"):
        buckets = report["components"][component]["length_buckets"]
        assert sum(buckets.values()) == report["components"][component]["records_written"]


def test_t_c1_cli_rejects_output_outside_allowed_roots(tmp_path: Path):
    """Вывод вне gb10-shared//tmp отклоняется (C-032/C-033), код возврата 2."""
    roots = make_roots(tmp_path)
    code = cpt.main([
        "build-cpt",
        "--out", str(TOOLS_DIR / "cpt-kds.jsonl.gz"),
        "--k-root", str(roots["k_root"]),
        "--d-root", str(roots["d_root"]),
        "--s-root", str(roots["s_root"]),
    ])
    assert code == 2


def test_t_pl1_plugin_table_in_report(tmp_path: Path):
    """Таблица «плагин → в/вне → почему» строится по факту содержимого."""
    out_dir = tmp_path / "out"
    report = cpt.run_build_cpt(
        out=out_dir / "cpt-kds-v0.1.jsonl.gz",
        report=out_dir / "report.json",
        codec="gzip",
        **make_roots(tmp_path),
    )
    table = {row["plugin"]: row for row in report["plugin_table"]}

    assert set(table) >= {"laguna", "agent-harness", "banking", "_archive"}
    assert table["laguna"]["in"] is True
    assert table["laguna"]["skills"] == 2
    assert table["laguna"]["records"] == 2
    assert table["banking"]["in"] is False
    assert table["banking"]["reason"]
    assert table["_archive"]["in"] is False


def test_t_pl1_allowlist_matches_card(tmp_path: Path):
    """Константа — фильтр домена axiom из карточки ADR-020 (22 плагина)."""
    assert cpt.PLUGIN_ALLOWLIST == {
        "laguna", "agentic-rl", "data-curator", "verification", "effort", "frontier-lab",
        "frontier-intelligence", "document-tools", "misc-tools", "agent-infra",
        "agent-harnesses", "agent-harness", "cpt", "pretrain", "patterns-resilience",
        "patterns-integration", "aws-builders", "arch-core", "dka", "kimi",
        "tui-agent-skills", "arch-distilled",
    }


# --------------------------------------------------------------------------- #
# T-p1 — каталоги по умолчанию
# --------------------------------------------------------------------------- #


def test_t_p1_default_output_roots():
    """Проба пишет в /tmp, бой — в канонический корень датасета на gb10-shared."""
    assert cpt.default_out(probe=True).startswith("/tmp/axiom-cpt-probe/")
    assert cpt.default_out(probe=False).startswith(
        str(Path("~/gb10-shared/datasets/axiom-domain-ds-v1").expanduser())
    )
    assert cpt.default_out(probe=False).endswith("cpt-kds-v0.1.jsonl.zst")


def test_t_p1_probe_run_writes_under_tmp(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """--probe без --out кладёт шарды в /tmp/axiom-cpt-probe (каталог пробы)."""
    probe_root = tmp_path / "probe"
    monkeypatch.setattr(cpt, "PROBE_ROOT", str(probe_root))
    roots = make_roots(tmp_path)
    code = cpt.main([
        "build-cpt", "--probe", "--codec", "gzip", "--limit-files", "1",
        "--k-root", str(roots["k_root"]),
        "--d-root", str(roots["d_root"]),
        "--s-root", str(roots["s_root"]),
    ])
    assert code == 0
    assert (probe_root / "cpt-kds-v0.1-00000.jsonl.gz").exists()


def test_t_p1_probe_report_path_default(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Отчёт пробы по умолчанию ложится рядом с шардами и содержит те же числа."""
    probe_root = tmp_path / "probe"
    monkeypatch.setattr(cpt, "PROBE_ROOT", str(probe_root))
    roots = make_roots(tmp_path)
    assert cpt.main([
        "build-cpt", "--probe", "--codec", "gzip", "--limit-files", "1",
        "--k-root", str(roots["k_root"]),
        "--d-root", str(roots["d_root"]),
        "--s-root", str(roots["s_root"]),
    ]) == 0
    report = json.loads((probe_root / "report-cpt.json").read_text(encoding="utf-8"))
    assert report["probe"] is True
    assert report["totals"]["records"] == 3
