"""T-q1..T-q5 — шард Q (decay-фаза ADR-021): сужёный фильтр W + код с тестами.

Проверяются свойства, от которых зависит decay-выборка:

* **T-q1** — сужёный фильтр веб-части: edu-порог ``int_score >= 4`` и окно длины
  (короткие и слишком длинные документы отбрасываются, причины — в отчёте);
* **T-q2** — кодовая часть: в Q проходят только записи с тест-маркерами
  (``def test_``/``unittest``/``pytest``), остальные фильтры шарда C действуют,
  найденные маркеры фиксируются в мете записи;
* **T-q3** — доля кода: микс 85/15 по approx-токенам измеряется на выходе
  (отчёт ``mix`` сходится с содержимым шардов, отклонение — в пределах допуска);
* **T-q4** — дедуп против W/C: точный дубль и near-дубль опорной записи не
  попадают в Q, непохожая запись попадает; при выключенной опоре прогон
  отказывается стартовать, а не «дедуплицирует молча точным хешем»;
* **T-q5** — resume: прерванный прогон продолжается с места (max_records →
  полный), ни одна запись не потеряна и не размножена; смена потока в том же
  манифесте — отказ.

Все фикстуры синтетические (локальные jsonl), сеть не нужна.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

TOOLS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS_DIR))

from prep_pretrain import common, fineweb, stack  # noqa: E402

MB = 1024 * 1024


# --------------------------------------------------------------------------- #
# Фикстуры
# --------------------------------------------------------------------------- #


def write_jsonl(path: Path, records: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return path


def incompressible(index: int, chars: int) -> str:
    """Детерминированный несжимаемый текст (ротация шардов предсказуема)."""
    blocks: list[str] = []
    length = 0
    counter = 0
    while length < chars:
        block = hashlib.sha256(f"q{index}:{counter}".encode()).hexdigest()
        blocks.append(block)
        length += len(block)
        counter += 1
    return "".join(blocks)[:chars]


def web_record(index: int, chars: int = 4000, int_score: int = 4) -> dict:
    """Запись FineWeb-Edu: несжимаемый текст, edu-оценка управляемая."""
    return {
        "text": incompressible(index, chars),
        "id": f"<urn:uuid:{index:032d}>",
        "url": f"http://example.invalid/{index}",
        "dump": "CC-MAIN-TEST",
        "language": "en",
        "int_score": int_score,
    }


TEST_BODY = "import pytest\n\n\ndef test_addition():\n    assert 1 + 1 == 2\n"


def code_record(index: int, *, tests: bool = True, language: str = "python",
                bytes_size: int = 2000, license: object = "mit") -> dict:
    """Запись codeparrot-clean: поле ``content``, тест-маркер по флагу."""
    head = "# module %d\n" % index
    body = TEST_BODY if tests else "def add(a, b):\n    return a + b\n"
    filler = "x = 1  # padding comment\n" * max(0, (bytes_size - len(head) - len(body)) // 25)
    return {
        "content": head + body + filler,
        # Поля языка у codeparrot-clean в записи нет (корпус одноязычный) —
        # фикстура его добавляет только там, где проверяется фильтр языка.
        "lang": language,
        "license": license,
        "size": bytes_size,
        "repo_name": f"owner/repo{index}",
        "path": f"tests/test_mod{index}.py",
        "hash": f"sha{index}",
    }


WEB_RECORDS = 400
CODE_RECORDS = 60
#: Из 400 веб-записей порог int_score >= 4 проходит половина (200), из 60
#: кодовых тест-маркер есть у половины (30) — этого хватает, чтобы доля кода
#: держалась у цели, а не упиралась в исчерпание источника.
WEB_KEPT = WEB_RECORDS // 2
CODE_KEPT = CODE_RECORDS // 2


@pytest.fixture()
def web_source(tmp_path: Path) -> dict:
    """Веб-источник: половина записей проходит сужёный порог int_score >= 4."""
    records = [
        web_record(i, int_score=4 if i % 2 == 0 else 3) for i in range(WEB_RECORDS)
    ]
    path = write_jsonl(tmp_path / "src" / "web.jsonl", records)
    return {"kind": "local", "glob": str(path)}


@pytest.fixture()
def code_source(tmp_path: Path) -> dict:
    """Кодовый источник: каждая вторая запись — с тестами."""
    records = [code_record(i, tests=(i % 2 == 0)) for i in range(CODE_RECORDS)]
    path = write_jsonl(tmp_path / "src" / "code.jsonl", records)
    return {"kind": "local", "glob": str(path)}


def read_shard(path: Path) -> list[dict]:
    """Прочитать jsonl-шард (zstd) — независимо от писателя."""
    import zstandard

    text = zstandard.ZstdDecompressor().decompress(
        path.read_bytes(), max_output_size=256 * MB
    )
    return [json.loads(line) for line in text.decode("utf-8").splitlines() if line]


def records_of(report: dict, out: Path) -> list[dict]:
    return [r for entry in report["shards"] for r in read_shard(out / entry["file"])]


def prepare(
    tmp_path: Path,
    web_source: dict,
    code_source: dict | None = None,
    *,
    prior: bool = False,
    target_tokens: int = 0,
    **kwargs,
) -> tuple[dict, Path]:
    out = kwargs.pop("out", tmp_path / "Q")
    report = fineweb.prepare_q(
        out_dir=out,
        target_tokens=target_tokens,
        web_spec=web_source,
        code_spec=code_source or {"kind": "local", "glob": str(tmp_path / "src" / "none.jsonl")},
        prior_enabled=prior,
        shard_bytes=64 * 1024,
        level=1,
        dedup_window=1000,
        manifest_path=out / "manifest-q.json",
        report_path=out / "report-q.json",
        **kwargs,
    )
    return report, out


# --------------------------------------------------------------------------- #
# T-q1: сужёный фильтр веб-части
# --------------------------------------------------------------------------- #


def test_t_q1_quality_filter_thresholds_and_length(tmp_path: Path) -> None:
    records = [
        web_record(0, chars=4000, int_score=5),   # верхняя полка — проходит
        web_record(1, chars=4000, int_score=4),   # порог — проходит
        web_record(2, chars=4000, int_score=3),   # ниже сужёного порога
        {"text": "short but scored well " * 5, "int_score": 5},   # короче min_chars
        {"text": incompressible(9, 120_000), "int_score": 5},     # длиннее max_chars
        {"text": "   \n ", "int_score": 5},                       # пустой текст
        {"text": incompressible(10, 4000)},                       # нет int_score
    ]
    path = write_jsonl(tmp_path / "src" / "web.jsonl", records)

    normalizer = fineweb.QualityDocuments()
    kept = [normalizer(record) for record in records]
    assert [parsed is not None for parsed in kept] == [True, True, False, False, False, False, False]
    dropped = normalizer.stats
    assert dropped["dropped_low_int_score"] == 2, "score 3 и отсутствие поля"
    assert dropped["dropped_short"] == 1
    assert dropped["dropped_oversized"] == 1
    assert dropped["dropped_empty_text"] == 1

    # Тот же отбор в прогоне: в шард идут только документы, прошедшие все ступени.
    report, out = prepare(
        tmp_path, {"kind": "local", "glob": str(path)}, prior=False, web_yield=1.0
    )
    assert report["totals"]["records"] == 2
    stored = records_of(report, out)
    assert {record["meta"]["int_score"] for record in stored} == {4, 5}
    assert all(record["meta"]["component"] == "web" for record in stored)
    assert all(
        fineweb.DEFAULT_Q_MIN_CHARS <= len(record["text"]) <= fineweb.DEFAULT_Q_MAX_CHARS
        for record in stored
    )
    web_rules = report["rules"]["components"]["web"]
    assert web_rules == {
        "source": "FineWeb-Edu (тот же источник, что шард W)",
        "min_int_score": fineweb.DEFAULT_Q_MIN_INT_SCORE,
        "min_chars": fineweb.DEFAULT_Q_MIN_CHARS,
        "max_chars": fineweb.DEFAULT_Q_MAX_CHARS,
    }


def test_t_q1_w_shard_filter_is_untouched(tmp_path: Path) -> None:
    """Шард W сохраняет прежнее поведение: порог выключен, длина не ограничена снизу."""
    records = [web_record(0, chars=300, int_score=3), web_record(1, chars=5000, int_score=2)]
    path = write_jsonl(tmp_path / "src" / "w.jsonl", records)
    out = tmp_path / "W"
    report = fineweb.prepare_w(
        out_dir=out,
        target_tokens=0,
        source_spec={"kind": "local", "glob": str(path)},
        shard_bytes=64 * 1024,
        level=1,
        manifest_path=out / "manifest-w.json",
        report_path=out / "report-w.json",
    )
    assert report["totals"]["records"] == 2, "шард W не сужается фильтрами Q"
    assert "dropped_short" not in report["counters"]["dropped_by_rule"]
    assert report["counters"]["dropped_by_rule"]["dropped_low_int_score"] == 0


# --------------------------------------------------------------------------- #
# T-q2: кодовые примеры с тестами
# --------------------------------------------------------------------------- #


def test_t_q2_only_code_with_tests_gets_in(tmp_path: Path) -> None:
    normalizer = fineweb.TestCodeDocuments(source_name="codeparrot-clean")
    assert normalizer(code_record(0, tests=True)) is not None
    marker_meta = normalizer(code_record(0, tests=True))[1]
    assert marker_meta["tests"] == "def test_,pytest"
    assert marker_meta["lang"] == "python"
    assert "component" not in marker_meta, "компонент ставит нормализатор микса, не фильтр"

    for marker in ("unittest", "pytest", "def test_"):
        assert fineweb.test_markers(f"x = 1\n# {marker}\n") == [marker]
    assert fineweb.test_markers("def add(a, b):\n    return a + b\n") == []

    assert normalizer(code_record(1, tests=False)) is None
    assert normalizer.stats["dropped_no_tests"] == 1
    # Штатные фильтры шарда C продолжают действовать на кодовой части.
    assert normalizer(code_record(2, tests=True, bytes_size=40)) is None
    assert normalizer.stats["dropped_too_small"] == 1
    assert normalizer(code_record(3, tests=True, language="rust")) is None
    assert normalizer.stats["dropped_language"] == 1
    # Лицензионный фильтр у codeparrot-clean выключен политикой источника.
    assert normalizer(code_record(4, tests=True, license="gpl-3.0")) is not None
    assert normalizer.stats["dropped_license"] == 0
    assert normalizer.license_preselected is True


def test_t_q2_code_reaches_shard_with_markers(tmp_path: Path, web_source: dict,
                                               code_source: dict) -> None:
    report, out = prepare(tmp_path, web_source, code_source)
    stored = records_of(report, out)
    code = [record for record in stored if record["meta"]["component"] == "code"]
    assert code, "кодовая часть не дошла до шарда"
    for record in code:
        assert fineweb.test_markers(record["text"]) != []
        assert record["meta"]["tests"]
        assert record["meta"]["lang"] == "python"
        assert record["meta"]["license"] == "mit"
    dropped = report["counters"]["dropped_by_rule"]
    assert dropped["code:dropped_no_tests"] == 30, "каждая вторая запись — без тестов"


# --------------------------------------------------------------------------- #
# T-q3: доля кода в миксе
# --------------------------------------------------------------------------- #


def test_t_q3_mix_share_matches_shard_contents(tmp_path: Path, web_source: dict,
                                               code_source: dict) -> None:
    # Цель обрывает поток раньше исчерпания источников: иначе, как на реальном
    # прогоне, конец потока задавала бы доступность источников, а не микс.
    # Заявленные урожайности = 1 токен на символ: у синтетических записей обеих
    # частей выход на символ одинаков, поэтому доля по входу равна доле по выходу.
    report, out = prepare(
        tmp_path, web_source, code_source, code_share=0.15, web_yield=1.0, code_yield=1.0,
        target_tokens=40_000,
    )
    stored = records_of(report, out)
    web_tokens = sum(common.approx_tokens(r["text"]) for r in stored if r["meta"]["component"] == "web")
    code_tokens = sum(common.approx_tokens(r["text"]) for r in stored if r["meta"]["component"] == "code")
    share = code_tokens / (web_tokens + code_tokens)

    assert code_tokens and web_tokens, "обе части микса должны присутствовать"
    assert abs(share - 0.15) <= 0.03, f"доля кода ушла от цели: {share:.3f}"
    mix = report["mix"]
    assert mix["code_share_target"] == 0.15
    assert abs(mix["code_share"] - share) <= 0.01, "отчёт и содержимое шардов разошлись"
    assert mix["code"]["tokens"] == code_tokens and mix["web"]["tokens"] == web_tokens
    assert mix["code_records_filtered"] >= mix["code"]["records"]
    assert mix["source_exhausted"] == {"web": False, "code": False}
    assert set(mix["measured_yield"]) == {"web", "code"}
    assert "до точного дедупа" in mix["basis"]
    # Фактические урожайности совпали с заявленными — рекомендация не сдвигает
    # ручку без причины (при отклонении она вывела бы её на цель).
    assert mix["measured_yield"]["web"] == pytest.approx(0.25, abs=0.02)
    assert mix["recommended_code_yield"] == pytest.approx(1.0, rel=0.15)


def test_t_q3_governor_is_position_pure() -> None:
    """Решение о источнике зависит от позиций в потоках, а не от результатов фильтров."""
    governor = fineweb.MixGovernor(
        code_share=0.15, web_yield=fineweb.DEFAULT_WEB_YIELD, code_yield=fineweb.DEFAULT_CODE_YIELD
    )
    assert governor.pull_code() is True, "на пустом потоке первым идёт код"
    pulls = []
    for _ in range(12):
        component = "code" if governor.pull_code() else "web"
        pulls.append(component)
        governor.note(component, 4000)
    # После каждой кодовой записи идёт серия веб-записей: доля держится у цели.
    assert pulls[0] == "code"
    assert pulls.count("code") <= 3
    assert (governor.expected_code_tokens / (governor.expected_web_tokens + governor.expected_code_tokens)) < 0.3
    assert governor.state()["pulls"] == {"web": pulls.count("web"), "code": pulls.count("code")}


def test_t_q3_code_exhaustion_is_reported(tmp_path: Path, web_source: dict) -> None:
    """Кодовый источник кончился — доля ниже цели, и это видно, а не спрятано."""
    one_code = write_jsonl(tmp_path / "src" / "code-one.jsonl", [code_record(0, tests=True)])
    report, _ = prepare(
        tmp_path, web_source, {"kind": "local", "glob": str(one_code)},
        code_share=0.15, web_yield=1.0, code_yield=1.0,
    )
    assert report["mix"]["source_exhausted"]["code"] is True
    assert report["mix"]["recommended_code_yield"] is None
    assert "исчерпан" in report["mix"]["recommendation_note"]


# --------------------------------------------------------------------------- #
# T-q4: дедуп против записей W/C
# --------------------------------------------------------------------------- #


def reference_shard(tmp_path: Path, texts: list[str]) -> Path:
    """Опорный шард в формате W/C: те же поля, что пишет пайплайн."""
    out = tmp_path / "prior" / "W"
    records = [{"text": text, "meta": {"source": "prior", "id": str(index)}}
               for index, text in enumerate(texts)]
    return write_jsonl(out / "W-00000.jsonl", records)


def test_t_q4_prior_near_dup_drops_copies(tmp_path: Path, web_source: dict) -> None:
    base = incompressible(4242, 6000)
    near = base + incompressible(999, 300)  # 95 % общего текста — near-dup
    reference = reference_shard(tmp_path, [base])

    web = write_jsonl(
        tmp_path / "src" / "web-q.jsonl",
        [
            {"text": base, "int_score": 5, "id": "exact-copy"},
            {"text": near, "int_score": 5, "id": "near-copy"},
            {"text": incompressible(7, 6000), "int_score": 5, "id": "fresh"},
        ],
    )
    report, out = prepare(
        tmp_path, {"kind": "local", "glob": str(web)}, prior=True,
        prior_dirs=[tmp_path / "prior"], web_yield=1.0,
    )

    prior = report["prior"]
    assert prior["enabled"] is True
    assert prior["reference_records"] == 1
    assert prior["files_read"] == 1
    assert prior["dropped_exact"] == 1, "точная копия опорной записи"
    assert prior["dropped_near"] == 1, "near-дубль (Jaccard выше порога)"
    assert prior["checked"] == 3
    assert report["counters"]["dropped_by_rule"]["web:dropped_prior_duplicate"] == 2

    stored = records_of(report, out)
    assert [record["meta"]["id"] for record in stored] == ["fresh"]
    assert report["rules"]["prior_dedup"]["enabled"] is True


def test_t_q4_prior_dedup_requires_reference(tmp_path: Path, web_source: dict) -> None:
    """Нет опорных шардов — прогон не стартует: тихой подмены «точным хешем» нет."""
    with pytest.raises(FileNotFoundError, match="опорные шарды W/C не найдены"):
        fineweb.prepare_q(
            out_dir=tmp_path / "Q",
            target_tokens=0,
            web_spec=web_source,
            code_spec={"kind": "local", "glob": str(tmp_path / "nope.jsonl")},
            prior_dirs=[tmp_path / "empty"],
            manifest_path=tmp_path / "Q" / "manifest-q.json",
            report_path=tmp_path / "Q" / "report-q.json",
            level=1,
        )

    # Явное отключение фиксируется в отчёте, а не молчит.
    report, _ = prepare(tmp_path, web_source, prior=False)
    assert report["prior"] == {"enabled": False}
    assert report["rules"]["prior_dedup"]["enabled"] is False


def test_t_q4_prior_index_is_bounded(tmp_path: Path) -> None:
    """Опора ограничена окном записей: индекс не растёт со всем корпусом."""
    references = [incompressible(index, 900) for index in range(5)]
    shard = reference_shard(tmp_path, references)
    index = fineweb.PriorNearDupIndex(max_records=3)
    state = fineweb.load_prior_reference(index, [shard])
    assert index.reference_records == 3, "индекс взял ровно окно"
    assert state["truncated"] is True
    assert state["records"] == 3
    assert len(state["files"]) == 1
    # Проверяемый документ в индекс не добавляется: проверка сколько угодно раз.
    fresh = incompressible(77, 900)
    assert index.duplicate(fresh) is False
    assert index.duplicate(fresh) is False
    assert index.reference_records == 3

    with pytest.raises(ValueError):
        fineweb.PriorNearDupIndex(max_records=0)
    tiny = fineweb.PriorNearDupIndex(max_records=1)
    tiny.index(incompressible(1, 900))
    with pytest.raises(ValueError, match="полон"):
        tiny.index(incompressible(2, 900))


# --------------------------------------------------------------------------- #
# T-q5: resume и защита курсора
# --------------------------------------------------------------------------- #


def test_t_q5_resume_continues_without_loss(tmp_path: Path, web_source: dict,
                                            code_source: dict) -> None:
    out = tmp_path / "Q"
    common_kwargs = dict(
        out_dir=out,
        target_tokens=0,
        web_spec=web_source,
        code_spec=code_source,
        prior_enabled=False,
        shard_bytes=64 * 1024,
        level=1,
        manifest_path=out / "manifest-q.json",
        report_path=out / "report-q.json",
    )
    first = fineweb.prepare_q(max_records=20, **common_kwargs)
    assert first["stop_reason"] == "max_records"
    assert first["totals"]["records"] > 0
    consumed = first["source_records_consumed"]
    first_texts = [record["text"] for record in records_of(first, out)]

    second = fineweb.prepare_q(**common_kwargs)
    assert second["source_records_skipped_on_resume"] == consumed
    assert second["stop_reason"] == "source_exhausted"
    stored = records_of(second, out)
    texts = [record["text"] for record in stored]
    assert len(texts) == len(set(texts)), "дубли через границу resume"
    assert set(first_texts) <= set(texts), "шарды первого прогона потерялись"
    # Оба прогона читали один и тот же источник: после resume записи не теряются.
    assert second["totals"]["records"] == WEB_KEPT + CODE_KEPT, (
        "ожидались все веб-записи выше порога и все код-записи с тестами"
    )
    assert common.verify_manifest(
        json.loads((out / "manifest-q.json").read_text(encoding="utf-8")), out
    )["bad"] == []
    assert second["mix"]["code_share"] > 0

    # Повторный прогон ничего не дописывает: курсор стоит на исчерпании.
    again = fineweb.prepare_q(**common_kwargs)
    assert again["totals"]["records"] == second["totals"]["records"]


def test_t_q5_foreign_stream_in_same_manifest_is_refused(
    tmp_path: Path, web_source: dict, code_source: dict
) -> None:
    out = tmp_path / "Q"
    common_kwargs = dict(
        out_dir=out,
        target_tokens=0,
        prior_enabled=False,
        shard_bytes=64 * 1024,
        level=1,
        manifest_path=out / "manifest-q.json",
        report_path=out / "report-q.json",
    )
    fineweb.prepare_q(web_spec=web_source, code_spec=code_source, **common_kwargs)
    with pytest.raises(ValueError, match="другим потоком Q"):
        fineweb.prepare_q(
            web_spec=web_source, code_spec=code_source, code_share=0.4, **common_kwargs
        )
    # Явный --restart снимает отказ: прежние шарды не нужны.
    restarted = fineweb.prepare_q(
        web_spec=web_source, code_spec=code_source, code_share=0.4,
        restart=True, **common_kwargs,
    )
    assert restarted["source_records_skipped_on_resume"] == 0
    assert restarted["mix"]["code_share_target"] == 0.4


def test_t_q5_restart_rebuilds_shards(tmp_path: Path, web_source: dict) -> None:
    out = tmp_path / "Q"
    kwargs = dict(
        out_dir=out,
        target_tokens=0,
        web_spec=web_source,
        code_spec={"kind": "local", "glob": str(tmp_path / "src" / "code.jsonl")},
        prior_enabled=False,
        shard_bytes=64 * 1024,
        level=1,
        manifest_path=out / "manifest-q.json",
        report_path=out / "report-q.json",
    )
    fineweb.prepare_q(max_records=10, **kwargs)
    report = fineweb.prepare_q(restart=True, **kwargs)
    on_disk = sorted(path.name for path in out.glob("Q-*.jsonl.zst"))
    assert on_disk == sorted(entry["file"] for entry in report["shards"])
    assert report["source_records_skipped_on_resume"] == 0
