"""T-p1..T-d1 — пайплайн подготовки претрейн-датасета L3 (ADR-021).

Проверяются свойства, от которых зависит боевая загрузка 17B/3B:

* **T-p1** — ротация шард-файлов по размеру сжатого вывода, sha256 каждого шарда
  в манифесте совпадает с хешем файла на диске;
* **T-p2** — resume: после прерывания прогон продолжается с места, ни один
  документ не теряется и не дублируется;
* **T-p3** — фильтры шарда C: язык, лицензия (белый список), длина файла
  (микробные выбрасываются, длинные усекаются);
* **T-p4** — approx-счётчик `max(1, len(text)//4)` и его сумма в манифесте;
* **T-d1** — страховочный документный дедуп с ограниченным окном (LRU);
* **T-c-source / T-c-lic / T-c-resume** — источник ``codeparrot-clean`` (план B
  ADR-021): поле ``content`` из streaming-потока доходит до шарда, лицензионный
  фильтр к нему не применяется, чужой курсор манифеста не наследуется;
* плюс границы: запрет вывода вне gb10-shared/tmp и снятие socks-прокси.

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
    """Псевдослучайный (но детерминированный) текст: sha256-блоки подряд.

    Нужен, чтобы размер сжатого шарда был предсказуем — на повторах zstd
    ужимает текст в разы и ротация по размеру не проверяется.
    """
    blocks: list[str] = []
    length = 0
    counter = 0
    while length < chars:
        block = hashlib.sha256(f"{index}:{counter}".encode()).hexdigest()
        blocks.append(block)
        length += len(block)
        counter += 1
    return "".join(blocks)[:chars]


def web_record(index: int, chars: int = 400) -> dict:
    """Запись FineWeb-подобного источника: тексты несжимаемые и уникальные."""
    body = incompressible(index, chars)
    return {
        "text": body,
        "id": f"<urn:uuid:{index:032d}>",
        "url": f"http://example.invalid/{index}",
        "dump": "CC-MAIN-TEST",
        "language": "en",
        "int_score": 3,
        "score": 2.5,
    }


def code_record(
    index: int,
    language: str = "python",
    license: object = "mit",
    bytes_size: int = 600,
    text: str | None = None,
) -> dict:
    return {
        "content": text if text is not None else f"# file {index}\n" + "x = 1\n" * (bytes_size // 6),
        "lang": language,
        "license": license,
        "repo_name": f"owner/repo{index}",
        "path": f"src/mod{index}.py",
        "size": bytes_size,
    }


@pytest.fixture()
def web_source(tmp_path: Path) -> dict:
    records = [web_record(i) for i in range(120)]
    path = write_jsonl(tmp_path / "src" / "web.jsonl", records)
    return {"kind": "local", "glob": str(path)}


@pytest.fixture()
def source_spec(web_source: dict) -> dict:
    return web_source


def read_shard(path: Path) -> list[dict]:
    """Прочитать jsonl-шард (zstd) — независимо от писателя."""
    import zstandard

    raw = path.read_bytes()
    text = zstandard.ZstdDecompressor().decompress(raw, max_output_size=256 * MB)
    return [json.loads(line) for line in text.decode("utf-8").splitlines() if line]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


# --------------------------------------------------------------------------- #
# T-p1: ротация шардов и sha256 в манифесте
# --------------------------------------------------------------------------- #


def test_t_p1_shard_rotation_and_manifest_hashes(tmp_path: Path, source_spec: dict) -> None:
    out = tmp_path / "W"
    shard_bytes = 8 * 1024
    report = fineweb.prepare_w(
        out_dir=out,
        target_tokens=0,
        source_spec=source_spec,
        shard_bytes=shard_bytes,
        codec="zstd",
        level=1,
        flush_every=1024,
        dedup_window=10_000,
        manifest_path=out / "manifest-w.json",
        report_path=out / "report-w.json",
    )

    shards = report["shards"]
    assert len(shards) >= 3, f"ротация не сработала: шардов {len(shards)}"
    assert report["stop_reason"] == "source_exhausted"
    assert report["totals"]["records"] == 120

    files = sorted(p.name for p in out.glob("W-*.jsonl.zst"))
    assert files == sorted(entry["file"] for entry in shards), "манифест и файлы разошлись"

    for entry in shards:
        path = out / entry["file"]
        assert path.exists()
        assert entry["bytes"] == path.stat().st_size, "bytes в манифесте — не размер файла"
        assert entry["sha256"] == sha256_file(path), "sha256 в манифесте — не хеш файла"
        assert entry["approx_tokens"] > 0 and entry["records"] > 0
        # Ротация идёт по размеру сжатого вывода: все шарды кроме последнего —
        # не меньше порога (последний закрыт остатком источника).
    for entry in shards[:-1]:
        assert entry["bytes"] >= shard_bytes, f"шард {entry['file']} меньше порога ротации"

    # Ни одного «.part» и ни одного файла вне манифеста.
    assert not list(out.glob("*.part"))

    # Числа отчёта сходятся с файлами.
    total_records = sum(len(read_shard(out / entry["file"])) for entry in shards)
    assert total_records == report["totals"]["records"] == 120
    assert report["totals"]["bytes"] == sum(e["bytes"] for e in shards)
    assert report["totals"]["approx_tokens"] == sum(
        common.approx_tokens(record["text"])
        for entry in shards
        for record in read_shard(out / entry["file"])
    )

    # Манифест на диске — валидный JSON и совпадает с отчётом.
    manifest = json.loads((out / "manifest-w.json").read_text(encoding="utf-8"))
    assert len(manifest["shards"]) == len(shards)
    assert manifest["totals"]["records"] == 120

    # verify_manifest подтверждает хеши пересчётом с диска.
    verification = common.verify_manifest(manifest, out)
    assert verification["bad"] == []
    assert verification["ok"] == len(shards)


def test_t_p1_gzip_codec_is_supported(tmp_path: Path, source_spec: dict) -> None:
    out = tmp_path / "W-gz"
    report = fineweb.prepare_w(
        out_dir=out,
        target_tokens=0,
        source_spec=source_spec,
        shard_bytes=4 * 1024,
        codec="gzip",
        level=1,
        manifest_path=out / "manifest-w.json",
        report_path=out / "report-w.json",
    )
    assert report["shards"]
    assert all(entry["file"].endswith(".jsonl.gz") for entry in report["shards"])
    import gzip

    first = out / report["shards"][0]["file"]
    with gzip.open(first, "rt", encoding="utf-8") as handle:
        assert json.loads(handle.readline())["text"]
    assert first.stat().st_size == report["shards"][0]["bytes"]
    assert sha256_file(first) == report["shards"][0]["sha256"]


# --------------------------------------------------------------------------- #
# T-p2: resume после прерывания
# --------------------------------------------------------------------------- #


class CrashAfter:
    """Нормализатор-обёртка: падает на N-й записи, имитируя обрыв прогона."""

    def __init__(self, inner, crash_at: int) -> None:
        self.inner = inner
        self.crash_at = crash_at
        self.seen = 0
        self.stats: dict[str, int] = {}

    def __call__(self, record: dict):
        self.seen += 1
        if self.seen >= self.crash_at:
            raise RuntimeError("имитация обрыва прогона")
        return self.inner(record)


def test_t_p2_resume_continues_after_crash(tmp_path: Path, source_spec: dict) -> None:
    out = tmp_path / "W"
    shard_bytes = 4 * 1024
    common_kwargs = dict(
        shard="W",
        out_dir=out,
        target_tokens=0,
        source_spec=source_spec,
        manifest_path=out / "manifest-w.json",
        report_path=out / "report-w.json",
        shard_bytes=shard_bytes,
        flush_every=1024,
        level=1,
    )

    crashing = CrashAfter(fineweb.FinewebDocuments(), crash_at=40)
    with pytest.raises(RuntimeError, match="имитация обрыва"):
        common.run_shard(normalize=crashing, **common_kwargs)

    manifest_path = out / "manifest-w.json"
    first_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    # Прерванный шард дописан и учтён, «полу-файлов» не осталось.
    assert not list(out.glob("*.part"))
    assert first_manifest["shards"], "после обрыва манифест пуст — resume невозможен"
    for entry in first_manifest["shards"]:
        assert (out / entry["file"]).exists()
        assert entry["sha256"] == sha256_file(out / entry["file"])
    consumed_before = first_manifest["source_records"]
    assert 0 < consumed_before <= 40
    records_before = first_manifest["totals"]["records"]

    # Продолжение с места: первый прогон не повторяется.
    resuming = fineweb.FinewebDocuments()
    report = common.run_shard(normalize=resuming, **common_kwargs)

    assert report["source_records_skipped_on_resume"] == consumed_before
    assert resuming.stats["dropped_empty_text"] == 0
    assert report["totals"]["records"] == 120, "после resume потеряны или размножены записи"

    texts = [
        record["text"]
        for entry in report["shards"]
        for record in read_shard(out / entry["file"])
    ]
    assert len(texts) == len(set(texts)) == 120, "дубли через границу resume"
    expected = {web_record(i)["text"] for i in range(120)}
    assert set(texts) == expected, "часть документов потеряна при resume"
    assert report["totals"]["records"] > records_before

    # Манифест после resume консистентен, повторный прогон ничего не делает.
    final = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert common.verify_manifest(final, out)["bad"] == []
    again = common.run_shard(normalize=fineweb.FinewebDocuments(), **common_kwargs)
    assert again["totals"]["records"] == 120, "повторный запуск дописал лишнее"
    assert again["stop_reason"] == "source_exhausted"


def test_t_p2_restart_starts_over(tmp_path: Path, source_spec: dict) -> None:
    out = tmp_path / "W"
    kwargs = dict(
        shard="W",
        out_dir=out,
        target_tokens=0,
        source_spec=source_spec,
        manifest_path=out / "manifest-w.json",
        report_path=out / "report-w.json",
        shard_bytes=8 * 1024,
        level=1,
    )
    common.run_shard(normalize=fineweb.FinewebDocuments(), max_records=30, **kwargs)
    stopped = json.loads((out / "manifest-w.json").read_text(encoding="utf-8"))
    assert stopped["stop_reason"] == "max_records"

    report = common.run_shard(normalize=fineweb.FinewebDocuments(), restart=True, **kwargs)
    assert report["source_records_skipped_on_resume"] == 0
    assert report["totals"]["records"] == 120
    # Файлы от прежнего прогона перезаписаны, лишних шардов не осталось.
    on_disk = sorted(p.name for p in out.glob("W-*.jsonl.zst"))
    assert on_disk == sorted(entry["file"] for entry in report["shards"])


# --------------------------------------------------------------------------- #
# T-p3: фильтры шарда C (язык / лицензия / длина)
# --------------------------------------------------------------------------- #


@pytest.fixture()
def code_source(tmp_path: Path) -> dict:
    big_text = "y = 2\n" * 30_000  # ~180 КБ — больше верхней границы
    records = [
        code_record(0, "python", "mit", 600),
        code_record(1, "rust", ["apache-2.0"], 600),
        code_record(2, "go", "BSD-3-Clause", 600),
        code_record(3, "javascript", "isc", 600),
        code_record(4, "shell", "0bsd", 600),
        code_record(5, "python", "gpl-3.0", 600),            # лицензия вне белого списка
        code_record(6, "python", ["mit", "gpl-3.0"], 600),   # смесь — консервативно мимо
        code_record(7, "python", None, 600),                 # лицензия не указана
        code_record(8, "c", "mit", 600),                     # язык вне списка
        code_record(9, "python", "mit", 40),                 # микробный (< 256 Б)
        code_record(10, "rust", "mit", text=big_text),       # длинный — усекается
        code_record(11, "go", "mit", text="   \n  "),        # пустой текст
    ]
    path = write_jsonl(tmp_path / "src" / "code.jsonl", records)
    return {"kind": "local", "glob": str(path)}


def test_t_p3_language_license_length_filters(tmp_path: Path, code_source: dict) -> None:
    out = tmp_path / "C"
    report = stack.prepare_c(
        out_dir=out,
        target_tokens=0,
        source_spec=code_source,
        shard_bytes=64 * 1024,
        level=1,
        dedup_window=1000,
        manifest_path=out / "manifest-c.json",
        report_path=out / "report-c.json",
    )
    rules = report["rules"]
    assert rules["languages"] == list(stack.LANGUAGES)
    assert set(rules["allowed_licenses"]) == {
        "mit", "apache-2.0", "bsd-3-clause", "bsd-2-clause", "isc", "0bsd",
    }

    dropped = report["counters"]["dropped_by_rule"]
    assert dropped["dropped_license"] == 3, "GPL, смесь mit+gpl и отсутствие лицензии"
    assert dropped["dropped_language"] == 1, "файл на языке вне объявленных"
    assert dropped["dropped_too_small"] == 1, "микробный файл (< 256 Б)"
    assert dropped["dropped_empty_text"] == 1, "пустой текст"
    assert dropped["truncated_oversized"] == 1, "длинный файл усечён, а не выброшен"

    assert report["totals"]["records"] == 6
    by_lang = report["rules"]["kept_by_language"]
    assert by_lang == {"python": 1, "rust": 2, "go": 1, "javascript": 1, "shell": 1}
    assert sorted(by_lang) == sorted(stack.LANGUAGES), "каждый объявленный язык представлен"

    stored = [record for entry in report["shards"] for record in read_shard(out / entry["file"])]
    for record in stored:
        size = len(record["text"].encode("utf-8"))
        assert stack.MIN_FILE_BYTES <= size <= stack.MAX_FILE_BYTES
        assert record["meta"]["lang"] in stack.LANGUAGES
        assert set(record["meta"]["license"].split(",")) <= stack.ALLOWED_LICENSES
        assert record["meta"]["source"]
    truncated = [r for r in stored if r["meta"]["lang"] == "rust" and len(r["text"]) > 10_000]
    assert len(truncated) == 1
    assert len(truncated[0]["text"].encode("utf-8")) == stack.MAX_FILE_BYTES


def test_t_p3_license_normalisation() -> None:
    assert stack.normalize_licenses("MIT") == ("mit",)
    assert stack.normalize_licenses(["MIT", "Apache-2.0"]) == ("mit", "apache-2.0")
    assert stack.normalize_licenses("['BSD-3-Clause']") == ("bsd-3-clause",)
    assert stack.normalize_licenses(None) == ()
    assert stack.normalize_licenses("") == ()

    assert stack.licenses_allowed("mit")
    assert stack.licenses_allowed(["apache-2.0", "bsd-2-clause"])
    assert stack.licenses_allowed("bsd-3-clause-clear") is False
    assert stack.licenses_allowed("gpl-3.0") is False
    assert stack.licenses_allowed(["mit", "gpl-3.0"]) is False, "смесь лицензий — мимо"
    assert stack.licenses_allowed(None) is False


def test_t_p3_nested_metadata_and_assuming_single_language_source() -> None:
    """Поля бывают вложены в metadata, а у одноязычного корпуса языка в записи нет."""
    nested = stack.StackFiles(source_name="common-pile-stackv2")
    parsed = nested(
        {
            "text": "z" * 400,
            "metadata": {"license": "MIT", "language": "Python", "path": "a/b.py", "repo_name": "o/r"},
        }
    )
    assert parsed is not None, "поля внутри metadata должны читаться"
    text, meta = parsed
    assert meta["lang"] == "python" and meta["license"] == "mit"
    assert meta["path"] == "a/b.py" and meta["repo"] == "o/r"

    # Язык записи не объявлен, но корпус одноязычный по построению.
    assumed = stack.StackFiles(source_name="codeparrot-clean")
    parsed = assumed({"content": "y" * 400, "license": "apache-2.0", "size": 400})
    assert parsed is not None and parsed[1]["lang"] == "python"

    # Без поля языка и без свойства источника запись отбрасывается, а не угадывается.
    unknown = stack.StackFiles(source_name="stack-dedup-v1")
    assert unknown({"content": "q" * 400, "license": "mit"}) is None
    assert unknown.stats["dropped_language"] == 1


def test_t_p3_source_registry_declares_stack_priority() -> None:
    """Реестр источников: приоритет ADR-021 и честная пометка гейтов."""
    assert stack.AUTO_ORDER[0] == "stack-v2-dedup"
    assert stack.AUTO_ORDER[1] == "stack-dedup-v1"
    assert stack.SOURCES["stack-v2-dedup"].gated is True
    assert stack.SOURCES["stack-dedup-v1"].gated is True
    assert stack.DEFAULT_SOURCE_NAME == "stack-dedup-v1"
    # Языковые конфиги v2 — ровно те, что нужны шарду C.
    assert stack.SOURCES["stack-v2-dedup"].config_for("python") == "Python"
    assert stack.SOURCES["stack-v2-dedup"].config_for("shell") == "Shell"


# --------------------------------------------------------------------------- #
# T-c: источник codeparrot-clean (план B ADR-021, python-only)
# --------------------------------------------------------------------------- #

CODEPARROT_REPO = "codeparrot/codeparrot-clean"
CODEPARROT_NOTE = "python-only fallback (ADR-021 план B)"


def codeparrot_record(index: int, *, license: object = "mit", bytes_size: int = 600) -> dict:
    """Запись codeparrot-clean: поле ``content``, поля языка НЕТ (корпус python)."""
    return {
        "content": f"# file {index}\n" + "x = 1\n" * (bytes_size // 6),
        "license": license,
        "size": bytes_size,
        "repo_name": f"owner/repo{index}",
        "path": f"src/mod{index}.py",
        "hash": f"sha{index}",
    }


class FakeHfStream:
    """Синтетический streaming-источник: тот же путь кода, что у HF-потока.

    Подменяет ``common.iter_hf_stream`` — точка, из которой ``stack`` берёт
    настоящий ``datasets``-поток (сеть в тестах не нужна, вызов виден).
    """

    def __init__(self, records: list[dict]) -> None:
        self.records = records
        self.calls: list[dict] = []

    def __call__(self, repo, config=None, split="train", data_files=None, **_):
        self.calls.append(
            {"repo": repo, "config": config, "split": split, "data_files": data_files}
        )
        return iter([dict(record) for record in self.records])

    @property
    def last_call(self) -> dict:
        assert self.calls, "поток источника не запрашивался"
        return self.calls[-1]


def test_t_c_source_streams_content_into_shards(tmp_path: Path, monkeypatch) -> None:
    """T-c-source: streaming-датасет с полем ``content`` → записи шарда C.

    Проверяется весь путь: реестр → ``iter_stack_documents`` → HF-поток →
    фильтры → jsonl-шард + манифест с оговоркой источника.
    """
    records = [codeparrot_record(i) for i in range(30)]
    stream = FakeHfStream(records)
    monkeypatch.setattr(common, "iter_hf_stream", stream)

    out = tmp_path / "C-cp"
    report = stack.prepare_c(
        out_dir=out,
        target_tokens=0,
        source_name="codeparrot-clean",
        shard_bytes=64 * 1024,
        level=1,
        dedup_window=1000,
        manifest_path=out / "manifest-c.json",
        report_path=out / "report-c.json",
    )

    # Поток запрошен у правильного датасета: конфиг default, split train.
    assert stream.last_call["repo"] == CODEPARROT_REPO
    assert stream.last_call["config"] is None, "конфиг default"
    assert stream.last_call["split"] == "train"

    # Поле content → text: записи дошли до шарда, ничего не потеряно.
    assert report["totals"]["records"] == len(records)
    assert report["stop_reason"] == "source_exhausted"
    stored = [
        record for entry in report["shards"] for record in read_shard(out / entry["file"])
    ]
    assert len(stored) == len(records)
    assert all(record["text"].startswith("# file ") for record in stored)
    assert {record["meta"]["lang"] for record in stored} == {"python"}
    assert all(record["meta"]["source"] == CODEPARROT_REPO for record in stored)
    assert stored[0]["meta"]["path"] == "src/mod0.py"
    assert stored[0]["meta"]["repo"] == "owner/repo0"
    assert report["counters"]["dropped_by_rule"]["dropped_empty_text"] == 0

    # Манифест: имя источника, python-only и оговорка плана B.
    manifest = json.loads((out / "manifest-c.json").read_text(encoding="utf-8"))
    assert manifest["source"]["name"] == "codeparrot-clean"
    assert manifest["source"]["note"] == CODEPARROT_NOTE
    assert manifest["source"]["languages"] == ["python"]
    assert manifest["source"]["license_filter"] == "preselected"
    assert report["source"]["note"] == CODEPARROT_NOTE
    assert report["rules"]["languages"] == ["python"], "языки сужены до фактических"
    assert report["rules"]["license_filter"] == "preselected"
    assert report["rules"]["kept_by_language"] == {"python": len(records)}
    # Фильтр выключен — значит распределение лицензий обязано быть измерено
    # и лежать в отчёте: корпус не «разрешён по умолчанию».
    assert report["rules"]["licenses_seen"] == {"mit": len(records)}
    assert common.verify_manifest(manifest, out)["bad"] == []


def test_t_c_source_resume_continues_same_source(tmp_path: Path, monkeypatch) -> None:
    """Resume того же источника работает как раньше (курсор не сбрасывается)."""
    stream = FakeHfStream([codeparrot_record(i) for i in range(30)])
    monkeypatch.setattr(common, "iter_hf_stream", stream)
    out = tmp_path / "C-cp"
    kwargs = dict(
        out_dir=out,
        target_tokens=0,
        source_name="codeparrot-clean",
        shard_bytes=64 * 1024,
        level=1,
        manifest_path=out / "manifest-c.json",
        report_path=out / "report-c.json",
    )
    first = stack.prepare_c(max_records=10, **kwargs)
    assert first["totals"]["records"] == 10
    assert first["stop_reason"] == "max_records"

    second = stack.prepare_c(**kwargs)
    assert "manifest_source_reset" not in second
    assert second["source_records_skipped_on_resume"] == first["source_records_consumed"]
    assert second["totals"]["records"] == 30, "тот же источник — потерь и дублей нет"


def test_t_c_lic_license_filter_skipped_for_codeparrot() -> None:
    """T-c-lic: у codeparrot-clean лицензионный фильтр не применяется (не поле).

    Датасет отфильтрован по лицензиям на своей стороне, ``license`` в записи —
    справка о происхождении. Белый список к нему не применяется: запись без
    поля лицензии и запись с лицензией вне списка остаются в корпусе.
    """
    preselected = stack.StackFiles(languages=("python",), source_name="codeparrot-clean")
    assert preselected.license_preselected is True
    assert preselected({"content": "y" * 400}) is not None, "нет поля license — не причина отбоя"
    assert preselected({"content": "y" * 400, "license": "gpl-3.0"}) is not None
    assert preselected.stats["dropped_license"] == 0, "фильтр выключен политикой источника"
    # Язык и длина при этом работают: выключен ровно один шаг.
    assert preselected({"content": "y" * 400, "lang": "rust"}) is None
    assert preselected.stats["dropped_language"] == 1
    assert preselected({"content": "y" * 10}) is None
    assert preselected.stats["dropped_too_small"] == 1

    # Контроль: у источника с белым списком те же записи отбраковываются.
    control = stack.StackFiles(languages=("python",), source_name="stack-dedup-v1")
    assert control.license_preselected is False
    assert control({"content": "y" * 400, "lang": "python"}) is None
    assert control({"content": "y" * 400, "lang": "python", "license": "gpl-3.0"}) is None
    assert control.stats["dropped_license"] == 2
    assert stack.SOURCES["codeparrot-clean"].license_preselected is True
    assert all(
        not spec.license_preselected
        for name, spec in stack.SOURCES.items()
        if name != "codeparrot-clean"
    ), "выключение фильтра — свойство одного источника, а не реестра"


def test_t_c_resume_foreign_cursor_is_not_inherited(tmp_path: Path, monkeypatch) -> None:
    """Курсор манифеста принадлежит прежнему источнику — наследовать нельзя.

    Пустой манифест (прошлый прогон не дал шардов) лечится сбросом курсора;
    непустой — отказ, потому что шарды разных источников смешивать нельзя.
    """
    out = tmp_path / "C"
    tiny = FakeHfStream([{"content": "x", "lang": "python", "license": "mit"}])
    monkeypatch.setattr(common, "iter_hf_stream", tiny)
    first = stack.prepare_c(
        out_dir=out,
        target_tokens=0,
        source_name="stack-dedup-v1",
        manifest_path=out / "manifest-c.json",
        report_path=out / "report-c.json",
        level=1,
    )
    assert first["totals"]["records"] == 0 and first["shards"] == []
    assert first["source_records_consumed"] > 0, "курсор чужого источника продвинулся"

    stream = FakeHfStream([codeparrot_record(i) for i in range(5)])
    monkeypatch.setattr(common, "iter_hf_stream", stream)
    kwargs = dict(
        out_dir=out,
        target_tokens=0,
        source_name="codeparrot-clean",
        manifest_path=out / "manifest-c.json",
        report_path=out / "report-c.json",
        level=1,
    )
    second = stack.prepare_c(**kwargs)
    reset = second["manifest_source_reset"]
    assert reset["previous_source"] == "stack-dedup-v1"
    assert reset["current_source"] == "codeparrot-clean"
    assert second["source_records_skipped_on_resume"] == 0, "чужой курсор не промотал поток"
    assert second["totals"]["records"] == 5, "живой датасет не «исчерпан» чужим курсором"

    # Шарды уже есть: смена источника — отказ, а не тихая мешанина.
    with pytest.raises(ValueError, match="разных источников"):
        stack.prepare_c(**{**kwargs, "source_name": "stack-smol-xl"})


# --------------------------------------------------------------------------- #
# T-p4: approx-счётчик
# --------------------------------------------------------------------------- #


def test_t_p4_approx_token_counter() -> None:
    assert common.approx_tokens("") == 1
    assert common.approx_tokens("abc") == 1
    assert common.approx_tokens("a" * 4) == 1
    assert common.approx_tokens("a" * 400) == 100
    assert common.approx_tokens("a" * 401) == 100


def test_t_p4_totals_match_sum_over_records(tmp_path: Path, source_spec: dict) -> None:
    out = tmp_path / "W"
    report = fineweb.prepare_w(
        out_dir=out,
        target_tokens=0,
        source_spec=source_spec,
        shard_bytes=16 * 1024,
        level=1,
        manifest_path=out / "manifest-w.json",
        report_path=out / "report-w.json",
    )
    expected = sum(common.approx_tokens(web_record(i)["text"]) for i in range(120))
    assert report["totals"]["approx_tokens"] == expected
    assert report["counters"]["approx_tokens"] == expected
    per_shard = sum(entry["approx_tokens"] for entry in report["shards"])
    assert per_shard == expected, "сумма по шардам расходится с итогом"


def test_t_p4_target_tokens_stops_run(tmp_path: Path, source_spec: dict) -> None:
    out = tmp_path / "W"
    record_tokens = common.approx_tokens(web_record(0)["text"])
    report = fineweb.prepare_w(
        out_dir=out,
        target_tokens=record_tokens * 10,
        source_spec=source_spec,
        shard_bytes=64 * 1024,
        level=1,
        manifest_path=out / "manifest-w.json",
        report_path=out / "report-w.json",
    )
    assert report["stop_reason"] == "target_tokens"
    assert report["totals"]["approx_tokens"] >= record_tokens * 10
    assert report["totals"]["approx_tokens"] < record_tokens * 11
    assert report["source_records_consumed"] == report["totals"]["records"]


# --------------------------------------------------------------------------- #
# T-d1: страховочный дедуп
# --------------------------------------------------------------------------- #


def test_t_d1_duplicate_documents_are_dropped(tmp_path: Path, source_spec: dict) -> None:
    out = tmp_path / "W"
    report = fineweb.prepare_w(
        out_dir=out,
        target_tokens=0,
        source_spec=source_spec,
        shard_bytes=32 * 1024,
        level=1,
        manifest_path=out / "manifest-w.json",
        report_path=out / "report-w.json",
    )
    assert report["counters"]["dropped_dedup"] == 0, "уникальный источник не должен терять записи"

    # Дубли в источнике (в т.ч. через границу шардов) отбрасываются.
    dup_path = write_jsonl(
        tmp_path / "src" / "dup.jsonl",
        [web_record(i) for i in range(40)] + [web_record(0), web_record(3), web_record(0)],
    )
    dup_out = tmp_path / "W-dup"
    report = fineweb.prepare_w(
        out_dir=dup_out,
        target_tokens=0,
        source_spec={"kind": "local", "glob": str(dup_path)},
        shard_bytes=8 * 1024,
        level=1,
        manifest_path=dup_out / "manifest-w.json",
        report_path=dup_out / "report-w.json",
        flush_every=1024,
    )
    assert report["counters"]["dropped_dedup"] == 3
    assert report["totals"]["records"] == 40
    texts = [r["text"] for e in report["shards"] for r in read_shard(dup_out / e["file"])]
    assert len(texts) == len(set(texts)) == 40
    assert report["dedup_window"] == common.DEFAULT_DEDUP_WINDOW


def test_t_d1_lru_window_evicts_old_hashes() -> None:
    window = common.BoundedHashSet(capacity=2)
    assert window.duplicate("a") is False
    assert window.duplicate("b") is False
    assert window.duplicate("a") is True, "свежий хеш должен помниться"
    assert window.duplicate("c") is False, "вытесняет 'b' (LRU)"
    assert window.size == 2
    assert window.duplicate("b") is False, "вытесненный хеш забыт — это граница окна"
    assert window.hits == 1, "hits считает только подтверждённые повторы"

    with pytest.raises(ValueError):
        common.BoundedHashSet(capacity=0)


def test_t_d1_code_shard_dedups_tool(tmp_path: Path, code_source: dict) -> None:
    """Дедуп работает и на шарде C: одинаковые файлы из разных репозиториев — один раз."""
    duplicated = code_source["glob"].replace("code.jsonl", "code-dup.jsonl")
    with open(code_source["glob"], encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle]
    twin = dict(records[0], repo_name="other/repo", path="lib/other.py")
    write_jsonl(Path(duplicated), records + [twin])
    out = tmp_path / "C-dup"
    report = stack.prepare_c(
        out_dir=out,
        target_tokens=0,
        source_spec={"kind": "local", "glob": duplicated},
        shard_bytes=64 * 1024,
        level=1,
        manifest_path=out / "manifest-c.json",
        report_path=out / "report-c.json",
    )
    assert report["counters"]["dropped_dedup"] == 1
    assert report["totals"]["records"] == 6


# --------------------------------------------------------------------------- #
# Границы: каталоги вывода и прокси
# --------------------------------------------------------------------------- #


def test_output_outside_allowed_roots_is_refused(tmp_path: Path) -> None:
    """Вывод вне gb10-shared/tmp отклоняется — сигнал отката (а) контракта."""
    home_dir = Path.home()
    with pytest.raises(ValueError, match="вне разрешённых корней"):
        common.ensure_output_allowed(home_dir / "axiom-pretrain-escape")
    assert common.ensure_output_allowed(tmp_path) == tmp_path.resolve()
    assert common.ensure_output_allowed(
        Path.home() / "gb10-shared" / "datasets" / "axiom-pretrain-l3"
    )


def test_source_specs_parse() -> None:
    assert common.parse_source_spec("hf:HuggingFaceFW/fineweb-edu:sample-100BT") == {
        "kind": "hf",
        "repo": "HuggingFaceFW/fineweb-edu",
        "config": "sample-100BT",
        "split": "train",
    }
    assert common.parse_source_spec("hf:bigcode/the-stack-dedup")["config"] is None
    assert common.parse_source_spec("local:/tmp/x/*.jsonl") == {
        "kind": "local",
        "glob": "/tmp/x/*.jsonl",
    }
    with pytest.raises(ValueError):
        common.parse_source_spec("s3://bucket/prefix")


def test_socks_proxy_env_is_stripped() -> None:
    """httpx не умеет socks-схему: ALL_PROXY=socks://… валит запрос до сети."""
    env = {
        "ALL_PROXY": "socks://127.0.0.1:7890/",
        "all_proxy": "socks5://127.0.0.1:7890/",
        "HTTPS_PROXY": "http://127.0.0.1:12080/",
        "HTTP_PROXY": "http://127.0.0.1:12080/",
    }
    changed = common.sanitize_proxy_env(env)
    assert sorted(changed) == ["ALL_PROXY", "all_proxy"]
    assert "ALL_PROXY" not in env and "all_proxy" not in env
    assert env["HTTPS_PROXY"] == "http://127.0.0.1:12080/", "рабочие http-прокси не трогаем"
