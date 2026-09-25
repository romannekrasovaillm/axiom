"""Стадия `sft` конвейера A4 (ADR-005 п. 1) — приёмочные тесты дельты.

Источник истины — `docs/specs/SFT-STAGE.delta.md` §4, а не реализация
(`tools/run_sft_smoke.py`). Сценарии:

1. **Симлинк, а не копия** (C-032/C-033): путь-копия в рабочем каталоге —
   отказ до всякого обучения; симлинк, ведущий вне канонического
   `~/gb10-shared`, — тоже отказ.
2. **Карточка датасета** (ADR-004): sha256 содержимого сходится с независимо
   посчитанным хешем, лицензия непуста и объявлена у каждого блока, форма
   читается `net.data.DatasetCard`.
3. **Журнал стадии без абсолютных путей** (ADR-014 п. 8, §4.5): после
   смоук-прогона ни одна строка артефакта не начинается с `/`; есть сид,
   число шагов, `loss_first/loss_last` с падением, `tree_hash`.
4. **`tree_hash` после round-trip** (§4.1): чекпойнт стадии восстанавливается
   побитово, хеш совпадает с записанным в журнале и с манифестом чекпойнта.
5. **Детерминизм при пиннутом сиде** (§4.1, AD-11): повторный прогон с тем же
   сидом даёт тот же первый лосс.

Смоук-прогоны тестов идут на CPU (`NET_JAX_BACKEND=cpu`) и в смоук-масштабе:
тест мерит провод стадии (данные → токенизация → шаг → чекпойнт → журнал), а
не железо; пиннинг бэкенда приёмки (ADR-010) от этого не зависит. Временные
каталоги создаются ВНУТРИ репозитория (фикстура `repo_scratch`): пути журнала
обязаны быть относительными от его корня, а `~/gb10-shared` подменяется
скретчем, чтобы тест не трогал канонический диск.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Iterator

import pytest

NET_DIR = Path(__file__).resolve().parent.parent
CASE_DIR = NET_DIR.parent
REPO_ROOT = CASE_DIR.parent.parent
RUNNER = CASE_DIR / "tools" / "run_sft_smoke.py"

#: Документы реального набора (симлинк на ~/gb10-shared) для смоук-окна теста.
SHARED_DATASET = Path.home() / "gb10-shared" / "datasets" / "sft_train_v12.jsonl"

_tmp_counter = [0]

#: Независимая проверка round-trip В ТОМ ЖЕ бэкенде, что и прогон стадии.
#: Orbax пишет топологию шардирования чекпойнта: восстановление чекпойнта,
#: снятого на CPU, на GPU-топологии падает («Device cpu:0 was not found in
#: jax.local_devices()»), а не возвращает те же значения. Поэтому проверка
#: идёт отдельным процессом с тем же выбором бэкенда, что у прогона; пиннинг
#: берётся из conftest приёмки (ADR-010: одна точка пиннинга).
_VERIFY_PROGRAM = """
import importlib.util, json, sys
spec = importlib.util.spec_from_file_location("rss_verify", sys.argv[1])
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
runner._load_acceptance_conftest()          # выбор бэкенда до импорта net.*
from net import checkpoint, model           # noqa: E402
import jax.random as jr                     # noqa: E402
cfg = runner.build_model_config(int(sys.argv[3]), sys.argv[4])
structure = model.init_params(jr.PRNGKey(0), cfg)   # та же структура, иные значения
restored = checkpoint.load_checkpoint(sys.argv[2], target=structure)
print(checkpoint.tree_hash(restored))
"""


def _load_runner():
    """Загрузить `tools/run_sft_smoke.py` по пути (модуль вне пакета net)."""
    spec = importlib.util.spec_from_file_location("run_sft_smoke_test", RUNNER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["run_sft_smoke_test"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def repo_scratch() -> Iterator[Path]:
    """Временный каталог ВНУТРИ репозитория, удаляется после теста.

    Журнал стадии несёт пути, относительные от корня репозитория (ADR-014
    п. 8), поэтому выход теста обязан лежать внутри репо; каталог evidence/
    кейса тесты не затрагивают.
    """
    _tmp_counter[0] += 1
    scratch = Path(tempfile.mkdtemp(
        prefix=f".sft-test-{_tmp_counter[0]}-", dir=str(NET_DIR / "tests")
    ))
    try:
        yield scratch
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def _mini_dataset(scratch: Path, docs: int = 3) -> Path:
    """Смоук-набор: голова реального v12, положенная на скретч-«сетевой диск».

    Возвращает симлинк в рабочем каталоге — ровно та монтировка, которой
    требует C-033 (данные канонически на диске, в рабочем каталоге симлинк).
    """
    shared = scratch / "shared"
    (shared / "datasets").mkdir(parents=True, exist_ok=True)
    target = shared / "datasets" / "sft_mini.jsonl"
    lines: list[str] = []
    if SHARED_DATASET.is_file():
        with open(SHARED_DATASET, encoding="utf-8") as handle:
            for index, line in enumerate(handle):
                if index >= docs:
                    break
                lines.append(line)
    else:  # pragma: no cover — набор не смонтирован: честный синтетический вход
        for index in range(docs):
            lines.append(json.dumps({
                "messages": [
                    {"role": "system", "content": "system prompt"},
                    {"role": "user", "content": f"вопрос {index}: обучение модели"},
                    {"role": "assistant", "content": "<think>шаг</think>ответ"},
                ],
                "source": "synthetic_fallback",
            }, ensure_ascii=False) + "\n")
    target.write_text("".join(lines), encoding="utf-8")
    link = scratch / "data.jsonl"
    link.symlink_to(target)
    return link


def _run_stage(
    scratch: Path, data: Path, *, steps: int = 8, out: str = "out",
    extra: tuple[str, ...] = (),
) -> subprocess.CompletedProcess:
    """Смоук-прогон стадии подпроцессом; окружение — скретч-диск и CPU."""
    env = dict(os.environ)
    env["GB10_SHARED"] = str(scratch / "shared")
    env["NET_JAX_BACKEND"] = "cpu"
    cmd = [
        sys.executable, str(RUNNER),
        "--data", str(data),
        "--out", str(scratch / out),
        "--ckpt-dir", str(scratch / out / "checkpoint"),
        "--steps", str(steps),
        "--seq-len", "128",
        "--pool-docs", "3",
        "--doc-chars", "400",
        "--model-preset", "tiny",
        "--card-scan-lines", "0",
        *extra,
    ]
    return subprocess.run(
        cmd, cwd=str(REPO_ROOT), env=env, capture_output=True, text=True, timeout=900,
    )


def _journal(scratch: Path, out: str = "out") -> dict:
    path = scratch / out / "stage-journal.json"
    assert path.is_file(), "журнал стадии не записан"
    return json.loads(path.read_text(encoding="utf-8"))


def _repo_path(relative: str) -> Path:
    """Путь журнала (относительный от корня репо либо ``~/…``) -> Path."""
    if relative.startswith("~/"):
        return Path.home() / relative[2:]
    if relative.startswith("/"):  # pragma: no cover — журнал так не пишет
        return Path(relative)
    return REPO_ROOT / relative


def _absolute_paths(node, trail: str = "") -> list[str]:
    """Рекурсивно собрать строки артефакта, начинающиеся с `/` (абсолютные)."""
    found: list[str] = []
    if isinstance(node, str):
        if node.startswith("/"):
            found.append(f"{trail}: {node}")
    elif isinstance(node, dict):
        for key, value in node.items():
            found.extend(_absolute_paths(value, f"{trail}.{key}"))
    elif isinstance(node, list):
        for index, value in enumerate(node):
            found.extend(_absolute_paths(value, f"{trail}[{index}]"))
    return found


# ---------------------------------------------------------------------------
# 1. Симлинк, а не копия (C-032/C-033)
# ---------------------------------------------------------------------------


def test_copy_instead_of_symlink_refused(repo_scratch):
    """Копия набора в рабочем каталоге — отказ (симлинк обязателен)."""
    module = _load_runner()
    shared = repo_scratch / "shared"
    shared.mkdir(parents=True, exist_ok=True)
    copy = repo_scratch / "sft_copy.jsonl"
    copy.write_text('{"messages": [{"role": "user", "content": "x"}]}\n', encoding="utf-8")

    with pytest.raises(module.StageError, match="не симлинк, а копия"):
        module.validate_data_path(copy, shared, REPO_ROOT)


def test_symlink_outside_shared_disk_refused(repo_scratch):
    """Симлинк, ведущий вне канонического диска, — отказ (C-032/C-033)."""
    module = _load_runner()
    shared = repo_scratch / "shared"
    shared.mkdir(parents=True, exist_ok=True)
    outside = repo_scratch / "outside.jsonl"
    outside.write_text('{"messages": []}\n', encoding="utf-8")
    link = repo_scratch / "data.jsonl"
    link.symlink_to(outside)

    with pytest.raises(module.StageError, match="вне канонического диска"):
        module.validate_data_path(link, shared, REPO_ROOT)


def test_symlink_inside_shared_disk_accepted(repo_scratch):
    """Симлинк на файл канонического диска принимается и описывается в журнал."""
    module = _load_runner()
    link = _mini_dataset(repo_scratch)
    described = module.validate_data_path(link, repo_scratch / "shared", REPO_ROOT)

    assert described["is_symlink"] is True
    assert described["kind"] == "file"
    assert described["target"].startswith("кейсы/axiom/net/tests/.sft-test-")
    assert not described["target"].startswith("/")


# ---------------------------------------------------------------------------
# 2. Карточка датасета: sha256 и лицензия (ADR-004)
# ---------------------------------------------------------------------------


def test_card_carries_sha256_and_license(repo_scratch):
    """Карточка несёт sha256 содержимого (потоково) и лицензию каждого блока."""
    module = _load_runner()
    link = _mini_dataset(repo_scratch)
    card = module.build_dataset_card(link, scan_lines=0)

    expected = hashlib.sha256(link.resolve().read_bytes()).hexdigest()
    assert card["sha256"] == expected, "sha256 карточки не равен хешу содержимого"
    assert card["hash"] == expected
    assert card["license"].strip(), "лицензия карточки пуста"
    assert card["blocks"], "состав блоков не объявлен"
    for block in card["blocks"]:
        assert block["license"].strip(), f"блок {block['source']} без лицензии"
        assert block["provenance"].strip(), f"блок {block['source']} без происхождения"
    assert card["lines"] > 0
    assert card["bytes"] > 0


def test_card_shape_compatible_with_datasetcard(repo_scratch):
    """Форма карточки читается `net.data.DatasetCard` (ADR-004)."""
    module = _load_runner()
    link = _mini_dataset(repo_scratch)
    card = module.build_dataset_card(link, scan_lines=0)

    datacard = module.card_to_datacard(card)
    assert datacard.name == "sft_mini"
    assert datacard.hash == card["sha256"]
    assert datacard.license == card["license"]
    assert datacard.domain == "sft_messages"


def test_unknown_source_gets_honest_license(repo_scratch):
    """Блок вне реестра лицензий получает «не установлена», а не выдумку."""
    module = _load_runner()
    shared = repo_scratch / "shared"
    (shared / "datasets").mkdir(parents=True, exist_ok=True)
    target = shared / "datasets" / "unknown.jsonl"
    target.write_text(
        json.dumps({"messages": [{"role": "user", "content": "x"}],
                    "source": "our_experiment_register_entry_absent"},
                   ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    link = repo_scratch / "unknown.jsonl"
    link.symlink_to(target)

    card = module.build_dataset_card(link, scan_lines=0)
    assert card["blocks"][0]["license"].startswith("не установлена")


# ---------------------------------------------------------------------------
# 3-5. Смоук-прогон: журнал, чекпойнт, детерминизм
# ---------------------------------------------------------------------------


def test_smoke_journal_has_no_absolute_paths(repo_scratch):
    """Смоук-журнал валиден и не содержит ни одного абсолютного пути (§4.5)."""
    module = _load_runner()
    data = _mini_dataset(repo_scratch)
    proc = _run_stage(repo_scratch, data)
    assert proc.returncode == 0, proc.stdout + proc.stderr

    journal = _journal(repo_scratch)
    assert journal["schema"] == module.JOURNAL_SCHEMA
    assert journal["stage"] == "sft"
    assert journal["status"] == "executed"
    assert journal["scale"] == "smoke"
    assert journal["seed"] == 1337
    assert journal["steps"] == 8
    assert journal["qat_weights"] == "on"
    assert journal["loss_first"] > journal["loss_last"], "лосс не упал на смоуке"
    assert journal["loss_fell"] is True
    assert len(journal["checkpoint"]["tree_hash"]) == 64
    assert journal["dataset_card"]["sha256"]
    assert journal["dataset_symlink"]["is_symlink"] is True

    offences = _absolute_paths(journal)
    assert not offences, "абсолютные пути в журнале: " + "; ".join(offences)


def test_checkpoint_tree_hash_survives_roundtrip(repo_scratch):
    """tree_hash чекпойнта стадии совпадает после round-trip (§4.1)."""
    module = _load_runner()
    data = _mini_dataset(repo_scratch)
    proc = _run_stage(repo_scratch, data)
    assert proc.returncode == 0, proc.stdout + proc.stderr

    journal = _journal(repo_scratch)
    described = journal["checkpoint"]
    assert described["roundtrip_ok"] is True
    assert described["symlink"] is True, "копия весов в рабочем каталоге (C-032)"

    from net import checkpoint as checkpoint_mod

    canonical = _repo_path(described["canonical"])
    assert canonical.is_dir(), f"каталог чекпойнта не найден: {canonical}"
    manifest = checkpoint_mod.read_manifest(canonical / "manifest.json")
    assert manifest["checkpoint_hash"] == described["tree_hash"]
    assert manifest["format"] == "orbax"

    # независимая проверка round-trip: восстанавливаем в дерево той же структуры
    # (инициализация другим сидом даёт те же формы/типы, но другие значения) и
    # считаем хеш заново — он обязан совпасть с хешем в журнале.  Отдельным
    # процессом с тем же бэкендом, что у прогона стадии (см. _VERIFY_PROGRAM).
    env = dict(os.environ)
    env["NET_JAX_BACKEND"] = "cpu"
    verify = subprocess.run(
        [sys.executable, "-c", _VERIFY_PROGRAM, str(RUNNER), str(canonical),
         str(journal["model_config"]["vocab_size"]),
         journal["model_config"]["preset"]],
        cwd=str(REPO_ROOT), env=env, capture_output=True, text=True, timeout=300,
    )
    assert verify.returncode == 0, verify.stdout + verify.stderr
    assert verify.stdout.strip().splitlines()[-1] == described["tree_hash"]


def test_same_seed_same_first_loss(repo_scratch):
    """Повторный прогон при том же сиде даёт тот же первый лосс (AD-11)."""
    data = _mini_dataset(repo_scratch)
    first = _run_stage(repo_scratch, data, steps=3, out="out")
    assert first.returncode == 0, first.stdout + first.stderr
    journal_first = _journal(repo_scratch, "out")

    second = _run_stage(repo_scratch, data, steps=3, out="out2")
    assert second.returncode == 0, second.stdout + second.stderr
    journal_second = _journal(repo_scratch, "out2")

    assert journal_first["loss_first"] == journal_second["loss_first"], (
        "первый лосс зависит от прогона, а не только от сида"
    )
    assert journal_first["loss_curve"] == journal_second["loss_curve"]
    # тот же сид — тот же пул (шаффл пиннут), значит и тот же чекпойнт
    assert (
        journal_first["checkpoint"]["tree_hash"]
        == journal_second["checkpoint"]["tree_hash"]
    )
