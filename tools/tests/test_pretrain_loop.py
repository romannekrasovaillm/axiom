"""Претрейн-луп L3 (V-4): даталоадер, расписание, resume, стоп-правило.

Источник истины — TASK.md (V-4) и ADR-021 (микс W/C ~85/15, ADR-008/009 — стек),
ADR-008 (лимит стоимости до запуска, AD-8) и `net/checkpoint.py` (tree_hash).

Сценарии:

* **L-1** — WSD-расписание: warmup → stable → decay последние 5%;
* **L-2** — манифест шарда читается как контракт (файлы, sha256, records);
* **L-3** — потоковость: за один батч читается ограниченное число документов,
  а не весь шард (корпус не поднимается в память целиком);
* **L-4** — детерминированный шаффл по сиду, устойчивый к `PYTHONHASHSEED`;
* **L-5** — микс W/C держит объявленную пропорцию по токенам;
* **L-6** — resume из курсора-манифеста: поток после resume побитово равен
  продолжению непрерывного прогона (без потерь и дублей);
* **L-7** — стоп-правило AD-8: превышение `target_tokens`/часов/лимита
  останавливает прогон с причиной в журнале; отсутствие сметы — отказ старта;
* **L-8** — ретенция чекпойнтов `keep_last N` + cursor-манифест;
* **L-9** — метрики jsonl: loss / tok-per-s / MFU;
* **L-10** — parity шага с существующим тренером скелета
  (`tools/run_sft_smoke.run_training`).

Тесты файловые и CPU-пиннутые (`NET_JAX_BACKEND=cpu` ДО первого импорта jax):
смоук-провод меряется на синтетическом мини-корпусе, а не на каноническом
диске (`~/gb10-shared` не трогается — все каталоги в `tmp_path`).
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

TOOLS_DIR = Path(__file__).resolve().parents[1]
CASE_DIR = TOOLS_DIR.parent

# Пиннинг бэкенда ДО первого импорта jax (ADR-010): тест остаётся файловым.
os.environ.setdefault("NET_JAX_BACKEND", "cpu")

for _path in (str(CASE_DIR), str(TOOLS_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import numpy as np  # noqa: E402

from net import train_loop as tl  # noqa: E402


# ---------------------------------------------------------------------------
# Фикстуры: синтетический мини-корпус + манифест канонической формы
# ---------------------------------------------------------------------------


def _approx_tokens(text: str) -> int:
    """Та же оценка, что у пайплайна подготовки (запись манифеста)."""
    return max(1, len(text) // 4)


def write_shard_set(
    root: Path,
    name: str,
    docs: list[str],
    *,
    shards: int = 1,
) -> Path:
    """Синтетический шард-набор канонической формы + манифест.

    Пишет ``<root>/<NAME>/<NAME>-0000N.jsonl.zst`` и ``manifest-<l>.json`` ровно
    в том контракте, что отдаёт ``tools/prep_pretrain`` (файл, sha256, bytes,
    approx_tokens, records, totals, source, target_tokens, codec).
    """
    import zstandard as zstd

    out_dir = root / name
    out_dir.mkdir(parents=True, exist_ok=True)
    chunk = max(1, (len(docs) + shards - 1) // shards)
    entries = []
    for index, start in enumerate(range(0, len(docs), chunk)):
        part = docs[start : start + chunk]
        payload = "".join(
            json.dumps({"text": text}, ensure_ascii=False) + "\n" for text in part
        ).encode("utf-8")
        path = out_dir / f"{name}-{index:05d}.jsonl.zst"
        path.write_bytes(zstd.ZstdCompressor(level=3).compress(payload))
        entries.append(
            {
                "file": path.name,
                "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "approx_tokens": sum(_approx_tokens(text) for text in part),
                "records": len(part),
            }
        )
    manifest = {
        "version": "axiom-pretrain-l3/1",
        "shard": name,
        "shards": entries,
        "source_records": len(docs),
        "totals": {
            "shards": len(entries),
            "bytes": sum(e["bytes"] for e in entries),
            "approx_tokens": sum(e["approx_tokens"] for e in entries),
            "records": sum(e["records"] for e in entries),
        },
        "source": {"kind": "synthetic", "name": f"synthetic-{name}"},
        "target_tokens": sum(e["approx_tokens"] for e in entries),
        "codec": "zstd",
    }
    manifest_path = out_dir / f"manifest-{name.lower()}.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    return manifest_path


def synthetic_docs(prefix: str, count: int, words: int, *, offset: int = 0) -> list[str]:
    """Детерминированные документы: слова несут индекс, поток различим."""
    return [
        " ".join(f"{prefix}{offset + i}_{w}" for w in range(words))
        for i in range(count)
    ]


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    """Мини-корпус: W из 3 шардов, C из 1 шарда."""
    root = tmp_path / "axiom-pretrain-l3"
    write_shard_set(root, "W", synthetic_docs("w", 60, 12), shards=3)
    write_shard_set(root, "C", synthetic_docs("c", 30, 12), shards=1)
    return root


def word_encoder(text: str) -> list[int]:
    """Дешёвый детерминированный энкодер (без BPE) для проводных тестов."""
    ids = []
    for word in text.split():
        digest = hashlib.sha256(word.encode("utf-8")).digest()
        ids.append(3 + int.from_bytes(digest[:2], "big") % 2045)
    return ids


def make_loader(corpus_root: Path, **overrides):
    params = dict(
        shard_root=corpus_root,
        streams=("W", "C"),
        encode=word_encoder,
        seq_len=16,
        batch_size=2,
        seed=7,
        mix={"W": 0.85, "C": 0.15},
        shuffle_window=8,
    )
    params.update(overrides)
    return tl.PretrainMixLoader(**params)


# ---------------------------------------------------------------------------
# L-1 — WSD-расписание
# ---------------------------------------------------------------------------


def test_l1_wsd_schedule_warmup_stable_decay():
    """Warmup линеен, середина стабильна, последние 5% — decay."""
    lr = tl.wsd_schedule(1.0, 100, warmup_ratio=0.10, decay_ratio=0.05)
    assert lr(0) == pytest.approx(0.0, abs=1e-9)
    assert lr(5) == pytest.approx(0.5, abs=1e-6)
    assert lr(10) == pytest.approx(1.0, abs=1e-6)
    assert lr(50) == pytest.approx(1.0, abs=1e-6)
    assert lr(94) == pytest.approx(1.0, abs=1e-6)  # последний шаг до decay
    assert lr(95) == pytest.approx(1.0, abs=1e-6)  # decay стартует с пика
    assert lr(96) < lr(95)                          # и дальше только вниз
    assert lr(99) == pytest.approx(0.0, abs=1e-6)  # минимум на последнем шаге
    stable = [lr(step) for step in range(10, 95)]
    assert stable == sorted(stable), "плато не должно дрейфовать"


def test_l1_wsd_decay_covers_last_five_percent_by_default():
    """Дефолтная доля decay — 5% от общего числа шагов."""
    lr = tl.wsd_schedule(1.0, 1000)
    assert lr(949) == pytest.approx(1.0, abs=1e-6)
    assert lr(950) == pytest.approx(1.0, abs=1e-6)  # decay начинается здесь
    assert lr(951) < 1.0
    assert lr(999) == pytest.approx(0.0, abs=1e-6)


def test_l1_wsd_never_exceeds_peak():
    """Расписание не превышает пиковый LR ни на одном шаге."""
    lr = tl.wsd_schedule(0.3, 40, warmup_ratio=0.25)
    assert max(lr(step) for step in range(40)) <= 0.3 + 1e-9


# ---------------------------------------------------------------------------
# L-2 — манифест шарда как контракт
# ---------------------------------------------------------------------------


def test_l2_load_shard_set_reads_manifest_contract(corpus: Path):
    """Шард-набор читается из манифеста: файлы, sha256, records, токены."""
    shard_set = tl.load_shard_set(corpus / "W" / "manifest-w.json")
    assert shard_set.name == "W"
    assert len(shard_set.entries) == 3
    assert sum(entry.records for entry in shard_set.entries) == 60
    assert shard_set.total_tokens == sum(e.approx_tokens for e in shard_set.entries)
    for entry in shard_set.entries:
        assert entry.path.is_file()
        assert len(entry.sha256) == 64


def test_l2_load_shard_set_rejects_unknown_shard(tmp_path: Path):
    """Неизвестное имя шарда — отказ, а не молчаливый пропуск."""
    root = tmp_path / "axiom-pretrain-l3"
    write_shard_set(root, "W", synthetic_docs("w", 4, 4))
    path = root / "W" / "manifest-w.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data["shard"] = "Z"
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(tl.PretrainDataError):
        tl.load_shard_set(path, allowed=("W", "C"))


# ---------------------------------------------------------------------------
# L-3 — потоковость (корпус не поднимается в память целиком)
# ---------------------------------------------------------------------------


def test_l3_loader_reads_only_what_is_needed(corpus: Path):
    """Один батч читает ограниченное окно документов, а не весь шард."""
    loader = make_loader(corpus)
    next(loader)
    stats = loader.stats()
    assert stats["documents_read"] < 60, "прочитан весь шард вместо окна"
    assert stats["documents_read"] <= loader.shuffle_window + 2


def test_l3_shard_documents_are_streamed_lazily(corpus: Path):
    """Итератор документов шарда — ленивый: взят первый, прочитан один."""
    entry = tl.load_shard_set(corpus / "W" / "manifest-w.json").entries[0]
    stream = tl.iter_shard_docs(entry.path)
    assert json.loads(next(stream))["text"].startswith("w0_")
    assert stream.documents_read == 1


# ---------------------------------------------------------------------------
# L-4 — детерминированный шаффл по сиду
# ---------------------------------------------------------------------------


def test_l4_window_permutation_is_a_permutation():
    """Перестановка окна — биекция на диапазоне окна."""
    perm = tl.window_permutation(seed=3, stream="W", shard_index=0, epoch=0, window_index=1, size=8)
    assert sorted(perm) == list(range(8))


def test_l4_window_permutation_depends_on_seed():
    """Разные сиды дают разные перестановки (шаффл не тождественный)."""
    kwargs = dict(stream="W", shard_index=0, epoch=0, window_index=0, size=16)
    assert tl.window_permutation(seed=1, **kwargs) != tl.window_permutation(seed=2, **kwargs)


def test_l4_window_permutation_is_stable_across_processes(tmp_path: Path):
    """Сид+окно дают ту же перестановку при другом PYTHONHASHSEED (AD-11)."""
    script = (
        "import sys; sys.path.insert(0, r'%s');"
        "from net import train_loop as tl;"
        "print(','.join(map(str, tl.window_permutation("
        "seed=5, stream='W', shard_index=0, epoch=0, window_index=2, size=12))))"
        % str(CASE_DIR)
    )
    env = dict(os.environ, PYTHONHASHSEED="12345", NET_JAX_BACKEND="cpu")
    first = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, env=env, check=True
    ).stdout.strip()
    env["PYTHONHASHSEED"] = "999"
    second = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, env=env, check=True
    ).stdout.strip()
    expected = ",".join(
        str(i)
        for i in tl.window_permutation(
            seed=5, stream="W", shard_index=0, epoch=0, window_index=2, size=12
        )
    )
    assert first == second == expected


def test_l4_loader_shuffle_is_reproducible(corpus: Path):
    """Два лоадера с одним сидом дают побитово тот же поток батчей."""
    left = [np.asarray(b) for b in make_loader(corpus)]
    right = [np.asarray(b) for b in make_loader(corpus)]
    assert len(left) == len(right)
    for a, b in zip(left, right):
        assert np.array_equal(a, b)


# ---------------------------------------------------------------------------
# L-5 — микс W/C
# ---------------------------------------------------------------------------


def test_l5_mix_ratio_follows_declared_weights(corpus: Path):
    """Пропорция по токенам держит объявленные веса с допуском."""
    loader = make_loader(corpus, mix={"W": 0.85, "C": 0.15})
    for _ in loader:
        pass
    share = loader.stats()["token_share"]
    assert share["W"] == pytest.approx(0.85, abs=0.05), share
    assert share["C"] == pytest.approx(0.15, abs=0.05), share


def test_l5_mix_ratio_changes_with_weights(corpus: Path):
    """Другие веса — другая доля (правило читает веса, а не константу)."""
    loader = make_loader(corpus, mix={"W": 0.5, "C": 0.5})
    for _ in loader:
        pass
    share = loader.stats()["token_share"]
    assert share["C"] == pytest.approx(0.5, abs=0.05), share


# ---------------------------------------------------------------------------
# L-6 — resume из курсора-манифеста
# ---------------------------------------------------------------------------


def test_l6_cursor_roundtrips_through_json(corpus: Path):
    """Курсор сериализуется и восстанавливается без потерь."""
    loader = make_loader(corpus)
    for _ in range(4):
        next(loader)
    cursor = loader.cursor(step=4)
    restored = tl.MixCursor.from_json(json.loads(json.dumps(cursor.to_json())))
    assert restored == cursor


def test_l6_resume_reproduces_uninterrupted_stream(corpus: Path):
    """Поток после resume равен продолжению непрерывного прогона (L-6)."""
    continuous = [np.asarray(batch) for batch in make_loader(corpus)]

    first_leg = make_loader(corpus)
    prefix = [np.asarray(next(first_leg)) for _ in range(3)]
    cursor = first_leg.cursor(step=3)

    resumed = make_loader(corpus, cursor=cursor)
    tail = [np.asarray(batch) for batch in resumed]

    assert len(tail) == len(continuous) - 3
    for index, batch in enumerate(tail):
        assert np.array_equal(batch, continuous[3 + index]), f"расхождение на батче {index}"


def test_l6_resume_across_shard_boundary(corpus: Path):
    """Resume на границе шарда не теряет и не дублирует документы."""
    continuous = [np.asarray(batch) for batch in make_loader(corpus)]
    loader = make_loader(corpus)
    # Провязываем поток до перехода во второй шард W (окно 8 → 60/3 = 20 на шард).
    seen = []
    while loader.stats()["streams"]["W"]["shard_index"] == 0:
        seen.append(np.asarray(next(loader)))
        if len(seen) > len(continuous):
            pytest.fail("шард не переключился: поток не продвигается")
    cursor = loader.cursor(step=len(seen))
    resumed = make_loader(corpus, cursor=cursor)
    tail = [np.asarray(batch) for batch in resumed]
    assert len(tail) == len(continuous) - len(seen)
    for index, batch in enumerate(tail):
        assert np.array_equal(batch, continuous[len(seen) + index])


def test_l6_cursor_reports_no_loss_no_duplication(corpus: Path):
    """Число токенов и документов в курсоре — суммарные, без двойного счёта."""
    loader = make_loader(corpus)
    for _ in range(3):
        next(loader)
    cursor = loader.cursor(step=3)
    streams = {item.name: item for item in cursor.streams}
    assert streams["W"].docs + streams["C"].docs == loader.stats()["documents_read"]
    assert sum(item.tokens for item in cursor.streams) == cursor.tokens_total


# ---------------------------------------------------------------------------
# L-7 — стоп-правило AD-8
# ---------------------------------------------------------------------------


def write_budget(path: Path, **overrides) -> Path:
    """Смета канонической формы (AD-8/C-041) + поле ``target_tokens``."""
    payload = {
        "schema": "budget-estimate/v1",
        "run_ref": "pretrain-smoke",
        "gpu_type": "RTX 4080 SUPER (локальная)",
        "gpu_hours_estimate": 1.0,
        "usd_estimate": 0.2,
        "limit_usd": 1.0,
        "target_tokens": 1_000_000,
        "budget_method": "калибровка по arXiv 2412.19437: 343 TFLOP/s эффективных на H800",
        "stop_rule": "остановка после ближайшего чекпойнта при превышении лимита",
        "created_at": "2026-09-30T10:00:00+03:00",
        "approved_by": "architect",
    }
    payload.update(overrides)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


def test_l7_load_budget_reads_limits(tmp_path: Path):
    """Смета читается: target_tokens, часы, лимит и метод."""
    path = write_budget(tmp_path / "pretrain-smoke.json")
    budget = tl.load_budget(path)
    assert budget.present is True
    assert budget.target_tokens == 1_000_000
    assert budget.gpu_hours_estimate == 1.0
    assert budget.limit_usd == 1.0
    assert "2412.19437" in budget.budget_method


def test_l7_missing_budget_blocks_start(tmp_path: Path):
    """Отсутствие сметы — блокирующее условие запуска (AD-8)."""
    budget = tl.load_budget(tmp_path / "нет-такого.json")
    assert budget.present is False
    with pytest.raises(tl.PretrainBudgetError):
        tl.require_budget(budget)


def test_l7_missing_budget_allowed_by_explicit_limit(tmp_path: Path):
    """Явный лимит смоука снимает блокировку, но фиксируется как факт."""
    budget = tl.load_budget(tmp_path / "нет-такого.json")
    resolved = tl.require_budget(budget, explicit_limit_usd=1.0)
    assert resolved.limit_usd == 1.0
    assert resolved.present is False


def test_l7_no_breach_inside_limits(tmp_path: Path):
    """В пределах сметы стоп-правило молчит."""
    budget = tl.load_budget(write_budget(tmp_path / "b.json"))
    assert tl.budget_breach(budget, tokens_seen=10, gpu_hours=0.1, usd_spent=0.01) is None


def test_l7_token_breach_stops_with_reason(tmp_path: Path):
    """Превышение target_tokens останавливает прогон с причиной в журнале."""
    budget = tl.load_budget(write_budget(tmp_path / "b.json"))
    reason = tl.budget_breach(budget, tokens_seen=1_000_001, gpu_hours=0.1, usd_spent=0.01)
    assert reason is not None and "target_tokens" in reason


def test_l7_hours_and_usd_breach_stop_with_reason(tmp_path: Path):
    """Превышение часов и лимита USD — тоже остановка."""
    budget = tl.load_budget(write_budget(tmp_path / "b.json"))
    hours = tl.budget_breach(budget, tokens_seen=10, gpu_hours=1.5, usd_spent=0.0)
    assert hours is not None and "gpu_hours" in hours
    usd = tl.budget_breach(budget, tokens_seen=10, gpu_hours=0.0, usd_spent=2.0)
    assert usd is not None and "limit_usd" in usd


# ---------------------------------------------------------------------------
# L-8 — чекпойнты: ретенция keep_last + resume из курсора-манифеста
# ---------------------------------------------------------------------------


def tiny_params():
    """Минимальный pytree параметров и состояния оптимизатора для чекпойнта."""
    import jax.numpy as jnp

    params = {"w": jnp.arange(12, dtype=jnp.float32).reshape(3, 4), "b": jnp.ones((3,))}
    state = {"w": jnp.zeros((3, 4), dtype=jnp.float32), "b": jnp.zeros((3,))}
    return params, state


def test_l8_checkpoint_roundtrip_and_cursor(tmp_path: Path):
    """Чекпойнт сохраняется с tree_hash, курсор-манифест указывает на него."""
    params, state = tiny_params()
    manager = tl.CheckpointManager(tmp_path / "ckpt", keep_last=2)
    record = manager.save(step=5, params=params, optimizer_state=state, cursor={"step": 5})
    assert len(record["tree_hash"]) == 64
    assert Path(record["path"]).is_dir()

    latest = manager.latest()
    assert latest["step"] == 5
    assert latest["tree_hash"] == record["tree_hash"]

    restored_params, restored_state, cursor = manager.resume(
        target_params=params, target_state=state
    )
    assert cursor == {"step": 5}
    from net import checkpoint as checkpoint_mod

    assert checkpoint_mod.tree_hash(restored_params) == record["tree_hash"]
    assert np.allclose(np.asarray(restored_state["w"]), 0.0)


def test_l8_keep_last_prunes_old_checkpoints(tmp_path: Path):
    """keep_last N оставляет N последних чекпойнтов, старые удаляются."""
    params, state = tiny_params()
    manager = tl.CheckpointManager(tmp_path / "ckpt", keep_last=2)
    for step in (1, 2, 3, 4):
        manager.save(step=step, params=params, optimizer_state=state, cursor={"step": step})
    steps = sorted(record.step for record in manager.list())
    assert steps == [3, 4]
    assert manager.latest()["step"] == 4


def test_l8_resume_continues_without_loss(tmp_path: Path, corpus: Path):
    """Сквозной resume: чекпойнт + курсор дают продолжение без потерь и дублей."""
    continuous = [np.asarray(batch) for batch in make_loader(corpus)]

    loader = make_loader(corpus)
    consumed = [np.asarray(next(loader)) for _ in range(2)]
    cursor = loader.cursor(step=2)
    params, state = tiny_params()
    manager = tl.CheckpointManager(tmp_path / "ckpt", keep_last=1)
    manager.save(step=2, params=params, optimizer_state=state, cursor=cursor.to_json())

    restored_params, _state, raw_cursor = manager.resume(
        target_params=params, target_state=state
    )
    resumed_loader = make_loader(corpus, cursor=tl.MixCursor.from_json(raw_cursor))
    tail = [np.asarray(batch) for batch in resumed_loader]

    assert len(tail) == len(continuous) - 2
    for index, batch in enumerate(tail):
        assert np.array_equal(batch, continuous[2 + index])


# ---------------------------------------------------------------------------
# L-9 — метрики jsonl: loss / ток-в-с / MFU
# ---------------------------------------------------------------------------


def test_l9_metrics_appends_json_lines(tmp_path: Path):
    """Метрики дописываются построчно в jsonl и читаются обратно."""
    path = tmp_path / "metrics.jsonl"
    writer = tl.MetricsWriter(path)
    writer.log({"step": 1, "loss": 9.0, "tokens_per_sec": 100.0, "mfu": 0.5})
    writer.log({"step": 2, "loss": 8.0, "tokens_per_sec": 110.0, "mfu": 0.51})
    rows = tl.MetricsWriter.read(path)
    assert [row["step"] for row in rows] == [1, 2]
    assert rows[1]["loss"] == 8.0
    assert rows[0]["schema"] == tl.METRICS_SCHEMA


def test_l9_mfu_from_flops_and_peak():
    """MFU = достигнутые TFLOP/s / объявленный пик (6·N·tokens на шаг)."""
    flops = tl.step_flops(active_params=1_000_000_000, tokens=8192)
    assert flops == pytest.approx(6 * 1_000_000_000 * 8192)
    value = tl.mfu(flops, seconds=1.0, peak_tflops=100.0)
    assert value == pytest.approx(49.152 / 100.0, rel=1e-6)


def test_l9_mfu_is_none_without_declared_peak():
    """Без объявленного пика MFU не выдумывается — вместо числа None."""
    flops = tl.step_flops(active_params=10, tokens=10)
    assert tl.mfu(flops, seconds=1.0, peak_tflops=None) is None
    assert tl.mfu(flops, seconds=0.0, peak_tflops=100.0) is None


# ---------------------------------------------------------------------------
# L-10 — цикл обучения и parity с существующим тренером скелета
# ---------------------------------------------------------------------------


def acceptance_conftest():
    """Численные смоук-конфиги приёмки сети (``net/tests/conftest.py``)."""
    import importlib.util

    path = CASE_DIR / "net" / "tests" / "conftest.py"
    spec = importlib.util.spec_from_file_location("net_acceptance_conftest", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parity_setup(steps: int = 3):
    """Конфиг и пул последовательностей ``(T,)`` — форма пула ``run_sft_smoke``."""
    import jax.numpy as jnp
    import jax.random as jr

    cfg = acceptance_conftest().tiny_config()
    key = jr.PRNGKey(11)
    pool = [
        jr.randint(jr.fold_in(key, index), (16,), 0, cfg.vocab_size, dtype=jnp.int32)
        for index in range(steps)
    ]
    return cfg, pool


def batched(pool):
    """Батчи ``(B, T)`` для ``train`` из пула последовательностей ``(T,)``."""
    import numpy as np

    return iter([np.asarray(sequence)[None, :] for sequence in pool])


def free_budget() -> "tl.Budget":
    """Смета без границ: тесты гоняют провод, а не лимит стоимости.

    ``train`` требует смету явно (AD-8), поэтому тест обязан её предъявить —
    отсутствие сметы проверяется отдельным тестом L-10.
    """
    return tl.Budget(run_ref="test", path=Path("/dev/null"), present=True)


def test_l10_step_parity_with_existing_trainer():
    """Шаг ``train`` побитово совпадает с ``run_sft_smoke.run_training``.

    Существующий тренер — оракул проводки (loss, оптимизатор, расписание,
    сид); расхождение здесь означало бы, что претрейн-луп обучает не тем
    шагом, что приёмка скелета.
    """
    import run_sft_smoke as runner

    cfg, pool = parity_setup(steps=3)
    reference_losses, reference_steps, reference_params, reference_hash, _ = runner.run_training(
        pool, cfg, steps=3, seed=7, lr=1e-2, warmup_ratio=0.01, qat_weights=False
    )

    result = tl.train(
        cfg,
        batched(pool),
        train_config=tl.TrainConfig(
            steps=3, lr=1e-2, seed=7, warmup_ratio=0.01, schedule="cosine",
            param_dtype="float32", grad_checkpointing=False,
        ),
        budget=free_budget(),
    )

    assert result.steps_done == reference_steps == 3
    for index, (mine, reference) in enumerate(zip(result.losses, reference_losses)):
        assert mine == pytest.approx(reference, rel=0, abs=0), f"лосс шага {index}"
    assert result.tree_hash == reference_hash, "параметры после шагов разошлись"


def dtype_names(tree) -> set:
    """Множество имён dtype по листьям pytree (сравнимо без магии jnp-классов)."""
    import jax

    return {leaf.dtype.name for leaf in jax.tree_util.tree_leaves(tree)}


def max_leaf_delta(left, right) -> float:
    """Наибольшее покомпонентное расхождение двух деревьев (в float64)."""
    import jax
    import numpy as np

    deltas = jax.tree_util.tree_map(
        lambda a, b: float(
            np.abs(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)).max()
        ),
        left,
        right,
    )
    return max(jax.tree_util.tree_leaves(deltas))


def test_l10_bf16_params_with_fp32_master():
    """bf16-параметры обучаются fp32-мастером: мастер остаётся fp32 и меняется."""
    cfg, pool = parity_setup(steps=2)
    from net import model

    import jax.random as jr

    initial = model.init_params(jr.PRNGKey(7), cfg)
    result = tl.train(
        cfg,
        batched(pool),
        train_config=tl.TrainConfig(
            steps=2, lr=1e-2, seed=7, param_dtype="bfloat16", schedule="cosine"
        ),
        budget=free_budget(),
    )
    assert dtype_names(result.params) == {"bfloat16"}, dtype_names(result.params)
    assert dtype_names(result.master_params) == {"float32"}, dtype_names(result.master_params)
    # шаг не вырожден: мастер реально обновился
    assert max_leaf_delta(initial, result.master_params) > 0.0
    # bf16-веса — это усечение мастера, а не независимая копия
    assert max_leaf_delta(
        result.params,
        jax_numpy_cast(result.master_params, "bfloat16"),
    ) == 0.0


def jax_numpy_cast(tree, dtype: str):
    """Привести дерево к dtype — в тесте, чтобы не тянуть jax в модуль тестов."""
    import jax
    import jax.numpy as jnp

    return jax.tree_util.tree_map(lambda leaf: leaf.astype(jnp.dtype(dtype)), tree)


def test_l10_grad_checkpointing_keeps_numerics():
    """Grad-checkpointing (remat) не меняет численный результат шага."""
    cfg, pool = parity_setup(steps=2)
    plain = tl.train(
        cfg, batched(pool), train_config=tl.TrainConfig(steps=2, lr=1e-2, seed=7),
        budget=free_budget(),
    )
    remat = tl.train(
        cfg,
        batched(pool),
        train_config=tl.TrainConfig(steps=2, lr=1e-2, seed=7, grad_checkpointing=True),
        budget=free_budget(),
    )
    assert remat.grad_checkpointing is True
    for mine, reference in zip(remat.losses, plain.losses):
        assert mine == pytest.approx(reference, rel=1e-5, abs=1e-5)
    # Побитового совпадения нет и быть не должно: remat пересобирает backward,
    # XLA фьюзит его иначе, и ассоциация сложений float32 меняется. Требование —
    # численная эквивалентность, а не один и тот же хеш.
    tolerance = max_leaf_delta(plain.params, remat.params)
    assert tolerance < 1e-5, f"remat разошёлся с обычным шагом на {tolerance}"


def test_l11_per_layer_policy_skips_outer_coarse_wrap(monkeypatch):
    """L-11: модельная политика ``per_layer`` отменяет внешнюю coarse-обёртку.

    Вложение ``jax.checkpoint``(coarse) поверх послойных remat внутри
    ``compute_loss`` приводит к тому, что XLA инлайнит внутренние remat —
    компилированный граф совпадает с нематериализованным (OOM 888 ГиБ на
    пилоте GB10 03.10.2026). Обёртка применяется только при политике ``none``.
    """
    import dataclasses

    cfg, pool = parity_setup(steps=1)
    cfg = dataclasses.replace(cfg, grad_ckpt_policy="per_layer")
    calls: list[str] = []
    real = tl._checkpoint_policy

    def spy(name):
        calls.append(name)
        return real(name)

    monkeypatch.setattr(tl, "_checkpoint_policy", spy)
    result = tl.train(
        cfg,
        batched(pool),
        train_config=tl.TrainConfig(steps=1, lr=1e-2, seed=7, grad_checkpointing=True),
        budget=free_budget(),
    )
    assert result.steps_done == 1
    assert calls == [], f"coarse-обёртка применилась поверх per_layer: {calls}"


def test_l11_none_policy_keeps_outer_coarse_wrap(monkeypatch):
    """L-11: при политике ``none`` coarse-обёртка по флагу остаётся."""
    cfg, pool = parity_setup(steps=1)
    calls: list[str] = []
    real = tl._checkpoint_policy

    def spy(name):
        calls.append(name)
        return real(name)

    monkeypatch.setattr(tl, "_checkpoint_policy", spy)
    result = tl.train(
        cfg,
        batched(pool),
        train_config=tl.TrainConfig(steps=1, lr=1e-2, seed=7, grad_checkpointing=True),
        budget=free_budget(),
    )
    assert result.steps_done == 1
    assert calls == ["full"], f"coarse-обёртка потерялась при политике none: {calls}"


def test_l10_wsd_schedule_reaches_the_loop():
    """В цикле действует WSD: decay последних 20% опускает LR ниже пика."""
    cfg, pool = parity_setup(steps=20)
    result = tl.train(
        cfg,
        batched(pool),
        train_config=tl.TrainConfig(
            steps=20, lr=1e-2, seed=7, schedule="wsd", warmup_ratio=0.10, decay_ratio=0.20
        ),
        budget=free_budget(),
    )
    assert result.steps_done == 20
    assert result.lr_history[9] == pytest.approx(1e-2, rel=1e-6)   # плато
    assert result.lr_history[-1] < result.lr_history[9] * 0.5      # decay дошёл


def test_l10_data_exhaustion_stops_gracefully():
    """Исчерпание батчей — штатная остановка с причиной, а не исключение."""
    cfg, pool = parity_setup(steps=2)
    result = tl.train(
        cfg, batched(pool), train_config=tl.TrainConfig(steps=5, seed=7), budget=free_budget()
    )
    assert result.steps_done == 2
    assert result.stop_reason is not None and "data" in result.stop_reason


def test_l10_stop_rule_halts_on_target_tokens(tmp_path: Path):
    """Превышение target_tokens останавливает цикл с причиной (AD-8)."""
    cfg, pool = parity_setup(steps=5)
    budget = tl.load_budget(write_budget(tmp_path / "b.json", target_tokens=32))
    result = tl.train(
        cfg,
        batched(pool),
        train_config=tl.TrainConfig(steps=5, lr=1e-2, seed=7),
        budget=budget,
    )
    assert result.steps_done < 5, "стоп-правило не сработало"
    assert result.stop_reason and "target_tokens" in result.stop_reason
    assert result.stopped_by_budget is True


def test_l10_missing_budget_blocks_training(tmp_path: Path):
    """Без сметы цикл не стартует (AD-8), а не «едет молча»."""
    cfg, pool = parity_setup(steps=1)
    budget = tl.load_budget(tmp_path / "нет.json")
    with pytest.raises(tl.PretrainBudgetError):
        tl.train(
            cfg, batched(pool), train_config=tl.TrainConfig(steps=1), budget=budget
        )


def test_l10_metrics_and_checkpoints_are_written(tmp_path: Path):
    """Цикл пишет метрики jsonl и чекпойнты с курсором."""
    cfg, pool = parity_setup(steps=2)
    metrics = tmp_path / "metrics.jsonl"
    result = tl.train(
        cfg,
        batched(pool),
        train_config=tl.TrainConfig(
            steps=2,
            lr=1e-2,
            seed=7,
            checkpoint_every=1,
            keep_last=2,
            ckpt_dir=tmp_path / "ckpt",
            metrics_path=metrics,
            peak_tflops=100.0,
        ),
        budget=free_budget(),
    )
    rows = tl.MetricsWriter.read(metrics)
    assert len(rows) == 2
    assert rows[0]["step"] == 1
    assert rows[0]["tokens_per_sec"] > 0
    assert rows[0]["tflops_achieved"] > 0
    assert rows[0]["mfu"] is not None
    assert rows[0]["mfu"] == pytest.approx(rows[0]["tflops_achieved"] / 100.0, rel=1e-9)
    assert rows[0]["mfu_params_only"] is True
    assert result.checkpoint is not None
    assert result.checkpoint["step"] == 2
    assert len(result.checkpoint["tree_hash"]) == 64
    manager = tl.CheckpointManager(tmp_path / "ckpt", keep_last=2)
    assert manager.latest()["step"] == 2


def test_l10_resume_continues_training_identically(tmp_path: Path):
    """Resume из чекпойнта продолжает обучение без потерь (params + данные).

    Расписание считается от ``total_steps`` (горизонт прогона), а не от длины
    ноги: иначе вторая нога поехала бы по другому LR — классическая ошибка
    resume, из-за которой продолжение не равно непрерывному прогону.
    """
    cfg, pool = parity_setup(steps=4)
    continuous = tl.train(
        cfg, batched(pool), train_config=tl.TrainConfig(steps=4, lr=1e-2, seed=7),
        budget=free_budget(),
    )

    first_leg = tl.train(
        cfg,
        batched(pool[:2]),
        train_config=tl.TrainConfig(
            steps=2,
            total_steps=4,
            lr=1e-2,
            seed=7,
            ckpt_dir=tmp_path / "ckpt",
            checkpoint_every=2,
        ),
        budget=free_budget(),
    )
    assert first_leg.steps_done == 2

    manager = tl.CheckpointManager(tmp_path / "ckpt", keep_last=2)
    second_leg = tl.train(
        cfg,
        batched(pool[2:]),
        train_config=tl.TrainConfig(steps=2, total_steps=4, lr=1e-2, seed=7),
        resume_from=manager,
        budget=free_budget(),
    )
    assert second_leg.steps_done == 2, "resume не продолжил, а начал заново"
    assert second_leg.tree_hash == continuous.tree_hash, "resume разошёлся с непрерывным прогоном"
    for mine, reference in zip(second_leg.losses, continuous.losses[2:]):
        assert mine == pytest.approx(reference, rel=1e-6, abs=1e-6)


# ---------------------------------------------------------------------------
# L-11 — CLI: портируемость журнала и отказы, которые не молчат
# ---------------------------------------------------------------------------


def pretrain_cli():
    """CLI стадии претрейна (``tools/pretrain_run.py``)."""
    import pretrain_run

    return pretrain_run


def test_l11_portable_paths_strip_build_machine_paths():
    """Пути сборочной машины в журнал не попадают (ADR-014 п. 8)."""
    cli = pretrain_cli()
    journal = cli.portable_paths(
        {
            "estimate": f"{CASE_DIR}/evidence/budget/x.json — нет сметы",
            "out": f"{Path.home()}/gb10-shared/runs/pretrain-l3",
        },
        CASE_DIR,
    )
    assert str(CASE_DIR) not in json.dumps(journal, ensure_ascii=False)
    assert str(Path.home()) not in json.dumps(journal, ensure_ascii=False)
    assert journal["out"] == "~/gb10-shared/runs/pretrain-l3"


def test_l11_absolute_path_leaks_reports_leftovers():
    """Оставшийся абсолютный путь — находка, а не тишина."""
    cli = pretrain_cli()
    leaks = cli.absolute_path_leaks({"a": {"b": ["/var/tmp/остаток"]}})
    assert leaks and "journal.a.b[0]" in leaks[0]
    assert cli.absolute_path_leaks({"a": "~/gb10-shared/x", "b": "evidence/budget/x.json"}) == []


def test_l11_cli_refuses_without_budget_and_writes_portable_journal(tmp_path: Path):
    """Отсутствие сметы — отказ с кодом 1 и журналом без путей машины."""
    cli = pretrain_cli()
    out_dir = tmp_path / "out"
    code = cli.main(
        [
            "--run-ref",
            "нет-такой-сметы",
            "--out",
            str(out_dir),
            "--steps",
            "1",
            "--budget-file",
            str(CASE_DIR / "evidence" / "budget" / "нет-такой-сметы.json"),
        ]
    )
    assert code == 1
    journal = json.loads((out_dir / "journal.json").read_text(encoding="utf-8"))
    assert journal["status"] == "absent"
    assert "смет" in journal["refusal"]
    text = json.dumps(journal, ensure_ascii=False)
    assert str(CASE_DIR) not in text, "абсолютный путь кейса утёк в журнал"
    assert str(Path.home()) not in text, "домашний путь утёк в журнал"


def test_l11_cli_refuses_l3_full_without_grad_checkpointing(tmp_path: Path):
    """l3-full без remat — отказ (урок OOM 956 ГиБ), а не тихий запуск."""
    cli = pretrain_cli()
    code = cli.main(
        [
            "--run-ref",
            "pretrain-nogc",
            "--model-preset",
            "l3-full",
            "--no-grad-checkpointing",
            "--budget-limit-usd",
            "5",
            "--out",
            str(tmp_path / "out"),
            "--steps",
            "1",
        ]
    )
    assert code == 1
    journal = json.loads((tmp_path / "out" / "journal.json").read_text(encoding="utf-8"))
    assert "grad-checkpointing" in journal["refusal"]


def test_l11_resume_cursor_restored_from_checkpoint(tmp_path: Path, corpus: Path):
    """Курсор данных восстанавливается из чекпойнта (иначе resume идёт с нуля).

    Дефект, найденный сквозным GPU-прогоном: веса восстанавливались, а поток
    данных начинался заново — лосс выглядел правдоподобно, но прогон повторял
    уже пройденные документы.
    """
    cli = pretrain_cli()
    params, state = tiny_params()
    continuous = [np.asarray(batch) for batch in make_loader(corpus)]

    loader = make_loader(corpus)
    for _ in range(2):
        next(loader)
    cursor = loader.cursor(step=2)
    manager = tl.CheckpointManager(tmp_path / "ckpt", keep_last=1)
    manager.save(step=2, params=params, optimizer_state=state, cursor=cursor.to_json())

    restored = cli.resume_cursor(manager)
    assert restored == cursor, "курсор данных не восстановился из чекпойнта"
    resumed = make_loader(corpus, cursor=restored)
    tail = [np.asarray(batch) for batch in resumed]
    assert len(tail) == len(continuous) - 2
    for index, batch in enumerate(tail):
        assert np.array_equal(batch, continuous[2 + index])

    assert cli.resume_cursor(None) is None
    empty = tl.CheckpointManager(tmp_path / "пусто", keep_last=1)
    assert cli.resume_cursor(empty) is None


# ---------------------------------------------------------------------------
# L-12 — decay-фаза (ADR-021): третий поток Q, переключение по шагу, resume
# ---------------------------------------------------------------------------


@pytest.fixture
def corpus_with_q(tmp_path: Path) -> Path:
    """Мини-корпус с decay-шардом Q: W (3 шарда), C (1), Q (1)."""
    root = tmp_path / "axiom-pretrain-l3-q"
    write_shard_set(root, "W", synthetic_docs("w", 60, 12), shards=3)
    write_shard_set(root, "C", synthetic_docs("c", 30, 12), shards=1)
    write_shard_set(root, "Q", synthetic_docs("q", 40, 12), shards=1)
    return root


def make_q_loader(corpus_root: Path, *, decay_start: int, **overrides):
    """Даталоадер с decay-фазой: микс W/C, а с шага ``decay_start`` — шард Q."""
    params = dict(
        shard_root=corpus_root,
        streams=("W", "C"),
        encode=word_encoder,
        seq_len=16,
        batch_size=2,
        seed=7,
        mix={"W": 0.85, "C": 0.15},
        shuffle_window=8,
        decay_stream="Q",
        decay_start=decay_start,
    )
    params.update(overrides)
    return tl.PretrainMixLoader(**params)


def test_l12_decay_start_step_matches_wsd_boundary():
    """Граница данных = граница LR WSD (одна формула, не две копии)."""
    assert tl.decay_start_step(100, warmup_ratio=0.01, decay_ratio=0.05) == 95
    assert tl.decay_start_step(1000, warmup_ratio=0.01, decay_ratio=0.05) == 950
    assert tl.decay_start_step(100, warmup_ratio=0.10, decay_ratio=0.05) == 95
    # decay_ratio=0 — decay-фазы нет (граница за последним шагом)
    assert tl.decay_start_step(50, decay_ratio=0.0) == 50


def test_l12_phase_switches_at_the_boundary(corpus_with_q: Path):
    """Фаза stable на первых ``decay_start`` батчах, decay — начиная со следующего."""
    loader = make_q_loader(corpus_with_q, decay_start=3)
    seen = []
    for _ in loader:
        seen.append(loader.phase)
    assert seen[:3] == [tl.PHASE_STABLE] * 3, seen
    assert seen[3:] and all(phase == tl.PHASE_DECAY for phase in seen[3:]), seen
    assert "phase" in loader.stats() and loader.stats()["decay_start"] == 3


def test_l12_decay_phase_feeds_only_q_stream(corpus_with_q: Path):
    """После границы W/C заморожены, растёт только Q — микс не подмешивается."""
    loader = make_q_loader(corpus_with_q, decay_start=3)
    for _ in range(3):
        next(loader)
    frozen = loader.stats()["streams"]
    stable_w, stable_c, stable_q = (
        frozen["W"]["tokens"],
        frozen["C"]["tokens"],
        frozen["Q"]["tokens"],
    )
    assert stable_q == 0, "Q не должен течь в стабильной фазе"
    for _ in range(3):
        next(loader)
    after = loader.stats()["streams"]
    assert after["W"]["tokens"] == stable_w, "W подмешался в decay-фазу"
    assert after["C"]["tokens"] == stable_c, "C подмешался в decay-фазу"
    assert after["Q"]["tokens"] > stable_q, "Q не питает decay-фазу"
    # доля Q растёт только за счёт decay-шагов (W/C заморожены)
    assert loader.stats()["token_share"]["Q"] > 0.0


def test_l12_decay_batches_equal_q_only_reader(corpus_with_q: Path):
    """Decay-поток — тот же reader и сид, что у Q-only набора (свой порядок)."""
    loader = make_q_loader(corpus_with_q, decay_start=3)
    for _ in range(3):
        next(loader)
    decay_batches = [np.asarray(next(loader)) for _ in range(3)]

    q_only = tl.PretrainMixLoader(
        shard_root=corpus_with_q,
        streams=("Q",),
        encode=word_encoder,
        seq_len=16,
        batch_size=2,
        seed=7,
        mix={"Q": 1.0},
        shuffle_window=8,
    )
    expected = [np.asarray(next(q_only)) for _ in range(3)]
    for index, (mine, reference) in enumerate(zip(decay_batches, expected)):
        assert np.array_equal(mine, reference), f"Q-поток расходится на батче {index}"


def test_l12_cursor_carries_phase_and_roundtrips(corpus_with_q: Path):
    """Курсор несёт фазу-владельца хвоста и переживает json без потерь."""
    loader = make_q_loader(corpus_with_q, decay_start=2)
    for _ in range(3):
        next(loader)
    cursor = loader.cursor(step=3)
    assert cursor.phase == tl.PHASE_DECAY
    restored = tl.MixCursor.from_json(json.loads(json.dumps(cursor.to_json())))
    assert restored == cursor


def test_l12_resume_across_phase_boundary_matches_continuous(corpus_with_q: Path):
    """Resume ровно на границе фаз воспроизводит непрерывный прогон."""
    continuous = [np.asarray(batch) for batch in make_q_loader(corpus_with_q, decay_start=3)]

    leg = make_q_loader(corpus_with_q, decay_start=3)
    for _ in range(3):
        next(leg)
    cursor = leg.cursor(step=3)  # снимок на последнем stable-шаге
    assert cursor.phase == tl.PHASE_STABLE

    resumed = make_q_loader(corpus_with_q, decay_start=3, cursor=cursor)
    tail = [np.asarray(batch) for batch in resumed]
    assert len(tail) == len(continuous) - 3
    for index, batch in enumerate(tail):
        assert np.array_equal(batch, continuous[3 + index]), f"расхождение на батче {index}"


def test_l12_resume_inside_decay_matches_continuous(corpus_with_q: Path):
    """Resume внутри decay-фазы воспроизводит хвост Q-потока без потерь."""
    continuous = [np.asarray(batch) for batch in make_q_loader(corpus_with_q, decay_start=3)]

    leg = make_q_loader(corpus_with_q, decay_start=3)
    for _ in range(5):
        next(leg)
    cursor = leg.cursor(step=5)
    assert cursor.phase == tl.PHASE_DECAY

    resumed = make_q_loader(corpus_with_q, decay_start=3, cursor=cursor)
    tail = [np.asarray(batch) for batch in resumed]
    assert len(tail) == len(continuous) - 5
    for index, batch in enumerate(tail):
        assert np.array_equal(batch, continuous[5 + index])


def test_l12_decay_requires_boundary_and_distinct_stream(corpus_with_q: Path):
    """Некорректная конфигурация decay-фазы — отказ, а не тихий микс."""
    with pytest.raises(tl.PretrainDataError):
        make_q_loader(corpus_with_q, decay_start=None)  # type: ignore[arg-type]
    with pytest.raises(tl.PretrainDataError):
        tl.PretrainMixLoader(
            shard_root=corpus_with_q,
            streams=("W", "C", "Q"),
            encode=word_encoder,
            seq_len=16,
            mix={"W": 0.7, "C": 0.15, "Q": 0.15},
            decay_stream="Q",
            decay_start=2,
        )


def test_l12_metrics_record_phase(tmp_path: Path):
    """jsonl метрик несёт фазу шага: stable до границы, decay после."""
    cfg, pool = parity_setup(steps=4)

    class StubLoader:
        decay_start = 2

    metrics = tmp_path / "metrics.jsonl"
    tl.train(
        cfg,
        batched(pool),
        train_config=tl.TrainConfig(steps=4, lr=1e-2, seed=7, metrics_path=metrics),
        budget=free_budget(),
        loader=StubLoader(),
    )
    rows = tl.MetricsWriter.read(metrics)
    assert [row["phase"] for row in rows] == [
        tl.PHASE_STABLE,
        tl.PHASE_STABLE,
        tl.PHASE_DECAY,
        tl.PHASE_DECAY,
    ]


def test_l12_metrics_without_decay_are_stable(tmp_path: Path):
    """Без decay-шарда метрика phase стабильна — фаза не выдумывается."""
    cfg, pool = parity_setup(steps=3)
    metrics = tmp_path / "metrics.jsonl"
    tl.train(
        cfg,
        batched(pool),
        train_config=tl.TrainConfig(steps=3, lr=1e-2, seed=7, metrics_path=metrics),
        budget=free_budget(),
    )
    rows = tl.MetricsWriter.read(metrics)
    assert [row["phase"] for row in rows] == [tl.PHASE_STABLE] * 3


def test_l12_cli_decay_flags_and_paths(tmp_path: Path):
    """CLI принимает decay-флаги, не ломая прежние, и резолвит путь шарда Q."""
    cli = pretrain_cli()
    args = cli.parse_args(["--shard-root", str(tmp_path / "ds"), "--decay-stream", "Q"])
    assert args.decay_stream == "Q"
    assert args.decay_seed is None
    paths = cli.resolve_paths(args)
    assert paths["decay_shard_root"] == tmp_path / "ds"

    off = cli.parse_args([])
    assert off.decay_stream is None, "decay-фаза не должна включаться молча"
    assert isinstance(cli.resolve_paths(off)["decay_shard_root"], Path)


def test_l12_resume_in_decay_requires_decay_stream(corpus_with_q: Path):
    """Resume decay-курсора без объявленного Q-шарда — отказ, а не потеря потока."""
    loader = make_q_loader(corpus_with_q, decay_start=2)
    for _ in range(3):
        next(loader)
    cursor = loader.cursor(step=3)
    with pytest.raises(tl.PretrainDataError):
        make_loader(corpus_with_q, cursor=cursor)


# ---------------------------------------------------------------------------
# К3 — бюджетные счётчики накопительны по прогону (resume не обнуляет)
# ---------------------------------------------------------------------------


def test_k3_resume_accumulates_tokens_and_gpu_hours(tmp_path: Path):
    """Resume складывает tokens_seen/gpu_hours ног, а не начинает с нуля.

    Без этого пороги сметы ($225/$260) недостижимы: каждая нога выходила «в
    пределах лимита», spend считался только от последней ноги.
    """
    cfg, pool = parity_setup(steps=4)
    first = tl.train(
        cfg,
        batched(pool[:2]),
        train_config=tl.TrainConfig(
            steps=2,
            total_steps=4,
            lr=1e-2,
            seed=7,
            ckpt_dir=tmp_path / "ckpt",
            checkpoint_every=2,
            metrics_path=tmp_path / "metrics.jsonl",
        ),
        budget=free_budget(),
    )
    assert first.tokens_seen > 0

    manager = tl.CheckpointManager(tmp_path / "ckpt", keep_last=2)
    second = tl.train(
        cfg,
        batched(pool[2:]),
        train_config=tl.TrainConfig(steps=2, total_steps=4, lr=1e-2, seed=7),
        resume_from=manager,
        budget=free_budget(),
    )
    assert second.tokens_seen == first.tokens_seen * 2, "tokens_seen не накопился"
    assert second.budget_report["tokens_seen_resumed_base"] == first.tokens_seen
    assert second.budget_report["gpu_hours_actual"] >= second.budget_report["gpu_hours_resumed_base"]


# ---------------------------------------------------------------------------
# К5 — stop-файл останавливает луп (килл-свитч ватчдога)
# ---------------------------------------------------------------------------


def test_k5_stop_file_halts_the_loop(tmp_path: Path):
    """Существование stop-файла останавливает прогон с причиной в результате."""
    cfg, pool = parity_setup(steps=5)
    stop_file = tmp_path / "stop"
    stop_file.write_text("spend $225 >= cap $225", encoding="utf-8")
    result = tl.train(
        cfg,
        batched(pool),
        train_config=tl.TrainConfig(steps=5, seed=7, stop_file=stop_file),
        budget=free_budget(),
    )
    assert result.steps_done == 1, "луп не встал на первом же шаге"
    assert result.stop_reason is not None and result.stop_reason.startswith("stop-file")
    assert result.stopped_by_budget is True
    assert "spend $225" in result.stop_reason


def test_k5_absent_stop_file_does_not_stop(tmp_path: Path):
    """Пока stop-файла нет, луп идёт по плану (стоп-файл не выдумывается)."""
    cfg, pool = parity_setup(steps=2)
    result = tl.train(
        cfg,
        batched(pool),
        train_config=tl.TrainConfig(steps=2, seed=7, stop_file=tmp_path / "нет-файла"),
        budget=free_budget(),
    )
    assert result.steps_done == 2 and result.stop_reason is None


# ---------------------------------------------------------------------------
# H1 — decay-окно против объёма Q
# ---------------------------------------------------------------------------


def test_h1_decay_window_fits_inside_q():
    """Окно decay внутри Q — доля сохраняется."""
    plan = tl.decay_window_plan(
        total_steps=100, decay_ratio=0.05, batch_size=1, seq_len=8192,
        available_tokens=5 * 8192,
    )
    assert plan.adjusted is False
    assert plan.decay_steps == 5
    assert plan.decay_ratio == 0.05


def test_h1_decay_window_reduced_when_q_smaller():
    """Q меньше окна — доля уменьшается до влезающей (граница: ровно 3 шага)."""
    plan = tl.decay_window_plan(
        total_steps=100, decay_ratio=0.05, batch_size=1, seq_len=8192,
        available_tokens=3 * 8192,
    )
    assert plan.adjusted is True
    assert plan.decay_steps == 3
    assert plan.decay_ratio == pytest.approx(0.03)
    assert plan.window_tokens <= plan.available_tokens


def test_h1_exact_fit_is_not_adjusted():
    """Ровно влезает (window == Q) — не уменьшаем: граница включительна."""
    plan = tl.decay_window_plan(
        total_steps=100, decay_ratio=0.05, batch_size=1, seq_len=8192,
        available_tokens=5 * 8192,
    )
    assert plan.adjusted is False and plan.decay_steps == 5


def test_h1_unknown_q_is_not_assumed_to_fit():
    """Неизвестный объём Q — проверка не выполняется, доля не выдумывается."""
    plan = tl.decay_window_plan(
        total_steps=100, decay_ratio=0.05, batch_size=1, seq_len=8192,
        available_tokens=None,
    )
    assert plan.adjusted is False
    assert "неизвестен" in plan.reason


# ---------------------------------------------------------------------------
# H4 — resume пиннит seed/ratios/data_kind
# ---------------------------------------------------------------------------


def test_h4_resume_seed_mismatch_is_refused(tmp_path: Path, corpus: Path):
    """Продолжение другим сидом — отказ, а не молча другая траектория."""
    cli = pretrain_cli()
    params, state = tiny_params()
    manager = tl.CheckpointManager(tmp_path / "ckpt", keep_last=1)
    manager.save(
        step=2,
        params=params,
        optimizer_state=state,
        cursor={
            "schema": tl.CURSOR_SCHEMA,
            "step": 2,
            "tokens_total": 0,
            "streams": [],
            "pending_tokens": [],
            "phase": tl.PHASE_STABLE,
            "run": {
                "seed": 4242,
                "total_steps": 4,
                "warmup_ratio": 0.01,
                "decay_ratio": 0.05,
                "data_kind": "raw",
            },
        },
    )
    out_dir = tmp_path / "out"
    code = cli.main(
        [
            "--run-ref", "pretrain-h4",
            "--resume",
            "--seed", "1337",
            "--shard-root", str(corpus),
            "--out", str(out_dir),
            "--ckpt-dir", str(tmp_path / "ckpt"),
            "--steps", "1",
            "--budget-limit-usd", "5",
        ]
    )
    assert code == 1
    journal = json.loads((out_dir / "journal.json").read_text(encoding="utf-8"))
    assert "seed" in journal["refusal"], journal["refusal"]


def test_h4_validate_resume_pins_matches_and_mismatches():
    """Сверка пинов: совпадение молчит, расхождение называет поле."""
    run = {"seed": 7, "warmup_ratio": 0.01, "decay_ratio": 0.05, "data_kind": "raw"}
    assert cli_validate(run, seed=7, warmup=0.01, decay=0.05, kind="raw") == []
    bad = cli_validate(run, seed=7, warmup=0.02, decay=0.05, kind="raw")
    assert bad and "warmup_ratio" in bad[0]


def cli_validate(run, *, seed, warmup, decay, kind):
    cli = pretrain_cli()
    return cli.validate_resume_pins(
        run, seed=seed, warmup_ratio=warmup, decay_ratio=decay, data_kind=kind, enabled=True
    )


# ---------------------------------------------------------------------------
# H2/H3/H5/H6 — точечные дефекты данных и следов
# ---------------------------------------------------------------------------


def test_h2_pick_stream_renormalizes_over_live(tmp_path: Path):
    """При исчерпании потока оставшиеся миксуются по перенормированным долям (H2).

    Сценарий, где старое правило (вес и счётчик мёртвого потока остаются в расчёте)
    выбирает соло-поток B, а перенормировка — C: A израсходован (weight 0.8,
    1000 токенов), B — 100, C — 0.  Старое: deficitB = 0.15·1100 − 100 = 65 >
    deficitC = 0.05·1100 = 55 → B (соло).  Новое: live-доли 0.75/0.25 → deficitC =
    25 > deficitB = −25 → C.
    """
    root = tmp_path / "ds3"
    write_shard_set(root, "A", synthetic_docs("a", 40, 8))
    write_shard_set(root, "B", synthetic_docs("b", 40, 8))
    write_shard_set(root, "C", synthetic_docs("c", 40, 8))
    loader = tl.PretrainMixLoader(
        shard_root=root,
        streams=("A", "B", "C"),
        encode=word_encoder,
        seq_len=16,
        batch_size=1,
        seed=3,
        mix={"A": 0.8, "B": 0.15, "C": 0.05},
        shuffle_window=8,
    )
    loader._streams["A"].exhausted = True
    loader._streams["A"].tokens = 1000
    loader._streams["B"].tokens = 100
    loader._streams["C"].tokens = 0
    assert loader._pick_stream() == "C", "дефицит считается по мёртвому потоку (соло-хвост)"


def test_h3_pack_batch_does_not_insert_fake_eos():
    """Упаковка не вставляет EOS на разрезе записи и не теряет токены (H3)."""
    tokens = list(range(10, 10 + 2 * 3))  # две записи по T-1=3
    batch = tl.pack_batch(tokens, 4)
    assert batch.shape == (2, 4)
    assert np.all(batch[:, 0] == tl.BOS_ID)
    assert batch[:, 1:].reshape(-1).tolist() == tokens, "поток искажён упаковкой"
    assert tl.EOS_ID not in batch[:, 1:].tolist(), "фальшивый EOS на разрезе записи"


def test_h3_documents_are_joined_by_eos_without_breaks(corpus: Path):
    """Документ не разрезается фальшивым EOS: между EOS — ровно один документ."""
    loader = make_loader(corpus, streams=("W",), mix={"W": 1.0}, seq_len=8, batch_size=1)
    stream: list[int] = []
    for batch in loader:
        for row in batch:
            stream.extend(int(token) for token in row[1:])
    segments: list[list[int]] = []
    current: list[int] = []
    for token in stream:
        if token == tl.EOS_ID:
            segments.append(current)
            current = []
        else:
            current.append(token)
    expected = [word_encoder(text) for text in synthetic_docs("w", 60, 12)]
    assert segments, "поток не содержит закрытых документов"
    for index, segment in enumerate(segments):
        assert segment in expected, f"сегмент {index} — не целый документ (документ разрезан)"


def test_h5_metrics_read_skips_broken_lines(tmp_path: Path):
    """Оборванная строка metrics.jsonl не роняет чтение всего журнала (H5)."""
    path = tmp_path / "metrics.jsonl"
    path.write_text(
        json.dumps({"schema": tl.METRICS_SCHEMA, "step": 1, "loss": 9.0}) + "\n"
        + '{"schema": "pretrain-metrics/v1", "step": 2, "lo'  # обрыв преемпшна
        + "\n"
        + json.dumps({"schema": tl.METRICS_SCHEMA, "step": 3, "loss": 7.0}) + "\n",
        encoding="utf-8",
    )
    rows = tl.MetricsWriter.read(path)
    assert [row["step"] for row in rows] == [1, 3]


def test_h6_blank_lines_do_not_shift_resume_offset(tmp_path: Path):
    """Пустые строки шарда не сдвигают позицию документов при resume (H6)."""
    import zstandard as zstd

    path = tmp_path / "shard-00000.jsonl.zst"
    payload = (
        json.dumps({"text": "doc0"}) + "\n"
        + "\n"
        + json.dumps({"text": "doc1"}) + "\n"
        + "   \n"
        + json.dumps({"text": "doc2"}) + "\n"
    ).encode("utf-8")
    path.write_bytes(zstd.ZstdCompressor(level=3).compress(payload))

    stream = tl.iter_shard_docs(path, skip=1)
    assert json.loads(next(stream))["text"] == "doc1", "skip=1 перескочил документ из-за пустой строки"
    assert json.loads(next(stream))["text"] == "doc2"


def test_k2_resolve_packed_autodetects_manifests(tmp_path: Path):
    """--packed авто-включается, если манифесты tokens/ на месте (К2)."""
    cli = pretrain_cli()
    shard_root = tmp_path / "ds"
    args = cli.parse_args(["--shard-root", str(shard_root)])
    assert cli.resolve_packed(args, shard_root / "tokens", ("W", "C")) is False
    tokens_root = shard_root / "tokens"
    for name in ("W", "C"):
        (tokens_root / name).mkdir(parents=True, exist_ok=True)
        (tokens_root / name / f"manifest-{name.lower()}.json").write_text("{}", encoding="utf-8")
    assert cli.resolve_packed(args, tokens_root, ("W", "C")) is True
    forced = cli.parse_args(["--no-packed"])
    assert cli.resolve_packed(forced, tokens_root, ("W", "C")) is False


def write_packed_set(root: Path, name: str, *, vocab_size: int, tokenizer_hash: str, records: int = 4, seq_len: int = 16):
    """Синтетический packed-набор: .bin + манифест контракта tokens/ (К2)."""
    tokens_root = root / "tokens"
    out_dir = tokens_root / name
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = np.zeros((records, seq_len), dtype=np.uint32)
    rows[:, 0] = tl.BOS_ID
    rows[:, 1:] = np.arange(3, 3 + seq_len - 1, dtype=np.uint32)
    path = out_dir / f"{name}-00000.bin"
    path.write_bytes(rows.tobytes(order="C"))
    manifest = {
        "version": "axiom-pretrain-tokens/1",
        "shard": name,
        "seq_len": seq_len,
        "dtype": "uint32",
        "record_layout": "bos + (seq_len-1) токенов потока",
        "bos_id": tl.BOS_ID,
        "eos_id": tl.EOS_ID,
        "pad_id": tl.PAD_ID,
        "tokenizer_hash": tokenizer_hash,
        "tokenizer": {"file": "tokenizer.model", "vocab_size": vocab_size, "tokenizer_hash": tokenizer_hash},
        "shards": [
            {
                "file": path.name,
                "source": f"{name}-00000.jsonl.zst",
                "source_sha256": "0" * 64,
                "records": records,
                "tokens": records * seq_len,
                "stream_tokens": records * (seq_len - 1),
                "pad_tokens": 0,
                "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        ],
    }
    (out_dir / f"manifest-{name.lower()}.json").write_text(
        json.dumps(manifest, ensure_ascii=False), encoding="utf-8"
    )
    return tokens_root


def test_k2_packed_tokenizer_pin_reads_manifest_vocab(tmp_path: Path):
    """Vocab модели берётся из манифеста packed (id в .bin покрыты), хеш сверяется (К2)."""
    cli = pretrain_cli()
    tokens_root = write_packed_set(tmp_path / "ds", "W", vocab_size=160000, tokenizer_hash="a" * 64)
    write_packed_set(tmp_path / "ds", "C", vocab_size=160000, tokenizer_hash="a" * 64)
    pin = cli._packed_tokenizer_pin(tl, tokens_root, ("W", "C"), None)
    assert pin["vocab_size"] == 160000
    assert pin["hash"] == "a" * 64


def test_k2_packed_tokenizer_pin_refuses_mixed_tokenizers(tmp_path: Path):
    """Разные токенизаторы в потоках — отказ контракта данных, не молчаливый микс."""
    cli = pretrain_cli()
    tokens_root = write_packed_set(tmp_path / "ds", "W", vocab_size=160000, tokenizer_hash="a" * 64)
    write_packed_set(tmp_path / "ds", "C", vocab_size=160000, tokenizer_hash="b" * 64)
    with pytest.raises(tl.PretrainDataError):
        cli._packed_tokenizer_pin(tl, tokens_root, ("W", "C"), None)
