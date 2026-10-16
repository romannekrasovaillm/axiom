"""ADR-054 Am.21 (цикл-12, веха-10) — асинхронные метрики шага (CPU).

Перф-зонд цикла-8 показал futex/absl-цепочки вокруг ``jax
BlockUntilReadyWithCancel ← PyArray BatchedCopyToHostAsyncHelper``: ``device_get``
loss на КАЖДОМ шаге — синхронная копия на хост, ~25% сэмплов.  На host-bound
шаге (Am.15) это чистый минус.  Дельта: значения loss окна собираются на
устройстве, хост получает их одним ``device_get`` раз в K шагов (K =
``log_every`` или 10; последний шаг ноги — принудительный сброс).  Строки
``pretrain-metrics/v1`` формируются каждый шаг и пишутся на границах сброса с
маркером ``async_metrics``.  Гейт ``AXIOM_ASYNC_METRICS`` (дефолт ``0``).

Что пинуется здесь (CPU, без GPU):

* **(а) гейт-off побитово прежний** — прогон ``train`` на текущей ревизии (гейт
  не задан) против прогона ``net/`` базовой ревизии ``BASELINE_COMMIT`` (архив
  ``git archive``): loss-репризы, tree_hash и строки метрик (шаг, loss) — точно;
  плюс семантика гейта (дефолт/нормализация/fail-closed) и маркер строк
  ``async_metrics=0``;
* **(б) паритет с синхронным путём при нулевом допуске** — gate-on против
  gate-off на одинаковых данных: ``repr`` loss-ов совпадают ПОСТРОЧНО, строки
  метрик (шаг, loss) тоже (те же числа, реже — копия едет раз в K шагов), а
  число ``device_get``-копий loss падает с «каждый шаг» (и «на микробатч» в
  Python-ветке накопления) до «на границу сброса»; три пути шага: одиночный
  микробатч, Python-ветка накопления, jitted-аккумуляция (Am.16);
* **(в) nan ловится на границе сброса** — симулированный nan в окне (маркерный
  токен в данных при патче ``compute_loss``): gate-on падает
  :class:`tl.PretrainNanError` на ПЕРВОЙ границе сброса, накрывающей nan-шаг,
  уже ПОСЛЕ записи строк окна (nan-значения в журнале — разбор не потерян);
  gate-off (без гейта) nan не роняет — синхронный путь не слабее прежнего, а
  гейт добавляет контроль;
* **(г) хвост и каденс сброса** — (i) при ``log_every=0`` K=10: один сброс на
  выходе ноги, строки и loss-траектория полны; (ii) кончился поток батчей
  (``break`` до конца ноги): остаток окна сбрасывается принудительно — ни строка,
  ни loss-значение не теряются.

Не пинуется здесь: фактические ток/с и число хост-синхронизаций на GB10 (замер
архитектора, цикл-12 A/B), полный регресс ``net/tests`` на CPU (существующие
дефекты окружения без ``~/gb10-shared`` — вне дельты).
"""

from __future__ import annotations

import contextlib
import io
import json
import math
import os
import subprocess
import sys
import tarfile
from pathlib import Path

import jax
import numpy as np
import pytest

from conftest import tiny_config

from net import train_loop as tl

#: Геометрия смоука: одна последовательность, короткий контекст (CPU-время).
SEQ_LEN = 16
#: Цель накопления — два микробатча на шаг (tokens_per_micro = 1 * SEQ_LEN).
ACCUM2 = 2 * SEQ_LEN

#: Базовая ревизия (ADR-054 Am.21 — до дельты асинхронных метрик): гейт-off
#: обязан воспроизводить её побитово.  ``git archive`` + skip, если ревизия
#: недоступна (форма test_36_accum_jit).
BASELINE_COMMIT = "9527c64"
NET_DIR = Path(__file__).resolve().parents[1]
CASE_DIR = NET_DIR.parent

#: Маркерный токен nan-впрыска: валидный id (последний), которого НЕТ в
#: случайных данных (генератор ниже держит диапазон ``[0, marker)``).
NAN_TOKEN = 63


# ---------------------------------------------------------------------------
# Общие помощники
# ---------------------------------------------------------------------------


def _batches(cfg, count: int, *, seed: int = 5):
    """Детерминированный поток ``(1, T) int32`` — одинаковые данные для ног."""
    rng = np.random.default_rng(seed)
    return [
        rng.integers(0, cfg.vocab_size, size=(1, SEQ_LEN), dtype=np.int32)
        for _ in range(count)
    ]


def _batches_with_marker(cfg, count: int, *, at: int, seed: int = 5):
    """Поток без маркерного токена в случайных данных + маркер ровно на ``at``."""
    rng = np.random.default_rng(seed)
    batches = [
        rng.integers(0, NAN_TOKEN, size=(1, SEQ_LEN), dtype=np.int32)
        for _ in range(count)
    ]
    batches[at] = batches[at].copy()
    batches[at][0, 0] = NAN_TOKEN
    return batches


def _budget():
    return tl.Budget(run_ref="async-metrics", path=Path("/dev/null"), present=True)


def _train_leg(
    cfg,
    *,
    steps: int,
    accum: int = 0,
    log_every: int = 3,
    metrics_path=None,
    batches=None,
    seed: int = 7,
    lr: float = 1e-2,
):
    """Нога ``train`` без управления env (гейты выставляют контексты)."""
    return tl.train(
        cfg,
        iter(batches if batches is not None else _batches(cfg, steps * 2)),
        train_config=tl.TrainConfig(
            steps=steps,
            lr=lr,
            seed=seed,
            schedule="cosine",
            param_dtype="float32",
            accum_tokens=accum,
            micro_batch=1,
            metrics_path=metrics_path,
            log_every=log_every,
            kpi_every=20,
        ),
        budget=_budget(),
    )


@contextlib.contextmanager
def gate_env(value: str | None):
    """Временное значение ``AXIOM_ASYNC_METRICS`` (``None`` — переменную снять)."""
    saved = os.environ.get(tl.ASYNC_METRICS_ENV)
    try:
        if value is None:
            os.environ.pop(tl.ASYNC_METRICS_ENV, None)
        else:
            os.environ[tl.ASYNC_METRICS_ENV] = value
        yield
    finally:
        if saved is None:
            os.environ.pop(tl.ASYNC_METRICS_ENV, None)
        else:
            os.environ[tl.ASYNC_METRICS_ENV] = saved


@contextlib.contextmanager
def accum_env(value: str | None):
    """Временное значение ``AXIOM_ACCUM_JIT`` (для jitted-пути накопления)."""
    saved = os.environ.get(tl.ACCUM_JIT_ENV)
    try:
        if value is None:
            os.environ.pop(tl.ACCUM_JIT_ENV, None)
        else:
            os.environ[tl.ACCUM_JIT_ENV] = value
        yield
    finally:
        if saved is None:
            os.environ.pop(tl.ACCUM_JIT_ENV, None)
        else:
            os.environ[tl.ACCUM_JIT_ENV] = saved


@contextlib.contextmanager
def count_loss_copies():
    """Счётчик ``device_get``, вызванных ИЗ ``net/train_loop.py`` (кадр — файл).

    Глобальный патч ``jax.device_get`` с фильтром по кадру вызова: копии loss
    (синхронный путь — кадр ``train``, асинхронный — кадр
    ``_flush_metrics_window``) считаются, а чужие копии (``checkpoint.tree_hash``
    и прочая периферия) — нет.
    """
    real = jax.device_get
    calls: list[str] = []

    def wrapper(x, *args, **kwargs):
        frame = sys._getframe(1)
        if os.path.abspath(frame.f_code.co_filename) == os.path.abspath(tl.__file__):
            calls.append(frame.f_code.co_name)
        return real(x, *args, **kwargs)

    jax.device_get = wrapper
    try:
        yield calls
    finally:
        jax.device_get = real


def _rows(metrics_path) -> list[dict]:
    return tl.MetricsWriter.read(metrics_path)


def _loss_signature(result) -> list[str]:
    return [repr(x) for x in result.losses]


def _row_signature(rows: list[dict]) -> list[tuple[int, str]]:
    return [(int(row["step"]), repr(row["loss"])) for row in rows]


#: Поля строки метрик, детерминированные по шагу (не зависят от таймингов):
#: паритет gate-on/gate-off обязан держать их равными.
_DETERMINISTIC_FIELDS = (
    "step",
    "lr",
    "phase",
    "tokens",
    "tokens_seen",
    "microbatches",
    "accum_jit",
    "optimizer_classification",
    "master_dtype",
    "master_comp",
    "grad_ckpt_policy",
    "mfu_params_only",
    "mla_bwd_bf16",
    "ce_bf16",
)


# ---------------------------------------------------------------------------
# Гейт: семантика значения (дефолт, значения, fail-closed)
# ---------------------------------------------------------------------------


def test_gate_default_is_off_and_values_normalize():
    with gate_env(None):
        assert tl.async_metrics_mode() is False, "дефолт — прежний синхронный путь"
    with gate_env("1"):
        assert tl.async_metrics_mode() is True
    with gate_env(" 0 "):
        assert tl.async_metrics_mode() is False, "пробелы нормализуются"
    with gate_env(""):
        assert tl.async_metrics_mode() is False, "пустая переменная = не задана"
    assert tl.async_metrics_mode("1") is True, "явный аргумент сильнее окружения"
    assert tl.async_metrics_mode("0") is False
    assert tl.ASYNC_METRICS_DEFAULT_EVERY == 10, "K без log_every — десять шагов"


def test_gate_unknown_value_fails_closed():
    """Опечатка — ошибка, не тихий дефолт: тихая нога мерила бы не тот путь."""
    for bad in ("true", "yes", "on", "2", "01", "async"):
        with gate_env(bad):
            with pytest.raises(tl.AsyncMetricsError, match=tl.ASYNC_METRICS_ENV):
                tl.async_metrics_mode()


def test_train_refuses_unknown_gate_value():
    """Та же fail-closed дверь на сборке цикла (до первой компиляции)."""
    cfg = tiny_config()
    with gate_env("async"):
        with pytest.raises(tl.AsyncMetricsError, match=tl.ASYNC_METRICS_ENV):
            _train_leg(cfg, steps=1)


# ---------------------------------------------------------------------------
# (а) гейт-off: побитово прежний путь — базовая ревизия + маркер строк
# ---------------------------------------------------------------------------

#: Проба гейт-off: те же данные/шаги, печатает JSON последней строкой.
#: Запускается (i) в подпроцессе поверх архива базовой ревизии и (ii) на
#: текущей рабочей ревизии; ``conftest`` импортируется ДО jax — пины ADR-010
#: действуют на процесс только до создания jax-клиента.
_PROBE = r'''
import json, sys
sys.path.insert(0, "net/tests")
import conftest  # noqa: F401 — пины ADR-010 до import jax
import numpy as np
from conftest import tiny_config
from net import checkpoint, train_loop as tl

metrics_path = sys.argv[1]
cfg = tiny_config()
rng = np.random.default_rng(5)
batches = [rng.integers(0, cfg.vocab_size, size=(1, 16), dtype=np.int32) for _ in range(4)]
result = tl.train(
    cfg, iter(batches),
    train_config=tl.TrainConfig(steps=2, lr=1e-2, seed=7, schedule="cosine",
                                param_dtype="float32", accum_tokens=32,
                                micro_batch=1, log_every=0, metrics_path=metrics_path),
    budget=tl.Budget(run_ref="gate-off", path=__import__("pathlib").Path("/dev/null"),
                     present=True),
)
rows = tl.MetricsWriter.read(metrics_path)
print(json.dumps({
    "losses": [repr(x) for x in result.losses],
    "params_hash": checkpoint.tree_hash(result.params),
    "master_hash": checkpoint.tree_hash(result.master_params),
    "steps_done": result.steps_done,
    "rows": [[row["step"], repr(row["loss"])] for row in rows],
}))
'''


def _run_probe(cwd: Path, metrics_path: Path, *, script: Path | None = None) -> dict:
    """Проба в подпроцессе: env без гейтов — дефолтное поведение.

    ``script`` — файл пробы (архив базовой ревизии); без него тот же исходник
    исполняется через ``-c`` (``sys.path[0] = cwd``), чтобы не писать файлов
    в рабочее дерево текущей ревизии.
    """
    env = dict(os.environ)
    for key in (tl.ASYNC_METRICS_ENV, tl.ACCUM_JIT_ENV, "AXIOM_MASTER_DTYPE",
                "AXIOM_MASTER_COMP"):
        env.pop(key, None)
    command = (
        [sys.executable, str(script), str(metrics_path)]
        if script is not None
        else [sys.executable, "-c", _PROBE, str(metrics_path)]
    )
    done = subprocess.run(
        command, cwd=cwd, env=env, capture_output=True, text=True
    )
    if done.returncode != 0:  # pragma: no cover — диагностика прогона
        raise AssertionError(f"проба упала ({cwd}):\n{done.stderr[-2000:]}")
    lines = [line for line in done.stdout.strip().splitlines() if line.strip()]
    return json.loads(lines[-1])


def _extract_baseline(tmp_path: Path) -> Path:
    """``net/`` базовой ревизии + проба рядом (cwd=root: ``from net import``)."""
    root = tmp_path / "baseline"
    root.mkdir()
    try:
        archive = subprocess.run(
            ["git", "-C", str(CASE_DIR), "archive", BASELINE_COMMIT, "net/"],
            check=True,
            capture_output=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        pytest.skip(f"baseline {BASELINE_COMMIT} недоступен: {exc}")
    with tarfile.open(fileobj=io.BytesIO(archive.stdout)) as bundle:
        bundle.extractall(root)
    (root / "probe.py").write_text(_PROBE, encoding="utf-8")
    return root


def test_a_gate_off_is_bit_identical_to_baseline(tmp_path):
    """Гейт-off (дефолт) воспроизводит базовую ревизию побитово.

    Сравниваются loss-репризы, tree_hash деревьев и строки метрик (шаг, loss) —
    точно: любая новая арифметика на дефолтном пути видна строкой, а не «в
    допуске».  Проба гоняет 2 шага по 2 микробатча — именно Python-ветку
    накопления, которую async-дельта трогает больше всего (per-micro копии).
    """
    baseline = _extract_baseline(tmp_path)
    base_out = _run_probe(baseline, tmp_path / "base-metrics.jsonl",
                          script=baseline / "probe.py")
    current_out = _run_probe(CASE_DIR, tmp_path / "cur-metrics.jsonl")
    assert current_out == base_out, (
        "гейт-off разошёлся с базовой ревизией: "
        f"current={current_out!r} baseline={base_out!r}"
    )


def test_a_gate_off_marks_rows_and_keeps_per_step_copies(tmp_path):
    """Структурный пин дефолта: маркер ``async_metrics=0`` и копия на КАЖДОМ шаге.

    Число ``device_get``-копий loss в гейт-off равно числу шагов (single) и
    числу микробатчей (Python-ветка) — дельта не включена.
    """
    cfg = tiny_config()
    metrics = tmp_path / "off.jsonl"
    with gate_env(None), count_loss_copies() as calls:
        result = _train_leg(cfg, steps=2, accum=ACCUM2, metrics_path=metrics)
    assert calls == ["train"] * 4, f"Python-ветка: 2 микробатча × 2 шага, не {calls!r}"
    rows = _rows(metrics)
    assert [row["async_metrics"] for row in rows] == [0, 0]
    assert _row_signature(rows) == [(1, repr(result.losses[0])), (2, repr(result.losses[1]))]

    metrics_single = tmp_path / "off-single.jsonl"
    with gate_env(None), count_loss_copies() as calls:
        _train_leg(cfg, steps=3, accum=0, metrics_path=metrics_single)
    assert calls == ["train"] * 3, f"одиночная ветка: копия на шаг, не {calls!r}"


# ---------------------------------------------------------------------------
# (б) паритет с синхронным путём при нулевом допуске + каденс копий
# ---------------------------------------------------------------------------

#: Пути шага: (имя, accum_tokens, AXIOM_ACCUM_JIT, ожидаемых копий за 6 шагов
#: при gate-off).  Копии: single — 1/шаг; Python-ветка — 2/шаг (по микробатчу);
#: jitted-аккумуляция — 1/шаг (loss возвращает один jit).
_STEP_PATHS = (
    ("single", 0, None, 6),
    ("python-accum2", ACCUM2, None, 12),
    ("jitted-accum2", ACCUM2, "1", 6),
)


@pytest.mark.parametrize("name,accum,accum_gate,off_copies", _STEP_PATHS)
def test_b_gate_on_equals_sync_with_zero_tolerance(
    name, accum, accum_gate, off_copies, tmp_path
):
    """gate-on == gate-off точно (``repr``), копий в K раз меньше.

    Строки метрик: состав совпадает, значения loss — те же числа (допуск нуля
    не «в пределах», а равенство ``repr``: значения не пересчитываются, только
    копия едет реже).  Каденс: 6 шагов при K=3 — ровно два сброса.
    """
    cfg = tiny_config()
    steps, log_every = 6, 3
    batches = _batches(cfg, steps * 2)
    metrics_off = tmp_path / f"{name}-off.jsonl"
    metrics_on = tmp_path / f"{name}-on.jsonl"

    with accum_env(accum_gate), gate_env(None), count_loss_copies() as off_calls:
        off = _train_leg(cfg, steps=steps, accum=accum, log_every=log_every,
                         metrics_path=metrics_off, batches=list(batches))
    with accum_env(accum_gate), gate_env("1"), count_loss_copies() as on_calls:
        on = _train_leg(cfg, steps=steps, accum=accum, log_every=log_every,
                        metrics_path=metrics_on, batches=list(batches))

    off_rows, on_rows = _rows(metrics_off), _rows(metrics_on)
    assert len(off_rows) == len(on_rows) == steps, "строка на каждый шаг"
    assert _loss_signature(off) == _loss_signature(on), (
        f"loss-траектории разошлись ({name}): "
        f"off={_loss_signature(off)} on={_loss_signature(on)}"
    )
    assert _row_signature(off_rows) == _row_signature(on_rows), (
        f"строки метрик разошлись ({name})"
    )
    # Детерминированные поля строк совпадают; loss уже сравнён точным repr.
    for off_row, on_row in zip(off_rows, on_rows):
        assert {key: off_row[key] for key in _DETERMINISTIC_FIELDS} == {
            key: on_row[key] for key in _DETERMINISTIC_FIELDS
        }
        assert off_row["async_metrics"] == 0 and on_row["async_metrics"] == 1
    # Каденс копий: синхронный путь — как объявлено, асинхронный — сбросы
    # (6 шагов, K=3: сбросы на 3 и 6).
    assert off_calls == ["train"] * off_copies, f"off-копии ({name}): {off_calls!r}"
    assert on_calls == ["_flush_metrics_window"] * 2, f"on-копии ({name}): {on_calls!r}"


def test_b_default_k_is_ten_and_last_step_forces_flush(tmp_path):
    """Без ``log_every`` K=10: окно не закрывается в цикле — сброс на выходе ноги.

    Четыре шага < K: одна копия на всю ногу, но строки и loss-траектория полны
    (принудительный сброс последнего шага), и числа совпадают с синхронным путём
    точно.
    """
    cfg = tiny_config()
    metrics_off = tmp_path / "k10-off.jsonl"
    metrics_on = tmp_path / "k10-on.jsonl"
    with gate_env(None):
        off = _train_leg(cfg, steps=4, accum=0, log_every=0, metrics_path=metrics_off)
    with gate_env("1"), count_loss_copies() as on_calls:
        on = _train_leg(cfg, steps=4, accum=0, log_every=0, metrics_path=metrics_on)

    assert on_calls == ["_flush_metrics_window"], f"K=10, 4 шага: {on_calls!r}"
    on_rows = _rows(metrics_on)
    assert len(on_rows) == 4
    assert _loss_signature(off) == _loss_signature(on)
    assert _row_signature(_rows(metrics_off)) == _row_signature(on_rows)


def test_b_async_without_metrics_keeps_loss_trajectory(tmp_path):
    """Без журнала (``metrics=None``) траектория loss всё равно полна и точна.

    Манифест (``TrainResult.losses``) собирается из сбросов окна — гейт не
    зависит от наличия ``metrics.jsonl``; при этом ни одной строки не пишется.
    """
    cfg = tiny_config()
    with gate_env(None):
        off = _train_leg(cfg, steps=5, accum=0, log_every=2)
    with gate_env("1"):
        on = _train_leg(cfg, steps=5, accum=0, log_every=2)
    assert _loss_signature(off) == _loss_signature(on)
    assert on.metrics_path is None
    assert on.steps_done == 5


# ---------------------------------------------------------------------------
# (в) nan в окне ловится на границе сброса
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def nan_injection(monkeypatch, *, token: int = NAN_TOKEN):
    """Патч ``model.compute_loss``: nan-loss на батчах с маркерным токеном.

    Внутри jit ``jnp.where`` остаётся data-dependent узлом графа — nan
    появляется ровно на том шаге, чей батч несёт маркер (данные без маркера
    генерирует :func:`_batches_with_marker`).
    """
    from net import model as model_mod

    real = model_mod.compute_loss
    import jax.numpy as jnp

    def patched(params, cfg, batch, **kwargs):
        loss = real(params, cfg, batch, **kwargs)
        hit = jnp.any(jnp.asarray(batch) == token)
        return jnp.where(hit, jnp.full_like(loss, jnp.nan), loss)

    monkeypatch.setattr(model_mod, "compute_loss", patched)
    try:
        yield
    finally:
        monkeypatch.setattr(model_mod, "compute_loss", real)


def test_c_nan_in_window_fails_at_flush_boundary_with_rows_written(monkeypatch, tmp_path):
    """nan на шаге 5 (окно 4..6) роняет ногу сбросом на шаге 6 — не раньше.

    Строки окаймления (4..6) уже записаны: ``nan`` уезжает в журнал как есть —
    разбор не теряется, а прогон не продолжается молча.  Первый сброс (окно
    1..3) проходит чисто и домена не роняет.
    """
    cfg = tiny_config()
    batches = _batches_with_marker(cfg, 12, at=4)
    metrics = tmp_path / "nan-on.jsonl"
    with nan_injection(monkeypatch):
        with gate_env("1"), count_loss_copies() as calls:
            with pytest.raises(tl.PretrainNanError) as excinfo:
                _train_leg(cfg, steps=6, accum=0, log_every=3, metrics_path=metrics,
                           batches=batches)
    message = str(excinfo.value)
    assert "loss шага 5" in message, f"nan-шаг не назван: {message!r}"
    assert "окно 4..6" in message, f"окно не названо: {message!r}"
    assert calls == ["_flush_metrics_window"] * 2, (
        f"падение обязано быть на СБРОСЕ (шаги 3 и 6), а не раньше: {calls!r}"
    )
    rows = _rows(metrics)
    assert [row["step"] for row in rows] == [1, 2, 3, 4, 5, 6], (
        "строки окна с nan обязаны быть записаны до падения"
    )
    assert math.isfinite(rows[0]["loss"]) and math.isfinite(rows[2]["loss"])
    assert math.isnan(rows[4]["loss"]), "nan-значение шага — в журнале как есть"
    assert all(row["async_metrics"] == 1 for row in rows)


def test_c_gate_off_records_nan_without_failing(monkeypatch, tmp_path):
    """Гейт-off nan не роняет: прежний путь не слабее, детектор — свойство гейта.

    Синхронный путь пишет nan-значение шага в строку (и в loss-траекторию) и
    продолжает — ровно как до дельты; ``PretrainNanError`` — добавленный
    контроль async-режима, а не ослабление синхронного.
    """
    cfg = tiny_config()
    batches = _batches_with_marker(cfg, 12, at=4)
    metrics = tmp_path / "nan-off.jsonl"
    with nan_injection(monkeypatch):
        with gate_env(None):
            result = _train_leg(cfg, steps=6, accum=0, log_every=3, metrics_path=metrics,
                                batches=batches)
    assert result.stop_reason is None, "синхронный путь nan'ом не останавливается"
    assert len(result.losses) == 6
    assert math.isnan(result.losses[4])
    rows = _rows(metrics)
    assert _row_signature(rows) == [(step, repr(x)) for step, x in
                                    zip(range(1, 7), result.losses)]
    assert math.isnan(rows[4]["loss"])
    assert all(row["async_metrics"] == 0 for row in rows)


# ---------------------------------------------------------------------------
# (г) хвост: кончился поток батчей — остаток окна сброшен
# ---------------------------------------------------------------------------


def test_g_exhausted_stream_flushes_tail_rows_and_losses(tmp_path):
    """``break`` до конца ноги не теряет хвост: принудительный сброс на выходе.

    Заказано 6 шагов, в потоке 4 батча: нога останавливается «data exhausted»,
    но строки и loss-траектория несут все 4 исполненных шага — те же числа, что
    в синхронной ноге.
    """
    cfg = tiny_config()
    batches = _batches(cfg, 4)
    metrics_off = tmp_path / "tail-off.jsonl"
    metrics_on = tmp_path / "tail-on.jsonl"
    with gate_env(None):
        off = _train_leg(cfg, steps=6, accum=0, log_every=3, metrics_path=metrics_off,
                         batches=list(batches))
    with gate_env("1"), count_loss_copies() as on_calls:
        on = _train_leg(cfg, steps=6, accum=0, log_every=3, metrics_path=metrics_on,
                        batches=list(batches))

    assert off.steps_done == on.steps_done == 4
    assert on.stop_reason and on.stop_reason.startswith("data exhausted")
    assert _loss_signature(off) == _loss_signature(on)
    assert _row_signature(_rows(metrics_off)) == _row_signature(_rows(metrics_on))
    assert [row["step"] for row in _rows(metrics_on)] == [1, 2, 3, 4]
    # Сброс на 3-м шаге (K=3) и принудительный — на выходе (шаг 4).
    assert on_calls == ["_flush_metrics_window"] * 2
