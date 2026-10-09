"""Претрейн-луп скелета L3 (V-4) — потоковый даталоадер, цикл, resume, стоп.

Модуль реализует главный кодовый блок до старта 20B: прогон претрейна по
шард-файлам W/C (ADR-021) существующим скелетом сети (``net/model.py``,
``net/optimizer.py``, ``net/checkpoint.py``).  Разделение ответственности:

* **данные** — :class:`PretrainMixLoader`: потоковое чтение ``*.jsonl.zst``
  шардов W/C по манифестам, микс ~85/15 по токенам (ADR-021), детерминированный
  шаффл по сиду, токенизация и упаковка в T-последовательности.  Корпус в
  память не поднимается: живое окно — ``shuffle_window`` документов плюс
  недобранные токены одной последовательности.  На последних ``decay_ratio``
  шагах (граница — :func:`decay_start_step`) микс замещается decay-шардом Q тем
  же читателем;
* **курсор** — :class:`MixCursor`: манифест-курсор (шаг, документы, токены,
  остаток токенов).  Resume из него даёт **тот же** поток, что продолжил бы
  непрерывный прогон: без потерь и дублей (проверяется тестом L-6);
* **цикл** — :func:`train`: loss и оптимизатор берутся из ``net/`` без правок,
  bf16-параметры с fp32-мастером, опциональный grad-checkpointing (обязателен
  для ``l3-full`` — урок OOM 956 ГиБ), WSD-расписание (warmup-stable-decay);
* **след** — метрики jsonl (loss / ток-в-с / MFU) и журнал остановки.

Что модуль **не** делает: не пишет в ``evidence/``, не трогает конфиг сети и не
подменяет стоп-правило AD-8 — смета читается из ``evidence/budget/<run_ref>.json``
(или из явно указанного файла) и её отсутствие блокирует запуск.

Отдельно: **явно объявленный бэкенд** (``NET_JAX_BACKEND=cpu|gpu``, ADR-010)
держится уже на импорте этого модуля — см. :func:`apply_declared_backend_pinning`.

Запуск — через CLI ``tools/pretrain_run.py``.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import random
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence

import numpy as np

#: Допустимые имена шардов претрейн-микса (ADR-021: W — веб, C — код).
#: Стабильная фаза: микс W/C.  Decay-шард объявлен отдельно (``DECAY_SHARD``):
#: он не входит в микс, а замещает его на последних ``decay_ratio`` шагах.
SHARD_NAMES = ("W", "C")

#: Шард decay-фазы (ADR-021): сужёный высококачественный микс ~1B токенов.
DECAY_SHARD = "Q"

#: Фазы данных прогона: ``stable`` (микс W/C ~85/15) и ``decay`` (шард Q).
PHASE_STABLE = "stable"
PHASE_DECAY = "decay"
PHASES = (PHASE_STABLE, PHASE_DECAY)

#: Версия контракта курсора (манифест resume).
CURSOR_SCHEMA = "pretrain-cursor/v1"

#: Идентификаторы специальных токенов — те же, что у ``net.data.pack_sequence``.
PAD_ID = 0
BOS_ID = 1
EOS_ID = 2


class PretrainDataError(RuntimeError):
    """Данные претрейна не соответствуют контракту (шард, манифест, курсор)."""


class PretrainBackendError(RuntimeError):
    """Объявленный бэкенд (ADR-010) не применён — прогон не стартует."""


# ---------------------------------------------------------------------------
# 0. Пиннинг бэкенда (ADR-010): до первого импорта jax
# ---------------------------------------------------------------------------

#: Переменная явного выбора бэкенда (ADR-010): ``auto`` (дефолт) | ``gpu`` | ``cpu``.
BACKEND_ENV = "NET_JAX_BACKEND"

#: Значения ``NET_JAX_BACKEND``, означающие **явный** выбор (``auto``/пусто — не выбор).
DECLARED_BACKENDS = ("cpu", "gpu")

#: Единая точка пиннинга (ADR-010) — приёмочный conftest сети: ``JAX_PLATFORMS``,
#: политика точности, при гейтовом профиле — ``XLA_FLAGS`` детерминизма (ADR-013).
ACCEPTANCE_CONFTEST = Path(__file__).resolve().parent / "tests" / "conftest.py"


def declared_backend() -> str | None:
    """Явно объявленный бэкенд или ``None`` (режим ``auto``: выбор за jax)."""
    value = (os.environ.get(BACKEND_ENV) or "").strip().lower()
    return value if value in DECLARED_BACKENDS else None


def _load_acceptance_pinning() -> Any:
    """Загрузить ``net/tests/conftest.py`` — единую точку пиннинга (ADR-010).

    Пиннинг применяется телом модуля: сначала ``select_backend()`` (объявленный
    ``NET_JAX_BACKEND=cpu`` ставит ``JAX_PLATFORMS=cpu``), затем политика
    точности матмулов.  Дублировать это здесь значило бы завести вторую точку
    пиннинга — тот же довод, что у ``run_sft_smoke._load_acceptance_conftest``.
    """
    import importlib.util

    if not ACCEPTANCE_CONFTEST.is_file():  # pragma: no cover — сломанная установка
        raise PretrainBackendError(
            f"объявлен {BACKEND_ENV}={declared_backend()}, но точка пиннинга "
            f"отсутствует: {ACCEPTANCE_CONFTEST} (ADR-010)"
        )
    spec = importlib.util.spec_from_file_location(
        "net_tests_conftest_for_pretrain", ACCEPTANCE_CONFTEST
    )
    if spec is None or spec.loader is None:  # pragma: no cover — сломанная установка
        raise PretrainBackendError(
            f"точка пиннинга не загружается: {ACCEPTANCE_CONFTEST} (ADR-010)"
        )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def apply_declared_backend_pinning() -> dict[str, Any]:
    """Применить явно объявленный бэкенд (ADR-010) до инициализации jax.

    ``JAX_PLATFORMS`` читается jax при **импорте** и дальше не перечитывается:
    выбор, применённый позже, молча теряется (замер: ``JAX_PLATFORMS=cpu``,
    выставленный после ``import jax``, оставляет массивы на ``CudaDevice``).
    Поэтому тест претрейн-лупа пинует себя ``NET_JAX_BACKEND=cpu`` до первого
    импорта jax, а модуль, который тест импортирует первым, обязан пиннинг
    применить — иначе объявленный выбор не держится, и численные проверки
    выполняются не на том бэкенде, на каком объявлены.

    Почему это не косметика: на GPU XLA не гарантирует побитового совпадения
    ядер между **раздельно** скомпилированными программами (ADR-013), поэтому
    требование T-L10 «шаг лупа побитово равен тренеру скелета» на GPU
    выполнимо только под ``--xla_gpu_deterministic_ops`` (замер: без флага два
    раздельно скомпилированных шага расходятся на 1 ULP градиента embedding —
    5,96e-08 на масштабе 0,63; под флагом — ноль расхождений).  На CPU шаг
    точен, и объявленный ``cpu`` — тот бэкенд, где проверка имеет смысл.

    Если jax уже импортирован, ``JAX_PLATFORMS`` переигрывается через
    ``config.jax_platforms`` (действует, пока не создан клиент) и результат
    проверяется: пиннинг, который не применился, не выдаёт себя за применённый
    (ADR-011) — расхождение объявленного и фактического бэкенда даёт warning.

    Возвращается запись для журнала: ``declared``, ``platforms``, ``applied``.
    При ``auto`` (переменная не выставлена) модуль не трогает ни jax, ни
    окружение — возвращается ``{"declared": None, "applied": False}``.
    """
    declared = declared_backend()
    if declared is None:
        return {"declared": None, "platforms": None, "applied": False}

    if "jax" not in sys.modules:
        # Штатный путь (тесты и стейджи): пиннинг применяется телом conftest до
        # первого импорта jax — тогда JAX_PLATFORMS читается уже пиннутым, и
        # заодно применяется политика точности матмулов (ADR-010).
        _load_acceptance_pinning()
        return {"declared": declared, "platforms": [declared], "applied": True}

    # jax импортирован раньше (например, общей сессией тестов): переменную он
    # уже прочитал, поэтому выбор переигрывается конфигом — это действует, пока
    # не создан клиент.  Факт проверяется, а не предполагается (ADR-011):
    # несостоявшийся пиннинг обязан быть виден, а не выглядеть применённым.
    import jax

    jax.config.update("jax_platforms", declared)
    platforms = sorted({device.platform for device in jax.devices()})
    if declared not in platforms:
        import warnings

        warnings.warn(
            f"объявлен {BACKEND_ENV}={declared}, а вычисления идут на {platforms}: "
            "пиннинг бэкенда не применён (ADR-010) — выбор читается jax при "
            "импорте, поэтому net.train_loop должен быть импортирован раньше jax",
            RuntimeWarning,
            stacklevel=2,
        )
        return {"declared": declared, "platforms": platforms, "applied": False}
    return {"declared": declared, "platforms": platforms, "applied": True}


#: Факт пиннинга бэкенда на импорте модуля (ADR-010) — до первого импорта jax.
ACCEPTANCE_PINNING = apply_declared_backend_pinning()


# ---------------------------------------------------------------------------
# 1. Расписание LR: WSD (warmup — stable — decay)
# ---------------------------------------------------------------------------


def decay_start_step(
    total_steps: int, *, warmup_ratio: float = 0.01, decay_ratio: float = 0.05
) -> int:
    """Первый 0-based шаг decay-фазы (граница LR **и** данных).

    По ADR-021 последние ``decay_ratio`` шагов прогона идут по decay-шарду Q,
    и граница данных обязана совпасть с границей LR: иначе Q кормился бы на
    стабильном LR (или наоборот), и «decay-фаза» была бы объявлением, а не
    фактом.  Поэтому и :func:`wsd_schedule`, и даталоадер читают **одну** формулу
    ``total_steps - round(total_steps · decay_ratio)`` с клампом по warmup, а не
    каждая свою копию.

    Возвращается 0-based индекс: для ``total_steps=100, decay_ratio=0.05``
    граница равна 95, то есть decay-фаза — шаги 95..99 (последние 5%).
    """
    if total_steps <= 0:
        raise ValueError("total_steps должен быть > 0")
    if not 0.0 <= decay_ratio <= 1.0:
        raise ValueError("decay_ratio должен быть в [0, 1]")
    warmup_steps = max(1, int(round(total_steps * warmup_ratio)))
    decay_steps = max(0, int(round(total_steps * decay_ratio)))
    return max(warmup_steps, total_steps - decay_steps)


@dataclass(frozen=True)
class DecayWindowPlan:
    """План decay-фазы по объёму шарда Q (H1): хватает ли окна."""

    decay_ratio: float
    decay_steps: int
    window_tokens: int
    available_tokens: int
    adjusted: bool
    reason: str


def decay_window_plan(
    *,
    total_steps: int,
    decay_ratio: float,
    batch_size: int,
    seq_len: int,
    available_tokens: int | None,
) -> DecayWindowPlan:
    """Сверить decay-окно ``decay_steps·B·T`` с объёмом Q и уменьшить долю при нехватке (H1).

    Decay-фаза питается шардом Q за один проход; если окно шире Q, лоадер
    исчерпает Q **до конца прогона** и последние шаги упадут в
    ``data exhausted`` — финиш прогона теряется.  Поэтому до старта считается
    окно ``round(total_steps·decay_ratio)·B·T`` и сравнивается с доступным объёмом
    Q: при нехватке ``decay_ratio`` уменьшается до влезающего (окно = Q), а причина
    возвращается для журнала.  ``available_tokens is None`` (Q не объявлен или
    объём неизвестен) — проверка не выполняется, доля не меняется: неизвестность
    не выдаётся за «влезает».
    """
    if total_steps <= 0:
        raise ValueError("total_steps должен быть > 0")
    if not 0.0 <= decay_ratio <= 1.0:
        raise ValueError("decay_ratio должен быть в [0, 1]")
    requested = max(0, int(round(total_steps * decay_ratio)))
    per_step = int(batch_size) * int(seq_len)
    if available_tokens is None or per_step <= 0:
        return DecayWindowPlan(
            decay_ratio=decay_ratio,
            decay_steps=requested,
            window_tokens=requested * per_step,
            available_tokens=-1 if available_tokens is None else int(available_tokens),
            adjusted=False,
            reason="объём decay-шарда неизвестен — проверка не выполнялась",
        )
    window = requested * per_step
    available = int(available_tokens)
    if window <= available:
        return DecayWindowPlan(
            decay_ratio=decay_ratio,
            decay_steps=requested,
            window_tokens=window,
            available_tokens=available,
            adjusted=False,
            reason=f"окно {window} ≤ Q {available} — доля сохранена",
        )
    # Сколько шагов влезает в Q.  Максимум — запрошенные шаги, минимум — 0
    # (0 означает «decay-фазы нет»: пустой Q кормить нечем, честнее не объявлять
    # фазу, чем упасть на исчерпании).
    fits = min(requested, available // per_step)
    adjusted_ratio = fits / total_steps if total_steps else 0.0
    return DecayWindowPlan(
        decay_ratio=adjusted_ratio,
        decay_steps=fits,
        window_tokens=fits * per_step,
        available_tokens=available,
        adjusted=True,
        reason=(
            f"окно {window} > Q {available}: decay_ratio уменьшен "
            f"{decay_ratio:.4f} → {adjusted_ratio:.4f} ({fits} шагов), "
            "иначе Q исчерпается до конца прогона (H1)"
        ),
    )


def wsd_schedule(
    peak_lr: float,
    total_steps: int,
    *,
    warmup_ratio: float = 0.01,
    decay_ratio: float = 0.05,
    min_ratio: float = 0.0,
) -> Callable[[float], float]:
    """WSD: линейный warmup → стабильное плато → decay последние ``decay_ratio``.

    Отличие от ``net/optimizer.cosine_schedule`` (косинус на всём прогоне):
    стабильная фаза держит LR на пике, а decay приходит только в конце — это
    расписание фазы претрейна ADR-021 (~19B на W+C, затем ~1B на decay-шарде Q).
    Доля decay отсчитывается от **общего** числа шагов (дефолт 5%), граница — та
    же, что у данных (:func:`decay_start_step`).
    """
    if total_steps <= 0:
        raise ValueError("total_steps должен быть > 0")
    if not 0.0 <= min_ratio <= 1.0:
        raise ValueError("min_ratio должен быть в [0, 1]")
    warmup_steps = max(1, int(round(total_steps * warmup_ratio)))
    decay_start = decay_start_step(
        total_steps, warmup_ratio=warmup_ratio, decay_ratio=decay_ratio
    )
    # Прогон исполняет шаги ``0..total_steps-1``: decay приходит к минимуму на
    # последнем исполненном шаге, а не на «шаге total_steps», которого не будет.
    last_step = total_steps - 1

    def lr_at(step: float) -> float:
        step = float(step)
        if step < warmup_steps:
            return peak_lr * (step / warmup_steps)
        if step >= decay_start and last_step > decay_start:
            progress = (step - decay_start) / (last_step - decay_start)
            progress = min(max(progress, 0.0), 1.0)
            shape = 0.5 * (1.0 + math.cos(math.pi * progress))
            return peak_lr * (min_ratio + (1.0 - min_ratio) * shape)
        return peak_lr

    return lr_at


# ---------------------------------------------------------------------------
# 2. Манифест шард-набора
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ShardEntry:
    """Один шард-файл из манифеста (контракт ``tools/prep_pretrain``)."""

    file: str
    path: Path
    bytes: int
    sha256: str
    approx_tokens: int
    records: int


@dataclass(frozen=True)
class ShardSet:
    """Шард-набор (W или C): манифест + его шарды."""

    name: str
    manifest_path: Path
    entries: tuple[ShardEntry, ...]
    source: dict

    @property
    def total_tokens(self) -> int:
        return sum(entry.approx_tokens for entry in self.entries)

    @property
    def total_records(self) -> int:
        return sum(entry.records for entry in self.entries)


def load_shard_set(
    manifest_path: str | Path, *, allowed: Sequence[str] = SHARD_NAMES
) -> ShardSet:
    """Прочитать ``manifest-{w,c}.json`` и вернуть набор шардов.

    Манифест — это контракт данных (ADR-004/ADR-021): имя шарда, файлы, их
    sha256 и ``records``.  Неизвестное имя шарда или пустой список файлов —
    отказ: молчаливая подстановка другого набора запрещена (иначе прогон
    уехал бы по данным, не тем, что запиннены карточкой).
    """
    manifest_path = Path(manifest_path)
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise PretrainDataError(f"манифест не читается: {manifest_path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise PretrainDataError(f"манифест не разбирается: {manifest_path}: {exc}") from exc

    name = data.get("shard")
    if name not in allowed:
        raise PretrainDataError(
            f"шард {name!r} не из объявленного набора {tuple(allowed)}: {manifest_path}"
        )
    files = data.get("shards") or []
    if not files:
        raise PretrainDataError(f"манифест без шардов: {manifest_path}")
    entries = []
    for item in files:
        entry_path = manifest_path.parent / item["file"]
        if not entry_path.is_file():
            raise PretrainDataError(f"шард из манифеста отсутствует: {entry_path}")
        entries.append(
            ShardEntry(
                file=item["file"],
                path=entry_path,
                bytes=int(item.get("bytes", entry_path.stat().st_size)),
                sha256=str(item.get("sha256", "")),
                approx_tokens=int(item.get("approx_tokens", 0)),
                records=int(item.get("records", 0)),
            )
        )
    return ShardSet(
        name=name,
        manifest_path=manifest_path,
        entries=tuple(entries),
        source=dict(data.get("source") or {}),
    )


# ---------------------------------------------------------------------------
# 3. Потоковое чтение шарда
# ---------------------------------------------------------------------------


class ShardDocStream:
    """Ленивый поток документов одного ``.jsonl.zst`` (одна строка в памяти).

    Курсор resume (``skip``) проматывает строки **без** их разбора: на границе
    окна достаточно перечитать окно, а не токенизировать пролог шарда заново.
    """

    def __init__(self, path: str | Path, *, skip: int = 0):
        self.path = Path(path)
        self.skip = int(skip)
        self.documents_read = 0
        self._iterator: Iterator[str] | None = None
        self._handle = None
        self._reader = None
        self._text = None

    def __iter__(self) -> "ShardDocStream":
        return self

    def _open(self) -> Iterator[str]:
        import zstandard as zstd

        self._handle = open(self.path, "rb")
        self._reader = zstd.ZstdDecompressor().stream_reader(self._handle)
        self._text = io.TextIOWrapper(self._reader, encoding="utf-8", newline="\n")
        # H6: ``skip`` отсчитывает **непустые** строки — ту же единицу, в которой
        # измеряется ``shard_doc_offset``.  Раньше проматывались все строки подряд,
        # и пустая строка в шарде сдвигала позицию: resume внутри окна переигрывал
        # или терял документы.  Пропуск считается до фильтра пустых, чтобы нумерация
        # «документ N» оставалась согласованной с курсором.
        skipped = 0
        for line in self._text:
            stripped = line.strip()
            if not stripped:
                continue
            if skipped < self.skip:
                skipped += 1
                continue
            yield stripped

    def __next__(self) -> str:
        if self._iterator is None:
            self._iterator = self._open()
        line = next(self._iterator)
        self.documents_read += 1
        return line

    def close(self) -> None:
        for handle in (self._text, self._reader, self._handle):
            if handle is not None:
                try:
                    handle.close()
                except Exception:  # закрытие потока не должно ронять прогон
                    pass
        self._text = self._reader = self._handle = None
        self._iterator = None

    def __del__(self) -> None:
        self.close()


def iter_shard_docs(path: str | Path, *, skip: int = 0) -> ShardDocStream:
    """Поток документов шарда (``skip`` — сколько строк промахнуть на resume)."""
    return ShardDocStream(path, skip=skip)


# ---------------------------------------------------------------------------
# 4. Детерминированный шаффл окна
# ---------------------------------------------------------------------------


def window_permutation(
    *,
    seed: int,
    stream: str,
    shard_index: int,
    epoch: int,
    window_index: int,
    size: int,
) -> list[int]:
    """Перестановка окна ``size`` документов, детерминированная по сиду.

    Сид разворачивается в целое через SHA-256 от строкового ключа: ``random``
    хеширует кортежи через ``hash()``, который зависит от ``PYTHONHASHSEED``, —
    этот путь дал бы разный шаффл в разных процессах (AD-11 требует
    воспроизводимости от прогона к прогону, а не от процесса к процессу).
    """
    if size < 0:
        raise ValueError("size должен быть >= 0")
    key = f"{seed}|{stream}|{shard_index}|{epoch}|{window_index}|{size}".encode("utf-8")
    int_seed = int.from_bytes(hashlib.sha256(key).digest()[:8], "big")
    order = list(range(size))
    random.Random(int_seed).shuffle(order)
    return order


# ---------------------------------------------------------------------------
# 5. Курсор resume
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StreamCursor:
    """Позиция одного потока-шарда в курсоре."""

    name: str
    shard_index: int
    shard_doc_offset: int
    docs: int
    tokens: int
    epoch: int

    def to_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "shard_index": self.shard_index,
            "shard_doc_offset": self.shard_doc_offset,
            "docs": self.docs,
            "tokens": self.tokens,
            "epoch": self.epoch,
        }

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> "StreamCursor":
        return cls(
            name=str(data["name"]),
            shard_index=int(data["shard_index"]),
            shard_doc_offset=int(data["shard_doc_offset"]),
            docs=int(data.get("docs", 0)),
            tokens=int(data.get("tokens", 0)),
            epoch=int(data.get("epoch", 0)),
        )


@dataclass(frozen=True)
class MixCursor:
    """Манифест-курсор прогона: где поток стоит и что осталось не упаковано.

    ``pending_tokens`` — токены, уже извлечённые из документов, но не добравшие
    до последовательности.  Без них resume потерял бы хвост (или подсунул его
    дважды), поэтому остаток хранится в курсоре явно.  ``phase`` называет фазу-
    владельца хвоста: при resume на границе фаз остаток стабильной фазы не должен
    утечь в Q-поток.
    """

    step: int
    tokens_total: int
    streams: tuple[StreamCursor, ...]
    pending_tokens: tuple[int, ...]
    schema: str = CURSOR_SCHEMA
    #: Фаза, которой принадлежат ``pending_tokens`` (владелец хвоста): stable|decay.
    #: Нужна для корректного resume на границе фаз — хвост стабильной фазы не
    #: должен попасть в Q-поток, а позиция Q должна восстановиться со своего места.
    phase: str = PHASE_STABLE

    def to_json(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "step": self.step,
            "tokens_total": self.tokens_total,
            "streams": [item.to_json() for item in self.streams],
            "pending_tokens": list(self.pending_tokens),
            "phase": self.phase,
        }

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> "MixCursor":
        schema = str(data.get("schema", CURSOR_SCHEMA))
        if schema != CURSOR_SCHEMA:
            raise PretrainDataError(
                f"курсор схемы {schema!r} не поддерживается (ожидается {CURSOR_SCHEMA})"
            )
        phase = str(data.get("phase", PHASE_STABLE))
        if phase not in PHASES:
            raise PretrainDataError(f"фаза курсора {phase!r} не из {PHASES}")
        return cls(
            step=int(data.get("step", 0)),
            tokens_total=int(data.get("tokens_total", 0)),
            streams=tuple(StreamCursor.from_json(item) for item in data.get("streams", [])),
            pending_tokens=tuple(int(token) for token in data.get("pending_tokens", [])),
            phase=phase,
        )


# ---------------------------------------------------------------------------
# 6. Потоковый микс W/C
# ---------------------------------------------------------------------------


class _StreamIterator:
    """Поток одного шард-набора: окна документов + позиция курсора."""

    def __init__(self, shard_set: ShardSet, *, window: int, seed: int, cursor: StreamCursor | None = None):
        self.shard_set = shard_set
        self.name = shard_set.name
        self.window = int(window)
        self.seed = int(seed)
        self.shard_index = cursor.shard_index if cursor else 0
        self.shard_doc_offset = cursor.shard_doc_offset if cursor else 0
        self.docs = cursor.docs if cursor else 0
        self.tokens = cursor.tokens if cursor else 0
        self.epoch = cursor.epoch if cursor else 0
        self._buffer: deque[str] = deque()
        self._reader: ShardDocStream | None = None
        self._shard_exhausted = False
        self.exhausted = False

    # -- позиция -----------------------------------------------------------

    @property
    def entry(self) -> ShardEntry:
        return self.shard_set.entries[self.shard_index]

    def cursor(self) -> StreamCursor:
        return StreamCursor(
            name=self.name,
            shard_index=self.shard_index,
            shard_doc_offset=self.shard_doc_offset,
            docs=self.docs,
            tokens=self.tokens,
            epoch=self.epoch,
        )

    # -- окна --------------------------------------------------------------

    def _fill(self) -> bool:
        """Набрать следующее окно; False — шард исчерпан."""
        if self._shard_exhausted or self.shard_index >= len(self.shard_set.entries):
            return False
        start = (self.shard_doc_offset // self.window) * self.window
        if self._reader is None:
            self._reader = iter_shard_docs(self.entry.path, skip=start)
        docs = []
        for _ in range(self.window):
            try:
                docs.append(next(self._reader))
            except StopIteration:
                break
        if not docs:
            self._reader.close()
            self._reader = None
            self._shard_exhausted = True
            return False
        order = window_permutation(
            seed=self.seed,
            stream=self.name,
            shard_index=self.shard_index,
            epoch=self.epoch,
            window_index=start // self.window,
            size=len(docs),
        )
        self._buffer.extend(docs[index] for index in order)
        # уже выданная часть окна (resume внутри окна) не выдаётся повторно
        for _ in range(self.shard_doc_offset - start):
            self._buffer.popleft()
        if len(docs) < self.window:
            self._shard_exhausted = True
        return True

    def _advance_shard(self) -> bool:
        self.shard_index += 1
        self.shard_doc_offset = 0
        self._shard_exhausted = False
        if self._reader is not None:
            self._reader.close()
            self._reader = None
        if self.shard_index >= len(self.shard_set.entries):
            # эпоха исчерпана: микс W/C рассчитан на один проход (17B/3B)
            self.exhausted = True
            return False
        return True

    def next_document(self) -> str:
        """Следующий документ потока или ``StopIteration`` при исчерпании."""
        if self.exhausted:
            raise StopIteration
        while not self._buffer:
            if not self._fill():
                if not self._advance_shard():
                    raise StopIteration
                continue
        document = self._buffer.popleft()
        self.shard_doc_offset += 1
        self.docs += 1
        return document


class PretrainMixLoader:
    """Потоковый даталоадер претрейн-микса: батчи ``(B, T)`` id-токенов.

    Микс задаётся весами по токенам (ADR-021: ~85/15).  Выбор потока на каждом
    документе — по максимальному дефициту ``weight_i * emitted_total - emitted_i``:
    правило детерминировано и восстанавливается из курсора, потому что опирается
    только на накопленные счётчики токенов.

    Корпус не поднимается в память: живое окно — ``shuffle_window`` документов
    на поток плюс ``pending``-токены одной недобранной последовательности.

    **Decay-фаза (ADR-021).**  Если объявлен ``decay_stream`` (Q), даталоадер
    двухфазный: до шага ``decay_start`` идёт микс ``streams`` (stable), с шага
    ``decay_start`` — только ``decay_stream`` (decay).  ``decay_stream`` читается
    **тем же** читателем шардов, что W/C (тот же ``_StreamIterator``), со своим
    сидом шаффла; дедуп не нужен — Q фильтрован при сборке шарда.  Границу фаз
    считает :func:`decay_start_step`, чтобы данные и WSD-LR шли в ногу.
    """

    def __init__(
        self,
        *,
        shard_root: str | Path,
        streams: Sequence[str] = SHARD_NAMES,
        encode: Callable[[str], Sequence[int]],
        seq_len: int,
        batch_size: int = 1,
        seed: int = 0,
        mix: Mapping[str, float] | None = None,
        shuffle_window: int = 1000,
        cursor: MixCursor | None = None,
        max_doc_tokens: int | None = None,
        bos_id: int = BOS_ID,
        eos_id: int = EOS_ID,
        pad_id: int = PAD_ID,
        text_key: str = "text",
        decay_stream: str | None = None,
        decay_start: int | None = None,
        decay_shard_root: str | Path | None = None,
        decay_seed: int | None = None,
    ):
        if not streams:
            raise PretrainDataError("не объявлено ни одного потока шардов")
        if seq_len < 3:
            raise ValueError("seq_len должен быть >= 3 (bos + eos + минимум один токен)")
        if batch_size < 1:
            raise ValueError("batch_size должен быть >= 1")
        if shuffle_window < 1:
            raise ValueError("shuffle_window должен быть >= 1")
        decay_enabled = decay_stream is not None
        if decay_enabled and decay_start is None:
            raise PretrainDataError(
                f"decay-шард {decay_stream!r} объявлен без decay_start: "
                "граница фаз неизвестна (см. decay_start_step)"
            )
        if decay_start is not None and not decay_enabled:
            raise PretrainDataError(
                "decay_start объявлен без decay_stream: неясно, чем питать decay-фазу"
            )
        if decay_enabled and decay_stream in streams:
            raise PretrainDataError(
                f"decay-шард {decay_stream!r} не может входить в стабильный микс "
                f"{tuple(streams)}: иначе он подмешивался бы всё время (ADR-021)"
            )
        if cursor is not None and cursor.phase == PHASE_DECAY and not decay_enabled:
            raise PretrainDataError(
                "курсор снят в decay-фазе, а decay-шард не объявлен: resume без него "
                "потерял бы Q-поток — объявите decay_stream/decay_start"
            )

        self.shard_root = Path(shard_root)
        self.seq_len = int(seq_len)
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.shuffle_window = int(shuffle_window)
        self.encode = encode
        self.max_doc_tokens = max_doc_tokens
        self.bos_id, self.eos_id, self.pad_id = bos_id, eos_id, pad_id
        self.text_key = text_key
        self.step = cursor.step if cursor else 0
        self.decay_stream = decay_stream
        self.decay_shard_root = (
            Path(decay_shard_root) if decay_shard_root is not None else None
        )
        #: Первый 0-based шаг decay-фазы; ``None`` — фаза выключена (микс всё время).
        self.decay_start = int(decay_start) if decay_start is not None else None
        #: Сид шаффла decay-шарда: свой порядок, отдельный от стабильного микса.
        self.decay_seed = int(decay_seed) if decay_seed is not None else self.seed

        weights = dict(mix) if mix else {name: 1.0 for name in streams}
        missing = [name for name in streams if name not in weights]
        if missing:
            raise PretrainDataError(f"для потоков {missing} не объявлены веса микса")
        if any(weight < 0 for weight in weights.values()):
            raise PretrainDataError("веса микса не могут быть отрицательными")
        if sum(weights[name] for name in streams) <= 0:
            raise PretrainDataError("сумма весов микса должна быть > 0")
        self.weights = {name: float(weights[name]) for name in streams}

        cursor_streams = {item.name: item for item in (cursor.streams if cursor else ())}
        self._streams: dict[str, _StreamIterator] = {}
        for name in streams:
            manifest = self.shard_root / name / f"manifest-{name.lower()}.json"
            shard_set = load_shard_set(manifest, allowed=tuple(streams))
            self._streams[name] = _StreamIterator(
                shard_set,
                window=self.shuffle_window,
                seed=self.seed,
                cursor=cursor_streams.get(name),
            )

        self._q_stream: _StreamIterator | None = None
        if decay_enabled:
            q_root = self.decay_shard_root or self.shard_root
            q_manifest = q_root / decay_stream / f"manifest-{decay_stream.lower()}.json"
            q_set = load_shard_set(q_manifest, allowed=(decay_stream,))
            self._q_stream = _StreamIterator(
                q_set,
                window=self.shuffle_window,
                seed=self.decay_seed,
                cursor=cursor_streams.get(decay_stream),
            )

        # pending разделён по фазам: хвост стабильной фазы не должен утечь в Q и
        # наоборот.  Восстанавливается в фазу-владельца, названную курсором.
        owner = cursor.phase if cursor else PHASE_STABLE
        self._pending: dict[str, deque[int]] = {
            PHASE_STABLE: deque(),
            PHASE_DECAY: deque(),
        }
        if cursor and cursor.pending_tokens:
            self._pending[owner].extend(cursor.pending_tokens)
        self._last_phase = owner

        self._documents_read = 0
        self._dropped_empty = 0
        self._truncated = 0
        self._exhausted = False

    # -- контракт итератора ------------------------------------------------

    def __iter__(self) -> "PretrainMixLoader":
        return self

    def _phase_for_batch(self, step: int) -> str:
        """Фаза батча с абсолютным номером ``step`` (1-based)."""
        if self.decay_start is None:
            return PHASE_STABLE
        return PHASE_DECAY if (step - 1) >= self.decay_start else PHASE_STABLE

    @property
    def phase(self) -> str:
        """Фаза последнего выданного батча (stable до первого — ``stable``)."""
        return self._last_phase

    def __next__(self) -> np.ndarray:
        if self._exhausted:
            raise StopIteration
        phase = self._phase_for_batch(self.step + 1)
        pending = self._pending[phase]
        # Запись = [BOS] + (T-1) токенов потока (H3): T-1, а не T, потому что BOS
        # занимает слот записи и не является токеном потока.  Так запись совпадает
        # с раскладкой претокенизированного ``.bin`` (``PACKED_RECORD_LAYOUT``).
        needed = (self.seq_len - 1) * self.batch_size
        while len(pending) < needed:
            if not self._fill_pending(phase):
                self._exhausted = True
                raise StopIteration
        chunk = [pending.popleft() for _ in range(needed)]
        self.step += 1
        self._last_phase = phase
        return pack_batch(chunk, self.seq_len, self.bos_id, self.eos_id, self.pad_id)

    def _fill_pending(self, phase: str) -> bool:
        """Добрать pending текущей фазы; False — фаза исчерпана.

        Эпоха заканчивается на **первом** исчерпании активной фазы: микс W/C
        объявлен как один проход по 17B/3B (ADR-021), и когда один шард кончился,
        удержать объявленную пропорцию уже нельзя — продолжать значит молча уехать
        по другой пропорции, чем запиннена в карточке.  То же и для decay: Q —
        один проход ~1B, кончился — эпоха кончилась.
        """
        if phase == PHASE_DECAY:
            stream = self._q_stream
            if stream is None or stream.exhausted:
                return False
            return self._absorb_document(stream, PHASE_DECAY)
        name = self._pick_stream()
        if name is None:
            return False
        return self._absorb_document(self._streams[name], PHASE_STABLE)

    def _absorb_document(self, stream: _StreamIterator, phase: str) -> bool:
        try:
            raw = stream.next_document()
        except StopIteration:
            stream.exhausted = True
            return False
        self._documents_read += 1
        tokens = self._document_tokens(raw)
        if not tokens:
            return True
        self._pending[phase].extend(tokens)
        stream.tokens += len(tokens)
        return True

    def _document_tokens(self, raw: str) -> list[int]:
        """Токены одной строки шарда + закрывающий EOS (общий путь фаз).

        H3: документы в потоке разделяются **EOS** — ровно так же, как их упаковывает
        претокенизация (``tools/pretokenize.py``: между двумя EOS лежит один документ).
        Без разделителя склейка учила бы модель продолжать чужой документ, а граница
        оставалась ненаблюдаемой.  EOS — разделитель стыка, а не «конец записи»:
        запись режется по ``T-1`` (:func:`pack_batch`) без вставки фальшивого EOS на
        разрезе, поэтому длинный документ, переехавший на границу записи, своего
        настоящего EOS не теряет и чужого не получает.
        """
        text = ""
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                text = str(parsed.get(self.text_key) or "")
        except json.JSONDecodeError:
            text = ""
        tokens = list(self.encode(text)) if text else []
        if self.max_doc_tokens is not None and len(tokens) > self.max_doc_tokens:
            self._truncated += 1
            tokens = tokens[: self.max_doc_tokens]
        if not tokens:
            self._dropped_empty += 1
            return []
        tokens.append(self.eos_id)
        return tokens

    def _pick_stream(self) -> str | None:
        """Поток с максимальным дефицитом против объявленной пропорции.

        Дефицит считается от накопленных счётчиков токенов, поэтому после resume
        выбор продолжается ровно там, где остановился непрерывный прогон.

        H2: при исчерпании потока его вес и счётчик **исключаются** из расчёта, а
        веса оставшихся перенормируются.  Иначе хвост эпохи шёл бы «соло»: мёртвый
        поток продолжал тянуть долю на себя в ``total`` и ``weights``, дефицит
        живого искажался, и остаток корпуса съедал один поток.  Перенормировка
        держит объявленную **относительную** пропорцию оставшихся (для двух и
        более живых потоков), а один живой — это неизбежное соло последнего
        потока, а не молчаливое искажение микса.
        """
        live = [name for name in self._streams if not self._streams[name].exhausted]
        if not live:
            return None
        emitted = {name: self._streams[name].tokens for name in self._streams}
        total = sum(emitted[name] for name in live)
        weight_sum = sum(self.weights[name] for name in live)
        best_name, best_deficit = None, None
        for name in live:
            share = self.weights[name] / weight_sum if weight_sum > 0 else 1.0 / len(live)
            deficit = share * total - emitted[name]
            if best_deficit is None or deficit > best_deficit:
                best_name, best_deficit = name, deficit
        return best_name

    # -- след и курсор -----------------------------------------------------

    def _all_streams(self) -> dict[str, _StreamIterator]:
        streams = dict(self._streams)
        if self._q_stream is not None and self.decay_stream is not None:
            streams[self.decay_stream] = self._q_stream
        return streams

    def cursor(self, *, step: int | None = None) -> MixCursor:
        streams = tuple(item.cursor() for item in self._all_streams().values())
        return MixCursor(
            step=self.step if step is None else int(step),
            tokens_total=sum(item.tokens for item in streams),
            streams=streams,
            pending_tokens=tuple(self._pending[self._last_phase]),
            phase=self._last_phase,
        )

    def stats(self) -> dict[str, Any]:
        streams = {name: item.cursor() for name, item in self._all_streams().items()}
        total_tokens = sum(item.tokens for item in streams.values())
        share = {
            name: (item.tokens / total_tokens if total_tokens else 0.0)
            for name, item in streams.items()
        }
        return {
            "documents_read": self._documents_read,
            "documents_dropped_empty": self._dropped_empty,
            "documents_truncated": self._truncated,
            "tokens_total": total_tokens,
            "token_share": share,
            "pending_tokens": len(self._pending[self._last_phase]),
            "phase": self._last_phase,
            "decay_stream": self.decay_stream,
            "decay_start": self.decay_start,
            "streams": {
                name: {
                    "shard_index": item.shard_index,
                    "shard_doc_offset": item.shard_doc_offset,
                    "docs": item.docs,
                    "tokens": item.tokens,
                    "epoch": item.epoch,
                }
                for name, item in streams.items()
            },
        }


def pack_batch(
    tokens: Sequence[int],
    seq_len: int,
    bos_id: int = BOS_ID,
    eos_id: int = EOS_ID,  # noqa: ARG001 — аргумент сохранён для совместимости вызова
    pad_id: int = PAD_ID,  # noqa: ARG001 — EOS/PAD сюда не вставляются (см. ниже)
) -> np.ndarray:
    """Упаковать непрерывный поток токенов в ``(B, T)`` без фальшивых EOS (H3).

    Запись — ровно ``[BOS] + (T-1) токенов потока``: та же раскладка, что у
    претокенизированного ``.bin`` (``PACKED_RECORD_LAYOUT``), поэтому raw-путь и
    packed-путь дают модели один и тот же вид строки.  Прежний путь через
    ``net.data.pack_sequence`` брал ``tokens[:T-2]`` и дописывал ``[eos]``: на
    каждом разрезе записи рождался **фальшивый EOS** (документ не кончался, его
    просто разрезала граница записи), а два последних токена окна терялись.
    Настоящие границы документов ставит поток (``_document_tokens`` добавляет EOS
    к каждому документу), а не упаковщик — упаковщик только режет по ``T-1``.

    ``eos_id``/``pad_id`` принимаются для совместимости сигнатуры и не
    используются: вставлять их здесь значило бы вернуть тот самый фальшивый EOS.
    """
    body = int(seq_len) - 1
    if body < 2:
        raise ValueError("seq_len должен быть >= 3 (bos + минимум два токена потока)")
    if len(tokens) % body != 0:
        raise ValueError("длина потока токенов должна быть кратна seq_len - 1")
    rows = []
    for start in range(0, len(tokens), body):
        row = np.empty(seq_len, dtype=np.int32)
        row[0] = bos_id
        row[1:] = np.asarray(tokens[start : start + body], dtype=np.int32)
        rows.append(row)
    return np.stack(rows, axis=0).astype(np.int32)


# ---------------------------------------------------------------------------
# 6-бис. Читатель претокенизированного корпуса (``tools/pretokenize.py``)
# ---------------------------------------------------------------------------
#
# Поток ``.jsonl.zst`` (``PretrainMixLoader`` выше) токенизирует документы на
# каждом шаге; претокенизированный корпус — это уже готовые id, упакованные в
# записи ``T`` (AD-004: токенизация делается один раз локально и уезжает на
# аренду как данные).  Различие только в источнике токенов, поэтому читатель
# отдаёт луп **тот же** контракт: ``(B, T) int32`` с BOS в позиции 0, где
# ``compute_loss`` сдвигает сам.  Никакой арифметики границ здесь нет — границы
# записи лежат в файле, и их раскладка сверяется с манифестом при открытии.


#: Схема манифеста претокенизированного шард-набора (``tools/pretokenize.py``).
PACKED_MANIFEST_SCHEMA = "axiom-pretrain-tokens/1"

#: Раскладка записи ``.bin``, которую ждёт читатель (сверяется с манифестом).
#: Запись = ``[bos] + (seq_len - 1) токенов потока``: строка модели целиком.
PACKED_RECORD_LAYOUT = "bos + (seq_len-1) токенов потока"

#: Сколько записей читать за один заход (512 × 8192 × 4 Б ≈ 16 МиБ).
PACKED_READ_BLOCK = 512


@dataclass(frozen=True)
class PackedShardEntry:
    """Один ``.bin`` шард из манифеста претокенизации."""

    file: str
    path: Path
    source: str
    source_sha256: str
    records: int
    slots: int
    stream_tokens: int
    pad_tokens: int
    bytes: int
    sha256: str


@dataclass(frozen=True)
class PackedShardSet:
    """Шард-набор готовых токенов (W или C): манифест + его ``.bin``.

    Два счётчика токенов различаются осознанно и не взаимозаменяемы:

    * ``total_slots`` — столько id отдаст читатель (``records × seq_len``):
      сюда входят служебные BOS (по одному на запись) и PAD хвоста;
    * ``total_stream_tokens`` — токены самого потока (документы + EOS + кодовые
      префиксы).  Именно он сопоставим с ``ShardSet.total_tokens`` шард-набора
      jsonl, поэтому смета и пропорции микса считаются по нему.
    """

    name: str
    manifest_path: Path
    seq_len: int
    entries: tuple[PackedShardEntry, ...]
    tokenizer_hash: str
    source: dict

    @property
    def total_records(self) -> int:
        return sum(entry.records for entry in self.entries)

    @property
    def total_slots(self) -> int:
        return sum(entry.slots for entry in self.entries)

    @property
    def total_stream_tokens(self) -> int:
        return sum(entry.stream_tokens for entry in self.entries)


def packed_manifest_path(tokens_root: str | Path, stream: str) -> Path:
    """Путь манифеста претокенизированного набора (``tokens/W/manifest-w.json``)."""
    return Path(tokens_root) / stream / f"manifest-{stream.lower()}.json"


def load_packed_shard_set(
    manifest_path: str | Path, *, allowed: Sequence[str] = SHARD_NAMES
) -> PackedShardSet:
    """Прочитать манифест претокенизации и вернуть набор ``.bin`` шардов.

    Манифест — такой же контракт данных, как ``manifest-{w,c}.json`` (AD-4):
    имя шарда, файлы, их sha256 и раскладка записи.  Неизвестное имя шарда,
    пустой список, чужая схема или **другая раскладка записи** — отказ: читатель,
    который «догадается» про границы, вернёт молча сдвинутый поток, и это
    вылезло бы только на лоссе.
    """
    manifest_path = Path(manifest_path)
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise PretrainDataError(f"манифест tokens не читается: {manifest_path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise PretrainDataError(f"манифест tokens не разбирается: {manifest_path}: {exc}") from exc

    if data.get("version") != PACKED_MANIFEST_SCHEMA:
        raise PretrainDataError(
            f"схема манифеста tokens {data.get('version')!r} != {PACKED_MANIFEST_SCHEMA!r}: "
            f"{manifest_path}"
        )
    name = data.get("shard")
    if name not in allowed:
        raise PretrainDataError(
            f"шард {name!r} не из объявленного набора {tuple(allowed)}: {manifest_path}"
        )
    seq_len = int(data.get("seq_len") or 0)
    if seq_len < 3:
        raise PretrainDataError(f"seq_len манифеста не объявлен или мал: {seq_len}")
    layout = data.get("record_layout")
    if layout != PACKED_RECORD_LAYOUT:
        raise PretrainDataError(
            f"раскладка записи {layout!r} != ожидаемой {PACKED_RECORD_LAYOUT!r}: {manifest_path}"
        )
    if data.get("dtype") != "uint32":
        raise PretrainDataError(f"dtype {data.get('dtype')!r} != 'uint32': {manifest_path}")
    # Специальные токены обязаны совпасть с константами лупа: id запечены в файл,
    # и разошедшийся BOS — это сдвиг всей последовательности, а не косметика.
    for key, expected in (("bos_id", BOS_ID), ("eos_id", EOS_ID), ("pad_id", PAD_ID)):
        actual = int(data.get(key, expected))
        if actual != expected:
            raise PretrainDataError(
                f"{key} манифеста {actual} != {expected} (net.data.pack_sequence): {manifest_path}"
            )
    files = data.get("shards") or []
    if not files:
        raise PretrainDataError(f"манифест без шардов: {manifest_path}")
    entries = []
    for item in files:
        path = manifest_path.parent / item["file"]
        if not path.is_file():
            raise PretrainDataError(f"шард готовых токенов отсутствует: {path}")
        records = int(item.get("records", 0))
        if records <= 0:
            raise PretrainDataError(f"шард без записей: {path}")
        entries.append(
            PackedShardEntry(
                file=item["file"],
                path=path,
                source=str(item.get("source", "")),
                source_sha256=str(item.get("source_sha256", "")),
                records=records,
                slots=int(item.get("tokens", records * seq_len)),
                stream_tokens=int(item.get("stream_tokens", 0)),
                pad_tokens=int(item.get("pad_tokens", 0)),
                bytes=int(item.get("bytes", path.stat().st_size)),
                sha256=str(item.get("sha256", "")),
            )
        )
    return PackedShardSet(
        name=name,
        manifest_path=manifest_path,
        seq_len=seq_len,
        entries=tuple(entries),
        tokenizer_hash=str(data.get("tokenizer_hash", "")),
        source=dict(data.get("tokenizer") or {}),
    )


class PackedShardReader:
    """Ленивый поток записей одного ``.bin``: ``(seq_len,) int32`` за шаг.

    Проверки при открытии — не формальность: длина файла обязана быть кратна
    записи (иначе последняя запись битая), а число записей — совпасть с
    манифестом (иначе файл от другой ревизии).  Недосчитанный шард читается как
    ошибка, а не как «на один батч меньше»: укороченный поток незаметно меняет
    эпоху.
    """

    def __init__(self, entry: PackedShardEntry, seq_len: int, *, skip: int = 0):
        self.entry = entry
        self.seq_len = int(seq_len)
        self.record_bytes = self.seq_len * 4  # uint32
        # Поля инициализируются до проверок: отказ в __init__ оставляет объект
        # без ``_handle``, и ``__del__`` не должен падать вторым исключением.
        self._handle = None
        self._block: np.ndarray | None = None
        self._index = 0
        self._emitted = int(skip)
        size = entry.path.stat().st_size
        if size % self.record_bytes != 0:
            raise PretrainDataError(
                f"размер {entry.path} ({size} Б) не кратен записи {self.record_bytes} Б"
            )
        on_disk = size // self.record_bytes
        if entry.records and on_disk != entry.records:
            raise PretrainDataError(
                f"{entry.path}: записей на диске {on_disk}, а в манифесте {entry.records}"
            )
        if skip < 0 or skip > on_disk:
            raise PretrainDataError(
                f"{entry.path}: смещение resume {skip} вне [0, {on_disk}] записей"
            )
        self.records = on_disk
        #: Сколько записей шарда уже пройдено — позиция для курсора resume.
        self.offset = int(skip)

    def __iter__(self) -> "PackedShardReader":
        return self

    def _fill(self) -> bool:
        if self._handle is None:
            self._handle = open(self.entry.path, "rb")
            # Resume: пропущенное смещение не читается блоками, а перематывается
            # указателем — это O(1), а не O(skip) записей.
            if self._emitted:
                self._handle.seek(self._emitted * self.record_bytes)
        count = min(PACKED_READ_BLOCK, self.records - self._emitted)
        if count <= 0:
            return False
        raw = self._handle.read(count * self.record_bytes)
        if len(raw) != count * self.record_bytes:
            raise PretrainDataError(f"{self.entry.path}: файл кончился раньше манифеста")
        self._block = np.frombuffer(raw, dtype=np.uint32).reshape(count, self.seq_len)
        self._emitted += count
        self._index = 0
        return True

    def __next__(self) -> np.ndarray:
        """Следующая запись как ``(seq_len,) int32`` (id < vocab, старший бит пуст)."""
        if self._block is None or self._index >= self._block.shape[0]:
            if not self._fill():
                raise StopIteration
        row = self._block[self._index]
        self._index += 1
        self.offset = self._emitted - (self._block.shape[0] - self._index)
        return row.astype(np.int32, copy=False)

    def close(self) -> None:
        if self._handle is not None:
            try:
                self._handle.close()
            except Exception:  # закрытие не должно ронять прогон
                pass
        self._handle = None

    def __del__(self) -> None:
        self.close()


class PackedTokenLoader:
    """Батчи ``(B, T) int32`` из готовых ``.bin`` — пара к ``PretrainMixLoader``.

    Пропорция микса держится на **гранулярности батча**: поток выбирается по
    максимальному дефициту ``weight * emitted - emitted_stream`` (то же правило,
    что у ``PretrainMixLoader``), а внутри потока шарды идут по порядку манифеста.
    Окно шаффла к готовым токенам не применяется осознанно: документы уже упакованы
    в записи, и «шаффлить» их значило бы перемешивать фиксированные окна контекста.

    Как и ``PretrainMixLoader``, читатель поддерживает **resume из курсора** (К2/К3):
    ``cursor()`` отдаёт ``MixCursor`` той же формы, а конструктор принимает его
    обратно — позиция потока (шард + смещение записей в нём) восстанавливается без
    потерь и дублей, поэтому аренда interruptible переживает преемпшн.  Decay-фаза
    (ADR-021) поддержана тем же способом: с шага ``decay_start`` батчи берутся
    только из шарда Q.
    """

    def __init__(
        self,
        *,
        tokens_root: str | Path,
        streams: Sequence[str] = SHARD_NAMES,
        seq_len: int,
        batch_size: int = 1,
        mix: Mapping[str, float] | None = None,
        cursor: MixCursor | None = None,
        decay_stream: str | None = None,
        decay_start: int | None = None,
        decay_shard_root: str | Path | None = None,
    ):
        if not streams:
            raise PretrainDataError("не объявлено ни одного потока шардов")
        if seq_len < 3:
            raise ValueError("seq_len должен быть >= 3")
        if batch_size < 1:
            raise ValueError("batch_size должен быть >= 1")
        decay_enabled = decay_stream is not None
        if decay_enabled and decay_start is None:
            raise PretrainDataError(
                f"decay-шард {decay_stream!r} объявлен без decay_start: граница фаз неизвестна"
            )
        if decay_start is not None and not decay_enabled:
            raise PretrainDataError("decay_start объявлен без decay_stream")
        if decay_enabled and decay_stream in streams:
            raise PretrainDataError(
                f"decay-шард {decay_stream!r} не может входить в стабильный микс {tuple(streams)}"
            )
        if cursor is not None and cursor.phase == PHASE_DECAY and not decay_enabled:
            raise PretrainDataError(
                "курсор снят в decay-фазе, а decay-шард не объявлен: resume потерял бы Q-поток"
            )

        self.tokens_root = Path(tokens_root)
        self.decay_shard_root = (
            Path(decay_shard_root) if decay_shard_root is not None else self.tokens_root
        )
        self.seq_len = int(seq_len)
        self.batch_size = int(batch_size)
        weights = dict(mix) if mix else {name: 1.0 for name in streams}
        missing = [name for name in streams if name not in weights]
        if missing:
            raise PretrainDataError(f"для потоков {missing} не объявлены веса микса")
        if any(weight < 0 for weight in weights.values()):
            raise PretrainDataError("веса микса не могут быть отрицательными")
        if sum(weights[name] for name in streams) <= 0:
            raise PretrainDataError("сумма весов микса должна быть > 0")
        self.weights = {name: float(weights[name]) for name in streams}
        self.decay_stream = decay_stream
        self.decay_start = int(decay_start) if decay_start is not None else None

        cursor_streams = {item.name: item for item in (cursor.streams if cursor else ())}
        self._sets: dict[str, PackedShardSet] = {}
        self._readers: dict[str, PackedShardReader] = {}
        self._shard_index: dict[str, int] = {}
        self._offset: dict[str, int] = {}          # записей выдано из текущего шарда
        self._records: dict[str, int] = {}         # записей выдано из потока всего
        self._emitted: dict[str, int] = {}         # слотов (records×T) выдано из потока
        self._dropped_tail: dict[str, int] = {}
        self._exhausted: set[str] = set()

        for name in streams:
            shard_set = load_packed_shard_set(
                packed_manifest_path(self.tokens_root, name), allowed=tuple(streams)
            )
            if shard_set.seq_len != self.seq_len:
                raise PretrainDataError(
                    f"seq_len манифеста {name} ({shard_set.seq_len}) != заказанного {self.seq_len}"
                )
            self._sets[name] = shard_set
            self._open_stream(name, shard_set, cursor_streams.get(name))

        self._q_stream: str | None = None
        if decay_enabled:
            assert decay_stream is not None
            q_set = load_packed_shard_set(
                packed_manifest_path(self.decay_shard_root, decay_stream),
                allowed=(decay_stream,),
            )
            if q_set.seq_len != self.seq_len:
                raise PretrainDataError(
                    f"seq_len манифеста {decay_stream} ({q_set.seq_len}) != {self.seq_len}"
                )
            self._sets[decay_stream] = q_set
            self._open_stream(decay_stream, q_set, cursor_streams.get(decay_stream))
            self._q_stream = decay_stream

        self.step = cursor.step if cursor else 0
        self._last_phase = cursor.phase if cursor else PHASE_STABLE

    def _open_stream(
        self, name: str, shard_set: PackedShardSet, cursor: StreamCursor | None
    ) -> None:
        """Открыть поток на позиции курсора (или с начала) и восстановить счётчики."""
        shard_index = int(cursor.shard_index) if cursor else 0
        offset = int(cursor.shard_doc_offset) if cursor else 0
        if shard_index >= len(shard_set.entries):
            raise PretrainDataError(
                f"курсор {name} указывает на шард {shard_index}, а их {len(shard_set.entries)}"
            )
        entry = shard_set.entries[shard_index]
        if offset > entry.records:
            raise PretrainDataError(
                f"курсор {name}: смещение {offset} больше записей шарда {entry.records}"
            )
        self._shard_index[name] = shard_index
        self._offset[name] = offset
        if cursor is not None:
            self._records[name] = int(cursor.docs)
            self._emitted[name] = int(cursor.tokens)
        else:
            self._records[name] = 0
            self._emitted[name] = 0
        self._dropped_tail[name] = 0
        self._readers[name] = PackedShardReader(entry, self.seq_len, skip=offset)

    def __iter__(self) -> "PackedTokenLoader":
        return self

    def _phase_for_batch(self, step: int) -> str:
        """Фаза батча с абсолютным номером ``step`` (1-based)."""
        if self.decay_start is None:
            return PHASE_STABLE
        return PHASE_DECAY if (step - 1) >= self.decay_start else PHASE_STABLE

    @property
    def phase(self) -> str:
        """Фаза последнего выданного батча (stable до первого)."""
        return self._last_phase

    def __next__(self) -> np.ndarray:
        """Следующий батч ``(B, T)``.

        Батч всегда полный: у лупа фиксированная форма ``(B, T)`` (JAX-граф
        компилируется под неё), поэтому недобранный хвост потока — это конец
        эпохи, а не батч другого размера.  Добор идёт **сквозь** границу шарда,
        так что на стыке ``.bin`` теряются не записи, а только хвост потока
        (``≤ B-1`` записей за эпоху, счётчик — в ``stats()``).
        """
        phase = self._phase_for_batch(self.step + 1)
        name = self._q_stream if phase == PHASE_DECAY else self._pick_stream()
        if name is None:
            raise StopIteration
        rows: list[np.ndarray] = []
        while len(rows) < self.batch_size:
            try:
                rows.append(next(self._readers[name]))
            except StopIteration:
                if self._advance_shard(name):
                    continue
                self._exhausted.add(name)
                break
        if len(rows) < self.batch_size:
            self._dropped_tail[name] += len(rows)
            raise StopIteration
        batch = np.stack(rows, axis=0)
        self._emitted[name] += batch.size
        self._records[name] += len(rows)
        self._offset[name] = self._readers[name].offset
        self.step += 1
        self._last_phase = phase
        return batch

    def _pick_stream(self) -> str | None:
        """Поток с максимальным дефицитом против объявленной пропорции.

        H2: мёртвые потоки исключаются из расчёта, веса оставшихся перенормируются —
        иначе хвост эпохи шёл бы «соло» с искажённым дефицитом.

        Итерируем по ``self.weights`` (потоки стабильного микса), а не по
        ``self._sets``: в ``_sets`` лежит ещё и decay-шард Q, у которого веса в
        миксе нет — выбор его здесь сломал бы как пропорцию, так и индекс весов.
        """
        live = [name for name in self.weights if name not in self._exhausted]
        if not live:
            return None
        total = sum(self._emitted[name] for name in live)
        weight_sum = sum(self.weights[name] for name in live)
        best_name, best_deficit = None, None
        for name in live:
            share = self.weights[name] / weight_sum if weight_sum > 0 else 1.0 / len(live)
            deficit = share * total - self._emitted[name]
            if best_deficit is None or deficit > best_deficit:
                best_name, best_deficit = name, deficit
        return best_name

    def _advance_shard(self, name: str) -> bool:
        """Перейти к следующему ``.bin`` потока; ``False`` — поток кончился."""
        reader = self._readers[name]
        reader.close()
        entries = self._sets[name].entries
        index = self._shard_index[name] + 1
        while index < len(entries):
            next_reader = PackedShardReader(entries[index], self.seq_len)
            if next_reader.records > 0:
                self._readers[name] = next_reader
                self._shard_index[name] = index
                self._offset[name] = 0
                return True
            index += 1
        return False

    # -- след ---------------------------------------------------------------

    def cursor(self, *, step: int | None = None) -> MixCursor:
        """Курсор потока в форме ``MixCursor`` — контракт resume, общий с raw-путём."""
        streams = []
        for name in self._sets:
            streams.append(
                StreamCursor(
                    name=name,
                    shard_index=self._shard_index[name],
                    shard_doc_offset=self._offset[name],
                    docs=self._records[name],
                    tokens=self._emitted[name],
                    epoch=0,
                )
            )
        return MixCursor(
            step=self.step if step is None else int(step),
            tokens_total=sum(self._emitted.values()),
            streams=tuple(streams),
            pending_tokens=(),
            phase=self._last_phase,
        )

    def stats(self) -> dict[str, Any]:
        total = sum(self._emitted.values())
        return {
            "step": self.step,
            "batch_size": self.batch_size,
            "seq_len": self.seq_len,
            "slots_total": total,
            "token_share": {
                name: (emitted / total if total else 0.0)
                for name, emitted in self._emitted.items()
            },
            "dropped_tail_records": dict(self._dropped_tail),
            "phase": self._last_phase,
            "decay_stream": self.decay_stream,
            "decay_start": self.decay_start,
            "records_total": sum(self._records.values()),
            "streams": {
                name: {
                    "shard_index": self._shard_index[name],
                    "shards": len(self._sets[name].entries),
                    "slots": self._emitted[name],
                    "records": self._records[name],
                    "stream_tokens": self._sets[name].total_stream_tokens,
                    "exhausted": name in self._exhausted,
                    "tokenizer_hash": self._sets[name].tokenizer_hash,
                }
                for name in self._sets
            },
        }

    def close(self) -> None:
        for reader in self._readers.values():
            reader.close()


# ---------------------------------------------------------------------------
# 7. Стоп-правило AD-8: смета до запуска, превышение останавливает прогон
# ---------------------------------------------------------------------------


class PretrainBudgetError(RuntimeError):
    """Смета отсутствует или непригодна — запуск блокирован (AD-8)."""


@dataclass(frozen=True)
class Budget:
    """Смета прогона (AD-8/C-041) + поле ``target_tokens`` претрейна.

    ``target_tokens`` — объём корпуса, на который рассчитан прогон; вместе с
    ``gpu_hours_estimate``/``limit_usd`` он образует стоп-правило: превышение
    любого из трёх останавливает прогон (после ближайшего чекпойнта).
    """

    run_ref: str
    path: Path
    present: bool
    target_tokens: int | None = None
    gpu_hours_estimate: float | None = None
    usd_estimate: float | None = None
    limit_usd: float | None = None
    budget_method: str = ""
    stop_rule: str = ""
    approved_by: str = ""
    raw: dict = field(default_factory=dict)

    def with_explicit_limit(self, limit_usd: float) -> "Budget":
        return dataclass_replace(self, limit_usd=float(limit_usd))


def dataclass_replace(instance: Any, **changes: Any) -> Any:
    """``dataclasses.replace`` без импорта на уровне модуля (мелочь, но явно)."""
    import dataclasses

    return dataclasses.replace(instance, **changes)


def load_budget(path: str | Path) -> Budget:
    """Прочитать смету прогона; отсутствующий файл — факт ``present=False``."""
    path = Path(path)
    raw: dict[str, Any] = {}
    present = path.is_file()
    if present:
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise PretrainBudgetError(f"смета не читается: {path}: {exc}") from exc

    def _number(key: str) -> float | None:
        value = raw.get(key)
        return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None

    target = raw.get("target_tokens")
    return Budget(
        run_ref=str(raw.get("run_ref") or path.stem),
        path=path,
        present=present,
        target_tokens=int(target) if isinstance(target, int) and not isinstance(target, bool) else None,
        gpu_hours_estimate=_number("gpu_hours_estimate"),
        usd_estimate=_number("usd_estimate"),
        limit_usd=_number("limit_usd"),
        budget_method=str(raw.get("budget_method") or ""),
        stop_rule=str(raw.get("stop_rule") or ""),
        approved_by=str(raw.get("approved_by") or ""),
        raw=raw,
    )


def require_budget(budget: Budget, *, explicit_limit_usd: float | None = None) -> Budget:
    """Пропустить прогон к старту или отказать.

    AD-8: «отсутствие сметы — блокирующее условие запуска».  Явный лимит
    (смоук-режим) снимает блокировку, но не выдаёт себя за смету: ``present``
    остаётся ``False`` и попадает в журнал как факт.
    """
    if budget.present:
        return budget
    if explicit_limit_usd is not None:
        return budget.with_explicit_limit(explicit_limit_usd)
    raise PretrainBudgetError(
        f"смета отсутствует: {budget.path} — запуск блокирован (AD-8); "
        "для смоука укажите явный --budget-limit-usd"
    )


def budget_breach(
    budget: Budget, *, tokens_seen: int, gpu_hours: float, usd_spent: float
) -> str | None:
    """Причина остановки или ``None``, если прогон в пределах сметы.

    Проверяются все три границы сметы по отдельности — причина называет ту,
    что сработала, чтобы журнал читался как план работ, а не как «что-то упало».
    """
    if budget.target_tokens is not None and tokens_seen > budget.target_tokens:
        return (
            f"target_tokens: пройдено {tokens_seen} токенов при цели "
            f"{budget.target_tokens} (AD-8)"
        )
    if budget.gpu_hours_estimate is not None and gpu_hours > budget.gpu_hours_estimate:
        return (
            f"gpu_hours: израсходовано {gpu_hours:.4f} ч при оценке "
            f"{budget.gpu_hours_estimate} ч (AD-8)"
        )
    if budget.limit_usd is not None and usd_spent > budget.limit_usd:
        return (
            f"limit_usd: израсходовано {usd_spent:.4f} USD при лимите "
            f"{budget.limit_usd} USD (AD-8)"
        )
    return None


# ---------------------------------------------------------------------------
# 8. Чекпойнты: Orbax + tree_hash, ретенция keep_last, курсор-манифест
# ---------------------------------------------------------------------------

#: Имя манифеста-курсора прогона в каталоге чекпойнтов.
CURSOR_MANIFEST_NAME = "cursor.json"


@dataclass(frozen=True)
class CheckpointRecord:
    """Запись о чекпойнте: шаг, каталог, tree_hash и курсор данных."""

    step: int
    path: Path
    tree_hash: str
    cursor: dict

    def to_json(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "path": self.path.name,
            "tree_hash": self.tree_hash,
            "cursor": self.cursor,
        }

    @classmethod
    def from_json(cls, data: Mapping[str, Any], directory: Path) -> "CheckpointRecord":
        return cls(
            step=int(data["step"]),
            path=directory / str(data["path"]),
            tree_hash=str(data["tree_hash"]),
            cursor=dict(data.get("cursor") or {}),
        )


class CheckpointManager:
    """Orbax-чекпойнты стадии претрейна с ретенцией и курсором resume.

    Раскладка::

        <directory>/cursor.json          — манифест-курсор: последний шаг + курсор данных
        <directory>/step-00000010/       — orbax-дерево {params, opt_state} + manifest.json

    ``tree_hash`` считается ``net/checkpoint.tree_hash`` — той же функцией, что
    в стадии SFT, поэтому хеш чекпойнта сопоставим между стадиями конвейера.
    """

    def __init__(self, directory: str | Path, *, keep_last: int = 2, prefix: str = "step-"):
        if keep_last < 1:
            raise ValueError("keep_last должен быть >= 1")
        self.directory = Path(directory)
        self.keep_last = int(keep_last)
        self.prefix = prefix

    @property
    def cursor_path(self) -> Path:
        return self.directory / CURSOR_MANIFEST_NAME

    def step_dir(self, step: int) -> Path:
        return self.directory / f"{self.prefix}{step:08d}"

    def save(
        self,
        *,
        step: int,
        params: Any,
        optimizer_state: Any,
        cursor: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Сохранить чекпойнт, обновить курсор-манифест и применить ретенцию.

        Параметры и состояние оптимизатора лежат в **отдельных** orbax-деревьях
        (``<step>/params``, ``<step>/opt_state``), а не в одном словаре: так
        ``tree_hash`` записи — это хеш дерева параметров, той же формы и той же
        функцией, что в стадии SFT, и хеши стадий сопоставимы.  Хеш словаря с
        путями ``['params', ...]`` дал бы другое число для тех же весов.
        """
        from net import checkpoint as checkpoint_mod

        target = self.step_dir(step)
        target.mkdir(parents=True, exist_ok=True)
        digest = checkpoint_mod.save_checkpoint(params, target / "params")
        state_digest = checkpoint_mod.save_checkpoint(optimizer_state, target / "opt_state")
        meta = {
            "step": int(step),
            "tree_hash": digest,
            "optimizer_state_hash": state_digest,
            "format": "orbax",
        }
        (target / "meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8"
        )
        record = CheckpointRecord(
            step=int(step), path=target, tree_hash=digest, cursor=dict(cursor)
        )
        self._write_cursor(record)
        self.prune()
        return {
            "step": record.step,
            "path": str(record.path),
            "tree_hash": digest,
            "optimizer_state_hash": state_digest,
        }

    def _write_cursor(self, record: CheckpointRecord) -> None:
        payload = {"schema": CURSOR_SCHEMA, **record.to_json()}
        tmp = self.cursor_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
        tmp.replace(self.cursor_path)

    def _checkpoint_dirs(self) -> list[tuple[int, Path]]:
        if not self.directory.is_dir():
            return []
        found = []
        for path in self.directory.iterdir():
            if path.is_dir() and path.name.startswith(self.prefix):
                suffix = path.name[len(self.prefix) :]
                if suffix.isdigit():
                    found.append((int(suffix), path))
        return sorted(found)

    def list(self) -> list[CheckpointRecord]:
        """Все наличные чекпойнты, по возрастанию шага."""
        records = []
        for step, path in self._checkpoint_dirs():
            meta = path / "meta.json"
            digest = ""
            if meta.is_file():
                try:
                    digest = str(json.loads(meta.read_text(encoding="utf-8")).get("tree_hash", ""))
                except (OSError, json.JSONDecodeError):
                    digest = ""
            records.append(CheckpointRecord(step=step, path=path, tree_hash=digest, cursor={}))
        return records

    def latest(self) -> dict[str, Any] | None:
        """Запись курсора-манифеста (шаг, каталог, tree_hash, курсор данных)."""
        if not self.cursor_path.is_file():
            return None
        try:
            payload = json.loads(self.cursor_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        record = CheckpointRecord.from_json(payload, self.directory)
        return {
            "step": record.step,
            "path": str(record.path),
            "tree_hash": record.tree_hash,
            "cursor": record.cursor,
        }

    def resume(self, *, target_params: Any, target_state: Any):
        """Восстановить последний чекпойнт: (params, opt_state, курсор данных)."""
        from net import checkpoint as checkpoint_mod

        latest = self.latest()
        if latest is None:
            raise PretrainDataError(
                f"курсор-манифест не найден: {self.cursor_path} — resume невозможен"
            )
        directory = Path(latest["path"])
        if not directory.is_dir():
            raise PretrainDataError(f"каталог чекпойнта отсутствует: {directory}")
        restored_params = checkpoint_mod.load_checkpoint(directory / "params", target=target_params)
        restored_state = checkpoint_mod.load_checkpoint(
            directory / "opt_state", target=target_state
        )
        digest = checkpoint_mod.tree_hash(restored_params)
        if latest["tree_hash"] and digest != latest["tree_hash"]:
            raise PretrainDataError(
                f"round-trip чекпойнта не совпал по tree_hash: {digest} != {latest['tree_hash']}"
            )
        return restored_params, restored_state, latest["cursor"]

    def prune(self) -> list[int]:
        """Оставить ``keep_last`` последних чекпойнтов; вернуть удалённые шаги."""
        import shutil

        dirs = self._checkpoint_dirs()
        removed = []
        for step, path in dirs[: max(0, len(dirs) - self.keep_last)]:
            shutil.rmtree(path)
            removed.append(step)
        return removed


# ---------------------------------------------------------------------------
# 9. Метрики прогона: loss / ток-в-с / MFU
# ---------------------------------------------------------------------------

#: Схема строки метрик прогона.
METRICS_SCHEMA = "pretrain-metrics/v1"

#: ADR-048: метки режима классификации параметров оптимизатора.  ``"adr-048"`` —
#: действующая классификация (embeddings / tied LM head / ViT -> AdamW),
#: ``"legacy"`` — прежняя «любой ``ndim == 2`` -> Muon».
CLASSIFICATION_ADR048 = "adr-048"
CLASSIFICATION_LEGACY = "legacy"


def optimizer_classification(legacy_muon_all_2d: bool) -> str:
    """Метка режима классификации оптимизатора для журнала (ADR-048).

    Один-в-один с флагом ``legacy_muon_all_2d``: ``True`` — прежнее поведение
    (``"legacy"``), ``False`` — классификация ADR-048 (``"adr-048"``).  Отдельная
    метка (а не голое булево поле) нужна, чтобы ноги A/B сшивались по журналам
    без чтения флага «наоборот».
    """
    return CLASSIFICATION_LEGACY if legacy_muon_all_2d else CLASSIFICATION_ADR048


class MetricsWriter:
    """Построчный jsonl-журнал метрик: одна строка на шаг, дописывается сразу."""

    def __init__(self, path: str | Path, *, schema: str = METRICS_SCHEMA):
        self.path = Path(path)
        self.schema = schema

    def log(self, record: Mapping[str, Any]) -> None:
        payload = {"schema": self.schema, **dict(record)}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    @staticmethod
    def read(path: str | Path) -> list[dict]:
        """Прочитать метрики построчно, устойчиво к обрыву записи (H5).

        jsonl дописывается на живой машине и может быть оборван преемпшном на
        середине строки: без построчного ``try`` одно битое окончание роняло
        чтение **всего** журнала.  Битую строку пропускаем — она не метрика, но и
        не повод потерять всё остальное (диагностика преемпшна особенно ценна
        именно тогда, когда файл оборван).
        """
        path = Path(path)
        if not path.is_file():
            return []
        rows = []
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            try:
                parsed = json.loads(stripped)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                rows.append(parsed)
        return rows


def step_flops(active_params: int, tokens: int, *, factor: float = 6.0) -> float:
    """FLOPs шага обучения: ``factor · N_active · tokens`` (forward+backward).

    Оценка по числу **активных** параметров (``net.model.active_param_count``)
    и только по параметрической части: вклад внимания (квадратичный по T) сюда
    не входит и в журнале помечается явно (``mfu_params_only``), чтобы MFU не
    выдавал себя за полный замер.
    """
    return float(factor) * float(active_params) * float(tokens)


def tflops_achieved(flops: float, *, seconds: float) -> float | None:
    """Достигнутые TFLOP/s — измерение, не зависящее от объявленного пика."""
    if seconds <= 0:
        return None
    return flops / seconds / 1e12


def mfu(flops: float, *, seconds: float, peak_tflops: float | None) -> float | None:
    """Model FLOPs Utilization: достигнутые TFLOP/s против объявленного пика.

    Без объявленного пика (или при нулевом времени шага) возвращается ``None``:
    необъявленный пик не даёт права ни на число, ни на вердикт.
    """
    if peak_tflops is None or peak_tflops <= 0 or seconds <= 0:
        return None
    achieved_tflops = flops / seconds / 1e12
    return achieved_tflops / float(peak_tflops)


# ---------------------------------------------------------------------------
# 9-бис. Пофазовый профиль шага (opt-in, диагностика)
# ---------------------------------------------------------------------------
#
# Фазы kda/mla/moe/ce fused внутри одного ``jax.jit(jax.value_and_grad(loss_fn))``
# (см. ``train`` ниже).  Разделение их на отдельные jit меняет граф вычислений, а
# вместе с ним — числа (loss/grads/tree_hash): это не дефект, а физика JIT.
# Поэтому профиль **не** режет штатный граф.  Он исполняет ДОПОЛНИТЕЛЬНЫЙ
# декомпозированный прогон тех же входов отдельными jitted-функциями, меряет
# каждую ногу host-таймером ``perf_counter`` + ``jax.block_until_ready`` (та же
# конвенция, что в ``net/tests/cost_method.py``) и выбрасывает результат:
# тренировка идёт штатным fused-путём, веса не меняются (паритет T-PP-3).
#
# Ноги делятся на два слоя.
#
# * **Сверка (reconciliation).**  Ноги ``sec_forward`` / ``sec_backward`` /
#   ``sec_backopt`` (``RECONCILIATION_LEGS``) меряются **реальными графами шага**:
#   ``sec_forward`` — тот же ``loss_fn`` под ``jax.jit``, ``sec_backward`` —
#   разность «fwd+bwd минус forward» на том же ``grad_fn`` (обратный проход нельзя
#   вызвать сам по себе: он живёт на активациях прямого, поэтому честная оценка —
#   разность, а не отдельный ``jax.grad``-проход), ``sec_backopt`` — шаг
#   оптимизатора.  Их сумма — вычислительная часть шага; ``sec_other`` =
#   ``шаг − сумма`` — **остаток** (хост-диспетчеризация, приведение деревьев,
#   ``device_get`` и прочая неарифметика), а НЕ «прочее», и подаётся как остаток.
#   Доля покрытия ``reconciliation_pct`` уезжает в метрики; ниже
#   ``UNEXPLAINED_SHARE_THRESHOLD`` срабатывает находка ``unexplained_share_high`` —
#   непокрытая доля шага не растворяется в тишине (ADR-011).
# * **Декомпозиция (attribution).**  Ноги ``sec_kda`` / ``sec_mla`` / ``sec_moe`` /
#   ``sec_ce`` — изолированные проходы по своему стеку слоёв (``sec_kda`` — все
#   KDA-слои, ``sec_mla`` — все MLA с пробросом пула ADR-012, ``sec_moe`` — все
#   канальные MLP/MoE, ``sec_ce`` — головы NTP+MTP): они отвечают на вопрос «какой
#   компонент прямого прохода доминирует».  Проходы не интерливаются, как в
#   ``model.forward`` (AttnRes-коррекция и остаточная арифметика между ними в
#   разбиении не участвуют), поэтому они НЕ суммируются со сверкой — иначе
#   forward считался бы дважды.  ``sec_loader`` меряется отдельно и в сверку не
#   входит: получение батча идёт ДО ``tick``, то есть вне ``step_seconds`` (см.
#   ветку лоадера в ``train``), и подаётся как справка, а не как нога шага.
# * **Декомпозиция backward (attribution).**  Ноги ``sec_bwd_kda`` / ``sec_bwd_mla``
#   / ``sec_bwd_moe`` / ``sec_bwd_ce`` / ``sec_bwd_other`` — отдельные
#   ``jax.jit(jax.grad(...))``-проходы по скалярному отклику того же подмножества,
#   что и forward-ноги (оболочка ``sec_bwd_other`` — лукап эмбеддинга + финальная
#   RMSNorm).  Мотивация: backward l3-full — 73 % шага, и удвоение пика матмулов
#   (bf16) ускоряет его лишь ×1.05, то есть он упирается в память/трафик
#   активаций, а не в FLOPs; разложить это можно только по компонентам.
#   Grad по подмножеству **не равен** «доле» полного backward из-за общих
#   активаций, поэтому сумма ног подаётся рядом с ``sec_backward`` как ориентир
#   (``bwd_reconciliation``, с пометкой и находкой ``backward_legs_diverge`` при
#   уходе дальше ``BWD_RECONCILIATION_TOLERANCE``), а не как партиция — расхождение
#   ожидаемо, но обязано быть видно.
# * **Память ног.**  Рядом с временем каждой ноги подаётся пиковая XLA-память её
#   скомпилированного исполнителя (``PHASE_MEMORY_FIELD``, разбор
#   ``memory_analysis``): она отделяет «упор в память» от «упор в арифметику» и
#   не входит в арифметику шага.  Меряется один раз на прогреве (компиляция), не
#   на каждом профильном шаге.
#
# Фаза, которую собрать или исполнить не удалось, записывается ``null``:
# неизмеренное не выдаётся за измеренное (ADR-011).

#: Поля-секунды в записи метрик (контракт дельты) — в порядке вывода.
PHASE_PROFILE_FIELDS = (
    "sec_kda",
    "sec_mla",
    "sec_moe",
    "sec_ce",
    "sec_bwd_kda",
    "sec_bwd_mla",
    "sec_bwd_moe",
    "sec_bwd_ce",
    "sec_bwd_other",
    "sec_backopt",
    "sec_forward",
    "sec_backward",
    "sec_loader",
    "sec_other",
)

#: Ноги, сумма которых сверяется со временем шага.  Декомпозиция прямого прохода
#: (kda/mla/moe/ce) сюда НЕ входит — она не партиционирует шаг; ``sec_loader``
#: тоже вне: лоадер меряется до ``tick`` и в ``step_seconds`` не попадает.
#: Ноги backward-декомпозиции (``BWD_LEG_FIELDS``) — тоже вне: они меряют не
#: части шага, а стоимость обратного прохода подмножеств (см. их сверку ниже).
RECONCILIATION_LEGS = ("sec_forward", "sec_backward", "sec_backopt")

#: Порог покрытия шага измеренными ногами: ниже — находка, а не тишина.
UNEXPLAINED_SHARE_THRESHOLD = 0.95
FINDING_UNEXPLAINED_SHARE_HIGH = "unexplained_share_high"

#: Ноги backward-декомпозиции (диагностика): каждая — свой ``jax.grad``-проход по
#: скалярному отклику своего подмножества модели (``sec_bwd_kda`` — KDA-слои,
#: ``sec_bwd_mla`` — MLA с пулом ADR-012, ``sec_bwd_moe`` — канальные MLP/MoE,
#: ``sec_bwd_ce`` — головы NTP+MTP, ``sec_bwd_other`` — оболочка остатка: лукап
#: эмбеддинга + финальная RMSNorm).  Это **не** доли полного ``sec_backward``:
#: у подмножества с полным проходом общие активации (лукап эмбеддинга,
#: остаточный поток), поэтому сумма ног не обязана сходиться с ``sec_backward``.
#: Их стоимость — ключ к вопросу «на что уходит backward» (память активаций или
#: FLOPs): рядом с временем каждой ноги подаётся пиковая XLA-память её
#: исполнителя (``PHASE_MEMORY_FIELD``).
BWD_LEG_FIELDS = (
    "sec_bwd_kda",
    "sec_bwd_mla",
    "sec_bwd_moe",
    "sec_bwd_ce",
    "sec_bwd_other",
)

#: Поле-носитель пиковой XLA-памяти скомпилированных исполнителей ног (байт).
#: Ключи — имена ног из ``PHASE_PROFILE_FIELDS``; значение — разбор
#: ``memory_analysis`` (``_executable_memory``) или ``None`` (бэкенд не отдал).
#: Ключ ``sec_backward`` несёт память графа fwd+bwd (``grad_fn``): отдельного
#: исполнителя «только обратный проход» у шага нет.
PHASE_MEMORY_FIELD = "phase_memory_bytes"

#: Поле-носитель сверки суммы backward-ног с полным ``sec_backward`` (ориентир).
BWD_RECONCILIATION_FIELD = "bwd_reconciliation"

#: Допуск ориентира: сумма backward-ног против полного ``sec_backward``.
#: Широкий (±20 %), потому что ноги и полный проход меряют разное: подмножества
#: делят общие активации, а нога ``sec_backward`` — ещё и разность времён.
BWD_RECONCILIATION_TOLERANCE = 0.20

#: Находка: сумма backward-ног ушла от полного backward дальше допуска.
FINDING_BACKWARD_LEGS_DIVERGE = "backward_legs_diverge"

#: Пометка к сверке backward-ног — «ориентир, не партиция» (видна в записи).
BWD_RECONCILIATION_NOTE = (
    "ориентир, не партиция: ноги — отдельные grad-проходы по подмножествам; "
    "с полным backward у них общие активации (лукап эмбеддинга, остаточный "
    "поток), поэтому расхождение ожидаемо; sec_backward — разность времён"
)


def reconcile_phases(
    step_seconds: float | None, legs: Mapping[str, Any]
) -> dict[str, Any]:
    """Сверка суммы измеренных ног со временем шага (чистая арифметика).

    Суммируются только ноги ``RECONCILIATION_LEGS``, присутствующие числом:
    отсутствующая (``None``) нога честно снижает покрытие и может поднять находку,
    а не подставляется нулём.  ``sec_other`` — **остаток** шага за вычетом суммы
    ног, не «прочее»; ``reconciliation_pct`` — покрытие в процентах; ``finding`` —
    ``unexplained_share_high`` при покрытии ниже ``UNEXPLAINED_SHARE_THRESHOLD``.

    Шаг неизвестен/неположителен (``None`` или ``<= 0``) — сверка неопределена,
    все три поля ``None``: без деления на ноль и без вымысла.
    """
    measured = {
        name: float(legs[name])
        for name in RECONCILIATION_LEGS
        if isinstance(legs.get(name), (int, float))
        and not isinstance(legs.get(name), bool)
    }
    if (
        not isinstance(step_seconds, (int, float))
        or isinstance(step_seconds, bool)
        or step_seconds <= 0
    ):
        return {"reconciliation_pct": None, "sec_other": None, "finding": None}
    reconciled = math.fsum(measured.values())
    span = float(step_seconds)
    pct = 100.0 * reconciled / span
    finding = (
        FINDING_UNEXPLAINED_SHARE_HIGH
        if pct < UNEXPLAINED_SHARE_THRESHOLD * 100.0
        else None
    )
    return {
        "reconciliation_pct": pct,
        "sec_other": span - reconciled,
        "finding": finding,
    }


def reconcile_backward_legs(
    sec_backward: float | None, legs: Mapping[str, Any]
) -> dict[str, Any]:
    """Сверка суммы backward-ног с полным ``sec_backward`` — **ориентир**.

    Ноги ``BWD_LEG_FIELDS`` складываются только те, что присутствуют числом:
    отсутствующая (``None``) не подставляется нулём (ADR-011).  ``pct`` — сумма
    ног к полному проходу в процентах; ``finding`` — ``backward_legs_diverge``
    при уходе от 100 % дальше ``BWD_RECONCILIATION_TOLERANCE``.  ``note`` —
    пометка о том, что это ориентир, а не партиция: ноги меряют графы
    подмножеств (у них общие активации с полным проходом), а ``sec_backward`` —
    разность времён, поэтому расхождение ожидаемо — но обязано быть видно, а не
    растворяться в тишине (ADR-011).

    Полный backward неизвестен/неположителен — сверка не определена: ``pct`` и
    ``finding`` ``None``, сумма ног (если есть) и пометка остаются.  Ни одна
    нога не измерена — сумма тоже ``None``.
    """
    measured = {
        name: float(legs[name])
        for name in BWD_LEG_FIELDS
        if isinstance(legs.get(name), (int, float))
        and not isinstance(legs.get(name), bool)
    }
    result: dict[str, Any] = {
        "legs_sum": math.fsum(measured.values()) if measured else None,
        "full_backward": None,
        "pct": None,
        "tolerance": BWD_RECONCILIATION_TOLERANCE,
        "finding": None,
        "note": BWD_RECONCILIATION_NOTE,
    }
    if not measured:
        return result
    if (
        not isinstance(sec_backward, (int, float))
        or isinstance(sec_backward, bool)
        or sec_backward <= 0
    ):
        return result
    total = result["legs_sum"]
    full = float(sec_backward)
    pct = 100.0 * total / full
    result["full_backward"] = full
    result["pct"] = pct
    result["finding"] = (
        FINDING_BACKWARD_LEGS_DIVERGE
        if abs(pct - 100.0) > BWD_RECONCILIATION_TOLERANCE * 100.0
        else None
    )
    return result


def _executable_memory(compiled) -> dict[str, int | None] | None:
    """Пиковая XLA-память скомпилированного исполнителя (байт); ``None`` — нет.

    ``peak_bytes`` — temp + аргументы + выход (та же свёртка, что в
    ``tools/remat_policy_smoke.py`` и ``tools/kda_phase_profile.py``).  Бэкенд не
    отдал разбор (``memory_analysis`` бросил или вернул ``None``) — честный
    ``None``, а не выдуманный ноль (ADR-011).
    """
    try:
        analysis = compiled.memory_analysis()
    except Exception:
        return None
    if analysis is None:
        return None
    temp = getattr(analysis, "temp_size_in_bytes", None)
    argument = getattr(analysis, "argument_size_in_bytes", None)
    output = getattr(analysis, "output_size_in_bytes", None)
    peak = getattr(analysis, "peak_memory_in_bytes", None)
    if peak is None and None not in (temp, argument, output):
        peak = int(temp) + int(argument) + int(output)
    return {
        "peak_bytes": None if peak is None else int(peak),
        "temp_bytes": None if temp is None else int(temp),
        "argument_bytes": None if argument is None else int(argument),
        "output_bytes": None if output is None else int(output),
    }


def render_phase_memory(memory: Mapping[str, Any] | None) -> str:
    """Строка пиковой памяти ног: ``имя=NN.NМиБ`` / ``имя=null`` (по полям ног).

    Чистая функция — годится и для печати, и для теста контракта записи.
    """
    memory = memory or {}
    parts = []
    for name in PHASE_PROFILE_FIELDS:
        entry = memory.get(name)
        peak = entry.get("peak_bytes") if isinstance(entry, Mapping) else None
        rendered = "null" if peak is None else f"{peak / (1024.0 * 1024.0):.1f}МиБ"
        parts.append(f"{name}={rendered}")
    return ", ".join(parts)


def kpi_window_values(
    step_tokens_per_sec,
    *,
    window: int,
    exclude_first: bool = True,
) -> list[float]:
    """Значения ток/с для KPI-медианы: хвост ``window`` шагов без первого.

    ADR-048 Amendment, «первый шаг вне KPI»: первый шаг ноги компилирует jit-граф
    (``grad_fn`` и шаг оптимизатора), и его время несоизмеримо со временем
    установившегося шага — включать его в медиану значило бы мерить компиляцию, а
    не пропускную способность.  Единственное место, где задано правило окна:
    им пользуются и медиана, и печатаемый размер окна.  Исключение — только из
    агрегата: построчные значения в ``metrics.jsonl`` остаются нетронутыми (сырой
    носитель), а ``exclude_first=False`` возвращает прежнее окно для «до/после».
    """
    values = list(step_tokens_per_sec)
    if exclude_first and len(values) > 1:
        values = values[1:]
    return values[-max(1, int(window)):]


def kpi_tokens_per_sec(
    step_tokens_per_sec,
    *,
    window: int,
    exclude_first: bool = True,
) -> float | None:
    """Медиана ток/с по окну :func:`kpi_window_values`; ``None`` — измерений нет."""
    values = kpi_window_values(
        step_tokens_per_sec, window=window, exclude_first=exclude_first
    )
    if not values:
        return None
    import numpy as np

    return float(np.median(values))


def _phase_kda_stack(params, cfg, input_ids, chunk_size):
    """KDA-фаза: проход по всем KDA-слоям на тех же входах (диагностика)."""
    import jax

    from net import kda as kda_mod
    from net import model as model_mod
    from net.norm import rms_norm

    h = params.embedding[input_ids]
    for index, block in enumerate(params.layers):
        if model_mod.layer_kind(cfg, index) != "kda":
            continue
        attn = jax.vmap(lambda xb: kda_mod.apply_kda(block.attn, cfg, xb, chunk_size))(
            rms_norm(h, block.norm_attn)
        )
        h = h + attn
    return h


def _phase_mla_stack(params, cfg, input_ids):
    """MLA-фаза: проход по всем MLA-слоям; пул ADR-012 пробрасывается, как в модели.

    Режим слоя читается у ``mla_mod.layer_mode`` по тому же порядковому номеру
    MLA-слоя, что и ``model._mla_ordinal_at`` (порядковый — со сдвигом на
    диагностический dense-standard-префикс).
    """
    from net import mla as mla_mod
    from net import model as model_mod
    from net.norm import rms_norm

    h = params.embedding[input_ids]
    pool = None
    prefix = int(cfg.dense_standard_layers)
    for index, block in enumerate(params.layers):
        kind = model_mod.layer_kind(cfg, index)
        if kind == "kda":
            continue
        x = rms_norm(h, block.norm_attn)
        if kind == "dense-standard":
            # Диагностический блок VERIFICATION-LEG идёт плотным оракулом
            # безусловно (та же ветка, что в ``model._block_delta``).
            out = mla_mod._dense_apply(block.attn, cfg, x)
        else:
            out, pool = mla_mod.apply_with_pool(
                block.attn,
                cfg,
                x,
                pool=pool,
                mode=mla_mod.layer_mode(cfg, (index - prefix) // 4),
            )
        h = h + out
    return h


def _phase_moe_stack(params, cfg, input_ids):
    """MoE-фаза: проход по канальному микшеру всех слоёв (dense MLP и LatentMoE)."""
    from net import mlp as mlp_mod
    from net import moe as moe_mod
    from net.norm import rms_norm

    h = params.embedding[input_ids]
    for block in params.layers:
        x = rms_norm(h, block.norm_mlp)
        if isinstance(block.mlp, moe_mod.LatentMoEParams):
            h = h + moe_mod.apply(block.mlp, cfg, x)
        else:
            h = h + mlp_mod.apply(block.mlp, cfg, x)
    return h


def _phase_ce_head(params, cfg, hidden, input_ids, chunk_size, ce_tokens):
    """CE-фаза: головы NTP (chunked или наивная) + MTP — та же формула, что в loss."""
    from net import model as model_mod

    if ce_tokens > 0:
        ntp = model_mod._chunked_cross_entropy(
            hidden[:, :-1], input_ids[:, 1:], params.embedding, ce_tokens
        )
    else:
        ntp = model_mod._cross_entropy(
            hidden[:, :-1] @ params.embedding.T, input_ids[:, 1:]
        )
    aux = model_mod.mtp_loss(
        params, cfg, hidden, input_ids, chunk_size, ce_chunk_tokens=ce_tokens
    )
    return ntp + aux


def _phase_other_stack(params, cfg, input_ids):
    """Остаток-оболочка: лукап эмбеддинга + финальная RMSNorm (диагностика).

    Это та часть графа ``model.forward``, что не попадает ни в один стековый
    разбор (``_phase_kda_stack`` и соседи): скаттер-аддиция градиента в таблицу
    эмбеддингов (``params.embedding[input_ids]``) и обратный проход финальной
    нормы (``rms_norm(h, params.norm_final)``).  Своих слоёв у ноги нет, поэтому
    её backward — не «доля» полного прохода, а мера стоимости именно этой части.
    """
    from net.norm import rms_norm

    h = params.embedding[input_ids]
    return rms_norm(h, params.norm_final)


class _PhaseProfiler:
    """Декомпозированный прогон фаз с host-таймерами (только opt-in).

    Строится лишь в профильном режиме: дефолтный путь не платит ни компиляцией,
    ни временем.  Jitted-функции фаз создаются один раз; первый вызов каждой ноги
    компилирует граф, поэтому перед первым замером идёт прогревочный прогон, чьё
    время в замер не попадает (иначе в ``sec_*`` попала бы компиляция, а не
    стоимость фазы на шаге).  Отказ любой ноги (нет хука, ошибка компиляции) —
    ``null`` в её поле, а не падение тренировки: профиль диагностический.

    ``grad_fn``/``forward_fn`` приходят снаружи — это **реальные** графы шага
    (forward = тот же ``loss_fn`` под ``jax.jit``, fwd+bwd = тот же ``grad_fn``),
    поэтому ``sec_forward``/``sec_backward`` меряют шаг, а не его копию.  Ноги
    декомпозиции строятся внутри из ``cfg``.

    Ноги backward (``BWD_LEG_FIELDS``) — свои ``jax.grad``-проходы по скалярному
    отклику подмножества (``jnp.sum`` стекового выхода; ``sec_bwd_ce`` — прямо по
    лоссу головы NTP+MTP): grad по подмножеству не равен «доле» полного backward
    (общие активации), поэтому их сумма идёт ориентиром (``bwd_reconciliation``),
    а не партицией.
    """

    def __init__(self, cfg, train_config, *, grad_fn, forward_fn):
        import jax
        import jax.numpy as jnp

        chunk_size = int(train_config.chunk_size)
        ce_tokens = int(cfg.ce_chunk_tokens)
        self._warmed = False
        # Реальные графы шага — только для ног сверки (forward/backward).
        self._grad_fn = grad_fn
        self._forward_fn = forward_fn
        # ``cfg`` захвачен замыканием (статическая константа графа), а не передан
        # аргументом jit: конфиг модели — не данные шага.
        self._fns = {
            "kda": jax.jit(lambda p, ids: _phase_kda_stack(p, cfg, ids, chunk_size)),
            "mla": jax.jit(lambda p, ids: _phase_mla_stack(p, cfg, ids)),
            "moe": jax.jit(lambda p, ids: _phase_moe_stack(p, cfg, ids)),
            "ce": jax.jit(
                lambda p, hidden, ids: _phase_ce_head(
                    p, cfg, hidden, ids, chunk_size, ce_tokens
                )
            ),
        }
        # Ноги backward: отдельный ``jax.grad`` на подмножество.  ``sec_bwd_ce``
        # дифференцирует голову по параметрам (``argnums=0``); ``hidden`` —
        # данные (активация канального прохода), а не параметр.
        self._bwd_fns = {
            "bwd_kda": jax.jit(
                jax.grad(
                    lambda p, ids: jnp.sum(_phase_kda_stack(p, cfg, ids, chunk_size))
                )
            ),
            "bwd_mla": jax.jit(
                jax.grad(lambda p, ids: jnp.sum(_phase_mla_stack(p, cfg, ids)))
            ),
            "bwd_moe": jax.jit(
                jax.grad(lambda p, ids: jnp.sum(_phase_moe_stack(p, cfg, ids)))
            ),
            "bwd_ce": jax.jit(
                jax.grad(
                    lambda p, hidden, ids: _phase_ce_head(
                        p, cfg, hidden, ids, chunk_size, ce_tokens
                    ),
                    argnums=0,
                )
            ),
            "bwd_other": jax.jit(
                jax.grad(lambda p, ids: jnp.sum(_phase_other_stack(p, cfg, ids)))
            ),
        }
        #: Пиковая XLA-память ног: заполняется один раз на прогреве (там же, где
        #: компилируются исполнители), читается справкой на каждом профильном шаге.
        #: Все ключи ``PHASE_PROFILE_FIELDS`` присутствуют сразу (``None`` — нет
        #: исполнителя/бэкенд не отдал разбор).
        self._memory: dict[str, Any] = {
            name: None for name in PHASE_PROFILE_FIELDS
        }

    def measure(
        self,
        *,
        params,
        batch,
        grads,
        master,
        state,
        lr,
        step_fn,
        step_seconds,
        loader_seconds,
    ):
        """Секунды фаз дополнительного прогона; недоступная фаза — ``None``.

        Возвращает запись со всеми полями ``PHASE_PROFILE_FIELDS`` плюс
        ``reconciliation_pct``/``sec_other``/``finding`` (сверка
        ``reconcile_phases``), карту пиковой памяти ног (``PHASE_MEMORY_FIELD``,
        снятую на прогреве) и сверку backward-ног (``BWD_RECONCILIATION_FIELD``,
        ориентир с пометкой).  ``loader_seconds`` — справка вне шага: лоадер
        меряется до ``tick`` и в ``step_seconds`` не входит, поэтому он НЕ нога
        сверки.
        """
        if not self._warmed:
            self._warmed = True
            self._sweep(
                params=params, batch=batch, grads=grads, master=master,
                state=state, lr=lr, step_fn=step_fn, time_it=False,
            )
        record = self._sweep(
            params=params, batch=batch, grads=grads, master=master,
            state=state, lr=lr, step_fn=step_fn, time_it=True,
        )
        record["sec_loader"] = (
            float(loader_seconds)
            if isinstance(loader_seconds, (int, float))
            and not isinstance(loader_seconds, bool)
            else None
        )
        record.update(reconcile_phases(step_seconds, record))
        record[PHASE_MEMORY_FIELD] = dict(self._memory)
        record[BWD_RECONCILIATION_FIELD] = reconcile_backward_legs(
            record.get("sec_backward"), record
        )
        return record

    def _sweep(self, *, params, batch, grads, master, state, lr, step_fn, time_it):
        import jax

        record: dict[str, float | None] = {name: None for name in PHASE_PROFILE_FIELDS}

        def run(thunk):
            if not time_it:
                value = thunk()
                jax.block_until_ready(value)
                return None, value
            start = time.perf_counter()
            value = thunk()
            jax.block_until_ready(value)
            return time.perf_counter() - start, value

        def remember_memory(field, lower, graph):
            """Пиковая память исполнителя ноги — один раз, на прогреве."""
            if time_it:
                return
            try:
                memory = _executable_memory(lower().compile())
            except Exception:
                memory = None
            self._memory[field] = None if memory is None else {"graph": graph, **memory}

        def attempt(field, thunk, lower=None, graph=None):
            try:
                elapsed, value = run(thunk)
            except Exception:
                return None, None
            record[field] = elapsed
            if lower is not None:
                remember_memory(field, lower, graph or "forward")
            return elapsed, value

        # Скрытое состояние для CE-ноги — выход канального прохода: настоящая
        # активация той же формы (B, T, hidden), поэтому голова меряется без
        # второго прохода по бэкбону; её стоимость задаётся формой
        # (B, T, vocab, ce_chunk_tokens), а не значениями.
        hidden = None
        for field, name in (("sec_kda", "kda"), ("sec_mla", "mla"), ("sec_moe", "moe")):
            fn = self._fns[name]
            _, value = attempt(
                field,
                lambda fn=fn: fn(params, batch),
                lambda fn=fn: fn.lower(params, batch),
            )
            if name == "moe" and value is not None:
                hidden = value
        if hidden is None:
            hidden = params.embedding[batch]
        ce_fn = self._fns["ce"]
        attempt(
            "sec_ce",
            lambda: ce_fn(params, hidden, batch),
            lambda: ce_fn.lower(params, hidden, batch),
        )
        attempt(
            "sec_backopt",
            lambda: step_fn(master, grads, state, lr),
            lambda: step_fn.lower(master, grads, state, lr),
            graph="optimizer-step",
        )

        # Backward-декомпозиция: те же подмножества, что и forward-ноги, но grad
        # собственного скалярного отклика.  ``sec_bwd_ce`` берёт ту же ``hidden``,
        # что и ``sec_ce``; ``sec_bwd_other`` — оболочку (эмбеддинг + финальная
        # норма).  Результаты выбрасываются: это диагностика.
        for field, name in (
            ("sec_bwd_kda", "bwd_kda"),
            ("sec_bwd_mla", "bwd_mla"),
            ("sec_bwd_moe", "bwd_moe"),
            ("sec_bwd_other", "bwd_other"),
        ):
            fn = self._bwd_fns[name]
            attempt(
                field,
                lambda fn=fn: fn(params, batch),
                lambda fn=fn: fn.lower(params, batch),
                graph="grad",
            )
        bwd_ce = self._bwd_fns["bwd_ce"]
        attempt(
            "sec_bwd_ce",
            lambda: bwd_ce(params, hidden, batch),
            lambda: bwd_ce.lower(params, hidden, batch),
            graph="grad",
        )

        # Ноги сверки: forward — реальный ``loss_fn`` под jit, backward — разность
        # «fwd+bwd минус forward» на реальном ``grad_fn``.  Оба прохода читают те
        # же параметры и выбрасывают результат: веса не меняются.  Память ноги
        # ``sec_backward`` — память графа fwd+bwd: отдельного backward-исполнителя
        # у шага нет, и разбор именно этого графа и есть ключ к «упор в память».
        forward_seconds, _ = attempt(
            "sec_forward",
            lambda: self._forward_fn(params, batch),
            lambda: self._forward_fn.lower(params, batch),
        )
        grad_memory = None
        try:
            if not time_it:
                grad_memory = _executable_memory(
                    self._grad_fn.lower(params, batch).compile()
                )
        except Exception:
            grad_memory = None
        if grad_memory is not None:
            self._memory["sec_backward"] = {"graph": "fwd+bwd", **grad_memory}
        fwd_bwd_seconds = None
        try:
            fwd_bwd_seconds, _ = run(lambda: self._grad_fn(params, batch))
        except Exception:
            fwd_bwd_seconds = None
        if forward_seconds is not None and fwd_bwd_seconds is not None:
            # Разность может уйти в минус на шуме таймера (forward в fwd+bwd
            # дешевле изолированного) — тогда обратный проход не различим, 0.0.
            record["sec_backward"] = max(0.0, fwd_bwd_seconds - forward_seconds)
        return record


# ---------------------------------------------------------------------------
# 10. Цикл претрейна
# ---------------------------------------------------------------------------


@dataclass
class TrainConfig:
    """Параметры цикла претрейна.

    ``steps`` — сколько шагов исполняет **эта нога**; ``total_steps`` — горизонт
    расписания (по умолчанию равен ``steps``).  Разделение нужно для resume:
    продолжение обязано идти по тому же LR, что непрерывный прогон, а не
    растягивать warmup/decay на длину ноги.
    """

    steps: int = 1
    total_steps: int | None = None
    lr: float = 1e-2
    schedule: str = "cosine"  # cosine | wsd
    warmup_ratio: float = 0.01
    decay_ratio: float = 0.05
    seed: int = 0
    chunk_size: int = 64
    #: Микробатч (число последовательностей в одном forward), ADR-031 delta B.
    #: Сам размер батча задаёт даталоадер (``pretrain_run.py`` передаёт сюда
    #: значение ``--micro-batch``); поле фиксирует его для журнала и проверки.
    #: ``1`` — текущее поведение: один forward на шаг оптимизатора.
    micro_batch: int = 1
    #: Накопление градиентов до шага оптимизатора (ADR-031: батч 256K
    #: накоплением).  ``0`` — выключено (один микробатч на шаг, текущее
    #: поведение); ``T > 0`` — накопить ``ceil(T / (micro_batch * seq_len))``
    #: микробатчей и шагнуть один раз.  Токены в метрике шага — фактически
    #: потреблённые накоплением (не номинальный батч).
    accum_tokens: int = 0
    #: Каждые ``kpi_every`` шагов оптимизатора печатать медиану ток/с по
    #: последним ``kpi_window`` шагам (носитель KPI — ``metrics.jsonl``).
    kpi_every: int = 20
    kpi_window: int = 20
    #: Opt-in пофазовый профиль шага (диагностика узкого места).  ``False`` —
    #: продакшн-путь: ни jit-функций фаз, ни host-таймеров, ни полей в метриках.
    #: ``True`` — на каждом kpi-интервальном шаге исполняется **дополнительный**
    #: декомпозированный прогон тех же входов (см. раздел «9-бис»), его числа
    #: уезжают в ``metrics.jsonl`` полями ``sec_*``/``phase_profile`` и
    #: выбрасываются: градиенты и шаг оптимизатора берутся штатным fused-путём,
    #: поэтому веса и ``tree_hash`` профильного прогона совпадают с дефолтным.
    phase_profile: bool = False
    #: ``True`` — remat графа: память активаций падает ценой пересчёта.
    grad_checkpointing: bool = False
    #: ``full`` (nothing_saveable) | ``selective`` (dots без batch-осей).
    grad_checkpointing_policy: str = "full"
    #: ``float32`` — прямая точность (parity со старым тренером);
    #: ``bfloat16`` — bf16-параметры при fp32-мастере.
    param_dtype: str = "float32"
    checkpoint_every: int = 0  # 0 — чекпойнты по числу шагов выключены
    #: К1 (runbook §3): каденс чекпойнтов по **минутам**, а не шагам.  Нужен на
    #: аренде: длительность шага плавает (преемпшн, соседи по железу), и «30 минут»
    #: — риск-метрика потери, тогда как «N шагов» — нет.  Обе настройки складываются
    #: по ИЛИ: что наступит раньше, то и сохраняет.
    checkpoint_every_min: float | None = None
    keep_last: int = 2
    ckpt_dir: Path | None = None
    metrics_path: Path | None = None
    #: Пик железа для MFU; не объявлен — MFU не выдумывается (None).
    peak_tflops: float | None = None
    peak_tflops_source: str = ""
    #: Ставка аренды: без неё стоп-правило по USD не оценивается (и это видно).
    usd_per_gpu_hour: float | None = None
    log_every: int = 0
    #: К5: файл-стоп (килл-свитч ватчдога).  Существование файла останавливает
    #: прогон на ближайшем чекпойнте — независимо от того, дошло ли дело до
    #: бюджетного порога.  Путь разрешает CLI (argv → env STOP_FILE → курсор).
    stop_file: Path | None = None
    #: Как часто проверять ``stop_file`` (в шагах); 1 — каждый шаг.
    stop_check_every: int = 1
    #: Вид источника данных (``raw`` | ``packed``): попадает в курсор и сверяется
    #: при resume (H4) — продолжать packed-прогон raw-курсором значило бы молча
    #: поставить поток на чужую позицию.
    data_kind: str = "raw"
    #: ADR-048: число Newton-Schulz итераций Muon (дефолт — прежние 5; решение
    #: о снижении принимает архитектор по замеру, здесь только параметр).
    ns_steps: int = 5
    #: ADR-048: вернуть прежнюю классификацию параметров («любой ndim==2 ->
    #: Muon», embeddings/LM head включительно) — для честного сравнения «до/после»
    #: на одной ревизии кода.  ``False`` — решение ADR-048 (emb/head -> AdamW).
    legacy_muon_all_2d: bool = False


@dataclass
class TrainResult:
    """Итог ноги прогона: шаги, лоссы, веса, чекпойнт и причина остановки."""

    losses: list[float]
    steps_done: int
    start_step: int
    tokens_seen: int
    params: Any
    master_params: Any
    #: Хеш обслуживаемых весов (``params``) — сопоставим со стадией SFT.
    tree_hash: str
    #: Хеш fp32-мастера — ровно то, что лежит в чекпойнте.
    master_tree_hash: str
    lr_history: list[float]
    stop_reason: str | None
    stopped_by_budget: bool
    timings: dict
    checkpoint: dict | None
    metrics_path: str | None
    budget_report: dict
    grad_checkpointing: bool
    param_dtype: str


def _checkpoint_policy(name: str):
    """Политика remat по имени: ``full`` — не сохранять ничего."""
    import jax

    if name == "full":
        return jax.checkpoint_policies.nothing_saveable
    if name == "selective":
        return jax.checkpoint_policies.dots_with_no_batch_dims_saveable
    raise ValueError(f"неизвестная политика grad-checkpointing: {name!r}")


def train(
    cfg,
    batches: Iterable[Any],
    *,
    train_config: TrainConfig,
    budget: Budget,
    resume_from: CheckpointManager | None = None,
    loader: PretrainMixLoader | None = None,
) -> TrainResult:
    """Нога претрейна: шаги оптимизатора по потоку батчей.

    ``budget`` обязателен и не имеет значения по умолчанию: AD-8 объявляет
    отсутствие сметы блокирующим условием запуска, поэтому «забыть» про смету
    нельзя — только предъявить её (пусть и безлимитную в смоуке).

    ``loader`` нужен только для курсора resume: он отдаёт позицию потока данных
    на момент чекпойнта.  Без него чекпойнт восстанавливает веса, но не данные.

    **Профильный режим** (``train_config.phase_profile``, opt-in).  С флагом на
    каждом kpi-интервальном шаге исполняется ДОПОЛНИТЕЛЬНЫЙ декомпозированный
    прогон тех же входов (раздел «9-бис»), и его секунды уезжают в метрики полями
    ``sec_*`` + ``phase_profile: true``.  Числа фаз диагностические: результат
    декомпозиции выбрасывается, тренировка идёт штатным fused-путём (один
    ``jax.jit(value_and_grad(loss_fn))``), поэтому веса и ``tree_hash`` не зависят
    от флага.  Без флага не строится ни одной jit-функции фазы и не заводится ни
    одного host-таймера — продакшн-ветка байт-в-байт прежняя.
    """
    import jax
    import jax.numpy as jnp

    from net import checkpoint as checkpoint_mod
    from net import model, optimizer

    # Повторный гейт внутри цикла: лимит уже мог быть выдан смоук-режимом
    # (pretrain_run.py: explicit_limit_usd -> Budget.limit_usd), не требуем его снова.
    budget = require_budget(budget, explicit_limit_usd=getattr(budget, "limit_usd", None))
    if train_config.schedule not in ("cosine", "wsd"):
        raise ValueError(f"неизвестное расписание: {train_config.schedule!r}")
    if train_config.param_dtype not in ("float32", "bfloat16"):
        raise ValueError(f"неизвестная точность параметров: {train_config.param_dtype!r}")
    # ADR-031 delta B: micro-batch >= 1 and a non-negative accumulation target;
    # a zero/negative micro-batch would make the accumulation count ill-defined,
    # and a negative target has no meaning (0 already means "off").
    if int(train_config.micro_batch) < 1:
        raise ValueError(
            f"micro_batch must be >= 1, got {train_config.micro_batch!r}"
        )
    if int(train_config.accum_tokens) < 0:
        raise ValueError(
            f"accum_tokens must be >= 0 (0 = off), got {train_config.accum_tokens!r}"
        )

    manager = resume_from
    start_step = 0
    total_steps = train_config.total_steps or train_config.steps
    cursor_data: dict[str, Any] = {}

    master = model.init_params(jax.random.PRNGKey(train_config.seed), cfg)
    # ADR-048: классификация параметров по именам листьев — состояние и шаг
    # строятся по одному и тому же предикату (иначе форма состояния разошлась бы
    # с веткой обновления).
    state = optimizer.init_state(
        master, legacy_muon_all_2d=train_config.legacy_muon_all_2d
    )
    if manager is not None:
        latest = manager.latest()
        if latest is None:
            raise PretrainDataError(
                f"resume: чекпойнтов нет в {manager.directory} — нечего продолжать"
            )
        master, state, cursor_data = manager.resume(target_params=master, target_state=state)
        # Абсолютный шаг берётся из записи чекпойнта, а не из полезной нагрузки
        # курсора: без загрузчика данных в курсоре нет поля ``step``, и нога
        # молча поехала бы с нуля по расписанию.
        start_step = int(latest["step"])
        if train_config.total_steps is None:
            total_steps = int(cursor_data.get("run", {}).get("total_steps") or total_steps)
    elif train_config.total_steps is None:
        total_steps = train_config.steps

    # К3: бюджетные счётчики — накопительные по прогону, а не по ноге.  Без этого
    # resume обнулял ``tokens_seen``/``gpu_hours``, и пороги сметы ($225/$260)
    # становились недостижимы: каждая нога выходила «в пределах лимита».  База
    # приходит из курсора предыдущей ноги (``run``), к ней добавляется текущая.
    resume_run = dict(cursor_data.get("run") or {})
    tokens_seen = int(resume_run.get("tokens_seen_total") or 0)
    gpu_hours_base = float(resume_run.get("gpu_hours_total") or 0.0)

    _validate_schedule_horizon(start_step, total_steps, train_config)

    def lr_at(step: int) -> float:
        if train_config.schedule == "wsd":
            schedule = wsd_schedule(
                train_config.lr,
                total_steps,
                warmup_ratio=train_config.warmup_ratio,
                decay_ratio=train_config.decay_ratio,
            )
        else:
            schedule = optimizer.cosine_schedule(
                train_config.lr, total_steps, train_config.warmup_ratio
            )
        return float(schedule(step))

    def loss_fn(params, batch):
        return model.compute_loss(params, cfg, batch, chunk_size=train_config.chunk_size)

    # Coarse-обёртка поверх ВСЕГО loss применяется, только когда модельный
    # уровень не несёт свою remat-политику: послойный ``per_layer`` живёт
    # внутри ``compute_loss``, а вложенный в него внешний ``jax.checkpoint``
    # XLA инлайнит вместе со всеми внутренними remat-границами — компилированный
    # граф совпадает с нематериализованным побайтово (OOM 888 ГиБ на пилоте
    # GB10 03.10.2026 при байт-идентичном hlo_rematerialization-отчёте).
    if train_config.grad_checkpointing and cfg.grad_ckpt_policy == "none":
        policy = _checkpoint_policy(train_config.grad_checkpointing_policy)
        loss_fn = jax.checkpoint(loss_fn, policy=policy)

    grad_fn = jax.jit(jax.value_and_grad(loss_fn))
    step_fn = optimizer.make_step(
        cfg,
        ns_steps=train_config.ns_steps,
        legacy_muon_all_2d=train_config.legacy_muon_all_2d,
    )

    use_bf16 = train_config.param_dtype == "bfloat16"

    def working_params(fp32_params):
        return jax.tree_util.tree_map(lambda leaf: leaf.astype(jnp.bfloat16), fp32_params) if use_bf16 else fp32_params

    params = working_params(master)
    active_params = model.active_param_count(cfg)
    metrics = MetricsWriter(train_config.metrics_path) if train_config.metrics_path else None
    # ADR-048: режим классификации — условие прогона, а не деталь кода.  Метка
    # едет в каждую строку ``pretrain-metrics/v1``, чтобы A/B-ноги различались по
    # журналу без догадок (поле опциональное, схему не ломает).
    classification = optimizer_classification(train_config.legacy_muon_all_2d)
    # Профильный режим (opt-in, ``phase_profile``): точка ветвления.  С флагом
    # каждый kpi-интервальный шаг исполняет ДОПОЛНИТЕЛЬНО декомпозированный
    # прогон фаз — числа фаз диагностические, тренировка идёт штатным fused-путём
    # (раздел «9-бис»).  Без флага ``profiler is None``: ни jit-функций фаз, ни
    # host-таймеров, ни полей в метриках — продакшн-ветка не тронута.
    # Ноги сверки forward/backward меряются РЕАЛЬНЫМИ графами шага (тот же
    # ``loss_fn`` с возможным remat и тот же ``grad_fn``): копия графа мерила бы
    # не тот шаг, а сверка сравнивала бы разные вещи.  ``forward_fn`` заводится
    # только под флагом — дефолтная ветка не платит за ещё одну jit-компиляцию.
    if train_config.phase_profile:
        forward_fn = jax.jit(loss_fn)
        profiler = _PhaseProfiler(
            cfg, train_config, grad_fn=grad_fn, forward_fn=forward_fn
        )
    else:
        profiler = None
    if train_config.ckpt_dir is not None:
        manager = CheckpointManager(train_config.ckpt_dir, keep_last=train_config.keep_last)

    iterator = iter(batches)
    losses: list[float] = []
    lr_history: list[float] = []
    step_seconds: list[float] = []
    #: Ток/с каждого шага оптимизатора — носитель KPI-медианы (delta B).
    step_tokens_per_sec: list[float] = []
    stop_reason: str | None = None
    stopped_by_budget = False
    checkpoint_record: dict | None = None
    # Граница decay-фазы берётся у даталоадера (единственный источник истины):
    # метрика ``phase`` обязана называть ту же фазу, которой питался шаг, иначе
    # журнал разошёлся бы с данными.  Без даталоадера (тесты проводки) — stable.
    data_decay_start = getattr(loader, "decay_start", None) if loader is not None else None
    first_tick = 0.0
    started = time.time()
    stop_path = Path(train_config.stop_file) if train_config.stop_file is not None else None
    stop_check_every = max(1, int(train_config.stop_check_every))
    last_checkpoint_at = started
    usd_rate = train_config.usd_per_gpu_hour
    usd_spent = 0.0
    gpu_hours = gpu_hours_base

    micro_batch = max(1, int(train_config.micro_batch))
    accum_tokens = int(train_config.accum_tokens)

    for index in range(train_config.steps):
        absolute = start_step + index + 1
        # Delta B: pull the micro-batches of one optimizer step.  ``accum_tokens
        # <= 0`` is the pre-delta behaviour — exactly one item, so the numbers
        # below are bit-for-bit the ones the loop produced before.
        group: list[Any] = []
        target: int | None = None
        exhausted = False
        # Лоадер (получение батча) меряется только под флагом: он идёт ДО ``tick``
        # и потому в ``step_seconds`` штатного шага не входит, а в сверку не
        # берётся как нога шага — но без него «остаток» молча включал бы его.
        loader_start = time.perf_counter() if profiler is not None else None
        while target is None or len(group) < target:
            try:
                micro = next(iterator)
            except StopIteration:
                exhausted = True
                break
            group.append(micro)
            if target is None:
                tokens_per_micro = int(np.prod(np.shape(micro)))
                if accum_tokens <= 0:
                    target = 1
                else:
                    target = max(
                        1, -(-accum_tokens // max(1, tokens_per_micro))
                    )  # ceil(accum_tokens / tokens_per_micro)
        if not group:
            stop_reason = (
                f"data exhausted: поток батчей кончился на шаге {absolute - 1}"
                f" из {start_step + train_config.steps}"
            )
            break
        loader_seconds = (
            time.perf_counter() - loader_start if loader_start is not None else None
        )
        batch_tokens = sum(int(np.prod(np.shape(m))) for m in group)
        tick = time.time()
        if len(group) == 1:
            # No accumulation happened: identical to the pre-delta step.
            loss, grads = grad_fn(params, group[0])
            loss_value = float(jax.device_get(loss))
        else:
            # Token-weighted mean of the micro-batches' losses/gradients, so a
            # curriculum phase change inside the group does not bias the step.
            total = float(batch_tokens)
            loss_acc = 0.0
            grads = None
            for m in group:
                mt = float(np.prod(np.shape(m)))
                micro_loss, micro_grads = grad_fn(params, m)
                loss_acc += float(jax.device_get(micro_loss)) * mt
                scaled = jax.tree_util.tree_map(lambda leaf: leaf * mt, micro_grads)
                grads = (
                    scaled
                    if grads is None
                    else jax.tree_util.tree_map(lambda a, b: a + b, grads, scaled)
                )
            grads = jax.tree_util.tree_map(lambda leaf: leaf / total, grads)
            loss_value = loss_acc / total
        if use_bf16:
            grads = jax.tree_util.tree_map(lambda leaf: leaf.astype(jnp.float32), grads)
        master, state = step_fn(master, grads, state, lr_at(absolute - 1))
        params = working_params(master)
        elapsed = time.time() - tick
        if index == 0:
            first_tick = elapsed

        # Профиль шага: только под флагом и только на kpi-интервале (там же, где
        # печатается KPI-строка).  Исполняется ПОСЛЕ ``elapsed`` — иначе время
        # декомпозиции попало бы в ток/с штатного шага и испортило KPI.  Входы —
        # те же (первый микробатч группы, параметры и градиенты штатного шага);
        # результат выбрасывается, веса остаются из fused-пути (паритет T-PP-3).
        phase_record: dict[str, Any] | None = None
        if profiler is not None and (index + 1) % max(1, int(train_config.kpi_every)) == 0:
            phase_record = profiler.measure(
                params=params,
                batch=group[0],
                grads=grads,
                master=master,
                state=state,
                lr=lr_at(absolute - 1),
                step_fn=step_fn,
                step_seconds=elapsed,
                loader_seconds=loader_seconds,
            )

        losses.append(loss_value)
        lr_history.append(lr_at(absolute - 1))
        step_seconds.append(elapsed)
        step_tokens_per_sec.append(batch_tokens / elapsed if elapsed > 0 else 0.0)
        tokens_seen += batch_tokens
        # Накопительные счётчики: база прошлых ног + текущая нога (К3).
        gpu_hours = gpu_hours_base + (time.time() - started) / 3600.0
        if usd_rate is not None:
            usd_spent = gpu_hours * float(usd_rate)

        phase = (
            PHASE_DECAY
            if data_decay_start is not None and (absolute - 1) >= data_decay_start
            else PHASE_STABLE
        )
        if metrics is not None:
            record: dict[str, Any] = {
                "step": absolute,
                "loss": loss_value,
                "lr": lr_history[-1],
                "phase": phase,
                "tokens": batch_tokens,
                "tokens_seen": tokens_seen,
                "step_seconds": elapsed,
                "tokens_per_sec": batch_tokens / elapsed if elapsed > 0 else None,
                "tflops_achieved": tflops_achieved(
                    step_flops(active_params, batch_tokens), seconds=elapsed
                ),
                "mfu": mfu(
                    step_flops(active_params, batch_tokens),
                    seconds=elapsed,
                    peak_tflops=train_config.peak_tflops,
                ),
                "mfu_params_only": True,
                "gpu_hours": gpu_hours,
                "optimizer_classification": classification,
            }
            # Поля фаз — только на профильных (kpi-интервальных) шагах: дефолтная
            # запись остаётся ровно прежней.
            if phase_record is not None:
                record["phase_profile"] = True
                record.update(phase_record)
            metrics.log(record)
        if train_config.log_every and (
            index == 0 or absolute % train_config.log_every == 0
        ):
            print(
                f"[pretrain] шаг {absolute}: loss={loss_value:.4f} "
                f"lr={lr_history[-1]:.3e} {elapsed:.2f} с/шаг",
                flush=True,
            )

        # Delta B KPI: the owner's main metric during the batch-256K pilot is
        # tokens/second, and per-step numbers are noisy — print the median over
        # a rolling window every ``kpi_every`` optimizer steps.  The per-step
        # carrier stays ``metrics.jsonl`` (``tokens_per_sec`` above).
        kpi_every = max(1, int(train_config.kpi_every))
        if (index + 1) % kpi_every == 0:
            kpi_window = max(1, int(train_config.kpi_window))
            # Первый шаг ноги компилирует jit — он вне KPI-медианы (ADR-048 Amt.);
            # построчные значения остаются в metrics.jsonl как сырой носитель.
            window = kpi_window_values(step_tokens_per_sec, window=kpi_window)
            median_tps = kpi_tokens_per_sec(step_tokens_per_sec, window=kpi_window)
            print(
                f"[pretrain] KPI: медиана ток/с за последние {len(window)} шагов "
                f"= {median_tps:.1f} (шаг {absolute}, "
                f"микробатч={micro_batch}, накопление={accum_tokens or 'off'}; "
                f"first_step_excluded={len(step_tokens_per_sec) > 1})",
                flush=True,
            )
            if phase_record is not None:
                rendered = ", ".join(
                    f"{name}={'null' if phase_record.get(name) is None else format(phase_record[name], '.3f')}"
                    for name in PHASE_PROFILE_FIELDS
                )
                print(
                    f"[pretrain] фазы, с (декомпозированный прогон, диагностика): {rendered}",
                    flush=True,
                )
                reconciled = math.fsum(
                    float(phase_record[name])
                    for name in RECONCILIATION_LEGS
                    if isinstance(phase_record.get(name), (int, float))
                    and not isinstance(phase_record.get(name), bool)
                )
                pct = phase_record.get("reconciliation_pct")
                other = phase_record.get("sec_other")
                loader = phase_record.get("sec_loader")
                print(
                    f"[pretrain] сверка фаз: сумма ног "
                    f"({' + '.join(RECONCILIATION_LEGS)}) = {reconciled:.3f} с "
                    f"из шага {elapsed:.3f} с "
                    f"({format(pct, '.1f') if pct is not None else 'null'}%); "
                    f"sec_other={'null' if other is None else format(other, '.3f')} с "
                    f"— остаток (шаг − сумма ног), НЕ «прочее»; "
                    f"sec_loader={'null' if loader is None else format(loader, '.3f')} с "
                    f"(вне шага: меряется до tick)",
                    flush=True,
                )
                if phase_record.get("finding"):
                    print(
                        f"[pretrain] НАХОДКА: {phase_record['finding']} — измеренные "
                        f"ноги покрывают {format(pct, '.1f')}% шага (< 95%); остаток "
                        f"не растворяется в тишине (ADR-011)",
                        flush=True,
                    )
                rendered_bwd = ", ".join(
                    f"{name}={'null' if phase_record.get(name) is None else format(phase_record[name], '.3f')}"
                    for name in BWD_LEG_FIELDS
                )
                print(
                    f"[pretrain] backward по компонентам, с (отдельные grad-проходы, "
                    f"диагностика): {rendered_bwd}",
                    flush=True,
                )
                bwd = phase_record.get(BWD_RECONCILIATION_FIELD) or {}
                bwd_sum = bwd.get("legs_sum")
                bwd_full = bwd.get("full_backward")
                bwd_pct = bwd.get("pct")
                print(
                    f"[pretrain] сверка backward: сумма ног = "
                    f"{'null' if bwd_sum is None else format(bwd_sum, '.3f')} с из "
                    f"sec_backward={format(bwd_full, '.3f') if bwd_full is not None else 'null'} с "
                    f"({format(bwd_pct, '.1f') if bwd_pct is not None else 'null'}%) — "
                    f"{BWD_RECONCILIATION_NOTE}",
                    flush=True,
                )
                if bwd.get("finding"):
                    print(
                        f"[pretrain] НАХОДКА: {bwd['finding']} — сумма backward-ног ушла "
                        f"от полного прохода дальше "
                        f"±{format(BWD_RECONCILIATION_TOLERANCE * 100.0, '.0f')}% "
                        f"(ориентир): видно, что ноги меряют не доли шага",
                        flush=True,
                    )
                print(
                    f"[pretrain] пиковая XLA-память ног: "
                    f"{render_phase_memory(phase_record.get(PHASE_MEMORY_FIELD))}",
                    flush=True,
                )

        if train_config.ckpt_dir is not None:
            by_steps = bool(train_config.checkpoint_every) and (
                absolute % train_config.checkpoint_every == 0
            )
            by_time = bool(train_config.checkpoint_every_min) and (
                time.time() - last_checkpoint_at >= float(train_config.checkpoint_every_min) * 60.0
            )
            if by_steps or by_time:
                checkpoint_record = _save_checkpoint(
                    manager,
                    absolute,
                    master,
                    state,
                    loader,
                    train_config,
                    total_steps,
                    tokens_seen=tokens_seen,
                    gpu_hours=gpu_hours,
                )
                last_checkpoint_at = time.time()

        # К5: стоп-файл — килл-свитч ватчдога, независимый от бюджетного порога.
        # Проверяется рядом с budget_breach (та же точка «пора остановиться») каждые
        # stop_check_every шагов: файл создаётся на хосте/инстансе и синхронизируется,
        # поэтому луп обязан его видеть, иначе «стоп-файл» остаётся обещанием в смете.
        if stop_path is not None and index % stop_check_every == 0 and stop_path.is_file():
            try:
                detail = stop_path.read_text(encoding="utf-8").strip()
            except OSError:
                detail = ""
            stop_reason = f"stop-file: {stop_path.name}" + (f" ({detail})" if detail else "")
            stopped_by_budget = True
            if manager is not None:
                checkpoint_record = _save_checkpoint(
                    manager,
                    absolute,
                    master,
                    state,
                    loader,
                    train_config,
                    total_steps,
                    tokens_seen=tokens_seen,
                    gpu_hours=gpu_hours,
                )
            break

        reason = budget_breach(
            budget, tokens_seen=tokens_seen, gpu_hours=gpu_hours, usd_spent=usd_spent
        )
        if reason is not None:
            stop_reason = reason
            stopped_by_budget = True
            # AD-8: остановка после ближайшего чекпойнта — фиксируем состояние,
            # чтобы лимит не стоил потерянных шагов.
            if manager is not None:
                checkpoint_record = _save_checkpoint(
                    manager,
                    absolute,
                    master,
                    state,
                    loader,
                    train_config,
                    total_steps,
                    tokens_seen=tokens_seen,
                    gpu_hours=gpu_hours,
                )
            break

        # Delta B: the stream ran out mid-accumulation — the step above is the
        # honest one (tokens = what the group actually consumed); stop here.
        # The final checkpoint block below persists the state for this reason.
        if exhausted:
            stop_reason = (
                f"data exhausted: поток батчей кончился на шаге {absolute}"
                f" из {start_step + train_config.steps}"
            )
            break

    if (
        manager is not None
        and stop_reason is not None
        and (checkpoint_record is None or checkpoint_record["step"] != start_step + len(losses))
    ):
        checkpoint_record = _save_checkpoint(
            manager,
            start_step + len(losses),
            master,
            state,
            loader,
            train_config,
            total_steps,
            tokens_seen=tokens_seen,
            gpu_hours=gpu_hours,
        )

    gpu_hours = gpu_hours_base + (time.time() - started) / 3600.0
    if usd_rate is not None:
        usd_spent = gpu_hours * float(usd_rate)
    if stop_reason is None:
        stop_reason = (
            None if len(losses) >= train_config.steps else "остановка без причины: батчи кончились"
        )

    return TrainResult(
        losses=losses,
        steps_done=len(losses),
        start_step=start_step,
        tokens_seen=tokens_seen,
        params=params,
        master_params=master,
        tree_hash=checkpoint_mod.tree_hash(params),
        master_tree_hash=checkpoint_mod.tree_hash(master),
        lr_history=lr_history,
        stop_reason=stop_reason,
        stopped_by_budget=stopped_by_budget,
        timings={
            "first_step_seconds": round(first_tick, 4),
            "step_seconds_mean_tail": round(
                sum(step_seconds[1:]) / max(len(step_seconds) - 1, 1), 6
            ),
            "wall_seconds": round(time.time() - started, 3),
        },
        checkpoint=checkpoint_record,
        metrics_path=str(train_config.metrics_path) if train_config.metrics_path else None,
        budget_report={
            "run_ref": budget.run_ref,
            "estimate_present": budget.present,
            "estimate_path": str(budget.path),
            "target_tokens": budget.target_tokens,
            "gpu_hours_estimate": budget.gpu_hours_estimate,
            "limit_usd": budget.limit_usd,
            # ``gpu_hours_actual`` — накопительный счётчик прогона (сумма ног), он же
            # вход стоп-правила AD-8.  Для диагностики рядом лежат база из курсора и
            # длительность текущей ноги: видно, что «факт» не сбрасывался на resume.
            "gpu_hours_actual": round(gpu_hours, 6),
            "gpu_hours_leg": round((time.time() - started) / 3600.0, 6),
            "gpu_hours_resumed_base": round(gpu_hours_base, 6),
            "tokens_seen_total": tokens_seen,
            "tokens_seen_resumed_base": int(resume_run.get("tokens_seen_total") or 0),
            "usd_spent": round(usd_spent, 6),
            "usd_rate_declared": usd_rate is not None,
            "stop_reason": stop_reason,
        },
        grad_checkpointing=bool(train_config.grad_checkpointing),
        param_dtype=train_config.param_dtype,
    )


def _validate_schedule_horizon(start_step: int, total_steps: int, train_config: TrainConfig) -> None:
    """Горизонт расписания обязан накрывать ногу — иначе LR поедет молча."""
    if total_steps <= 0:
        raise ValueError("total_steps должен быть > 0")
    if start_step >= total_steps:
        raise ValueError(
            f"resume с шага {start_step} при горизонте расписания {total_steps}: "
            "нога не накрыта расписанием (укажите --total-steps больше)"
        )


def _save_checkpoint(
    manager: CheckpointManager,
    step: int,
    master: Any,
    state: Any,
    loader: Any,
    train_config: TrainConfig,
    total_steps: int,
    *,
    tokens_seen: int = 0,
    gpu_hours: float = 0.0,
) -> dict:
    """Сохранить чекпойнт шага вместе с курсором данных и параметрами прогона.

    В блок ``run`` кладутся не только параметры расписания, но и **накопительные**
    бюджетные счётчики (К3) и пути-опоры resume: ``stop_file`` (К5).  Блок ``run``
    — контракт resume: CLI сверяет по нему seed/ratios/горизонт/режим классификации
    оптимизатора с argv (H4), а ``train`` дочитывает из него базу счётчиков, чтобы
    нога не «обнуляла» смету.
    """
    cursor: dict[str, Any] = {}
    if loader is not None:
        cursor = loader.cursor(step=step).to_json()
    elif manager.latest() is not None:
        cursor = manager.latest().get("cursor") or {}
    record = manager.save(
        step=step,
        params=master,
        optimizer_state=state,
        cursor={
            **cursor,
            "run": {
                "seed": train_config.seed,
                "total_steps": total_steps,
                "lr": train_config.lr,
                "schedule": train_config.schedule,
                "warmup_ratio": train_config.warmup_ratio,
                "decay_ratio": train_config.decay_ratio,
                "param_dtype": train_config.param_dtype,
                "data_kind": train_config.data_kind,
                # ADR-048: режим классификации — условие траектории (иной
                # оптимизатор для emb/head).  Метка, а не голое булево поле:
                # CLI (H4) сверяет её с argv, и resume ноги с обратным
                # legacy_muon_all_2d отвергается, а не продолжается молча.
                "optimizer_classification": optimizer_classification(
                    train_config.legacy_muon_all_2d
                ),
                "tokens_seen_total": int(tokens_seen),
                "gpu_hours_total": round(float(gpu_hours), 6),
                "stop_file": str(train_config.stop_file) if train_config.stop_file else None,
            },
        },
    )
    return record


__all__ = [
    "ACCEPTANCE_CONFTEST",
    "ACCEPTANCE_PINNING",
    "BACKEND_ENV",
    "BOS_ID",
    "CURSOR_MANIFEST_NAME",
    "CURSOR_SCHEMA",
    "DECAY_SHARD",
    "DECLARED_BACKENDS",
    "EOS_ID",
    "METRICS_SCHEMA",
    "Budget",
    "CheckpointManager",
    "CheckpointRecord",
    "DecayWindowPlan",
    "MetricsWriter",
    "MixCursor",
    "PackedShardEntry",
    "PackedShardReader",
    "PackedShardSet",
    "PackedTokenLoader",
    "PAD_ID",
    "PHASES",
    "PHASE_DECAY",
    "PHASE_STABLE",
    "PretrainBackendError",
    "PretrainBudgetError",
    "PretrainDataError",
    "PretrainMixLoader",
    "SHARD_NAMES",
    "ShardDocStream",
    "ShardEntry",
    "ShardSet",
    "StreamCursor",
    "TrainConfig",
    "TrainResult",
    "apply_declared_backend_pinning",
    "budget_breach",
    "declared_backend",
    "decay_start_step",
    "decay_window_plan",
    "iter_shard_docs",
    "load_budget",
    "load_packed_shard_set",
    "load_shard_set",
    "mfu",
    "packed_manifest_path",
    "pack_batch",
    "require_budget",
    "step_flops",
    "tflops_achieved",
    "train",
    "window_permutation",
    "wsd_schedule",
]
