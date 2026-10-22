"""ADR-054 Am.16 (цикл-7, веха-7) — jitted-аккумуляция микробатчей (CPU).

Am.15 измеряет шаг l3-full B=2@8192 хост-bound: ~0.5 с GPU-работы против ~27 с
хост-машинерии, где каждый микробатч прежней accum-ветки — свой ``grad_fn``-вызов
+ ``device_get``-синхронизация + tree_map-скалирование в Python.  Дельта: один
jit-вызов на весь оптимизаторский шаг — ``lax.scan`` по ведущей оси пакета из N
микробатчей, градиенты и loss усредняются на устройстве с токен-весами (та же
формула, что в Python-ветке).  Гейт ``AXIOM_ACCUM_JIT`` (дефолт ``0``).

Что пинуется здесь (CPU, без GPU):

* **(а) гейт-off побитово прежний** — прогон ``train`` с накоплением на текущей
  ревизии (гейт не задан) против прогона ``net/`` базовой ревизии
  ``BASELINE_COMMIT`` (архив ``git archive``), сравнение loss-ов и tree_hash
  **точно**; плюс структурный пин: гейт-off не строит ни одной jitted-функции
  аккумуляции (spy на фабрику), метрики-строки несут ``accum_jit=0``;
* **(б) паритет** — jitted accum=4 против 4 одиночных ``grad_fn``-вызовов с тем
  же токен-взвешиванием: loss и grads allclose, допуск задокументирован
  фактическими значениями (fp32 измерено ``max|Δgrad| ≈ 1.0e-07``, rel
  ``3.5e-07``; bf16 ``max|Δgrad| ≈ 3.8e-06``, rel ``1.15e-03`` — цена иного
  порядка редукций scan-тела против отдельного ``grad_fn``, не другая формула);
  отдельно — невыровненные веса [1,2,3,4] (строка формулировки взвешивания);
* **(в) N=1** — jitted-вызов с пакетом из одного микробатча совпадает с
  одиночным ``grad_fn`` побитово (измерено 0.0 на CPU jax 0.11.2);
* **(г) donate-легальность** — донированный вариант зовётся ровно там, где
  буфер мёртв (bf16-рабочая копия), и не зовётся там, где ``params IS master``
  (fp32, bf16-мастер Am.13) или жив (``phase_profile``: профайлер читает
  ``params`` после градиента); после донированного вызова буфер помечен
  удалённым; bf16-нога с накоплением доезжает без «donated buffer reused»;
* **(д) переменная длина группы** — хвост потока (3 микробатча при цели 4)
  взвешивается как в Python-ветке, метрика несёт ``microbatches=3``; микробатчи
  РАЗНЫХ форм — честный отказ (:class:`tl.AccumJitError`), не тихая упаковка;
* **память** — пик временных буферов исполняемого графа не растёт с N
  (измерено: ``temp(N=4) = temp(N=2)``, ``temp(N=4)/temp(grad_fn) = 1.095``),
  в трассе ровно один ``scan`` и структура верхнего уровня не зависит от N —
  unroll-«пол» активаций (N копий fwd+bwd в одном jaxpr) не построен.

Не пинуется здесь: NVTX-графы/хост-эпохи и ток/с на GB10 (замер архитектора,
цикл-7 A/B: ``--accum-tokens 65536`` и ``131072``); полный регресс ``net/tests``
на CPU (существующие дефекты окружения без ``~/gb10-shared`` — вне дельты).
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import subprocess
import sys
import tarfile
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from conftest import tiny_config

from net import model
from net import train_loop as tl

#: Геометрия смоука: одна последовательность, короткий контекст (CPU-время).
SEQ_LEN = 16
CHUNK = 64
#: Цели накопления: 2 и 4 микробатча (tokens_per_micro = 1 * SEQ_LEN).
ACCUM2 = 2 * SEQ_LEN
ACCUM4 = 4 * SEQ_LEN

#: Базовая ревизия (ADR-054 Am.16 — до дельты jitted-аккумуляции): гейт-off
#: обязан воспроизводить её побитово.  ``git archive`` + skip, если ревизия
#: недоступна (форма test_29_compute_dtype).
BASELINE_COMMIT = "6d3eda2"
NET_DIR = Path(__file__).resolve().parents[1]
CASE_DIR = NET_DIR.parent

#: Фактически измеренный допуск паритета (jitted accum=4 ↔ 4 одиночных
#: ``grad_fn``, tiny-геометрия, CPU jax 0.11.2, seed 5, loss ≈ 4.61):
#:   fp32: max|Δloss| = 2.4e-07, max|Δgrad| = 1.0e-07, max rel|Δgrad| = 3.5e-07;
#:   bf16: max|Δloss| = 1.2e-07, max|Δgrad| = 3.8e-06, max rel = 1.15e-03
#:         (bf16-накопление — как в Python-ветке; остаток — порядок редукций).
#: Пороги ниже — с запасом ≥ 8x от измеренного.
FP32_RTOL, FP32_ATOL = 1e-5, 1e-6
BF16_RTOL, BF16_ATOL = 1e-2, 1e-5
#: Допуск train-ног с накоплением (2 шага, tiny): измерено fp32
#: max|Δparams| = 1.2e-07, bf16 max|Δmaster| = 9.7e-06; пороги с запасом ≥ 8x.
LEG_FP32_ATOL = 1e-6
LEG_BF16_ATOL = 1e-4
#: Допуск loss-строк train-ног на одинаковых данных: измерено ≤ 2.4e-07.
LEG_LOSS_ATOL = 1e-5


# ---------------------------------------------------------------------------
# Общие помощники
# ---------------------------------------------------------------------------


def _batches(cfg, steps: int, *, seed: int = 5, width: int = 1):
    """Детерминированный поток ``(width, T) int32`` — одинаковые данные для ног."""
    rng = np.random.default_rng(seed)
    return [
        rng.integers(0, cfg.vocab_size, size=(width, SEQ_LEN), dtype=np.int32)
        for _ in range(steps)
    ]


def _budget():
    return tl.Budget(run_ref="accum-jit", path=Path("/dev/null"), present=True)


def _train_leg(
    cfg,
    *,
    steps: int = 1,
    accum: int = ACCUM2,
    param_dtype: str = "float32",
    batches=None,
    metrics_path=None,
    kpi_every: int = 20,
    phase_profile: bool = False,
    seed: int = 7,
    lr: float = 1e-2,
):
    """Нога ``train`` без управления env (гейт/мастер выставляют контексты)."""
    return tl.train(
        cfg,
        iter(batches if batches is not None else _batches(cfg, steps * 2)),
        train_config=tl.TrainConfig(
            steps=steps,
            lr=lr,
            seed=seed,
            schedule="cosine",
            param_dtype=param_dtype,
            accum_tokens=accum,
            micro_batch=1,
            metrics_path=metrics_path,
            log_every=0,
            kpi_every=kpi_every,
            phase_profile=phase_profile,
        ),
        budget=_budget(),
    )


@contextlib.contextmanager
def gate_env(value: str | None):
    """Временное значение ``AXIOM_ACCUM_JIT`` (``None`` — переменную снять)."""
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
def master_mode(dtype: str | None = None, comp: str | None = None):
    """Временный режим мастера через env (``None`` — переменную снять)."""
    keys = ("AXIOM_MASTER_DTYPE", "AXIOM_MASTER_COMP")
    saved = {key: os.environ.get(key) for key in keys}
    try:
        for key, value in (("AXIOM_MASTER_DTYPE", dtype), ("AXIOM_MASTER_COMP", comp)):
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        yield
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _leaves(tree) -> list:
    return jax.tree_util.tree_leaves(tree)


def _max_abs(a, b) -> float:
    la, lb = _leaves(a), _leaves(b)
    assert len(la) == len(lb), f"разные деревья: {len(la)} против {len(lb)} листьев"
    return max(
        float(np.max(np.abs(np.asarray(x, np.float64) - np.asarray(y, np.float64))))
        for x, y in zip(la, lb)
    )


def _allclose(a, b, *, rtol: float, atol: float) -> list[str]:
    """Все листья пары близки; возврат — список нарушителей (пустой = ok)."""
    offenders = []
    for index, (x, y) in enumerate(zip(_leaves(a), _leaves(b))):
        if not bool(jnp.allclose(np.asarray(x), np.asarray(y), rtol=rtol, atol=atol)):
            gap = float(
                np.max(np.abs(np.asarray(x, np.float64) - np.asarray(y, np.float64)))
            )
            offenders.append(f"лист {index}: max|Δ| = {gap:.3e}")
    return offenders


def _loss_fn(cfg):
    def loss_fn(params, batch):
        return model.compute_loss(params, cfg, batch, chunk_size=CHUNK)

    return loss_fn


def _python_accum(grad_fn, params, group, weights=None):
    """Оракул: арифметика Python-ветки цикла один-в-один (train_loop :3124-3141).

    ``grad_fn`` зовётся на каждом микробатче, loss и градиенты масштабируются
    токенами микробатча, делятся на суммарные токены группы.  Возврат —
    ``(loss, grads)``; градиенты — в dtype ``grad_fn``.
    """
    if weights is None:
        weights = [float(np.prod(np.shape(m))) for m in group]
    total = float(np.sum([float(w) for w in weights]))
    loss_acc = 0.0
    grads = None
    for micro, weight in zip(group, weights):
        micro_loss, micro_grads = grad_fn(params, micro)
        loss_acc += float(jax.device_get(micro_loss)) * float(weight)
        scaled = jax.tree_util.tree_map(
            lambda leaf, w=float(weight): leaf * w, micro_grads
        )
        grads = (
            scaled
            if grads is None
            else jax.tree_util.tree_map(lambda a, b: a + b, grads, scaled)
        )
    grads = jax.tree_util.tree_map(lambda leaf: leaf / total, grads)
    return loss_acc / total, grads


def _spy_factory(monkeypatch):
    """Обернуть ``tl.make_jitted_accum`` наблюдателем аргумента ``donate``."""
    seen: list[dict] = []
    real = tl.make_jitted_accum

    def spy(loss_fn, *, donate=False):
        seen.append({"donate": donate})
        return real(loss_fn, donate=donate)

    monkeypatch.setattr(tl, "make_jitted_accum", spy)
    return seen


# ---------------------------------------------------------------------------
# Гейт: семантика значения (дефолт, значения, fail-closed)
# ---------------------------------------------------------------------------


def test_gate_default_is_off_and_values_normalize():
    with gate_env(None):
        assert tl.accum_jit_mode() is False, "дефолт — прежний Python-путь"
    with gate_env("1"):
        assert tl.accum_jit_mode() is True
    with gate_env(" 0 "):
        assert tl.accum_jit_mode() is False, "пробелы нормализуются"
    with gate_env(""):
        assert tl.accum_jit_mode() is False, "пустая переменная = не задана"
    assert tl.accum_jit_mode("1") is True, "явный аргумент сильнее окружения"
    assert tl.accum_jit_mode("0") is False


def test_gate_unknown_value_fails_closed():
    """Опечатка — ошибка, не тихий дефолт: тихая нога мерила бы не тот путь."""
    for bad in ("true", "yes", "on", "2", "01", "jitted"):
        with gate_env(bad):
            with pytest.raises(tl.AccumJitError, match="AXIOM_ACCUM_JIT"):
                tl.accum_jit_mode()


def test_train_refuses_unknown_gate_value():
    """Та же fail-closed дверь на сборке цикла (до первой компиляции)."""
    cfg = tiny_config()
    with gate_env("jitted"):
        with pytest.raises(tl.AccumJitError, match="AXIOM_ACCUM_JIT"):
            _train_leg(cfg)


# ---------------------------------------------------------------------------
# (а) гейт-off: побитово прежний путь — базовая ревизия + структурный пин
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

cfg = tiny_config()
rng = np.random.default_rng(5)
batches = [rng.integers(0, cfg.vocab_size, size=(1, 16), dtype=np.int32) for _ in range(4)]
result = tl.train(
    cfg, iter(batches),
    train_config=tl.TrainConfig(steps=2, lr=1e-2, seed=7, schedule="cosine",
                                param_dtype="float32", accum_tokens=32,
                                micro_batch=1, log_every=0),
    budget=tl.Budget(run_ref="gate-off", path=__import__("pathlib").Path("/dev/null"),
                     present=True),
)
print(json.dumps({
    "losses": [repr(x) for x in result.losses],
    "params_hash": checkpoint.tree_hash(result.params),
    "master_hash": checkpoint.tree_hash(result.master_params),
    "steps_done": result.steps_done,
}))
'''


def _run_probe(cwd: Path, *, script: Path | None = None) -> dict:
    """Проба в подпроцессе: env без гейта/мастера — дефолтное поведение.

    ``script`` — файл пробы (архив базовой ревизии); без него тот же исходник
    исполняется через ``-c`` (``sys.path[0] = cwd``), чтобы не писать файлов
    в рабочее дерево текущей ревизии.
    """
    env = dict(os.environ)
    env.pop(tl.ACCUM_JIT_ENV, None)
    env.pop("AXIOM_MASTER_DTYPE", None)
    env.pop("AXIOM_MASTER_COMP", None)
    command = (
        [sys.executable, str(script)]
        if script is not None
        else [sys.executable, "-c", _PROBE]
    )
    done = subprocess.run(
        command,
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
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

    Сравниваются loss-репризы и tree_hash деревьев (точно): любая новая
    арифметика на дефолтном пути видна строкой, а не «в допуске».  Проба
    гоняет 2 шага по 2 микробатча — именно accum-ветку, которую дельта не
    трогает.
    """
    baseline = _extract_baseline(tmp_path)
    base_out = _run_probe(baseline, script=baseline / "probe.py")
    current_out = _run_probe(CASE_DIR)
    assert current_out == base_out, (
        "гейт-off разошёлся с базовой ревизией: "
        f"current={current_out!r} baseline={base_out!r}"
    )


def test_a_gate_off_builds_no_jitted_accum_and_marks_metrics(monkeypatch, tmp_path):
    """Структурный пин: гейт-off не строит jitted-функцию; под гейтом — строит.

    Маркеры метрик — по факту пути: ``accum_jit=0`` при гейте-off и при шаге
    без накопления (1 микробатч), ``=1`` на jitted-шаге; ``microbatches`` —
    число микробатчей группы.
    """
    cfg = tiny_config()
    seen = _spy_factory(monkeypatch)
    metrics = tmp_path / "metrics.jsonl"

    with gate_env(None):
        off = _train_leg(cfg, steps=2, accum=ACCUM2, metrics_path=metrics)
    assert seen == [], "гейт-off построил jitted-аккумуляцию — дефолт не прежний"
    off_rows = tl.MetricsWriter.read(metrics)
    assert [(row["accum_jit"], row["microbatches"]) for row in off_rows] == [
        (0, 2),
        (0, 2),
    ]

    with gate_env("1"):
        on = _train_leg(cfg, steps=2, accum=ACCUM2, metrics_path=metrics)
    assert [call["donate"] for call in seen] == [False], (
        "fp32: строится ровно недонированный вариант (params IS master)"
    )
    on_rows = tl.MetricsWriter.read(metrics)[2:]
    assert [(row["accum_jit"], row["microbatches"]) for row in on_rows] == [
        (1, 2),
        (1, 2),
    ]

    # Шаг без накопления: накопления нет — маркер честно 0, микробатч один.
    with gate_env(None):
        _train_leg(cfg, steps=1, accum=0, metrics_path=metrics)
    single_rows = tl.MetricsWriter.read(metrics)[4:]
    assert [(row["accum_jit"], row["microbatches"]) for row in single_rows] == [(0, 1)]

    # Паритет ног: loss-ы совпадают; деревья — в задокументированном допуске.
    for a, b in zip(off.losses, on.losses):
        assert abs(a - b) < LEG_LOSS_ATOL, (a, b)
    assert _max_abs(off.params, on.params) < LEG_FP32_ATOL
    assert _max_abs(off.master_params, on.master_params) < LEG_FP32_ATOL


# ---------------------------------------------------------------------------
# (б) паритет jitted accum=4 против 4 одиночных grad_fn с тем же взвешиванием
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_b_parity_accum4_vs_single_calls(dtype):
    """loss и grads jitted accum=4 против Python-ветки (4 одиночных вызова).

    Допуски — измеренные фактические (шапка модуля): fp32 ~1e-07, bf16 ~4e-06
    (rel 1.2e-03).  bf16-нога доказывает, что jitted-путь работает с
    bf16-параметрами (Am.13/цикл-4) и накапливает в bf16 — как Python-ветка.
    """
    cfg = tiny_config()
    master = model.init_params(jr.PRNGKey(7), cfg)
    params = (
        jax.tree_util.tree_map(lambda x: x.astype(jnp.bfloat16), master)
        if dtype == "bfloat16"
        else master
    )
    loss_fn = _loss_fn(cfg)
    grad_fn = jax.jit(jax.value_and_grad(loss_fn))
    group = _batches(cfg, 4)

    ref_loss, ref_grads = _python_accum(grad_fn, params, group)
    packed, weights = tl.pack_microbatches(group)
    assert weights.tolist() == [float(SEQ_LEN)] * 4, "веса — токены микробатчей"
    accum = tl.make_jitted_accum(loss_fn, donate=False)
    loss, grads = accum(params, packed, weights)

    assert jax.tree_util.tree_leaves(grads)[0].dtype == jnp.dtype(dtype), (
        "dtype градиентов обязан совпасть с режимом (bf16-накопление — как в ветке)"
    )
    rtol, atol = (FP32_RTOL, FP32_ATOL) if dtype == "float32" else (BF16_RTOL, BF16_ATOL)
    assert abs(float(jax.device_get(loss)) - ref_loss) < atol
    offenders = _allclose(grads, ref_grads, rtol=rtol, atol=atol)
    assert not offenders, f"grads разошлись ({dtype}): {offenders}"

    # Невыровненные веса — та же формула Σ G_i·w_i / Σ w_i, не «среднее по N».
    weights = jnp.array([1.0, 2.0, 3.0, 4.0], jnp.float32)
    ref_w_loss, ref_w_grads = _python_accum(
        grad_fn, params, group, weights=[1.0, 2.0, 3.0, 4.0]
    )
    loss_w, grads_w = accum(params, packed, weights)
    assert abs(float(jax.device_get(loss_w)) - ref_w_loss) < atol
    offenders = _allclose(grads_w, ref_w_grads, rtol=rtol, atol=atol)
    assert not offenders, f"взвешенные grads разошлись ({dtype}): {offenders}"


# ---------------------------------------------------------------------------
# (в) N=1: jitted-пакет из одного микробатча == одиночный grad_fn
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_c_single_microbatch_is_bit_identical_to_grad_fn(dtype):
    """N=1 — тот же путь, что одиночный граф: измерено ровно 0.0 расхождения.

    Тест пинит факт побитового совпадения (scan длины 1 даёт тот же граф на
    CPU jax 0.11.2); сообщение назовёт расхождение числом, если смена стека
    сломает свойство.
    """
    cfg = tiny_config()
    master = model.init_params(jr.PRNGKey(11), cfg)
    params = (
        jax.tree_util.tree_map(lambda x: x.astype(jnp.bfloat16), master)
        if dtype == "bfloat16"
        else master
    )
    loss_fn = _loss_fn(cfg)
    grad_fn = jax.jit(jax.value_and_grad(loss_fn))
    micro = _batches(cfg, 1, seed=13)[0]

    ref_loss, ref_grads = grad_fn(params, micro)
    packed, weights = tl.pack_microbatches([micro])
    accum = tl.make_jitted_accum(loss_fn, donate=False)
    loss, grads = accum(params, packed, weights)

    assert float(jax.device_get(loss)) == float(jax.device_get(ref_loss)), (
        f"N=1 loss не побитово равен grad_fn ({dtype})"
    )
    assert _max_abs(grads, ref_grads) == 0.0, (
        f"N=1 grads не побитово равны grad_fn ({dtype})"
    )
    assert jax.tree_util.tree_leaves(grads)[0].dtype == jnp.dtype(dtype)


# ---------------------------------------------------------------------------
# (г) donate: правила мёртвого буфера и легальность в цикле
# ---------------------------------------------------------------------------


def test_d_donated_variant_is_real_and_params_die_after_call():
    """``donate=True`` — контракт владения: буфер помечен удалённым.

    Факт jax (0.11.2, CPU): чтение донированного дерева после вызова падает
    ``RuntimeError``.  Именно на это опирается цикл: в bf16-рабочей копии
    ``params`` после градиента мёртв; в fp32/bf16-мастере/profiler — жив
    (там донация не строится, тест ниже).
    """
    cfg = tiny_config()
    params = model.init_params(jr.PRNGKey(7), cfg)
    loss_fn = _loss_fn(cfg)
    group = _batches(cfg, 2)
    packed, weights = tl.pack_microbatches(group)

    donated = tl.make_jitted_accum(loss_fn, donate=True)
    loss, grads = donated(params, packed, weights)
    jax.block_until_ready(grads)
    assert np.isfinite(float(jax.device_get(loss)))
    with pytest.raises(RuntimeError, match="deleted"):
        np.asarray(jax.tree_util.tree_leaves(params)[0]) * 1.0

    # Арифметика донированного варианта — та же (свежее дерево, те же данные).
    params2 = model.init_params(jr.PRNGKey(7), cfg)
    plain = tl.make_jitted_accum(loss_fn, donate=True)
    loss2, _ = plain(params2, packed, weights)
    assert float(jax.device_get(loss2)) == float(jax.device_get(loss)), (
        "донация изменила арифметику"
    )


def test_d_donate_variant_follows_dead_buffer_rule(monkeypatch):
    """Donate-вариант строится ровно там, где входной буфер мёртв.

    * fp32 — ``params IS master``, его читает ``step_fn``: только donate=False;
    * bf16-рабочая копия — донированный, и 3-шаговая нога доезжает (буфер не
      переиспользуется: ошибка «donated buffer reused» уронила бы шаг 2);
    * bf16-мастер (Am.13) — ``params IS master``: только donate=False;
    * ``phase_profile`` — профайлер читает ``params`` после градиента: donate
      не строится (иначе ноги профиля получили бы удалённый буфер).
    """
    cfg = tiny_config()

    seen = _spy_factory(monkeypatch)
    with gate_env("1"):
        leg = _train_leg(cfg, steps=3, accum=ACCUM2, param_dtype="bfloat16")
    assert [call["donate"] for call in seen] == [False, True], (
        "bf16-рабочая копия: ожидались оба варианта, донированный — для шага"
    )
    assert leg.steps_done == 3 and all(np.isfinite(x) for x in leg.losses)

    monkeypatch.undo()
    seen = _spy_factory(monkeypatch)
    with gate_env("1"), master_mode("bf16", "kahan"):
        leg = _train_leg(cfg, steps=1, accum=ACCUM2, param_dtype="bfloat16")
    assert [call["donate"] for call in seen] == [False], (
        "bf16-мастер: params IS master, donate запрещён"
    )
    assert leg.steps_done == 1

    monkeypatch.undo()
    seen = _spy_factory(monkeypatch)
    with gate_env("1"), master_mode(None, None):
        _train_leg(cfg, steps=1, accum=ACCUM2, param_dtype="float32")
    assert [call["donate"] for call in seen] == [False], (
        "fp32: params IS master, donate запрещён"
    )

    # phase_profile: профайлер читает params — донированный вариант не строится.
    # Декомпозированные ноги профиля здесь не нужны: подменяем профайлер
    # зондом, который ДЕЙСТВИТЕЛЬНО читает params в measure() — контракт
    # «буфер жив» проверяется поведением, а не намерением.
    monkeypatch.undo()
    seen = _spy_factory(monkeypatch)
    reads: list[float] = []

    class _ProfilerProbe:
        def __init__(self, *args, **kwargs):
            pass

        def measure(self, *, params, **kwargs):
            leaf = jax.tree_util.tree_leaves(params)[0]
            reads.append(float(np.sum(np.asarray(leaf))))
            return {}

    monkeypatch.setattr(tl, "_PhaseProfiler", _ProfilerProbe)
    with gate_env("1"), master_mode(None, None):
        leg = _train_leg(
            cfg,
            steps=1,
            accum=ACCUM2,
            param_dtype="bfloat16",
            phase_profile=True,
            kpi_every=1,
        )
    assert [call["donate"] for call in seen] == [False], (
        "phase_profile: params жив для профайлера — donate не строится"
    )
    assert reads, "профайлер-зонд не увидел measure — шаг не дошёл до профиля"
    assert leg.steps_done == 1


# ---------------------------------------------------------------------------
# (д) переменная длина группы: хвост потока и честный отказ на разных формах
# ---------------------------------------------------------------------------


def test_e_tail_group_is_weighted_like_python(tmp_path):
    """Хвост потока: группа из 3 микробатчей при цели 4 — взвешивается честно.

    Поток даёт 3 микробатча: цель 4, поток кончился — группа N=3 (реальный
    факт шага, не номинальная цель).  Гейт-on и гейт-off считают тот же шаг;
    метрика называет число микробатчей группы.
    """
    cfg = tiny_config()
    batches3 = _batches(cfg, 3, seed=17)
    metrics = tmp_path / "metrics.jsonl"

    with gate_env(None):
        off = _train_leg(
            cfg, steps=1, accum=ACCUM4, batches=batches3, metrics_path=metrics
        )
    with gate_env("1"):
        on = _train_leg(
            cfg, steps=1, accum=ACCUM4, batches=batches3, metrics_path=metrics
        )

    assert off.steps_done == on.steps_done == 1
    assert abs(off.losses[0] - on.losses[0]) < LEG_LOSS_ATOL
    assert _max_abs(off.params, on.params) < LEG_FP32_ATOL
    rows = tl.MetricsWriter.read(metrics)
    assert [(row["accum_jit"], row["microbatches"]) for row in rows] == [
        (0, 3),
        (1, 3),
    ], "хвостовая группа обязана попасть в метрику числом фактических микробатчей"


def test_e_ragged_group_is_honest_refusal():
    """Микробатчи разных форм — ошибка упаковки, не тихая склейка/подмена весов.

    Штатный загрузчик кампании упаковывает фиксированный ``(micro_batch,
    seq_len)``, поэтому отказ достижим только на кастомном потоке (этот тест):
    разных форм в одном стеке быть не может, а рецепта взвешивания для них не
    существует — fail-closed на упаковке (форма остальных гейтов дельты).
    """
    ragged = [
        np.zeros((1, SEQ_LEN), np.int32),
        np.zeros((2, SEQ_LEN), np.int32),
    ]
    with pytest.raises(tl.AccumJitError, match="одинаковых форм"):
        tl.pack_microbatches(ragged)

    cfg = tiny_config()
    with gate_env("1"):
        with pytest.raises(tl.AccumJitError, match="одинаковых форм"):
            _train_leg(cfg, steps=1, accum=ACCUM4, batches=ragged)


# ---------------------------------------------------------------------------
# Память: scan — последовательный бюджет, а не HLO-«пол» из N копий fwd+bwd
# ---------------------------------------------------------------------------


def _trace_primitive_counts(fn, params, packed, weights) -> dict[str, int]:
    """Счётчики примитивов верхнего уровня трассы (для pin «один scan»)."""
    body = getattr(fn, "__wrapped__", fn)
    closed = jax.make_jaxpr(body)(params, packed, weights)
    counts: dict[str, int] = {}
    for eqn in closed.jaxpr.eqns:
        counts[eqn.primitive.name] = counts.get(eqn.primitive.name, 0) + 1
    return counts


def _executable_temp(fn, params, packed, weights) -> int:
    executable = fn.lower(params, packed, weights).compile()
    if not hasattr(executable, "memory_analysis"):  # pragma: no cover
        pytest.skip("memory_analysis() недоступен на этом стеке")
    analysis = executable.memory_analysis()
    return int(analysis.temp_size_in_bytes)


def test_memory_scan_body_budget_does_not_scale_with_n():
    """Пик временных буферов jitted-аккумуляции — бюджет тела, не N микробатчей.

    Измерено на CPU (jax 0.11.2, tiny): ``temp(N=4) = temp(N=2) = 2 173 544 Б``,
    ``temp(одиночный grad_fn) = 1 984 464 Б`` (отношение 1.095).  Если бы
    развёртка (или переассоциация XLA) материализовала активации всех N
    микробатчей, пик рос бы с N — тест назвал бы это числом до того, как
    цикл-7 пойдёт на GB10 («признак роста живых тензоров — остановиться и
    эскалировать», а не пушить).  Структурная половина: в трассе ровно один
    ``scan``, а число примитивов верхнего уровня не зависит от N.
    """
    cfg = tiny_config()
    params = model.init_params(jr.PRNGKey(7), cfg)
    loss_fn = _loss_fn(cfg)
    grad_fn = jax.jit(jax.value_and_grad(loss_fn))
    group = _batches(cfg, 4)

    single_exe = grad_fn.lower(params, group[0]).compile()
    if not hasattr(single_exe, "memory_analysis"):  # pragma: no cover
        pytest.skip("memory_analysis() недоступен на этом стеке")
    temp_single = int(single_exe.memory_analysis().temp_size_in_bytes)

    temps: dict[int, int] = {}
    counts: dict[int, dict[str, int]] = {}
    for n in (2, 4):
        packed, weights = tl.pack_microbatches(group[:n])
        fn = tl.make_jitted_accum(loss_fn, donate=False)
        temps[n] = _executable_temp(fn, params, packed, weights)
        counts[n] = _trace_primitive_counts(fn, params, packed, weights)

    assert temps[4] <= 1.25 * temp_single, (
        f"temp(accum N=4) = {temps[4]} > 1.25 * temp(grad_fn) = {1.25 * temp_single}: "
        "исполняемый граф материализует больше, чем бюджет одного микробатча?"
    )
    assert temps[4] <= 1.05 * temps[2], (
        f"temp(N=4) = {temps[4]} вырос против temp(N=2) = {temps[2]} — "
        "память начала масштабироваться с N (scan/unroll породил HLO-пол)"
    )
    for n in (2, 4):
        assert counts[n].get("scan") == 1, (
            f"N={n}: ожидался ровно один lax.scan верхнего уровня, "
            f"трасса: {counts[n]!r}"
        )
    assert sum(counts[2].values()) == sum(counts[4].values()), (
        "число примитивов верхнего уровня зависит от N — цикл развёрнут?"
    )
