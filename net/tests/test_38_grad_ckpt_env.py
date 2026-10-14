"""``AXIOM_GRAD_CKPT_POLICY`` — env-переопределение политики grad-checkpointing
для замеров (ADR-054 Am.18, микро-дельта к вехе-8, кампания MFU-55).

Зачем ручка.  Политику ``grad_ckpt_policy`` несёт пиннутый ``net/config.json``
(pretrain-пресет — ``per_layer``), CLI-ручки нет.  Веха-8 собрала
``per_layer_block`` (граница по KDA/MLA+MoE блоку — адрес рычага B=4, см.
``net/tests/test_37_per_layer_block.py``), но из раннера политика недостижима
без правки пина.  Микро-дельта: переменная окружения перекрывает
``cfg.grad_ckpt_policy`` **на сборке** — пин (AD-9) не трогается, A/B-ноги
«per_layer vs per_layer_block» идут по одному и тому же файлу.

Что пиннуется здесь (CPU, без GB10):

* **(а) без env — конфиг первичен**: чтение возвращает ``None`` (пустая и
  пробельная переменная — тоже «не задана»); ``load_config`` отдаёт объявленное
  значение; нога ``train`` пишет в журнал метрик **фактическое** значение
  конфига, а дефолтный путь побитово совпадает с ревизией до дельты
  (``BASELINE_COMMIT``) — ни замены cfg, ни лишней валидации;
* **(б) env=per_layer_block при конфиге per_layer**: ``load_config`` несёт
  переопределённое значение; ``train`` доводит его до сборщика графа (spy на
  ``net.model.compute_loss`` видит cfg с ``per_layer_block``) и до журнала
  метрик (поле ``grad_ckpt_policy`` — маркер A/B-ног, форма master_dtype/
  accum_jit);
* **(в) env=мусор — ошибка сборки, не запуск**: fail-closed чтение
  (:class:`net.config.GradCkptPolicyError`, образец ``AccumJitError``/
  ``master_dtype_mode``) — и на загрузке cfg, и на входе ``train`` до
  инициализации параметров, до компиляции и до первого батча (поток данных не
  читается, файл метрик не создаётся);
* **схема не обходится**: переопределённое значение проходит тот же
  ``validate_config`` (spy на схеме), а ``GRAD_CKPT_POLICIES`` схемы пиннут
  равным ``net.model.GRAD_CKPT_POLICIES`` — один контракт на три потребителя.

Не пинуется здесь: замеры B=4 и «политика → время/память» на GB10 (окно
кампании) — дельта лишь делает политику достижимой из раннера.
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

import numpy as np
import pytest

from conftest import tiny_config

from net import config as net_config
from net import model
from net import train_loop as tl

#: Геометрия смоука: одна последовательность, короткий контекст (CPU-время).
SEQ_LEN = 16

#: Ревизия до дельты (веха-8 в линии, env-ручки ещё нет): путь без переменной
#: обязан воспроизводить её побитово.  ``git archive`` + skip, если ревизия
#: недоступна (форма test_36/test_29).
BASELINE_COMMIT = "a1b425e"
NET_DIR = Path(__file__).resolve().parents[1]
CASE_DIR = NET_DIR.parent


# ---------------------------------------------------------------------------
# Общие помощники
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _cpu_kda_body(monkeypatch):
    """Явное committed KDA-тело: кампанийный дефолт ``persistent`` — GB10-кернел.

    Тот же suite-wide приём, что в test_17/23/31 (фикс a1b425e): окружение
    кампании не должно диктовать CPU-ногам, какое тело исполнять.
    """
    monkeypatch.setenv("AXIOM_KDA_CC_IMPL", "jax")


def _batches(cfg, steps: int, *, seed: int = 5, width: int = 1):
    """Детерминированный поток ``(width, T) int32`` — одинаковые данные для ног."""
    rng = np.random.default_rng(seed)
    return [
        rng.integers(0, cfg.vocab_size, size=(width, SEQ_LEN), dtype=np.int32)
        for _ in range(steps)
    ]


def _budget():
    return tl.Budget(run_ref="grad-ckpt-env", path=Path("/dev/null"), present=True)


def _train_leg(
    cfg,
    *,
    steps: int = 1,
    batches=None,
    metrics_path=None,
    seed: int = 7,
    lr: float = 1e-2,
):
    """Нога ``train`` (env политики выставляют тесты явно через monkeypatch)."""
    return tl.train(
        cfg,
        iter(batches if batches is not None else _batches(cfg, steps)),
        train_config=tl.TrainConfig(
            steps=steps,
            lr=lr,
            seed=seed,
            schedule="cosine",
            param_dtype="float32",
            micro_batch=1,
            metrics_path=metrics_path,
            log_every=0,
            kpi_every=20,
        ),
        budget=_budget(),
    )


def _records(path):
    rows = tl.MetricsWriter.read(path)
    assert rows, f"журнал метрик пуст: {path}"
    return rows


@contextlib.contextmanager
def policy_env(value: str | None):
    """Временное значение ``AXIOM_GRAD_CKPT_POLICY`` (``None`` — снять)."""
    key = net_config.GRAD_CKPT_POLICY_ENV
    saved = os.environ.get(key)
    try:
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value
        yield
    finally:
        if saved is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = saved


# ---------------------------------------------------------------------------
# Контракт значения: чтение, нормализация, fail-closed
# ---------------------------------------------------------------------------


def test_reader_defaults_to_none_and_normalizes(monkeypatch):
    """Не задана/пуста — ``None`` (конфиг первичен); значения нормализуются."""
    with policy_env(None):
        assert net_config.grad_ckpt_policy_mode() is None, "переменной нет"
    for blank in ("", "   ", "\t\n"):
        with policy_env(blank):
            assert net_config.grad_ckpt_policy_mode() is None, "пустая = не задана"
    for policy in net_config.GRAD_CKPT_POLICIES:
        with policy_env(policy):
            assert net_config.grad_ckpt_policy_mode() == policy
        with policy_env(f"  {policy.upper()} "):
            assert net_config.grad_ckpt_policy_mode() == policy, "регистр/пробелы"
    # Явный аргумент сильнее окружения (форма accum_jit_mode/master_dtype_mode).
    with policy_env("none"):
        assert net_config.grad_ckpt_policy_mode("per_layer_block") == "per_layer_block"


def test_reader_unknown_value_fails_closed():
    """Опечатка — :class:`GradCkptPolicyError`, не тихий откат в дефолт."""
    for bad in ("true", "yes", "on", "per-layer", "per_layer_blocked", "full", "none none"):
        with policy_env(bad):
            with pytest.raises(
                net_config.GradCkptPolicyError,
                match=net_config.GRAD_CKPT_POLICY_ENV,
            ):
                net_config.grad_ckpt_policy_mode()
    # Явный аргумент — та же дверь.
    with pytest.raises(net_config.GradCkptPolicyError):
        net_config.grad_ckpt_policy_mode("per_layer/block")


def test_policy_registry_agrees_with_model():
    """Три потребителя одного контракта не расходятся.

    Схема (``GRAD_CKPT_POLICIES``) валидирует значение и отдаёт его env-ручке,
    модель (``net.model.GRAD_CKPT_POLICIES``) строит по нему граф; test_37 пинует
    пару «схема ↔ модель», здесь к пину добавлен третий потребитель — чтение
    окружения обязано знать ровно те же имена (иначе ``per_layer_block`` был бы
    достижим из файла, но не из раннера, или наоборот).
    """
    assert net_config.GRAD_CKPT_POLICIES == tuple(model.GRAD_CKPT_POLICIES)
    assert set(net_config.GRAD_CKPT_POLICIES) == {"none", "per_layer", "per_layer_block"}


# ---------------------------------------------------------------------------
# Точка загрузки cfg: переопределение едет в cfg (носителе и метрик, и журнала)
# ---------------------------------------------------------------------------


def test_load_config_without_env_keeps_declared_policy():
    with policy_env(None):
        cfg = net_config.load_config(net_config.CONFIG_PATH)
    assert cfg.grad_ckpt_policy == "per_layer", "пин l3-full не тронут"
    net_config.validate_config(cfg)


def test_load_config_carries_env_override():
    with policy_env("per_layer_block"):
        cfg = net_config.load_config(net_config.CONFIG_PATH)
    assert cfg.grad_ckpt_policy == "per_layer_block"
    net_config.validate_config(cfg)  # то же значение про schema-дверь

    with policy_env("none"):
        cfg = net_config.load_config(net_config.CONFIG_PATH)
    assert cfg.grad_ckpt_policy == "none"


def test_load_config_refuses_unknown_policy_before_reading_model():
    """Ошибка сборки на загрузке: cfg с мусорным значением не рождается вовсе."""
    with policy_env("per_layer_block!"):
        with pytest.raises(
            net_config.GradCkptPolicyError,
            match=net_config.GRAD_CKPT_POLICY_ENV,
        ):
            net_config.load_config(net_config.CONFIG_PATH)


# ---------------------------------------------------------------------------
# (а) без env — конфиг первичен: фактическое значение в журнале + побитовый пин
# ---------------------------------------------------------------------------


def test_train_without_env_marks_config_policy_in_metrics(tmp_path):
    """Маркер журнала — значение конфига, а не дефолт ручки."""
    metrics = tmp_path / "metrics.jsonl"
    cfg = tiny_config(grad_ckpt_policy="per_layer")
    with policy_env(None):
        result = _train_leg(cfg, metrics_path=metrics)
    assert result.steps_done == 1
    for record in _records(metrics):
        assert record["grad_ckpt_policy"] == "per_layer"


#: Проба дефолтного пути: те же данные/шаги, печатает JSON последней строкой.
#: Запускается (i) в подпроцессе поверх архива BASELINE_COMMIT и (ii) на текущей
#: рабочей ревизии; ``conftest`` импортируется ДО jax — пины ADR-010 действуют
#: на процесс только до создания jax-клиента.
_PROBE = r'''
import json, sys
sys.path.insert(0, "net/tests")
import conftest  # noqa: F401 — пины ADR-010 до import jax
import numpy as np
from conftest import tiny_config
from net import checkpoint, train_loop as tl

cfg = tiny_config(grad_ckpt_policy="per_layer")
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
    """Проба в подпроцессе: env без ручки политики — дефолтное поведение."""
    env = dict(os.environ)
    for key in (
        net_config.GRAD_CKPT_POLICY_ENV,
        tl.ACCUM_JIT_ENV,
        "AXIOM_MASTER_DTYPE",
        "AXIOM_MASTER_COMP",
        "AXIOM_KDA_CC_IMPL",
    ):
        env.pop(key, None)
    command = (
        [sys.executable, str(script)]
        if script is not None
        else [sys.executable, "-c", _PROBE]
    )
    done = subprocess.run(command, cwd=cwd, env=env, capture_output=True, text=True)
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


def test_gate_off_is_bit_identical_to_baseline(tmp_path):
    """Без env — путь побитово прежний: сравнение с ревизией до дельты, точно.

    Сравниваются loss-репризы и tree_hash деревьев двух шагов с накоплением
    (2 микробатча на шаг): любая новая арифметика/замена cfg на дефолтном пути
    видна строкой, а не «в допуске».
    """
    baseline = _extract_baseline(tmp_path)
    base_out = _run_probe(baseline, script=baseline / "probe.py")
    current_out = _run_probe(CASE_DIR)
    assert current_out == base_out, (
        "дефолтный путь разошёлся с базовой ревизией: "
        f"current={current_out!r} baseline={base_out!r}"
    )


# ---------------------------------------------------------------------------
# (б) env=per_layer_block при конфиге per_layer: фактическая политика
# ---------------------------------------------------------------------------


def test_train_env_override_reaches_graph_builder_and_metrics(monkeypatch, tmp_path):
    """Переопределение доходит до ``compute_loss`` и до журнала — не только cfg.

    Spy на ``net.model.compute_loss`` — та самая точка, из которой
    ``net/model.py`` читает ``cfg.grad_ckpt_policy`` (механизм ``per_layer_block``
    пиннут test_37): без переопределения здесь оказался бы ``per_layer``
    конфига.  Поле метрик — маркер A/B-ног в формате master_dtype/accum_jit.
    """
    seen_policies: list[str] = []
    real = model.compute_loss

    def spy(params, cfg_arg, batch, *args, **kwargs):
        seen_policies.append(cfg_arg.grad_ckpt_policy)
        return real(params, cfg_arg, batch, *args, **kwargs)

    monkeypatch.setattr(model, "compute_loss", spy)
    metrics = tmp_path / "override.jsonl"
    cfg = tiny_config(grad_ckpt_policy="per_layer")
    with policy_env("per_layer_block"):
        result = _train_leg(cfg, metrics_path=metrics)
    assert result.steps_done == 1
    assert seen_policies == ["per_layer_block"], (
        "сборщик графа должен видеть переопределённую политику, "
        f"а увидел {seen_policies!r}"
    )
    for record in _records(metrics):
        assert record["grad_ckpt_policy"] == "per_layer_block"


def test_override_goes_through_the_same_validate_config(monkeypatch):
    """Схема не обходится: переопределённое значение валидируется тем же вызовом."""
    checked: list[str] = []
    real = net_config.validate_config

    def spy(cfg_arg):
        checked.append(cfg_arg.grad_ckpt_policy)
        return real(cfg_arg)

    monkeypatch.setattr(net_config, "validate_config", spy)
    cfg = tiny_config(grad_ckpt_policy="per_layer")
    with policy_env("per_layer_block"):
        _train_leg(cfg)
    assert checked and set(checked) == {"per_layer_block"}, (
        f"validate_config должен был увидеть переопределённое значение, а увидел {checked!r}"
    )


# ---------------------------------------------------------------------------
# (в) env=мусор — ошибка сборки, не запуск
# ---------------------------------------------------------------------------


class _NeverConsumed:
    """Поток батчей, который обязан остаться нетронутым при отказе сборки."""

    def __iter__(self):
        return self

    def __next__(self):  # pragma: no cover — исполнение = провал теста
        raise AssertionError("батчи не должны читаться: отказ должен быть на сборке")


def test_train_refuses_unknown_env_before_any_run(monkeypatch, tmp_path):
    """Мусор в переменной: ``GradCkptPolicyError`` до params/компиляции/данных."""
    metrics = tmp_path / "must-not-exist.jsonl"
    cfg = tiny_config(grad_ckpt_policy="per_layer")
    monkeypatch.setenv(net_config.GRAD_CKPT_POLICY_ENV, "per_layer_block2")
    with pytest.raises(
        net_config.GradCkptPolicyError, match=net_config.GRAD_CKPT_POLICY_ENV
    ):
        tl.train(
            cfg,
            _NeverConsumed(),
            train_config=tl.TrainConfig(
                steps=1, lr=1e-2, seed=7, schedule="cosine",
                param_dtype="float32", micro_batch=1,
                metrics_path=metrics, log_every=0,
            ),
            budget=_budget(),
        )
    assert not metrics.exists(), "отказ сборки не должен оставить журнал шагов"
