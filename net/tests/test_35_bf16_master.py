"""ADR-054 Am.13 (цикл-5) — bf16-мастер: единственное представление дерева.

Delta: ``AXIOM_MASTER_DTYPE=bf16`` держит дерево параметров в bf16 как
единственное представление (``params is master``, третий выход шага цикл-4
упразднён, grads bf16), fp32 остаётся в моментах оптимизатора и в арифметике
применения; ``AXIOM_MASTER_COMP=kahan|sr|none`` выбирает плечо компенсации
потери мелких обновлений ниже шага сетки bf16.

Что пинуется здесь (критерии приёмки Am.13):

* **(а) гейт-off побитово прежний** — прогон против ``net/`` базовой ревизии
  ``8387f4e`` (``git archive`` в подпроцесс, как в ``test_29_compute_dtype``:
  та же модель, те же батчи, те же шаги) — loss'ы и хеши деревьев совпадают
  строка к строке;
* **(б) 10-шаговый CPU-паритет** bf16-kahan против fp32-мастера на малом
  конфиге (lr=1e-3, кампанийный режим «обновление много меньше веса») с
  задокументированным дрейфом loss: измерено **max|dloss| = 3.2e-3** (kahan) и
  4.7e-3 (none) за 10 шагов — ожидание Am.13 (≤ 1e-2 абс.) выполнено;
  при lr=1e-2 тот же прогон даёт 6.9e-2/7.3e-2, но там траектория сама уходит
  на 9.8e-2 и «дрейф» меряет чувствительность, а не bf16-округление;
* **(в) контроль ценности компенсации** — плечо ``none`` не лучше ``kahan`` на
  тех же данных (и плато-сценарий примитива: kahan переносит остаток, none
  теряет его целиком); отдельно пин формулы kahan (см. ниже);
* **(г) sr** — воспроизводимость при фиксированном сиде прогона и участие
  ключа состояния в шаге;
* **(д) чекпойнт** — round-trip обоих форматов (fp32/bf16), кросс-resume
  fp32→bf16 (разовый каст, нулевой comp) и bf16→fp32 (апкаст с
  предупреждением), поле ``master_dtype`` в схеме; старые чекпойнты без поля
  читаются как fp32;
* **(е) donate** — ``donate (0,1,2)`` на step_fn легален во всех ветках режима
  (одиночный микробатч и accum), grad_fn — без donate по params, профайлерная
  ветка (donate=False) жива; дерево params на шаге — тот же объект, что мастер
  (второй копии нет);
* **(ж) контракты гейтов** — дефолты (fp32/none), fail-closed на опечатке,
  явный аргумент сильнее env, ``work_dtype`` при bf16-мастере — ошибка сборки,
  fp32-дерево на входе bf16-шага — отказ, состояние bf16-режима: моменты fp32
  + слот компенсации;
* **(з) метрика-строка** несёт ``opt_batch``/``master_dtype``/``master_comp``;
  связка ``bf16``-мастер × ``grouped`` **запрещена fail-closed** на сборке шага
  (фикс-раунд вехи-6: batched NS5 переупорядочивает редукции — в bf16-хранилище
  расхождение скатывается к шагу сетки, см. :func:`test_h2_...`), а разрешённые
  половины осей (``leaf`` × bf16 и ``grouped`` × fp32) живы.

Пин формулы kahan (ADR-054 Am.13 ↔ TASK.md п.2)
------------------------------------------------

В TASK.md формула записана как ``comp_new = bf16((t − master_f32) − y)``,
``t = master_f32 + y`` (fp32, **до** каста).  Это выражение — ошибка округления
fp32-сложения (``~2^-24`` относительно), а не потеря bf16-хранилища (``2^-9``):
плечо, реализованное буквально, **побитово совпадает с ``none``** (тест
:func:`test_c3_literal_formula_is_bit_identical_to_none`), то есть не
компенсирует ничего.  Реализовано чтение, в котором ``t`` — уже сохранённый
мастер (``comp_new = bf16((t_f32 − master_f32) − y)`` = знак перевёрнутого
остатка хранилища): буква формулы сохранена (``y = update − comp``, те же
операнды), а пара ``(master − comp)`` становится точным трекером fp32-цели
(``eff' = master_f32 + y``).  Расхождение зафиксировано в результате прогона
как ``conflicts_with_prior_decisions``: расхождение с буквой спеки при
сохранении её цели (компенсация мелких обновлений, выбранная первичной).
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import subprocess
import sys
import tarfile
import warnings
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from conftest import tiny_config

from net import checkpoint as checkpoint_mod
from net import master_dtype as md
from net import model, optimizer
from net import train_loop as tl

#: Геометрия смоука: одна последовательность, короткий контекст (CPU-время).
SEQ_LEN = 16
#: ``chunk_size`` — как в ``TrainConfig`` по умолчанию (на T=16 не режет).
CHUNK = 64
#: Базовая ревизия гейт-off (предок дельты Am.13 = rollback-якорь пакета).
BASELINE_COMMIT = "8387f4e"
NET_DIR = Path(__file__).resolve().parents[1]
CASE_DIR = NET_DIR.parent


# ---------------------------------------------------------------------------
# Общие помощники
# ---------------------------------------------------------------------------


def _batches(cfg, steps: int, *, seed: int = 5):
    """Детерминированный поток ``(1, T) int32`` — одинаковые данные для ног."""
    rng = np.random.default_rng(seed)
    return [
        rng.integers(0, cfg.vocab_size, size=(1, SEQ_LEN), dtype=np.int32)
        for _ in range(steps)
    ]


def _budget():
    return tl.Budget(run_ref="bf16-master", path=Path("/dev/null"), present=True)


@contextlib.contextmanager
def master_mode(dtype: str | None = None, comp: str | None = None):
    """Временный режим мастера через env (``None`` — переменную снять)."""
    keys = (md.MASTER_DTYPE_ENV, md.MASTER_COMP_ENV)
    saved = {key: os.environ.get(key) for key in keys}
    try:
        for key, value in ((md.MASTER_DTYPE_ENV, dtype), (md.MASTER_COMP_ENV, comp)):
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


@contextlib.contextmanager
def _no_warnings():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        yield


def _train_leg(cfg, steps: int, *, lr: float = 1e-2, seed: int = 7,
               batches=None, metrics_path=None, ckpt_dir=None,
               checkpoint_every: int = 0, total_steps: int | None = None,
               resume_from=None, param_dtype: str = "float32",
               accum_tokens: int = 0, phase_profile: bool = False):
    return tl.train(
        cfg,
        iter(batches if batches is not None else _batches(cfg, steps, seed=seed)),
        train_config=tl.TrainConfig(
            steps=steps, lr=lr, seed=seed, schedule="cosine", param_dtype=param_dtype,
            metrics_path=metrics_path, ckpt_dir=ckpt_dir,
            checkpoint_every=checkpoint_every,
            total_steps=total_steps, accum_tokens=accum_tokens,
            phase_profile=phase_profile, kpi_every=1, log_every=0,
        ),
        budget=_budget(),
        resume_from=resume_from,
    )


def _leaves(tree):
    return jax.tree_util.tree_leaves(tree)


def _dtype_names(tree) -> set[str]:
    return {np.dtype(jnp.asarray(leaf).dtype).name for leaf in _leaves(tree)}


def _max_leaf_diff(a, b) -> float:
    la, lb = _leaves(a), _leaves(b)
    assert len(la) == len(lb), f"разные деревья: {len(la)} против {len(lb)} листьев"
    return max(
        float(np.max(np.abs(np.asarray(x, np.float32) - np.asarray(y, np.float32))))
        for x, y in zip(la, lb)
    )


def _relative_leaf_gap(a, b) -> float:
    """Наибольший относительный зазор листа — мера, не зависящая от масштаба веса."""
    worst = 0.0
    for x, y in zip(_leaves(a), _leaves(b)):
        xa, ya = np.asarray(x, np.float32), np.asarray(y, np.float32)
        scale = max(float(np.max(np.abs(ya))), 1e-30)
        worst = max(worst, float(np.max(np.abs(xa - ya))) / scale)
    return worst


def _loss_drift(losses, reference) -> float:
    return max(abs(a - b) for a, b in zip(losses, reference))


# ---------------------------------------------------------------------------
# (а) Гейт-off побитово прежний — против базовой ревизии (git archive)
# ---------------------------------------------------------------------------

#: Проба гейт-off: те же данные/шаги, что в тестовом прогоне; печатает JSON,
#: который тест сравнивает с прогоном текущей ревизии.  Запускается в
#: подпроцессе с ``net`` = архив базовой ревизии (абсолютные импорты
#: ``from net import ...`` внутри пакета обязаны резолвиться в неё же).
#: ``conftest`` импортируется ДО ``jax``: пины ADR-010 (бэкенд, точность
#: matmul) действуют на процесс только до создания jax-клиента.
_GATE_OFF_PROBE = r'''
import json, sys
sys.path.insert(0, "net/tests")
import conftest  # noqa: F401 — пины ADR-010 до import jax
import numpy as np
from conftest import tiny_config
from net import checkpoint, train_loop as tl

cfg = tiny_config()  # та же геометрия, что у ноги теста (общая для обеих сторон)
rng = np.random.default_rng(5)
batches = [rng.integers(0, cfg.vocab_size, size=(1, 16), dtype=np.int32) for _ in range(3)]
result = tl.train(
    cfg, iter(batches),
    train_config=tl.TrainConfig(steps=3, lr=1e-2, seed=7, schedule="cosine",
                                param_dtype="float32", log_every=0),
    budget=tl.Budget(run_ref="gate-off", path=__import__("pathlib").Path("/dev/null"),
                     present=True),
)
print(json.dumps({
    "losses": [repr(x) for x in result.losses],
    "params_hash": checkpoint.tree_hash(result.params),
    "master_hash": checkpoint.tree_hash(result.master_params),
    "param_dtype": str(__import__("jax").tree_util.tree_leaves(result.params)[0].dtype),
}))
'''


@pytest.fixture(scope="module")
def baseline_gate_off() -> dict:
    """Прогон ``net/`` ревизии ``BASELINE_COMMIT`` в подпроцессе (см. пробу)."""
    root = NET_DIR / "tests" / ".baseline-gate-off"
    if root.exists():
        import shutil

        shutil.rmtree(root)
    root.mkdir()
    try:
        try:
            archive = subprocess.run(
                ["git", "-C", str(CASE_DIR), "archive", BASELINE_COMMIT, "net/"],
                check=True, capture_output=True,
            )
        except (subprocess.CalledProcessError, FileNotFoundError) as exc:  # pragma: no cover
            pytest.skip(f"baseline {BASELINE_COMMIT} недоступен: {exc}")
        with tarfile.open(fileobj=io.BytesIO(archive.stdout)) as bundle:
            bundle.extractall(root)
        env = dict(os.environ)
        env.pop(md.MASTER_DTYPE_ENV, None)
        env.pop(md.MASTER_COMP_ENV, None)
        probe = root / "probe.py"
        probe.write_text(_GATE_OFF_PROBE, encoding="utf-8")
        completed = subprocess.run(
            [sys.executable, str(probe)], cwd=root, env=env,
            capture_output=True, text=True,
        )
        if completed.returncode != 0:  # pragma: no cover
            raise AssertionError(
                f"прогон базовой ревизии упал: {completed.stderr[-2000:]}"
            )
        yield json.loads(completed.stdout.strip().splitlines()[-1])
    finally:
        import shutil

        shutil.rmtree(root, ignore_errors=True)


def test_a_gate_off_is_bitwise_identical_to_baseline(baseline_gate_off):
    """Гейт снят (дефолт) — loss'ы и деревья побитово прежние (ADR-054 Am.13).

    Сравниваются loss'ы (repr float — побитовое равенство), хеши параметров и
    мастера, dtype дерева.  Малейшее расхождение графа fp32-ветки (или утечка
    bf16-кода в дефолт) видно как другое число.
    """
    cfg = tiny_config()  # ровно та же геометрия, что в подпроцессной пробе
    with master_mode(None, None):
        result = _train_leg(cfg, 3, seed=7, batches=_batches(cfg, 3, seed=5))
    current = {
        "losses": [repr(x) for x in result.losses],
        "params_hash": checkpoint_mod.tree_hash(result.params),
        "master_hash": checkpoint_mod.tree_hash(result.master_params),
        "param_dtype": str(_leaves(result.params)[0].dtype),
    }
    assert current == baseline_gate_off, (
        "fp32-дефолт разошёлся с базовой ревизией — гейт-off не побитово прежний"
    )


# ---------------------------------------------------------------------------
# (ж) Контракты гейтов
# ---------------------------------------------------------------------------


def test_gate_defaults_and_fail_closed():
    """Дефолты (fp32/none), fail-closed на опечатке, явный аргумент сильнее env."""
    with master_mode(None, None):
        assert md.master_dtype_mode() == md.FP32
        assert md.master_comp_mode() == md.NONE
        with pytest.raises(md.MasterDtypeError):
            md.master_dtype_mode("fp16")
        with pytest.raises(md.MasterDtypeError):
            md.master_comp_mode("stochastic")
    with master_mode("BF16 ", " SR "):
        assert md.master_dtype_mode() == md.BF16, "регистр/пробелы нормализуются"
        assert md.master_comp_mode() == md.SR
    with master_mode("bf16", "sr"):
        # Явный аргумент сильнее окружения (как у AXIOM_OPT_BATCH).
        assert md.master_dtype_mode("fp32") == md.FP32
        assert md.master_comp_mode("none") == md.NONE
    with master_mode("bf16", None):
        assert md.master_comp_mode() == md.KAHAN, "дефолт компенсации при bf16"


def test_comp_at_fp32_is_ignored_with_warning():
    """``AXIOM_MASTER_COMP`` при fp32-мастере — игнор с предупреждением, не отказ."""
    with master_mode("fp32", "sr"):
        with pytest.warns(UserWarning, match="игнорируется"):
            assert md.master_comp_mode() == md.NONE
    with master_mode(None, "kahan"):
        with pytest.warns(UserWarning, match="master_comp"):
            assert md.master_comp_mode("kahan") == md.NONE
    with master_mode(None, "none"):
        with _no_warnings():
            assert md.master_comp_mode() == md.NONE, "согласованный none не шумит"
        with _no_warnings():
            assert md.master_comp_mode("none") == md.NONE


def test_work_dtype_with_bf16_master_is_build_error(cfg):
    """Третий выход упразднён: bf16-мастер + ``work_dtype`` — ошибка сборки."""
    with master_mode("bf16", "kahan"):
        with pytest.raises(ValueError, match="work_dtype"):
            optimizer.make_step(cfg, work_dtype="bfloat16")


def test_bf16_state_layout_moments_fp32_extra_slot(cfg):
    """Состояние bf16-режима: моменты fp32, слот компенсации под каждым листом.

    ``none`` — слот пустой (``None``, листьев не добавляет), kahan — bf16-буфер,
    sr — uint32-ключ; моменты fp32 во всех трёх плечах.  Число листьев
    состояния против fp32-раскладки — ровно +1 на каждый лист параметров
    (муон: ``m`` → ``(m, extra)``, adamw: ``(m, v)`` → ``(m, v, extra)``).
    """
    master = jax.tree_util.tree_map(
        lambda leaf: leaf.astype(jnp.bfloat16),
        model.init_params(jr.PRNGKey(3), cfg),
    )
    counts_fp32 = optimizer.state_leaf_counts(master, master_dtype="fp32")
    counts_bf16 = optimizer.state_leaf_counts(master, master_dtype="bf16")
    assert len(counts_fp32) == len(_leaves(master))
    assert all(b == a + 1 for a, b in zip(counts_fp32, counts_bf16))
    for comp, extra_dtype in (("none", None), ("kahan", "bfloat16"), ("sr", "uint32")):
        state = optimizer.init_state(
            master, master_dtype="bf16", master_comp=comp, key_seed=3
        )
        leaves = _leaves(state)
        dtypes = _dtype_names(state)
        assert "float32" in dtypes, "моменты обязаны быть fp32"
        if extra_dtype is None:
            assert dtypes == {"float32"}
            assert len(leaves) == sum(counts_fp32), "пустой слот не должен быть листом"
        else:
            assert extra_dtype in dtypes
            assert len(leaves) == sum(counts_bf16)


def test_bf16_step_rejects_fp32_master(cfg):
    """bf16-режим требует bf16-мастер у входа — иначе отказ, а не тихий fp32."""
    fp32_master = model.init_params(jr.PRNGKey(3), cfg)
    state = optimizer.init_state(fp32_master)
    step = optimizer.make_step(cfg, master_dtype="bf16", master_comp="none")
    grads = jax.tree_util.tree_map(
        lambda leaf: jnp.full_like(leaf, 1e-3, dtype=jnp.bfloat16), fp32_master
    )
    with pytest.raises(ValueError, match="bf16"):
        step(fp32_master, grads, state, 1e-2)


def test_third_output_is_gone_and_work_copy_is_master(cfg):
    """В bf16-мастере шаг двухвыходной; ``params`` на шаге — тот же объект, что мастер.

    Сравнение с прежним bf16-режимом (рабочая копия + fp32-мастер): там шаг
    трёхвыходной, а дерево параметров пересоздаётся кастом.  В цикле Am.13
    второй копии нет: аргумент ``params`` шага N+1 — **тот же массив**, что
    нулевой выход шага N (в рабочей копии это было бы другое дерево).
    """
    seen: list[tuple] = []
    real_make_step = optimizer.make_step

    def spy(*args, **kwargs):
        step = real_make_step(*args, **kwargs)

        def recording(params, grads, state, lr):
            out = step(params, grads, state, lr)
            seen.append((params, grads, state, out))
            return out

        return recording

    with master_mode("bf16", "kahan"):
        optimizer.make_step = spy
        try:
            result = _train_leg(cfg, 3, seed=7)
        finally:
            optimizer.make_step = real_make_step
    assert len(seen) == 3
    assert all(len(record[3]) == 2 for record in seen), "в bf16-мастере третьего выхода нет"
    assert _dtype_names(seen[0][0]) == {"bfloat16"}
    assert _dtype_names(seen[0][1]) == {"bfloat16"}, "grads — bf16"
    for previous, current in zip(seen, seen[1:]):
        assert _leaves(previous[3][0])[0] is _leaves(current[0])[0], (
            "дерево параметров пересоздано между шагами — рабочая копия вернулась"
        )
    assert result.steps_done == 3
    assert all(np.isfinite(loss) for loss in result.losses)


# ---------------------------------------------------------------------------
# (б)/(в)/(г) Паритет, контроль компенсации, воспроизводимость sr
# ---------------------------------------------------------------------------

#: 10 шагов паритета: lr малый — кампанийный режим, где обновление за шаг
#: много меньше веса (в боевой клетке B=2@8192 lr мал).  При lr=1e-2 на tiny-
#: конфиге траектория сама уходит на ~1e-1 за 10 шагов, и «дрейф» меряет
#: чувствительность траектории, а не bf16-округление: измерено на CPU —
#: max|dloss| = 6.9e-2 (kahan) / 7.3e-2 (none), т.е. порядка собственного хода
#: fp32-ноги (9.8e-2).  При lr=1e-3 те же 10 шагов: 3.2e-3 (kahan) / 4.7e-3
#: (none) — уже в допуске Am.13 (≤1e-2), и именно это число уходит в отчёт.
_PARITY_STEPS = 10
_PARITY_LR = 1e-3


@pytest.fixture(scope="module")
def parity_legs():
    """Одни и те же 10 шагов четырьмя ногами: fp32, bf16-kahan, bf16-none, bf16-sr."""
    cfg = tiny_config()
    batches = _batches(cfg, _PARITY_STEPS, seed=11)
    legs = {}
    with master_mode(None, None):
        legs["fp32"] = _train_leg(cfg, _PARITY_STEPS, lr=_PARITY_LR, batches=batches)
    for comp in ("kahan", "none", "sr"):
        with master_mode("bf16", comp):
            legs[f"bf16-{comp}"] = _train_leg(
                cfg, _PARITY_STEPS, lr=_PARITY_LR, batches=batches
            )
    return cfg, legs


def test_b_ten_step_parity_kahan_against_fp32_master(parity_legs):
    """10 шагов bf16-kahan против fp32-мастера: дрейф loss в допуске Am.13.

    Критерий приёмки Am.13 — «дрейф — задокументированная дельта (ожидание
    ≤ ~1e-2 абс. на 10 шагах)»: здесь он измеряется и печатается; превышение —
    вердикт в амендмент, а не подгон допуска (поэтому порог теста равен
    ожиданию, а фактическое число уходит в отчёт прогона).
    """
    _cfg, legs = parity_legs
    fp32, kahan = legs["fp32"], legs["bf16-kahan"]
    drift = _loss_drift(kahan.losses, fp32.losses)
    relative = _relative_leaf_gap(kahan.master_params, fp32.master_params)
    print(
        f"[паритет] kahan vs fp32: max|dloss|={drift:.3e} "
        f"|dloss_final|={abs(kahan.losses[-1] - fp32.losses[-1]):.3e} "
        f"отн. |dparam|={relative:.3e}"
    )
    assert drift <= 1e-2, (
        f"дрейф loss за 10 шагов {drift:.3e} > 1e-2 — критерий Am.13 нарушен"
    )
    assert all(np.isfinite(loss) for loss in kahan.losses)
    assert _dtype_names(kahan.master_params) == {"bfloat16"}


def test_c_none_arm_is_not_better_than_kahan(parity_legs):
    """Плечо ``none`` (контроль ущерба) не лучше kahan на тех же данных.

    Компенсация — механизм Am.13, а не декларация: если плечо без компенсации
    систематически ближе к fp32-траектории, ценность компенсации нулевая и
    вердикт идёт архитектору.  Критерий меряется по **дрейфу loss** — той же
    метрике, что критерий (б) (Am.13); заострённый сценарий ценности (плато
    мелких обновлений) — в :func:`test_c2_...`.  Относительный зазор параметров
    печатается диагностикой: на tiny-конфиге он смешивает масштабы (листья с
    малым |w| двигаются обновлением O(lr·upd) независимо от своего масштаба) и
    вердиктным числом не является.
    """
    _cfg, legs = parity_legs
    fp32 = legs["fp32"]
    kahan, none = legs["bf16-kahan"], legs["bf16-none"]
    drift_k = _loss_drift(kahan.losses, fp32.losses)
    drift_n = _loss_drift(none.losses, fp32.losses)
    print(
        f"[контроль] loss-дрейф kahan={drift_k:.3e} none={drift_n:.3e}; "
        f"диагностика отн. |dparam| kahan={_relative_leaf_gap(kahan.master_params, fp32.master_params):.3e} "
        f"none={_relative_leaf_gap(none.master_params, fp32.master_params):.3e}"
    )
    assert drift_n >= drift_k - 1e-9, (
        f"плечо none точнее kahan по loss ({drift_n:.3e} < {drift_k:.3e}) — "
        "компенсация не даёт ценности"
    )


def test_c2_kahan_carries_the_lost_remainder_none_loses_it():
    """Плато bf16: обновление ниже полушага сетки теряется ``none`` и переносится kahan.

    Сценарий, ради которого плечо существует (Am.13: «прямое обновление
    bf16-мастера теряет ``lr·update < |w|·2^-8``»): ``w = 1.0``, 2000
    обновлений ``+1e-5``.  Идеал fp32 — 1.020027; ``none`` не применяет ни
    одного обновления (1.0), kahan доводит трекер ``master_f32 − comp`` до
    идеала в пределах bf16-точности буфера (измерено: 1.0225, ошибка 2.4e-3
    против 2.0e-2 у ``none`` — на порядок меньше).
    """
    updates = 2000
    step = jnp.asarray(1e-5)
    ideal = np.float32(1.0)
    for _ in range(updates):
        ideal = np.float32(ideal + step)

    master = jnp.asarray(1.0, jnp.bfloat16)
    comp = jnp.asarray(0.0, jnp.bfloat16)
    for _ in range(updates):
        master, comp = md.kahan_apply(master, step, comp)
    kahan_value = float(master.astype(jnp.float32) - comp.astype(jnp.float32))

    plain = jnp.asarray(1.0, jnp.bfloat16)
    for _ in range(updates):
        plain = md.rn_apply(plain, step)
    none_value = float(plain)

    assert none_value == 1.0, "none перестал терять мелкие обновления — сценарий не тот"
    assert abs(kahan_value - float(ideal)) < 5e-3, (
        f"kahan не удержал плато: {kahan_value} против идеала {float(ideal)}"
    )
    assert abs(kahan_value - float(ideal)) < abs(none_value - float(ideal)) / 4


def test_c3_literal_formula_is_bit_identical_to_none():
    """Буквальная формула TASK.md (``(t − master_f32) − y``) тождественна ``none``.

    ``t`` — fp32-сумма до каста, поэтому ``(t − master_f32) − y`` есть ошибка
    округления fp32-сложения (Knuth TwoSum, ``~2^-24`` относительно), а не
    потеря bf16-хранилища (``2^-9``): буфер получает ~1e-8 вместо ~1e-3, и
    траектория совпадает с ``none`` побитово.  Тест фиксирует измеренную
    причину отклонения прочтения (докстринг модуля, ``result.json``).
    """
    updates = 500
    step = jnp.asarray(1e-5)
    literal_master = jnp.asarray(1.0, jnp.bfloat16)
    literal_comp = jnp.asarray(0.0, jnp.bfloat16)
    plain_master = jnp.asarray(1.0, jnp.bfloat16)
    for _ in range(updates):
        m32 = literal_master.astype(jnp.float32)
        y = step - literal_comp.astype(jnp.float32)
        t = m32 + y                                  # fp32, до каста
        literal_comp = ((t - m32) - y).astype(jnp.bfloat16)
        literal_master = t.astype(jnp.bfloat16)
        plain_master = md.rn_apply(plain_master, step)
    assert float(literal_master) == float(plain_master) == 1.0
    assert float(literal_comp) != 0.0  # буфер не пуст, но на порядки меньше потери
    assert abs(float(literal_comp)) < 1e-6, (
        "буквальная формула вдруг начала копить bf16-остаток — пересмотри прочтение"
    )


def test_d_sr_is_reproducible_under_fixed_seed(parity_legs):
    """sr-плечо воспроизводимо: тот же сид прогона — те же loss'ы и дерево.

    Поток случайности sr заводится ключом в состоянии (сид прогона): повторный
    прогон с тем же сидом обязан совпасть побитово, иначе SR нельзя ставить в
    батарею (Am.13: «seeded, воспроизводимо»).
    """
    cfg, legs = parity_legs
    reference = legs["bf16-sr"]
    with master_mode("bf16", "sr"):
        repeated = _train_leg(
            cfg, _PARITY_STEPS, lr=_PARITY_LR,
            batches=_batches(cfg, _PARITY_STEPS, seed=11),
        )
    assert [repr(x) for x in repeated.losses] == [repr(x) for x in reference.losses]
    assert _max_leaf_diff(repeated.master_params, reference.master_params) == 0.0


def test_d2_sr_key_participates_and_kahan_ignores_it(cfg):
    """Ключ состояния — источник случайности sr: смена сида меняет шаг (sr), не (kahan).

    Оптимизаторный уровень: одинаковые мастер/грады/моменты, разные ``key_seed``.
    sr обязан дать другое дерево (ключ едет в шаге), kahan — то же самое
    (буфер компенсации от сида не зависит, арифметика детерминирована).
    """
    master = jax.tree_util.tree_map(
        lambda leaf: leaf.astype(jnp.bfloat16),
        model.init_params(jr.PRNGKey(3), cfg),
    )
    grads = jax.tree_util.tree_map(
        lambda leaf: jnp.full_like(leaf, 1e-3, dtype=jnp.bfloat16), master
    )
    for comp, must_differ in (("sr", True), ("kahan", False), ("none", False)):
        state_a = optimizer.init_state(
            master, master_dtype="bf16", master_comp=comp, key_seed=1
        )
        state_b = optimizer.init_state(
            master, master_dtype="bf16", master_comp=comp, key_seed=2
        )
        with master_mode("bf16", comp):
            step = optimizer.make_step(cfg, master_dtype="bf16", master_comp=comp)
            a_master, _ = step(master, grads, state_a, 1e-2)
            b_master, _ = step(master, grads, state_b, 1e-2)
        gap = _max_leaf_diff(a_master, b_master)
        if must_differ:
            assert gap > 0.0, "смена key_seed не изменила sr-шаг — ключ не участвует"
        else:
            assert gap == 0.0, f"{comp} зависит от key_seed — арифметика не детерминирована"


def test_d3_stochastic_rounding_is_unbiased_and_deterministic():
    """Примитив SR: детерминирован при ключе и не смещён (E ≈ x) за много проб.

    Округление вверх случается с вероятностью дробной части (трюк «прибавить
    16 случайных бит мантиссы и отрезать их»), поэтому среднее по потоку ключей
    сходится к исходному значению, а не к границе сетки.
    """
    values = jnp.asarray(
        np.concatenate(
            [
                np.linspace(0.5, 1.5, 201, dtype=np.float32),
                np.linspace(-1.5, -0.5, 201, dtype=np.float32),
            ]
        )
    )
    key = jr.PRNGKey(0)
    probes = 800
    total = jnp.zeros_like(values)
    for index in range(probes):
        total = total + md.stochastic_round_bf16(values, jr.fold_in(key, index)).astype(
            jnp.float32
        )
    mean = np.asarray(total / probes)
    source = np.asarray(values)
    half_ulp = np.abs(source) * 2.0 ** -9
    assert float(np.max(np.abs(mean - source))) < float(np.max(half_ulp)) / 2.0, (
        "среднее SR ушло от значения дальше половины полушага сетки — смещение"
    )
    again = md.stochastic_round_bf16(values, jr.PRNGKey(7))
    assert np.array_equal(
        np.asarray(again), np.asarray(md.stochastic_round_bf16(values, jr.PRNGKey(7)))
    ), "SR не детерминирован при том же ключе"


# ---------------------------------------------------------------------------
# (д) Чекпойнт: схема, round-trip, кросс-resume
# ---------------------------------------------------------------------------


def test_e_manifest_schema_carries_master_dtype(cfg, tmp_path):
    """Схема манифеста несёт ``master_dtype``; старый манифест читается как fp32."""
    fp32_params = model.init_params(jr.PRNGKey(0), cfg)
    bf16_params = jax.tree_util.tree_map(
        lambda leaf: leaf.astype(jnp.bfloat16), fp32_params
    )
    d_fp32 = tmp_path / "fp32"
    d_bf16 = tmp_path / "bf16"
    checkpoint_mod.save_checkpoint(fp32_params, d_fp32)
    checkpoint_mod.save_checkpoint(bf16_params, d_bf16, master_dtype="bf16")
    assert checkpoint_mod.read_manifest(d_fp32 / "manifest.json")["master_dtype"] == "fp32"
    assert checkpoint_mod.read_manifest(d_bf16 / "manifest.json")["master_dtype"] == "bf16"
    assert checkpoint_mod.read_manifest_master_dtype(d_fp32) == "fp32"
    assert checkpoint_mod.read_manifest_master_dtype(d_bf16) == "bf16"
    # Чекпойнт прежней ревизии: поля нет — читатель видит fp32, не падает.
    legacy = tmp_path / "legacy"
    legacy.mkdir()
    (legacy / "manifest.json").write_text(
        json.dumps({"checkpoint_hash": "abc", "format": "orbax"}), encoding="utf-8"
    )
    assert checkpoint_mod.read_manifest_master_dtype(legacy) == "fp32"
    assert checkpoint_mod.checkpoint_master_dtype({}) == "fp32"
    with pytest.raises(ValueError):
        checkpoint_mod.checkpoint_master_dtype({"master_dtype": "fp16"})


def test_e2_roundtrip_both_formats_and_hash_is_of_stored_tree(cfg, tmp_path):
    """Round-trip обоих форматов: байты совпадают, хеш — хранимого дерева."""
    fp32_params = model.init_params(jr.PRNGKey(0), cfg)
    bf16_params = jax.tree_util.tree_map(
        lambda leaf: leaf.astype(jnp.bfloat16), fp32_params
    )
    for name, params, dtype in (("fp32", fp32_params, "fp32"), ("bf16", bf16_params, "bf16")):
        directory = tmp_path / name
        digest = checkpoint_mod.save_checkpoint(params, directory, master_dtype=dtype)
        restored = checkpoint_mod.load_checkpoint(directory, target=params)
        assert checkpoint_mod.tree_hash(restored) == digest
        assert _max_leaf_diff(restored, params) == 0.0
        other_dtype = "fp32" if dtype == "bf16" else "bf16"
        other = checkpoint_mod.load_checkpoint(
            directory, target=params, master_dtype=other_dtype
        )
        expected = "float32" if other_dtype == "fp32" else "bfloat16"
        assert _dtype_names(other) == {expected}, (
            "читатель, требующий dtype, обязан получить ровно его"
        )


def test_e3_cross_resume_fp32_to_bf16_and_back(tmp_path):
    """Кросс-resume: fp32-чекпойнт → bf16-нога (каст, нулевой comp) и обратно.

    Обе стороны контракта Am.13 п.3: продолжение в bf16-режиме — каст хранимого
    fp32-дерева и свежие слоты компенсации (нулевой буфер); продолжение в
    fp32-режиме — апкаст bf16-дерева **с предупреждением** (апкаст не
    возвращает потерянные хранилищем обновления — это осознанное действие).
    """
    cfg = tiny_config()
    d1 = tmp_path / "run1"
    with master_mode(None, None):
        first = _train_leg(cfg, 2, seed=7, ckpt_dir=d1, checkpoint_every=1, total_steps=10)
    with master_mode("bf16", "kahan"):
        second = _train_leg(
            cfg, 1, seed=7, ckpt_dir=d1, total_steps=10,
            resume_from=tl.CheckpointManager(d1),
        )
    assert first.steps_done == 2
    assert second.start_step == 2
    assert _dtype_names(second.master_params) == {"bfloat16"}
    assert np.isfinite(second.losses[0])
    d2 = tmp_path / "run2"
    with master_mode("bf16", "kahan"):
        third = _train_leg(cfg, 2, seed=7, ckpt_dir=d2, checkpoint_every=1, total_steps=10)
    with master_mode(None, None), pytest.warns(UserWarning, match="апкаст"):
        fourth = _train_leg(
            cfg, 1, seed=7, ckpt_dir=d2, total_steps=10,
            resume_from=tl.CheckpointManager(d2),
        )
    assert third.steps_done == 2
    assert fourth.start_step == 2
    assert _dtype_names(fourth.master_params) == {"float32"}
    assert np.isfinite(fourth.losses[0])


def test_e4_stored_master_mode_reads_meta_and_defaults(cfg, tmp_path):
    """``stored_master_mode``: bf16-запись читается как bf16, старая — fp32/none."""
    d = tmp_path / "ck"
    with master_mode("bf16", "sr"):
        _train_leg(cfg, 1, seed=7, ckpt_dir=d, checkpoint_every=1, total_steps=4)
    manager = tl.CheckpointManager(d)
    assert manager.stored_master_mode() == ("bf16", "sr")
    meta = json.loads((d / "step-00000001" / "meta.json").read_text(encoding="utf-8"))
    assert meta["master_dtype"] == "bf16" and meta["master_comp"] == "sr"
    for path in d.iterdir():
        if path.is_dir():
            payload = json.loads((path / "meta.json").read_text(encoding="utf-8"))
            payload.pop("master_dtype")
            payload.pop("master_comp")
            (path / "meta.json").write_text(json.dumps(payload), encoding="utf-8")
    assert manager.stored_master_mode() == ("fp32", "none"), "старая запись = fp32/none"


def test_e5_adapt_state_preserves_moments_and_zeroes_comp(cfg):
    """``adapt_state``: моменты переносятся, слот компенсации заводится свежим."""
    params = model.init_params(jr.PRNGKey(3), cfg)
    fp32_state = optimizer.init_state(params)
    for comp in ("kahan", "sr", "none"):
        adapted = optimizer.adapt_state(
            fp32_state, params, saved_master_dtype="fp32",
            saved_master_comp="none", master_dtype="bf16", master_comp=comp,
        )
        expected = optimizer.init_state(
            params, master_dtype="bf16", master_comp=comp
        )
        assert jax.tree_util.tree_structure(adapted) == jax.tree_util.tree_structure(
            expected
        ), "адаптация fp32→bf16 разошлась с раскладкой шага"
    # Обратно: bf16-состояние → fp32-раскладка (слоты отбрасываются, моменты целы).
    bf16_state = optimizer.init_state(
        params, master_dtype="bf16", master_comp="kahan", key_seed=5
    )
    back = optimizer.adapt_state(
        bf16_state, params, saved_master_dtype="bf16", saved_master_comp="kahan",
        master_dtype="fp32", master_comp="none",
    )
    assert jax.tree_util.tree_structure(back) == jax.tree_util.tree_structure(fp32_state)
    for restored, original in zip(_leaves(back), _leaves(fp32_state)):
        assert np.array_equal(np.asarray(restored), np.asarray(original))
    with pytest.raises(ValueError, match="раскладке"):
        optimizer.adapt_state(
            jnp.zeros((2, 2), jnp.float32), params, saved_master_dtype="fp32",
            saved_master_comp="none", master_dtype="fp32", master_comp="none",
        )


# ---------------------------------------------------------------------------
# (е) donate: легальность во всех ветках режима
# ---------------------------------------------------------------------------


def test_f_step_donate_legal_in_bf16_master(cfg):
    """``donate (0,1,2)`` легален в bf16-мастере: паритет и серия шагов.

    Шаг получает donate, потому что ``params is master`` и буфер градиента не
    читается никем после вызова (третьего дерева в режиме нет вовсе).  Проверка
    паритета: donate — контракт владения буферами, не арифметика (Am.1), и в
    bf16-мастере он легален для master/grads/state во всех ветках.
    """
    master = jax.tree_util.tree_map(
        lambda leaf: leaf.astype(jnp.bfloat16),
        model.init_params(jr.PRNGKey(3), cfg),
    )
    grads = jax.tree_util.tree_map(
        lambda leaf: jnp.full_like(leaf, 1e-3, dtype=jnp.bfloat16), master
    )
    with master_mode("bf16", "kahan"):
        state = optimizer.init_state(master, master_dtype="bf16", master_comp="kahan")
        step_plain = optimizer.make_step(cfg, master_dtype="bf16", master_comp="kahan")
        a_master, a_state = step_plain(master, grads, state, 1e-2)
        # Копия градиентов для донированных шагов делается ДО первой донации:
        # после ``step_donated`` исходный буфер мёртв (проверяется ниже).
        template = jax.tree_util.tree_map(lambda leaf: jnp.array(leaf), grads)
        step_donated = optimizer.make_step(
            cfg, master_dtype="bf16", master_comp="kahan", donate=True
        )
        b_master, b_state = step_donated(master, grads, state, 1e-2)
        assert _max_leaf_diff(a_master, b_master) == 0.0, "donate изменил арифметику"
        assert _max_leaf_diff(a_state, b_state) == 0.0
        # Контракт донации соблюдён самой средой: переданные буферы мертвы.
        dead = 0
        for donated_tree in (master, grads, state):
            for leaf in _leaves(donated_tree):
                try:
                    np.asarray(leaf)
                except RuntimeError:
                    dead += 1
        assert dead > 0, "donate не освободил ни одного буфера — контракт не проверен"
        for _ in range(3):
            fresh = jax.tree_util.tree_map(lambda leaf: jnp.array(leaf), template)
            b_master, b_state = step_donated(b_master, fresh, b_state, 1e-2)
        assert all(np.isfinite(np.asarray(leaf, np.float32)).all() for leaf in _leaves(b_master))


def test_f2_train_loop_donation_and_accum_branch(cfg):
    """Цикл: donate=True без профайлера, accum-ветка (2 микробатча) легальна.

    В accum-ветке ``params`` читается на каждом микровызове градиента — donate
    градиента там был бы use-after-free, поэтому цикл берёт недонированный
    ``grad_fn``; donate шага при этом остаётся легальным (его входы — master,
    grads, state — после вызова мертвы по построению режима).
    """
    seen: list[dict] = []
    real_make_step = optimizer.make_step

    def spy(*args, **kwargs):
        seen.append(dict(kwargs))
        return real_make_step(*args, **kwargs)

    with master_mode("bf16", "kahan"):
        optimizer.make_step = spy
        try:
            single = _train_leg(cfg, 2, seed=7)
            accum = _train_leg(cfg, 1, seed=7, accum_tokens=SEQ_LEN * 2)
        finally:
            optimizer.make_step = real_make_step
    assert seen and all(record.get("donate") is True for record in seen)
    assert single.steps_done == 2 and accum.steps_done == 1
    assert all(np.isfinite(loss) for loss in single.losses + accum.losses)


def test_f3_profile_branch_keeps_grads_alive(cfg):
    """Профайлерная ветка (``phase_profile``) — donate=False: grads читаются после шага."""
    profiled_seen: list[dict] = []
    real_make_step = optimizer.make_step

    def spy(*args, **kwargs):
        profiled_seen.append(dict(kwargs))
        return real_make_step(*args, **kwargs)

    with master_mode("bf16", "kahan"):
        optimizer.make_step = spy
        try:
            result = _train_leg(cfg, 2, seed=7, phase_profile=True)
        finally:
            optimizer.make_step = real_make_step
    assert profiled_seen and profiled_seen[0].get("donate") is False
    assert result.steps_done == 2 and all(np.isfinite(loss) for loss in result.losses)


# ---------------------------------------------------------------------------
# (з) Метрика-строка и ортогональность осей режимов
# ---------------------------------------------------------------------------


def test_h_metrics_rows_carry_active_modes(cfg, tmp_path):
    """Каждая строка метрик несёт ``opt_batch``/``master_dtype``/``master_comp``."""
    metrics = tmp_path / "metrics.jsonl"
    with master_mode("bf16", "sr"):
        result = _train_leg(cfg, 2, seed=7, metrics_path=metrics)
    assert result.steps_done == 2
    rows = tl.MetricsWriter.read(metrics)
    assert len(rows) == 2
    for row in rows:
        assert row["master_dtype"] == "bf16"
        assert row["master_comp"] == "sr"
        assert row["opt_batch"] in optimizer.OPT_BATCH_MODES
    metrics2 = tmp_path / "metrics-fp32.jsonl"
    with master_mode(None, None):
        _train_leg(cfg, 1, seed=7, metrics_path=metrics2)
    row = tl.MetricsWriter.read(metrics2)[0]
    assert row["master_dtype"] == "fp32" and row["master_comp"] == "none"


def test_h2_opt_batch_grouped_is_rejected_in_bf16_master(cfg):
    """``bf16``-мастер × ``AXIOM_OPT_BATCH=grouped`` — fail-closed на сборке (Am.13).

    Фикс-раунд вехи-6: связка отвергнута, а не «тихо живёт».  Диагноз измерен
    на CPU (small-конфиг), а не унаследован:

    * применение компенсации в grouped-пути **симметрично** leaf — каждый лист
      применяется со своим ``comp``-буфером (штемпель: ручная реконструкция
      шага-2 побитово повторяет обе ноги), так что подозрение (а) TASK.md не
      подтверждается;
    * корень — batched NS5: он переупорядочивает редукции (Am.11), дельта
      обновления на бакете ``expert_d`` (6,16,32) — до ~1.1e-5 абс. (~2e-5 отн.);
    * в bf16-хранилище микро-сдвиг после ``lr`` (~1.6e-8) у части элементов
      ложится вблизи середины округления — сохранённые листья расходятся **ровно
      на шаг сетки bf16** (соседние представимые значения; проверено побитово:
      диф 1.907e-06 = next-представимое bf16 при |w|~4.4e-4);
    * формула порога старого теста ``|y|·2^-8`` недооценивает шаг сетки: у bf16
      семь бит мантиссы, шаг в 1..2 раза больше, поэтому «ровно одно округление
      хранилища» по этой формуле неотличимо от расходимости — по построению
      зелёного «как есть» не существует (батчинг меняет порядок редукций).
      Решение — запрет на сборке шага с читаемой ошибкой (как
      :class:`optimizer.UnclassifiedMatrixError`), а не тихий дрейф.

    Разрешённые половины осей остаются живы: ``leaf`` × bf16 и ``grouped`` × fp32
    (вторая — прежняя дорога вехи-5, ``test_34``).
    """
    with master_mode("bf16", "kahan"):
        with pytest.raises(optimizer.IncompatibleModeError, match="grouped"):
            optimizer.make_step(cfg, master_dtype="bf16", opt_batch="grouped")
        # Окружение — тот же запрет; явный leaf-аргумент снимает связку.
        previous = os.environ.get(optimizer.OPT_BATCH_ENV)
        os.environ[optimizer.OPT_BATCH_ENV] = optimizer.OPT_BATCH_GROUPED
        try:
            with pytest.raises(
                optimizer.IncompatibleModeError, match=optimizer.OPT_BATCH_ENV
            ):
                optimizer.make_step(cfg, master_dtype="bf16")
            optimizer.make_step(cfg, master_dtype="bf16", opt_batch="leaf")
        finally:
            if previous is None:
                os.environ.pop(optimizer.OPT_BATCH_ENV, None)
            else:
                os.environ[optimizer.OPT_BATCH_ENV] = previous
    # fp32 × grouped — прежняя дорога, запрет её не касается.
    optimizer.make_step(cfg, master_dtype="fp32", opt_batch="grouped")
