"""Стадия `rl_base_scheme` конвейера A4 (ADR-005 п. 2) — приёмочные тесты дельты.

Источник истины — `docs/specs/RL-STAGE.delta.md` §4, а не реализация
(`tools/run_rl_smoke.py`). Сценарии:

1. **Симлинк, а не копия** (C-032): чекпойнт-копия в рабочем каталоге —
   отказ до всякого исполнения; симлинк, ведущий вне канонического
   `~/gb10-shared`, — тоже отказ; симлинк внутри диска принимается и
   описывается относительным путём.
2. **`qat_weights` наследуется** из журнала SFT (ADR-005 п. 7, решение по
   вопросу 5 SFT-дельты): `on` переносится, отсутствие поля трактуется как
   `off`, нераспознанное значение не «додумывается» до `on`.
3. **Схема базы** (ADR-005 п. 2): преимущество центрируется по группе,
   токен-уровневая маска по log-ratio срезает вклад токенов за порогом,
   нулевое преимущество даёт нулевой градиент, положительное — сдвигает
   параметры (провод обновления, а не обучение).
4. **Стоп-правило** (ADR-005 п. 6): отсутствие положительной дельты на
   holdout → решение `stop`; отсутствие замера → `not-evaluated` (не
   «продолжаем» по умолчанию).
5. **Журнал стадии** валиден и без абсолютных путей (ADR-014 п. 8): после
   смоук-прогона есть сид, задачи, вердикты, награды, `qat_weights`,
   `tree_hash` чекпойнта; смета AD-8/C-041 сопоставлена (смоук — без сметы).
6. **Детерминизм при пиннутом сиде** (спека §4.1, AD-11): повторный прогон с
   тем же сидом даёт тот же набор задач и ту же первую награду.

Смоук-прогоны тестов идут на CPU (`NET_JAX_BACKEND=cpu`): тест мерит провод
стадии, а не железо. Временные каталоги создаются ВНУТРИ репозитория
(фикстура `repo_scratch`), потому что пути журнала обязаны быть
относительными от его корня; `~/gb10-shared` подменяется скретчем, чтобы
тест не трогал канонический диск.  Чекпойнт входа создаётся **отдельным
процессом** с тем же выбором бэкенда, что у прогона стадии: Orbax пишет
топологию шардирования, и чекпойнт, снятый на другом бэкенде, не
восстановится (та же дисциплина, что в `test_20_sft_stage.py`).
"""

from __future__ import annotations

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
RUNNER = CASE_DIR / "tools" / "run_rl_smoke.py"

#: Модельный vocab смоука: тот же, что посчитает стадия из канонического BPE.
MODEL_VOCAB = 1024

_tmp_counter = [0]


#: Создание чекпойнта входа в отдельном процессе с пиннутым бэкендом
#: (ADR-010: выбор бэкенда делает conftest приёмки, а не вызывающая оболочка).
_MAKE_CKPT_PROGRAM = """
import importlib.util, sys
spec = importlib.util.spec_from_file_location("rrs_make", sys.argv[1])
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
sft = runner._load_sft_runner()
cfg = sft.build_model_config(int(sys.argv[4]), sys.argv[3], sys.argv[5] == "on")
import jax.random as jr          # noqa: E402 — после пиннинга бэкенда
from net import checkpoint, model  # noqa: E402
params = model.init_params(jr.PRNGKey(int(sys.argv[6])), cfg)
print(checkpoint.save_checkpoint(params, sys.argv[2]))
"""


def _load_runner():
    """Загрузить `tools/run_rl_smoke.py` по пути (модуль вне пакета net)."""
    spec = importlib.util.spec_from_file_location("run_rl_smoke_test", RUNNER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["run_rl_smoke_test"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def repo_scratch() -> Iterator[Path]:
    """Временный каталог ВНУТРИ репозитория, удаляется после теста."""
    _tmp_counter[0] += 1
    scratch = Path(tempfile.mkdtemp(
        prefix=f".rl-test-{_tmp_counter[0]}-", dir=str(NET_DIR / "tests")
    ))
    try:
        yield scratch
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def _sft_input(
    scratch: Path, *, qat_weights: str | None = "on", journal: bool = True,
    tree_hash: str | None = None,
) -> tuple[Path, Path]:
    """Вход стадии: чекпойнт (симлинк на скретч-диск) и журнал SFT.

    ``qat_weights=None`` — журнал без поля (спека §4.1: трактуется как ``off``).
    """
    link = scratch / "sft-ckpt"
    journal_path = scratch / "sft-stage-journal.json"
    if journal:
        body: dict = {
            "schema": "sft-stage-journal/v1",
            "stage": "sft",
            "status": "executed",
            "model_config": {"preset": "tiny", "vocab_size": MODEL_VOCAB},
            "checkpoint": {"tree_hash": tree_hash or ""},
        }
        if qat_weights is not None:
            body["qat_weights"] = qat_weights
        journal_path.write_text(
            json.dumps(body, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    return link, journal_path


def _make_input_checkpoint(
    scratch: Path, *, qat_weights: str | None = "on", journal: bool = True
) -> tuple[Path, Path]:
    """Чекпойнт на скретч-«сетевом диске» + симлинк в рабочем каталоге.

    Возвращает ``(симлинк, путь журнала)`` — ровно та монтировка, которую
    требует C-032 (веса канонически на диске, в рабочем каталоге симлинк).
    """
    shared = scratch / "shared"
    canonical = shared / "checkpoints" / "sft-mini"
    canonical.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["GB10_SHARED"] = str(shared)
    env["NET_JAX_BACKEND"] = "cpu"
    proc = subprocess.run(
        [sys.executable, "-c", _MAKE_CKPT_PROGRAM, str(RUNNER), str(canonical),
         "tiny", str(MODEL_VOCAB), "on" if qat_weights == "on" else "off", "1337"],
        cwd=str(REPO_ROOT), env=env, capture_output=True, text=True, timeout=600,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    digest = proc.stdout.strip().splitlines()[-1]
    link, journal_path = _sft_input(
        scratch, qat_weights=qat_weights, journal=journal, tree_hash=digest
    )
    link.symlink_to(canonical)
    return link, journal_path


def _run_stage(
    scratch: Path, *, link: Path, journal: Path, steps: int = 2, group: int = 2,
    train_tasks: int = 1, eval_tasks: int = 1, out: str = "out",
    extra: tuple[str, ...] = (),
) -> subprocess.CompletedProcess:
    """Смоук-прогон стадии подпроцессом; окружение — скретч-диск и CPU."""
    env = dict(os.environ)
    env["GB10_SHARED"] = str(scratch / "shared")
    env["NET_JAX_BACKEND"] = "cpu"
    cmd = [
        sys.executable, str(RUNNER),
        "--sft-ckpt", str(link),
        "--sft-journal", str(journal),
        "--out", str(scratch / out),
        "--ckpt-dir", str(scratch / out / "checkpoint"),
        "--steps", str(steps),
        "--train-tasks", str(train_tasks),
        "--eval-tasks", str(eval_tasks),
        "--group-size", str(group),
        "--max-new-tokens", "2",
        "--model-preset", "tiny",
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
# 1. Симлинк, а не копия (C-032)
# ---------------------------------------------------------------------------


def test_copy_instead_of_symlink_refused(repo_scratch):
    """Копия чекпойнта в рабочем каталоге — отказ (копии весов запрещены)."""
    module = _load_runner()
    shared = repo_scratch / "shared"
    shared.mkdir(parents=True, exist_ok=True)
    copy = repo_scratch / "ckpt-copy"
    copy.mkdir()
    (copy / "manifest.json").write_text("{}", encoding="utf-8")

    with pytest.raises(module.StageError, match="не симлинк, а копия"):
        module.validate_checkpoint_symlink(copy, shared, REPO_ROOT)


def test_symlink_outside_shared_disk_refused(repo_scratch):
    """Симлинк, ведущий вне канонического диска, — отказ (C-032)."""
    module = _load_runner()
    shared = repo_scratch / "shared"
    shared.mkdir(parents=True, exist_ok=True)
    outside = repo_scratch / "outside-ckpt"
    outside.mkdir()
    link = repo_scratch / "sft-ckpt"
    link.symlink_to(outside)

    with pytest.raises(module.StageError, match="вне канонического диска"):
        module.validate_checkpoint_symlink(link, shared, REPO_ROOT)


def test_symlink_inside_shared_disk_accepted(repo_scratch):
    """Симлинк на чекпойнт канонического диска принимается и описывается."""
    module = _load_runner()
    link, _ = _make_input_checkpoint(repo_scratch)
    described, resolved = module.validate_checkpoint_symlink(
        link, repo_scratch / "shared", REPO_ROOT
    )

    assert described["is_symlink"] is True
    assert described["format"] == "orbax"
    assert resolved.is_dir()
    assert described["canonical"].startswith("~") or not described["canonical"].startswith("/")
    assert described["path"].startswith("кейсы/axiom/net/tests/.rl-test-")


# ---------------------------------------------------------------------------
# 2. qat_weights наследуется из журнала SFT
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("journal_value", "expected", "source_fragment"),
    [
        ("on", "on", "sft-stage-journal:qat_weights"),
        ("off", "off", "sft-stage-journal:qat_weights"),
        (None, "off", "отсутствует"),
        ("включено", "off", "не распознано"),
    ],
)
def test_qat_weights_inherited_from_sft_journal(repo_scratch, journal_value, expected,
                                                source_fragment):
    """`qat_weights` переносится из журнала SFT; нет поля — `off` (спека §3.1)."""
    module = _load_runner()
    _, journal_path = _sft_input(repo_scratch, qat_weights=journal_value)
    described = module.read_sft_journal(journal_path, REPO_ROOT)

    assert described["qat_weights"] == expected
    assert source_fragment in described["qat_weights_source"]


def test_missing_sft_journal_refused(repo_scratch):
    """Без журнала SFT вход не определён: стадия отказывается, а не угадывает."""
    module = _load_runner()
    with pytest.raises(module.StageError, match="журнал стадии SFT не найден"):
        module.read_sft_journal(repo_scratch / "нет.json", REPO_ROOT)


# ---------------------------------------------------------------------------
# 3. Схема базы: преимущество, маска по log-ratio, градиент
# ---------------------------------------------------------------------------


def test_group_advantages_are_mean_centered():
    """ADR-005 п. 2: награда ответа минус среднее по K ответам группы."""
    module = _load_runner()
    assert module.group_advantages([0.5, 0.5, 0.5]) == [0.0, 0.0, 0.0]
    advantages = module.group_advantages([1.0, 0.0, 0.5])
    assert sum(advantages) == pytest.approx(0.0)
    assert advantages[0] > 0 > advantages[1]


def _model_setup(vocab: int = 64, preset: str = "tiny", qat: bool = False):
    """Модель, конфиг и синтетический «роллаут» для проверок цели схемы базы."""
    module = _load_runner()
    sft = module._load_sft_runner()
    cfg = sft.build_model_config(vocab, preset, qat)
    import jax.random as jr

    from net import model as model_mod

    params = model_mod.init_params(jr.PRNGKey(0), cfg)
    record = {"prompt_ids": [5, 6, 7, 8], "gen_ids": [9, 10, 11]}
    return module, cfg, params, record


def _objective(module, cfg, record, advantage, *, reference, tau_log=0.2, beta=0.0,
               qat=False, total_tokens=3):
    return module.make_rollout_objective(
        cfg, record, advantage, total_tokens,
        tau_log=tau_log, beta=beta, qat=qat, reference=reference,
    )


def test_token_mask_cuts_tokens_beyond_threshold():
    """Маска по log-ratio выключает вклад токенов за порогом (K2.5 :421–433)."""
    module, cfg, params, record = _model_setup()

    ((loss_on, (share_on, _)), _) = _objective(
        module, cfg, record, 1.0, reference=params, tau_log=0.2
    )(params, params)
    ((loss_cut, (share_cut, _)), _) = _objective(
        module, cfg, record, 1.0, reference=params, tau_log=-1.0
    )(params, params)

    assert float(share_on) == pytest.approx(1.0), "on-policy маска обязана пропускать все токены"
    assert float(share_cut) == pytest.approx(0.0), "порог -1 обязан вырезать все токены"
    assert float(loss_on) != 0.0, "on-policy маска не должна срезать весь сигнал"
    assert float(loss_cut) == pytest.approx(0.0), (
        "при вырезанной маске policy-член обязан обнулиться"
    )


def test_zero_advantage_gives_zero_gradient():
    """Вырожденная группа (advantage = 0) не двигает параметры — видно числом."""
    module, cfg, params, record = _model_setup()
    ((loss, _), grads) = _objective(module, cfg, record, 0.0, reference=params)(
        params, params
    )

    assert float(loss) == pytest.approx(0.0)
    assert module.gradient_norm(grads) == pytest.approx(0.0, abs=1e-12)


def test_positive_advantage_moves_parameters():
    """Ненулевое преимущество даёт ненулевой градиент и сдвиг параметров."""
    module, cfg, params, record = _model_setup()
    ((_, _), grads) = _objective(module, cfg, record, 1.0, reference=params)(
        params, params
    )
    assert module.gradient_norm(grads) > 0.0

    from net import optimizer as optimizer_mod

    step = optimizer_mod.make_step(cfg)
    state = optimizer_mod.init_state(params)
    updated, _ = step(params, grads, state, 1e-2)
    assert module._tree_hash(updated) != module._tree_hash(params)


def test_reference_kl_term_is_not_degenerate():
    """β-член считается к опорной политике, а не к текущей (иначе он фиктивен)."""
    module, cfg, params, record = _model_setup()
    import jax.numpy as jnp
    import jax.random as jr

    from net import model as model_mod

    other = model_mod.init_params(jr.PRNGKey(7), cfg)
    ((loss, _), _) = _objective(
        module, cfg, record, 0.0, reference=other, beta=1.0
    )(params, params)

    assert jnp.isfinite(loss)
    assert float(loss) != pytest.approx(0.0), (
        "KL к отличной опорной политике обязан быть ненулевым"
    )


def test_kl_estimator_is_flat_at_the_reference():
    """Относительная энтропия обязана быть плоской при π_θ = π_ref.

    Голая разность `log π_θ − log π_ref` плоской не является: её градиент в
    опорной точке равен `∇ log π_θ`, то есть тянет политику к собственным
    сэмплам (дообучение вместо регуляризации).  Несмещённая k3-оценка
    `e^d − d − 1` даёт и нулевое значение, и нулевой градиент — иначе вырожденная
    группа (advantage = 0) «обучалась» бы на пустом сигнале, что и было измерено
    на первом прогоне стадии.
    """
    module, cfg, params, record = _model_setup()
    ((loss, _), grads) = _objective(
        module, cfg, record, 0.0, reference=params, beta=1.0
    )(params, params)

    assert float(loss) == pytest.approx(0.0, abs=1e-6)
    assert module.gradient_norm(grads) == pytest.approx(0.0, abs=1e-6)


def test_chunk_padding_gradient_is_finite_and_loss_unchanged():
    """Регрессия находки: паддинг чанков KDA больше не отравляет градиент.

    `net/kda.py:apply_chunked` дополняет последовательность нулями до кратности
    чанку; прежде backward `net/norm.py:l2_norm` на строго нулевом векторе был
    нечисловым (0/0 в производной ‖x‖) и NaN доходил до KDA-параметров.  Фикс
    сети (нулевой тангенс нормы в нуле, forward дословный) снял дефект; полный
    регрессионный набор — `net/tests/test_22_l2norm_padding_gradient.py`.

    Страховка дельты (`safe_chunk`, чанк = длина последовательности) остаётся
    корректной и по-прежнему даёт **тот же** loss: семантика KDA не меняется.
    """
    module = _load_runner()
    sft = module._load_sft_runner()
    import jax
    import jax.numpy as jnp
    import jax.random as jr

    from net import model as model_mod
    from net.norm import l2_norm

    # минимальный воспроизводимый случай самой нормы: градиент конечен и нулевой
    primitive = jax.grad(lambda y: jnp.sum(l2_norm(y)))(jnp.zeros((2, 4)))
    assert bool(jnp.all(jnp.isfinite(primitive))), (
        "l2_norm обязан давать числовой градиент на нулевом векторе"
    )
    assert bool(jnp.all(primitive == 0.0)), (
        "в нуле норма недифференцируема; конвенция фикса — нулевой тангенс"
    )

    cfg = sft.build_model_config(64, "tiny", False)
    params = model_mod.init_params(jr.PRNGKey(0), cfg)
    prompt_ids, gen_ids = list(range(1, 4)), [9, 10, 11]
    seq_len = len(prompt_ids) + len(gen_ids)

    ids = jnp.asarray([prompt_ids + gen_ids], dtype=jnp.int32)

    def loss_at(chunk: int, params):
        logits = model_mod.forward(params, cfg, ids, chunk_size=chunk)
        logp = jax.nn.log_softmax(logits[:, :-1], axis=-1)
        picked = jnp.take_along_axis(logp, ids[:, 1:][..., None], axis=-1)[0, :, 0]
        return -jnp.sum(picked[-len(gen_ids):])

    padded_chunk = seq_len + 8  # pad > 0: заведомо нулевые строки внутри чанка
    loss_padded, grad_padded = jax.value_and_grad(lambda p: loss_at(padded_chunk, p))(params)
    loss_safe, grad_safe = jax.value_and_grad(
        lambda p: loss_at(module.safe_chunk(seq_len), p)
    )(params)

    assert bool(jnp.all(jnp.isfinite(module.gradient_norm(grad_padded)))), (
        "паддинг чанка не имеет права давать нечисловой градиент"
    )
    assert bool(jnp.all(jnp.isfinite(module.gradient_norm(grad_safe)))), (
        "safe_chunk (pad = 0) обязан оставаться числовым"
    )
    assert float(loss_padded) == pytest.approx(float(loss_safe)), (
        "значение loss обязано совпадать: страховка не меняет семантику KDA"
    )
    # Паддинг не влияет и на сам градиент, а не только на его численность.
    for a, b in zip(
        jax.tree_util.tree_leaves(grad_padded), jax.tree_util.tree_leaves(grad_safe)
    ):
        assert jnp.allclose(a, b, rtol=1e-5, atol=1e-6)


def test_nonfinite_gradient_is_detected_before_the_step():
    """Страж шага: нечисловой градиент распознаётся по листьям, а не по норме."""
    module = _load_runner()
    import jax.numpy as jnp

    clean = {"a": jnp.array([1.0, 2.0]), "b": (jnp.array(0.5),)}
    poisoned = {"a": jnp.array([1.0, float("nan")]), "b": (jnp.array(0.5),)}

    assert module.gradient_diagnostics(clean)["finite"] is True
    broken = module.gradient_diagnostics(poisoned)
    assert broken["finite"] is False
    assert broken["bad_leaves"] == 1
    assert broken["total_leaves"] == 2
    assert broken["bad_paths"] == ["a"]


def test_json_float_drops_nonfinite_values():
    """В журнал не попадает NaN: артефакт обязан читаться строгим JSON-парсером."""
    module = _load_runner()
    assert module._json_float(float("nan")) is None
    assert module._json_float(float("inf")) is None
    assert module._json_float(0.5) == 0.5


# ---------------------------------------------------------------------------
# 4. Стоп-правило (ADR-005 п. 6)
# ---------------------------------------------------------------------------


def test_stop_rule_stops_without_positive_delta():
    """Нет положительной дельты на holdout → продолжение прекращается."""
    module = _load_runner()
    decision = module.stop_rule(0.25, 0.25)
    assert decision["evaluated"] is True
    assert decision["positive"] is False
    assert decision["decision"] == "stop"

    worse = module.stop_rule(0.25, 0.10)
    assert worse["positive"] is False
    assert worse["decision"] == "stop"


def test_stop_rule_continues_on_positive_delta():
    """Положительная дельта — единственное основание продолжать RL."""
    module = _load_runner()
    decision = module.stop_rule(0.25, 0.30)
    assert decision["positive"] is True
    assert decision["decision"] == "continue"
    assert decision["delta"] == pytest.approx(0.05)


def test_stop_rule_without_measurement_is_not_evaluated():
    """Отсутствие замера — не «продолжаем»: решение помечено непроведённым (спека §6)."""
    module = _load_runner()
    for before, after in ((None, 0.3), (0.3, None), (None, None)):
        decision = module.stop_rule(before, after)
        assert decision["evaluated"] is False
        assert decision["decision"] == "not-evaluated"


# ---------------------------------------------------------------------------
# 5-6. Смоук-прогон: журнал, чекпойнт, детерминизм
# ---------------------------------------------------------------------------


def test_smoke_journal_has_no_absolute_paths(repo_scratch):
    """Журнал стадии валиден, полон по спеке §3.4 и без абсолютных путей."""
    module = _load_runner()
    link, journal_path = _make_input_checkpoint(repo_scratch, qat_weights="on")
    proc = _run_stage(repo_scratch, link=link, journal=journal_path)
    assert proc.returncode == 0, proc.stdout + proc.stderr

    journal = _journal(repo_scratch)
    assert journal["schema"] == module.JOURNAL_SCHEMA
    assert journal["stage"] == "rl_base_scheme"
    assert journal["status"] == "executed"
    assert journal["scale"] == "smoke"
    assert journal["seed"] == 1337
    assert journal["steps"] == 2
    assert journal["qat_weights"] == "on"
    assert journal["input"]["checkpoint"]["is_symlink"] is True
    assert journal["update_path"]["skipped_nonfinite"] == 0, (
        "страховка не должна срабатывать на длинах без паддинга чанков"
    )
    assert journal["input"]["checkpoint"]["matches_journal_hash"] is True
    assert len(journal["checkpoint"]["tree_hash"]) == 64
    assert journal["checkpoint"]["roundtrip_ok"] is True
    assert journal["checkpoint"]["symlink"] is True

    # поля спеки §3.4
    assert journal["tasks_used"], "журнал не назвал задачи обновления"
    assert set(journal["verdict_mix"]) == {"passed", "failed"}
    assert journal["verdict_mix"]["passed"] + journal["verdict_mix"]["failed"] == 4
    assert journal["reward_mean_first"] is not None
    assert journal["reward_mean_last"] is not None
    assert journal["holdout_before"] is not None
    assert journal["holdout_after"] is not None
    assert journal["wall_clock_s"] > 0
    assert 0.0 <= journal["mix_share"] <= 1.0
    assert journal["scheme"]["hyperparameters"]["group_size_K"] == 2

    # каждая попытка несёт задачу, вердикт и награду
    assert len(journal["attempts"]) == 4
    for attempt in journal["attempts"]:
        assert attempt["task_id"] in journal["tasks_used"]
        assert isinstance(attempt["passed"], bool)
        assert isinstance(attempt["reward_raw"], float)
        assert attempt["level"] == "L0"

    offences = _absolute_paths(journal)
    assert not offences, "абсолютные пути в журнале: " + "; ".join(offences)

    # Run Manifest вердиктных прогонов §8 записан и лежит внутри репозитория
    assert len(journal["comparison"]["runs"]) == 2
    for run in journal["comparison"]["runs"]:
        manifest_path = _repo_path(run["manifest"])
        assert manifest_path.is_file(), f"нет манифеста вердиктного прогона: {run['manifest']}"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert manifest["env_version"] == "environment-v1"
        assert manifest["decoding"]["temperature"] == 0.0


def test_qat_off_propagates_into_journal(repo_scratch):
    """Журнал без поля qat_weights → стадия идёт с `off` (наследование, спека §3.1)."""
    link, journal_path = _make_input_checkpoint(repo_scratch, qat_weights=None)
    proc = _run_stage(repo_scratch, link=link, journal=journal_path, steps=1,
                      group=1, eval_tasks=0)
    assert proc.returncode == 0, proc.stdout + proc.stderr

    journal = _journal(repo_scratch)
    assert journal["qat_weights"] == "off"
    assert journal["model_config"]["qat_enabled"] is False
    assert journal["stop_rule"]["decision"] == "not-evaluated"
    assert journal["holdout_before"] is None


def test_copy_checkpoint_refused_by_stage(repo_scratch):
    """Стадия отказывает на копии чекпойнта: журнал пишется честно, статус absent."""
    source_link, journal_path = _make_input_checkpoint(repo_scratch)
    copy_dir = repo_scratch / "ckpt-as-copy"
    shutil.copytree(source_link.resolve(), copy_dir)

    proc = _run_stage(repo_scratch, link=copy_dir, journal=journal_path, steps=1,
                      group=1, eval_tasks=0)
    assert proc.returncode == 1, proc.stdout + proc.stderr

    journal = _journal(repo_scratch)
    assert journal["status"] == "absent"
    assert "не симлинк, а копия" in journal["error"]
    assert not _absolute_paths(journal), "отказ тоже обязан писать относительные пути"


def test_same_seed_same_tasks_and_first_reward(repo_scratch):
    """Повтор при том же сиде даёт тот же набор задач и ту же первую награду."""
    link, journal_path = _make_input_checkpoint(repo_scratch)
    first = _run_stage(repo_scratch, link=link, journal=journal_path, steps=1,
                       group=1, eval_tasks=0, out="out1")
    assert first.returncode == 0, first.stdout + first.stderr
    journal_first = _journal(repo_scratch, "out1")

    second = _run_stage(repo_scratch, link=link, journal=journal_path, steps=1,
                        group=1, eval_tasks=0, out="out2")
    assert second.returncode == 0, second.stdout + second.stderr
    journal_second = _journal(repo_scratch, "out2")

    assert journal_first["tasks_used"] == journal_second["tasks_used"]
    assert journal_first["reward_mean_first"] == journal_second["reward_mean_first"]
    assert (
        journal_first["environment"]["train_tasks"]
        == journal_second["environment"]["train_tasks"]
    )
    # тот же сид и тот же чекпойнт входа — тот же хеш чекпойнта стадии
    assert (
        journal_first["checkpoint"]["tree_hash"]
        == journal_second["checkpoint"]["tree_hash"]
    )
