#!/usr/bin/env python3
"""Стадия ``rl_base_scheme`` конвейера A4 — цикл RL по схеме базы (ADR-005 п. 2).

Реализация дельты ``docs/specs/RL-STAGE.delta.md`` §3.  Прогон по умолчанию —
**смоук**: короткий цикл (``--steps`` <= 100), доказывающий провод
«генерация задач → исполнение политикой → механический вердикт среды →
награда → обновление по схеме базы → чекпойнт → журнал», а не обучение
модели.  Масштаб (шаги, задачи, роллауты, часы) пишется в журнал явно:
подмена смоука закрытием стадии запрещена (спека §5).

Что делает прогон (нумерация — по спеке §3):

1. **Вход.** Чекпойнт стадии SFT — **симлинк** на канонический диск
   ``~/gb10-shared`` (копия весов в рабочем каталоге — отказ до всякого
   исполнения, C-032); признак ``qat_weights`` наследуется из журнала SFT
   (``evidence/a4-run-wire/sft/stage-journal.json``); отсутствие поля
   трактуется как ``off`` (SFT-дельта, решение по вопросу 5).  Конфигурация
   сети и ``vocab_size`` берутся из журнала SFT — чекпойнт обязан
   восстанавливаться в ту же структуру, иначе загрузка молча сломала бы
   преемственность стадии.
2. **Среда v1 как есть** (``env/``): генерация задач (``env/generate.py``),
   исполнение политикой (``env/net_executor.py``), механический вердикт
   (``env/verifier.py``), детерминированная награда (``env/reward.py``).
   Среда не переписывается; из неё берутся существующие функции.
   Режим смоука — детерминированный: сид пиннут, decoding-параметры
   записаны в журнал, вердиктные замеры идут при temperature = 0.
3. **Цикл** (спека §3.2): на каждом шаге задача ротируется по пиннутому
   списку, исполняется K роллаутов (группа), считается групповая награда
   (взвешенная по сложности задачи, ADR-005 п. 2) и обновление по схеме базы.
4. **Схема базы** (ADR-005 п. 2; ``docs/research/k3-tech-report-extract.md``
   :421–433 для K2.5 и :287 для K1.5):

   * преимущество — групповое, без критика: ``a_i = r_i − mean(r_1..r_K)``
     (награда ответа минус среднее по K ответам группы);
   * токен-уровневая маска по log-ratio:
     ``1[|log π_θ − log π_old| ≤ τ_log]`` — клиппинг ограничивает off-policy
     дрейф строго через log-ratio **независимо от знака преимущества**, то
     есть работает градиентной маской, а не PPO-клиппингом (K2.5);
   * relative-entropy регуляризация к опорной политике (``π_ref`` — тот же
     чекпойнт SFT): ``β·KL_sampled(π_θ ‖ π_ref)`` — «variant of online policy
     mirror descent» (K1.5 :287); член per-token, как пер-токенная
     регуляризация K3 (:889);
   * оптимизатор — существующий ``net/optimizer.py`` (Per-Head Muon + weight
     clipping = MuonClip), cosine + warmup.
     Числа α, β, τ, K, λ источником не раскрыты (``docs/FIDELITY-TO-K3.md``:
     «свои числа») → взяты наши и записаны в журнал явно.

5. **Чекпойнт** после стадии (Orbax, ``tree_hash``): канонически на
   ``~/gb10-shared/checkpoints/rl-smoke``, в репозитории — симлинк (C-032);
   round-trip проверяется на месте.  Хеш *до* цикла фиксируется отдельно:
   видно, изменились ли веса, и это не выдаётся за обучение.
6. **Стоп-правило** (ADR-005 п. 6, спека §3.3): замер на holdout-сплите до и
   после цикла **тем же харнессом**; продолжение без положительной дельты
   прекращается.  Замер идёт при temperature = 0 (вердиктный режим §5.1) и
   оставляет Run Manifest на каждый прогон (§8).
7. **След стадии:** ``evidence/a4-run-wire/rl/stage-journal.json`` с полями
   спеки §3.4; пути **только относительные** от корня репозитория (ADR-014
   п. 8) либо в форме ``~/…`` для канонического диска.
8. **Стоимость** (AD-8/C-041): печать фактических GPU-часов и сопоставление
   со сметой ``evidence/budget/<run-id>.json``, если она есть; её отсутствие
   для смоука фиксируется как факт, а не замалчивается.

Контур награды — среда v1 без модельного судьи (AD-2): вердикт и награда
считаются из артефактов прогона.  Судейская модель (LLM), LLM-клиенты и внешние
API в путь награды не вводятся — это форсирует поведенческий страж C-039
(``tools/check_reward_isolation.py``), который обязан остаться зелёным.

Две численные страховки, найденные при проводке RL (обе записаны в журнал
полем ``findings``, обе — не «подгонка», а отказ от заведомо дефектного пути):

1. **Паддинг чанков KDA.** ``net/kda.py:apply_chunked`` дополняет
   последовательность нулями до кратности ``chunk_size``; на глубоких нулевых
   строках backward ``net/norm.py:l2_norm`` даёт NaN (0/0 в производной
   ``‖x‖``, при ``swish(x) → -0.0`` нулевая голова достижима и в forward).
   Значение loss при этом не меняется (чанки семантически эквивалентны —
   ``net/kda.py``, ``net/tests/test_02_kda_parity.py``), поэтому loss-путь
   считается с ``chunk = длина последовательности`` (pad = 0); SFT-стадия не
   сталкивалась с дефектом, потому что её длины кратны чанку (8192 % 64 == 0,
   128 % 64 == 0).  Дефект — в сети, а не в дельте: правка ``net/`` — отдельное
   решение (меняет паритет чекпойнта SFT).
2. **Страж нечислового градиента.** Шаг с нечисловым градиентом **не
   применяется**: NaN-веса в чекпойнт не пишутся, факт и листья фиксируются в
   журнале.  Молчаливое применение такого шага — брак артефакта, а не обучение.

Запуск (гейтовый профиль, GPU обязателен: чекпойнт SFT снят на GPU-топологии,
в ~/venv-axiom без CUDA-библиотек JAX падает на CPU — см. net/README.md):

    export LD_LIBRARY_PATH=$(ls -d ~/venv-axiom/lib/python3.11/site-packages/nvidia/*/lib | tr '\n' ':')
    NET_GATE_PROFILE=1 ~/venv-axiom/bin/python tools/run_rl_smoke.py \\
        --out evidence/a4-run-wire/rl --steps 10

Выход: ``0`` — стадия исполнена (цикл прокручен, чекпойнт и журнал записаны);
``1`` — стадия не исполнена (отказ входа, сбой цикла, ошибка) — журнал при этом
пишется честно, с фактическим статусом и причиной.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import shutil
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

CASE_DIR = Path(__file__).resolve().parent.parent
if str(CASE_DIR) not in sys.path:
    sys.path.insert(0, str(CASE_DIR))

#: Схема журнала стадии.
JOURNAL_SCHEMA = "rl-stage-journal/v1"

#: Имя стадии в эталонном наборе A4 (ADR-014).
STAGE = "rl_base_scheme"

#: Каталог следа стадии по умолчанию (относительно корня репозитория).
DEFAULT_OUT = "evidence/a4-run-wire/rl"

#: Вход: чекпойнт стадии SFT (симлинк; копия — отказ) и её журнал.
DEFAULT_SFT_CKPT = "evidence/a4-run-wire/sft/checkpoint"
DEFAULT_SFT_JOURNAL = "evidence/a4-run-wire/sft/stage-journal.json"

#: Каталог канонического хранения чекпойнта RL-стадии на сетевом диске.
#: Отдельный от ``checkpoints/sft-smoke``: перезапись чекпойнта предыдущей
#: стадии уничтожила бы её след.
CHECKPOINT_SUBDIR = "checkpoints/rl-smoke"

#: Веса сложности задачи (ADR-005 п. 2: награда «взвешенная по сложности»).
#: Источник числа не раскрывает → это **наше** число (``docs/FIDELITY-TO-K3.md``,
#: графа «свои числа»); записывается в журнал вместе с определением.
LEVEL_WEIGHTS: dict[str, float] = {"L0": 1.0, "L1": 1.25, "L2": 1.5, "L3": 2.0}

#: Замер «до/после» идёт по split'у, не участвующему в обновлении.
EVAL_LEVEL = "L1"
TRAIN_LEVEL = "L0"


_SFT_RUNNER_CACHE: list = []


def _load_sft_runner():
    """Загрузить ``tools/run_sft_smoke.py`` (модуль вне пакета).

    Переиспользуются **только** помощники предыдущей стадии: определение
    корня репозитория, приведение путей к относительным (ADR-014 п. 8),
    канонический диск ``~/gb10-shared``, единая точка пиннинга приёмки
    (``net/tests/conftest.py``, ADR-010/ADR-013), конфиг смоука, канонический
    токенизатор, смета (AD-8) и атомарная запись журнала стадии.  Дублировать
    их значило бы завести вторую точку пиннинга и второе правило путей.
    """
    if _SFT_RUNNER_CACHE:
        return _SFT_RUNNER_CACHE[0]
    spec = importlib.util.spec_from_file_location(
        "run_sft_smoke_for_rl", CASE_DIR / "tools" / "run_sft_smoke.py"
    )
    if spec is None or spec.loader is None:  # pragma: no cover — сломанная установка
        raise RuntimeError("не найден tools/run_sft_smoke.py (помощники стадии SFT)")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _SFT_RUNNER_CACHE.append(module)
    return module


class StageError(RuntimeError):
    """Стадия не исполнена: вход недопустим или инвариант нарушен."""


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# 1. Вход стадии: чекпойнт SFT (симлинк) и её журнал
# ---------------------------------------------------------------------------


def validate_checkpoint_symlink(
    path: Path, shared: Path, repo_root: Optional[Path]
) -> tuple[dict, Path]:
    """Чекпойнт — симлинк на канонический диск; копия или чужой путь — отказ.

    Семантика та же, что у данных SFT-стадии (C-032: веса канонически на
    ``~/gb10-shared``, в рабочем каталоге только симлинки), но формулировка —
    про чекпойнт: функция SFT-стадии говорит «набор/датасет», называть так
    веса было бы враньём в диагностике.

    Возвращает ``(описание для журнала, канонический путь)``: описание содержит
    только относительные пути (ADR-014 п. 8), а абсолютный путь нужен, чтобы
    загрузить чекпойнт, и в журнал не попадает.
    """
    sft = _load_sft_runner()
    path = Path(path)
    repo_root = repo_root if repo_root is not None else sft.detect_repo_root()
    if not path.exists():
        raise StageError(
            f"чекпойнт стадии SFT не найден: {sft.repo_rel(path, repo_root)}"
        )
    if not path.is_symlink():
        raise StageError(
            "путь к чекпойнту не симлинк, а копия в рабочем каталоге: копии "
            "весов запрещены (C-032), чекпойнт монтируется симлинком на "
            "~/gb10-shared"
        )
    shared = Path(shared)
    try:
        target = Path(os.readlink(path))
    except OSError as exc:  # pragma: no cover — гонка с удалением
        raise StageError(f"симлинк чекпойнта не читается: {exc}") from exc
    resolved = target if target.is_absolute() else (path.parent / target)
    try:
        resolved = resolved.resolve()
        shared_resolved = shared.resolve()
    except OSError as exc:
        raise StageError(f"симлинк чекпойнта не разрешается: {exc}") from exc
    if not resolved.exists():
        raise StageError(
            "симлинк чекпойнта ведёт на несуществующий путь: "
            f"{sft.repo_rel(resolved, repo_root)}"
        )
    if not resolved.is_dir():
        raise StageError(
            "чекпойнт стадии SFT — не каталог (Orbax пишет каталог): "
            f"{sft.repo_rel(resolved, repo_root)}"
        )
    if shared_resolved not in resolved.parents and resolved != shared_resolved:
        raise StageError(
            "симлинк чекпойнта ведёт вне канонического диска ~/gb10-shared: "
            f"{sft.repo_rel(resolved, repo_root)} (C-032)"
        )
    described = {
        "path": sft.repo_rel_link(path, repo_root),
        "is_symlink": True,
        "canonical": sft.repo_rel(resolved, repo_root),
        "shared_root": sft.repo_rel(shared_resolved, repo_root),
        "format": "orbax",
    }
    return described, resolved


def read_sft_journal(path: Path, repo_root: Optional[Path]) -> dict:
    """Журнал SFT как вход RL-стадии: признак ``qat_weights`` и конфиг сети.

    Отсутствие *поля* ``qat_weights`` трактуется как ``off`` (SFT-дельта,
    решение по вопросу 5; спека §3.1).  Отсутствие самого *журнала* — отказ:
    без него неизвестны ни признак QAT, ни структура чекпойнта, и стадия
    «догадалась бы» о входе вместо того, чтобы его проверить.
    """
    sft = _load_sft_runner()
    path = Path(path)
    if not path.is_file():
        raise StageError(
            "журнал стадии SFT не найден: "
            f"{sft.repo_rel(path, repo_root)} — вход RL-стадии не определён "
            "(признак qat_weights и конфиг чекпойнта берутся из него)"
        )
    try:
        journal = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise StageError(f"журнал SFT не читается: {exc}") from exc
    if not isinstance(journal, dict):
        raise StageError("журнал SFT не объект JSON")
    if journal.get("stage") not in (None, "sft"):
        raise StageError(
            f"журнал {sft.repo_rel(path, repo_root)} — не журнал стадии SFT "
            f"(stage={journal.get('stage')!r}): вход RL-стадии обязан быть следом SFT"
        )

    raw = journal.get("qat_weights")
    if raw == "on":
        qat, source = "on", "sft-stage-journal:qat_weights"
    elif raw == "off":
        qat, source = "off", "sft-stage-journal:qat_weights"
    elif raw is None:
        qat, source = "off", "поле qat_weights отсутствует → off (SFT-дельта, вопрос 5)"
    else:
        qat, source = "off", (
            f"значение qat_weights={raw!r} не распознано → off "
            "(«включено» не додумывается)"
        )

    model_config = journal.get("model_config") or {}
    checkpoint = journal.get("checkpoint") or {}
    return {
        "path": sft.repo_rel(path, repo_root),
        "schema": journal.get("schema"),
        "stage": journal.get("stage"),
        "status": journal.get("status"),
        "qat_weights": qat,
        "qat_weights_source": source,
        "model_preset": model_config.get("preset"),
        "model_vocab_size_from_journal": model_config.get("vocab_size"),
        "checkpoint_tree_hash_from_journal": checkpoint.get("tree_hash"),
    }


# ---------------------------------------------------------------------------
# 2. Конфигурация и веса: преемственность структуры чекпойнта
# ---------------------------------------------------------------------------


def model_vocab_size(tokenizer) -> int:
    """Модельный vocab смоука: покрывает все испускаемые id токенизатора.

    Та же арифметика, что у SFT-стадии: чекпойнт SFT снят с этим vocab, и
    другая величина сделала бы загрузку невозможной (и это было бы скрыто
    до первого forward'а).
    """
    max_id = max(tokenizer._merge_id.values()) if tokenizer._merge_id else 0
    max_emitted = max(3 + 256 - 1, max_id)
    return 1 << max(10, int(max_emitted).bit_length())


def load_policy(cfg, ckpt_canonical: Path, seed: int, expected_tree_hash: Optional[str]):
    """Параметры политики из чекпойнта SFT + честное сравнение ``tree_hash``.

    Несовпадение хеша не «исправляется» и не игнорируется: оно попадает в
    журнал флагом ``matches_journal_hash``.  Значения арифметики здесь не
    пересчитываются (значения читаются из чекпойнта), поэтому расхождение
    означало бы, что загружен не тот чекпойнт.
    """
    import jax.random as jr

    from net import checkpoint as checkpoint_mod
    from net import model as model_mod

    structure = model_mod.init_params(jr.PRNGKey(seed), cfg)
    params = checkpoint_mod.load_checkpoint(str(ckpt_canonical), target=structure)
    digest = checkpoint_mod.tree_hash(params)
    return params, digest, (expected_tree_hash is None or digest == expected_tree_hash)


# ---------------------------------------------------------------------------
# 3. Схема базы: преимущество, маска по log-ratio, KL-регуляризация
# ---------------------------------------------------------------------------


STOP_RULE_TEXT = (
    "ADR-005 п. 6: продолжение обучения без положительной дельты на holdout "
    "прекращается"
)


def stop_rule(before: Optional[float], after: Optional[float]) -> dict:
    """Стоп-правило ADR-005 п. 6 по замеру «до/после» (чистая функция).

    Отсутствие замера — не «ноль» и не «продолжаем»: решение помечается
    ``not-evaluated``, иначе смоук без holdout выглядел бы как разрешение
    продолжать (спека §6: честная пометка непроведённого замера).
    """
    if before is None or after is None:
        return {
            "evaluated": False,
            "decision": "not-evaluated",
            "reason": "holdout-сплит недоступен: замер «до/после» не проведён "
                      "(спека §6: стадия честно помечает замер как непроведённый)",
            "rule": STOP_RULE_TEXT,
        }
    delta = after - before
    return {
        "evaluated": True,
        "holdout_before": before,
        "holdout_after": after,
        "delta": round(delta, 6),
        "positive": bool(delta > 0),
        "decision": "continue" if delta > 0 else "stop",
        "rule": STOP_RULE_TEXT,
        "reason": (
            "дельта положительна — продолжение RL допустимо"
            if delta > 0 else
            "дельта не положительна: полный RL не запускается (смоук остаётся "
            "смоуком — он и не заявлял обучения)"
        ),
    }


def level_weight(level: str) -> float:
    """Вес сложности задачи (наше число, ADR-005 п. 2 «взвешенный по сложности»)."""
    return LEVEL_WEIGHTS.get(level, 1.0)


def group_advantages(values: list[float]) -> list[float]:
    """Групповое преимущество без критика: ``a_i = r_i − mean(r)``.

    Нормализация на std намеренно не добавлена: ADR-005 п. 2 описывает схему
    базы как «награда ответа минус среднее по K ответам группы»; деление на
    std — другой алгоритм, и подменять его молча нельзя.
    """
    if not values:
        return []
    mean = sum(values) / len(values)
    return [value - mean for value in values]


def safe_chunk(seq_len: int) -> int:
    """Чанк KDA, при котором pad = 0: длина последовательности.

    ``apply_chunked`` дополняет вход нулями до кратности чанку, а backward
    ``l2_norm`` на строго нулевом векторе даёт NaN (0/0 в производной ``‖x‖``);
    NaN-строки не влияют на значение loss, но отравляют градиент KDA-параметров
    (проверено: тот же loss 12.4911 при pad = 57 и при pad = 0, градиент —
    нечисловой только в первом случае).  Чанк, равный длине, убирает паддинг и
    **не меняет семантику**: обе формы KDA эквивалентны (``net/kda.py``,
    ``net/tests/test_02_kda_parity.py``).
    """
    return max(int(seq_len), 1)


def response_logprobs(params, cfg, prompt_ids: list[int], gen_ids: list[int]):
    """Лог-вероятности токенов ответа под политикой ``params``.

    Токен-уровневость здесь не украшение: и маска по log-ratio, и KL-член
    считаются по токенам **ответа** (K2.5 :421–433; K3 :889 — пер-токенная
    регуляризация).  Промпт из суммы исключается — иначе маска мерила бы
    дрейф на токенах, которые политика не выбирала.

    ``chunk_size`` поднимается до длины последовательности (``safe_chunk``):
    pad = 0 — иначе backward уходит в NaN на заведомо нулевых строках паддинга.
    """
    import jax
    import jax.numpy as jnp

    from net import model as model_mod

    if not gen_ids:
        return jnp.zeros((0,))
    ids = jnp.asarray([list(prompt_ids) + list(gen_ids)], dtype=jnp.int32)
    logits = model_mod.forward(
        params, cfg, ids, chunk_size=safe_chunk(ids.shape[1])
    )
    logp = jax.nn.log_softmax(logits[:, :-1], axis=-1)      # (1, T-1, V)
    targets = ids[:, 1:]                                     # (1, T-1)
    picked = jnp.take_along_axis(logp, targets[..., None], axis=-1)[0, :, 0]
    return picked[-len(gen_ids):]                            # (n_gen,)


def make_rollout_objective(cfg, record: dict, advantage: float, total_tokens: int, *,
                           tau_log: float, beta: float, qat: bool, reference):
    """Jit-цель одного роллаута: терм схемы базы + диагностика, одним графом.

    .. code-block:: text

        L_i = [ −a_i · Σ_t 1[|log π_θ(y_t) − log π_old(y_t)| ≤ τ_log] · log π_θ(y_t)
                + β · Σ_t (e^{d_t} − d_t − 1),   d_t = log π_ref(y_t) − log π_θ(y_t)
              ] / N_tokens

    ``log π_old`` (политика, породившая роллаут) и ``log π_ref`` (опорная
    политика — чекпойнт SFT) считаются **внутри** графа под stop-gradient:
    градиент идёт только через текущую политику, а маска меряет дрейф к
    политике-источнику, поэтому ограничивает off-policy дрейф внутренних эпох.

    Регуляризатор — несмещённая (k3) оценка ``KL(π_θ ‖ π_ref)`` по токенам
    ответа: неотрицательна и **плоская в точке π_θ = π_ref** (и значение, и
    градиент равны нулю).  Голая разность ``log π_θ − log π_ref`` этим свойством
    не обладает: её градиент в опорной точке равен ``∇ log π_θ``, то есть
    работает как дообучение на собственных сэмплах, а не как относительная
    энтропия — измерено на первом прогоне (ненулевой градиент при нулевом
    преимуществе), поэтому оценка заменена.

    Jit обязателен: без него шаг на маленьком конфиге — десятки секунд (eager
    диспетчеризация сотен мелких ядер), с ним — миллисекунды.  Ключ кэша jit —
    форма (длина последовательности), поэтому смоук-масштаб выбирается так,
    чтобы различных длин было немного: компиляция идёт один раз на длину.

    Возвращает ``(loss, (mask_share, masked_tokens))`` — диагностика едет тем же
    графом, отдельного прохода за ней не нужно.
    """
    import jax
    import jax.numpy as jnp

    from net import quant as quant_mod

    prompt_ids, gen_ids = record["prompt_ids"], record["gen_ids"]
    is_zero_term = advantage == 0.0 and beta == 0.0

    def objective(params, old_params):
        if is_zero_term or not gen_ids:
            # Вклад тождественно нулевой: 0·log π считать нельзя — log π бывает
            # -inf, и 0·(-inf) = NaN (вырожденная группа дала бы NaN-градиент
            # вместо честного нуля).
            return jnp.zeros(()), (jnp.ones(()), jnp.zeros(()))
        policy = quant_mod.apply_fake_quant_tree(params) if qat else params
        source = quant_mod.apply_fake_quant_tree(old_params) if qat else old_params
        ref = quant_mod.apply_fake_quant_tree(reference) if qat else reference
        logp = response_logprobs(policy, cfg, prompt_ids, gen_ids)
        if logp.shape[0] == 0:
            return jnp.zeros(()), (jnp.ones(()), jnp.zeros(()))
        logp_old = jax.lax.stop_gradient(response_logprobs(source, cfg, prompt_ids, gen_ids))
        logp_ref = jax.lax.stop_gradient(response_logprobs(ref, cfg, prompt_ids, gen_ids))
        drift = logp - logp_old
        mask = (jnp.abs(drift) <= tau_log).astype(logp.dtype)
        policy_term = -advantage * jnp.sum(mask * logp)
        # k3-оценка относительной энтропии: неотрицательна и плоская при π_θ = π_ref
        log_ratio_ref = jax.lax.stop_gradient(logp_ref) - logp
        regularizer = beta * jnp.sum(jnp.exp(log_ratio_ref) - log_ratio_ref - 1.0)
        loss = (policy_term + regularizer) / total_tokens
        return loss, (jnp.mean(mask), jnp.sum(mask))

    return jax.jit(jax.value_and_grad(objective, has_aux=True))


def add_grads(total, grads):
    """Накопление градиентов группы (линейность: сумма = градиент суммы)."""
    import jax

    if total is None:
        return grads
    return jax.tree_util.tree_map(lambda a, b: a + b, total, grads)


def gradient_norm(grads) -> float:
    """Глобальная L2-норма градиента по дереву параметров (диагностика)."""
    import jax
    import jax.numpy as jnp

    total = 0.0
    for leaf in jax.tree_util.tree_leaves(grads):
        total += float(jnp.sum(jnp.square(leaf)))
    return math.sqrt(total)


def _leaf_path(path) -> str:
    """Путь листа в дереве параметров как читаемая строка (``layers[0].attn.W_q``)."""
    parts: list[str] = []
    for entry in path:
        name = getattr(entry, "name", None)
        index = getattr(entry, "idx", None)
        if name is not None:
            parts.append(str(name))
        elif index is not None:
            parts.append(f"[{index}]")
        else:
            parts.append(str(getattr(entry, "key", "?")))
    return ".".join(parts).replace(".[", "[")


def gradient_diagnostics(grads) -> dict:
    """Числовая пригодность градиента: сколько листьев нечисловы и какие.

    Страж перед шагом оптимизатора: применить шаг с NaN — значит записать
    NaN-веса в чекпойнт стадии (артефакт станет браком, а не свидетельством).
    """
    import jax
    import jax.numpy as jnp

    bad: list[str] = []
    total = 0
    for path, leaf in jax.tree_util.tree_flatten_with_path(grads)[0]:
        total += 1
        if not bool(jnp.all(jnp.isfinite(leaf))):
            bad.append(_leaf_path(path) or "<корень>")
    return {
        "finite": not bad,
        "bad_leaves": len(bad),
        "total_leaves": total,
        "bad_paths": bad,
    }


def _json_float(value) -> Optional[float]:
    """Число для журнала: NaN/Inf → None (в JSON нет NaN — артефакт обязан читаться)."""
    number = float(value)
    return round(number, 8) if math.isfinite(number) else None


# ---------------------------------------------------------------------------
# 4. Среда: задачи, роллаут, механический вердикт, награда
# ---------------------------------------------------------------------------


def generate_tasks(seed: int, train_per_source: int, eval_tasks: int, out_dir: Path) -> dict:
    """Пиннутая генерация задач среды (``env/generate.py``) + размеченный split.

    Обновление идёт по задачам уровня L0 (H = 0: скрытые правила в них не
    участвуют), замер «до/после» — по задачам уровня L1 (H > 0): скрытый
    набор применяется **только на вердикте** замерных задач (ENVIRONMENT-V1
    §6) и не попадает в обучение.  Это и есть «holdout, не использованный ни
    в SFT, ни в RL» в объёме смоука; определение сплита пишется в журнал.

    Доля general-purpose задач (ADR-005 п. 4) фиксируется как доля задач
    ``source=real`` (живая задача на чистом кейсе, keep-gates-implement) —
    единственное механическое разделение, которое даёт ``env/generate.py``;
    определение записывается в журнал, а не подразумевается.
    """
    from env import generate as generate_mod

    grid = (
        ("corruption", TRAIN_LEVEL, train_per_source),
        ("real", TRAIN_LEVEL, train_per_source),
        ("corruption", EVAL_LEVEL, eval_tasks),
    )
    tasks_dir = out_dir / "env" / "tasks"
    summary = generate_mod.generate(CASE_DIR, tasks_dir, seed=seed, grid=grid, holdout_count=0)
    public = sorted((tasks_dir / "public").glob("*.json"))
    specs: dict[str, dict] = {}
    for path in public:
        spec = json.loads(path.read_text(encoding="utf-8"))
        specs[spec["id"]] = spec
    train_ids = [
        item["id"] for item in summary["public"]
        if item["level"] == TRAIN_LEVEL
    ]
    eval_ids = [
        item["id"] for item in summary["public"]
        if item["level"] == EVAL_LEVEL
    ]
    return {
        "tasks_dir": tasks_dir,
        "specs": specs,
        "train_ids": train_ids,
        "eval_ids": eval_ids,
        "hidden_constraints": tasks_dir / "holdout" / "hidden_constraints.yaml",
        "hidden_constraints_sha256": summary["hidden_constraints_sha256"],
        "public_rules": summary["public_rules"],
        "generated": summary,
    }


def hidden_for(spec: dict, env_tasks: dict):
    """Скрытый набор правил применяется только там, где задача его объявляет."""
    from env.util import EMPTY_HIDDEN_SHA256

    declared = spec["verifier"]["hidden_constraints_sha256"]
    if declared == EMPTY_HIDDEN_SHA256:
        return None
    path = env_tasks["hidden_constraints"]
    return path if path.exists() else None


def base_violations_at(ws_dir: Path, spec: dict, bin_path: str, hidden) -> frozenset:
    """Error-сигнатуры ``(rule, file)`` базового состояния **по пути роллаута**.

    Тонкость, найденная при проводке RL и записанная в журнал: у поведенческих
    правил CONSTRAINTS (C-038…C-043) сигнатура нарушения включает путь
    workspace'а, поэтому базовое и финальное состояния обязаны измеряться
    **по одному и тому же пути** — иначе разница путей сама породит «новые»
    нарушения и награда перестанет быть функцией состояния агента (AD-11).
    """
    from env import verifier as verifier_mod

    return verifier_mod.collect_violations(ws_dir, spec, bin=bin_path, hidden_constraints=hidden)


def execute_rollout(spec: dict, base_ws: Path, ws_dir: Path, *, policy_params, cfg, tokenizer,
                    decoding: dict, max_new_tokens: int, bin_path: str, hidden,
                    base_violations: frozenset, weight: float) -> dict:
    """Один роллаут: исполнение политикой → механический вердикт → награда.

    Используются существующие функции среды: ``env/net_executor.run_net_model``
    (копия снапшота кейса → генерация → ответ в workspace) и
    ``env/verifier.verify`` + ``env/reward.compute`` (вердикт и награда §5).
    Run Manifest на каждый роллаут не пишется: на смоук-масштабе след стадии —
    журнал (спека §3.4), а вердиктные замеры «до/после» манифест получают
    (они — вердиктные прогоны §5.1, а не RL-роллауты).
    """
    from env import net_executor
    from env import reward as reward_mod
    from env import verifier as verifier_mod

    net_run = net_executor.run_net_model(
        spec, base_ws, ws_dir,
        params=policy_params, cfg=cfg, tokenizer=tokenizer,
        decoding=decoding, max_new_tokens=max_new_tokens,
    )
    verdict = verifier_mod.verify(
        spec, ws_dir, bin=bin_path, hidden_constraints=hidden, run_task_tests=True
    )
    kind = spec.get("objective", {}).get("kind", "restore-gates")
    reward = reward_mod.compute(
        objective_kind=kind,
        passed=verdict.passed,
        tests_passed=verdict.tests_passed,
        base_violations=base_violations,
        final_violations=verdict.violations,
        spent_tokens=net_run.spent_tokens,
        max_tokens=int(spec.get("max_tokens", 0)),
    )
    return {
        "net_run": net_run,
        "verdict": verdict,
        "reward": reward,
        "reward_weighted": weight * reward.total,
        "prompt_ids": list(tokenizer.encode(str(spec.get("prompt", "")))),
    }


def write_attempt_manifest(path: Path, *, spec: dict, ws_dir: Path, net_run, verdict,
                           reward, cfg, decoding: dict, bin_path: str, run_id: str,
                           started: str, finished: str, repo_root: Optional[Path]) -> dict:
    """Run Manifest §8 для вердиктного прогона (temperature = 0)."""
    from env import manifest as manifest_mod
    from env.net_executor import manifest_mod_sha256, model_snapshot_sha256
    from env.util import tree_sha256, write_json
    from env.verifier import arch_ml_build_hash

    manifest = manifest_mod.build_manifest(
        run_id=run_id,
        task_spec=spec,
        arch_ml_build=arch_ml_build_hash(bin_path),
        constraints_sha256=manifest_mod_sha256(ws_dir, spec),
        hidden_constraints_sha256=spec["verifier"]["hidden_constraints_sha256"],
        workspace_sha256=tree_sha256(ws_dir),
        model_base="net-l3-skeleton-rl-smoke",
        model_snapshot_sha256=model_snapshot_sha256(cfg),
        decoding=decoding,
        attempts_used=1,
        usage={
            "tokens_in": net_run.tokens_in,
            "tokens_out": net_run.tokens_out,
            "cost_usd": 0.0,
            "host": "net-executor",
        },
        timing={"started": started, "finished": finished, "resume_count": 0},
        verdict={
            "pass": verdict.passed,
            "reward": reward.to_manifest_dict(),
            "issues_warn": verdict.warn_issues,
        },
    )
    manifest["model"]["routing_seed"] = int(cfg.routing_seed)
    for message in manifest_mod.full_validate(manifest):
        raise StageError(f"Run Manifest вердиктного прогона не валиден: {message}")
    write_json(path, manifest)
    sft = _load_sft_runner()
    return {"path": sft.repo_rel(path, repo_root), "run_id": run_id}


# ---------------------------------------------------------------------------
# 5. Чекпойнт стадии
# ---------------------------------------------------------------------------


def save_stage_checkpoint(params, ckpt_dir: Path, shared: Path, tree_digest: str):
    """Orbax-чекпойнт RL-стадии: канонически на диске, в репозитории симлинк.

    Логика та же, что у SFT-стадии (``tools/run_sft_smoke.py``), но **свой**
    канонический подкаталог: переиспользовать функцию SFT нельзя — она пишет
    в ``checkpoints/sft-smoke`` и затёрла бы чекпойнт предыдущей стадии, то
    есть уничтожила бы её след (AD-4).
    """
    from net import checkpoint as checkpoint_mod

    sft = _load_sft_runner()
    shared = Path(shared)
    canonical = shared / CHECKPOINT_SUBDIR
    canonical.mkdir(parents=True, exist_ok=True)
    ckpt_dir = Path(ckpt_dir)
    if ckpt_dir.is_symlink():
        pass  # повторный прогон: симлинк уже смонтирован
    elif ckpt_dir.exists():
        raise StageError(
            f"каталог чекпойнта {sft.repo_rel(ckpt_dir, sft.detect_repo_root())} — "
            "не симлинк: копия весов в рабочем каталоге запрещена (C-032)"
        )
    else:
        ckpt_dir.parent.mkdir(parents=True, exist_ok=True)
        ckpt_dir.symlink_to(canonical)

    digest = checkpoint_mod.save_checkpoint(params, canonical)
    if digest != tree_digest:
        raise StageError(f"tree_hash до сохранения {tree_digest} != после {digest}")
    restored = checkpoint_mod.load_checkpoint(canonical, target=params)
    if checkpoint_mod.tree_hash(restored) != digest:
        raise StageError("round-trip чекпойнта не совпал по tree_hash")
    return {
        "path": sft.repo_rel_link(ckpt_dir, sft.detect_repo_root()),
        "canonical": sft.repo_rel(canonical, sft.detect_repo_root()),
        "symlink": True,
        "format": "orbax",
        "tree_hash": digest,
        "roundtrip_ok": True,
    }


# ---------------------------------------------------------------------------
# Схема базы: описание для журнала
# ---------------------------------------------------------------------------


def scheme_block(args) -> dict[str, Any]:
    """Схема обновления в журнал: класс, механика, числа и их происхождение."""
    return {
        "family": "log-ratio / mirror-descent class (ADR-005 п. 2)",
        "basis": [
            "ADR-005 п. 2 (решение владельца 2026-09-11: воспроизводим схему базы, "
            "а не наш GRPO)",
            "K2.5 :421–433 — групповое преимущество + токен-уровневый клиппинг по "
            "log-ratio в роли маски градиента (docs/research/k3-tech-report-extract.md)",
            "K1.5 :287 — «variant of online policy mirror descent», relative-entropy "
            "regularized policy optimization",
            "K3 :889 — пер-токенная регуляризация против устаревших данных",
        ],
        "advantage": "group mean-centered: a_i = r_i − mean(r_1..r_K), без критика",
        "token_mask": (
            f"1[|log π_θ − log π_old| ≤ τ_log], τ_log={args.tau_log}: ограничивает "
            "off-policy дрейф по log-ratio независимо от знака преимущества "
            "(маска градиента, не PPO-клиппинг)"
        ),
        "regularizer": (
            f"β·KL_sampled(π_θ ‖ π_ref), β={args.beta}, π_ref = чекпойнт SFT "
            "(relative-entropy регуляризация, K1.5)"
        ),
        "optimizer": (
            "net/optimizer.py: Per-Head Muon + weight clipping (MuonClip), "
            f"cosine + warmup {args.warmup_ratio}"
        ),
        "hyperparameters": {
            "group_size_K": args.group_size,
            "tau_log": args.tau_log,
            "beta": args.beta,
            "lr": args.lr,
            "inner_epochs": args.inner_epochs,
            "warmup_ratio": args.warmup_ratio,
        },
        "numbers_source": (
            "α, β, τ, K, λ источником не раскрыты (docs/FIDELITY-TO-K3.md, графа "
            "«свои числа») → числа наши; схема — базы"
        ),
        "forbidden": "судейская модель и внешние API в контуре награды (AD-2/C-039)",
    }


# ---------------------------------------------------------------------------
# Прогон стадии
# ---------------------------------------------------------------------------


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Стадия rl_base_scheme конвейера A4: смоук-цикл RL + журнал стадии",
    )
    parser.add_argument("--sft-ckpt", default=DEFAULT_SFT_CKPT,
                        help="чекпойнт стадии SFT; обязан быть симлинком на ~/gb10-shared")
    parser.add_argument("--sft-journal", default=DEFAULT_SFT_JOURNAL,
                        help="журнал стадии SFT: источник признака qat_weights и конфига")
    parser.add_argument("--out", default=DEFAULT_OUT,
                        help="каталог следа стадии (внутри репозитория)")
    parser.add_argument("--steps", type=int, default=10,
                        help="число шагов обновления (смоук: <= 100, спека §3.2)")
    parser.add_argument("--train-tasks", type=int, default=2,
                        help="задач уровня L0 на каждый источник {corruption, real}")
    parser.add_argument("--eval-tasks", type=int, default=2,
                        help="задач уровня L1 для замера «до/после» (вне обновления)")
    parser.add_argument("--group-size", type=int, default=4, help="K роллаутов на задачу")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="temperature RL-роллаутов (ENVIRONMENT-V1 §5.1: > 0)")
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--inner-epochs", type=int, default=2,
                        help="внутренних эпох на шаг: вторая эпоха идёт по устаревшим "
                             "роллаутам — здесь маска по log-ratio и работает")
    parser.add_argument("--tau-log", type=float, default=0.2,
                        help="порог маски по log-ratio (наше число)")
    parser.add_argument("--beta", type=float, default=1e-3,
                        help="вес KL к опорной политике (наше число; 0 — выключить)")
    parser.add_argument("--lr", type=float, default=1e-3, help="пиковый LR (cosine + warmup)")
    parser.add_argument("--warmup-ratio", type=float, default=0.01)
    parser.add_argument("--model-preset", choices=("small", "tiny"), default=None,
                        help="пресет скелета L3 (по умолчанию — из журнала SFT)")
    parser.add_argument("--seed", type=int, default=1337, help="пиннутый сид (AD-11)")
    parser.add_argument("--ckpt-dir", default=None,
                        help="симлинк каталога чекпойнта (по умолчанию <out>/checkpoint)")
    parser.add_argument("--budget-limit-usd", type=float, default=None,
                        help="явный лимит смоука, если сметы evidence/budget/<run-id>.json нет")
    parser.add_argument("--json", action="store_true", help="печать журнала в stdout")
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.steps > 100:
        parser.error("смоук-режим: --steps <= 100 (спека §3.2); полный прогон — "
                     "отдельное решение со сметой (AD-8/C-041)")
    return args


def _rollout_seed(seed: int, task_seed: int, step: int, rollout: int) -> int:
    """Детерминированный сид роллаута: (глобальный сид, задача, шаг, индекс).

    Детерминизм при пиннутом сиде — требование спеки §4.1: тот же сид обязан
    дать тот же набор задач и ту же первую награду.
    """
    return (seed * 1_000_003 + task_seed * 1_009 + step * 31 + rollout) % (2**31 - 1)


def run_stage(args: argparse.Namespace) -> tuple[dict[str, Any], bool]:
    """Исполнить стадию; вернуть (журнал, успех).  Журнал пишется всегда."""
    sft = _load_sft_runner()
    started = time.time()
    started_wall = datetime.now(timezone.utc)
    run_date = started_wall.date().isoformat()
    run_id = f"rl-smoke-{run_date}"
    repo_root = sft.detect_repo_root()
    out_dir = sft.case_path(args.out)
    ckpt_dir = sft.case_path(args.ckpt_dir) if args.ckpt_dir else out_dir / "checkpoint"
    shared = sft.shared_root()

    conftest = sft._load_acceptance_conftest()   # пиннинг бэкенда до импорта net.*
    journal: dict[str, Any] = {
        "schema": JOURNAL_SCHEMA,
        "stage": STAGE,
        "status": "absent",
        "scale": "smoke",
        "run_ref": sft.repo_rel(out_dir / "stage-journal.json", repo_root),
        "run_id": run_id,
        "started_at": started_wall.isoformat(),
        "seed": args.seed,
        "steps": args.steps,
        "backend": sft._backend_block(conftest),
        "scheme": scheme_block(args),
        "notes": [
            "Смоук-масштаб: доказательство провода стадии (задачи → роллауты → "
            "механический вердикт → награда → обновление → чекпойнт → журнал), "
            "не обучение модели (спека §5).",
            "Контур награды — среда v1 как есть: вердикт env/verifier.py, награда "
            "env/reward.py; судейская модель и внешние API в контур не вводятся "
            "(AD-2), страж C-039 обязан остаться зелёным.",
            "Плотная компонента награды — существующая soft-доля env/reward.py "
            "(спека §2.2, ADR-005 п. 2): бинарный гейт на скелете даёт разреженный "
            "сигнал, soft остаётся механической долей гейтов, а не модельной оценкой.",
        ],
    }

    def fail(status: str, message: str) -> tuple[dict[str, Any], bool]:
        journal["status"] = status
        journal["error"] = _sanitize(message, repo_root)
        journal["wall_clock_s"] = round(time.time() - started, 3)
        journal["finished_at"] = datetime.now(timezone.utc).isoformat()
        sft._write_journal(journal, out_dir)
        print(f"[rl] {status}: {journal['error']}", file=sys.stderr, flush=True)
        return journal, False

    work_root: Optional[Path] = None
    try:
        # --- 1. вход: чекпойнт SFT (симлинк) и признак qat_weights ----------
        ckpt_link = sft.case_path(args.sft_ckpt)
        sft_journal_path = sft.case_path(args.sft_journal)
        print(f"[rl] вход: чекпойнт {sft.repo_rel_link(ckpt_link, repo_root)}", flush=True)
        described, ckpt_resolved = validate_checkpoint_symlink(ckpt_link, shared, repo_root)
        sft_journal = read_sft_journal(sft_journal_path, repo_root)
        journal["input"] = {"checkpoint": described, "sft_journal": sft_journal}
        journal["qat_weights"] = sft_journal["qat_weights"]
        qat = sft_journal["qat_weights"] == "on"
        print(
            f"[rl] симлинк ок -> {described['canonical']}; "
            f"qat_weights={journal['qat_weights']} ({sft_journal['qat_weights_source']})",
            flush=True,
        )

        # --- 2. конфигурация и веса ----------------------------------------
        tokenizer, tokenizer_hash = sft.canonical_tokenizer()
        computed_vocab = model_vocab_size(tokenizer)
        from_journal = sft_journal["model_vocab_size_from_journal"]
        if from_journal is not None and int(from_journal) != computed_vocab:
            return fail(
                "failed",
                f"vocab чекпойнта SFT ({from_journal}) не совпал с посчитанным "
                f"({computed_vocab}): чекпойнт не восстановится в ту же структуру",
            )
        preset = args.model_preset or sft_journal["model_preset"] or "small"
        cfg = sft.build_model_config(computed_vocab, preset, qat)
        journal["tokenizer"] = {
            "hash": tokenizer_hash,
            "source": "net/tokenizer.py canonical (net/config.json:tokenizer_hash)",
            "model_vocab_size": computed_vocab,
            "preset": preset,
        }
        journal["model_config"] = {
            "preset": preset,
            "vocab_size": cfg.vocab_size,
            "qat_enabled": bool(cfg.qat_enabled),
            "hidden": cfg.hidden,
            "num_layers": cfg.num_layers,
            "routing_seed": int(cfg.routing_seed),
            "dtype": "bf16",
            "note": "структура — из журнала SFT: чекпойнт обязан грузиться в неё",
        }
        policy, tree_hash_loaded, matches = load_policy(
            cfg, ckpt_resolved, args.seed,
            sft_journal["checkpoint_tree_hash_from_journal"],
        )
        reference = policy  # опорная политика π_ref схемы базы (K1.5): чекпойнт SFT
        journal["checkpoint_before_tree_hash"] = tree_hash_loaded
        journal["parameter_norm_before"] = round(_param_norm(policy), 6)
        journal["input"]["checkpoint"]["matches_journal_hash"] = bool(matches)
        if not matches:
            journal["notes"].append(
                "ВНИМАНИЕ: tree_hash загруженного чекпойнта не совпал с хешем в "
                "журнале SFT — преемственность стадии не подтверждена (факт "
                "зафиксирован, а не замаскирован)"
            )
        print(f"[rl] политика: tree_hash={tree_hash_loaded[:16]}… (из чекпойнта SFT)",
              flush=True)

        # --- 3. среда: генерация задач и сплит -----------------------------
        env_tasks = generate_tasks(
            args.seed, args.train_tasks, args.eval_tasks, out_dir
        )
        train_ids, eval_ids = env_tasks["train_ids"], env_tasks["eval_ids"]
        if not train_ids:
            return fail("absent", "сплит обновления пуст: генерация не дала задач L0")
        journal["environment"] = {
            "version": "environment-v1",
            "tasks_dir": sft.repo_rel(env_tasks["tasks_dir"], repo_root),
            "hidden_constraints": _optional_rel(
                env_tasks["hidden_constraints"], repo_root, sft
            ),
            "hidden_constraints_sha256": env_tasks["hidden_constraints_sha256"],
            "public_rules": env_tasks["public_rules"],
            "train_tasks": train_ids,
            "eval_tasks": eval_ids,
            "train_level": TRAIN_LEVEL,
            "eval_level": EVAL_LEVEL,
            "holdout_definition": (
                "замер «до/после» идёт по задачам уровня L1, не участвующим в "
                "обновлении; скрытый набор правил применяется только на вердикте "
                "этих задач (ENVIRONMENT-V1 §6). Это НЕ holdout-пул NFR-001 "
                "(≥50 задач с невыпущенными кейсами) — в смоуке он недоступен, "
                "и стадия называет замер по имени, а не выдаёт его за полный"
            ),
        }
        journal["mix_share"] = (
            round(sum(1 for t in train_ids if t.startswith("real")) / len(train_ids), 6)
        )
        journal["mix_definition"] = (
            "general-purpose = задачи source=real (keep-gates-implement: живая "
            "задача на чистом кейсе); domain = source=corruption (restore-gates). "
            "Определение наше: env/generate.py различает ровно эти два источника, "
            "а ADR-005 п. 4 требует долю — она фиксируется здесь и не "
            "подразумевается"
        )
        print(
            f"[rl] задачи: обновление {train_ids} (L0, H=0), замер {eval_ids} (L1); "
            f"mix_share={journal['mix_share']}",
            flush=True,
        )

        # --- 4. замер «до» -------------------------------------------------
        from env import verifier as verifier_mod

        bin_path = verifier_mod.arch_ml_bin()
        work_root = Path(tempfile.mkdtemp(prefix="rl-smoke-rollouts-"))
        manifests_dir = out_dir / "run-manifests"
        comparison = {
            "harness": "env/net_executor.py + env/verifier.py + env/reward.py (тот же, что у baseline)",
            "decoding": {"temperature": 0.0, "top_p": args.top_p,
                         "max_new_tokens": args.max_new_tokens},
            "timing_note": (
                "timing вердиктных манифестов фиксирован датой прогона ради "
                "воспроизводимости (A5: повтор даёт тот же манифест); фактическое "
                "время — в этом журнале"
            ),
            "runs": [],
        }
        journal["comparison"] = comparison

        def verdict_run(spec: dict, phase: str, policy_params) -> dict:
            """Вердиктный прогон (temperature = 0) + Run Manifest §8."""
            ws_dir = work_root / f"eval-{spec['id']}"
            hidden = hidden_for(spec, env_tasks)
            copy_snapshot(env_tasks["tasks_dir"] / "public" / spec["id"], ws_dir)
            base = base_violations_at(ws_dir, spec, bin_path, hidden)
            decoding = {
                "temperature": 0.0, "top_p": args.top_p,
                "seed": int(spec.get("seed", args.seed)),
                "max_new_tokens": int(args.max_new_tokens),
            }
            result = execute_rollout(
                spec, env_tasks["tasks_dir"] / "public" / spec["id"], ws_dir,
                policy_params=policy_params, cfg=cfg, tokenizer=tokenizer,
                decoding=decoding, max_new_tokens=args.max_new_tokens,
                bin_path=bin_path, hidden=hidden, base_violations=base,
                weight=level_weight(spec["difficulty"]["level"]),
            )
            manifest = write_attempt_manifest(
                manifests_dir / f"{phase}-{spec['id']}.json",
                spec=spec, ws_dir=ws_dir, net_run=result["net_run"],
                verdict=result["verdict"], reward=result["reward"], cfg=cfg,
                decoding={k: decoding[k] for k in ("temperature", "top_p", "seed")},
                bin_path=bin_path, run_id=f"rl-{phase}-{spec['id']}",
                started=f"{run_date}T00:00:00+00:00",
                finished=f"{run_date}T00:00:01+00:00", repo_root=repo_root,
            )
            entry = {
                "task_id": spec["id"],
                "phase": phase,
                "passed": bool(result["verdict"].passed),
                "reward": round(result["reward"].total, 6),
                "reward_weighted": round(result["reward_weighted"], 6),
                "manifest": manifest["path"],
                "decoding_seed": decoding["seed"],
            }
            comparison["runs"].append(entry)
            return entry

        def measure_split(phase: str, policy_params) -> Optional[dict]:
            if not eval_ids:
                return None
            entries = [
                verdict_run(env_tasks["specs"][task_id], phase, policy_params)
                for task_id in eval_ids
            ]
            rewards = [item["reward_weighted"] for item in entries]
            return {
                "mean_reward": round(sum(rewards) / len(rewards), 6),
                "tasks": eval_ids,
                "passed": sum(1 for item in entries if item["passed"]),
                "per_task": entries,
            }

        print("[rl] замер «до»: holdout-сплит, temperature=0 ...", flush=True)
        holdout_before = measure_split("before", quantized(policy, cfg, qat))
        journal["holdout_before"] = (
            None if holdout_before is None else holdout_before["mean_reward"]
        )

        # --- 5. цикл обновления -------------------------------------------
        from net import optimizer as optimizer_mod

        ckpt_work_dir = work_root / "train"
        base_cache: dict[str, frozenset] = {}
        state = optimizer_mod.init_state(policy)
        optimizer_step = optimizer_mod.make_step(cfg)
        total_inner = max(args.steps * args.inner_epochs, 1)
        lr_at = optimizer_mod.cosine_schedule(args.lr, total_inner, args.warmup_ratio)

        steps_detail: list[dict] = []
        attempts: list[dict] = []
        verdict_pass = 0
        verdict_fail = 0
        degenerate_groups = 0
        skipped_epochs = 0
        reward_first: Optional[float] = None
        reward_last: Optional[float] = None
        inner_index = 0

        print(
            f"[rl] цикл: {args.steps} шагов × K={args.group_size} роллаутов, "
            f"inner_epochs={args.inner_epochs}, lr={args.lr}, τ_log={args.tau_log}, "
            f"β={args.beta}",
            flush=True,
        )
        for step in range(args.steps):
            task_id = train_ids[step % len(train_ids)]
            spec = env_tasks["specs"][task_id]
            base_ws = env_tasks["tasks_dir"] / "public" / task_id
            ws_dir = ckpt_work_dir / task_id
            hidden = hidden_for(spec, env_tasks)
            weight = level_weight(spec["difficulty"]["level"])
            if task_id not in base_cache:
                # База меряется по тому же пути, по которому пойдёт роллаут:
                # иначе разница путей сама станет «нарушением» (см. base_violations_at).
                copy_snapshot(base_ws, ws_dir)
                base_cache[task_id] = base_violations_at(ws_dir, spec, bin_path, hidden)
            base = base_cache[task_id]
            rollout_policy = quantized(policy, cfg, qat)
            # π_old: политика, породившая роллауты ЭТОГО шага. Внутренние эпохи
            # обновляют policy, а маска по log-ratio меряет дрейф именно к ней.
            source_params = policy

            records: list[dict] = []
            rewards: list[float] = []
            for rollout in range(args.group_size):
                decoding = {
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                    "seed": _rollout_seed(args.seed, int(spec["seed"]), step, rollout),
                    "max_new_tokens": int(args.max_new_tokens),
                }
                result = execute_rollout(
                    spec, base_ws, ws_dir,
                    policy_params=rollout_policy, cfg=cfg,
                    tokenizer=tokenizer, decoding=decoding,
                    max_new_tokens=args.max_new_tokens, bin_path=bin_path,
                    hidden=hidden, base_violations=base, weight=weight,
                )
                gen_ids = list(result["net_run"].generated_ids)
                records.append({
                    "prompt_ids": result["prompt_ids"],
                    "gen_ids": gen_ids,
                })
                rewards.append(float(result["reward_weighted"]))
                if result["verdict"].passed:
                    verdict_pass += 1
                else:
                    verdict_fail += 1
                attempts.append({
                    "step": step,
                    "task_id": task_id,
                    "rollout": rollout,
                    "level": spec["difficulty"]["level"],
                    "weight": weight,
                    "decoding_seed": decoding["seed"],
                    "passed": bool(result["verdict"].passed),
                    "reward_raw": round(float(result["reward"].total), 6),
                    "reward_weighted": round(float(result["reward_weighted"]), 6),
                    "spent_tokens": int(result["net_run"].spent_tokens),
                    "generated_tokens": len(gen_ids),
                })

            advantages = group_advantages(rewards)
            spread = max(rewards) - min(rewards) if rewards else 0.0
            if spread == 0.0:
                degenerate_groups += 1
            if reward_first is None:
                reward_first = sum(rewards) / len(rewards)
            reward_last = sum(rewards) / len(rewards)

            loss_value = 0.0
            grad_norm = 0.0
            shares: list[float] = []
            applied_epochs = 0
            bad_paths: list[str] = []
            total_tokens = sum(len(item["gen_ids"]) for item in records) or 1
            objectives = [
                make_rollout_objective(
                    cfg, record, advantage, total_tokens,
                    tau_log=args.tau_log, beta=args.beta, qat=qat, reference=reference,
                )
                for record, advantage in zip(records, advantages)
            ]
            for _ in range(args.inner_epochs):
                grads_total = None
                loss_total = 0.0
                share_values: list[float] = []
                for objective in objectives:
                    (value, (_share, _masked)), grads = objective(policy, source_params)
                    loss_total += float(value)
                    share_values.append(float(_share))
                    grads_total = add_grads(grads_total, grads)
                loss_value = loss_total
                shares.append(
                    sum(share_values) / len(share_values) if share_values else 1.0
                )
                grad_norm = gradient_norm(grads_total)
                diagnostics = gradient_diagnostics(grads_total)
                if diagnostics["finite"]:
                    policy, state = optimizer_step(
                        policy, grads_total, state, lr_at(inner_index)
                    )
                    applied_epochs += 1
                else:
                    # Шаг не применяется: NaN-веса в чекпойнт стадии не пишутся.
                    skipped_epochs += 1
                    if not bad_paths:
                        bad_paths = diagnostics["bad_paths"][:8]
                inner_index += 1

            entry = {
                "step": step,
                "task_id": task_id,
                "group_size": len(rewards),
                "reward_mean": round(sum(rewards) / len(rewards), 6),
                "reward_spread": round(spread, 6),
                "advantage_scale": round(max((abs(a) for a in advantages), default=0.0), 6),
                "loss": _json_float(loss_value),
                "grad_norm": _json_float(grad_norm),
                "mask_share": round(sum(shares) / len(shares), 6) if shares else None,
                "degenerate_group": spread == 0.0,
                "epochs": len(shares),
                "epochs_applied": applied_epochs,
                "nonfinite_gradient": applied_epochs < len(shares),
            }
            if bad_paths:
                entry["nonfinite_leaf_paths"] = bad_paths
            steps_detail.append(entry)
            print(
                f"[rl] шаг {step + 1}/{args.steps}: {task_id} K={len(rewards)} "
                f"reward={entry['reward_mean']:.4f} (разброс {entry['reward_spread']:.4f}) "
                f"loss={entry['loss']} |grad|={entry['grad_norm']} "
                f"mask={entry['mask_share']} "
                f"применено эпох={applied_epochs}/{len(shares)}",
                flush=True,
            )

        journal["steps_detail"] = steps_detail
        journal["attempts"] = attempts
        journal["steps"] = len(steps_detail)
        journal["tasks_used"] = sorted({item["task_id"] for item in attempts})
        journal["verdict_mix"] = {"passed": verdict_pass, "failed": verdict_fail}
        journal["reward_mean_first"] = None if reward_first is None else round(reward_first, 6)
        journal["reward_mean_last"] = None if reward_last is None else round(reward_last, 6)
        journal["groups"] = {
            "count": len(steps_detail),
            "degenerate": degenerate_groups,
            "definition": (
                "группа вырождена, если награда внутри группы одна и та же "
                "(разброс 0) — тогда преимущество ADR-005 п. 2 равно нулю и "
                "градиент группы нулевой; фиксируется числом, а не замалчивается"
            ),
            "note": (
                "ENVIRONMENT-V1 §5.1/§10: вырожденность групп — предмет кальбровки "
                "(precondition RL: pass-rate K3 ∈ [10%, 90%]); здесь она измерена и "
                "записана как факт смоука"
            ),
        }
        journal["update_path"] = {
            "inner_steps": inner_index,
            "applied": inner_index - skipped_epochs,
            "skipped_nonfinite": skipped_epochs,
            "guard": (
                "шаг с нечисловым градиентом не применяется: NaN-веса в чекпойнт "
                "стадии не пишутся, факт и листья фиксируются здесь"
            ),
            "weight_decay_note": (
                "веса меняются и при нулевом градиенте: MuonClip применяет "
                "weight decay каждый шаг (net/optimizer.py), поэтому "
                "weights_changed — не мера обученности, а факт обновления; "
                "сигнал описывают advantage_scale и grad_norm"
            ),
        }
        journal["findings"] = [
            {
                "id": "kda-chunk-padding-nan-backward",
                "kind": "defect (сеть, не дельта)",
                "where": "net/kda.py:apply_chunked + net/norm.py:l2_norm",
                "what": (
                    "apply_chunked дополняет последовательность нулями до кратности "
                    "chunk_size; backward l2_norm на строго нулевом векторе даёт NaN "
                    "(0/0 в производной ‖x‖), и NaN доходит до KDA-параметров "
                    "(W_q/W_k/conv_q/conv_k) при нулевом upstream-градиенте паддинга"
                ),
                "repro": (
                    "jax.grad(lambda y: jnp.sum(l2_norm(y)))(jnp.zeros((2, 4))) → "
                    "нечисловой градиент; у модели значение loss одно и то же при "
                    "pad=57 и pad=0 (12.4911), а градиент нечисловой только при pad>0"
                ),
                "why_sft_missed": (
                    "SFT-стадия и приёмочные тесты используют длины, кратные чанку "
                    "(8192 % 64 == 0, 128 % 64 == 0) — паддинга не возникает"
                ),
                "mitigation_in_delta": (
                    "loss-путь считается с chunk = длина последовательности (pad = 0): "
                    "семантика та же (обе формы KDA эквивалентны), дефектный backward "
                    "не задействован"
                ),
                "out_of_scope_fix": (
                    "правка net/ (eps внутрь нормы либо stop-gradient на паддинге) — "
                    "отдельное решение: меняет числа паритета и чекпойнт SFT"
                ),
                "status": "открыто, вынесено архитектору",
            },
            {
                "id": "nonfinite-gradient-guard",
                "kind": "страховка дельты",
                "what": (
                    "перед каждым шагом оптимизатора градиент проверяется на "
                    "числовую пригодность; нечисловой шаг не применяется"
                ),
                "applied": inner_index - skipped_epochs,
                "skipped": skipped_epochs,
            },
        ]
        if skipped_epochs:
            journal["notes"].append(
                f"Нечисловой градиент: пропущено шагов {skipped_epochs} из "
                f"{inner_index} — веса не отравлены (см. findings)"
            )

        # --- 6. замер «после» и стоп-правило -------------------------------
        print("[rl] замер «после»: holdout-сплит, temperature=0 ...", flush=True)
        holdout_after = measure_split("after", quantized(policy, cfg, qat))
        journal["holdout_after"] = (
            None if holdout_after is None else holdout_after["mean_reward"]
        )
        journal["stop_rule"] = stop_rule(
            None if holdout_before is None else holdout_before["mean_reward"],
            None if holdout_after is None else holdout_after["mean_reward"],
        )

        # --- 7. чекпойнт ---------------------------------------------------
        hash_after = _tree_hash(policy)
        norm_after = _param_norm(policy)
        journal["weights_changed"] = bool(hash_after != tree_hash_loaded)
        journal["parameter_norm_after"] = round(norm_after, 6)
        journal["parameter_norm_delta"] = round(
            norm_after - journal["parameter_norm_before"], 6
        )
        journal["checkpoint"] = save_stage_checkpoint(policy, ckpt_dir, shared, hash_after)
        print(
            f"[rl] чекпойнт: tree_hash={hash_after[:16]}…, "
            f"round-trip ok -> {journal['checkpoint']['path']}",
            flush=True,
        )

        # --- 8. стоимость и статус -----------------------------------------
        wall_clock = time.time() - started
        journal["wall_clock_s"] = round(wall_clock, 3)
        journal["gpu_hours_actual"] = round(wall_clock / 3600.0, 6)
        journal["budget"] = sft.budget_report(run_id, wall_clock / 3600.0,
                                              args.budget_limit_usd)
        journal["rollout_workdir"] = (
            "временный каталог вне репозитория, удалён после прогона: копий весов "
            "и данных в рабочем каталоге не остаётся (C-032/C-033)"
        )
        journal["scope"] = {
            "rollouts": len(attempts),
            "steps": len(steps_detail),
            "tasks": {"train": len(train_ids), "eval": len(eval_ids)},
            "window": (
                "СМОУК: стадия исполнена в смоук-масштабе (шаги и роллауты в этом "
                "журнале); закрытием стадии «в полном объёме» не является (спека §5)"
            ),
            "calibration_precondition": (
                "ENVIRONMENT-V1 §10: precondition запуска RL — кальбровка K3 "
                "(pass-rate ∈ [10%, 90%]); она не проведена, поэтому вырожденность "
                "групп ожидаема и записана числом"
            ),
        }
        journal["status"] = "executed"
        journal["finished_at"] = datetime.now(timezone.utc).isoformat()
        sft._write_journal(journal, out_dir)

        print(
            f"[rl] стадия: executed; {len(steps_detail)} шагов, {len(attempts)} "
            f"роллаутов, вердикты passed/failed={verdict_pass}/{verdict_fail}; "
            f"{wall_clock:.1f} с ({journal['gpu_hours_actual']:.6f} GPU-ч) — СМОУК",
            flush=True,
        )
        print(f"[rl] стоп-правило: {journal['stop_rule']['decision']} — "
              f"{journal['stop_rule']['reason']}", flush=True)
        print(f"[rl] смета (AD-8/C-041): {journal['budget']['verdict']}", flush=True)
        print(f"[rl] журнал: {journal['run_ref']}", flush=True)
        shutil.rmtree(work_root, ignore_errors=True)
        return journal, True
    except StageError as exc:
        return fail("absent", str(exc))
    except Exception as exc:  # noqa: BLE001 — журнал обязан быть записан честно
        import traceback

        traceback.print_exc()
        return fail("failed", f"{type(exc).__name__}: {exc}")
    finally:
        # Рабочий каталог роллаутов — вне репозитория и удаляется в любом исходе:
        # копий весов/данных и снапшотов кейса после стадии не остаётся.
        if work_root is not None:
            shutil.rmtree(work_root, ignore_errors=True)


# ---------------------------------------------------------------------------
# Мелкие помощники (jax/среда импортируются лениво и только после пиннинга)
# ---------------------------------------------------------------------------


def _optional_rel(path: Path, repo_root, sft) -> Optional[str]:
    return None if path is None else (
        sft.repo_rel(path, repo_root) if path.exists() else "не создан генерацией"
    )


def _sanitize(message: str, repo_root: Optional[Path]) -> str:
    """Убрать абсолютные пути из текста ошибки (журнал обязан быть портируемым).

    Диагностика исключений приходит от ОС и содержит абсолютные пути; в журнал
    стадии они попасть не должны (ADR-014 п. 8) — иначе «переносимый» артефакт
    выдаёт личные пути сборочной машины.
    """
    text = str(message)
    for absolute, replacement in (
        (str(repo_root) if repo_root is not None else None, ""),
        (str(Path.home()), "~"),
    ):
        if absolute:
            text = text.replace(absolute + "/", replacement + "/")
            if replacement == "":
                text = text.replace(absolute, "")
            else:
                text = text.replace(absolute, replacement)
    return text


def copy_snapshot(src: Path, dst: Path) -> None:
    """Снапшот кейса через ``env.util.copy_case_snapshot`` (среда как есть)."""
    from env.util import copy_case_snapshot

    copy_case_snapshot(src, dst)


def quantized(params, cfg, qat: bool):
    """Политика в том виде, в каком она исполняется и считается в loss (QAT).

    При включённом QAT (ADR-005 п. 7: включается со SFT) и роллаут, и loss идут
    по fake-quant'нутым весам, а мастер-веса полной точности обновляет
    оптимизатор — иначе политика, породившая роллауты, и политика в loss были
    бы разными.
    """
    if not qat:
        return params
    from net import quant as quant_mod

    return quant_mod.apply_fake_quant_tree(params)


def _tree_hash(params) -> str:
    from net import checkpoint as checkpoint_mod

    return checkpoint_mod.tree_hash(params)


def _param_norm(params) -> float:
    import jax
    import jax.numpy as jnp

    total = 0.0
    for leaf in jax.tree_util.tree_leaves(params):
        total += float(jnp.sum(jnp.square(leaf)))
    return math.sqrt(total)


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)
    journal, ok = run_stage(args)
    if args.json:
        print(json.dumps(journal, ensure_ascii=False, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
