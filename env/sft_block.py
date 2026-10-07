"""E-2: генератор env-блока SFT-микса по SFT-STAGE.delta §8.5.

Назначение — закрыть пробел «интерфейс среды = 0% в v12-normalized»
(LAG-ADR-061/062): блок эталонных траекторий работы с 4 инструментами
ENVIRONMENT-V1 §13 (``list_files``, ``read_file``, ``edit_file``,
``run_gates``) + ``finish``. Без него RL-замер после SFT измеряет незнакомый
API, а не способность.

Сценарии (§8.5):

* **S1 «ремонт порчи»** (~70%) — чистый кейс → детерминированная порча
  мутатором (:mod:`env.corruption`, L0/L1) → reference-траектория
  ``list_files → read_file → edit_file → … → run_gates → finish``;
  восстановление известно конструктивно из Damage-листа.
* **S2 «чистый прогон»** (~30%) — кейс без порчи → ``run_gates → finish``
  (чтение вердикта и pass-семантика).

Инварианты
----------
* **AD-2 — наблюдения только реальные.** Каждый ``<tool_response>`` — результат
  фактического исполнения соответствующего инструмента на рабочей копии кейса
  (не выдуманная строка).
* **Единый источник формата.** Парсер ходов, обёртка наблюдений, инструменты и
  путь верификации берутся из пиннутого раннера ``tools/rollout_harness.py``
  (§13). Парсер здесь НЕ дублируется: формат хода/наблюдения байт-в-байт тот
  же, что у RL-раннера.
* **Маскирование.** Ход ассистента = ``<think>…</think>``? +
  ``<tool_call>{"name":…,"args":{…}}</tool_call>`` (плоский JSON, как в v12);
  наблюдение — сообщение роли ``tool`` в ``<tool_response>…</tool_response>``.
  Поле ``assistant_mask`` сообщения = 1 для ассистента (в лоссе) и 0 для
  system/user/tool (маскировано) — та же семантика, что ``assistant_mask``
  журнала раннера.
* **Детерминизм.** ``--seed S`` → байт-идентичный выход: сортированные обходы,
  фиксированный порядок полей JSON, относительные (нормализованные) пути в
  наблюдении вердикта.
* **Параллелизм (E-2.7).** ``--workers N`` (дефолт 1 — поведение неизменно):
  шарды ``pos % N`` исполняются :class:`multiprocessing.Pool` (CPU-bound гейты,
  не потоки). Содержимое траектории от ``N`` не зависит, записи собираются в
  порядке глобальных индексов → итоговый jsonl инвариантен по ``N``. Падение
  одной траектории не роняет пул: она исключается, ``summary['fails']`` несёт
  счётчик, строка — в stderr (прогресс-лог).

CLI::

    python3 -m env.sft_block --n-s1 70 --n-s2 30 --seed 20261005 \
        --workers 8 --out data/datasets/sft_env_block_v1-mini.jsonl --validate
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import json
import multiprocessing
import pickle
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable, Optional

# ── Единый источник формата: пиннутый раннер RL (§13) ──────────────────────
# env/ импортирует tools/rollout_harness.py намеренно: интерфейс агента env-блока
# обязан совпадать с раннером RL до байта (порядок анти-LAG-061/062). Парсер,
# обёртка наблюдений, инструменты и путь верификации НЕ дублируются.
ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "tools")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import rollout_harness as rh  # noqa: E402  (пиннутый интерфейс §13)

from env import corruption as corruption_mod  # noqa: E402
from env.util import (  # noqa: E402
    EMPTY_HIDDEN_SHA256,
    copy_case_snapshot,
    sha256_bytes,
    workspace_size_cap,
)

SCHEMA = "axiom-sft-env-block/1"
SOURCE = "env_block_v1"
GENERATOR = "env/sft_block.py"

#: Порог незавершённых tool_call (§8.2.1) и классы стража C-044 (§8.1).
UNFINISHED_MAX = 0.05
GUARD = "tools/check_sft_structure.py"

#: Тексты задач — как в кейсах пула (restore/verify), русский.
PROMPT_S1 = (
    "Восстанови архитектурные гейты кейса до зелёного статуса: fitness "
    "(CONSTRAINTS.yaml), spine и trace должны пройти без error-находок, "
    "не внося новых нарушений."
)
PROMPT_S2 = (
    "Проверь текущее состояние архитектурных гейтов кейса (fitness по "
    "CONSTRAINTS.yaml, spine, trace) и заверши работу, сообщив вердикт."
)

SYSTEM_PROMPT = rh.SYSTEM_PROMPT

#: Шаблоны think (английский, процедурный, 1–3 предложения), §8.5.
THINK_LIST = (
    "I need to inspect the case tree to locate the files before making any "
    "changes. Let me list the workspace contents."
)
THINK_READ = (
    "The listing shows the case layout. I will read {path} to see its current "
    "state before editing."
)
THINK_EDIT = (
    "The content around the damaged region in {path} is clear. I will restore "
    "the removed content with a single exact-match edit."
)
THINK_GATES_FIXED = (
    "The repair is applied. Let me run the gates to confirm the corrupted case "
    "is restored to its clean state."
)
THINK_GATES_CLEAN = (
    "The case carries no damage, so I expect the gates to describe the clean "
    "state. Let me run the gates and read the verdict."
)
THINK_FINISH = (
    "I have the verdict from run_gates. I will finish and report the outcome."
)


# ── Порча (только восстановимая 4 инструментами) ───────────────────────────


#: Ходы эталонной траектории атома ``drift_config_value`` (ADR-036, дельта D7):
#: гейты → ADR → конфиг → правка → гейты → finish. Значение берётся из ADR
#: В ВОРКСПЕЙСЕ, а не из чистой копии (урок DEF-1: артефакт сообщён).
CONFIG_DRIFT_SEQUENCE = (
    "run_gates", "read_file", "read_file", "edit_file", "run_gates", "finish",
)


def _config_leaf(dotted: str) -> str:
    return dotted.split(".")[-1]


def _expected_from_adr(adr_text: str, value: Any) -> Any:
    """Подтверждает ожидаемое значение по тексту ADR воркспейса.

    Поддерживает числовое и булево представление; не найдено — ValueError
    (траектория не выдумывает значение из чистой копии).
    """
    reps = [str(value), str(value).lower(), json.dumps(value)]
    if isinstance(value, bool):
        reps += ["true" if value else "false", "True" if value else "False"]
    for rep in reps:
        if rep and rep in adr_text:
            return value
    raise ValueError(f"значение {value!r} не подтверждено ADR воркспейса")


def build_config_drift_trajectory(
    ws_dir: Path,
    verifier: Any,
    *,
    claim: dict[str, Any],
    adr_rel: str,
) -> list[dict[str, Any]]:
    """Эталонная SFT-траектория починки атома ``drift_config_value`` (§D7).

    Последовательность ``CONFIG_DRIFT_SEQUENCE``: ``run_gates`` → ``read_file``
    ADR → ``read_file`` конфига → ``edit_file`` → ``run_gates`` → ``finish``.
    Ожидаемое значение берётся из ADR **в воркспейсе** (не из чистой копии).
    """
    import sys as _sys

    if str(ROOT / "tools") not in _sys.path:
        _sys.path.insert(0, str(ROOT / "tools"))
    import rollout_harness as rh

    binding = claim.get("binding") or {}
    config_rel = str(binding.get("file"))
    key = str(binding.get("path"))
    leaf = _config_leaf(key)
    tools = rh.WorkspaceTools(ws_dir, verifier)
    traj = _Trajectory(tools)
    traj.start(PROMPT_S2)

    traj.step("Run the gates to see the verdict.", "run_gates", {})
    adr_text = traj.step("Read the ADR the message named.", "read_file", {"path": adr_rel})
    expected = _expected_from_adr(adr_text, binding.get("value"))
    config_text = traj.step("Read the config binding.", "read_file", {"path": config_rel})
    current = json.loads(config_text)
    for part in key.split("."):
        current = current[part]
    old = f'"{leaf}": {json.dumps(current)}'
    new = f'"{leaf}": {json.dumps(expected)}'
    traj.step("Restore the value the ADR fixes.", "edit_file",
              {"path": config_rel, "old": old, "new": new})
    traj.step("Re-run the gates to confirm.", "run_gates", {})
    traj.finish("The binding matches the ADR again.", "Gates verdict confirmed.")
    return traj.messages


def _damage_effective(clean_root: Path, damage: corruption_mod.Damage) -> bool:
    """Порча реально применима (иначе мутатор молча ничего не сделал)."""
    target = clean_root / damage.file
    if not target.is_file():
        return False
    if damage.kind == "drift_config_value":
        return damage.path is not None and damage.new_value is not None
    text = target.read_text(encoding="utf-8")
    if damage.kind == "remove_adr_section":
        return damage.section is not None and damage.section in text
    if damage.kind in ("break_affects", "break_verified_by"):
        field = "affects" if damage.kind == "break_affects" else "verified_by"
        return any(ln.lstrip().startswith(field + ":") for ln in text.splitlines())
    return False


def plan_repairable(
    clean_root: Path, seed: int, level: str, atoms_version: str = "v1"
) -> list[corruption_mod.Damage]:
    """Damage-лист мутатора, суженный до восстановимого 4 инструментами §13.

    ``break_ad_link`` (удаление файла) исключается: интерфейс агента v1 не несёт
    инструмента создания файла, поэтому такое повреждение конструктивно
    невосстановимо. ``drift_config_value`` (``atoms_version: v2``) восстановим:
    ``edit_file`` правит значение ``net/config.json`` по тексту ADR из воркспейса.
    """
    planned = corruption_mod.plan_damages(clean_root, seed, level, atoms_version=atoms_version)
    out = [
        d
        for d in planned
        if d.kind != "break_ad_link" and _damage_effective(clean_root, d)
    ]
    if not out:
        raise ValueError(f"нет восстановимых повреждений для {level} (seed={seed})")
    return out


def _insertion_edit(damaged: str, clean: str) -> tuple[str, str]:
    """(old, new) для ``edit_file``: чистый текст = damaged со вставкой.

    Оба повреждения (``remove_adr_section``, ``break_verified_by``) — чистые
    вставки: clean длиннее damaged на ``L`` символов в точке ``p`` (общий
    префикс). Окно расширяется до единственного вхождения ``old`` в damaged;
    вырожденный fallback — полный текст файла.
    """
    if damaged == clean:
        raise ValueError("тексты совпадают — правка не нужна")
    p = 0
    m = min(len(damaged), len(clean))
    while p < m and damaged[p] == clean[p]:
        p += 1
    L = len(clean) - len(damaged)
    if L <= 0:
        return damaged, clean  # не вставка — полная замена (страховка)
    for ctx in (24, 48, 96, 192, 384, 768, 1536, 3072, 6144):
        a = max(0, p - ctx)
        b = min(len(damaged), p + ctx)
        old = damaged[a:b]
        if old and damaged.count(old) == 1:
            return old, clean[a : b + L]
    return damaged, clean


# ── Вердикт: реальный + нормализация путей (детерминизм) ────────────────────


def _rel_file(value: Any, workspace: Path) -> str:
    """Путь нарушения относительно workspace (устраняет машинный префикс)."""
    if not value or not isinstance(value, str):
        return ""
    try:
        p = Path(value)
        if p.is_absolute():
            rel = p.resolve().relative_to(Path(workspace).resolve())
            return rel.as_posix() or "."
        return p.as_posix()
    except (ValueError, OSError):
        return Path(value).as_posix()


def _normalize_verdict(raw: dict[str, Any], workspace: Path) -> dict[str, Any]:
    """Нормализует реальный вердикт: сортировка + относительные пути.

    Значения вердикта (``passed``/``tests_passed``/правила) не меняются —
    меняется только представление путей, иначе абсолютный tmp-путь рабочей
    копии протёк бы в наблюдение и сломал байт-детерминизм.
    """
    violations = [
        {"rule": str(v.get("rule", "")), "file": _rel_file(v.get("file"), workspace)}
        for v in raw.get("violations", [])
    ]
    violations.sort(key=lambda d: (d["rule"], d["file"]))
    return {
        "passed": bool(raw.get("passed")),
        "violations": violations,
        "tests_passed": bool(raw.get("tests_passed")),
        "gates": {k: bool(v) for k, v in sorted(raw.get("gates", {}).items())},
    }


class BlockVerifier:
    """Реальный путь верификации env (``EnvVerifier``) + нормализация вердикта."""

    def __init__(
        self,
        spec: dict[str, Any],
        base_ws: Path,
        *,
        bin: Optional[str] = None,
        gates_timeout: float = rh.DEFAULT_GATES_TIMEOUT,
    ) -> None:
        self._env = rh.EnvVerifier(
            spec, base_ws, bin=bin, gates_timeout=gates_timeout
        )

    def run_gates(self, workspace: Path) -> dict[str, Any]:
        return _normalize_verdict(self._env.run_gates(Path(workspace)), workspace)


def _default_verifier_factory(
    spec: dict[str, Any], ws: Path, *, bin: Optional[str] = None
) -> BlockVerifier:
    """Модульная фабрика реального верификатора.

    Отдельная от ``generate_block`` функция — чтобы :class:`functools.partial`
    от неё был picklable: воркеры пула получают фабрику через pickle (E-2.7).
    """
    return BlockVerifier(spec, ws, bin=bin)


# ── Сборка ходов траектории ────────────────────────────────────────────────


def _tool_call(name: str, args: dict[str, Any]) -> str:
    payload = json.dumps({"name": name, "args": args}, ensure_ascii=False, sort_keys=True)
    return f"<tool_call>{payload}</tool_call>"


def _think(text: str) -> str:
    return f"<think>\n{text}\n</think>"


def _masked(role: str, content: str) -> dict[str, Any]:
    return {"role": role, "content": content, "assistant_mask": 0}


def _assistant(content: str) -> dict[str, Any]:
    return {"role": "assistant", "content": content, "assistant_mask": 1}


class _Trajectory:
    """Конструктор одной траектории: реальные вызовы инструментов §13."""

    def __init__(self, tools: rh.WorkspaceTools) -> None:
        self.tools = tools
        self.messages: list[dict[str, Any]] = []
        self.turns = 0
        self.tool_calls = 0

    def start(self, prompt: str) -> None:
        self.messages.append(_masked("system", SYSTEM_PROMPT))
        self.messages.append(_masked("user", prompt))

    def step(self, think: str, name: str, args: dict[str, Any]) -> str:
        """Ход ассистента (think + tool_call) → реальное наблюдение роли tool."""
        content = f"{_think(think)}\n{_tool_call(name, args)}"
        self.messages.append(_assistant(content))
        self.turns += 1
        call = rh.ToolCall(name=name, args=args, raw=_tool_call(name, args))
        outcome = self.tools.dispatch(call)
        body = outcome.body
        self.messages.append(_masked("tool", rh.wrap_tool_response(body)))
        self.tool_calls += 1
        return body

    def finish(self, think: str, summary: str) -> None:
        """Финальный finish-ход + содержательный ответ (для стража — no_answer)."""
        content = f"{_think(think)}\n{_tool_call('finish', {})}\n\n{summary}"
        self.messages.append(_assistant(content))
        self.turns += 1


def _verdict_text(verdict: dict[str, Any]) -> str:
    """Краткая текстовая сводка реального вердикта (для финального ответа)."""
    return f"passed={verdict['passed']}, violations={len(verdict.get('violations', []))}"


def _finish_summary(action: str, verdict: dict[str, Any]) -> str:
    """Содержательный финальный ответ: реальный вердикт + интерпретация.

    При красном базисе (предсуществующий стенд-долг, а не порча/действия
    траектории) это прямо отражено — иначе SFT учил бы заявлять успех при
    красных гейтах.
    """
    text = f"{action} Gates verdict: {_verdict_text(verdict)}."
    if not verdict["passed"]:
        text += (
            " The listed violations are pre-existing in the case baseline "
            "(stand debt), not introduced by this run."
        )
    return text


def _make_spec(source: str, kind: str, tests_cmd: str = "true") -> dict[str, Any]:
    return {
        "id": "env-block",
        "source": source,
        "prompt": PROMPT_S1 if kind == "restore-gates" else PROMPT_S2,
        "objective": {"kind": kind, "tests_cmd": tests_cmd},
        "verifier": {
            "constraints": "CONSTRAINTS.yaml",
            "spine": True,
            "trace": True,
            "hidden_constraints_sha256": EMPTY_HIDDEN_SHA256,
        },
        "max_tokens": 131072,
    }


def _damage_seed(base_seed: int, index: int) -> int:
    """Seed порчи, отличный от RL-пула (у RL — малые ``base*10000+…``)."""
    digest = hashlib.sha256(f"env-block-damage:{base_seed}:{index}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


# ── S1: ремонт порчи ───────────────────────────────────────────────────────


def build_s1(
    clean_root: Path,
    seed: int,
    index: int,
    level: str,
    *,
    verifier_factory: Callable[[dict[str, Any], Path], Any],
    workdir: Path,
) -> dict[str, Any]:
    """Tраектория S1: порча → восстановление по Damage-листу → вердикт."""
    damage_seed = _damage_seed(seed, index)
    damages = plan_repairable(clean_root, damage_seed, level)

    ws = workdir / f"s1-{index:05d}"
    copy_case_snapshot(clean_root, ws)
    for d in damages:
        corruption_mod.apply_damage(ws, d)
    workspace_size_cap(ws)

    # Восстановительные правки известны конструктивно из Damage-листа.
    edits = []
    for d in damages:
        clean_text = (clean_root / d.file).read_text(encoding="utf-8")
        damaged_text = (ws / d.file).read_text(encoding="utf-8")
        old, new = _insertion_edit(damaged_text, clean_text)
        edits.append((d, old, new))

    spec = _make_spec("corruption", "restore-gates")
    verifier = verifier_factory(spec, ws)
    tools = rh.WorkspaceTools(ws, verifier)
    traj = _Trajectory(tools)
    traj.start(PROMPT_S1)

    traj.step(THINK_LIST, "list_files", {})
    for d, old, new in edits:
        traj.step(THINK_READ.format(path=d.file), "read_file", {"path": d.file})
        body = traj.step(
            THINK_EDIT.format(path=d.file),
            "edit_file",
            {"path": d.file, "old": old, "new": new},
        )
        if body != "ok":
            raise RuntimeError(f"reference-edit не прошёл для {d.file}: {body!r}")

    # Конструктивная проверка «порча прошла после фикса»: файл восстановлен
    # байт-в-байт к чистому состоянию.
    for d, _old, _new in edits:
        if (ws / d.file).read_bytes() != (clean_root / d.file).read_bytes():
            raise RuntimeError(f"восстановление не совпало с чистым: {d.file}")

    gates_body = traj.step(THINK_GATES_FIXED, "run_gates", {})
    verdict = json.loads(gates_body)
    files = ", ".join(sorted({d.file for d, _o, _n in edits}))
    summary = _finish_summary(
        f"Repair complete: restored content in {files}; the corruption is resolved.",
        verdict,
    )
    traj.finish(THINK_FINISH, summary)

    return _record(
        traj,
        task_type=f"env_repair_{level.lower()}",
        scenario="S1",
        case_id=clean_root.name,
        seed=seed,
        verdict=verdict,
        extra={
            "corruption": {
                "level": level,
                "seed": damage_seed,
                "damages": [d.as_dict() for d in damages],
            }
        },
    )


# ── S2: чистый прогон ──────────────────────────────────────────────────────


def build_s2(
    clean_root: Path,
    seed: int,
    index: int,
    *,
    verifier_factory: Callable[[dict[str, Any], Path], Any],
    workdir: Path,
) -> dict[str, Any]:
    """Траектория S2: кейс без порчи → run_gates → finish (чтение вердикта)."""
    ws = workdir / f"s2-{index:05d}"
    copy_case_snapshot(clean_root, ws)
    workspace_size_cap(ws)

    spec = _make_spec("real", "restore-gates")
    verifier = verifier_factory(spec, ws)
    tools = rh.WorkspaceTools(ws, verifier)
    traj = _Trajectory(tools)
    traj.start(PROMPT_S2)

    gates_body = traj.step(THINK_GATES_CLEAN, "run_gates", {})
    verdict = json.loads(gates_body)
    summary = _finish_summary(
        "Verification complete: the case carries no corruption; the verdict "
        "reflects the clean state.",
        verdict,
    )
    traj.finish(THINK_FINISH, summary)

    return _record(
        traj,
        task_type="env_verify",
        scenario="S2",
        case_id=clean_root.name,
        seed=seed,
        verdict=verdict,
    )


def _record(
    traj: _Trajectory,
    *,
    task_type: str,
    scenario: str,
    case_id: str,
    seed: int,
    verdict: dict[str, Any],
    extra: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "messages": traj.messages,
        "task_type": task_type,
        "n_turns": traj.turns,
        "n_tool_calls": traj.tool_calls,
        "source": SOURCE,
        "schema": SCHEMA,
        "generator": GENERATOR,
        "scenario": scenario,
        "case_id": case_id,
        "seed": seed,
        "verdict_passed": bool(verdict["passed"]),
        "verdict": verdict,
    }
    if extra:
        record.update(extra)
    if not verdict["passed"]:
        # Вердикт реальный; при красном базисе помечаем, что нарушения —
        # предсуществующие (стенд-долг), а не внесены ходом агента.
        rules = sorted({v["rule"].split(":")[0] for v in verdict.get("violations", [])})
        record["verdict_note"] = (
            "passed=false: pre-existing case violations ("
            + (", ".join(rules) if rules else "unknown")
            + "), not introduced by this trajectory"
        )
    return record


# ── Генерация блока ────────────────────────────────────────────────────────


#: Чистый кейс по умолчанию — пул E-1.5 (``env/data-regen/public/real-l0-00``),
#: если он есть на диске (пул вне VCS, регенерируется ``env.main generate``);
#: иначе — рабочий репозиторий (тогда кейс строится снапшотом E-1.5 на месте).
DEFAULT_POOL_CASE = ROOT / "env" / "data-regen" / "public" / "real-l0-00"


def default_clean_root() -> Path:
    return DEFAULT_POOL_CASE if (DEFAULT_POOL_CASE / "CONSTRAINTS.yaml").is_file() else ROOT


#: Префикс строки прогресс-лога E-2.7 (deliverable 2) в stderr.
PROGRESS_PREFIX = "[sft-block]"


def _progress_line(result: dict[str, Any]) -> str:
    """Строка по одной траектории: шард, индекс, ``pass``/``fail``.

    ``pass`` — траектория построена (в detail виден вердикт гейтов), ``fail`` —
    исключение при построении (в detail — его тип и текст).
    """
    status = "pass" if result["ok"] else "fail"
    if result["ok"]:
        detail = "verdict=" + (
            "passed" if result["record"]["verdict_passed"] else "failed"
        )
    else:
        detail = f"error={result['error']}"
    return (
        f"{PROGRESS_PREFIX} shard={result['shard']} index={result['pos']} "
        f"scenario={result['scenario']} {status} {detail}"
    )


def _log_progress(result: dict[str, Any]) -> None:
    """Печатает строку прогресса в stderr с немедленным flush (фон-контроль)."""
    print(_progress_line(result), file=sys.stderr, flush=True)


def _run_task(task: dict[str, Any]) -> dict[str, Any]:
    """Выполняет одно шард-задание (S1/S2-траекторию) и НИКОГДА не бросает.

    Падение траектории (в т.ч. ``TimeoutError`` из верификатора) возвращается
    как ``ok=False`` — пул воркеров не рушится, задача исключается из блока.
    Функция модульного уровня и принимает один pickle-совместимый аргумент:
    требование ``multiprocessing`` (spawn/fork шлют задание через pickle).
    """
    try:
        clean_root = Path(task["clean_root"])
        workdir = Path(task["workdir"])
        factory = task["verifier_factory"]
        if task["scenario"] == "S1":
            record = build_s1(
                clean_root, task["seed"], task["index"], task["level"],
                verifier_factory=factory, workdir=workdir,
            )
        else:
            record = build_s2(
                clean_root, task["seed"], task["index"],
                verifier_factory=factory, workdir=workdir,
            )
        return {
            "pos": task["pos"], "shard": task["shard"],
            "scenario": task["scenario"], "index": task["index"],
            "ok": True, "record": record, "error": None,
        }
    except Exception as exc:  # noqa: BLE001 — изоляция сбоя одной траектории
        return {
            "pos": task["pos"], "shard": task["shard"],
            "scenario": task["scenario"], "index": task["index"],
            "ok": False, "record": None,
            "error": f"{type(exc).__name__}: {exc}",
        }


def _build_tasks(
    *,
    n_s1: int,
    n_s2: int,
    seed: int,
    level: str,
    clean_root: Path,
    verifier_factory: Callable[[dict[str, Any], Path], Any],
    workdir: Path,
    workers: int,
) -> list[dict[str, Any]]:
    """Пул задач в порядке генерации (сначала S1, затем S2).

    Шард траектории — ``pos % workers`` (рекомендация E-2.7, внутри шарда —
    последовательный перебор). Содержимое траектории от шарда НЕ зависит:
    шард лишь исполнитель. Порядок результата восстанавливается по ``pos``,
    поэтому итоговый jsonl инвариантен по ``workers``.
    """
    tasks: list[dict[str, Any]] = []
    pos = 0
    for scenario, count, scenario_level in (
        ("S1", int(n_s1), level),
        ("S2", int(n_s2), None),
    ):
        for index in range(count):
            tasks.append({
                "pos": pos,
                "shard": pos % workers,
                "scenario": scenario,
                "index": index,
                "level": scenario_level,
                "seed": seed,
                "clean_root": str(clean_root),
                "workdir": str(workdir),
                "verifier_factory": verifier_factory,
            })
            pos += 1
    return tasks


def _pool_context() -> Any:
    """Контекст пула: fork на Linux (быстро для CPU-bound), иначе spawn."""
    methods = multiprocessing.get_all_start_methods()
    return multiprocessing.get_context("fork" if "fork" in methods else "spawn")


def _require_picklable(factory: Callable[[dict[str, Any], Path], Any]) -> None:
    """Рано и явно: фабрика верификатора обязана переживать pickle."""
    try:
        pickle.dumps(factory)
    except Exception as exc:
        raise ValueError(
            "workers>1 требует picklable verifier_factory (пул передаёт фабрику "
            "в воркеры через pickle); передайте функцию/класс модульного уровня "
            "или используйте workers=1"
        ) from exc


def _dispatch(tasks: list[dict[str, Any]], workers: int) -> list[dict[str, Any]]:
    """Исполняет задания: последовательно (workers=1) либо пулом процессов."""
    if not tasks:
        return []
    if workers <= 1:
        results: list[dict[str, Any]] = []
        for task in tasks:
            result = _run_task(task)
            _log_progress(result)
            results.append(result)
        return results
    _require_picklable(tasks[0]["verifier_factory"])
    with _pool_context().Pool(processes=workers) as pool:
        results = []
        for result in pool.imap_unordered(_run_task, tasks, chunksize=1):
            _log_progress(result)
            results.append(result)
    return results


def generate_block(
    *,
    n_s1: int,
    n_s2: int,
    seed: int,
    full_case_root: Optional[Path] = None,
    s1_level: str = "L0",
    bin: Optional[str] = None,
    verifier_factory: Optional[Callable[[dict[str, Any], Path], Any]] = None,
    workdir: Optional[Path] = None,
    workers: int = 1,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Генерирует env-блок; возвращает (records, summary).

    Порядок: сначала S1, затем S2 (детерминированно). Порча каждого S1 —
    собственный seed от ``(seed, index)``, отличный от seed'ов RL-пула.

    ``workers`` (E-2.7, дефолт 1 — поведение неизменно): >1 — задания
    раскладываются по шардам ``pos % workers`` и исполняются
    :class:`multiprocessing.Pool` (CPU-bound гейты, не потоки). Результат
    собирается в порядке глобальных индексов, поэтому файл инвариантен по
    ``workers`` (при picklable ``verifier_factory``); упавшие траектории
    исключаются, их число — в ``summary['fails']``.
    """
    if s1_level not in ("L0", "L1"):
        raise ValueError("s1_level должен быть 'L0' или 'L1'")
    workers = int(workers)
    if workers < 1:
        raise ValueError("workers должен быть >= 1")
    clean_root = Path(full_case_root) if full_case_root is not None else default_clean_root()
    if not (clean_root / "CONSTRAINTS.yaml").is_file():
        raise FileNotFoundError(f"чистый кейс без CONSTRAINTS.yaml: {clean_root}")

    if verifier_factory is None:
        # partial модульной фабрики — pickle-совместим для пула воркеров.
        verifier_factory = functools.partial(_default_verifier_factory, bin=bin)

    own_workdir = workdir is None
    workdir = Path(workdir) if workdir is not None else Path(
        tempfile.mkdtemp(prefix="sft-env-block-")
    )
    workdir.mkdir(parents=True, exist_ok=True)

    try:
        tasks = _build_tasks(
            n_s1=n_s1, n_s2=n_s2, seed=seed, level=s1_level,
            clean_root=clean_root, verifier_factory=verifier_factory,
            workdir=workdir, workers=workers,
        )
        results = _dispatch(tasks, workers)
    finally:
        if own_workdir:
            shutil.rmtree(workdir, ignore_errors=True)

    results.sort(key=lambda r: r["pos"])
    records = [r["record"] for r in results if r["ok"]]
    failures = [r for r in results if not r["ok"]]

    summary = block_summary(records)
    summary["seed"] = seed
    summary["clean_case_root"] = str(clean_root)
    summary["s1_level"] = s1_level
    summary["workers"] = workers
    summary["requested"] = int(n_s1) + int(n_s2)
    summary["fails"] = len(failures)
    summary["failures"] = [
        {"pos": r["pos"], "scenario": r["scenario"], "index": r["index"],
         "error": r["error"]}
        for r in failures
    ]
    return records, summary


def block_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Счётчики блока для карточки пиннинга (§8.5)."""
    total = len(records)
    s1 = sum(1 for r in records if r["scenario"] == "S1")
    s2 = total - s1
    multi = sum(1 for r in records if r["n_tool_calls"] >= 2)
    turns = sum(r["n_turns"] for r in records)
    calls = sum(r["n_tool_calls"] for r in records)
    passed = sum(1 for r in records if r["verdict_passed"])
    return {
        "records": total,
        "s1": s1,
        "s2": s2,
        "s1_share": round(s1 / total, 6) if total else 0.0,
        "s2_share": round(s2 / total, 6) if total else 0.0,
        "multi_action_records": multi,
        "multi_action_share": round(multi / total, 6) if total else 0.0,
        "turns": turns,
        "tool_calls": calls,
        "verdict_passed": passed,
    }


def write_block(records: list[dict[str, Any]], out: Path) -> str:
    """Пишет jsonl байт-детерминированно; возвращает sha256 содержимого."""
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(r, ensure_ascii=False, sort_keys=True) for r in records]
    data = ("\n".join(lines) + "\n").encode("utf-8") if lines else b""
    out.write_bytes(data)
    return sha256_bytes(data)


# ── Валидация стражем C-044 ────────────────────────────────────────────────


def validate_block(path: Path, *, quiet: bool = True) -> tuple[bool, dict[str, Any]]:
    """Прогон стража ``tools/check_sft_structure.py`` по блоку (§8.5, C-044).

    Требования: 0 ``unclosed_think``, 0 ``tool_call_in_think``,
    ``unfinished_tool_call`` ≤ 5 %. Фейл (в т.ч. сатурация/нечитаемость) —
    ``ok=False`` (в CLI → ненулевой exit).
    """
    import subprocess

    path = Path(path)
    with tempfile.TemporaryDirectory(prefix="sft-env-guard-") as td:
        report_path = Path(td) / "report.json"
        cmd = [
            sys.executable, str(ROOT / "tools" / "check_sft_structure.py"),
            "--input", str(path), "--strict", "--quiet", "--json", str(report_path),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if not report_path.is_file():
            return False, {
                "verdict": "cannot-check",
                "reason": (proc.stderr or proc.stdout or "нет отчёта")[:400],
                "exit_code": proc.returncode,
            }
        report = json.loads(report_path.read_text(encoding="utf-8"))

    classes = report.get("classes", {})
    unclosed = int(classes.get("unclosed_think", {}).get("count", 0))
    in_think = int(classes.get("tool_call_in_think", {}).get("count", 0))
    no_answer = int(classes.get("no_answer", {}).get("count", 0))
    unfinished = report.get("unfinished_tool_call", {})
    unfinished_share = float(unfinished.get("share", 0.0))
    ok = (
        proc.returncode == 0
        and not report.get("saturated", False)
        and unclosed == 0
        and in_think == 0
        and no_answer == 0
        and unfinished_share <= UNFINISHED_MAX
    )
    summary = {
        "verdict": report.get("verdict"),
        "exit_code": proc.returncode,
        "records": report.get("records", 0),
        "assistant_messages": report.get("assistant_messages", 0),
        "unclosed_think": unclosed,
        "tool_call_in_think": in_think,
        "no_answer": no_answer,
        "unfinished_tool_call": unfinished.get("count", 0),
        "unfinished_tool_call_share": unfinished_share,
        "threshold": UNFINISHED_MAX,
        "ok": ok,
    }
    return ok, summary


# ── CLI ────────────────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="env.sft_block",
        description="E-2: генератор env-блока SFT-микса (SFT-STAGE §8.5)",
    )
    p.add_argument("--n-s1", type=int, default=70, help="число траекторий S1 (ремонт)")
    p.add_argument("--n-s2", type=int, default=30, help="число траекторий S2 (чистый прогон)")
    p.add_argument("--seed", type=int, required=True, help="базовый seed генерации")
    p.add_argument("--out", type=Path, required=True, help="выходной jsonl")
    p.add_argument(
        "--full-case-root", type=Path, default=None,
        help="корень чистого кейса (по умолчанию — корень репозитория)",
    )
    p.add_argument("--s1-level", choices=("L0", "L1"), default="L0")
    p.add_argument("--bin", default=None, help="бинарь arch-ml (иначе ENV_ARCH_ML_BIN/PATH)")
    p.add_argument(
        "--workers", type=int, default=1,
        help="число воркеров multiprocessing.Pool (дефолт 1 — последовательно); "
             "шарды pos %% workers, результат инвариантен по workers",
    )
    p.add_argument("--validate", action="store_true", help="прогнать стража C-044 по блоку")
    p.add_argument("--summary-out", type=Path, default=None, help="куда записать сводку (JSON)")
    p.add_argument("--quiet", action="store_true")
    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    records, summary = generate_block(
        n_s1=args.n_s1, n_s2=args.n_s2, seed=args.seed,
        full_case_root=args.full_case_root, s1_level=args.s1_level, bin=args.bin,
        workers=args.workers,
    )
    sha = write_block(records, args.out)
    summary["out"] = str(args.out)
    summary["sha256"] = sha

    if args.validate:
        ok, guard = validate_block(args.out)
        summary["guard"] = guard
        if not ok:
            if not args.quiet:
                print(json.dumps(summary, ensure_ascii=False, indent=2))
            print("error: страж C-044 забраковал блок", file=sys.stderr)
            return 1

    if args.summary_out:
        args.summary_out.parent.mkdir(parents=True, exist_ok=True)
        args.summary_out.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    if not args.quiet:
        print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
