"""Агентный harness-луп роллаутов RL — интерфейс агента v1 (ENVIRONMENT-V1 §13, E-1).

Эпизод: модель ходит по *workspace-копии* кейса четырьмя инструментами
(``list_files``, ``read_file``, ``edit_file``, ``run_gates``), ход ассистента —
``<think>…</think>``? + ``<tool_call>{"name": ..., "args": {...}}</tool_call>``
(плоский JSON, непрерывность доменного SFT v12), наблюдение — сообщение роли
``tool`` в обёртке ``<tool_response>…</tool_response>``. Завершение — явный
``finish``-ход (правки после него игнорируются; запускается финальный вердикт
через ``env.run.evaluate_run``) ИЛИ исчерпание ``attempts`` ИЛИ
``budget_seconds`` ИЛИ лимит роллаут-токенов.

Генератор — подключаемый LM-адаптер (:class:`LMAdapter`): ``MockLM``
(скриптованные ходы, тесты, CPU) и заготовка ``JaxLM`` (реализация генерации
— ЗА рамками пакета, только интерфейс и TODO-точка). GPU/сеть не используются
(§7): все инструменты — файловые операции над локальной копией, ``run_gates``
— существующий путь верификации ``env``.

Детерминизм (AD-4/AD-11): один seed → байт-идентичная последовательность
наблюдений и байт-идентичный журнал; наблюдения кэпированы и детерминированы.
Журнал несёт ``token_ids``, ``behavior_logprobs``, ``policy_version``,
``reward``, ``assistant_mask`` (модель-токены = 1, промпт/tool/observation = 0)
и sha256-хеши текстов ходов/наблюдений.

Shaping arm ``laguna`` (ADR-027, §13): ошибка парсинга tool_call −0.1; менее
``n_min`` вызовов инструментов («сдался») −0.1; таймаут/бюджет исчерпан — 0.0;
поверх — бинарный вердикт и soft формулы среды v1 без изменений. В arm
``frognano`` префикс Лагуны не применяется.
"""

from __future__ import annotations

import json
import re
import shutil
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional, Protocol, Sequence

from env.util import sha256_text

# ── Константы интерфейса агента v1 (§13) ───────────────────────────────────

TOOL_NAMES: tuple[str, ...] = ("list_files", "read_file", "edit_file", "run_gates")
FINISH_NAME = "finish"
ARMS: tuple[str, ...] = ("laguna", "frognano")

#: Кэп наблюдения ``read_file`` в символах (обрезка с пометкой, §13).
READ_CAP_CHARS = 8000
#: Кэп числа записей ``list_files`` (компактность наблюдения).
LIST_CAP_ENTRIES = 200
#: Дефолтный лимит роллаут-токенов, ADR-022 п.7 (диапазон 8–32k).
DEFAULT_MAX_ROLLOUT_TOKENS = 16384
#: Дефолт «сдался» — менее n_min вызовов инструментов за эпизод (§13).
DEFAULT_N_MIN_TOOL_CALLS = 2
#: Дефолтный таймаут одного вызова ``run_gates`` (сек).
DEFAULT_GATES_TIMEOUT = 120.0
#: Дефолтный таймаут финального вердикта ``evaluate_run`` (сек).
DEFAULT_EVAL_TIMEOUT = 600.0
#: Жёсткий предохранитель от бесконечного цикла (не часть контракта §13).
DEFAULT_MAX_TURNS = 256

PARSE_ERROR_PENALTY = -0.1
GIVE_UP_PENALTY = -0.1
BUDGET_REWARD = 0.0
#: Терминации, трактуемые как «таймаут/бюджет исчерпан» (§13).
_BUDGET_TERMINATIONS = frozenset({"budget", "tokens", "timeout"})

SYSTEM_PROMPT = (
    "Ты — инженер-исполнитель в кейсе axiom. Работай только четырьмя "
    "инструментами над рабочей копией кейса. Ход ассистента: необязательный "
    "<think>…</think> и ровно один <tool_call>{\"name\": ..., \"args\": {...}}"
    "</tool_call> (плоский JSON). Инструменты: list_files(dir?), "
    "read_file(path, offset?, limit?), edit_file(path, old, new), run_gates(scope?). "
    "Заверши работу ходом <tool_call>{\"name\": \"finish\", \"args\": {}}</tool_call>. "
    "Сети нет; только эти инструменты."
)

# ── Ошибки ─────────────────────────────────────────────────────────────────


class ToolTimeout(RuntimeError):
    """Инструмент не уложился в отведённый таймаут."""


class WorkspaceEscape(ValueError):
    """Путь инструмента выходит за пределы workspace-копии."""


# ── Разбор хода ассистента ─────────────────────────────────────────────────

_THINK_RE = re.compile(r"^\s*<think>(.*?)</think>", re.S)
_TOOL_CALL_OPEN = "<tool_call>"
_TOOL_CALL_CLOSE = "</tool_call>"


@dataclass(frozen=True)
class ToolCall:
    name: str
    args: dict[str, Any]
    raw: str  # точный JSON-текст вызова (для хеша журнала)


@dataclass(frozen=True)
class ParsedTurn:
    think: Optional[str]
    call: Optional[ToolCall]
    parse_error: bool
    error: str = ""


def _extract_json_object(s: str, start: int) -> Optional[str]:
    """Сбалансированный объект JSON от ``start`` (учитывает строки и escape).

    Плоский JSON v12 всё же может нести ``{``/``}`` внутри args (например,
    ``edit_file`` правит код), поэтому нежадная регулярка непригодна.
    """
    depth = 0
    in_str = False
    escaped = False
    for i in range(start, len(s)):
        ch = s[i]
        if in_str:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_str = False
        elif ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return s[start : i + 1]
    return None


def parse_assistant_turn(text: str) -> ParsedTurn:
    """Разбирает ход ассистента: think + один плоский tool_call (§13)."""
    think_m = _THINK_RE.match(text)
    think = think_m.group(1).strip() if think_m else None
    open_at = text.find(_TOOL_CALL_OPEN)
    if open_at < 0:
        return ParsedTurn(think, None, True, "tool_call не найден")
    brace_at = text.find("{", open_at + len(_TOOL_CALL_OPEN))
    if brace_at < 0:
        return ParsedTurn(think, None, True, "tool_call: JSON-объект не найден")
    raw = _extract_json_object(text, brace_at)
    if raw is None:
        return ParsedTurn(think, None, True, "tool_call: незакрытый JSON-объект")
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        return ParsedTurn(think, None, True, f"tool_call: невалидный JSON ({exc.msg})")
    if not isinstance(payload, dict):
        return ParsedTurn(think, None, True, "tool_call: JSON должен быть объектом")
    name = payload.get("name")
    if not isinstance(name, str) or not name:
        return ParsedTurn(think, None, True, "tool_call: поле name обязательно")
    args = payload.get("args", {})
    if not isinstance(args, dict):
        return ParsedTurn(think, None, True, "tool_call: args должен быть объектом")
    return ParsedTurn(think, ToolCall(name, args, raw), False)


def wrap_tool_response(body: str) -> str:
    """Наблюдение роли ``tool`` в обёртке v12 (§13)."""
    return f"<tool_response>\n{body}\n</tool_response>"


# ── LM-адаптер ─────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class GenerationResult:
    text: str
    token_ids: list[int]
    behavior_logprobs: list[float]
    policy_version: str


class LMAdapter(Protocol):
    """Подключаемый генератор (реализация генерации — вне пакета)."""

    policy_version: str

    def encode(self, text: str) -> list[int]:
        """Детерминированное кодирование промпта/наблюдения в token_ids."""

    def generate(
        self, messages: list[dict[str, str]], seed: int, max_tokens: int
    ) -> GenerationResult:
        """Один ход ассистента: текст + token_ids + behavior_logprobs."""


def _mock_logprobs(token_ids: Sequence[int]) -> list[float]:
    """Детерминированные behavior-logprobs MockLM (не зависят от времени)."""
    return [round(-0.01 * ((i % 7) + 1), 6) for i in range(len(token_ids))]


class MockLM:
    """Скриптованный LM для тестов: ходы выдаются по списку (CPU, без сети).

    Когда скрипт исчерпан, возвращается ``default_turn`` (по умолчанию —
    ``list_files``), что позволяет моделировать «незавершающийся» эпизод.
    """

    def __init__(
        self,
        script: Sequence[str] = (),
        *,
        policy_version: str = "mock-v1",
        default_turn: Optional[str] = None,
    ) -> None:
        self._script = list(script)
        self._index = 0
        self.policy_version = policy_version
        self._default_turn = default_turn or (
            '<tool_call>{"name": "list_files", "args": {}}</tool_call>'
        )

    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))

    def generate(
        self, messages: list[dict[str, str]], seed: int, max_tokens: int
    ) -> GenerationResult:
        if self._index < len(self._script):
            text = self._script[self._index]
            self._index += 1
        else:
            text = self._default_turn
        token_ids = self.encode(text)
        return GenerationResult(
            text=text,
            token_ids=token_ids,
            behavior_logprobs=_mock_logprobs(token_ids),
            policy_version=self.policy_version,
        )


class JaxLM:
    """Заготовка JAX-адаптера генерации (E-1: реализация — ЗА рамками пакета).

    TODO(E-1): подключить train-aware генератор (TITO, tokenizer v12,
    jax-модель на стенде); вернуть token_ids, behavior_logprobs и
    policy_version из чекпойнта. Генерация в этом пакете не реализуется.
    """

    def __init__(self, policy_version: str = "jax-pending") -> None:
        self.policy_version = policy_version

    def encode(self, text: str) -> list[int]:
        raise NotImplementedError("JaxLM.encode: TODO(E-1) — tokenizer v12 вне пакета")

    def generate(
        self, messages: list[dict[str, str]], seed: int, max_tokens: int
    ) -> GenerationResult:
        raise NotImplementedError(
            "JaxLM.generate: TODO(E-1) — train-aware генерация вне пакета (E-1)"
        )


# ── Инструменты над workspace-копией ───────────────────────────────────────


@dataclass(frozen=True)
class ToolOutcome:
    body: str
    timed_out: bool = False


class WorkspaceTools:
    """Четыре инструмента v1 поверх workspace-КОПИИ кейса (сети нет, §7)."""

    def __init__(
        self,
        root: Path,
        verifier: "Verifier",
        *,
        read_cap: int = READ_CAP_CHARS,
        list_cap: int = LIST_CAP_ENTRIES,
        gates_timeout: float = DEFAULT_GATES_TIMEOUT,
    ) -> None:
        self.root = Path(root).resolve()
        self.verifier = verifier
        self.read_cap = read_cap
        self.list_cap = list_cap
        self.gates_timeout = gates_timeout

    # -- путь ---------------------------------------------------------------

    def _resolve(self, path: str) -> Path:
        if not isinstance(path, str) or not path:
            raise WorkspaceEscape("path обязателен")
        candidate = Path(path)
        if candidate.is_absolute():
            raise WorkspaceEscape(f"абсолютный путь запрещён: {path}")
        resolved = (self.root / candidate).resolve()
        if resolved != self.root and self.root not in resolved.parents:
            raise WorkspaceEscape(f"путь вне workspace: {path}")
        return resolved

    # -- инструменты --------------------------------------------------------

    def list_files(self, dir: str = ".") -> str:
        base = self._resolve(dir)
        if not base.is_dir():
            return f"error: не каталог: {dir}"
        entries: list[tuple[str, int]] = []
        for p in sorted(base.rglob("*")):
            if p.is_file():
                rel = p.relative_to(self.root).as_posix()
                entries.append((rel, p.stat().st_size))
        lines = [f"{rel}\t{size}" for rel, size in entries[: self.list_cap]]
        if len(entries) > self.list_cap:
            lines.append(f"...[обрезано: {len(entries) - self.list_cap} записей]")
        if not lines:
            return "(пусто)"
        return "\n".join(lines)

    def read_file(self, path: str, offset: int = 0, limit: Optional[int] = None) -> str:
        """Содержимое файла; ``offset``/``limit`` — в символах, кэп — read_cap."""
        p = self._resolve(path)
        if not p.is_file():
            return f"error: не файл: {path}"
        text = p.read_text(encoding="utf-8", errors="replace")
        start = max(0, int(offset))
        chunk = text[start:] if limit is None else text[start : start + max(0, int(limit))]
        truncated = False
        if len(chunk) > self.read_cap:
            chunk = chunk[: self.read_cap]
            truncated = True
        if truncated:
            chunk += f"\n...[обрезано: показано {len(chunk)} из {len(text)} символов]"
        return chunk

    def edit_file(self, path: str, old: str, new: str) -> str:
        p = self._resolve(path)
        if not p.is_file():
            return f"error: не файл: {path}"
        if not isinstance(old, str) or old == "":
            return "error: old должен быть непустой строкой"
        text = p.read_text(encoding="utf-8")
        count = text.count(old)
        if count != 1:
            return f"error: ожидалось ровно 1 совпадение, найдено {count}"
        p.write_text(text.replace(old, new, 1), encoding="utf-8")
        return "ok"

    def run_gates(self, scope: Optional[str] = None) -> str:
        summary = self.verifier.run_gates(self.root)
        return json.dumps(summary, ensure_ascii=False, sort_keys=True)

    # -- диспетчер ----------------------------------------------------------

    def dispatch(self, call: ToolCall) -> ToolOutcome:
        kwargs = dict(call.args)
        try:
            if call.name == "list_files":
                return ToolOutcome(self.list_files(**_pick(kwargs, ("dir",))))
            if call.name == "read_file":
                return ToolOutcome(
                    self.read_file(**_pick(kwargs, ("path", "offset", "limit")))
                )
            if call.name == "edit_file":
                return ToolOutcome(self.edit_file(**_pick(kwargs, ("path", "old", "new"))))
            if call.name == "run_gates":
                return ToolOutcome(
                    self.run_gates(**_pick(kwargs, ("scope",))),
                )
        except WorkspaceEscape as exc:
            return ToolOutcome(f"error: {exc}")
        except ToolTimeout:
            return ToolOutcome("error: таймаут инструмента run_gates", timed_out=True)
        except (TypeError, ValueError) as exc:  # неверные аргументы инструмента
            return ToolOutcome(f"error: аргументы инструмента: {exc}")
        return ToolOutcome(f"error: неизвестный инструмент: {call.name}")


def _pick(kwargs: dict[str, Any], allowed: Sequence[str]) -> dict[str, Any]:
    unknown = set(kwargs) - set(allowed)
    if unknown:
        raise TypeError(f"неизвестные аргументы: {sorted(unknown)}")
    return {k: kwargs[k] for k in allowed if k in kwargs}


# ── Верификатор (путь ``env``) ─────────────────────────────────────────────


@dataclass(frozen=True)
class EvalResult:
    passed: bool
    reward_total: float
    reward_parts: dict[str, Any]
    violations: list[Any]
    tests_passed: bool


class Verifier(Protocol):
    """Интерфейс верификатора: быстрый вердикт гейтов и финальная оценка."""

    def run_gates(self, workspace: Path) -> dict[str, Any]:
        ...

    def evaluate(self, workspace: Path, spent_tokens: int) -> EvalResult:
        ...


def _run_with_timeout(fn: Callable[[], Any], timeout: float) -> Any:
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(fn)
        try:
            return future.result(timeout=timeout)
        except FuturesTimeoutError as exc:
            raise ToolTimeout(f"таймаут {timeout}с") from exc


class EnvVerifier:
    """Существующий путь верификации ``env``: ``collect_gates`` / ``evaluate_run``.

    ``run_gates`` — компактный JSON-вердикт (passed / violations / tests_passed);
    ``evaluate`` — финальный вердикт + награда формулы среды v1 через
    ``env.run.evaluate_run``. Оба вызова ограничены таймаутом.
    """

    def __init__(
        self,
        task_spec: dict[str, Any],
        base_ws: Path,
        *,
        bin: Optional[str] = None,
        hidden_constraints: Optional[Path] = None,
        gates_timeout: float = DEFAULT_GATES_TIMEOUT,
        eval_timeout: float = DEFAULT_EVAL_TIMEOUT,
    ) -> None:
        self.task_spec = task_spec
        self.base_ws = Path(base_ws)
        self.bin = bin
        self.hidden_constraints = hidden_constraints
        self.gates_timeout = gates_timeout
        self.eval_timeout = eval_timeout

    def run_gates(self, workspace: Path) -> dict[str, Any]:
        from env.verifier import collect_gates, run_tests

        def _call() -> dict[str, Any]:
            gates = collect_gates(
                Path(workspace), self.task_spec, bin=self.bin,
                hidden_constraints=self.hidden_constraints,
            )
            named = {k: v for k, v in gates.items() if not k.startswith("_")}
            violations: list[tuple[str, str]] = []
            for gate in named.values():
                for item in gate.errors:
                    violations.append((item["rule"], item.get("file", "")))
            violations.sort()
            return {
                "passed": all(g.passed for g in named.values()),
                "violations": [{"rule": r, "file": f} for r, f in violations],
                "tests_passed": run_tests(Path(workspace), self.task_spec),
                "gates": {k: g.passed for k, g in sorted(named.items())},
            }

        return _run_with_timeout(_call, self.gates_timeout)

    def evaluate(self, workspace: Path, spent_tokens: int) -> EvalResult:
        from env.run import evaluate_run

        def _call() -> EvalResult:
            rr = evaluate_run(
                self.task_spec, self.base_ws, Path(workspace), spent_tokens,
                bin=self.bin, hidden_constraints=self.hidden_constraints,
            )
            return EvalResult(
                passed=rr.verdict.passed,
                reward_total=rr.reward.total,
                reward_parts=rr.reward.to_manifest_dict(),
                violations=sorted([list(v) for v in rr.verdict.violations]),
                tests_passed=rr.verdict.tests_passed,
            )

        return _run_with_timeout(_call, self.eval_timeout)


# ── Shaping (§13, ADR-027) ─────────────────────────────────────────────────


def prefix_penalty(
    arm: str, *, parse_errors: int, tool_calls: int, n_min: int
) -> float:
    """Префикс Лагуны: parse-error −0.1; «сдался» (< n_min вызовов) −0.1."""
    _check_arm(arm)
    if arm != "laguna":
        return 0.0
    penalty = 0.0
    if parse_errors > 0:
        penalty += PARSE_ERROR_PENALTY
    if tool_calls < n_min:
        penalty += GIVE_UP_PENALTY
    return penalty


def episode_reward(
    base_total: float,
    *,
    arm: str,
    parse_errors: int,
    tool_calls: int,
    termination: str,
    n_min: int,
) -> float:
    """Награда эпизода: формула среды v1 + префикс Лагуны (arm ``laguna``)."""
    _check_arm(arm)
    if arm == "laguna" and termination in _BUDGET_TERMINATIONS:
        return BUDGET_REWARD
    return base_total + prefix_penalty(
        arm, parse_errors=parse_errors, tool_calls=tool_calls, n_min=n_min
    )


def _check_arm(arm: str) -> None:
    if arm not in ARMS:
        raise ValueError(f"arm должен быть одним из {ARMS}, получено {arm!r}")


# ── Журнал роллаута ────────────────────────────────────────────────────────


@dataclass
class _Segment:
    role: str
    kind: str  # "assistant" | "tool" | "prompt"
    text: str
    token_ids: list[int]
    behavior_logprobs: list[float]
    assistant_mask: list[int]
    text_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "kind": self.kind,
            "text": self.text,
            "text_sha256": self.text_sha256,
            "token_ids": list(self.token_ids),
            "behavior_logprobs": list(self.behavior_logprobs),
            "assistant_mask": list(self.assistant_mask),
        }


@dataclass
class AttemptJournal:
    attempt_index: int
    termination: str
    finished: bool
    policy_version: str
    segments: list[_Segment] = field(default_factory=list)
    tool_calls: int = 0
    parse_errors: int = 0
    tool_timeouts: int = 0
    turns: int = 0
    tokens_used: int = 0
    base_reward: float = 0.0
    reward: float = 0.0
    reward_parts: dict[str, Any] = field(default_factory=dict)
    verdict_passed: Optional[bool] = None
    violations: list[Any] = field(default_factory=list)
    tests_passed: Optional[bool] = None
    tool_call_hashes: list[str] = field(default_factory=list)
    observation_hashes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "attempt_index": self.attempt_index,
            "termination": self.termination,
            "finished": self.finished,
            "policy_version": self.policy_version,
            "tool_calls": self.tool_calls,
            "parse_errors": self.parse_errors,
            "tool_timeouts": self.tool_timeouts,
            "turns": self.turns,
            "tokens_used": self.tokens_used,
            "base_reward": self.base_reward,
            "reward": self.reward,
            "reward_parts": self.reward_parts,
            "verdict_passed": self.verdict_passed,
            "tests_passed": self.tests_passed,
            "violations": self.violations,
            "tool_call_hashes": list(self.tool_call_hashes),
            "observation_hashes": list(self.observation_hashes),
            "token_ids": [t for s in self.segments for t in s.token_ids],
            "behavior_logprobs": [
                lp for s in self.segments for lp in s.behavior_logprobs
            ],
            "assistant_mask": [m for s in self.segments for m in s.assistant_mask],
            "messages": [s.to_dict() for s in self.segments],
        }


@dataclass
class EpisodeJournal:
    task_id: str
    seed: int
    arm: str
    policy_version: str
    aggregate: str
    attempts_allowed: int
    attempts_used: int
    final_attempt_index: int
    termination: str
    reward: float
    verdict_passed: Optional[bool]
    attempts: list[AttemptJournal] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "seed": self.seed,
            "arm": self.arm,
            "policy_version": self.policy_version,
            "aggregate": self.aggregate,
            "attempts_allowed": self.attempts_allowed,
            "attempts_used": self.attempts_used,
            "final_attempt_index": self.final_attempt_index,
            "termination": self.termination,
            "reward": self.reward,
            "verdict_passed": self.verdict_passed,
            "attempts": [a.to_dict() for a in self.attempts],
        }

    def to_json(self) -> str:
        """Байт-детерминированная сериализация журнала (sort_keys)."""
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True, indent=2)


# ── Конфигурация эпизода ───────────────────────────────────────────────────


@dataclass
class EpisodeConfig:
    arm: str = "laguna"
    n_min: int = DEFAULT_N_MIN_TOOL_CALLS
    aggregate: str = "last"
    budget_seconds: Optional[float] = None
    max_rollout_tokens: int = DEFAULT_MAX_ROLLOUT_TOKENS
    max_turns: int = DEFAULT_MAX_TURNS
    gates_timeout: float = DEFAULT_GATES_TIMEOUT
    eval_timeout: float = DEFAULT_EVAL_TIMEOUT
    read_cap: int = READ_CAP_CHARS
    list_cap: int = LIST_CAP_ENTRIES

    def __post_init__(self) -> None:
        _check_arm(self.arm)
        if self.aggregate not in ("last", "best"):
            raise ValueError("aggregate должен быть 'last' или 'best'")


# ── Копия workspace ────────────────────────────────────────────────────────


def _ignore_runtime(_dir: str, names: list[str]) -> list[str]:
    return [n for n in names if n == "__pycache__" or n.endswith(".pyc")]


def copy_workspace(src: Path, dst: Path) -> None:
    """Детерминированная копия workspace-кейса (кэши интерпретатора исключены)."""
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst, ignore=_ignore_runtime, symlinks=False)


# ── Эпизод ─────────────────────────────────────────────────────────────────


def _append_masked(
    journal: AttemptJournal,
    lm: LMAdapter,
    role: str,
    kind: str,
    text: str,
    *,
    register_observation: bool,
) -> None:
    """Промпт/наблюдение: токены маскированы (assistant_mask = 0)."""
    token_ids = lm.encode(text)
    journal.segments.append(
        _Segment(
            role=role,
            kind=kind,
            text=text,
            token_ids=token_ids,
            behavior_logprobs=[0.0] * len(token_ids),
            assistant_mask=[0] * len(token_ids),
            text_sha256=sha256_text(text),
        )
    )
    if register_observation:
        journal.observation_hashes.append(sha256_text(text))


def _append_assistant(journal: AttemptJournal, gen: GenerationResult) -> None:
    journal.segments.append(
        _Segment(
            role="assistant",
            kind="assistant",
            text=gen.text,
            token_ids=list(gen.token_ids),
            behavior_logprobs=list(gen.behavior_logprobs),
            assistant_mask=[1] * len(gen.token_ids),
            text_sha256=sha256_text(gen.text),
        )
    )


def _append_tool(journal: AttemptJournal, lm: LMAdapter, body: str) -> None:
    _append_masked(
        journal, lm, "tool", "tool", wrap_tool_response(body),
        register_observation=True,
    )


def run_attempt(
    task_spec: dict[str, Any],
    base_ws: Path,
    workspace: Path,
    lm: LMAdapter,
    verifier: Verifier,
    config: EpisodeConfig,
    *,
    attempt_index: int,
    seed: int,
    clock: Callable[[], float],
) -> AttemptJournal:
    """Один проход эпизода по свежей копии workspace."""
    journal = AttemptJournal(
        attempt_index=attempt_index,
        termination="attempts_exhausted",
        finished=False,
        policy_version=lm.policy_version,
    )
    # Промпт: system + user (маскируется в loss, в хеши наблюдений не входит).
    _append_masked(
        journal, lm, "system", "prompt", SYSTEM_PROMPT, register_observation=False
    )
    _append_masked(
        journal, lm, "user", "prompt", str(task_spec.get("prompt", "")),
        register_observation=False,
    )
    tokens_used = sum(len(s.token_ids) for s in journal.segments)

    tools = WorkspaceTools(
        workspace, verifier,
        read_cap=config.read_cap, list_cap=config.list_cap,
        gates_timeout=config.gates_timeout,
    )
    budget = config.budget_seconds
    if budget is None:
        budget = float(task_spec.get("budget_seconds", 0) or 0) or None
    started = clock()
    messages: list[dict[str, str]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": str(task_spec.get("prompt", ""))},
    ]

    while True:
        if journal.turns >= config.max_turns:
            journal.termination = "turns"
            break
        if budget is not None and (clock() - started) > budget:
            journal.termination = "budget"
            break
        remaining = config.max_rollout_tokens - tokens_used
        if remaining <= 0:
            journal.termination = "tokens"
            break

        gen = lm.generate(messages, seed=seed, max_tokens=max(1, remaining))
        _append_assistant(journal, gen)
        journal.turns += 1
        tokens_used += len(gen.token_ids)
        messages.append({"role": "assistant", "content": gen.text})

        parsed = parse_assistant_turn(gen.text)
        if parsed.parse_error:
            journal.parse_errors += 1
            body = f"error: {parsed.error}"
            _append_tool(journal, lm, body)
            messages.append({"role": "tool", "content": wrap_tool_response(body)})
        elif parsed.call is not None and parsed.call.name == FINISH_NAME:
            journal.finished = True
            journal.termination = "finish"
            journal.tool_call_hashes.append(sha256_text(parsed.call.raw))
            break
        else:
            assert parsed.call is not None
            journal.tool_call_hashes.append(sha256_text(parsed.call.raw))
            outcome = tools.dispatch(parsed.call)
            if outcome.timed_out:
                journal.tool_timeouts += 1
                _append_tool(journal, lm, outcome.body)
                journal.termination = "timeout"
                break
            journal.tool_calls += 1
            _append_tool(journal, lm, outcome.body)
            messages.append(
                {"role": "tool", "content": wrap_tool_response(outcome.body)}
            )

        if tokens_used >= config.max_rollout_tokens:
            journal.termination = "tokens"
            break

    journal.tokens_used = tokens_used
    evaluation = verifier.evaluate(workspace, tokens_used)
    journal.base_reward = evaluation.reward_total
    journal.reward_parts = evaluation.reward_parts
    journal.verdict_passed = evaluation.passed
    journal.violations = evaluation.violations
    journal.tests_passed = evaluation.tests_passed
    journal.reward = episode_reward(
        evaluation.reward_total,
        arm=config.arm,
        parse_errors=journal.parse_errors,
        tool_calls=journal.tool_calls,
        termination=journal.termination,
        n_min=config.n_min,
    )
    return journal


def run_episode(
    task_spec: dict[str, Any],
    workspace_src: Path,
    lm: LMAdapter,
    *,
    verifier: Optional[Verifier] = None,
    config: Optional[EpisodeConfig] = None,
    seed: Optional[int] = None,
    workdir: Optional[Path] = None,
    clock: Callable[[], float] = time.monotonic,
) -> EpisodeJournal:
    """Роллаут-эпизод: попытки по свежим копиям до finish/исчерпания лимитов.

    ``workspace_src`` — оригинал кейса (не мутируется); каждая попытка работает
    на своей копии. Возвращает журнал, пригодный для байт-сравнения (AD-4/AD-11).
    """
    config = config or EpisodeConfig()
    _check_arm(config.arm)
    spec_seed = int(task_spec.get("seed", 0)) if seed is None else int(seed)
    attempts_allowed = max(1, int(task_spec.get("attempts", 1)))
    base_ws = Path(workspace_src)
    if not base_ws.is_dir():
        raise FileNotFoundError(f"workspace-кейс не найден: {base_ws}")
    verifier = verifier or EnvVerifier(
        task_spec, base_ws, gates_timeout=config.gates_timeout, eval_timeout=config.eval_timeout
    )

    own_workdir = workdir is None
    workdir = (
        Path(workdir)
        if workdir is not None
        else Path(tempfile.mkdtemp(prefix="rollout-harness-"))
    )
    workdir.mkdir(parents=True, exist_ok=True)

    attempts: list[AttemptJournal] = []
    for attempt_index in range(attempts_allowed):
        ws = workdir / f"attempt-{attempt_index}"
        copy_workspace(base_ws, ws)
        journal = run_attempt(
            task_spec, base_ws, ws, lm, verifier, config,
            attempt_index=attempt_index, seed=spec_seed, clock=clock,
        )
        attempts.append(journal)
        if journal.finished:
            break

    final = attempts[-1]
    if config.aggregate == "last":
        chosen = final
    else:  # "best"
        chosen = max(attempts, key=lambda a: (a.reward, -a.attempt_index))
    episode = EpisodeJournal(
        task_id=str(task_spec.get("id", "")),
        seed=spec_seed,
        arm=config.arm,
        policy_version=lm.policy_version,
        aggregate=config.aggregate,
        attempts_allowed=attempts_allowed,
        attempts_used=len(attempts),
        final_attempt_index=chosen.attempt_index,
        termination=chosen.termination,
        reward=chosen.reward,
        verdict_passed=chosen.verdict_passed,
        attempts=attempts,
    )
    if own_workdir:
        # Рабочая копия одноразовая: не тащим её за пределами эпизода.
        shutil.rmtree(workdir, ignore_errors=True)
    return episode


__all__ = [
    "ARMS",
    "AttemptJournal",
    "DEFAULT_GATES_TIMEOUT",
    "DEFAULT_MAX_ROLLOUT_TOKENS",
    "DEFAULT_N_MIN_TOOL_CALLS",
    "EnvVerifier",
    "EpisodeConfig",
    "EpisodeJournal",
    "EvalResult",
    "GenerationResult",
    "JaxLM",
    "LMAdapter",
    "MockLM",
    "ParsedTurn",
    "ToolCall",
    "ToolOutcome",
    "ToolTimeout",
    "Verifier",
    "WorkspaceEscape",
    "WorkspaceTools",
    "copy_workspace",
    "episode_reward",
    "parse_assistant_turn",
    "prefix_penalty",
    "run_attempt",
    "run_episode",
    "wrap_tool_response",
]
