#!/usr/bin/env python3
"""ADR-041 — префлайт-дисциплина памяти JAX-прогонов на совмещённом стенде.

Инцидент 08.10.2026 (повторение 26.09.2026): ночной ``nsys``-прогон l3-full на
GB10 стартовал без ``XLA_PYTHON_CLIENT_MEM_FRACTION``, JAX зарезервировал
дефолтные ~75 % устройства, совмещённый стенд ушёл в global OOM (34 события
``oom-kill``) и ребутнулся. Словесное правило не переживает смену контекста —
поэтому дисциплина механизирована здесь.

Модуль даёт две вещи (обе — **до** инициализации рантайма JAX):

* :func:`ensure_mem_fraction` — выставляет ``XLA_PYTHON_CLIENT_MEM_FRACTION``
  (дефолт ``0.5``, перекрывается одноимённой переменной окружения) **до**
  ``import jax``. Каждый JAX-инструмент ``tools/*.py`` вызывает её первой
  строкой исполняемой части; наличие вызова проверяет правило CONSTRAINTS
  (ADR-041 п. 3).
* :func:`preflight_gate` — снимает состояние стенда (``nvidia-smi
  --query-compute-apps`` + ``free -g``) и **fail-closed** отказывает в старте,
  если на устройстве есть ЧУЖИЕ compute-процессы, а явного разрешения владельца
  на совмещение нет. Обёртка :func:`gate_or_exit` для инструментов применяет
  этот отказ **только на распознанном стенде** (маркеры :data:`STAND_MARKERS`:
  ``GB10|Spark|Grace``, ADR-041 п.3 + Amendment 2 п.3) и вызывается **только из
  прогонных путей**; проверяющие инструменты несут лишь
  :func:`ensure_mem_fraction` (ADR-041 п.2) — контрольный контур не должен
  падать из-за чужой нагрузки.

Модуль ``stdlib-only``: ничего не импортирует из ``jax`` (иначе лимит
выставлялся бы слишком поздно), не делает сетевых вызовов и ничего не пишет за
пределы собственного вывода.

Разрешение владельца на совмещение — текстовый файл в лок-каталоге
(``$GB10_LOCK_DIR`` или ``~/gb10-shared/.locks/``). Формат содержимого
свободный, но файл обязан читаться как текст и быть непустым, а имя (или текст)
— явно говорить о разрешении (``allow``/``permit``/``colocat``/``co-run``/
``share``/``совмест``). Нагрузочные маркеры ``*.lock`` (конвенция C-040) — это
не разрешение, они игнорируются.

CLI (проверка состояния стенда вручную)::

    python3 tools/jax_preflight.py --gate          # fail-closed: чужие -> отказ
    python3 tools/jax_preflight.py --gate --json   # машиночитаемое состояние
    python3 tools/jax_preflight.py --lock-dir DIR  # иной каталог разрешений

Код возврата: ``0`` — состояние позволяет стартовать; ненулевой —
отказ/проблема (fail-closed через ``SystemExit``).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Optional

#: Переменная, которой XLA управляет резервом памяти устройства.
ENV_VAR = "XLA_PYTHON_CLIENT_MEM_FRACTION"

#: Дефолт: половина памяти устройства. На совмещённом стенде резерв JAX по
#: умолчанию (~75 %) гарантирует OOM-каскад; 0.5 — безопасный старт, а более
#: тесное совмещение задаётся переменной окружения (ADR-041 п. 1).
DEFAULT_MEM_FRACTION = "0.5"

#: Переключатель префлайт-гейта для инструментов: ``0``/``false``/``off``/``no``
#: — не проверять совмещение вовсе; иное значение (или отсутствие) — проверять
#: штатно. Enforce ограничен стендом GB10 (ADR-041 п.3): **ни одно** значение
#: переменной не расширяет блокировку за пределы GB10 — на дев-ПК чужие
#: процессы остаются предупреждением, а не отказом.
GATE_ENV = "JAX_PREFLIGHT_GATE"

#: Таймаут сенсорных подпроцессов, с.
_SUBPROCESS_TIMEOUT = 15

#: Каталог разрешений/локов по умолчанию (конвенция C-040).
DEFAULT_LOCK_DIR = Path.home() / "gb10-shared" / ".locks"

#: Набор маркеров имени общего стенда (ADR-041 п.3; Amendment 2 п.3). Literal
#: «GB10» хрупок: стенд, чьё ``nvidia-smi``-имя не содержит этой подстроки,
#: получил бы advisory-режим вместо fail-closed — то есть отказ **в сторону
#: разрешения**, худшая сторона ошибки для защиты от OOM (инцидент 08.10).
#: Набор ``GB10|Spark|Grace`` покрывает DGX Spark / GB10 / Grace Blackwell;
#: ложное срабатывание на не-стенде практически исключено (таких имён в контуре
#: нет), а блокировать локальную работу на дев-ПК чужие процессы по-прежнему не
#: могут (имя не совпадает).
STAND_MARKERS = r"GB10|Spark|Grace"

#: Переменная окружения, переопределяющая набор маркеров (регулярное
#: выражение). Пустое значение — дефолтный набор; битый шаблон **не сужает**
#: защиту (остаётся дефолт с предупреждением), иначе опечатка молча отключила
#: бы fail-closed на стенде.
STAND_RE_ENV = "JAX_PREFLIGHT_STAND_RE"

#: Скомпилированный дефолтный шаблон (имя сохранено для совместимости).
SHARED_STAND_RE = re.compile(STAND_MARKERS, re.IGNORECASE)

#: Признак явного разрешения на совмещение (имя файла или его текст).
PERMISSION_RE = re.compile(r"(allow|permit|colocat|co-?run|share|совмест)", re.IGNORECASE)

#: Значения ``JAX_PREFLIGHT_GATE``, выключающие проверку (для читаемости логов).
_SKIP_VALUES = frozenset({"0", "false", "off", "no"})

__all__ = [
    "ENV_VAR",
    "DEFAULT_MEM_FRACTION",
    "DEFAULT_LOCK_DIR",
    "STAND_MARKERS",
    "STAND_RE_ENV",
    "ensure_mem_fraction",
    "preflight_gate",
    "gate_or_exit",
    "is_shared_stand",
    "main",
]


# ---------------------------------------------------------------------------
# Лимит памяти (до import jax)
# ---------------------------------------------------------------------------


def ensure_mem_fraction(default: str = DEFAULT_MEM_FRACTION) -> str:
    """Выставляет ``XLA_PYTHON_CLIENT_MEM_FRACTION``, если он не задан.

    Вызывать ДО ``import jax``: XLA читает переменную в момент инициализации
    рантайма. Значение из окружения неприкосновенно (перекрывает дефолт).
    Возвращает фактическое значение; логирует одной строкой в ``stderr``.
    """
    if not os.environ.get(ENV_VAR):
        os.environ[ENV_VAR] = str(default)
        source = "default"
    else:
        source = "env"
    value = os.environ[ENV_VAR]
    print(
        f"[jax-preflight] {ENV_VAR}={value} (source: {source})",
        file=sys.stderr,
    )
    return value


# ---------------------------------------------------------------------------
# Сенсоры состояния стенда
# ---------------------------------------------------------------------------


def _run(args: list[str]) -> Optional[str]:
    """Запускает сенсор; ``None`` — команда недоступна или не удалась.

    Сеть не используется; подпроцесс ограничен таймаутом, окружение — с
    ``LC_ALL=C`` (стабильный разбор ``free`` независимо от локали).
    """
    env = dict(os.environ)
    env["LC_ALL"] = "C"
    try:
        proc = subprocess.run(
            args,
            capture_output=True,
            text=True,
            timeout=_SUBPROCESS_TIMEOUT,
            env=env,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout


def _parse_compute_apps(csv_text: str) -> list[dict[str, Any]]:
    """Парсит ``pid, process_name, used_memory`` из nvidia-smi в список словарей."""
    procs: list[dict[str, Any]] = []
    for line in csv_text.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2:
            continue
        try:
            pid = int(parts[0])
        except ValueError:
            continue
        name = parts[1] or "<unknown>"
        used_mib = 0
        if len(parts) >= 3:
            digits = re.match(r"(\d+)", parts[2])
            if digits:
                used_mib = int(digits.group(1))
        procs.append({"pid": pid, "name": name, "used_mib": used_mib})
    return procs


def _mem_available_gb() -> Optional[float]:
    """Свободная (доступная) память хоста в ГиБ из ``free -g``; ``None`` — нет данных."""
    out = _run(["free", "-g"])
    if not out:
        return None
    for line in out.splitlines():
        stripped = line.strip()
        if not stripped or ":" not in stripped:
            continue
        label = stripped.split(":", 1)[0].strip().lower()
        if label not in ("mem", "память"):
            continue
        numbers = re.findall(r"-?\d+", stripped.split(":", 1)[1])
        if numbers:
            # Последняя колонка `free` — «available» / «доступно».
            return float(numbers[-1])
    return None


def _gpu_name() -> Optional[str]:
    """Имя первого GPU из nvidia-smi; ``None`` — nvidia-smi недоступен."""
    out = _run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"])
    if not out:
        return None
    first = out.splitlines()[0].strip() if out.strip() else ""
    return first or None


def _stat_ppid(pid: int) -> Optional[int]:
    """Родитель ``pid`` из ``/proc/<pid>/stat``; ``None`` — процесса/поля нет."""
    try:
        data = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    rparen = data.rfind(")")
    if rparen < 0:
        return None
    fields = data[rparen + 2:].split()
    if len(fields) < 2:
        return None
    try:
        return int(fields[1])
    except ValueError:
        return None


def _own_pids(root_pid: Optional[int] = None) -> set[int]:
    """Множество PID, принадлежащих нам: сам процесс, предки и потомки.

    Без ``/proc`` (не-Linux) — только собственный PID. «Чужой» = compute-процесс
    вне этого множества.
    """
    root = os.getpid() if root_pid is None else root_pid
    own: set[int] = {root}
    pid = root
    for _ in range(128):  # предки
        ppid = _stat_ppid(pid)
        if ppid is None or ppid <= 0 or ppid in own:
            break
        own.add(ppid)
        pid = ppid
    try:  # потомки
        children: dict[int, list[int]] = {}
        for entry in os.listdir("/proc"):
            if not entry.isdigit():
                continue
            child = int(entry)
            ppid = _stat_ppid(child)
            if ppid is not None:
                children.setdefault(ppid, []).append(child)
        stack = [root]
        while stack:
            current = stack.pop()
            for child in children.get(current, []):
                if child not in own:
                    own.add(child)
                    stack.append(child)
    except OSError:
        pass
    return own


def _permission_grant(lock_dir: Path) -> Optional[str]:
    """Путь файла-разрешения на совмещение или ``None``.

    Разрешением считается обычный непустой текстовый файл в ``lock_dir``, чьё
    имя ИЛИ содержимое явно говорит о разрешении. Нагрузочные маркеры ``*.lock``
    (C-040) разрешением не являются.
    """
    try:
        if not lock_dir.is_dir():
            return None
    except OSError:  # каталог есть, но недоступен (чужой/сетевой) — не разрешение
        return None
    try:
        entries = sorted(lock_dir.iterdir())
    except OSError:
        return None
    for entry in entries:
        if not entry.is_file() or entry.suffix == ".lock":
            continue
        try:
            text = entry.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if not text.strip():
            continue
        if PERMISSION_RE.search(entry.name) or PERMISSION_RE.search(text):
            return str(entry)
    return None


def _stand_re() -> re.Pattern[str]:
    """Шаблон распознавания стенда: дефолтные маркеры или переопределение.

    Переопределение ``STAND_RE_ENV`` — ради стенда с нестандартным именем;
    битый шаблон не сужает защиту, а откатывается к дефолту с предупреждением
    (молчаливое отключение fail-closed на стенде недопустимо).
    """
    override = os.environ.get(STAND_RE_ENV, "").strip()
    if not override:
        return SHARED_STAND_RE
    try:
        return re.compile(override, re.IGNORECASE)
    except re.error as exc:
        print(
            f"[jax-preflight] предупреждение: {STAND_RE_ENV}={override!r} не "
            f"компилируется ({exc}); используется дефолтный набор маркеров "
            f"{STAND_MARKERS} — сужение защиты запрещено",
            file=sys.stderr,
        )
        return SHARED_STAND_RE


def is_shared_stand(state: dict[str, Any]) -> bool:
    """True, если распознан общий стенд (маркеры ``GB10|Spark|Grace``)."""
    device = state.get("device")
    return bool(device) and bool(_stand_re().search(str(device)))


# ---------------------------------------------------------------------------
# Гейт состояния стенда
# ---------------------------------------------------------------------------


def preflight_gate(
    lock_dir: Optional[str | Path] = None,
    require_lock: bool = True,
    own_pid: Optional[int] = None,
) -> dict[str, Any]:
    """Снимает состояние стенда; fail-closed при чужих процессах без разрешения.

    Возвращает структуру ``{own_pids, foreign_procs, mem_available_gb, ok,
    reason, gpu, device}``. При обнаружении чужих compute-процессов и
    ``require_lock=True`` без файла-разрешения поднимает ``SystemExit``
    (ненулевой код возврата — не ``RuntimeError``). Машина без ``nvidia-smi``
    (CPU/ПК) не роняет вызов: ``ok=True``, ``reason="no-gpu"``.
    """
    lock_path = Path(lock_dir) if lock_dir is not None else DEFAULT_LOCK_DIR
    mem_gb = _mem_available_gb()
    own = _own_pids(own_pid)

    apps_text = _run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader",
        ]
    )
    if apps_text is None:
        return {
            "own_pids": sorted(own),
            "foreign_procs": [],
            "mem_available_gb": mem_gb,
            "ok": True,
            "reason": "no-gpu",
            "gpu": "none",
            "device": None,
        }

    foreign = [p for p in _parse_compute_apps(apps_text) if p["pid"] not in own]
    state: dict[str, Any] = {
        "own_pids": sorted(own),
        "foreign_procs": foreign,
        "mem_available_gb": mem_gb,
        "ok": True,
        "reason": "free",
        "gpu": "present",
        "device": _gpu_name(),
    }
    if not foreign:
        return state

    grant = _permission_grant(lock_path)
    if grant:
        state["reason"] = f"co-run-allowed ({grant})"
        return state
    if not require_lock:
        state["reason"] = "foreign-procs (lock not required)"
        return state

    listed = ", ".join(
        f"{p['name']} (pid {p['pid']}, {p['used_mib']} MiB)" for p in foreign
    )
    raise SystemExit(
        "[jax-preflight] ОТКАЗ: на устройстве чужие compute-процессы: "
        f"{listed}. Совмещение запрещено (AD-7/C-040, ADR-041) без явного "
        f"разрешения владельца — текстовый файл с непустым текстом в {lock_path} "
        "(имя/текст со словом allow|permit|colocat|co-run|share|совмест). "
        "Освободите устройство или положите разрешение и повторите запуск."
    )


def gate_or_exit(
    lock_dir: Optional[str | Path] = None,
    verbose: bool = False,
) -> Optional[dict[str, Any]]:
    """Гейт для инструментов: вызывается перед реальным стартом прогона.

    Режим из ``JAX_PREFLIGHT_GATE``: выключен (``0``/``false``/``off``/``no``) —
    ``None`` (проверка пропущена); любое иное значение (или не задан) — проверка
    штатная. Enforce (``SystemExit``) выполняется **только** на распознанном
    стенде (маркеры :data:`STAND_MARKERS` — ``GB10|Spark|Grace``, ADR-041 п.3 +
    Amendment 2 п.3); на любой другой машине — нет ``nvidia-smi`` или имя GPU не
    из набора — чужие процессы дают предупреждение в ``stderr`` и прогон
    продолжается. Ни одно значение переменной строгость за пределы стенда не
    расширяет: блокировать локальную работу на дев-ПК из-за чужого процесса —
    вредная строгость (ложный FAIL контрольного контура).

    Fail-closed на стенде происходит внутри :func:`preflight_gate`
    (``SystemExit``) при отсутствии разрешения владельца.
    """
    mode = os.environ.get(GATE_ENV, "").strip().lower()
    if mode in _SKIP_VALUES:
        return None

    state = preflight_gate(lock_dir=lock_dir, require_lock=False)
    foreign = state.get("foreign_procs") or []

    if state.get("gpu") == "none":
        print(
            "[jax-preflight] предупреждение: nvidia-smi недоступен — проверка "
            "совмещения пропущена (стенд GB10 не распознан)",
            file=sys.stderr,
        )
        return state
    if not foreign:
        if verbose:
            print(
                f"[jax-preflight] preflight_gate: свободно "
                f"(mem_available_gb={state.get('mem_available_gb')})",
                file=sys.stderr,
            )
        return state
    if is_shared_stand(state):
        # Enforce: поднимет SystemExit, если разрешения владельца нет.
        state = preflight_gate(lock_dir=lock_dir, require_lock=True)
        if verbose:
            print(f"[jax-preflight] preflight_gate: {state['reason']}", file=sys.stderr)
        return state
    print(
        "[jax-preflight] предупреждение: чужие compute-процессы на не-GB10 GPU "
        "(не распознан стенд Spark/Grace) — совмещение разрешено "
        "(enforce только на стенде, ADR-041 п.3)",
        file=sys.stderr,
    )
    return state


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="ADR-041: префлайт памяти JAX-прогонов")
    parser.add_argument(
        "--gate",
        action="store_true",
        help="fail-closed проверка совмещения (чужие процессы -> отказ)",
    )
    parser.add_argument("--lock-dir", default=None, help=f"каталог разрешений (дефолт {DEFAULT_LOCK_DIR})")
    parser.add_argument("--json", action="store_true", help="машиночитаемое состояние")
    args = parser.parse_args(argv)

    ensure_mem_fraction()
    state = preflight_gate(lock_dir=args.lock_dir, require_lock=bool(args.gate))
    if args.json:
        print(json.dumps(state, ensure_ascii=False, indent=2))
    else:
        print(
            f"[jax-preflight] gate: ok={state['ok']} reason={state['reason']} "
            f"gpu={state['gpu']} device={state['device']} "
            f"mem_available_gb={state['mem_available_gb']} "
            f"foreign={len(state['foreign_procs'])}"
        )
    return 0 if state["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
