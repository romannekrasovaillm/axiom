#!/usr/bin/env python3
"""C-042 — поведенческий страж пиннинга бэкенда и политики точности (ADR-010).

Проверяет **фактическую конфигурацию JAX**, а не текст: импортирует
``net/tests/conftest.py`` (единственная точка пиннинга), применяет пиннинг в
своём процессе и читает обратно ``jax.config.jax_default_matmul_precision`` и
``jax.devices()``.  Правило-близнец «в conftest есть строка
``jax_default_matmul_precision``» было бы documentary и запрещено ADR-011:
оно зеленеет и когда настройку переименовали, и когда её перекрыли позже.

ADR-013 расширяет правило: в гейтовом профиле проверяется ещё и **фактическое
наличие** ``--xla_gpu_deterministic_ops`` в ``XLA_FLAGS`` процесса прогона
(runtime-факт, а не строка в conftest) — без него GPU-прогон не гарантирует
побитового совпадения артефактов, и A5 («повторный прогон → тот же манифест»)
держится случайно.  Наличие флага в окружении — доказательство, что XLA его
применил: прогон, который флаг не принял (опечатка в имени), умирает на разборе
``XLA_FLAGS`` при создании клиента, поэтому успешная инициализация устройства
и флаг в переменной не могут разойтись.

Два режима:

* по умолчанию — страж пиннинга (правило C-042).  Печатает политику точности,
  список устройств, профиль (gate/local) и режим детерминизма.  Код 0 — PASS,
  1 — FAIL.  В гейтовом профиле отсутствие CUDA-устройства — FAIL: приёмочный
  вердикт A4 на CPU не производится (ADR-010, решение 2); отсутствие флага
  детерминизма — тоже FAIL (ADR-013, решение 3).
* ``--measure-criterion2`` — замер критерия 2 (паритет рекуррентной и
  чанкованной форм KDA, порог 1e-4) **под пиннингом**: подтверждает, что
  расхождение 3.37e-04 было политикой точности, а не дефектом алгоритма.
  Код 0 — в пределах порога, 1 — порог нарушен, 2 — SKIP (нет CUDA-устройства,
  критерий на доступном бэкенде не измерен; ложный PASS не выдаётся).

Запуск (из каталога кейса)::

    python3 tools/check_precision_pinning.py
    python3 tools/check_precision_pinning.py --measure-criterion2
    python3 tools/check_precision_pinning.py --profile local

Интерпретатор: пиннинг применяется к JAX, поэтому страж сам находит
интерпретатор, в котором ``import jax`` проходит (``--python``, ``KK_PYTHON``,
``NET_JAX_PYTHON``, ``VIRTUAL_ENV``, ``~/venv-axiom/bin/python``, системный), и
перезапускает себя в нём.  Если такого интерпретатора нет — FAIL: страж,
не способный проверить поведение, не имеет права зеленеть.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import subprocess
import sys
import types
from pathlib import Path

SCRIPT = Path(__file__).resolve()
CASE_DIR = SCRIPT.parents[1]
DEFAULT_CONFTEST = CASE_DIR / "net" / "tests" / "conftest.py"

#: Страж импортирует модули каталога кейса (``net/tests/conftest.py`` →
#: ``net/config.py``), а ``command_succeeds``-правило исполняется гейтом
#: **внутри проверяемого workspace**.  Байткод-кеш рядом с исходником (``.pyc``)
#: маршалит абсолютный путь исходника (``co_filename``), поэтому импорт в
#: workspace делает байты workspace зависящими от его пути — и ``tree_sha256``
#: (``env/util.py``, хеш манифеста A5) расходится у двух идентичных прогонов в
#: разные каталоги.  Верификатор не имеет права менять артефакт, который
#: проверяет: ни в своём процессе, ни в перезапущенном (``reexec`` наследует
#: окружение, поэтому переменная, а не только ``sys``-флаг — её видят и
#: подпроцессы вроде ``can_import_jax``).
sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

#: Ожидаемая политика точности — из решения ADR-010 (не из conftest: иначе
#: ослабленный conftest сам себе выписывал бы ожидание).
EXPECTED_PRECISION = "highest"

#: Порог критерия 2 (MODEL-L3-SKELETON.md §6, ADR-010: не меняется).
CRITERION2_THRESHOLD = 1e-4

#: Маркер профиля — тот же, что читает conftest (выбор бэкенда NET_JAX_BACKEND
#: стражу не нужен: его применяет conftest, а страж читает результат).
GATE_ENV = "NET_GATE_PROFILE"

#: Флаг детерминизма XLA — из решения ADR-013 (не из conftest: иначе
#: ослабленный conftest сам себе выписывал бы ожидание).
EXPECTED_DETERMINISM_FLAG = "--xla_gpu_deterministic_ops"

#: Переменная, в которой флаг обязан быть у процесса прогона (XLA разбирает её
#: один раз, при создании клиента).
XLA_FLAGS_ENV = "XLA_FLAGS"

#: Значения маркера, означающие локальный профиль (остальные — гейтовый).
_LOCAL_VALUES = {"0", "false", "no", "off", "local"}

#: Служебные переменные перезапуска (защита от петли).
CHILD_ENV = "_KK_PRECISION_PINNING_CHILD"
PARENT_ENV = "_KK_PRECISION_PINNING_PARENT"

#: Переменные с путём к интерпретатору и кандидаты по умолчанию.
PYTHON_ENVS = ("KK_PYTHON", "NET_JAX_PYTHON")
PYTHON_CANDIDATES = ("~/venv-axiom/bin/python", "~/.venv/bin/python", "python3", "python")

EXIT_PASS = 0
EXIT_FAIL = 1
EXIT_SKIP = 2


class GuardError(Exception):
    """Конфигурация пиннинга не читается или не применяется."""


# ---------------------------------------------------------------------------
# Выбор интерпретатора с JAX
# ---------------------------------------------------------------------------


def jax_available_here() -> bool:
    """Есть ли jax в текущем интерпретаторе — без его импорта.

    Импортировать jax здесь нельзя: плагин CUDA проверяет библиотеки в момент
    импорта и при неудаче навсегда падает на CPU.  Импорт делает conftest —
    после предзагрузки CUDA-библиотек венва.
    """
    try:
        return importlib.util.find_spec("jax") is not None
    except (ImportError, ValueError):
        return False


def can_import_jax(python: str, timeout: int = 600) -> bool:
    """Пробный запуск: ``import jax`` в указанном интерпретаторе."""
    try:
        proc = subprocess.run(
            [python, "-c", "import jax"],
            capture_output=True,
            timeout=timeout,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return proc.returncode == 0


def python_candidates() -> list[str]:
    candidates: list[str] = []
    for name in PYTHON_ENVS:
        value = (os.environ.get(name) or "").strip()
        if value:
            candidates.append(value)
    virtual_env = (os.environ.get("VIRTUAL_ENV") or "").strip()
    if virtual_env:
        candidates.append(str(Path(virtual_env) / "bin" / "python"))
    candidates.append(sys.executable)
    candidates.extend(str(Path(p).expanduser()) for p in PYTHON_CANDIDATES)
    seen: set[str] = set()
    unique: list[str] = []
    for candidate in candidates:
        if candidate not in seen:
            seen.add(candidate)
            unique.append(candidate)
    return unique


def find_jax_python(explicit: str | None = None) -> str | None:
    """Интерпретатор, в котором ``import jax`` проходит (``None`` — не найден)."""
    if explicit:
        return explicit if can_import_jax(explicit) else None
    if jax_available_here():
        return sys.executable
    for candidate in python_candidates():
        if candidate == sys.executable:
            continue
        if can_import_jax(candidate):
            return candidate
    return None


def reexec(python: str, argv: list[str]) -> int:
    """Перезапуск стража в найденном интерпретаторе (код возврата — его)."""
    env = dict(os.environ)
    env[CHILD_ENV] = "1"
    env[PARENT_ENV] = sys.executable
    try:
        proc = subprocess.run([python, str(SCRIPT), *argv], env=env)
    except OSError as exc:
        print(f"ошибка: не удалось запустить интерпретатор {python}: {exc}", file=sys.stderr)
        return EXIT_FAIL
    return proc.returncode


# ---------------------------------------------------------------------------
# Загрузка conftest сети
# ---------------------------------------------------------------------------


def ensure_importable_pytest() -> bool:
    """Гарантирует импортируемость ``pytest`` для загрузки conftest.

    Стражу нужен только пиннинг; если pytest в интерпретаторе нет, фикстуры
    подменяются минимальной заглушкой — иначе проверка поведения подменялась
    бы проверкой состава окружения.  Возвращает True, если pytest настоящий.
    """
    try:
        import pytest  # noqa: F401
        return True
    except ImportError:
        stub = types.ModuleType("pytest")

        def fixture(*args, **kwargs):
            if args and callable(args[0]) and not kwargs:
                return args[0]
            return lambda func: func

        stub.fixture = fixture
        sys.modules["pytest"] = stub
        return False


def load_conftest(path: Path):
    """Импортирует conftest сети по пути (побочный эффект — применение пиннинга).

    Из conftest стражу нужны ровно две вещи: объявленная политика
    (``PINNED_MATMUL_PRECISION``) и точка входа ``apply_precision_pinning()``.
    Устройства и профиль страж читает сам — иначе ослабленный conftest мог бы
    ослабить и проверку.
    """
    if not path.is_file():
        raise GuardError(f"conftest сети не найден: {path}")
    ensure_importable_pytest()
    spec = importlib.util.spec_from_file_location("kk_net_tests_conftest", path)
    if spec is None or spec.loader is None:
        raise GuardError(f"conftest не импортируется как модуль: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:  # пиннинг не применился — это и есть FAIL стража
        raise GuardError(f"применение conftest не удалось ({path}): {exc!r}") from exc
    return module


# ---------------------------------------------------------------------------
# Профиль, устройства, политика — из runtime, а не из текста
# ---------------------------------------------------------------------------


def profile_name() -> str:
    """Профиль прогона: gate требует CUDA-устройство, local — нет.

    Fail-safe: любое непустое значение, кроме явного local-маркера, считается
    гейтовым (опечатка делает прогон строже, а не мягче).
    """
    raw = os.environ.get(GATE_ENV)
    if raw is None or not raw.strip():
        return "local"
    return "local" if raw.strip().lower() in _LOCAL_VALUES else "gate"


def jax_module():
    """jax текущего процесса.

    Импортируется только после ``load_conftest``: conftest предзагружает
    CUDA-библиотеки венва, и импорт раньше предзагрузки навсегда увёл бы
    плагин на CPU.
    """
    import jax

    return jax


def describe_devices() -> list[str]:
    """``jax.devices()`` как список ``repr`` (``CudaDevice(id=0)``), не падает."""
    try:
        return [repr(device) for device in jax_module().devices()]
    except Exception as exc:
        return [f"<недоступны: {exc}>"]


def gpu_devices() -> list[str]:
    """CUDA-устройства текущего бэкенда (пусто на CPU)."""
    try:
        return [repr(d) for d in jax_module().devices() if d.platform == "gpu"]
    except Exception:
        return []


def precision_policy() -> object:
    """Фактическая политика точности процесса (None — не задана)."""
    return jax_module().config.jax_default_matmul_precision


def xla_flags_value() -> str:
    """Сырое значение ``XLA_FLAGS`` проверяемого процесса (пусто — не задана)."""
    return os.environ.get(XLA_FLAGS_ENV, "")


def determinism_in_env() -> bool:
    """Есть ли флаг детерминизма в ``XLA_FLAGS`` **этого** процесса (ADR-013).

    Сопоставляется имя флага, а не токен целиком: XLA принимает и
    ``--xla_gpu_deterministic_ops``, и ``--xla_gpu_deterministic_ops=true``.
    """
    return any(
        token.split("=", 1)[0] == EXPECTED_DETERMINISM_FLAG
        for token in xla_flags_value().split()
        if token
    )


def determinism_mode() -> str:
    """Режим детерминизма процесса: on/off + откуда флаг взялся."""
    if not determinism_in_env():
        return "off"
    return "on (gate)" if profile_name() == "gate" else "on (из окружения)"


def _header(conftest, conftest_path: Path, python: str, real_pytest: bool) -> list[str]:
    parent = os.environ.get(PARENT_ENV)
    interpreter = f"{python}"
    if parent and parent != python:
        interpreter += f" (переход из {parent})"
    preloaded = getattr(conftest, "PRELOADED_CUDA_LIBS", None)
    return [
        "страж пиннинга точности (ADR-010, C-042)",
        f"  conftest: {conftest_path}",
        f"  интерпретатор: {interpreter}",
        f"  pytest: {'есть' if real_pytest else 'нет (заглушка для импорта conftest)'}",
        f"  профиль: {profile_name()} ({GATE_ENV}={os.environ.get(GATE_ENV, '<unset>')})",
        f"  политика точности: jax_default_matmul_precision={precision_policy()}",
        f"  устройства JAX: {describe_devices()}",
        f"  детерминизм XLA: {determinism_mode()} "
        f"({XLA_FLAGS_ENV}={xla_flags_value() or '<unset>'!r})",
        f"  предзагружено CUDA-библиотек: {len(preloaded) if preloaded is not None else '—'}",
    ]


def check_pinning(conftest) -> list[str]:
    """Применяет пиннинг и проверяет фактическую конфигурацию. Список нарушений."""
    problems: list[str] = []
    declared = getattr(conftest, "PINNED_MATMUL_PRECISION", None)
    if declared != EXPECTED_PRECISION:
        problems.append(
            f"conftest объявляет политику {declared!r}, ADR-010 требует "
            f"{EXPECTED_PRECISION!r}"
        )

    # ADR-013, решение 3: в гейтовом профиле — фактическое наличие флага
    # детерминизма в окружении прогона. Проверяется до раннего выхода ниже:
    # отсутствие флага и отсутствие точки входа пиннинга — разные нарушения,
    # и вердикт обязан назвать оба.
    if profile_name() == "gate" and not determinism_in_env():
        problems.append(
            f"гейтовый профиль требует {EXPECTED_DETERMINISM_FLAG} в "
            f"{XLA_FLAGS_ENV}: без него GPU-прогон не гарантирует побитового "
            f"совпадения артефактов (A5; ADR-013, решение 1/3); фактически "
            f"{XLA_FLAGS_ENV}={xla_flags_value() or '<unset>'!r}"
        )

    apply = getattr(conftest, "apply_precision_pinning", None)
    if apply is None:
        problems.append(
            "conftest не предоставляет точку входа apply_precision_pinning() — "
            "проверить применение пиннинга нечем"
        )
        return problems
    try:
        apply()
    except Exception as exc:
        problems.append(f"применение пиннинга упало: {exc!r}")
        return problems

    effective = precision_policy()
    if effective != EXPECTED_PRECISION:
        problems.append(
            "пиннинг не действует: jax_default_matmul_precision="
            f"{effective!r}, ожидалось {EXPECTED_PRECISION!r}"
        )

    if profile_name() == "gate" and not gpu_devices():
        problems.append(
            "гейтовый профиль требует CUDA-устройство, jax.devices()="
            f"{describe_devices()} (ADR-010, решение 2/3)"
        )
    return problems


def run_pinning_guard(conftest, conftest_path: Path, python: str, real_pytest: bool) -> int:
    lines = _header(conftest, conftest_path, python, real_pytest)
    problems = check_pinning(conftest)

    if problems:
        lines.append("вердикт: FAIL")
        for problem in problems:
            lines.append(f"  - {problem}")
        if profile_name() == "local" and not gpu_devices():
            lines.append(
                "  (предварительный прогон: профиль local не требует GPU; "
                "гейтовый профиль — NET_GATE_PROFILE=1 — потребует)"
            )
        print("\n".join(lines))
        return EXIT_FAIL

    lines.append("вердикт: PASS")
    if profile_name() == "local" and not gpu_devices():
        lines.append(
            "  предварительный прогон: GPU нет, профиль local его не требует"
        )
    print("\n".join(lines))
    return EXIT_PASS


# ---------------------------------------------------------------------------
# Режим 2: критерий 2 под пиннингом
# ---------------------------------------------------------------------------


def measure_criterion2(conftest, case_dir: Path) -> int:
    """Замер критерия 2 той же функцией, что и тест (net/tests/test_02)."""
    lines = [
        "критерий 2 (паритет рекуррентной и чанкованной форм KDA) под пиннингом",
        f"  политика точности: jax_default_matmul_precision={precision_policy()}",
        f"  устройства JAX: {describe_devices()}",
        f"  профиль: {profile_name()}",
        f"  порог: {CRITERION2_THRESHOLD:.0e}",
    ]

    gpus = gpu_devices()
    if not gpus:
        lines.append(
            "вердикт: SKIP — нет CUDA-устройства: критерий 2 на доступном "
            "бэкенде не измерен (предварительный прогон, ложный PASS не выдаётся)"
        )
        print("\n".join(lines))
        return EXIT_SKIP

    make_config = getattr(conftest, "small_config", None)
    if make_config is None:
        lines.append(
            "вердикт: FAIL — conftest не предоставляет small_config(): "
            "замерить критерий 2 той же конфигурацией, что сьют, нечем"
        )
        print("\n".join(lines))
        return EXIT_FAIL

    sys.path.insert(0, str(case_dir))
    sys.path.insert(0, str(case_dir / "net" / "tests"))
    import test_02_kda_parity as criterion2  # та же функция _run, что в сьюте

    cfg = make_config()
    worst = 0.0
    try:
        for seq_len, chunk in ((32, 1), (32, 8), (32, 16), (64, 16)):
            delta = criterion2._run(cfg, seq_len=seq_len, chunk=chunk)
            worst = max(worst, delta)
            lines.append(f"  T={seq_len} chunk={chunk}: max|rec-chunked|={delta:.3e}")
    except Exception as exc:
        lines.append(f"вердикт: FAIL — замер не выполнен: {exc!r}")
        print("\n".join(lines))
        return EXIT_FAIL

    if worst <= CRITERION2_THRESHOLD:
        lines.append(
            f"вердикт: PASS — максимальное расхождение {worst:.3e} "
            f"в пределах порога {CRITERION2_THRESHOLD:.0e} (бэкенд: {gpus[0]})"
        )
        print("\n".join(lines))
        return EXIT_PASS

    lines.append(
        f"вердикт: FAIL — расхождение {worst:.3e} превышает порог "
        f"{CRITERION2_THRESHOLD:.0e}: паритет форм KDA нарушен (ADR-010)"
    )
    print("\n".join(lines))
    return EXIT_FAIL


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="C-042: поведенческий страж пиннинга бэкенда и политики "
        "точности матмулов (ADR-010).",
    )
    parser.add_argument(
        "--conftest",
        default=str(DEFAULT_CONFTEST),
        help="путь к conftest сети (по умолчанию net/tests/conftest.py кейса)",
    )
    parser.add_argument(
        "--profile",
        choices=("gate", "local"),
        default=None,
        help="профиль прогона: gate требует CUDA-устройство, local — нет "
        "(по умолчанию из NET_GATE_PROFILE)",
    )
    parser.add_argument(
        "--measure-criterion2",
        action="store_true",
        help="замерить критерий 2 под пиннингом вместо проверки пиннинга",
    )
    parser.add_argument(
        "--python",
        default=None,
        help="интерпретатор с jax (по умолчанию подбирается автоматически)",
    )
    parser.add_argument(
        "--case-dir",
        default=str(CASE_DIR),
        help="каталог кейса (для импорта net/ и net/tests/)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    case_dir = Path(args.case_dir)
    conftest_path = Path(args.conftest)

    # Профиль ставится в окружение до импорта conftest: conftest читает тот же
    # маркер, поэтому страж и сьют разрешают профиль одинаково.
    if args.profile:
        os.environ[GATE_ENV] = "1" if args.profile == "gate" else "0"

    current_argv = list(sys.argv[1:] if argv is None else argv)

    # Явно указанный интерпретатор — приоритет: пиннинг применяется к JAX.
    if args.python:
        if not can_import_jax(args.python):
            print(
                f"вердикт: FAIL — указанный интерпретатор не импортирует jax: "
                f"{args.python}",
                file=sys.stderr,
            )
            return EXIT_FAIL
        if Path(args.python).resolve() != Path(sys.executable).resolve():
            return reexec(args.python, current_argv)

    # Автоподбор, если в текущем интерпретаторе jax нет: без jax проверять нечего.
    if not jax_available_here():
        if os.environ.get(CHILD_ENV) == "1":
            print(
                "вердикт: FAIL — jax недоступен в этом интерпретаторе "
                f"({sys.executable}), и перезапуск уже выполнялся",
                file=sys.stderr,
            )
            return EXIT_FAIL
        python = find_jax_python(None)
        if python is None:
            print(
                "страж пиннинга точности (ADR-010, C-042)\n"
                "вердикт: FAIL — не найден интерпретатор, в котором проходит "
                "import jax; проверить пиннинг нечем.\n"
                "  кандидаты: " + ", ".join(python_candidates()) + "\n"
                "  подсказка: укажите --python <путь> или KK_PYTHON",
                file=sys.stderr,
            )
            return EXIT_FAIL
        return reexec(python, current_argv)

    try:
        real_pytest = ensure_importable_pytest()
        # conftest импортирует net.config: каталог кейса должен быть на пути
        # (при запуске через pytest это cwd, при запуске скриптом — нет).
        sys.path.insert(0, str(case_dir))
        conftest = load_conftest(conftest_path)
    except GuardError as exc:
        print(f"страж пиннинга точности (ADR-010, C-042)\nвердикт: FAIL — {exc}")
        return EXIT_FAIL

    if args.measure_criterion2:
        return measure_criterion2(conftest, case_dir)
    return run_pinning_guard(conftest, conftest_path, sys.executable, real_pytest)


if __name__ == "__main__":
    raise SystemExit(main())
