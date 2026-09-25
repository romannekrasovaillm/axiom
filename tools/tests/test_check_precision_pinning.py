"""Тесты поведенческого стража пиннинга точности (tools/check_precision_pinning.py, C-042).

Страж обязан проверять **runtime-конфигурацию JAX** после применения
``net/tests/conftest.py``, а не искать строку в файле (ADR-010, ADR-011).
Сценарии:

* положительный: рабочий conftest — PASS, в отчёте видны политика точности,
  список устройств и профиль (gate/local);
* негативный: conftest без пиннинга → FAIL (конфигурация не запиннена);
* негативный: conftest с ослабленной политикой (``high``) → FAIL;
* негативный: conftest без точки входа пиннинга → FAIL (проверять нечем);
* негативный: гейтовый профиль без CUDA-устройства → FAIL, а не SKIP;
* негативный (ADR-013): гейтовый профиль без ``--xla_gpu_deterministic_ops``
  в окружении → FAIL, и нарушение названо в вердикте;
* положительный (ADR-013): гейтовый профиль с пиннингом детерминизма → PASS,
  флаг виден в отчёте;
* локальный профиль флаг не ставит и объявляет это в отчёте;
* критерий 2 под пиннингом на доступном бэкенде в пределах порога; если
  CUDA-устройства нет — SKIP с явной причиной, без ложного PASS.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

CASE_DIR = Path(__file__).resolve().parents[2]
SCRIPT = CASE_DIR / "tools" / "check_precision_pinning.py"

EXIT_PASS = 0
EXIT_FAIL = 1
EXIT_SKIP = 2

#: Флаг детерминизма (ADR-013) — ожидание из решения, не из conftest.
EXPECTED_DETERMINISM_FLAG = "--xla_gpu_deterministic_ops"

#: Негативные фикстуры: conftest-заменители, которые пиннинг не выполняют.
NO_PINNING_CONFTEST = '''"""Негативная фикстура C-042: пиннинг не применяется вовсе."""


def apply_precision_pinning():
    return None
'''

WEAKENED_CONFTEST = '''"""Негативная фикстура C-042: политика ослаблена до high."""
PINNED_MATMUL_PRECISION = "high"


def apply_precision_pinning():
    import jax

    jax.config.update("jax_default_matmul_precision", "high")
    return "high"
'''

NO_ENTRYPOINT_CONFTEST = '''"""Негативная фикстура C-042: точка входа пиннинга отсутствует."""
PINNED_MATMUL_PRECISION = "highest"
'''

#: Негативная фикстура ADR-013: точность запиннена, детерминизм — нет.
#: Гейтовый профиль обязан такую конфигурацию завернуть (C-042).
PRECISION_ONLY_CONFTEST = '''"""Негативная фикстура C-042/ADR-013: флаг детерминизма не ставится."""
PINNED_MATMUL_PRECISION = "highest"


def apply_precision_pinning():
    import jax

    jax.config.update("jax_default_matmul_precision", "highest")
    return "highest"
'''


def _run(*args: str, env_extra: dict[str, str] | None = None,
         timeout: int = 1800) -> subprocess.CompletedProcess[str]:
    """Запуск стража отдельным процессом с чистым профильным окружением.

    ``XLA_FLAGS`` снимается намеренно: предмет проверки — пиннинг, который
    ставит conftest, а не переменная, унаследованная из шелла прогоняющего.
    Иначе зелёный результат зависел бы от окружения — то, против чего и
    ADR-010, и ADR-013.
    """
    env = dict(os.environ)
    for name in ("NET_GATE_PROFILE", "NET_JAX_BACKEND", "_KK_PRECISION_PINNING_CHILD",
                 "XLA_FLAGS"):
        env.pop(name, None)
    if env_extra:
        env.update(env_extra)
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=str(CASE_DIR),
        env=env,
    )


def _write_conftest(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "conftest.py"
    path.write_text(body, encoding="utf-8")
    return path


def _report(result: subprocess.CompletedProcess[str]) -> str:
    return result.stdout + result.stderr


def _problems(result: subprocess.CompletedProcess[str]) -> str:
    """Секция нарушений вердикта (пусто, если страж прошёл)."""
    marker = "вердикт: FAIL"
    return result.stdout.split(marker, 1)[1] if marker in result.stdout else ""


def _tree_files(root: Path) -> set[str]:
    """Относительные пути всех файлов дерева — для проверки «не изменил»."""
    return {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}


def test_guard_passes_with_pinned_conftest() -> None:
    result = _run("--profile", "local")
    assert result.returncode == EXIT_PASS, _report(result)
    out = result.stdout
    assert "jax_default_matmul_precision=highest" in out
    assert "профиль: local" in out
    assert "устройства JAX:" in out
    assert "вердикт: PASS" in out


def test_guard_fails_without_pinning(tmp_path: Path) -> None:
    conftest = _write_conftest(tmp_path, NO_PINNING_CONFTEST)
    result = _run("--profile", "local", "--conftest", str(conftest))
    assert result.returncode == EXIT_FAIL, _report(result)
    assert "пиннинг не действует" in result.stdout
    assert "вердикт: FAIL" in result.stdout


def test_guard_fails_with_weakened_policy(tmp_path: Path) -> None:
    conftest = _write_conftest(tmp_path, WEAKENED_CONFTEST)
    result = _run("--profile", "local", "--conftest", str(conftest))
    assert result.returncode == EXIT_FAIL, _report(result)
    assert "'highest'" in result.stdout
    assert "jax_default_matmul_precision='high'" in result.stdout


def test_guard_fails_without_pinning_entry_point(tmp_path: Path) -> None:
    conftest = _write_conftest(tmp_path, NO_ENTRYPOINT_CONFTEST)
    result = _run("--profile", "local", "--conftest", str(conftest))
    assert result.returncode == EXIT_FAIL, _report(result)
    assert "apply_precision_pinning" in result.stdout


def test_guard_does_not_write_into_the_case_it_inspects(tmp_path: Path) -> None:
    """Страж не оставляет байткод в каталоге, который проверяет (инвариант A5).

    C-042 — правило ``command_succeeds``: гейт исполняет команду **внутри
    проверяемого workspace**.  Импорт ``net/tests/conftest.py`` пишет рядом с
    исходником ``__pycache__/*.pyc``, а байткод-кеш маршалит абсолютный путь
    исходника (``co_filename``) — хеш ``env/util.py:tree_sha256`` тогда зависит
    от пути workspace, и два идентичных прогона в разные каталоги дают разные
    манифесты (A5).  Верификатор не имеет права менять артефакт, который
    проверяет: после прогона в каталоге не должно появиться ни одного файла.
    """
    case = tmp_path / "case"
    conftest = case / "net" / "tests" / "conftest.py"
    conftest.parent.mkdir(parents=True)
    conftest.write_text(NO_PINNING_CONFTEST, encoding="utf-8")
    before = _tree_files(case)

    result = _run(
        "--profile", "local",
        "--case-dir", str(case),
        "--conftest", str(conftest),
    )
    # Страж здесь обязан упасть (пиннинга в фикстуре нет) — важно, что он
    # действительно загрузил conftest, то есть импорт состоялся.
    assert result.returncode == EXIT_FAIL, _report(result)
    assert _tree_files(case) == before, (
        "страж изменил проверяемый каталог: "
        f"{sorted(_tree_files(case) - before)}"
    )


def test_gate_profile_without_gpu_fails_not_skips() -> None:
    """Гейтовый профиль на CPU — FAIL: вердикт A4 на CPU не производится."""
    result = _run("--profile", "gate", env_extra={"NET_JAX_BACKEND": "cpu"})
    assert result.returncode == EXIT_FAIL, _report(result)
    assert "профиль: gate" in result.stdout
    assert "CUDA" in result.stdout


def test_gate_profile_env_marker_matches_cli() -> None:
    """Маркер окружения и опция CLI разрешают профиль одинаково."""
    result = _run(env_extra={"NET_JAX_BACKEND": "cpu", "NET_GATE_PROFILE": "1"})
    assert result.returncode == EXIT_FAIL, _report(result)
    assert "профиль: gate" in result.stdout


def test_gate_profile_requires_determinism_flag(tmp_path: Path) -> None:
    """ADR-013: гейтовый профиль без флага детерминизма — FAIL (runtime-факт).

    Фикстура пиннит точность, но ``XLA_FLAGS`` не трогает — ровно та
    конфигурация, которую ADR-013 запрещает: GPU-прогон объявлен гейтовым, а
    побитовой воспроизводимости операций у него нет (A5 держится случайно).
    Страж обязан это увидеть и назвать, а не позеленеть по строке в conftest.
    """
    conftest = _write_conftest(tmp_path, PRECISION_ONLY_CONFTEST)
    result = _run("--profile", "gate", "--conftest", str(conftest))
    assert result.returncode == EXIT_FAIL, _report(result)
    assert "детерминизм XLA: off" in result.stdout
    assert EXPECTED_DETERMINISM_FLAG in _problems(result), _report(result)


def test_gate_profile_pins_determinism_flag() -> None:
    """Гейтовый профиль с настоящим conftest: флаг в окружении, вердикт PASS.

    На машине без CUDA гейтовый профиль красный по ADR-010 (нет устройства), и
    вердикт A4 там не производится вовсе: это SKIP, но только если нарушение
    именно в устройстве — красное по флагу детерминизма не пропускается.
    """
    result = _run("--profile", "gate")
    if result.returncode == EXIT_FAIL and EXPECTED_DETERMINISM_FLAG not in _problems(result):
        pytest.skip(
            "нет CUDA-устройства: гейтовый вердикт на этой машине не "
            f"производится ({_problems(result).strip()[:160]})"
        )
    assert result.returncode == EXIT_PASS, _report(result)
    assert "детерминизм XLA: on (gate)" in result.stdout
    assert EXPECTED_DETERMINISM_FLAG in result.stdout


def test_local_profile_does_not_pin_determinism_flag() -> None:
    """Локальный профиль флаг не ставит (ADR-013, решение 1) и объявляет это."""
    result = _run("--profile", "local")
    assert result.returncode == EXIT_PASS, _report(result)
    assert "детерминизм XLA: off" in result.stdout
    assert "XLA_FLAGS='<unset>'" in result.stdout


def test_criterion2_within_threshold_under_pinning() -> None:
    """Критерий 2 под пиннингом: расхождение в пределах порога на доступном бэкенде."""
    result = _run("--measure-criterion2")
    if result.returncode == EXIT_SKIP:
        reason = next(
            (line for line in result.stdout.splitlines() if "SKIP" in line),
            "нет CUDA-устройства",
        )
        pytest.skip(f"критерий 2 не измерен: {reason}")
    assert result.returncode == EXIT_PASS, _report(result)
    assert "jax_default_matmul_precision=highest" in result.stdout
    assert "в пределах порога" in result.stdout
