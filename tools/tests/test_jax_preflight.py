"""Tests of ``tools/jax_preflight.py`` (ADR-041) — CPU-only, subprocess mocked.

Pins the four properties the preflight discipline rests on (инцидент 08.10):
the XLA memory limit is set **before** ``import jax`` and never overwrites an
explicit environment value; the stand gate is fail-closed on foreign compute
processes without an owner's permission file; a machine without ``nvidia-smi``
is not fatal. No GPU, no network, no ``jax`` import in the module under test.
"""

from __future__ import annotations

import importlib
import os
import re
import sys
from pathlib import Path

import pytest

TOOLS_DIR = Path(__file__).resolve().parents[1]
CASE_DIR = TOOLS_DIR.parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import jax_preflight  # noqa: E402

ENV = jax_preflight.ENV_VAR


class _Completed:
    """Минимальный результат ``subprocess.run`` (stdout/returncode)."""

    def __init__(self, stdout: str = "", returncode: int = 0) -> None:
        self.stdout = stdout
        self.stderr = ""
        self.returncode = returncode


def _patch_runner(
    monkeypatch: pytest.MonkeyPatch,
    *,
    apps: str | None = None,
    gpu_name: str = "NVIDIA GB10",
    free_out: str = "               total        used        free      shared  buff/cache   available\n"
    "Mem:             120          10           5           0           5         110",
    missing: tuple[str, ...] = (),
) -> None:
    """Мок ``subprocess.run`` внутри модуля: диспетчер по команде сенсора."""

    def fake_run(args, **kwargs):  # noqa: ANN001, ANN003
        name = args[0]
        if name in missing:
            raise FileNotFoundError(name)
        if name == "nvidia-smi":
            if any(str(a).startswith("--query-compute-apps") for a in args):
                return _Completed(apps if apps is not None else "")
            if any(str(a).startswith("--query-gpu=name") for a in args):
                return _Completed(gpu_name + "\n")
            raise AssertionError(f"unexpected nvidia-smi args: {args}")
        if name == "free":
            return _Completed(free_out)
        raise AssertionError(f"unexpected command: {args}")

    monkeypatch.setattr(jax_preflight.subprocess, "run", fake_run)


# --- лимит памяти: выставляется, не перетирается, логируется -----------------


def test_ensure_mem_fraction_sets_default_when_env_empty(monkeypatch, capsys):
    monkeypatch.delenv(ENV, raising=False)
    value = jax_preflight.ensure_mem_fraction()
    assert value == "0.5"
    assert os.environ[ENV] == "0.5"
    err = capsys.readouterr().err
    assert f"[jax-preflight] {ENV}=0.5 (source: default)" in err


def test_ensure_mem_fraction_preserves_env_value(monkeypatch, capsys):
    monkeypatch.setenv(ENV, "0.15")
    value = jax_preflight.ensure_mem_fraction()
    assert value == "0.15"
    assert os.environ[ENV] == "0.15"  # явное значение окружения не перетёрто
    assert f"(source: env)" in capsys.readouterr().err


def test_ensure_mem_fraction_logs_exactly_one_line(monkeypatch, capsys):
    monkeypatch.delenv(ENV, raising=False)
    jax_preflight.ensure_mem_fraction()
    err = capsys.readouterr().err.strip().splitlines()
    assert len(err) == 1
    assert re.fullmatch(
        rf"\[jax-preflight\] {re.escape(ENV)}=\S+ \(source: (env|default)\)", err[0]
    )


# --- порядок: модуль не импортирует jax (лимит ставится до инициализации) ----


def test_module_does_not_import_jax():
    """Проверка порядка: префлайт обязан быть импортируем без jax вовсе."""
    source = Path(jax_preflight.__file__).read_text(encoding="utf-8")
    assert not re.search(r"^\s*(?:import|from)\s+jax\b", source, re.MULTILINE)
    assert not hasattr(jax_preflight, "jax")  # символ jax не привязан в модуле


def test_no_network_in_source():
    """Selftest без сети: в модуле нет сетевых импортов/вызовов."""
    source = Path(jax_preflight.__file__).read_text(encoding="utf-8")
    for needle in ("socket", "urllib", "requests", "httpx", "http.client", "urlopen"):
        assert needle not in source


# --- префлайт-гейт: fail-closed / разрешение / no-gpu ------------------------


def test_gate_foreign_without_lock_exits(monkeypatch, tmp_path):
    _patch_runner(monkeypatch, apps="1714413, ./llama.cpp/llama-server, 7584 MiB")
    with pytest.raises(SystemExit) as exc:
        jax_preflight.preflight_gate(lock_dir=tmp_path)
    assert "чужие compute-процессы" in str(exc.value)


def test_gate_permission_file_allows_colocation(monkeypatch, tmp_path):
    (tmp_path / "allow-colocation.txt").write_text("owner: ok to share the stand\n", encoding="utf-8")
    _patch_runner(monkeypatch, apps="1714413, ./llama.cpp/llama-server, 7584 MiB")
    state = jax_preflight.preflight_gate(lock_dir=tmp_path)
    assert state["ok"] is True
    assert "co-run-allowed" in state["reason"]
    assert len(state["foreign_procs"]) == 1


def test_gate_empty_lock_file_is_not_permission(monkeypatch, tmp_path):
    (tmp_path / "allow-colocation.txt").write_text("   \n", encoding="utf-8")  # пусто — не разрешение
    _patch_runner(monkeypatch, apps="1714413, ./llama.cpp/llama-server, 7584 MiB")
    with pytest.raises(SystemExit):
        jax_preflight.preflight_gate(lock_dir=tmp_path)


def test_gate_no_nvidia_smi_is_not_fatal(monkeypatch, tmp_path):
    _patch_runner(monkeypatch, missing=("nvidia-smi",))
    state = jax_preflight.preflight_gate(lock_dir=tmp_path)
    assert state["ok"] is True
    assert state["reason"] == "no-gpu"
    assert state["gpu"] == "none"


def test_gate_own_process_is_not_foreign(monkeypatch, tmp_path):
    _patch_runner(monkeypatch, apps=f"{os.getpid()}, python3, 4096 MiB")
    state = jax_preflight.preflight_gate(lock_dir=tmp_path)
    assert state["ok"] is True
    assert state["foreign_procs"] == []


def test_gate_reports_mem_available(monkeypatch, tmp_path):
    _patch_runner(monkeypatch, apps="")
    state = jax_preflight.preflight_gate(lock_dir=tmp_path)
    assert state["mem_available_gb"] == 110.0


# --- gate_or_exit: политика стенда (enforce GB10 / advisory локально) --------


def test_gate_or_exit_enforces_on_shared_stand(monkeypatch, tmp_path):
    monkeypatch.delenv(jax_preflight.GATE_ENV, raising=False)
    _patch_runner(monkeypatch, apps="1714413, llama-server, 7584 MiB", gpu_name="NVIDIA GB10")
    with pytest.raises(SystemExit):
        jax_preflight.gate_or_exit(lock_dir=tmp_path)


def test_gate_or_exit_advisory_on_local_gpu(monkeypatch, tmp_path):
    monkeypatch.delenv(jax_preflight.GATE_ENV, raising=False)
    _patch_runner(
        monkeypatch,
        apps="1714413, llama-server, 7584 MiB",
        gpu_name="NVIDIA GeForce RTX 4080 SUPER",
    )
    state = jax_preflight.gate_or_exit(lock_dir=tmp_path)
    assert state is not None and state["ok"] is True  # локальный GPU не блокирует


def test_gate_or_exit_can_be_disabled(monkeypatch, tmp_path):
    monkeypatch.setenv(jax_preflight.GATE_ENV, "0")
    _patch_runner(monkeypatch, apps="1714413, llama-server, 7584 MiB", gpu_name="NVIDIA GB10")
    assert jax_preflight.gate_or_exit(lock_dir=tmp_path) is None


# --- регрессия: инструменты вызывают префлайт до import jax ------------------

_TOOLS_WITH_PREFLIGHT = (
    "pretrain_run.py",
    "bench_block_merge.py",
    "bench_kda_wyut.py",
    "check_precision_pinning.py",
    "d8_isolate.py",
    "pool_cost_probe.py",
    "run_a4_pipeline.py",
    "run_rl_smoke.py",
    "run_sft_smoke.py",
    "topk_exact_probe.py",
    "a4_manifest.py",
    "mfu_bf16_protocol.py",
    "loss_parity_bf16.py",
    "profile_mfu.py",
)


@pytest.mark.parametrize("name", _TOOLS_WITH_PREFLIGHT)
def test_tool_calls_preflight_before_import_jax(name: str):
    path = TOOLS_DIR / name
    if not path.is_file():  # например, профилировочные приборы живут в ветке BF16
        pytest.skip(f"{name} отсутствует в этой ветке")
    lines = path.read_text(encoding="utf-8").splitlines()
    preflight_at = next((i for i, ln in enumerate(lines) if "ensure_mem_fraction()" in ln), None)
    assert preflight_at is not None, f"{name}: нет вызова ensure_mem_fraction()"
    jax_at = next((i for i, ln in enumerate(lines) if re.match(r"^import jax\b", ln)), None)
    if jax_at is not None:  # лимит обязан стоять до инициализации рантайма JAX
        assert preflight_at < jax_at, f"{name}: ensure_mem_fraction() после import jax"


# --- ADR-041 Amendment 08.10: гейт только для прогонов, enforce только GB10 ---

#: Инструменты-ПРОВЕРКИ (контрольный контур): несут только лимит памяти.
_CHECKING_TOOLS = ("check_precision_pinning.py", "a4_manifest.py")

#: Инструменты-ПРОГОНЫ (обучение/профилировка): обязаны нести и fail-closed гейт.
_RUN_TOOLS_GATED = (
    "pretrain_run.py",
    "bench_block_merge.py",
    "bench_kda_wyut.py",
    "run_rl_smoke.py",
    "run_sft_smoke.py",
    "d8_isolate.py",
    "pool_cost_probe.py",
    "topk_exact_probe.py",
    "run_a4_pipeline.py",
    "mfu_bf16_protocol.py",
    "loss_parity_bf16.py",
    "profile_mfu.py",
)


@pytest.mark.parametrize("name", _CHECKING_TOOLS)
def test_checking_tool_survives_foreign_load(name, monkeypatch, capsys):
    """ADR-041 п.2: проверяющий инструмент не падает при чужой нагрузке.

    Его единственный префлайт-вызов — ``ensure_mem_fraction()`` (лимит памяти);
    fail-closed ``gate_or_exit()`` контрольному контуру не принадлежит. Даже при
    чужих compute-процессах на стенде GB10 SystemExit не поднимается, иначе
    C-042/C-038 дали бы ложный FAIL.
    """
    _patch_runner(monkeypatch, apps="1714413, llama-server, 7584 MiB", gpu_name="NVIDIA GB10")
    source = (TOOLS_DIR / name).read_text(encoding="utf-8")
    assert "gate_or_exit" not in source, "контрольный контур не должен нести fail-closed гейт"
    assert "ensure_mem_fraction()" in source  # лимит памяти обязателен всем

    value = jax_preflight.ensure_mem_fraction()  # не падает под чужой нагрузкой
    assert value
    assert capsys.readouterr().err  # префлайт залогирован, SystemExit не поднят


@pytest.mark.parametrize("name", _RUN_TOOLS_GATED)
def test_run_tool_keeps_fail_closed_gate(name):
    """ADR-041 п.2 (обратная сторона): прогонный путь обязан нести гейт."""
    if not (TOOLS_DIR / name).is_file():
        pytest.skip(f"{name} отсутствует в этой ветке")
    source = (TOOLS_DIR / name).read_text(encoding="utf-8")
    assert "gate_or_exit()" in source, f"{name}: прогонный путь потерял гейт стенда"


def test_gate_or_exit_local_gpu_warns_and_continues(monkeypatch, tmp_path, capsys):
    """Не-GB10 GPU: чужие процессы -> предупреждение в stderr, без SystemExit."""
    monkeypatch.delenv(jax_preflight.GATE_ENV, raising=False)
    _patch_runner(
        monkeypatch,
        apps="1714413, llama-server, 7584 MiB",
        gpu_name="NVIDIA GeForce RTX 4080 SUPER",
    )
    state = jax_preflight.gate_or_exit(lock_dir=tmp_path)
    assert state is not None and state["ok"] is True  # локальный GPU не блокирует
    assert "не-GB10" in capsys.readouterr().err  # но предупреждает


def test_gate_or_exit_no_nvidia_smi_warns_and_continues(monkeypatch, tmp_path, capsys):
    """Нет nvidia-smi (дев-ПК): предупреждение и продолжение, не SystemExit."""
    monkeypatch.delenv(jax_preflight.GATE_ENV, raising=False)
    _patch_runner(monkeypatch, missing=("nvidia-smi",))
    state = jax_preflight.gate_or_exit(lock_dir=tmp_path)
    assert state is not None and state["gpu"] == "none"
    assert "nvidia-smi недоступен" in capsys.readouterr().err


def test_gate_or_exit_force_value_does_not_extend_beyond_gb10(monkeypatch, tmp_path, capsys):
    """ADR-041 п.3: даже ``JAX_PREFLIGHT_GATE=1`` не блокирует не-GB10 GPU."""
    monkeypatch.setenv(jax_preflight.GATE_ENV, "1")
    _patch_runner(
        monkeypatch,
        apps="1714413, llama-server, 7584 MiB",
        gpu_name="NVIDIA RTX 4090",
    )
    state = jax_preflight.gate_or_exit(lock_dir=tmp_path)
    assert state is not None and state["ok"] is True
    assert "не-GB10" in capsys.readouterr().err


def test_gate_or_exit_enforces_on_gb10_with_permission_file(monkeypatch, tmp_path):
    """GB10 + файл-разрешение владельца -> enforce не срабатывает (прогон идёт)."""
    monkeypatch.delenv(jax_preflight.GATE_ENV, raising=False)
    (tmp_path / "allow-colocation.txt").write_text("owner: ok to share\n", encoding="utf-8")
    _patch_runner(monkeypatch, apps="1714413, llama-server, 7584 MiB", gpu_name="NVIDIA GB10")
    state = jax_preflight.gate_or_exit(lock_dir=tmp_path)
    assert state is not None and "co-run-allowed" in state["reason"]


# --- ADR-041 Amendment 08.10: read_locks устойчив к недоступному .locks -------

import check_gb10_single_load as gb10  # noqa: E402


def _raise_permission(*_args, **_kwargs):
    raise PermissionError(13, "Permission denied")


def test_read_locks_state_unavailable_on_permission_error(tmp_path, monkeypatch):
    """Каталог режима 000 (PermissionError) -> статус недоступности, не падение."""
    lock_dir = tmp_path / ".locks"
    lock_dir.mkdir()
    monkeypatch.setattr(gb10.os, "scandir", _raise_permission)

    state = gb10.read_locks_state(lock_dir)
    assert state.locks == []
    assert state.readable is False
    assert state.unavailable and "недоступен" in state.unavailable
    # совместимая обёртка тоже не падает и отдаёт пустой список
    assert gb10.read_locks(lock_dir) == []


def test_read_locks_state_missing_dir_is_empty_and_readable(tmp_path):
    """Отсутствие реестра — «ноль локов» и доступность (не NOT-VERIFIED)."""
    state = gb10.read_locks_state(tmp_path / "nope")
    assert state.locks == []
    assert state.readable is True
    assert state.unavailable is None


def test_main_lock_dir_unavailable_is_not_verified(tmp_path, monkeypatch, capsys):
    """Недоступный реестр при подтверждённом GB10 — NOT-VERIFIED, а не OK."""
    lock_dir = tmp_path / ".locks"
    lock_dir.mkdir()
    monkeypatch.setattr(gb10, "query_gpu_name", lambda: (True, "GB10"))
    monkeypatch.setattr(
        gb10, "query_compute_apps", lambda: (True, [gb10.GpuProcess(pid=1, name="python", used_mib=17536)])
    )
    monkeypatch.setattr(gb10.os, "scandir", _raise_permission)

    rc = gb10.main(["--lock-dir", str(lock_dir)])
    out = capsys.readouterr().out
    assert rc == gb10.EXIT_NOT_VERIFIED
    assert "НЕ ПРОВЕРЕНО" in out and "недоступен" in out


def test_main_lock_dir_ok_confirmed_fail_stays_fail(tmp_path, monkeypatch, capsys):
    """Подтверждённый FAIL (>=2 нагрузок) важнее недоступного реестра."""
    lock_dir = tmp_path / ".locks"
    lock_dir.mkdir()
    two = [gb10.GpuProcess(pid=1, name="vllm", used_mib=17536), gb10.GpuProcess(pid=2, name="sft", used_mib=17300)]
    monkeypatch.setattr(gb10, "query_gpu_name", lambda: (True, "GB10"))
    monkeypatch.setattr(gb10, "query_compute_apps", lambda: (True, two))
    monkeypatch.setattr(gb10.os, "scandir", _raise_permission)

    rc = gb10.main(["--lock-dir", str(lock_dir)])
    assert rc == gb10.EXIT_FAIL
    assert "FAIL" in capsys.readouterr().out


# --- ADR-041 Amendment 2 (08.10): детект стенда шире literal «GB10» -----------


def test_stand_markers_cover_gb10_spark_grace():
    """Дефолтные маркеры стенда: GB10 | Spark | Grace (Amendment 2 п.3).

    Literal «GB10» недостаточен: стенд с именем без этой подстроки получил бы
    advisory-режим вместо fail-closed — отказ в сторону разрешения.
    """
    assert "GB10" in jax_preflight.STAND_MARKERS
    for device in ("NVIDIA GB10", "NVIDIA DGX Spark", "Grace Blackwell", "nvidia gb10"):
        assert jax_preflight.is_shared_stand({"device": device}), device


def test_stand_markers_do_not_match_dev_gpu():
    """Дев-ПК (RTX) стендом не считается: чужие процессы там — advisory."""
    for device in ("NVIDIA GeForce RTX 4080 SUPER", "NVIDIA RTX 4090", "cpu", None, ""):
        assert not jax_preflight.is_shared_stand({"device": device}), device


def test_gate_or_exit_enforces_on_dgx_spark(monkeypatch, tmp_path):
    """Стенд назван «DGX Spark» (без подстроки GB10) — всё равно fail-closed."""
    monkeypatch.delenv(jax_preflight.GATE_ENV, raising=False)
    monkeypatch.delenv(jax_preflight.STAND_RE_ENV, raising=False)
    _patch_runner(monkeypatch, apps="1714413, llama-server, 7584 MiB", gpu_name="NVIDIA DGX Spark")
    with pytest.raises(SystemExit):
        jax_preflight.gate_or_exit(lock_dir=tmp_path)


def test_stand_re_env_override_widens_detection(monkeypatch, tmp_path):
    """Переопределение маркеров переменной окружения расширяет детект стенда."""
    monkeypatch.delenv(jax_preflight.GATE_ENV, raising=False)
    monkeypatch.setenv(jax_preflight.STAND_RE_ENV, "Titan")
    _patch_runner(monkeypatch, apps="1714413, llama-server, 7584 MiB", gpu_name="NVIDIA Titan V")
    with pytest.raises(SystemExit):
        jax_preflight.gate_or_exit(lock_dir=tmp_path)


def test_stand_re_env_override_is_honoured(monkeypatch, tmp_path, capsys):
    """Переопределение уважается буквально: «DGX Spark» мимо шаблона -> advisory."""
    monkeypatch.delenv(jax_preflight.GATE_ENV, raising=False)
    monkeypatch.setenv(jax_preflight.STAND_RE_ENV, "GB10")
    _patch_runner(monkeypatch, apps="1714413, llama-server, 7584 MiB", gpu_name="NVIDIA DGX Spark")
    state = jax_preflight.gate_or_exit(lock_dir=tmp_path)
    assert state is not None and state["ok"] is True
    assert "не-GB10" in capsys.readouterr().err


def test_stand_re_env_broken_pattern_keeps_default_guard(monkeypatch, capsys):
    """Битый шаблон не сужает защиту: остаются дефолтные маркеры + предупреждение."""
    monkeypatch.setenv(jax_preflight.STAND_RE_ENV, "[")
    assert jax_preflight.is_shared_stand({"device": "NVIDIA GB10"})  # дефолт защищает
    err = capsys.readouterr().err
    assert "не компилируется" in err and "сужение защиты запрещено" in err


# --- ADR-041 Amendment п.5: приборы стадии 2 несут префлайт на прогонных путях


@pytest.mark.parametrize("name", ("mfu_bf16_protocol", "loss_parity_bf16", "profile_mfu"))
def test_run_module_sets_mem_fraction_on_import(name, monkeypatch):
    """Прогонный прибор выставляет лимит памяти уже на импорте модуля.

    Регрессия слияния bf16-ветки: правка 4603902 (префлайт в этих приборах)
    была перетёрта версиями из ветки — C-053 красный, четыре теста падали.
    Проверка фактического эффекта (лимит в окружении), а не наличия строки:
    ``reload`` переисполняет шапку модуля.
    """
    monkeypatch.delenv(ENV, raising=False)
    module = importlib.import_module(name)
    importlib.reload(module)
    assert os.environ[ENV] == jax_preflight.DEFAULT_MEM_FRACTION


def test_nsys_wrapper_exports_mem_fraction():
    """nsys-обвязка (стендовый прогон) фиксирует лимит памяти и сама."""
    sh = (TOOLS_DIR / "profile_mfu_nsys.sh").read_text(encoding="utf-8")
    assert "export XLA_PYTHON_CLIENT_MEM_FRACTION=" in sh
