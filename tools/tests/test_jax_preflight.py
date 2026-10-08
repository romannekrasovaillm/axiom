"""Tests of ``tools/jax_preflight.py`` (ADR-041) — CPU-only, subprocess mocked.

Pins the four properties the preflight discipline rests on (инцидент 08.10):
the XLA memory limit is set **before** ``import jax`` and never overwrites an
explicit environment value; the stand gate is fail-closed on foreign compute
processes without an owner's permission file; a machine without ``nvidia-smi``
is not fatal. No GPU, no network, no ``jax`` import in the module under test.
"""

from __future__ import annotations

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
