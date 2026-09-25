"""Тесты поведенческого стража AD-7 (C-040): check_gb10_single_load.

Три обязательных исхода из постановки дельты:
  (i)   одна модельная нагрузка  -> OK;
  (ii)  две модельные нагрузки   -> FAIL;
  (iii) nvidia-smi недоступен    -> NOT-VERIFIED с ненулевым кодом (не PASS!).

Страж — точка отказа контура: ложный зелёный запрещён. Поэтому проверяется не
только сам прогон, но и различие трёх состояний по коду возврата и по выводу.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import check_gb10_single_load as gb10  # noqa: E402  (путь добавляется выше)


def _proc(pid: int, name: str = "python", used_mib: int = 17536) -> gb10.GpuProcess:
    return gb10.GpuProcess(pid=pid, name=name, used_mib=used_mib)


def _lock(pid: int, name: str = "load.lock", what: str = "inference") -> gb10.LoadLock:
    return gb10.LoadLock(name=name, pid=pid, what=what)


# --- (i) одна нагрузка -> OK -------------------------------------------------


def test_one_load_ok() -> None:
    result = gb10.assess([_proc(1)], [])

    assert result.status == gb10.STATUS_OK
    assert result.exit_code == gb10.EXIT_OK
    assert result.message.startswith("OK:")


def test_zero_loads_ok() -> None:
    """Пустой стенд — тоже допустимо: подтверждено ноль нагрузок."""
    result = gb10.assess([], [])

    assert result.status == gb10.STATUS_OK
    assert result.exit_code == gb10.EXIT_OK


def test_below_threshold_is_not_a_model_load() -> None:
    """Smoke-тест CUDA (мало памяти) не считается модельной нагрузкой."""
    result = gb10.assess([_proc(1, used_mib=200)], [])

    assert result.status == gb10.STATUS_OK
    assert "0" in result.message  # ноль модельных нагрузок


# --- (ii) две нагрузки -> FAIL ----------------------------------------------


def test_two_loads_fail() -> None:
    result = gb10.assess([_proc(1), _proc(2)], [])

    assert result.status == gb10.STATUS_FAIL
    assert result.exit_code == gb10.EXIT_FAIL
    assert result.message.startswith("FAIL:")


def test_lock_and_proc_different_pids_fail() -> None:
    """Один compute-процесс + один лок с другим pid — две нагрузки."""
    result = gb10.assess([_proc(1)], [_lock(2)])

    assert result.status == gb10.STATUS_FAIL


def test_lock_and_proc_same_pid_dedup_to_one() -> None:
    """Лок объявляет тот же процесс, что виден в nvidia-smi — одна нагрузка."""
    result = gb10.assess([_proc(1)], [_lock(1, what="inference")])

    assert result.status == gb10.STATUS_OK


# --- (iii) nvidia-smi недоступен -> NOT-VERIFIED (не PASS) -------------------


def test_nvidia_unavailable_not_verified(monkeypatch, tmp_path, capsys) -> None:
    """nvidia-smi отсутствует: ненулевой код, «НЕ ПРОВЕРЕНО», не зелёный."""
    monkeypatch.setattr(gb10, "query_gpu_name", lambda: (False, ""))
    monkeypatch.setattr(gb10, "query_compute_apps", lambda: (False, []))

    rc = gb10.main(["--lock-dir", str(tmp_path)])
    out = capsys.readouterr().out

    assert rc == gb10.EXIT_NOT_VERIFIED
    assert rc != 0
    assert "НЕ ПРОВЕРЕНО" in out


def test_stand_not_gb10_not_verified(monkeypatch, tmp_path, capsys) -> None:
    """GPU присутствует, но это не DGX Spark/GB10 — проверка неприменима."""
    monkeypatch.setattr(
        gb10, "query_gpu_name", lambda: (True, "NVIDIA GeForce RTX 4080 SUPER")
    )
    monkeypatch.setattr(gb10, "query_compute_apps", lambda: (False, []))

    rc = gb10.main(["--lock-dir", str(tmp_path)])
    out = capsys.readouterr().out

    assert rc == gb10.EXIT_NOT_VERIFIED
    assert "НЕ ПРОВЕРЕНО" in out
    assert "RTX 4080" in out


# --- различие трёх состояний -------------------------------------------------


def test_three_states_are_distinguishable() -> None:
    """Коды возврата и префиксы вывода различают три исхода; третий != первому."""
    ok = gb10.assess([_proc(1)], [])
    fail = gb10.assess([_proc(1), _proc(2)], [])
    nv = gb10.Result(
        gb10.STATUS_NOT_VERIFIED, "НЕ ПРОВЕРЕНО: стенд недоступен"
    )

    assert len({ok.exit_code, fail.exit_code, nv.exit_code}) == 3
    assert ok.exit_code == 0
    assert nv.exit_code != ok.exit_code
    assert {ok.status, fail.status, nv.status} == {
        gb10.STATUS_OK,
        gb10.STATUS_FAIL,
        gb10.STATUS_NOT_VERIFIED,
    }


# --- реестр локов ------------------------------------------------------------


def test_read_locks_ignores_stale_and_malformed(tmp_path: Path) -> None:
    """Мёртвый pid и неконформный маркер отбрасываются; живой — остаётся."""
    lock_dir = tmp_path / "locks"
    lock_dir.mkdir()
    alive = lock_dir / "alive.lock"
    alive.write_text(
        f'{{"pid": {_my_pid()}, "what": "inference"}}', encoding="utf-8"
    )
    (lock_dir / "stale.lock").write_text(
        f'{{"pid": {_dead_pid()}, "what": "crashed"}}', encoding="utf-8"
    )
    (lock_dir / "malformed.lock").write_text(
        '{"what": "нет pid"}', encoding="utf-8"
    )

    locks = gb10.read_locks(lock_dir)

    assert [lock.name for lock in locks] == ["alive.lock"]


def test_read_locks_missing_dir_is_empty(tmp_path: Path) -> None:
    """Отсутствие реестра — ноль локов, не ошибка и не NOT-VERIFIED."""
    assert gb10.read_locks(tmp_path / "nope") == []


def _my_pid() -> int:
    import os

    return os.getpid()


def _dead_pid() -> int:
    import subprocess

    proc = subprocess.Popen(["true"])
    pid = proc.pid
    proc.wait()
    return pid  # процесс уже завершился — pid почти наверняка мёртв


# --- парсинг nvidia-smi ------------------------------------------------------


def test_parse_compute_apps() -> None:
    procs = gb10.parse_compute_apps(
        "1234, python, 17536\n5678, vllm, 8192\n"
    )

    assert procs == [
        gb10.GpuProcess(pid=1234, name="python", used_mib=17536),
        gb10.GpuProcess(pid=5678, name="vllm", used_mib=8192),
    ]


# --- сквозной прогон main() для OK и FAIL -----------------------------------


def test_main_end_to_end_ok(monkeypatch, tmp_path, capsys) -> None:
    """На GB10 с одним compute-процессом main() возвращает 0 и «OK:»."""
    monkeypatch.setattr(gb10, "query_gpu_name", lambda: (True, "GB10"))
    monkeypatch.setattr(
        gb10, "query_compute_apps", lambda: (True, [_proc(1)])
    )

    rc = gb10.main(["--lock-dir", str(tmp_path)])
    out = capsys.readouterr().out

    assert rc == gb10.EXIT_OK
    assert out.startswith("OK:")


def test_main_end_to_end_fail(monkeypatch, tmp_path, capsys) -> None:
    """На GB10 с двумя compute-процессами main() возвращает 2 и «FAIL:»."""
    monkeypatch.setattr(gb10, "query_gpu_name", lambda: (True, "GB10"))
    monkeypatch.setattr(
        gb10, "query_compute_apps", lambda: (True, [_proc(1), _proc(2)])
    )

    rc = gb10.main(["--lock-dir", str(tmp_path)])
    out = capsys.readouterr().out

    assert rc == gb10.EXIT_FAIL
    assert out.startswith("FAIL:")
