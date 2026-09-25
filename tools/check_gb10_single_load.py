#!/usr/bin/env python3
"""C-040 — поведенческий страж AD-7 «одна модельная нагрузка на GB10 за раз».

Определяет, исполняется ли на стенде GB10 (DGX Spark, 128 ГБ unified) больше
одной модельной нагрузки одновременно. Проверяет не «наличие надписи в спайне»,
а состояние стенда: сколько GPU-нагрузок занято прямо сейчас.

Источники состояния стенда (два независимых сенсора, объединяются по PID):

1. ``nvidia-smi`` — авторитетный сенсор занятости GPU. Модельной нагрузкой
   считается CUDA-вычислительный процесс (``--query-compute-apps``),
   удерживающий не менее ``--threshold`` МиБ видеопамяти (по умолчанию 1024).
   На безголовом Spark нет Xorg/gnome-shell: compute-процесс с гигабайтами
   памяти — это инференс/rollout/среда, а не утилита.

2. Реестр маркеров-локов занятости — кооперативная декларация нагрузки,
   которую держат долгие прогоны. Конвенция (минимальная, введена этой
   дельтой): каталог ``$GB10_LOCK_DIR`` (по умолчанию ``~/gb10-shared/.locks/``).
   Выбран ``~/gb10-shared/.locks``, а не локальный ``.gb10-loads/``: реестр
   должен быть виден всем нагрузкам независимо от их рабочего каталога, а
   ``~/gb10-shared`` — канонический общий диск весов/датасетов (C-032/C-033).
   Формат маркера: файл ``*.lock`` с JSON-объектом, обязательное поле ``pid``
   (int), опциональное ``what`` (str, описание нагрузки)::

       {"pid": 12345, "what": "inference llama.cpp", "started_at": 1757000000}

   Маркер активен, только если ``pid`` жив (``os.kill(pid, 0)``). Мёртвый
   ``pid`` — брошенный маркер упавшего прогона: игнорируется, чтобы не давать
   вечный ложный FAIL. Маркер без целого ``pid`` — неконформный: игнорируется.

Три различимых исхода (третий не равен первому; ложный зелёный запрещён):

* ``OK`` (exit 0) — подтверждено: модельных нагрузок ноль или одна.
* ``FAIL`` (exit 2) — подтверждено: нагрузок больше одной.
* ``NOT-VERIFIED`` (exit 1) — стенд недоступен / nvidia-smi отсутствует /
  данных недостаточно. Печатает ``НЕ ПРОВЕРЕНО: <причина>``. Недоступность
  стенда доказательством «одна нагрузка» не считается.

Запуск::

    python3 tools/check_gb10_single_load.py [--lock-dir PATH] [--threshold MIB]

Переопределение реестра: переменная окружения ``GB10_LOCK_DIR``.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

#: Имя GPU на DGX Spark (GB10, Grace Blackwell). Консервативный шаблон:
#: нераспознанное имя -> NOT-VERIFIED, а не «зелёный на чужой машине».
GB10_NAME_RE = re.compile(r"GB10|Spark|Grace\s*Blackwell", re.IGNORECASE)

#: Порог отсечения «модельной нагрузки» по занятой видеопамяти, МиБ.
#: Целевая модель 30B/4-бит занимает ~17 ГБ; реальная нагрузка на порядки выше
#: порога, а CUDA-утилита (smoke-тест) — ниже.
DEFAULT_THRESHOLD_MIB = 1024

#: Реестр локов по умолчанию — общий диск, видимый всем нагрузкам.
DEFAULT_LOCK_DIR = Path.home() / "gb10-shared" / ".locks"

#: Коды возврата различимы; третий (NOT-VERIFIED) не равен первому (OK).
EXIT_OK = 0
EXIT_NOT_VERIFIED = 1
EXIT_FAIL = 2

STATUS_OK = "OK"
STATUS_FAIL = "FAIL"
STATUS_NOT_VERIFIED = "NOT-VERIFIED"


@dataclass(frozen=True)
class GpuProcess:
    """Один compute-процесс из ``nvidia-smi --query-compute-apps``."""

    pid: int
    name: str
    used_mib: int


@dataclass(frozen=True)
class LoadLock:
    """Активный маркер-лок занятости из реестра (pid всегда жив)."""

    name: str
    pid: int
    what: str


@dataclass
class Result:
    """Вердикт стража: различимые статус, сообщение, детали и код возврата."""

    status: str
    message: str
    details: list[str] = field(default_factory=list)

    @property
    def exit_code(self) -> int:
        return {
            STATUS_OK: EXIT_OK,
            STATUS_FAIL: EXIT_FAIL,
            STATUS_NOT_VERIFIED: EXIT_NOT_VERIFIED,
        }[self.status]


def _pid_alive(pid: int) -> bool:
    """Жив ли процесс ``pid`` (реестр локов: мёртвый pid — брошенный маркер)."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # процесс есть, но не наш — маркер всё ещё активен
    except OSError:
        return False
    return True


def parse_compute_apps(csv_text: str) -> list[GpuProcess]:
    """Парсит ``nvidia-smi --query-compute-apps=pid,process_name,used_memory
    --format=csv,noheader,nounits`` в список процессов."""
    procs: list[GpuProcess] = []
    for line in csv_text.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 2:
            continue
        try:
            pid = int(parts[0])
        except ValueError:
            continue
        name = parts[1]
        used_mib = 0
        if len(parts) >= 3:
            try:
                used_mib = int(float(parts[2]))
            except ValueError:
                used_mib = 0
        procs.append(GpuProcess(pid=pid, name=name, used_mib=used_mib))
    return procs


def read_locks(lock_dir: Path) -> list[LoadLock]:
    """Читает активные маркеры-локи (мёртвые/неконформные отбрасываются)."""
    locks: list[LoadLock] = []
    if not lock_dir.is_dir():
        return locks
    for path in sorted(lock_dir.glob("*.lock")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue  # нечитаемый/битый маркер не доказывает нагрузку и не роняет страж
        if not isinstance(data, dict):
            continue
        pid = data.get("pid")
        if not isinstance(pid, int):
            continue  # конвенция требует целый pid — иначе не проверить живость
        if not _pid_alive(pid):
            continue  # брошенный маркер упавшего прогона
        what = str(data.get("what") or path.stem)
        locks.append(LoadLock(name=path.name, pid=pid, what=what))
    return locks


def _run_nvidia_smi(*args: str) -> tuple[bool, str]:
    """Запускает nvidia-smi; возвращает (успех, stdout)."""
    try:
        proc = subprocess.run(
            ["nvidia-smi", *args],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except FileNotFoundError:
        return False, ""
    except subprocess.TimeoutExpired:
        return False, ""
    if proc.returncode != 0:
        return False, ""
    return True, proc.stdout


def query_gpu_name() -> tuple[bool, str]:
    """Возвращает (успех, имя GPU). Пустая строка — nvidia-smi недоступен."""
    ok, out = _run_nvidia_smi("--query-gpu=name", "--format=csv,noheader,nounits")
    if not ok:
        return False, ""
    first = out.strip().splitlines()
    name = first[0].strip() if first else ""
    return bool(name), name


def query_compute_apps() -> tuple[bool, list[GpuProcess]]:
    """Возвращает (успех, compute-процессы)."""
    ok, out = _run_nvidia_smi(
        "--query-compute-apps=pid,process_name,used_memory",
        "--format=csv,noheader,nounits",
    )
    if not ok:
        return False, []
    return True, parse_compute_apps(out)


def assess(
    procs: list[GpuProcess],
    locks: list[LoadLock],
    threshold_mib: int = DEFAULT_THRESHOLD_MIB,
) -> Result:
    """Чистая логика подсчёта нагрузок (OK/FAIL); сенсоры уже собраны.

    Модельные нагрузки — объединение по PID двух сенсоров: compute-процессов
    с памятью >= порога и активных локов. Локи только добавляют нагрузки,
    поэтому не могут создать ложный зелёный.
    """
    load_ids: dict[int | str, str] = {}
    for p in procs:
        if p.used_mib >= threshold_mib:
            load_ids[p.pid] = f"compute pid={p.pid} {p.name} {p.used_mib} МиБ"
    for lock in locks:
        load_ids[lock.pid] = f"lock pid={lock.pid} {lock.what}"

    count = len(load_ids)
    details = [f"  - {desc}" for desc in sorted(load_ids.values())]

    if count >= 2:
        return Result(
            STATUS_FAIL,
            f"FAIL: более одной модельной нагрузки — подтверждено "
            f"(модельных нагрузок: {count})",
            details,
        )
    return Result(
        STATUS_OK,
        f"OK: одна модельная нагрузка за раз — подтверждено "
        f"(модельных нагрузок: {count})",
        details,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="AD-7 (C-040): одна модельная нагрузка на GB10 за раз"
    )
    parser.add_argument(
        "--lock-dir",
        default=os.environ.get("GB10_LOCK_DIR") or str(DEFAULT_LOCK_DIR),
        help="каталог реестра локов занятости (по умолчанию ~/gb10-shared/.locks)",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=DEFAULT_THRESHOLD_MIB,
        help=f"порог памяти модельной нагрузки, МиБ (по умолчанию {DEFAULT_THRESHOLD_MIB})",
    )
    args = parser.parse_args(argv)

    locks = read_locks(Path(args.lock_dir))

    gpu_ok, gpu_name = query_gpu_name()
    if not gpu_ok:
        result = Result(
            STATUS_NOT_VERIFIED,
            "НЕ ПРОВЕРЕНО: nvidia-smi недоступен или стенд недоступен — "
            "занятость GPU не подтверждена",
        )
    elif not GB10_NAME_RE.search(gpu_name):
        result = Result(
            STATUS_NOT_VERIFIED,
            f"НЕ ПРОВЕРЕНО: стенд GB10 недоступен — GPU «{gpu_name}» "
            "не DGX Spark/GB10 (занятость чужой машины не считается проверкой)",
        )
    else:
        procs_ok, procs = query_compute_apps()
        if not procs_ok:
            result = Result(
                STATUS_NOT_VERIFIED,
                "НЕ ПРОВЕРЕНО: данных недостаточно — не удалось получить "
                "список GPU-процессов",
            )
        else:
            result = assess(procs, locks, threshold_mib=args.threshold)

    print(result.message)
    for detail in result.details:
        print(detail)
    return result.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
