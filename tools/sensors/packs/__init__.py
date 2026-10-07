"""Доменные пакеты экспортёров (ADR-038, дельта G2).

Пакет — каталог ``packs/<domain>/`` с ``PACK.yaml`` (список экспортёров и
фикстуры) и модулями экспортёров. :func:`discover_exporters` подхватывает все
пакеты и возвращает экземпляры, соответствующие протоколу
:class:`tools.sensors.protocol.Exporter`.

Ядро кейса (S-001…S-030) остаётся на прямой регистрации в ``model/sensors.yaml``
и пакетом не переписывается. Референсные пакеты чужих доменов (``service``,
``iac``, ``data``) — демонстрация универсальности протокола; в кейсе axiom они
не регистрируются и в правила кейса не входят.
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from ...miniyaml import MiniYamlError, load_file

#: Каталог всех пакетов (этот файл лежит в ``tools/sensors/packs/``).
PACKS_DIR = Path(__file__).resolve().parent


class PackError(ValueError):
    """Манифест пакета не читается или не соответствует схеме."""


@dataclass(frozen=True)
class PackInfo:
    name: str
    path: Path
    title: str
    exporters: tuple[dict[str, str], ...] = ()
    fixtures: Optional[str] = None


def load_manifest(pack_dir: Path) -> PackInfo:
    manifest = pack_dir / "PACK.yaml"
    try:
        data = load_file(manifest)
    except FileNotFoundError as exc:
        raise PackError(f"нет манифеста пакета: {manifest}") from exc
    except MiniYamlError as exc:
        raise PackError(f"{manifest}: невалидный YAML — {exc}") from exc
    if not isinstance(data, dict):
        raise PackError(f"{manifest}: ожидался объект")
    name = str(data.get("pack") or pack_dir.name)
    entries = data.get("exporters") or []
    if not isinstance(entries, list) or not entries:
        raise PackError(f"{manifest}: exporters — непустой список")
    exporters: list[dict[str, str]] = []
    for index, entry in enumerate(entries):
        if isinstance(entry, str):
            exporters.append({"module": entry})
        elif isinstance(entry, dict) and entry.get("module"):
            exporters.append({k: str(v) for k, v in entry.items()})
        else:
            raise PackError(f"{manifest}: exporters[{index}] — строка или {{module, class}}")
    return PackInfo(
        name=name, path=pack_dir, title=str(data.get("title") or name),
        exporters=tuple(exporters), fixtures=data.get("fixtures"),
    )


def discover_packs(packs_dir: str | Path | None = None) -> list[PackInfo]:
    base = Path(packs_dir) if packs_dir is not None else PACKS_DIR
    if not base.is_dir():
        return []
    packs: list[PackInfo] = []
    for child in sorted(base.iterdir()):
        if child.is_dir() and (child / "PACK.yaml").is_file():
            packs.append(load_manifest(child))
    return packs


def _load_module(path: Path):
    import sys

    repo_root = PACKS_DIR.parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    spec = importlib.util.spec_from_file_location(f"sensors_pack_{path.stem}_{abs(hash(str(path))) & 0xffff}", path)
    if spec is None or spec.loader is None:
        raise PackError(f"не удалось загрузить модуль {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def discover_exporters(
    packs_dir: str | Path | None = None,
    only: Optional[list[str]] = None,
) -> list[Any]:
    """Все экспортёры всех пакетов (или только указанных доменов)."""
    exporters: list[Any] = []
    wanted = set(only) if only else None
    for pack in discover_packs(packs_dir):
        if wanted is not None and pack.name not in wanted:
            continue
        for entry in pack.exporters:
            module_path = pack.path / f"{entry['module']}.py"
            if not module_path.is_file():
                raise PackError(f"{pack.name}: нет модуля экспортёра {module_path}")
            module = _load_module(module_path)
            class_name = entry.get("class")
            if class_name:
                cls = getattr(module, class_name, None)
                if cls is None:
                    raise PackError(f"{pack.name}/{entry['module']}: нет класса {class_name}")
                exporters.append(cls())
            elif hasattr(module, "EXPORTER"):
                exporters.append(module.EXPORTER)
            elif hasattr(module, "EXPORTERS"):
                exporters.extend(module.EXPORTERS)
            elif callable(getattr(module, "build_exporter", None)):
                exporters.append(module.build_exporter())
            else:
                raise PackError(
                    f"{pack.name}/{entry['module']}: нужен EXPORTER, EXPORTERS, build_exporter() или class в PACK.yaml"
                )
    return exporters


def fixture_path(pack_name: str, *parts: str, packs_dir: str | Path | None = None) -> Path:
    base = Path(packs_dir) if packs_dir is not None else PACKS_DIR
    return base / pack_name / "fixtures" / Path(*parts)


__all__ = [
    "PACKS_DIR",
    "PackError",
    "PackInfo",
    "load_manifest",
    "discover_packs",
    "discover_exporters",
    "fixture_path",
]
