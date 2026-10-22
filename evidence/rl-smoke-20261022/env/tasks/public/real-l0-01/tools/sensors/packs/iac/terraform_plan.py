"""Экспортёр IAC-002: terraform plan (дельта G3, ADR-038).

Читает JSON-вывод ``terraform show -json``: ресурсы по типам и число
создаваемых/удаляемых/изменяемых/заменяемых ресурсов. Только файл, без terraform.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from tools.sensors.protocol import BaseExporter, Fact, SensorSpec


class TerraformPlanExporter(BaseExporter):
    SPEC = SensorSpec(
        id="IAC-002",
        facts=("resources_total", "create_count", "update_count", "delete_count", "replace_count", "resources_by_type"),
        schema={
            "resources_total": {"unit": "count", "quality": "wrapped", "level": "component"},
            "create_count": {"unit": "count", "quality": "wrapped", "level": "end_to_end"},
            "update_count": {"unit": "count", "quality": "wrapped", "level": "end_to_end"},
            "delete_count": {"unit": "count", "quality": "wrapped", "level": "end_to_end"},
            "replace_count": {"unit": "count", "quality": "wrapped", "level": "component"},
            "resources_by_type": {"unit": "count", "quality": "wrapped", "level": "component"},
        },
        level="component",
        raw={"path": "<terraform-show.json>", "format": "json", "retention_days": None, "in_git": False},
        context=("repo",),
        pack="iac",
    )

    def collect(self, subject: dict[str, Any], *, input_path: Any = None, **_: Any) -> list[Fact]:
        if not input_path:
            return [self.unavailable(f, subject, "не передан файл terraform plan (input_path)") for f in self.SPEC.facts]
        path = Path(input_path)
        if not path.is_file():
            return [self.unavailable(f, subject, f"plan не найден: {path}") for f in self.SPEC.facts]
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return [self.unavailable(f, subject, f"plan не читается как JSON: {path}") for f in self.SPEC.facts]
        if not isinstance(data, dict):
            return [self.unavailable(f, subject, "plan — не объект JSON") for f in self.SPEC.facts]

        changes = data.get("resource_changes") or []
        by_type: Counter[str] = Counter()
        create = update = delete = replace = 0
        for change in changes:
            if not isinstance(change, dict):
                continue
            rtype = str(change.get("type") or "unknown")
            by_type[rtype] += 1
            actions = (change.get("change") or {}).get("actions") or []
            actions = [str(a) for a in actions]
            if actions == ["no-op"]:
                continue
            if "create" in actions and "delete" in actions:
                replace += 1
            elif "create" in actions:
                create += 1
            elif "delete" in actions:
                delete += 1
            elif "update" in actions:
                update += 1

        method = f"resource_changes из {path.name}: {len(changes)} записей"
        return [
            self.fact("resources_total", len(changes), subject=subject, method=method),
            self.fact("create_count", create, subject=subject, method=method),
            self.fact("update_count", update, subject=subject, method=method),
            self.fact("delete_count", delete, subject=subject, method=method),
            self.fact("replace_count", replace, subject=subject, method=method),
            self.fact("resources_by_type", dict(sorted(by_type.items())), subject=subject, method=method),
        ]


EXPORTER = TerraformPlanExporter()
