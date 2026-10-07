"""Экспортёр SVC-001: поверхность OpenAPI 3 (дельта G3, ADR-038).

Читает файл OpenAPI 3 (JSON; YAML — если доступен PyYAML) и считает число
эндпоинтов, объявленных схем аутентификации и эндпоинтов без аутентификации.
Демонстрация универсальности протокола на стандартном артефакте; живых
сервисов и сети не требует.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from tools.sensors.protocol import BaseExporter, Fact, SensorSpec

_HTTP_METHODS = ("get", "put", "post", "delete", "options", "head", "patch", "trace")


def _load_spec(path: Path) -> Optional[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        try:
            import yaml  # noqa: PLC0415 — YAML-ветка необязательна

            data = yaml.safe_load(text)
        except Exception:  # noqa: BLE001
            return None
    return data if isinstance(data, dict) else None


class OpenApiSurfaceExporter(BaseExporter):
    SPEC = SensorSpec(
        id="SVC-001",
        facts=("endpoint_count", "auth_schemes_declared", "endpoints_without_auth"),
        schema={
            "endpoint_count": {"unit": "count", "quality": "measured", "level": "end_to_end"},
            "auth_schemes_declared": {"unit": "count", "quality": "measured", "level": "component"},
            "endpoints_without_auth": {"unit": "count", "quality": "measured", "level": "end_to_end"},
        },
        level="end_to_end",
        raw={"path": "<openapi.json>", "format": "json", "retention_days": None, "in_git": False},
        context=("repo",),
        pack="service",
    )

    def collect(self, subject: dict[str, Any], *, input_path: Any = None, **_: Any) -> list[Fact]:
        if not input_path:
            return [self.unavailable(f, subject, "не передан файл OpenAPI (input_path)") for f in self.SPEC.facts]
        path = Path(input_path)
        if not path.is_file():
            return [self.unavailable(f, subject, f"файл OpenAPI не найден: {path}") for f in self.SPEC.facts]
        data = _load_spec(path)
        if data is None:
            return [self.unavailable(f, subject, f"OpenAPI не читается: {path}") for f in self.SPEC.facts]

        paths = data.get("paths") if isinstance(data.get("paths"), dict) else {}
        global_security = data.get("security") or []
        endpoints = 0
        without_auth = 0
        for item in paths.values():
            if not isinstance(item, dict):
                continue
            for method, operation in item.items():
                if method.lower() not in _HTTP_METHODS or not isinstance(operation, dict):
                    continue
                endpoints += 1
                effective = operation.get("security", global_security)
                if not effective:
                    without_auth += 1

        components = data.get("components") if isinstance(data.get("components"), dict) else {}
        schemes = components.get("securitySchemes") if isinstance(components.get("securitySchemes"), dict) else {}

        return [
            self.fact("endpoint_count", endpoints, subject=subject, method=f"разбор {path.name}"),
            self.fact("auth_schemes_declared", len(schemes), subject=subject, method=f"разбор {path.name}"),
            self.fact("endpoints_without_auth", without_auth, subject=subject, method=f"разбор {path.name}"),
        ]


EXPORTER = OpenApiSurfaceExporter()
