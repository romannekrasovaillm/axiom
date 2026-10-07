"""Экспортёр IAC-001: манифесты Kubernetes (дельта G3, ADR-038).

По YAML-манифестам считает: число Deployment, наличие requests/limits у
контейнеров и наличие PodDisruptionBudget по Deployment. Только файлы, без
кластера и сети.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from tools.sensors.protocol import BaseExporter, Fact, SensorSpec


def _load_documents(path: Path) -> Optional[list[dict[str, Any]]]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None
    try:
        import yaml  # noqa: PLC0415 — YAML-разбор манифестов

        docs = list(yaml.safe_load_all(text))
    except Exception:  # noqa: BLE001
        return None
    return [d for d in docs if isinstance(d, dict)]


def _selector_labels(spec: Any) -> dict[str, str]:
    if not isinstance(spec, dict):
        return {}
    match = spec.get("matchLabels")
    return match if isinstance(match, dict) else {}


def _covers(subset: dict[str, str], superset: dict[str, str]) -> bool:
    return bool(subset) and all(superset.get(k) == v for k, v in subset.items())


class K8sManifestsExporter(BaseExporter):
    SPEC = SensorSpec(
        id="IAC-001",
        facts=("deployment_count", "deployments_missing_requests", "deployments_missing_limits", "deployments_without_pdb"),
        schema={
            "deployment_count": {"unit": "count", "quality": "measured", "level": "component"},
            "deployments_missing_requests": {"unit": "count", "quality": "measured", "level": "end_to_end"},
            "deployments_missing_limits": {"unit": "count", "quality": "measured", "level": "end_to_end"},
            "deployments_without_pdb": {"unit": "count", "quality": "measured", "level": "end_to_end"},
        },
        level="component",
        raw={"path": "<manifests.yaml>", "format": "text", "retention_days": None, "in_git": False},
        context=("repo",),
        pack="iac",
    )

    def collect(self, subject: dict[str, Any], *, input_path: Any = None, **_: Any) -> list[Fact]:
        if not input_path:
            return [self.unavailable(f, subject, "не передан файл манифестов (input_path)") for f in self.SPEC.facts]
        path = Path(input_path)
        if not path.is_file():
            return [self.unavailable(f, subject, f"манифест не найден: {path}") for f in self.SPEC.facts]
        docs = _load_documents(path)
        if docs is None:
            return [self.unavailable(f, subject, f"манифест не читается (нет PyYAML?): {path}") for f in self.SPEC.facts]

        deployments = [d for d in docs if str(d.get("kind", "")).lower() == "deployment"]
        pdbs = [d for d in docs if str(d.get("kind", "")).lower() == "poddisruptionbudget"]
        pdb_selectors = [_selector_labels(p.get("spec", {}).get("selector", {})) for p in pdbs]

        missing_requests = 0
        missing_limits = 0
        without_pdb = 0
        for dep in deployments:
            containers = (
                dep.get("spec", {}).get("template", {}).get("spec", {}).get("containers", []) or []
            )
            need_req = False
            need_lim = False
            for container in containers:
                if not isinstance(container, dict):
                    continue
                resources = container.get("resources") or {}
                if not (resources.get("requests") or {}):
                    need_req = True
                if not (resources.get("limits") or {}):
                    need_lim = True
            missing_requests += int(need_req)
            missing_limits += int(need_lim)

            own = _selector_labels(dep.get("spec", {}).get("selector", {}))
            if not any(_covers(own, s) for s in pdb_selectors):
                without_pdb += 1

        method = f"разбор {path.name}: deployments={len(deployments)}, pdb={len(pdbs)}"
        return [
            self.fact("deployment_count", len(deployments), subject=subject, method=method),
            self.fact("deployments_missing_requests", missing_requests, subject=subject, method=method),
            self.fact("deployments_missing_limits", missing_limits, subject=subject, method=method),
            self.fact("deployments_without_pdb", without_pdb, subject=subject, method=method),
        ]


EXPORTER = K8sManifestsExporter()
