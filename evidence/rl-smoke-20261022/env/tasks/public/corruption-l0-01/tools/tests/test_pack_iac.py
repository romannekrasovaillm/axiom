"""Тест доменного пакета iac (ADR-038, дельта G3)."""

from __future__ import annotations

import pytest

from tools.sensors.packs import discover_exporters, fixture_path
from tools.sensors.protocol import check_conformance

SUBJECT = {"git_sha": "a" * 40, "git_dirty": False, "device_kind": "cpu"}


def _by_id() -> dict:
    return {e.describe().id: e for e in discover_exporters(only=["iac"])}


def test_iac_exporters_conform():
    assert check_conformance(discover_exporters(only=["iac"]), subject=SUBJECT) == []


def test_iac001_k8s_manifests_values():
    pytest.importorskip("yaml")
    exp = _by_id()["IAC-001"]
    facts = {f.fact: f.value for f in exp.collect(SUBJECT, input_path=fixture_path("iac", "manifests.yaml"))}
    assert facts["deployment_count"] == 2
    assert facts["deployments_missing_requests"] == 1
    assert facts["deployments_missing_limits"] == 1
    assert facts["deployments_without_pdb"] == 1


def test_iac001_missing_source_unverified():
    exp = _by_id()["IAC-001"]
    facts = exp.collect(SUBJECT, input_path=fixture_path("iac", "nope.yaml"))
    assert all(f.status == "unverified" for f in facts)


def test_iac002_terraform_plan_values():
    exp = _by_id()["IAC-002"]
    facts = {f.fact: f.value for f in exp.collect(SUBJECT, input_path=fixture_path("iac", "plan.json"))}
    assert facts["resources_total"] == 4
    assert facts["create_count"] == 1
    assert facts["update_count"] == 1
    assert facts["delete_count"] == 1
    assert facts["replace_count"] == 1
    assert facts["resources_by_type"]["aws_instance"] == 2


def test_iac002_broken_json_unverified():
    exp = _by_id()["IAC-002"]
    facts = exp.collect(SUBJECT, input_path=fixture_path("iac", "manifests.yaml"))
    assert all(f.status == "unverified" for f in facts)
