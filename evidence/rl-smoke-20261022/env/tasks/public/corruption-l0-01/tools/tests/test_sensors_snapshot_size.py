"""S-006 snapshot_size — тесты датчика."""

from tools.sensors import snapshot_size
from tools.sensors._common import REPO_ROOT


def test_selftest_green():
    assert snapshot_size.run_selftest() == 0


def test_measure_deterministic(tmp_path):
    a = snapshot_size.measure(REPO_ROOT, out_dir=tmp_path)
    b = snapshot_size.measure(REPO_ROOT, out_dir=tmp_path)
    assert a["snapshot_bytes"]["value"] == b["snapshot_bytes"]["value"]
    assert a["snapshot_bytes"]["value"] > 0
