"""S-002 param_count — тесты датчика."""

from tools.sensors import param_count
from tools.sensors._common import REPO_ROOT, load_config_json


def test_selftest_green():
    assert param_count.run_selftest() == 0


def test_measure_matches_declaration(tmp_path):
    declared = load_config_json(REPO_ROOT / "net" / "config.json")
    written = param_count.measure(REPO_ROOT / "net" / "config.json", out_dir=tmp_path)
    assert written["params_total"]["value"] == declared["actual_param_count"]
    assert written["params_active"]["value"] == declared["active_params_per_token"]
    assert written["params_total"]["note"] == ""
