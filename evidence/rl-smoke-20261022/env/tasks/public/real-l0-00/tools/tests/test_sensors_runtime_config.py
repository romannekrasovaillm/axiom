"""S-001 runtime_config — тесты датчика."""

from tools.sensors import runtime_config
from tools.sensors._common import REPO_ROOT


def test_selftest_green():
    assert runtime_config.run_selftest() == 0


def test_measure_writes_declared_structure(tmp_path):
    written = runtime_config.measure(REPO_ROOT / "net" / "config.json", out_dir=tmp_path)
    assert written["num_kda_layers"]["value"] == 18
    assert written["num_mla_layers"]["value"] == 6
    assert written["moe_top_k"]["value"] == 2
    assert written["moe_routed"]["value"] == 12
    assert written["moe_shared"]["value"] == 2
    assert written["vocab_size"]["value"] == 160000
    assert written["kda_impl_selected"]["value"] == "chunked"
    assert (tmp_path / "S-001.jsonl").is_file()
