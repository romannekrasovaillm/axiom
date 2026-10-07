"""S-014 wrap_bench — тесты датчика."""

from tools.sensors import wrap_bench as mod


def test_selftest_green():
    assert mod.run_selftest() == 0


def test_bench_fields_on_repo(tmp_path):
    written = mod.measure(out_dir=tmp_path)
    assert written["block_merge_cost_ratio"]["value"] is not None
    assert written["block_merge_recall"]["value"] is not None
    # evidence/kda-wyut отсутствует → честный unverified.
    assert written["kda_component_speedup"]["status"] == "unverified"
