"""S-020 power_thermal — тесты датчика."""

from tools.sensors import power_thermal as mod


def test_selftest_green():
    assert mod.run_selftest() == 0


def test_parse_and_median(tmp_path):
    series = mod.parse_csv("10, 100, 30, 50\n20, 200, 40, 70\n")
    assert series["power_draw_w"] == [10.0, 20.0]
    written = mod.measure("10, 100, 30, 50\n20, 200, 40, 70\n", out_dir=tmp_path)
    assert written["sm_clock_mhz"]["value"] == 150.0


def test_empty_unverified(tmp_path):
    written = mod.measure("", out_dir=tmp_path)
    assert written["temperature_c"]["status"] == "unverified"
