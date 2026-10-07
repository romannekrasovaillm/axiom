"""S-018 wrap_gb10_load — тесты датчика."""

from tools.sensors import wrap_gb10_load as mod


def test_selftest_green():
    assert mod.run_selftest() == 0


def test_missing_evidence_unverified(tmp_path):
    written = mod.measure(tmp_path / "none.txt", out_dir=tmp_path)
    assert written["model_loads_count"]["status"] == "unverified"


def test_parses_evidence(tmp_path):
    ev = tmp_path / "gb10-single-load-x.txt"
    ev.write_text("Вывод:     OK: одна модельная нагрузка за раз (модельных нагрузок: 0)\n", encoding="utf-8")
    written = mod.measure(ev, out_dir=tmp_path)
    assert written["model_loads_count"]["value"] == 0
