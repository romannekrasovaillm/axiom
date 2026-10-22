"""S-007 task_feasibility — тесты датчика."""

from tools.sensors import task_feasibility


def test_selftest_green():
    assert task_feasibility.run_selftest() == 0


def test_mechanics_without_verdicts(tmp_path):
    written = task_feasibility.measure(
        n=1, levels=("L0",), out_dir=tmp_path, run_verdicts=False
    )
    assert written["n_tasks"]["value"] == 1
    assert written["feasibility_rate"]["status"] == "unverified"
