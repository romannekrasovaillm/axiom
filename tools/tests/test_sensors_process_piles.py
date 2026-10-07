"""S-010 process_piles — тесты датчика."""

from tools.sensors import process_piles
from tools.sensors._common import REPO_ROOT


def test_selftest_green():
    assert process_piles.run_selftest() == 0


def test_classifier():
    assert process_piles._classify_paths(["CONSTRAINTS.yaml"], False) == process_piles.BEHAVIOUR
    assert process_piles._classify_paths(["docs/adr/ADR-001-x.md"], False) == process_piles.DOCS
    assert process_piles._classify_paths(["net/model.py"], False) == process_piles.WORK


def test_measure_on_repo(tmp_path):
    written = process_piles.measure(REPO_ROOT, out_dir=tmp_path)
    for key in ("docs_share", "behaviour_share", "work_share"):
        assert written[key]["status"] == "ok"
        assert 0.0 <= written[key]["value"] <= 1.0
    assert written["adr_numbering_collisions"]["value"] == 0
