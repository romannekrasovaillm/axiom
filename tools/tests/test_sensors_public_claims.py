"""S-030 public_claims — тесты датчика."""

from tools.sensors import public_claims as mod
from tools.sensors.fact import write_fact
from tools.sensors.subject import build_subject


def test_selftest_green():
    assert mod.run_selftest() == 0


def test_missing_card_unverified(tmp_path):
    written = mod.measure(card_path=tmp_path / "none.md", out_dir=tmp_path)
    assert written["public_card_matches_facts"]["status"] == "unverified"


def test_match_and_mismatch(tmp_path):
    subject = build_subject(repo_root=tmp_path, git_sha="a" * 40, dirty=False, device="cpu")
    write_fact("S-012", "tok_s_median_window", 800.0, unit="tok_s", quality="wrapped",
               method="f", subject=subject, out_dir=tmp_path)
    card = tmp_path / "card.md"
    card.write_text("800 tok/s", encoding="utf-8")
    assert mod.measure(card_path=card, out_dir=tmp_path)["public_card_matches_facts"]["value"] is True
    card.write_text("100 tok/s", encoding="utf-8")
    assert mod.measure(card_path=card, out_dir=tmp_path)["public_card_matches_facts"]["value"] is False
