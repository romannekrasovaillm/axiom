"""``--config-path`` in the SFT stage runner (VERIFICATION-LEG revision, E-4.1).

The leg runs two pretrains (dense + arch) from configs that must NOT be written
into ``net/config.json`` (the skeleton pin).  The runner contract is therefore
"an explicit config path overrides the preset", with ``None`` meaning the old
preset behaviour bit-for-bit.

What is pinned here:

* the flag parses (``Path``, default ``None``);
* ``build_model_config`` returns the declared file's architecture when the path
  is given and the frozen preset config when it is not — the presets themselves
  are untouched;
* the tokenizer pin and the journal's config provenance follow the *same* file
  the model is built from (reading the pin from another file would check the
  data against a config the run does not use);
* the error boundary: a broken or missing config file fails loudly instead of
  silently falling back to a preset.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

TOOLS_DIR = Path(__file__).resolve().parents[1]
CASE_DIR = TOOLS_DIR.parent
for _p in (str(TOOLS_DIR), str(CASE_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import run_sft_smoke as runner  # noqa: E402

DENSE = CASE_DIR / "net" / "config-dense124m.json"
ARCH = CASE_DIR / "net" / "config-arch124m.json"
PILOT = CASE_DIR / "net" / "config.json"


def test_flag_defaults_to_none_and_parses_a_path() -> None:
    assert runner.parse_args([]).config_path is None
    args = runner.parse_args(["--config-path", str(DENSE)])
    assert args.config_path == DENSE
    assert isinstance(args.config_path, Path)


def test_config_path_overrides_the_preset() -> None:
    dense = runner.build_model_config(256, "small", False, config_path=DENSE)
    assert dense.dense_standard_layers == 12
    assert dense.num_layers == 12 and dense.hidden == 768
    assert (dense.num_kda_layers, dense.num_mla_layers) == (0, 0)

    arch = runner.build_model_config(256, "small", False, config_path=ARCH)
    assert arch.dense_standard_layers == 0
    assert (arch.num_kda_layers, arch.num_mla_layers) == (9, 3)
    assert arch.num_layers == 12 and arch.hidden == 768


def test_presets_are_unchanged_without_the_flag() -> None:
    """``config_path=None`` is the previous behaviour — the frozen presets."""
    small = runner.build_model_config(256, "small", False)
    assert (small.num_layers, small.hidden) == (4, 64)
    full = runner.build_model_config(512, "l3-full", False)
    assert (full.num_layers, full.hidden) == (24, 1536)
    assert full.dense_standard_layers == 0


def test_stage_vocab_and_qat_overrides_still_apply() -> None:
    """The stage still owns vocab (data contract) and QAT (SFT stage) fields."""
    cfg = runner.build_model_config(1024, "small", True, config_path=DENSE)
    assert cfg.vocab_size == 1024
    assert cfg.qat_enabled is True
    # Architecture fields come from the file, not from the preset.
    assert cfg.num_layers == 12


def test_tokenizer_pin_follows_the_config_path() -> None:
    pilot_pin = runner.config_tokenizer_pin()
    assert pilot_pin == runner.config_tokenizer_pin(PILOT)  # default == pilot
    assert runner.config_tokenizer_pin(DENSE) == pilot_pin  # copied verbatim
    assert runner.config_tokenizer_pin(ARCH) == pilot_pin


def test_tokenizer_pin_reads_the_given_file(tmp_path) -> None:
    path = tmp_path / "cfg.json"
    path.write_text(json.dumps({"tokenizer_hash": "deadbeef"}), encoding="utf-8")
    assert runner.config_tokenizer_pin(path) == "deadbeef"
    # An absent pin is not silence-by-accident: it reads as empty.
    blank = tmp_path / "blank.json"
    blank.write_text("{}", encoding="utf-8")
    assert runner.config_tokenizer_pin(blank) == ""


def test_config_path_report_is_relative_and_hashed() -> None:
    report = runner.config_path_report(ARCH)
    assert not report["path"].startswith("/")
    assert report["path"].endswith("net/config-arch124m.json")
    assert len(report["sha256"]) == 64


def test_missing_config_file_fails_loudly(tmp_path) -> None:
    missing = tmp_path / "nope.json"
    with pytest.raises(FileNotFoundError):
        runner.load_config_path(missing)


def test_broken_config_file_fails_loudly(tmp_path) -> None:
    broken = tmp_path / "broken.json"
    broken.write_text("{not json", encoding="utf-8")
    with pytest.raises(json.JSONDecodeError):
        runner.load_config_path(broken)


def test_malformed_composition_is_not_silently_ignored(tmp_path) -> None:
    """A file that claims dense-standard but declares it wrongly is an error."""
    path = tmp_path / "bad-composition.json"
    path.write_text(json.dumps({
        "num_layers": 12, "num_kda_layers": 9, "num_mla_layers": 3,
        "layer_composition": {"kda": 8, "mla": 3},
    }), encoding="utf-8")
    with pytest.raises(ValueError):
        runner.load_config_path(path)
