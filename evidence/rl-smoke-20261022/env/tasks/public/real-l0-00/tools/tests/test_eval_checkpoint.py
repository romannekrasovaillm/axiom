"""Tests of ``tools/eval_checkpoint.py`` (ADR-055) — CPU-only, no checkpoint load.

Pinned properties:

* ``--help``/``--plan-only`` never import jax: the module under test has no
  top-level jax/numpy imports, and the plan path touches only the filesystem
  and the JSON config;
* pure helpers (``window_ids``, ``summarize``, ``parse_ckpt_spec``,
  ``plan_stream_counts``) behave as the contract says — full windows only,
  token-weighted mean (not mean of means), unique ``NAME=PATH``;
* every malformed input is fail-closed (exit 2), never a silent empty result;
* the report document has exactly schema ``axiom-eval-checkpoint/1``.

The tests do not load a checkpoint and do not run the model: the mandatory
acceptance route is the CPU plan + pure functions.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

TOOLS_DIR = Path(__file__).resolve().parents[1]
CASE_DIR = TOOLS_DIR.parent

# Пиннинг бэкенда до первого импорта jax (ADR-010) — тесты остаются файловыми.
os.environ.setdefault("NET_JAX_BACKEND", "cpu")

for _path in (str(CASE_DIR), str(TOOLS_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import eval_checkpoint as ec  # noqa: E402

TOOL = TOOLS_DIR / "eval_checkpoint.py"
SEQ = 16


# --------------------------------------------------------------------------- #
# Синтетика и запуск
# --------------------------------------------------------------------------- #


def _make_workspace(root: Path) -> dict[str, Path]:
    """Валидные синтетические входы: .npy, .bin и dummy <ckpt>/params."""
    domain = root / "domain.npy"
    ids = (np.arange(3 * SEQ + 5, dtype=np.uint32) % 4096).astype(np.uint32)
    np.save(domain, ids)

    shard = root / "W-00001.bin"
    (np.arange(2 * SEQ + 3, dtype="<u4") % 4096).astype("<u4").tofile(shard)

    ckpt = root / "ckpt-a"
    (ckpt / "params").mkdir(parents=True)
    (ckpt / "params" / "placeholder").write_text("dummy", encoding="utf-8")

    ckpt_b = root / "ckpt-b"
    (ckpt_b / "params").mkdir(parents=True)
    (ckpt_b / "params" / "placeholder").write_text("dummy", encoding="utf-8")
    return {"domain": domain, "shard": shard, "ckpt": ckpt, "ckpt_b": ckpt_b}


@pytest.fixture()
def ws(tmp_path: Path) -> dict[str, Path]:
    return _make_workspace(tmp_path)


def _base_args(ws: dict[str, Path], *, plan_only: bool = True) -> list[str]:
    args = [
        "--ckpt",
        f"a={ws['ckpt']}",
        "--domain-ids",
        str(ws["domain"]),
        "--w-shard",
        str(ws["shard"]),
        "--seq",
        str(SEQ),
        "--max-tokens",
        "50",
    ]
    if plan_only:
        args.insert(0, "--plan-only")
    return args


def _run_tool(args: list[str], cwd: Path = CASE_DIR) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env["NET_JAX_BACKEND"] = "cpu"
    return subprocess.run(
        [sys.executable, str(TOOL), *args],
        capture_output=True,
        text=True,
        cwd=str(cwd),
        env=env,
        timeout=120,
        check=False,
    )


def _run_python(code: str) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env["NET_JAX_BACKEND"] = "cpu"
    return subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=str(CASE_DIR),
        env=env,
        timeout=120,
        check=False,
    )


# --------------------------------------------------------------------------- #
# 1. --help / --plan-only без jax
# --------------------------------------------------------------------------- #


def test_help_works_and_module_has_no_top_level_jax():
    """``--help`` — код 0, а импорт модуля не тянет jax на верхнем уровне."""
    proc = _run_tool(["--help"])
    assert proc.returncode == 0, proc.stderr
    assert "usage:" in proc.stdout
    assert "--plan-only" in proc.stdout

    code = (
        "import contextlib, io, sys\n"
        f"sys.path.insert(0, {str(TOOLS_DIR)!r})\n"
        "import eval_checkpoint\n"
        "assert 'jax' not in sys.modules, 'top-level import pulled jax'\n"
        "try:\n"
        "    with contextlib.redirect_stdout(io.StringIO()):\n"
        "        eval_checkpoint.build_parser().parse_args(['--help'])\n"
        "except SystemExit as exc:\n"
        "    assert exc.code == 0, exc.code\n"
        "assert 'jax' not in sys.modules, '--help pulled jax'\n"
        "print('ok')\n"
    )
    proc = _run_python(code)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "ok"


def test_plan_only_returns_valid_json_and_does_not_import_jax(ws, tmp_path):
    out = tmp_path / "plan.json"
    code = (
        "import contextlib, io, json, sys\n"
        f"sys.path.insert(0, {str(TOOLS_DIR)!r})\n"
        "import eval_checkpoint as ec\n"
        f"args = {(_base_args(ws) + ['--out', str(out)])!r}\n"
        "buf = io.StringIO()\n"
        "with contextlib.redirect_stdout(buf):\n"
        "    rc = ec.main(args)\n"
        "assert rc == 0, rc\n"
        "assert 'jax' not in sys.modules, 'plan-only pulled jax'\n"
        "doc = json.loads(buf.getvalue())\n"
        "assert doc['schema'] == ec.REPORT_SCHEMA\n"
        "assert doc['mode'] == 'plan'\n"
        "print('ok')\n"
    )
    proc = _run_python(code)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "ok"


def test_plan_only_cli_prints_valid_json_and_writes_out(ws, tmp_path):
    out = tmp_path / "plan.json"
    proc = _run_tool(_base_args(ws) + ["--out", str(out)])
    assert proc.returncode == 0, proc.stderr

    doc = json.loads(proc.stdout)
    assert doc["schema"] == ec.REPORT_SCHEMA
    assert doc["mode"] == "plan"
    assert set(doc) == {
        "schema",
        "mode",
        "seq",
        "max_tokens",
        "checkpoints",
        "datasets",
        "meta",
    }
    assert set(doc["checkpoints"]) == {"a"}
    assert doc["checkpoints"]["a"]["path"] == str(ws["ckpt"])
    assert doc["checkpoints"]["a"]["params"] == str(ws["ckpt"] / "params")
    assert doc["checkpoints"]["a"]["params_present"] is True

    domain = doc["datasets"]["domain"]
    assert domain["tokens"] == 50  # max_tokens режет 53 -> 50
    assert domain["windows"] == 50 // SEQ
    assert domain["tail_tokens"] == 50 % SEQ
    assert domain["pred_tokens"] == (50 // SEQ) * (SEQ - 1)

    w_holdout = doc["datasets"]["w_holdout"]
    assert w_holdout["tokens"] == 2 * SEQ + 3
    assert w_holdout["windows"] == 2
    assert w_holdout["tail_tokens"] == 3

    assert doc["meta"]["w_shard"] == "W-00001.bin"
    assert doc["meta"]["w_shard_bytes"] == (2 * SEQ + 3) * 4
    assert doc["meta"]["domain_tail_tokens"] == 50 % SEQ
    assert doc["meta"]["w_tail_tokens"] == 3
    assert doc["meta"]["domain_ids_sha256"] == ec.domain_ids_sha256(
        np.load(ws["domain"])
    )
    assert doc["meta"]["config_sha256"] == hashlib.sha256(
        (CASE_DIR / "net" / "config.json").read_bytes()
    ).hexdigest()
    assert doc["meta"]["tokenizer_hash"] == ec.read_config_document().get(
        "tokenizer_hash"
    )
    assert doc["meta"]["git_sha"] is None or len(doc["meta"]["git_sha"]) == 40

    assert json.loads(out.read_text(encoding="utf-8")) == doc


def test_plan_only_without_w_shard_reports_none(ws, tmp_path):
    out = tmp_path / "plan-no-w.json"
    args = [
        "--plan-only",
        "--ckpt",
        f"a={ws['ckpt']}",
        "--domain-ids",
        str(ws["domain"]),
        "--seq",
        str(SEQ),
        "--max-tokens",
        "50",
        "--out",
        str(out),
    ]
    proc = _run_tool(args)
    assert proc.returncode == 0, proc.stderr
    doc = json.loads(out.read_text(encoding="utf-8"))
    assert doc["datasets"]["w_holdout"] is None
    assert doc["meta"]["w_shard"] == "none"
    assert doc["meta"]["w_shard_bytes"] == 0
    assert doc["meta"]["w_tokens"] == 0
    assert doc["meta"]["w_tail_tokens"] == 0


# --------------------------------------------------------------------------- #
# 2. Чистые функции
# --------------------------------------------------------------------------- #


def test_window_ids_keeps_only_full_windows_and_reports_tail():
    ids = np.arange(10)
    windows, tail = ec.window_ids(ids, 4)
    assert [window.tolist() for window in windows] == [
        [0, 1, 2, 3],
        [4, 5, 6, 7],
    ]
    assert tail == 2


def test_window_ids_exact_multiple_has_zero_tail():
    windows, tail = ec.window_ids(list(range(8)), 4)
    assert len(windows) == 2
    assert tail == 0


def test_window_ids_short_stream_has_no_windows():
    windows, tail = ec.window_ids(np.arange(3), 4)
    assert windows == []
    assert tail == 3


def test_window_ids_rejects_nonpositive_seq():
    with pytest.raises(ValueError):
        ec.window_ids(np.arange(4), 0)


def test_plan_stream_counts_caps_and_drops_tail():
    counts = ec.plan_stream_counts(total_tokens=53, seq=16, max_tokens=50)
    assert counts == {
        "tokens": 50,
        "windows": 3,
        "tail_tokens": 2,
        "pred_tokens": 3 * 15,
    }
    with pytest.raises(ValueError):
        ec.plan_stream_counts(total_tokens=10, seq=16, max_tokens=0)


def test_summarize_is_token_weighted_not_mean_of_means():
    stats = ec.summarize([1.0, 3.0], [10, 30])
    assert stats["loss"] == pytest.approx(2.5)  # (1*10 + 3*30) / 40
    assert stats["ppl"] == pytest.approx(math.exp(2.5))
    assert stats["windows"] == 2
    assert stats["pred_tokens"] == 40


def test_summarize_single_window_matches_exp():
    stats = ec.summarize([2.0], [7])
    assert stats["loss"] == pytest.approx(2.0)
    assert stats["ppl"] == pytest.approx(math.exp(2.0))
    assert stats["pred_tokens"] == 7


def test_summarize_is_fail_closed():
    with pytest.raises(ec.EvalError):
        ec.summarize([], [])
    with pytest.raises(ec.EvalError):
        ec.summarize([1.0, 2.0], [10])
    with pytest.raises(ec.EvalError):
        ec.summarize([1.0], [0])
    with pytest.raises(ec.EvalError):
        ec.summarize([float("nan")], [10])
    with pytest.raises(ec.EvalError):
        ec.summarize([float("inf")], [10])


def test_parse_ckpt_spec_preserves_order_and_allows_equals_in_path():
    parsed = ec.parse_ckpt_spec(["pretrain=/runs/a", "midtrain=/runs/b=c"])
    assert [(name, str(path)) for name, path in parsed] == [
        ("pretrain", "/runs/a"),
        ("midtrain", "/runs/b=c"),
    ]


def test_parse_ckpt_spec_is_fail_closed():
    with pytest.raises(ec.InputError):
        ec.parse_ckpt_spec([])
    with pytest.raises(ec.InputError):
        ec.parse_ckpt_spec(["no-equals"])
    with pytest.raises(ec.InputError):
        ec.parse_ckpt_spec(["=path"])
    with pytest.raises(ec.InputError):
        ec.parse_ckpt_spec(["name="])
    with pytest.raises(ec.InputError):
        ec.parse_ckpt_spec(["dup=/a", "dup=/b"])


# --------------------------------------------------------------------------- #
# 3. Fail-closed CLI: входы и чекпоинты
# --------------------------------------------------------------------------- #


def test_fail_closed_empty_npy(ws, tmp_path):
    empty = tmp_path / "empty.npy"
    np.save(empty, np.array([], dtype=np.uint32))
    args = _base_args(ws)
    args[args.index(str(ws["domain"]))] = str(empty)
    assert ec.main(args) == ec.EXIT_INPUT_ERROR


def test_fail_closed_broken_npy(ws, tmp_path):
    broken = tmp_path / "broken.npy"
    broken.write_bytes(b"not-an-npy")
    args = _base_args(ws)
    args[args.index(str(ws["domain"]))] = str(broken)
    assert ec.main(args) == ec.EXIT_INPUT_ERROR


def test_fail_closed_npy_ndim_not_one(ws, tmp_path):
    two_d = tmp_path / "two-d.npy"
    np.save(two_d, np.zeros((2, 3), dtype=np.uint32))
    args = _base_args(ws)
    args[args.index(str(ws["domain"]))] = str(two_d)
    assert ec.main(args) == ec.EXIT_INPUT_ERROR


def test_fail_closed_npy_wrong_dtype(ws, tmp_path):
    wrong = tmp_path / "int64.npy"
    np.save(wrong, np.arange(8, dtype=np.int64))
    args = _base_args(ws)
    args[args.index(str(ws["domain"]))] = str(wrong)
    assert ec.main(args) == ec.EXIT_INPUT_ERROR


def test_fail_closed_max_id_outside_vocab(ws, tmp_path):
    vocab = ec.config_vocab_size(ec.read_config_document())
    outside = tmp_path / "outside.npy"
    np.save(outside, np.array([0, vocab], dtype=np.uint32))
    args = _base_args(ws)
    args[args.index(str(ws["domain"]))] = str(outside)
    assert ec.main(args) == ec.EXIT_INPUT_ERROR


def test_fail_closed_empty_bin(ws, tmp_path):
    empty = tmp_path / "empty.bin"
    empty.write_bytes(b"")
    args = _base_args(ws)
    args[args.index(str(ws["shard"]))] = str(empty)
    assert ec.main(args) == ec.EXIT_INPUT_ERROR


def test_fail_closed_bin_not_multiple_of_uint32(ws, tmp_path):
    broken = tmp_path / "broken.bin"
    broken.write_bytes(b"\x00\x00\x00")
    args = _base_args(ws)
    args[args.index(str(ws["shard"]))] = str(broken)
    assert ec.main(args) == ec.EXIT_INPUT_ERROR


def test_fail_closed_bin_too_short_for_one_window(ws, tmp_path):
    short = tmp_path / "short.bin"
    np.arange(SEQ - 1, dtype="<u4").tofile(short)
    args = _base_args(ws)
    args[args.index(str(ws["shard"]))] = str(short)
    assert ec.main(args) == ec.EXIT_INPUT_ERROR


def test_fail_closed_duplicate_checkpoint_name(ws):
    args = _base_args(ws) + ["--ckpt", f"a={ws['ckpt_b']}"]
    assert ec.main(args) == ec.EXIT_INPUT_ERROR


def test_fail_closed_missing_checkpoint(ws, tmp_path):
    args = _base_args(ws)
    args[args.index(f"a={ws['ckpt']}")] = f"a={tmp_path / 'absent'}"
    assert ec.main(args) == ec.EXIT_INPUT_ERROR


def test_plan_only_marks_missing_params_dir_but_still_plans(ws, tmp_path, capsys):
    """Dummy-каталог без ``params/`` — валидный объект плана, но не прогона.

    План проверяет существование чекпоинта (каталог есть); грузимость
    (``params/``) требует сам прогон.  Отсутствие фиксируется флагом и
    предупреждением, а не молчанием.
    """
    bare = tmp_path / "no-params"
    bare.mkdir()
    args = _base_args(ws)
    args[args.index(f"a={ws['ckpt']}")] = f"a={bare}"
    assert ec.main(args) == ec.EXIT_OK
    doc = json.loads(capsys.readouterr().out)
    assert doc["checkpoints"]["a"]["params_present"] is False
    assert doc["checkpoints"]["a"]["params"] == str(bare / "params")


def test_fail_closed_plan_only_without_domain_ids(ws):
    args = ["--plan-only", "--ckpt", f"a={ws['ckpt']}", "--seq", str(SEQ)]
    assert ec.main(args) == ec.EXIT_INPUT_ERROR


def test_fail_closed_broken_ckpt_format(ws):
    args = ["--plan-only", "--ckpt", "broken", "--domain-ids", str(ws["domain"])]
    assert ec.main(args) == ec.EXIT_INPUT_ERROR


def test_fail_closed_max_tokens_smaller_than_seq(ws):
    args = _base_args(ws)
    args[args.index("50")] = str(SEQ - 1)
    assert ec.main(args) == ec.EXIT_INPUT_ERROR


# --------------------------------------------------------------------------- #
# 4. JSON-схема
# --------------------------------------------------------------------------- #


def _stat(loss: float, pred_tokens: int) -> dict[str, object]:
    return {
        "loss": loss,
        "ppl": math.exp(loss),
        "windows": 2,
        "pred_tokens": pred_tokens,
    }


def test_report_schema_keys_and_types():
    meta = {
        "w_shard": "W-00001.bin",
        "w_shard_bytes": 1234,
        "domain_ids_sha256": "a" * 64,
        "git_sha": "b" * 40,
        "tokenizer_hash": "c" * 64,
        "config_sha256": "d" * 64,
        "domain_tokens": 50,
        "domain_tail_tokens": 2,
        "w_tokens": 35,
        "w_tail_tokens": 3,
    }
    doc = ec.build_report(
        seq=16,
        max_tokens=50,
        checkpoints={
            "a": {
                "path": "/checkpoints/a",
                "domain": _stat(1.5, 30),
                "w_holdout": _stat(2.5, 20),
            }
        },
        meta=meta,
    )

    assert doc["schema"] == "axiom-eval-checkpoint/1"
    assert set(doc) == {"schema", "seq", "max_tokens", "checkpoints", "meta"}
    assert isinstance(doc["seq"], int) and doc["seq"] == 16
    assert isinstance(doc["max_tokens"], int) and doc["max_tokens"] == 50
    assert set(doc["checkpoints"]) == {"a"}
    entry = doc["checkpoints"]["a"]
    assert entry["path"] == "/checkpoints/a"
    for stream in ("domain", "w_holdout"):
        stat = entry[stream]
        assert set(stat) == {"loss", "ppl", "windows", "pred_tokens"}
        assert isinstance(stat["loss"], float)
        assert isinstance(stat["ppl"], float)
        assert isinstance(stat["windows"], int)
        assert isinstance(stat["pred_tokens"], int)
    assert doc["meta"] == meta


def test_report_allows_w_holdout_null():
    doc = ec.build_report(
        seq=8,
        max_tokens=8,
        checkpoints={
            "a": {
                "path": "/a",
                "domain": _stat(1.0, 7),
                "w_holdout": None,
            }
        },
        meta={},
    )
    assert doc["checkpoints"]["a"]["w_holdout"] is None


def test_build_meta_does_not_invent_absent_artifacts(tmp_path):
    ids = np.arange(4, dtype=np.uint32)
    meta = ec.build_meta(
        w_shard_path=None,
        w_shard_bytes=0,
        domain_ids=ids,
        config_path=tmp_path / "missing-config.json",
        config={},
        repo_root=tmp_path,  # не git-репозиторий
        domain_counts={"tokens": 4, "tail_tokens": 4},
        w_counts=None,
    )
    assert meta["w_shard"] == "none"
    assert meta["w_shard_bytes"] == 0
    assert meta["tokenizer_hash"] is None
    assert meta["config_sha256"] is None
    assert meta["git_sha"] is None  # не выдуман
    assert meta["domain_ids_sha256"] == hashlib.sha256(ids.tobytes()).hexdigest()
    assert meta["domain_tokens"] == 4
    assert meta["domain_tail_tokens"] == 4
    assert meta["w_tokens"] == 0
    assert meta["w_tail_tokens"] == 0
