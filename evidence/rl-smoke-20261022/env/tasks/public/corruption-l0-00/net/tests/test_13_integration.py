"""net ↔ env integration wire: task spec → model executor → verdict → manifest.

End-to-end on the tiny config: a corruption-type task spec is solved by the
(untrained) network through ``env.net_executor`` — prompt tokenisation
(byte-level BPE, net/tokenizer.py) → seeded generation (net/infer.py, NTP
head) → detokenised answer into the workspace → mechanical verdict (FAIL is
expected and valid: the wire, not the solution, is under test) → Run Manifest
written and accepted by the loader.  Pinning (AD-4): model snapshot hash of
the config JSON, routing seed, decoding parameters.  A rerun with the same
seed produces an identical manifest (A5 determinism).

The tiny model's embedding is 512-wide, so the tokenizer is trained with a
matching 512 vocabulary — same byte-level BPE algorithm/class as the pinned
BPE-160K, reduced vocab like the other tiny fixtures.
"""

from __future__ import annotations

import dataclasses
import os
from pathlib import Path

import pytest

import jax.random as jr

from env import generate as env_generate
from env import manifest as manifest_mod
from env import net_executor
from env.util import EMPTY_HIDDEN_SHA256, is_sha256_hex, read_json
from net import infer, model
from net.data import synthetic_corpus_texts
from net.tokenizer import BPETokenizer

CASE_DIR = Path(__file__).resolve().parents[2]

MAX_NEW_TOKENS = 32
DECODING_DEFAULT = {"temperature": 0.7, "top_p": 0.95}


def _find_bin() -> str | None:
    env = os.environ.get("ENV_ARCH_ML_BIN")
    if env and Path(env).is_file():
        return env
    repo_bin = CASE_DIR.parent.parent / "target" / "release" / "arch-ml"
    if repo_bin.is_file():
        return str(repo_bin)
    from env.verifier import arch_ml_available

    if arch_ml_available("arch-ml"):
        return "arch-ml"
    return None


def _tiny_setup(tiny_cfg):
    """Tiny config + matching-vocab tokenizer + seeded params."""
    cfg = dataclasses.replace(tiny_cfg, vocab_size=512)
    tok = BPETokenizer(vocab_size=cfg.vocab_size).train(
        synthetic_corpus_texts(seed=0, n_docs=4, words_per_doc=16), seed=0
    )
    params = model.init_params(jr.PRNGKey(0), cfg)
    return cfg, tok, params


def _make_task(tmp_path: Path):
    """One corruption-type L0 task spec + its base workspace."""
    data_dir = tmp_path / "tasks"
    env_generate.generate(
        CASE_DIR, data_dir, seed=0,
        grid=(("corruption", "L0", 1),), holdout_count=0,
    )
    spec = read_json(data_dir / "public" / "corruption-l0-00.json")
    return spec, data_dir / "public" / "corruption-l0-00"


def _run_task(spec, base_ws, out_dir, cfg, tok, params, bin):
    decoding = {**DECODING_DEFAULT, "seed": int(spec["seed"])}
    return net_executor.run_net_task(
        spec, base_ws, out_dir,
        params=params, cfg=cfg, tokenizer=tok, decoding=decoding,
        max_new_tokens=MAX_NEW_TOKENS, bin=bin,
        run_id="test-net-integration",
        started="2026-09-12T00:00:00+00:00",
        finished="2026-09-12T00:00:01+00:00",
        model_base="net-l3-skeleton-tiny-untrained",
        host="test-host",
    )


def _bytecode_files(ws_dir: Path) -> list[str]:
    """Относительные пути байткод-кеша в снапшоте (``__pycache__``, ``*.pyc``).

    Байткод несёт mtime исходника и его абсолютный путь (``co_filename``),
    поэтому снапшот с ним даёт разные ``workspace_sha256`` у двух одинаковых
    прогонов в разные каталоги — A5 требует пустоты этого списка.
    """
    out: list[str] = []
    for p in sorted(ws_dir.rglob("*")):
        rel = p.relative_to(ws_dir).as_posix()
        if p.is_dir() and p.name == net_executor.BYTECODE_DIR:
            out.append(rel)
        elif p.is_file() and p.name.endswith(net_executor.BYTECODE_SUFFIX):
            out.append(rel)
    return out


def test_workspace_snapshot_has_no_pyc(tmp_path, tiny_cfg):
    """Мини-прогон (те же фикстуры, без гейтов): снапшот без байткод-кеша.

    Полный прогон с гейтами проверяет то же в
    ``test_end_to_end_manifest_and_determinism``; здесь — сам снапшот: байткод
    в базовом кейсе (кеш рядом с исходником и «осиротевший» ``*.pyc`` старого
    layout) в снапшот не попадает, а источник не мутируется. Гейт исполняет
    ``command_succeeds`` внутри workspace и оставляет ``__pycache__/*.pyc``,
    маршалящий абсолютный путь исходника, — отсюда недетерминированный
    ``workspace_sha256`` до фикса.
    """
    cfg, tok, params = _tiny_setup(tiny_cfg)
    spec, base_ws = _make_task(tmp_path)

    cache = base_ws / "net" / net_executor.BYTECODE_DIR
    cache.mkdir(parents=True)
    (cache / "config.cpython-311.pyc").write_bytes(b"\x00fake-bytecode")
    stray = base_ws / "net" / "legacy.pyc"
    stray.write_bytes(b"\x00fake-bytecode")

    out_dir = tmp_path / "ws-snapshot"
    net_executor.run_net_model(
        spec, base_ws, out_dir,
        params=params, cfg=cfg, tokenizer=tok,
        decoding={**DECODING_DEFAULT, "seed": int(spec["seed"])},
        max_new_tokens=MAX_NEW_TOKENS,
    )

    assert _bytecode_files(out_dir) == []
    # Кейс скопирован (снапшот — копия, а не пустое дерево), источник цела.
    assert (out_dir / "CONSTRAINTS.yaml").is_file()
    assert (out_dir / net_executor.RESPONSE_FILE).is_file()
    assert cache.is_dir() and stray.is_file()


def test_generate_greedy_deterministic(tiny_cfg):
    """Same seed → identical token sequence; greedy is seed-independent."""
    cfg, _, params = _tiny_setup(tiny_cfg)
    ids = [4, 10, 20]
    a = infer.generate(params, cfg, ids, 8, 0.0, seed=0)
    b = infer.generate(params, cfg, ids, 8, 0.0, seed=999)
    assert a == b
    assert 0 < len(a) <= 8
    assert all(0 <= t < cfg.vocab_size for t in a)


def test_generate_sampled_seed_pinned(tiny_cfg):
    """Temperature sampling is deterministic under the pinned seed."""
    cfg, _, params = _tiny_setup(tiny_cfg)
    ids = [4, 10, 20]
    a = infer.generate(params, cfg, ids, 8, 0.7, seed=42, top_p=0.95)
    b = infer.generate(params, cfg, ids, 8, 0.7, seed=42, top_p=0.95)
    assert a == b
    assert len(a) <= 8


def test_end_to_end_manifest_and_determinism(tmp_path, tiny_cfg):
    """task spec → net executor → verdict (FAIL ok) → manifest (pinned, A5)."""
    bin = _find_bin()
    if bin is None:
        pytest.skip("arch-ml бинарь недоступен (нет ENV_ARCH_ML_BIN/target/release/arch-ml)")

    cfg, tok, params = _tiny_setup(tiny_cfg)
    spec, base_ws = _make_task(tmp_path)

    rr1, m1 = _run_task(spec, base_ws, tmp_path / "ws-a", cfg, tok, params, bin)
    rr2, m2 = _run_task(spec, base_ws, tmp_path / "ws-b", cfg, tok, params, bin)

    # Модель необучена: вердикт FAIL ожидаем — валиден сам провод.
    assert rr1.verdict.passed is False
    assert m1["verdict"]["pass"] is False

    # A5: повторный прогон с тем же seed → идентичный манифест.
    assert m1 == m2

    # A5-механизм: гейт исполняет command_succeeds внутри workspace и оставляет
    # там байткод-кеш с абсолютным путём исходника — снапшот его не несёт
    # (иначе два прогона в разные каталоги расходятся по workspace_sha256).
    assert _bytecode_files(tmp_path / "ws-a") == []
    assert _bytecode_files(tmp_path / "ws-b") == []

    # Манифест записывается и проходит валидацию загрузчика (AD-4).
    mpath = tmp_path / "run-manifest.json"
    manifest_mod.dump_manifest(m1, mpath)
    loaded = manifest_mod.load_manifest(mpath)
    assert loaded == m1

    # Пиннинг: снапшот модели, routing seed, decoding-параметры.
    assert is_sha256_hex(m1["model"]["snapshot_sha256"])
    assert m1["model"]["snapshot_sha256"] == net_executor.model_snapshot_sha256(cfg)
    assert m1["model"]["routing_seed"] == cfg.routing_seed
    assert m1["decoding"]["seed"] == int(spec["seed"])
    assert m1["decoding"]["temperature"] == DECODING_DEFAULT["temperature"]
    assert m1["decoding"]["top_p"] == DECODING_DEFAULT["top_p"]
    assert m1["decoding"]["max_new_tokens"] == MAX_NEW_TOKENS
    assert m1["usage"]["tokens_in"] > 0
    assert m1["usage"]["tokens_out"] > 0
    # L0-задача без скрытых правил: хеш пустого holdout-набора.
    assert spec["verifier"]["hidden_constraints_sha256"] == EMPTY_HIDDEN_SHA256
