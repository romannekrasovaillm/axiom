"""E-3: CPU-тесты eval-пакета претрейн-чекпойнта (SFT-STAGE §1.1).

Покрытие по задаче:

(а) holdout-сборка детерминирована (seed → идентичный файл);
(б) leak-фильтр отбрасывает текст с 12-граммовым пересечением с exclude;
(в) PPL на фиктивной модели (равномерное распределение над словарём 100) ≈ 100;
(г) адаптер-интерфейс соответствует §13 (структура ответа);
(д) отсутствие jax даёт понятную ошибку, не traceback.

GPU/сеть не используются: (а)/(б) — файловая синтетика в tmp, (в) — numpy-mock,
(г) — подменяемый бэкенд, (д) — ветка отсутствия jax (skip, если jax есть).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

CASE_DIR = Path(__file__).resolve().parents[2]
if str(CASE_DIR) not in sys.path:
    sys.path.insert(0, str(CASE_DIR))

from env import eval_holdout  # noqa: E402
from env import eval_ppl  # noqa: E402
from env import eval_prompts  # noqa: E402
from env.jaxlm_adapter import (  # noqa: E402
    RESULT_FIELDS,
    GenerationResult,
    JaxLMAdapter,
    JaxUnavailableError,
    jax_available,
    render_messages,
)

HAS_JAX = jax_available()


# --------------------------------------------------------------------------- #
# Вспомогательное
# --------------------------------------------------------------------------- #


def _write_jsonl(path: Path, texts: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps({"text": t}, ensure_ascii=False) + "\n" for t in texts),
        encoding="utf-8",
    )


def _words(prefix: str, count: int) -> str:
    return " ".join(f"{prefix}{i}" for i in range(count))


# --------------------------------------------------------------------------- #
# (а) Детерминизм сборки holdout
# --------------------------------------------------------------------------- #


def test_holdout_build_deterministic(tmp_path):
    source = tmp_path / "source.jsonl"
    _write_jsonl(source, [_words(f"doc{i}_", 30) for i in range(40)])
    out_a = tmp_path / "a.jsonl"
    out_b = tmp_path / "b.jsonl"

    report_a = eval_holdout.build_holdout(
        [source], out_a, share=0.5, min_texts=10, seed=7
    )
    report_b = eval_holdout.build_holdout(
        [source], out_b, share=0.5, min_texts=10, seed=7
    )

    assert out_a.read_bytes() == out_b.read_bytes()
    assert report_a["sha256"] == report_b["sha256"]
    assert report_a["n_texts"] == report_b["n_texts"] == 20
    assert report_a["seed"] == 7


def test_holdout_seed_changes_selection(tmp_path):
    source = tmp_path / "source.jsonl"
    _write_jsonl(source, [_words(f"doc{i}_", 30) for i in range(40)])
    out_a = tmp_path / "a.jsonl"
    out_b = tmp_path / "b.jsonl"
    eval_holdout.build_holdout([source], out_a, share=0.5, min_texts=10, seed=1)
    eval_holdout.build_holdout([source], out_b, share=0.5, min_texts=10, seed=2)
    # Разные seed — почти наверняка разные срезы; сравниваем содержимое.
    assert out_a.read_bytes() != out_b.read_bytes()


def test_holdout_records_have_text_and_source(tmp_path):
    source = tmp_path / "source.jsonl"
    _write_jsonl(source, [_words(f"doc{i}_", 30) for i in range(5)])
    out = tmp_path / "h.jsonl"
    report = eval_holdout.build_holdout([source], out, share=1.0, min_texts=5, seed=0)
    assert report["n_texts"] == 5
    for line in out.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        assert set(record) == {"text", "source_path"}
        assert record["source_path"] == str(source)


# --------------------------------------------------------------------------- #
# (б) Leak-фильтр: 12-граммовое пересечение с exclude
# --------------------------------------------------------------------------- #


def test_leak_filter_drops_shared_ngram(tmp_path):
    shared = _words("shared_", 12)
    clean = _words("clean_", 30)
    leaked = _words("pre_", 20) + " " + shared + " " + _words("post_", 20)
    source = tmp_path / "source.jsonl"
    _write_jsonl(source, [clean, leaked])

    exclude = tmp_path / "exclude.jsonl"
    _write_jsonl(exclude, [_words("other_", 15) + " " + shared])

    out = tmp_path / "h.jsonl"
    report = eval_holdout.build_holdout(
        [source], out, share=1.0, min_texts=2, seed=0, exclude=[exclude]
    )
    texts = [json.loads(line)["text"] for line in out.read_text(encoding="utf-8").splitlines()]
    assert report["n_texts"] == 1
    assert clean in texts
    assert leaked not in texts
    assert report["n_dropped_leak"] == 1


def test_no_exclude_keeps_everything(tmp_path):
    source = tmp_path / "source.jsonl"
    _write_jsonl(source, [_words("a_", 20), _words("b_", 20)])
    out = tmp_path / "h.jsonl"
    report = eval_holdout.build_holdout([source], out, share=1.0, min_texts=2, seed=0)
    assert report["n_texts"] == 2
    assert report["n_dropped_leak"] == 0


# --------------------------------------------------------------------------- #
# (в) PPL на mock-модели: равномерное распределение над словарём 100
# --------------------------------------------------------------------------- #


def _uniform_logits_fn(vocab: int = 100):
    import numpy as np

    def logits_fn(batch):
        b = len(batch)
        t = max(len(row) for row in batch)
        return np.zeros((b, t, vocab), dtype=np.float64)

    return logits_fn


def test_ppl_uniform_is_vocab_size():
    sentences = [
        list(range(1, 21)),
        list(range(3, 18)),
        list(range(50, 90)),
    ]
    result = eval_ppl.perplexity(_uniform_logits_fn(100), sentences, batch_size=1)
    assert result["ppl"] == pytest.approx(100.0, rel=0.01)
    assert result["n_texts"] == 3
    assert result["n_tokens"] > 0


def test_ppl_batching_matches_single():
    import random

    rng = random.Random(0)
    sentences = [[rng.randrange(1, 100) for _ in range(rng.randrange(5, 30))] for _ in range(9)]
    late = eval_ppl.perplexity(_uniform_logits_fn(100), sentences, batch_size=1)
    batched = eval_ppl.perplexity(_uniform_logits_fn(100), sentences, batch_size=4)
    assert batched["n_tokens"] == late["n_tokens"]
    assert batched["ppl"] == pytest.approx(late["ppl"], rel=1e-12)


def test_compute_ppl_formula():
    # N токенов с NLL = N*ln(V) → PPL = V.
    import math

    n = 10
    vocab = 50
    assert eval_ppl.compute_ppl(n * math.log(vocab), n) == pytest.approx(vocab, rel=1e-9)
    with pytest.raises(eval_ppl.PplError):
        eval_ppl.compute_ppl(0.0, 0)


def test_nll_from_logits_matches_manual():
    import math

    import numpy as np

    vocab = 4
    logits = np.zeros((1, 3, vocab))
    ids = np.array([[0, 1, 2]])
    nll, n_tokens = eval_ppl.nll_from_logits(logits, ids)
    assert n_tokens == 2  # предсказываются позиции 1 и 2
    assert nll == pytest.approx(2 * math.log(vocab), rel=1e-9)


# --------------------------------------------------------------------------- #
# (г) Адаптер §13: структура ответа
# --------------------------------------------------------------------------- #


class _FakeBackend:
    """Подменяемый бэкенд: интерфейс §13 без jax."""

    def __init__(self, vocab: int = 100) -> None:
        self.vocab = vocab

    @property
    def checkpoint_sha256(self) -> str:
        return "a" * 64

    def encode(self, text: str) -> list[int]:
        return [len(text) % self.vocab]

    def decode(self, ids) -> str:
        return "text:" + ",".join(str(int(i)) for i in ids)

    def generate_ids(self, prompt_ids, max_new_tokens, seed, temperature, top_p, no_repeat_ngram):
        return [1, 2, 3]

    def behavior_logprobs(self, prompt_ids, out_ids):
        return [-0.5] * len(out_ids)


def test_adapter_result_structure_matches_section13():
    adapter = JaxLMAdapter(backend=_FakeBackend())
    result = adapter.generate([{"role": "user", "content": "привет"}], seed=3, max_tokens=8)
    assert isinstance(result, GenerationResult)
    assert tuple(result.__dataclass_fields__) == RESULT_FIELDS
    assert isinstance(result.text, str)
    assert all(isinstance(t, int) for t in result.token_ids)
    assert all(isinstance(v, float) for v in result.behavior_logprobs)
    assert len(result.token_ids) == len(result.behavior_logprobs)
    assert isinstance(result.policy_version, str)
    assert result.policy_version == "jax-" + "a" * 12


def test_adapter_generate_signature_is_section13():
    import inspect

    params = list(inspect.signature(JaxLMAdapter.generate).parameters)
    assert params[:4] == ["self", "messages", "seed", "max_tokens"]


def test_render_messages_deterministic():
    messages = [
        {"role": "system", "content": "s"},
        {"role": "tool", "content": "<tool_response>x</tool_response>"},
    ]
    first = render_messages(messages)
    assert first == render_messages(messages)
    assert "<|system|>" in first and "<|tool|>" in first
    assert first.rstrip().endswith("<|assistant|>") or "<|assistant|>" in first


def test_prompts_are_twenty_and_unique():
    assert len(eval_prompts.PROMPTS) == 20
    assert len(set(eval_prompts.PROMPTS)) == 20
    assert all(p.strip() for p in eval_prompts.PROMPTS)


# --------------------------------------------------------------------------- #
# (д) Отсутствие jax — понятная ошибка, не traceback
# --------------------------------------------------------------------------- #


@pytest.mark.skipif(HAS_JAX, reason="jax доступен: ветка отсутствия не воспроизводится")
def test_adapter_without_jax_raises_clear_error(tmp_path):
    adapter = JaxLMAdapter(
        tmp_path / "ckpt", {"vocab_size": 100}
    )
    with pytest.raises(JaxUnavailableError) as excinfo:
        adapter.generate([{"role": "user", "content": "x"}], seed=0, max_tokens=4)
    assert "jax" in str(excinfo.value).lower()
    # Это RuntimeError, а не голый ImportError из недр импорта.
    assert isinstance(excinfo.value, RuntimeError)


@pytest.mark.skipif(HAS_JAX, reason="jax доступен: ветка отсутствия не воспроизводится")
def test_ppl_without_jax_raises_clear_error(tmp_path):
    holdout = tmp_path / "h.jsonl"
    _write_jsonl(holdout, [_words("a_", 20)])
    with pytest.raises(JaxUnavailableError) as excinfo:
        eval_ppl.run_ppl(
            tmp_path / "ckpt", holdout, None, config={"vocab_size": 100}
        )
    assert "jax" in str(excinfo.value).lower()


@pytest.mark.skipif(HAS_JAX, reason="jax доступен: ветка отсутствия не воспроизводится")
def test_prompts_cli_reports_error_without_jax(tmp_path, capsys):
    code = eval_prompts.main(
        [
            "--checkpoint", str(tmp_path / "ckpt"),
            "--config", str(_REPO_ROOT_CONFIG(tmp_path)),
            "--tokenizer", str(tmp_path / "tok.json"),
            "--out", str(tmp_path / "out.jsonl"),
        ]
    )
    assert code == 2
    err = capsys.readouterr().err
    assert "jax" in err.lower()


def _REPO_ROOT_CONFIG(tmp_path: Path) -> Path:
    """Минимальный JSON-конфиг для CLI-ветки (net/config.json не читаем без jax)."""
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"vocab_size": 100}), encoding="utf-8")
    return path
