"""E-5.3: канонический загрузчик токенизатора JaxLM-адаптера (§13).

Регрессия калибровки: ``--tokenizer-path`` стенда указывает на **манифест**
токенизатора AD-4 (``tools/bpe_train.py``: ``version/kind/vocab_size/merges/
specials/pad_id`` + ``tokenizer.file`` + ``tokenizer_hash``), а адаптер отдавал
этот манифест прямо в ``tokenizers.Tokenizer.from_file`` → падение с
``Unknown tokenizer version 'axiom-pretrain-tokenizer/1'``.

Пиннится: манифест → артефакт → ветка загрузчика; обязательная сверка
``tokenizer_hash`` (подмена артефакта после упаковки — отказ, AD-4); identity
фактического токенизатора в ``policy_version``.

Окружение без jax (системный python3): работают манифест-ветка и ветка
``tokenizers``; ветка JSON сети (``net.tokenizer.BPETokenizer``) требует
ML-стека — skip.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

CASE_DIR = Path(__file__).resolve().parents[2]
if str(CASE_DIR) not in sys.path:
    sys.path.insert(0, str(CASE_DIR))

from env.jaxlm_adapter import (  # noqa: E402
    JaxLMAdapter,
    TokenizerPin,
    load_tokenizer,
    resolve_tokenizer_pin,
)

tokenizers = pytest.importorskip(
    "tokenizers", reason="нужен пакет tokenizers: артефакт манифеста — tokens-json"
)

#: Корпус мини-артефакта: латиница + кириллица + код, чтобы round-trip был не вырожден.
CORPUS = [
    "hello world",
    "данные и код",
    "def foo(): return 42",
    "математика знание зрение",
    "token token token",
]
SPECIALS = ["<pad>", "<bos>", "<eos>", "<|code|>"]
MANIFEST_SCHEMA = "axiom-pretrain-tokenizer/1"


# --------------------------------------------------------------------------- #
# Мини-копия формата: артефакт ``tokenizers-json/v1`` + манифест AD-4
# --------------------------------------------------------------------------- #


def _train_artifact(path: Path) -> None:
    """Обучить крошечный байтовый BPE и сохранить артефакт ``tokenizers-json/v1``."""
    tok = tokenizers.Tokenizer(tokenizers.models.BPE(unk_token=None))
    tok.pre_tokenizer = tokenizers.pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = tokenizers.decoders.ByteLevel()
    trainer = tokenizers.trainers.BpeTrainer(
        vocab_size=300,
        special_tokens=list(SPECIALS),
        initial_alphabet=tokenizers.pre_tokenizers.ByteLevel.alphabet(),
    )
    tok.train_from_iterator(CORPUS, trainer)
    tok.save(str(path))


def _write_manifest(
    directory: Path,
    *,
    name: str = "tokenizer-manifest.json",
    artifact_name: str = "tokenizer.model",
    version: str = MANIFEST_SCHEMA,
    drop: tuple[str, ...] = (),
    **overrides,
) -> Path:
    """Мини-манифест формата ``tools/bpe_train.py`` с пином артефакта."""
    artifact = directory / artifact_name
    payload = {
        "version": version,
        "kind": "bpe",
        "seed": 0,
        "vocab_size": 512,
        "merges": 252,
        "specials": {name_: index for index, name_ in enumerate(SPECIALS)},
        "pad_id": 0,
        "bos_id": 1,
        "eos_id": 2,
        "code_prefix": "<|code|>",
        "code_prefix_id": 3,
        "tokenizer": {
            "file": artifact_name,
            "format": "tokenizers-json/v1",
            "bytes": artifact.stat().st_size if artifact.is_file() else 0,
        },
        "tokenizer_hash": (
            hashlib.sha256(artifact.read_bytes()).hexdigest()
            if artifact.is_file()
            else ""
        ),
    }
    payload.update(overrides)
    for key in drop:
        payload.pop(key, None)
    path = directory / name
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


@pytest.fixture()
def manifest_dir(tmp_path: Path) -> Path:
    """Каталог стенда: ``tokenizer.model`` + ``tokenizer-manifest.json`` рядом."""
    _train_artifact(tmp_path / "tokenizer.model")
    _write_manifest(tmp_path)
    return tmp_path


# --------------------------------------------------------------------------- #
# Манифест → артефакт: разворачивание и ветка загрузчика
# --------------------------------------------------------------------------- #


def test_raw_manifest_is_not_a_tokenizer_file(manifest_dir: Path) -> None:
    """Корень регрессии: сырой манифест ``from_file`` не принимает, загрузчик — да."""
    manifest = manifest_dir / "tokenizer-manifest.json"
    with pytest.raises(Exception) as excinfo:  # noqa: B017 — чужая библиотека, класс не нормируем
        tokenizers.Tokenizer.from_file(str(manifest))
    assert MANIFEST_SCHEMA in str(excinfo.value)

    tok = load_tokenizer(manifest)
    assert tok.decode(tok.encode("ok")) == "ok"


def test_manifest_resolves_artifact_and_round_trips(manifest_dir: Path) -> None:
    """Манифест разворачивается в артефакт; encode/decode дают исходный текст."""
    manifest = manifest_dir / "tokenizer-manifest.json"
    artifact = manifest_dir / "tokenizer.model"

    pin = resolve_tokenizer_pin(manifest)
    assert isinstance(pin, TokenizerPin)
    assert pin.manifest_path == manifest
    assert pin.artifact_path == artifact
    assert pin.tokenizer_hash == hashlib.sha256(artifact.read_bytes()).hexdigest()
    assert pin.vocab_size == 512
    assert pin.identity == f"{artifact.name}@{pin.tokenizer_hash[:12]}"

    tok = load_tokenizer(manifest)
    for text in CORPUS + ["Ünïcödé ✓", "  spaced  text "]:
        ids = tok.encode(text)
        assert all(isinstance(i, int) for i in ids), "шина ждёт list[int], не Encoding"
        assert ids, f"{text!r}: пустое кодирование — словарь не тот"
        assert tok.decode(ids) == text


def test_manifest_hash_pin_is_verified(manifest_dir: Path) -> None:
    """Подмена артефакта после упаковки ловится хешем (длина файла не меняется)."""
    artifact = manifest_dir / "tokenizer.model"
    blob = bytearray(artifact.read_bytes())
    blob[len(blob) // 2] ^= 0x01  # байт вместо байта: сверку ловит именно хеш
    artifact.write_bytes(bytes(blob))

    with pytest.raises(ValueError, match="tokenizer_hash"):
        load_tokenizer(manifest_dir / "tokenizer-manifest.json")


def test_manifest_without_pin_is_refused(manifest_dir: Path) -> None:
    """Манифест без ``tokenizer_hash`` — fail-closed: пин нечем проверить."""
    manifest = _write_manifest(manifest_dir, name="no-pin.json", drop=("tokenizer_hash",))
    with pytest.raises(ValueError, match="tokenizer_hash"):
        load_tokenizer(manifest)


def test_manifest_missing_artifact_is_clear_error(manifest_dir: Path) -> None:
    """Артефакт манифеста не найден — ошибка называет файл, а не падает в токенайзере."""
    manifest = _write_manifest(manifest_dir, name="missing.json", artifact_name="absent.model")
    with pytest.raises(FileNotFoundError, match="absent.model"):
        load_tokenizer(manifest)


def test_unsupported_manifest_schema_is_refused(manifest_dir: Path) -> None:
    manifest = _write_manifest(manifest_dir, name="v9.json", version="axiom-pretrain-tokenizer/9")
    with pytest.raises(ValueError, match="схема"):
        load_tokenizer(manifest)


def test_direct_artifact_path_still_loads(manifest_dir: Path) -> None:
    """Обратная совместимость: путь к самому артефакту (без манифеста) работает."""
    artifact = manifest_dir / "tokenizer.model"
    pin = resolve_tokenizer_pin(artifact)
    assert pin.manifest_path is None
    assert pin.tokenizer_hash is None
    assert pin.identity == artifact.name

    tok = load_tokenizer(artifact)
    assert tok.decode(tok.encode("abc")) == "abc"


def test_missing_path_is_clear_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="токенизатор не найден"):
        load_tokenizer(tmp_path / "nope.json")


# --------------------------------------------------------------------------- #
# Adapter §13: сборка и identity в policy_version
# --------------------------------------------------------------------------- #


class _PinnedFakeBackend:
    """Бэкенд §13 с пиннутым токенизатором: identity видна без jax/чекпойнта."""

    def __init__(
        self, *, identity: str = "", tokenizer_hash: str = "", sha: str = "a" * 64
    ) -> None:
        self._identity = identity
        self._tokenizer_hash = tokenizer_hash
        self._sha = sha

    @property
    def checkpoint_sha256(self) -> str:
        return self._sha

    @property
    def tokenizer_identity(self) -> str:
        return self._identity

    @property
    def tokenizer_hash(self) -> str:
        return self._tokenizer_hash

    def encode(self, text: str) -> list[int]:
        return [len(text) % 97]

    def decode(self, ids) -> str:
        return ",".join(str(int(i)) for i in ids)

    def generate_ids(self, prompt_ids, max_new_tokens, seed, temperature, top_p, no_repeat_ngram):
        return [1, 2, 3]

    def behavior_logprobs(self, prompt_ids, out_ids) -> list[float]:
        return [-0.5] * len(out_ids)


def test_adapter_builds_with_manifest_tokenizer_path(manifest_dir: Path) -> None:
    """Конструктор принимает манифест-путь и не трогает ML-стек до загрузки."""
    manifest = manifest_dir / "tokenizer-manifest.json"
    adapter = JaxLMAdapter("/nonexistent/ckpt", {"vocab_size": 512}, tokenizer_path=manifest)

    assert adapter._backend._tokenizer_path == manifest
    assert adapter._backend._loaded is False
    assert adapter._backend._tokenizer_pin is None


def test_policy_version_names_actual_tokenizer(manifest_dir: Path) -> None:
    """``policy_version`` называет путь/хеш фактического токенизатора, не пин чекпойнта."""
    pin = resolve_tokenizer_pin(manifest_dir / "tokenizer-manifest.json")
    adapter = JaxLMAdapter(
        backend=_PinnedFakeBackend(identity=pin.identity, tokenizer_hash=pin.tokenizer_hash)
    )

    assert adapter.tokenizer_identity == pin.identity
    assert adapter.tokenizer_hash == pin.tokenizer_hash
    assert adapter.policy_version == f"jax-{'a' * 12}+{pin.identity}"
    # Имя артефакта и его хеш — из фикстуры, а не выдуманы строкой.
    assert pin.artifact_path.name in adapter.policy_version
    assert pin.tokenizer_hash[:12] in adapter.policy_version


def test_policy_version_without_tokenizer_pin_is_unchanged() -> None:
    """Бэкенд без пиннутого токенизатора — прежний формат ``jax-<ckpt12>``."""
    adapter = JaxLMAdapter(backend=_PinnedFakeBackend(identity=""))
    assert adapter.tokenizer_identity == ""
    assert adapter.tokenizer_hash == ""
    assert adapter.policy_version == "jax-" + "a" * 12


def test_policy_version_explicit_pin_wins(manifest_dir: Path) -> None:
    pin = resolve_tokenizer_pin(manifest_dir / "tokenizer-manifest.json")
    adapter = JaxLMAdapter(
        backend=_PinnedFakeBackend(identity=pin.identity, tokenizer_hash=pin.tokenizer_hash),
        policy_version="jax-explicit",
    )
    assert adapter.policy_version == "jax-explicit"


# --------------------------------------------------------------------------- #
# Ветка JSON сети (``net.tokenizer.BPETokenizer``): требует ML-стека
# --------------------------------------------------------------------------- #


def test_net_bpe_json_artifact_branch(tmp_path: Path) -> None:
    """``specials``-список в JSON — артефакт сети, грузится ``BPETokenizer.load``."""
    pytest.importorskip("jax")
    from net import infer

    artifact = tmp_path / "net-bpe.json"
    infer.BPETokenizer(vocab_size=600).save(artifact)

    tok = load_tokenizer(artifact)
    assert isinstance(tok, infer.BPETokenizer)
    assert tok.decode(tok.encode("hello")) == "hello"


def test_manifest_pointing_to_net_bpe_artifact(tmp_path: Path) -> None:
    """Манифест со ссылкой на артефакт сети: ветка по содержимому, не по имени файла."""
    pytest.importorskip("jax")
    from net import infer

    artifact = tmp_path / "net-bpe.json"
    infer.BPETokenizer(vocab_size=600).save(artifact)
    manifest = _write_manifest(tmp_path, artifact_name="net-bpe.json")

    pin = resolve_tokenizer_pin(manifest)
    assert isinstance(pin.tokenizer, infer.BPETokenizer)
    assert pin.tokenizer_hash == hashlib.sha256(artifact.read_bytes()).hexdigest()
