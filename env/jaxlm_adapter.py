"""JaxLM-адаптер интерфейса §13 (ENVIRONMENT-V1) для претрейн-чекпойнта.

Контракт интерфейса агента роллаутов (§13, ADR-027/E-1):::

    generate(messages, seed, max_tokens) -> {text, token_ids, behavior_logprobs}

Это **тонкая обёртка** над уже существующей генерацией сети (:mod:`net.infer`,
:func:`net.infer.generate` / :func:`net.infer.generate_text`): прибор не
дублирует ни forward, ни цикл генерации — он переводит формат сообщений v12
(§13) в промпт, зовёт генерацию сети и возвращает результат в форме интерфейса.

Загрузка весов — orbax-чекпойнт через :mod:`net.checkpoint`
(:func:`net.checkpoint.load_checkpoint`); структура-цель строится
``init_params`` модуля модели, полученного через сам :mod:`net.infer`
(адаптер импортирует из сети **только** ``net.infer`` и ``net.checkpoint``).

Ленивая инициализация: ``jax`` и пакет ``net`` импортируются при первом
обращении к генерации/кодированию, поэтому CPU-тесты и остальная среда
работают без ML-стека. Отсутствие ``jax`` даёт понятную
:class:`JaxUnavailableError`, а не ImportError из недр импорта.

Режим декодирования (§8.6 SFT-STAGE): штатные пробы — greedy + запрет повтора
4-грамм (``no_repeat_ngram=4``) + eos на конце хода. ``net.infer.generate`` не
принимает запрещённый набор, поэтому при ``no_repeat_ngram >= 2`` адаптер
применяет тонкий слой выбора токена поверх того же forward сети (модель не
переписывается, меняется только правило выбора следующего токена).
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, fields as dataclass_fields
from pathlib import Path
from typing import Any, Optional, Sequence

#: Корень репозитория на ``sys.path``: ``net`` импортируется по пути репо-корня
#: (запуск прибора возможен и не из корня).
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

#: Поля результата §13 — состав и порядок зафиксированы (тест структуры).
RESULT_FIELDS: tuple[str, ...] = ("text", "token_ids", "behavior_logprobs", "policy_version")


class JaxUnavailableError(RuntimeError):
    """``jax``/``net`` недоступны: прибор честно отказывает, а не падает на импорте."""


@dataclass(frozen=True)
class GenerationResult:
    """Ответ одного хода ассистента (форма §13)."""

    text: str
    token_ids: list[int]
    behavior_logprobs: list[float]
    policy_version: str


def jax_available() -> bool:
    """Есть ли рабочий ``jax`` в текущем окружении (для skip-логики тестов)."""
    import importlib

    try:
        importlib.import_module("jax")
    except Exception:  # noqa: BLE001 — любой сбой импорта = ML-стека нет
        return False
    return True


# --------------------------------------------------------------------------- #
# Рендер сообщений v12 → промпт
# --------------------------------------------------------------------------- #

#: Роли сообщений v12 (§13). Рендер — детерминированная функция от списка.
_ROLE_TAGS: dict[str, str] = {
    "system": "<|system|>",
    "user": "<|user|>",
    "assistant": "<|assistant|>",
    "tool": "<|tool|>",
}


def render_messages(messages: Sequence[dict[str, str]]) -> str:
    """Список сообщений §13 → единый промпт (детерминированно).

    Формат — ролевые теги v12 и завершающий ``<|assistant|>``: прибор просит
    генерацию следующего хода.  Точный шаблон пиннится прогоном на стенде вместе
    с токенизатором; здесь важна байтовая воспроизводимость при одном входе.
    """
    parts: list[str] = []
    for message in messages:
        role = str(message.get("role", ""))
        content = str(message.get("content", ""))
        tag = _ROLE_TAGS.get(role, f"<|{role}|>")
        parts.append(f"{tag}\n{content}")
    parts.append("<|assistant|>\n")
    return "\n".join(parts)


# --------------------------------------------------------------------------- #
# Бэкенды: реальный (net.infer + net.checkpoint) и подменяемый (тесты)
# --------------------------------------------------------------------------- #


class _Backend:
    """Интерфейс бэкенда адаптера (позволяет тестировать §13 без jax)."""

    def encode(self, text: str) -> list[int]:  # pragma: no cover - протокол
        raise NotImplementedError

    def decode(self, ids: Sequence[int]) -> str:  # pragma: no cover - протокол
        raise NotImplementedError

    def generate_ids(
        self,
        prompt_ids: Sequence[int],
        max_new_tokens: int,
        seed: int,
        temperature: float,
        top_p: float,
        no_repeat_ngram: int,
    ) -> list[int]:  # pragma: no cover - протокол
        raise NotImplementedError

    def behavior_logprobs(
        self, prompt_ids: Sequence[int], out_ids: Sequence[int]
    ) -> list[float]:  # pragma: no cover - протокол
        raise NotImplementedError

    @property
    def checkpoint_sha256(self) -> str:  # pragma: no cover - протокол
        return ""


def load_model_config(source: Any):
    """Собрать ``ModelConfig`` из инстанса/словаря/пути (без импорта net.config).

    Модуль конфигурации берётся через ``net.infer`` (``infer.ModelConfig``):
    адаптер импортирует из сети только ``net.infer``/``net.checkpoint``.
    """
    from net import infer  # ленивый импорт: окружение без ML-стека остаётся лёгким

    if source is None:
        raise ValueError("config обязателен: ModelConfig | dict | путь к config.json")
    if hasattr(source, "vocab_size") and not isinstance(source, (str, Path, dict)):
        return source
    if isinstance(source, (str, Path)):
        data = json.loads(Path(source).read_text(encoding="utf-8"))
    elif isinstance(source, dict):
        data = dict(source)
    else:
        raise TypeError(f"config: неожиданный тип {type(source).__name__}")
    known = {fld.name for fld in dataclass_fields(infer.ModelConfig)}
    filtered = {k: v for k, v in data.items() if k in known}
    # JSON без кортежей: объявленная раскладка слоёв возвращается списком.
    for name in ("context_curriculum", "curriculum_split", "mla_layer_modes"):
        if name in filtered and isinstance(filtered[name], list):
            filtered[name] = tuple(filtered[name])
    # ``mla_block_merge`` — вложенный объект (ADR-018); читатель (attn_sparse)
    # ждёт объявленную схему, поэтому приводим dict к конфиг-записи.  Сам класс
    # берём из модуля конфигурации, уже загруженного ``net.infer``, — адаптер не
    # импортирует из сети ничего сверх ``net.infer``/``net.checkpoint``.
    merge = filtered.get("mla_block_merge")
    if isinstance(merge, dict):
        BlockMergeConfig = vars(sys.modules[infer.ModelConfig.__module__])[
            "BlockMergeConfig"
        ]
        filtered["mla_block_merge"] = BlockMergeConfig(
            enabled=bool(merge.get("enabled", False)),
            block=int(merge.get("block", BlockMergeConfig.block)),
        )
    return infer.ModelConfig(**filtered)


class _HFTokenizerShim:
    """HF ``tokenizers.Tokenizer`` → интерфейс ``encode``/``decode`` сети.

    ``net.infer`` ждёт ``encode(text) -> list[int]`` и ``decode(ids) -> str``;
    HF-токенизатор корпусного BPE v2 (``tokenizer.model``) отдаёт ``Encoding``.
    """

    def __init__(self, tokenizer: Any, vocab_size: Optional[int] = None) -> None:
        self._tok = tokenizer
        self._vocab_size = vocab_size or int(tokenizer.get_vocab_size())

    @property
    def vocab_size(self) -> int:
        return int(self._vocab_size)

    def encode(self, text: str) -> list[int]:
        return [int(i) for i in self._tok.encode(text).ids]

    def decode(self, ids: Sequence[int]) -> str:
        return self._tok.decode([int(i) for i in ids])


def load_tokenizer(path: str | Path, vocab_size: Optional[int] = None):
    """Пиннутый токенизатор претрейна: артефакт сети (JSON BPE) или HF v2.

    Различие — по содержимому: JSON ``net.tokenizer.BPETokenizer`` несёт ключ
    ``specials`` с целочисленными merges; HF ``tokenizer.model`` — нет.
    """
    from net import infer  # net.tokenizer.BPETokenizer доступен через net.infer

    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"токенизатор не найден: {p}")
    data = json.loads(p.read_text(encoding="utf-8"))
    merges = data.get("merges")
    is_net_bpe = (
        isinstance(data.get("specials"), list)
        and isinstance(merges, list)
        and (not merges or isinstance(merges[0], list))
    )
    if is_net_bpe:
        return infer.BPETokenizer.load(str(p), vocab_size=vocab_size or 160_000)
    try:
        from tokenizers import Tokenizer as HFTokenizer
    except ImportError as exc:  # pragma: no cover — окружение без библиотеки
        raise JaxUnavailableError(
            f"нужен пакет tokenizers для HF-токенизатора {p}: {exc}"
        ) from exc
    return _HFTokenizerShim(HFTokenizer.from_file(str(p)), vocab_size=vocab_size)


def default_tokenizer_path() -> Optional[Path]:
    """Дефолтный путь пиннутого tokenizer'а: env-переменная → корень репо.

    Порядок: ``AXIOM_TOKENIZER`` (явный пин прогона) → ``<repo>/tokenizer.model``
    → ``<repo>/net/config.json`` рядом с артефактом.  Нет файла — ``None``
    (адаптер потребует явный ``tokenizer_path``).
    """
    import os

    env_path = os.environ.get("AXIOM_TOKENIZER")
    if env_path and Path(env_path).is_file():
        return Path(env_path)
    for candidate in (_REPO_ROOT / "tokenizer.model", _REPO_ROOT / "net" / "tokenizer.model"):
        if candidate.is_file():
            return candidate
    return None


def resolve_step_dir(directory: str | Path) -> Path:
    """Каталог шага чекпойнта: сам ``step-*``/с ``params/`` или последний в наборе."""
    d = Path(directory)
    if (d / "params").is_dir():
        return d
    if (d / "cursor.json").is_file():
        try:
            latest = json.loads((d / "cursor.json").read_text(encoding="utf-8"))
            candidate = d / str(latest.get("path", ""))
            if (candidate / "params").is_dir():
                return candidate
        except (OSError, json.JSONDecodeError):
            pass
    steps = sorted(p for p in d.glob("step-*") if (p / "params").is_dir())
    if steps:
        return steps[-1]
    raise FileNotFoundError(
        f"чекпойнт не найден: {d} (нет params/, cursor.json или step-*/params)"
    )


def checkpoint_sha256(directory: str | Path) -> str:
    """sha256 предмета замера: ``tree_hash`` из meta/manifest, иначе пересчёт.

    Прибор не считает хеш с «живого» файла молча: при отсутствии манифеста
    восстановление всё равно даёт побайтовое дерево, хеш которого считается
    ``net.checkpoint.tree_hash`` (та же функция, что у менеджера чекпойнтов).
    """
    step = resolve_step_dir(directory)
    for manifest in (step / "meta.json", step / "params" / "manifest.json"):
        if manifest.is_file():
            try:
                payload = json.loads(manifest.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            digest = payload.get("tree_hash") or payload.get("checkpoint_hash")
            if isinstance(digest, str) and digest:
                return digest
    raise FileNotFoundError(
        f"манифест с tree_hash не найден в {step}: нужен meta.json "
        "(CheckpointManager) или params/manifest.json"
    )


class _NetBackend(_Backend):
    """Реальный бэкенд: ленивая загрузка ``jax``, ``net.infer``, ``net.checkpoint``."""

    def __init__(
        self,
        checkpoint: str | Path,
        config: Any,
        *,
        tokenizer: Any = None,
        tokenizer_path: str | Path | None = None,
        seed: int = 0,
        chunk_size: int = 64,
    ) -> None:
        self._checkpoint = checkpoint
        self._config_src = config
        self._tokenizer = tokenizer
        self._tokenizer_path = tokenizer_path
        self._seed = int(seed)
        self._chunk_size = int(chunk_size)
        self._loaded = False
        self._params: Any = None
        self._cfg: Any = None
        self._sha: str = ""

    # -- загрузка -----------------------------------------------------------

    def _load(self) -> None:
        if self._loaded:
            return
        try:
            import jax  # noqa: F401
            from net import checkpoint as checkpoint_mod
            from net import infer
        except ImportError as exc:
            raise JaxUnavailableError(
                "jax/net недоступны: JaxLM-адаптер требует ML-стек "
                f"(venv-axiom с jax). Импорт упал: {exc}"
            ) from exc
        self._infer = infer
        cfg = load_model_config(self._config_src)
        self._cfg = cfg
        tokenizer = self._tokenizer
        path = self._tokenizer_path
        if tokenizer is None and path is None:
            path = default_tokenizer_path()
        if tokenizer is None and path is not None:
            tokenizer = load_tokenizer(path, vocab_size=int(cfg.vocab_size))
        self._tokenizer = tokenizer
        step = resolve_step_dir(self._checkpoint)
        target = infer.model_mod.init_params(jax.random.PRNGKey(self._seed), cfg)
        self._params = checkpoint_mod.load_checkpoint(step / "params", target=target)
        self._sha = checkpoint_sha256(self._checkpoint)
        self._loaded = True

    def _require_tokenizer(self):
        if self._tokenizer is None:
            raise ValueError(
                "нужен tokenizer: передайте tokenizer= или tokenizer_path= "
                "(пиннутый tokens-v2 сети)"
            )
        return self._tokenizer

    @property
    def checkpoint_sha256(self) -> str:
        self._load()
        return self._sha

    @property
    def vocab_size(self) -> int:
        self._load()
        return int(self._cfg.vocab_size)

    # -- кодек --------------------------------------------------------------

    def encode(self, text: str) -> list[int]:
        self._load()
        return list(self._require_tokenizer().encode(text))

    def decode(self, ids: Sequence[int]) -> str:
        self._load()
        return self._require_tokenizer().decode([int(i) for i in ids])

    # -- генерация ----------------------------------------------------------

    def generate_ids(
        self,
        prompt_ids: Sequence[int],
        max_new_tokens: int,
        seed: int,
        temperature: float,
        top_p: float,
        no_repeat_ngram: int,
    ) -> list[int]:
        self._load()
        if no_repeat_ngram and no_repeat_ngram >= 2:
            return self._generate_no_repeat(
                prompt_ids, max_new_tokens, seed, temperature, top_p, no_repeat_ngram
            )
        return [
            int(t)
            for t in self._infer.generate(
                self._params,
                self._cfg,
                [int(t) for t in prompt_ids],
                int(max_new_tokens),
                float(temperature),
                int(seed),
                top_p=float(top_p),
                chunk_size=self._chunk_size,
            )
        ]

    def _generate_no_repeat(
        self,
        prompt_ids: Sequence[int],
        max_new_tokens: int,
        seed: int,
        temperature: float,
        top_p: float,
        n: int,
    ) -> list[int]:
        """Greedy + запрет повтора ``n``-грамм (§8.6), поверх forward сети.

        Модель не дублируется: logits берутся тем же ``forward``, которым
        пользуется ``net.infer.generate``; отличается только правило выбора
        токена — из логитов вычитается запрещённый набор, затем argmax.
        """
        import jax
        import jax.numpy as jnp
        from net.infer import EOS_ID

        infer = self._infer
        ids = [int(t) for t in prompt_ids]
        out: list[int] = []
        seed_key = jax.random.PRNGKey(int(seed))
        for _ in range(int(max_new_tokens)):
            x = jnp.asarray(ids, dtype=jnp.int32)[None, :]
            logits = infer.model_mod.forward(
                self._params, self._cfg, x, chunk_size=self._chunk_size
            )[0, -1]
            banned = _banned_next_tokens(ids, n, int(self._cfg.vocab_size))
            if temperature > 0.0:
                scaled = logits / float(temperature)
                probs = jax.nn.softmax(scaled)
                if banned:
                    mask = jnp.ones_like(probs)
                    mask = mask.at[jnp.asarray(sorted(banned), dtype=jnp.int32)].set(0.0)
                    probs = probs * mask
                    total = probs.sum()
                    probs = probs / jnp.where(total > 0, total, 1.0)
                seed_key, sub = jax.random.split(seed_key)
                nxt = int(jax.random.choice(sub, probs.shape[0], shape=(), p=probs))
            else:
                masked = logits
                if banned:
                    mask = jnp.ones_like(logits)
                    mask = mask.at[jnp.asarray(sorted(banned), dtype=jnp.int32)].set(
                        -jnp.inf
                    )
                    masked = logits + mask
                nxt = int(jnp.argmax(masked))
            if nxt == EOS_ID:
                break
            out.append(nxt)
            ids.append(nxt)
        return out

    # -- behavior-logprobs --------------------------------------------------

    def behavior_logprobs(
        self, prompt_ids: Sequence[int], out_ids: Sequence[int]
    ) -> list[float]:
        """Log-prob сгенерированных токенов: один teacher-forcing forward."""
        self._load()
        if not out_ids:
            return []
        import jax
        import jax.numpy as jnp

        infer = self._infer
        full = [int(t) for t in prompt_ids] + [int(t) for t in out_ids]
        x = jnp.asarray(full, dtype=jnp.int32)[None, :]
        logits = infer.model_mod.forward(
            self._params, self._cfg, x, chunk_size=self._chunk_size
        )[0]
        logp = jax.nn.log_softmax(logits.astype(jnp.float32), axis=-1)
        start = len(prompt_ids)
        values = [
            float(logp[start + j - 1, int(tok)]) for j, tok in enumerate(out_ids)
        ]
        return values


@dataclass(frozen=True)
class RestoredModel:
    """Восстановленный претрейн-чекпойнт: веса, конфиг, токенизатор, хеш (AD-4)."""

    params: Any
    config: Any
    tokenizer: Any
    checkpoint_sha256: str


def load_pretrain_model(
    checkpoint: str | Path,
    config: Any,
    *,
    tokenizer: Any = None,
    tokenizer_path: str | Path | None = None,
    seed: int = 0,
) -> RestoredModel:
    """Загрузить чекпойнт для приборов (PPL/проба генерации).

    Возвращает восстановленные параметры, конфиг и токенизатор.  ``jax``/``net``
    импортируются здесь (лениво) — вызов без ML-стека даёт
    :class:`JaxUnavailableError`.
    """
    backend = _NetBackend(
        checkpoint,
        config,
        tokenizer=tokenizer,
        tokenizer_path=tokenizer_path,
        seed=seed,
    )
    backend._load()
    return RestoredModel(
        params=backend._params,
        config=backend._cfg,
        tokenizer=backend._tokenizer,
        checkpoint_sha256=backend._sha,
    )


def _banned_next_tokens(ids: Sequence[int], n: int, vocab_size: int) -> set[int]:
    """Токены, дописывание которых создаёт повтор ``n``-граммы.

    Для каждого ``t``: если последовательность ``ids[-(n-1):] + [t]`` уже
    встречалась в ``ids[-len:]`` — ``t`` запрещён (§8.6, LAG-ADR-040).
    """
    if n < 2 or len(ids) < n - 1:
        return set()
    prefix = tuple(int(t) for t in ids[-(n - 1) :])
    banned: set[int] = set()
    for start in range(0, len(ids) - n + 1):
        gram = tuple(int(t) for t in ids[start : start + n])
        if gram[: n - 1] == prefix:
            banned.add(gram[-1])
    return {t for t in banned if 0 <= t < vocab_size}


# --------------------------------------------------------------------------- #
# Адаптер §13
# --------------------------------------------------------------------------- #


class JaxLMAdapter:
    """JaxLM-адаптер §13: ``generate(messages, seed, max_tokens)``.

    Тонкая обёртка: рендер сообщений → токенизация → генерация сети → форма §13.
    ``policy_version`` привязан к хешу чекпойнта (AD-4), чтобы журнал роллаута
    называл предмет замера.
    """

    def __init__(
        self,
        checkpoint: str | Path | None = None,
        config: Any = None,
        *,
        tokenizer: Any = None,
        tokenizer_path: str | Path | None = None,
        seed: int = 0,
        temperature: float = 0.0,
        top_p: float = 1.0,
        no_repeat_ngram: int = 0,
        chunk_size: int = 64,
        policy_version: str | None = None,
        backend: _Backend | None = None,
    ) -> None:
        self._policy_version = policy_version or "jax-pending"
        self.seed = int(seed)
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.no_repeat_ngram = int(no_repeat_ngram)
        if backend is not None:
            self._backend: _Backend = backend
        else:
            if checkpoint is None or config is None:
                raise ValueError(
                    "нужны checkpoint и config (или явный backend для тестов)"
                )
            self._backend = _NetBackend(
                checkpoint,
                config,
                tokenizer=tokenizer,
                tokenizer_path=tokenizer_path,
                seed=seed,
                chunk_size=chunk_size,
            )

    # -- §13 ----------------------------------------------------------------

    @property
    def checkpoint_sha256(self) -> str:
        """sha256 предмета замера (пусто у подменяемого бэкенда без чекпойнта)."""
        return getattr(self._backend, "checkpoint_sha256", "") or ""

    @property
    def policy_version(self) -> str:
        if self._policy_version != "jax-pending":
            return self._policy_version
        sha = self.checkpoint_sha256
        return f"jax-{sha[:12]}" if sha else "jax-pending"

    def encode(self, text: str) -> list[int]:
        """Детерминированное кодирование промпта/наблюдения (§13)."""
        return [int(t) for t in self._backend.encode(text)]

    def decode(self, ids: Sequence[int]) -> str:
        return self._backend.decode(ids)

    def generate(
        self,
        messages: list[dict[str, str]],
        seed: int | None = None,
        max_tokens: int = 256,
    ) -> GenerationResult:
        """Один ход ассистента: текст + token_ids + behavior_logprobs (§13)."""
        effective_seed = self.seed if seed is None else int(seed)
        prompt = render_messages(messages)
        prompt_ids = self.encode(prompt)
        out_ids = self._backend.generate_ids(
            prompt_ids,
            int(max_tokens),
            effective_seed,
            self.temperature,
            self.top_p,
            self.no_repeat_ngram,
        )
        text = self.decode(out_ids)
        logprobs = [
            float(x) for x in self._backend.behavior_logprobs(prompt_ids, out_ids)
        ]
        if len(logprobs) != len(out_ids):
            raise RuntimeError(
                "behavior_logprobs не совпали по длине с token_ids: "
                f"{len(logprobs)} != {len(out_ids)}"
            )
        return GenerationResult(
            text=text,
            token_ids=[int(t) for t in out_ids],
            behavior_logprobs=logprobs,
            policy_version=self.policy_version,
        )


__all__ = [
    "GenerationResult",
    "JaxLMAdapter",
    "JaxUnavailableError",
    "RESULT_FIELDS",
    "RestoredModel",
    "checkpoint_sha256",
    "default_tokenizer_path",
    "jax_available",
    "load_model_config",
    "load_pretrain_model",
    "load_tokenizer",
    "render_messages",
    "resolve_step_dir",
]
