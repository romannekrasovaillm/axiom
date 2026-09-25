#!/usr/bin/env python3
"""Оркестратор прогонного пути A4 — доступные стадии и честный журнал.

Контракт CLI (§5.2 дельты `docs/specs/A4-RUN.delta.md`):

    python3 tools/run_a4_pipeline.py --seed 0 --steps 1 --tasks 10 --out <каталог> [--manifest-out PATH]

Оркестратор исполняет **доступные** стадии эталонного набора `stage_set v1`
и честно помечает недоступные как `absent`/`skipped` с причиной. Никакой
подмены: частичный прогон не объявляется полным (`pipeline_complete=false`).

Доступные стадии (реализованы в коде кейса):

* `pretrain_checkpoint` — инициализация весов с нуля (seed-пиннинг,
  `net/model.py`), минимальный шаг оптимизации по существующему
  `net/optimizer.py`, сохранение Orbax-чекпойнтом (`net/checkpoint.py`),
  `tree_hash` весов.
* `rl_environment` — тот же чекпойнт прогоняется через `env/net_executor.py`
  на ≥`--tasks` задачах, собираются механические вердикты и Run Manifest
  среды.

Недоступные стадии помечаются честно:

* `spark_inference` — `skipped`: инференс исполняется локально (CPU JAX),
  что не является стендом gb10/DGX Spark (подмена стенда запрещена, ADR-010).
* `sft` — `absent`: кода SFT нет (ADR-005 п. 1 не реализован).
* `rl_base_scheme` — `absent`: кода RL по схеме базы нет (ADR-005 п. 2).

Манифест A4 собирает генератор `tools/a4_manifest.py` (владеет узел n1);
оркестратор вызывает его **подпроцессом** по контракту §5.1 с
`--weights-hash`, `--dataset-hash`, `--run-ref` и по одному `--stage` на
стадию. Манифест не фабрикуется оркестратором.

Пути и cwd (ADR-014 п. 8, K11–K12): оркестратор не зависит от текущего
рабочего каталога запуска. Корень репозитория определяется через
`git rev-parse --show-toplevel`; все пути, попадающие в манифест
(`--run-ref`, следы `stages[].evidence`) и в журнал прогона, — относительные
от этого корня. `--out`/`--manifest-out`, переданные относительными,
якорятся к корню кейса (не к cwd вызова). `--out` обязан лежать внутри
репозитория: иначе относительного пути для манифеста не существует, и
оркестратор завершается ненулевым кодом с явным сообщением ДО исполнения
стадий и вызова генератора (генератор §5.1 `--run-ref` вне репозитория всё
равно отклоняет — но оркестратор обязан сказать об этом понятно и заранее).
Генератор вызывается подпроцессом с cwd = корень репозитория, поэтому
относительные пути резолвятся от корня и на его стороне.

Модель для wire-прогона — малый конфиг (`small_config`, vocab 512), тот же,
что используют функциональные тесты `net/tests/` для CPU-прогонов: полный
конфиг 1B-параметров неисполним на CPU (тест 01 проверяет его лишь по формам).
Это не подмена стенда, а выбор модели; фиксируется в журнале.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Optional

# Корень кейса — родитель tools/. Добавляется в sys.path, чтобы `net`/`env`
# импортировались независимо от cwd вызова.
CASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CASE_DIR))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from net import checkpoint as checkpoint_mod  # noqa: E402
from net import model as model_mod  # noqa: E402
from net import optimizer as optimizer_mod  # noqa: E402
from net.config import ModelConfig  # noqa: E402
from net.data import synthetic_stream  # noqa: E402

GENERATOR = CASE_DIR / "tools" / "a4_manifest.py"
DEFAULT_MANIFEST = CASE_DIR / "evidence" / "a4-skeleton-run-manifest.json"

# Эталонный набор стадий (docs/specs/A4-RUN.delta.md §2, ADR-014).
STAGE_SET_V1 = (
    "pretrain_checkpoint",
    "spark_inference",
    "rl_environment",
    "sft",
    "rl_base_scheme",
)

# Малый конфиг для wire-прогона на CPU (зеркало net/tests/conftest.py
# ``small_config``, vocab 512 — функциональные тесты кейса).
SMALL_CONFIG = {
    "vocab_size": 512,
    "hidden": 64,
    "num_layers": 4,
    "num_kda_layers": 3,
    "num_mla_layers": 1,
    "num_heads": 4,
    "head_dim": 16,
    "kda_dk": 16,
    "kda_dv": 16,
    "kda_decay_rank": 16,
    "kda_short_conv_kernel": 4,
    "kda_g_min": -5.0,
    "mla_latent_dim": 32,
    "mla_head_dim": 16,
    "mlp_intermediate": 128,
    "siti_beta_gate": 4.0,
    "siti_beta_up": 25.0,
    "mtp_layers": 1,
    "mtp_loss_weight": 0.1,
    "attnres_blocks": 1,
    "attnres_block_size": 4,
    "vit_patch": 14,
    "vit_hidden": 32,
    "vit_depth": 2,
    "vit_heads": 2,
    "vit_mlp": 64,
    "image_size": 56,
    "moe_dense_layers": 1,
    "moe_latent_dim": 32,
    "moe_num_routed": 6,
    "moe_num_shared": 2,
    "moe_top_k": 2,
    "moe_expert_intermediate": 16,
    "moe_shared_intermediate": 32,
    "qb_weight": 0.01,
    "routing_seed": 0,
}

# Параметры минимального шага и инференса (детерминированы от seed).
BATCH_SIZE = 1
SEQ_LEN = 32
CHUNK_SIZE = 32
MAX_NEW_TOKENS = 8
DECODING_DEFAULT = {"temperature": 0.7, "top_p": 0.95}
INFER_PROMPT_IDS = [4, 10, 20, 30, 40, 50]


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def detect_repo_root() -> Optional[Path]:
    """Корень репозитория (`git rev-parse --show-toplevel`).

    Все пути манифеста и журнала прогона относительны от него (ADR-014 п. 8);
    в git-worktree это корень worktree. None — git недоступен или упал.
    """
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            timeout=30,
            cwd=str(CASE_DIR),
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    root = (proc.stdout or "").strip()
    return Path(root).resolve() if root else None


def repo_rel(path: Path, repo_root: Path) -> str:
    """Путь относительно корня репозитория (posix-форма для JSON-артефактов).

    Вызывается только для путей, гарантированно внутри репозитория
    (принадлежность --out проверяется в main до запуска стадий).
    """
    return path.resolve().relative_to(repo_root).as_posix()


def repo_rel_or_abs(path: Path, repo_root: Path) -> str:
    """Относительный путь, если внутри репозитория; иначе абсолютный как есть.

    Для `--manifest-out`: пользователь вправе направить файл манифеста куда
    угодно (например, в tmp) — сам этот путь в манифест не попадает.
    """
    resolved = path.resolve()
    try:
        return resolved.relative_to(repo_root).as_posix()
    except ValueError:
        return str(resolved)


def parse_environment_version() -> Optional[str]:
    """Версия среды из docs/specs/ENVIRONMENT-V1.md (зеркало разбора
    генератора). Передаётся генератору явно: он работает с cwd = корень
    репозитория, откуда файл спеки кейса ему не виден."""
    spec = CASE_DIR / "docs" / "specs" / "ENVIRONMENT-V1.md"
    if not spec.is_file():
        return None
    match = re.search(r"\*\*Версия:\*\*\s*([^\s()]+)", spec.read_text(encoding="utf-8"))
    return match.group(1).strip() if match else None


def detect_backend() -> dict[str, str]:
    """Платформа/устройство/версия JAX — пиннинг бэкенда (ADR-010)."""
    device = jax.devices()[0]
    platform = getattr(device, "platform", "cpu")
    kind = getattr(device, "device_kind", None) or platform
    try:
        precision = str(jax.config.jax_default_matmul_precision) or "default"
    except Exception:  # noqa: BLE001 — поле необязательно, читаемость важнее
        precision = "unknown"
    return {
        "platform": platform,
        "device_kind": kind,
        "jax_version": jax.__version__,
        "matmul_precision": precision,
    }


def make_batches(seed: int, steps: int, cfg: ModelConfig) -> list[Any]:
    """Детерминированные синтетические батчи для шагов оптимизации."""
    key = jax.random.PRNGKey(seed)
    stream = synthetic_stream(key, BATCH_SIZE, SEQ_LEN, cfg.vocab_size, steps)
    return [batch for batch in stream]


def dataset_digest(batches: list[Any]) -> tuple[str, bytes]:
    """sha256 реально использованного синтетического датасета (байты int32)."""
    h = hashlib.sha256()
    payload = bytearray()
    for b in batches:
        raw = jnp.asarray(b).astype(jnp.int32).tobytes()
        payload.extend(raw)
        h.update(raw)
    return h.hexdigest(), bytes(payload)


def run_pretrain_checkpoint(
    cfg: ModelConfig,
    seed: int,
    steps: int,
    batches: list[Any],
    out_dir: Path,
) -> tuple[str, Path, int]:
    """init с нуля → `steps` шагов оптимизации → Orbax-чекпойнт → tree_hash.

    Возвращает ``(model_weights_sha256, checkpoint_dir, executed_steps)``.
    """
    key = jax.random.PRNGKey(seed)
    params = model_mod.init_params(key, cfg)

    loss_fn = jax.jit(
        lambda p, ids: model_mod.compute_loss(p, cfg, ids, chunk_size=CHUNK_SIZE)
    )
    grad_fn = jax.jit(jax.value_and_grad(loss_fn))
    step_fn = jax.jit(optimizer_mod.make_step(cfg))
    lr_at = optimizer_mod.cosine_schedule(peak_lr=1e-3, total_steps=max(steps, 1))

    state = optimizer_mod.init_state(params)
    executed = 0
    for s in range(steps):
        ids = jnp.asarray(batches[s], dtype=jnp.int32)
        _, grads = grad_fn(params, ids)
        params, state = step_fn(params, grads, state, lr_at(s))
        executed += 1

    checkpoint_dir = out_dir / "checkpoints" / "pretrain"
    digest = checkpoint_mod.save_checkpoint(params, checkpoint_dir)
    return digest, checkpoint_dir, executed


def run_local_inference(cfg: ModelConfig, params, out_dir: Path, seed: int) -> Path:
    """Детерминированный инференс (net/infer.py) на том же чекпойнте → журнал."""
    from net import infer

    out_ids = infer.generate(
        params, cfg, INFER_PROMPT_IDS, MAX_NEW_TOKENS, 0.0, seed=seed
    )
    journal = out_dir / "inference" / "journal.json"
    journal.parent.mkdir(parents=True, exist_ok=True)
    journal.write_text(
        json.dumps(
            {
                "device_kind": detect_backend()["device_kind"],
                "prompt_ids": INFER_PROMPT_IDS,
                "max_new_tokens": MAX_NEW_TOKENS,
                "temperature": 0.0,
                "seed": seed,
                "generated_ids": out_ids,
                "note": "Локальный инференс (CPU JAX) — провод «чекпойнт → генерация»; "
                "это НЕ стадия spark_inference (стенд gb10/DGX Spark).",
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return journal


def run_rl_environment(
    cfg: ModelConfig,
    params,
    tokenizer,
    seed: int,
    tasks: int,
    out_dir: Path,
    repo_root: Path,
) -> tuple[list[dict], Path]:
    """≥`tasks` задач через env/net_executor.py → вердикты + Run Manifest среды.

    Пути манифестов задач в сводке — относительные от корня репозитория.
    """
    from env import generate as env_generate
    from env import manifest as env_manifest
    from env import net_executor
    from env.util import read_json

    tasks_dir = out_dir / "env" / "tasks"
    env_generate.generate(
        CASE_DIR, tasks_dir, seed=seed,
        grid=(("corruption", "L0", tasks),), holdout_count=0,
    )

    manifests_dir = out_dir / "env" / "run-manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)

    verdicts: list[dict] = []
    for i in range(tasks):
        task_id = f"corruption-l0-{i:02d}"
        spec = read_json(tasks_dir / "public" / f"{task_id}.json")
        base_ws = tasks_dir / "public" / task_id
        out_ws = out_dir / "env" / "workspaces" / task_id
        decoding = {**DECODING_DEFAULT, "seed": int(spec["seed"])}
        run_id = f"a4-wire-{task_id}"
        started = "2026-09-13T00:00:00+00:00"
        finished = "2026-09-13T00:00:01+00:00"
        rr, manifest = net_executor.run_net_task(
            spec, base_ws, out_ws,
            params=params, cfg=cfg, tokenizer=tokenizer, decoding=decoding,
            max_new_tokens=MAX_NEW_TOKENS, bin=None,
            run_id=run_id, started=started, finished=finished,
            model_base="net-l3-skeleton-small-untrained",
            host="net-executor",
        )
        env_manifest.dump_manifest(manifest, manifests_dir / f"{task_id}.json")
        verdicts.append(
            {
                "task_id": task_id,
                "seed": int(spec["seed"]),
                "passed": bool(rr.verdict.passed),
                "reward": rr.reward.to_manifest_dict(),
                "manifest": repo_rel(manifests_dir / f"{task_id}.json", repo_root),
            }
        )

    summary = out_dir / "env" / "summary.json"
    summary.parent.mkdir(parents=True, exist_ok=True)
    summary.write_text(
        json.dumps(
            {
                "tasks": len(verdicts),
                "verdicts": verdicts,
                "pass_count": sum(1 for v in verdicts if v["passed"]),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return verdicts, summary


def assemble_stages(
    model_weights_sha256: str,
    checkpoint_dir: Path,
    executed_steps: int,
    env_summary: Path,
    env_verdicts: list[dict],
    inference_journal: Path,
    dataset_sha256: str,
    repo_root: Path,
) -> list[dict[str, Any]]:
    """Эталонный набор v1 со статусами и evidence (честно, без подмены).

    Пути в evidence — относительные от корня репозитория (ADR-014 п. 8):
    именно эти строки через `--stage` попадают в манифест без изменений.
    """
    passed = sum(1 for v in env_verdicts if v["passed"])
    return [
        {
            "name": "pretrain_checkpoint",
            "status": "executed",
            "evidence": [
                f"checkpoint={repo_rel(checkpoint_dir, repo_root)}",
                f"tree_hash={model_weights_sha256}",
                f"steps={executed_steps}",
            ],
        },
        {
            "name": "spark_inference",
            "status": "skipped",
            "evidence": [
                "причина: инференс исполнен локально на CPU JAX, а не на стенде "
                "gb10/DGX Spark; подмена стенда запрещена (ADR-010). "
                f"Локальный журнал: {repo_rel(inference_journal, repo_root)}",
            ],
        },
        {
            "name": "rl_environment",
            "status": "executed",
            "evidence": [
                f"run_manifest={repo_rel(env_summary, repo_root)}",
                f"tasks={len(env_verdicts)}",
                f"pass={passed}",
            ],
        },
        {
            "name": "sft",
            "status": "absent",
            "evidence": ["причина: кода SFT нет (ADR-005 п. 1 не реализован)"],
        },
        {
            "name": "rl_base_scheme",
            "status": "absent",
            "evidence": ["причина: кода RL по схеме базы нет (ADR-005 п. 2 не реализован)"],
        },
    ]


def call_generator(
    weights_hash: str,
    dataset_hash: str,
    run_ref: Path,
    stages: list[dict[str, Any]],
    manifest_out: Path,
    repo_root: Path,
) -> dict[str, Any]:
    """Вызов генератора манифеста подпроцессом по контракту §5.1.

    `--run-ref` и `--output` (когда тот внутри репозитория) передаются
    относительными от корня репозитория, а подпроцессу задаётся cwd = корень
    репозитория: генератор §5.1 резолвит и нормализует пути именно от него
    (§4 п. 6–7). Версия среды передаётся явно — из cwd корня репозитория
    файл спеки кейса генератору не виден.
    """
    cmd = [
        sys.executable,
        str(GENERATOR),
        "--weights-hash", weights_hash,
        "--dataset-hash", dataset_hash,
        "--run-ref", repo_rel(run_ref, repo_root),
        "--output", repo_rel_or_abs(manifest_out, repo_root),
    ]
    environment_version = parse_environment_version()
    if environment_version:
        cmd += ["--environment-version", environment_version]
    for st in stages:
        evidence = ";".join(str(e) for e in st["evidence"])
        cmd += ["--stage", f'{st["name"]}={st["status"]}:{evidence}']

    proc = subprocess.run(
        cmd, cwd=str(repo_root), capture_output=True, text=True, timeout=120,
    )
    return {
        "attempted": True,
        "returncode": proc.returncode,
        "stdout": proc.stdout.strip(),
        "stderr": proc.stderr.strip(),
    }


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Оркестратор прогона A4: доступные стадии + честный журнал.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--tasks", type=int, default=10)
    parser.add_argument(
        "--out",
        required=True,
        help="каталог артефактов прогона; относительный считается от корня "
        "кейса (не от cwd); обязан лежать внутри репозитория "
        "(ADR-014 п. 8, K11–K12)",
    )
    parser.add_argument(
        "--manifest-out",
        default=None,
        help="путь файла манифеста (по умолчанию evidence/a4-skeleton-run-manifest.json "
        "кейса); относительный считается от корня кейса; может лежать где "
        "угодно — в манифест этот путь не попадает",
    )
    args = parser.parse_args(argv)

    repo_root = detect_repo_root()
    if repo_root is None:
        print(
            "ошибка: не удалось определить корень репозитория "
            "(git rev-parse --show-toplevel); пути манифеста A4 обязаны быть "
            "относительными от него (ADR-014 п. 8, K11–K12). Прогон отменён.",
            file=sys.stderr,
        )
        return 2

    # Относительные --out/--manifest-out якорятся к корню кейса (не к cwd
    # вызова) — оркестратор не зависит от каталога запуска.
    out_arg = Path(args.out)
    out_dir = (
        out_arg.resolve() if out_arg.is_absolute() else (CASE_DIR / out_arg).resolve()
    )
    if not out_dir.is_relative_to(repo_root):
        print(
            f"ошибка: каталог --out '{out_dir}' находится вне репозитория "
            f"'{repo_root}'. Артефакты прогона A4 обязаны лежать внутри "
            f"репозитория: все пути в манифесте — относительные от его корня "
            f"(ADR-014 п. 8, §4 п. 6–7), а --run-ref вне репозитория генератор "
            f"§5.1 отклоняет. Укажите --out внутри репозитория (относительный "
            f"путь считается от корня кейса). Прогон отменён ДО исполнения "
            f"стадий и генерации манифеста.",
            file=sys.stderr,
        )
        return 2
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.manifest_out:
        mo_arg = Path(args.manifest_out)
        manifest_out = (
            mo_arg.resolve() if mo_arg.is_absolute() else (CASE_DIR / mo_arg).resolve()
        )
    else:
        manifest_out = DEFAULT_MANIFEST

    cfg = ModelConfig(**SMALL_CONFIG)
    backend = detect_backend()

    # --- доступная стадия: pretrain_checkpoint ------------------------------
    batches = make_batches(args.seed, args.steps, cfg)
    dataset_sha256, dataset_payload = dataset_digest(batches)

    weights_hash, checkpoint_dir, executed_steps = run_pretrain_checkpoint(
        cfg, args.seed, args.steps, batches, out_dir
    )

    # Датасет: хеш реально использованного синтетического шарда (нет реального).
    dataset_dir = out_dir / "dataset"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    (dataset_dir / "synthetic-batch.bin").write_bytes(dataset_payload)
    (dataset_dir / "dataset-card.json").write_text(
        json.dumps(
            {
                "name": "synthetic-batch",
                "domain": "synthetic",
                "source": "net/data.py synthetic_stream (seed-pinned)",
                "license": "generated",
                "hash": dataset_sha256,
                "note": "Реального шарда на ~/gb10-shared не смонтировано; "
                "захэширован реально использованный синтетический батч.",
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    # rl_environment и инференс берут ТОТ ЖЕ чекпойнт, что и pretrain_checkpoint:
    # восстанавливаем веса из Orbax по структуре init-дерева.
    from net.tokenizer import BPETokenizer
    from net.data import synthetic_corpus_texts

    tokenizer = BPETokenizer(vocab_size=cfg.vocab_size).train(
        synthetic_corpus_texts(seed=0, n_docs=4, words_per_doc=16), seed=0
    )
    params = checkpoint_mod.load_checkpoint(
        checkpoint_dir, target=model_mod.init_params(jax.random.PRNGKey(args.seed), cfg)
    )

    # --- локальный инференс (не стадия, а провод чекпойнт → генерация) -------
    inference_journal = run_local_inference(cfg, params, out_dir, args.seed)

    # --- доступная стадия: rl_environment -----------------------------------
    env_verdicts, env_summary = run_rl_environment(
        cfg, params, tokenizer, args.seed, args.tasks, out_dir, repo_root
    )

    # --- сборка стадий и журнала прогона ------------------------------------
    stages = assemble_stages(
        weights_hash, checkpoint_dir, executed_steps, env_summary,
        env_verdicts, inference_journal, dataset_sha256, repo_root,
    )

    run_ref = out_dir / "run-journal.json"
    journal = {
        "schema": "a4-run-journal/v1",
        "seed": args.seed,
        "steps": args.steps,
        "tasks": args.tasks,
        "config": cfg.as_dict(),
        "backend": backend,
        "model_weights_sha256": weights_hash,
        "dataset_sha256": dataset_sha256,
        "run_ref": repo_rel(run_ref, repo_root),
        "stages": stages,
        "pipeline_complete": False,
        "note": "Модель для wire-прогона — small_config (vocab 512), как в "
        "функциональных тестах net/tests/; полный конфиг 1B неисполним на CPU.",
    }
    run_ref.write_text(
        json.dumps(journal, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    # --- вызов генератора манифеста (подпроцесс, контракт §5.1) -------------
    gen_result = call_generator(
        weights_hash, dataset_sha256, run_ref, stages, manifest_out, repo_root
    )
    journal["manifest_generation"] = gen_result
    journal["manifest_out"] = repo_rel_or_abs(manifest_out, repo_root)
    run_ref.write_text(
        json.dumps(journal, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    pipeline_complete = False
    summary = {
        "run_ref": repo_rel(run_ref, repo_root),
        "model_weights_sha256": weights_hash,
        "dataset_sha256": dataset_sha256,
        "stages": [
            {"name": st["name"], "status": st["status"], "evidence": st["evidence"]}
            for st in stages
        ],
        "pipeline_complete": pipeline_complete,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
