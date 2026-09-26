"""Дельта: пресет ``l3-full`` смоук-лупа SFT — полный конфиг скелета L3.

Источник истины — декларативный ``net/config.json`` (AD-9/C-035: конфиг
первичен), а не реализация пресета: тесты сверяют конфиг стадии с файлом, а
не повторяют числа из кода.

* **T-p1** — пресет ``l3-full`` строит конфиг, ключевые поля которого
  совпадают с ``net/config.json`` (слои, KDA/MLA-паттерн, MoE 12+2/top-2,
  MTP, AttnRes, иерархический пул); единственное расхождение с декларативными
  числами — подставленный vocab смоука, и оно проверяется точным тождеством
  числа параметров;
* **T-p2** — численные пресеты ``small``/``tiny`` заморожены: их поля
  совпадают и с значениями, на которых стоят приёмочные тесты сети, и с
  ``net/tests/conftest.py`` напрямую (сравнение «до/после» дельты);
* **T-p3** — журнал стадии получает ``notes`` с ``preset=l3-full``,
  ``params_total`` и ``state_estimate_gb``; вердикт оценки ресурсов
  (ok/warn/not_verified) включая WARN при нехватке свободной памяти.

Тесты файловые и **не трогают GPU** (владелец: GPU-прогоны запрещены): бэкенд
пиннуется на CPU до первого импорта jax, счёт параметров идёт по формам
(``jax.eval_shape``, без аллокации), свободная память устройства
подменяется в тестах вердикта.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import sys
from pathlib import Path

import pytest

TOOLS_DIR = Path(__file__).resolve().parents[1]
CASE_DIR = TOOLS_DIR.parent

# Пиннинг бэкенда ДО первого импорта jax: net/tests/conftest.py читает
# NET_JAX_BACKEND при импорте, и тест обязан остаться файловым (CPU).
os.environ.setdefault("NET_JAX_BACKEND", "cpu")

for _path in (str(TOOLS_DIR), str(CASE_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import run_sft_smoke as runner  # noqa: E402 — после правки sys.path
from net.config import ModelConfig, validate_config  # noqa: E402

CONFIG_PATH = CASE_DIR / "net" / "config.json"


def declared() -> dict:
    """Декларативный конфиг сети как он записан в net/config.json."""
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# T-p1 — пресет l3-full собирается из декларативного конфига
# ---------------------------------------------------------------------------


def test_t_p1_l3_full_fields_match_declarative_config():
    """Ключевые поля конфига стадии совпадают с net/config.json."""
    file = declared()
    cfg = runner.build_model_config(512, "l3-full", True)
    for field in (
        "hidden",
        "num_layers",
        "num_kda_layers",
        "num_mla_layers",
        "num_heads",
        "head_dim",
        "kda_dk",
        "kda_dv",
        "kda_decay_rank",
        "kda_short_conv_kernel",
        "kda_g_min",
        "mla_latent_dim",
        "mla_head_dim",
        "mla_top_k",
        "mla_index_heads",
        "mla_index_dim",
        "mla_pool_block",
        "mla_pool_size",
        "swa_window",
        "swa_share_kda_projections",
        "qat_kv_enabled",
        "attn_dense_reference",
        "mlp_intermediate",
        "siti_beta_gate",
        "siti_beta_up",
        "moe_dense_layers",
        "moe_latent_dim",
        "moe_num_routed",
        "moe_num_shared",
        "moe_top_k",
        "moe_expert_intermediate",
        "moe_shared_intermediate",
        "qb_weight",
        "routing_seed",
        "mtp_layers",
        "mtp_loss_weight",
        "attnres_blocks",
        "attnres_block_size",
        "vit_patch",
        "vit_hidden",
        "vit_depth",
        "vit_heads",
        "vit_mlp",
        "image_size",
        "pretrain_dtype",
        "weight_decay",
        "warmup_ratio",
        "lr_schedule",
    ):
        assert getattr(cfg, field) == file[field], (
            f"{field}: конфиг стадии {getattr(cfg, field)!r} != "
            f"net/config.json {file[field]!r}"
        )

    # 3:1 KDA:MLA — паттерн скелета (18 + 6 из 24 слоёв), MoE 12+2 top-2.
    assert (cfg.num_kda_layers, cfg.num_mla_layers) == (18, 6)
    assert cfg.num_kda_layers + cfg.num_mla_layers == cfg.num_layers == 24
    assert (cfg.moe_num_routed, cfg.moe_num_shared, cfg.moe_top_k) == (12, 2, 2)
    assert (cfg.mtp_layers, cfg.attnres_blocks) == (1, 2)
    # JSON не знает кортежей: списки конфига возвращаются неизменяемыми.
    assert isinstance(cfg.context_curriculum, tuple)
    assert isinstance(cfg.curriculum_split, tuple)
    assert isinstance(cfg.mla_layer_modes, tuple)
    assert cfg.context_curriculum == tuple(file["context_curriculum"])
    assert cfg.mla_layer_modes == tuple(file["mla_layer_modes"])
    assert len(cfg.mla_layer_modes) == cfg.num_mla_layers
    validate_config(cfg)  # структурные инварианты скелета держатся


def test_t_p1_vocab_substitution_is_the_only_parameter_delta():
    """Единственное расхождение с net/config.json — подставленный vocab смоука.

    Тождество точное: разница числа параметров равна
    ``(декларативный vocab - смоук-vocab) * hidden`` (связанный embedding, один
    раз).  Так «полный конфиг» проверяется не списком полей, а числом.
    """
    file = declared()
    cfg = runner.build_model_config(1024, "l3-full", True)
    assert cfg.vocab_size == 1024  # vocab подставляет стадия, как для small/tiny

    from net.model import active_param_count, param_count

    expected_total = file["actual_param_count"] - (
        file["vocab_size"] - cfg.vocab_size
    ) * cfg.hidden
    assert param_count(cfg) == expected_total
    # active-per-token не зависит от vocab (embedding — lookup, не вычисление):
    # счётчик стадии обязан совпасть с декларативным пиннутым числом.
    assert active_param_count(cfg) == file["active_params_per_token"]


def test_t_p1_qat_flag_follows_run_flag():
    """QAT включается флагом стадии (ADR-005 п. 7), KV-ветка — из конфига."""
    file = declared()
    on = runner.build_model_config(512, "l3-full", True)
    off = runner.build_model_config(512, "l3-full", False)
    assert on.qat_enabled is True
    assert off.qat_enabled is False
    assert on.qat_kv_enabled is off.qat_kv_enabled is bool(file["qat_kv_enabled"])


def test_t_p1_report_names_unmapped_fields_and_hashes_config():
    """Шаг 1 дельты: неперенесённые поля названы, конфиг пиннут хешем."""
    file = declared()
    report = runner.l3_config_report()
    assert report["path"] == "net/config.json"
    assert report["sha256"] == hashlib.sha256(CONFIG_PATH.read_bytes()).hexdigest()
    assert report["declared_vocab_size"] == file["vocab_size"]
    assert report["declared_param_count"] == file["actual_param_count"]
    assert report["declared_active_params"] == file["active_params_per_token"]

    # Неперенесённое — метаданные прогона/бюджета, а не архитектура сети:
    # ни одно поле dataclass'а не потеряно молча.
    known = {field.name for field in dataclasses.fields(ModelConfig)}
    assert set(report["unmapped"]) == set(file) - known
    assert report["unmapped"], "список неперенесённых полей не должен быть пуст"
    assert "actual_param_count" in report["unmapped"]
    assert "deviations" in report["unmapped"]
    assert set(report["unmapped"]).isdisjoint(known)
    # Обратная сторона: ни одно поле конфига не приходит из дефолтов dataclass'а
    # в обход декларативного файла — иначе «конфиг первичен» было бы неправдой.
    assert known <= set(file)


# ---------------------------------------------------------------------------
# T-p2 — численные пресеты small/tiny заморожены
# ---------------------------------------------------------------------------


def test_t_p2_small_preset_frozen():
    """small: поля не изменились дельтой (значения, что стоят тесты сети)."""
    cfg = runner.build_model_config(512, "small", False)
    frozen = {
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
    for field, value in frozen.items():
        assert getattr(cfg, field) == value, f"small.{field} изменился: {getattr(cfg, field)!r}"
    assert cfg.num_heads * cfg.head_dim == cfg.hidden
    assert cfg.vocab_size == 512  # vocab подставляет стадия
    assert cfg.qat_enabled is False


def test_t_p2_tiny_preset_frozen():
    """tiny: поля не изменились дельтой (значения, что стоят тесты сети)."""
    cfg = runner.build_model_config(64, "tiny", False)
    frozen = {
        "hidden": 16,
        "num_layers": 4,
        "num_kda_layers": 3,
        "num_mla_layers": 1,
        "num_heads": 2,
        "head_dim": 8,
        "kda_dk": 8,
        "kda_dv": 8,
        "kda_decay_rank": 8,
        "mla_latent_dim": 16,
        "mla_head_dim": 8,
        "mlp_intermediate": 32,
        "moe_latent_dim": 8,
        "moe_num_routed": 4,
        "moe_top_k": 2,
        "moe_num_shared": 1,
        "moe_expert_intermediate": 16,
        "moe_shared_intermediate": 32,
    }
    for field, value in frozen.items():
        assert getattr(cfg, field) == value, f"tiny.{field} изменился: {getattr(cfg, field)!r}"
    assert cfg.num_heads * cfg.head_dim == cfg.hidden
    assert cfg.vocab_size == 64
    assert cfg.qat_enabled is False


def test_t_p2_small_tiny_identical_to_acceptance_conftest():
    """Замена базы для l3-full не тронула пресеты приёмки: сравнение с conftest.

    ``build_model_config`` для ``small``/``tiny`` обязан быть побитово тем же
    ``dataclasses.replace(conftest.<preset>_config(), vocab_size=…,
    qat_enabled=…)``, что и до дельты.
    """
    conftest = runner._load_acceptance_conftest()
    for preset, base, vocab in (
        ("small", conftest.small_config(), 512),
        ("tiny", conftest.tiny_config(), 64),
    ):
        for qat in (True, False):
            assert runner.build_model_config(vocab, preset, qat) == dataclasses.replace(
                base, vocab_size=vocab, qat_enabled=qat
            )


def test_t_p2_unknown_preset_still_falls_back_to_small():
    """Неизвестный пресет по-прежнему означает small (поведение не менялось)."""
    conftest = runner._load_acceptance_conftest()
    assert runner.build_model_config(512, "unknown", False) == dataclasses.replace(
        conftest.small_config(), vocab_size=512, qat_enabled=False
    )


# ---------------------------------------------------------------------------
# T-p3 — journal notes: preset, params, оценка состояния, вердикт
# ---------------------------------------------------------------------------

FAKE_ESTIMATE = {
    "preset": "l3-full",
    "built_vocab_size": 1024,
    "params_total": 764_659_345,
    "params_active": 502_399_236,
    "params_total_declared": 1_009_632_913,
    "params_active_declared": 502_399_236,
    "bytes_per_param": 16,
    "state_estimate_bytes": 12_234_549_520,
    "state_estimate_gb": 11.394,
    "state_estimate_gb_declared": 15.045,
    "free_device_bytes": 17_179_869_184,
    "free_device_gb": 16.0,
    "device_memory_source": "test",
    "param_counting": "test",
    "verdict": "ok",
    "note": "оценка состояния 15.045 ГиБ <= свободной памяти 16.0 ГиБ (test)",
}


def test_t_p3_notes_carry_preset_params_and_estimate():
    """Контракт журнала: ``preset=l3-full, params_total=…, state_estimate_gb=…``."""
    report = runner.l3_config_report()
    notes = runner.l3_full_notes(FAKE_ESTIMATE, report)
    text = "\n".join(notes)
    assert "preset=l3-full, params_total=764659345, state_estimate_gb=11.394" in text
    assert "params_active=502399236" in text
    assert "16 Б/параметр" in text
    assert "вердикт=ok" in text
    # Неперенесённые поля конфига названы в журнале, а не потеряны.
    for field in report["unmapped"]:
        assert field in text
    assert report["path"] in text and report["sha256"][:16] in text
    # vocab подставлен — зафиксировано явно.
    assert "160000 → 1024" in text


def test_t_p3_warn_entry_when_estimate_exceeds_free_memory():
    """WARN — отдельная строка журнала, когда состояние не влезает."""
    report = runner.l3_config_report()
    estimate = dict(FAKE_ESTIMATE, verdict="warn", free_device_gb=4.0,
                    note="оценка состояния 15.045 ГиБ > свободной памяти 4.0 ГиБ")
    text = "\n".join(runner.l3_full_notes(estimate, report))
    assert "WARN" in text
    assert "вердикт=warn" in text


def test_t_p3_notes_survive_unmeasured_inputs():
    """«Не посчитано» печатается как «не посчитано», а не как ноль/успех."""
    report = {"path": "net/config.json", "sha256": "", "unmapped": [],
              "declared_vocab_size": None}
    estimate = dict(
        FAKE_ESTIMATE, params_total=None, params_active=None,
        state_estimate_gb=None, verdict="not_verified", note="не сверено",
    )
    text = "\n".join(runner.l3_full_notes(estimate, report))
    assert "params_total=не посчитано" in text
    assert "state_estimate_gb=не посчитано" in text
    assert "поля net/config.json без аналога" in text


def test_t_p3_state_estimate_is_16_bytes_per_param():
    """Оценка состояния — 16 Б/параметр (bf16-веса + fp32-мастер + m/v + градиенты)."""
    assert runner.STATE_BYTES_PER_PARAM == 16
    assert runner.state_estimate_bytes(1_000_000_000) == 16_000_000_000
    assert runner.state_estimate_bytes(0) == 0
    assert runner.gib(1 << 30) == 1.0
    assert runner.gib(None) is None


def test_t_p3_verdict_tracks_free_memory(monkeypatch):
    """Вердикт оценки: ok / warn / not_verified — по свободной памяти устройства."""
    cfg = runner.build_model_config(64, "tiny", False)
    report = runner.l3_config_report()

    monkeypatch.setattr(runner, "device_memory_free_bytes", lambda: (None, "test"))
    assert runner.assess_l3_resources(cfg, report)["verdict"] == "not_verified"

    huge = 1 << 40  # 1 ТиБ — заведомо больше состояния
    monkeypatch.setattr(runner, "device_memory_free_bytes", lambda: (huge, "test"))
    ok = runner.assess_l3_resources(cfg, report)
    assert ok["verdict"] == "ok"
    assert ok["free_device_bytes"] == huge
    # Оценка сравнивается по консервативной (большей) величине: декларативный
    # полный конфиг 160K, а не только построенный смоук-vocab.
    assert ok["state_estimate_gb_declared"] > ok["state_estimate_gb"]

    monkeypatch.setattr(runner, "device_memory_free_bytes", lambda: (0, "test"))
    warn = runner.assess_l3_resources(cfg, report)
    assert warn["verdict"] == "warn"
    assert "WARN" in "\n".join(runner.l3_full_notes(warn, report))


def test_t_p3_estimate_counts_params_by_shape():
    """Оценка считается по формам конфига и не поднимает исключений."""
    cfg = runner.build_model_config(64, "tiny", False)
    estimate = runner.assess_l3_resources(cfg, runner.l3_config_report())
    assert estimate["preset"] == "l3-full"
    assert isinstance(estimate["params_total"], int) and estimate["params_total"] > 0
    assert estimate["state_estimate_bytes"] == 16 * estimate["params_total"]
    assert estimate["bytes_per_param"] == 16
    assert estimate["verdict"] in {"ok", "warn", "not_verified"}
    assert estimate["param_counting"].startswith("net.model.param_count")
    for key in (
        "params_total", "params_active", "state_estimate_gb",
        "free_device_bytes", "free_device_gb", "device_memory_source",
        "verdict", "note", "built_vocab_size",
    ):
        assert key in estimate, f"поле оценки {key} отсутствует"


def test_t_p3_cli_accepts_l3_full_preset():
    """CLI: пресет ``l3-full`` доступен, численные пресеты не убраны."""
    args = runner.parse_args(
        ["--data", "d.jsonl", "--out", "o", "--model-preset", "l3-full"]
    )
    assert args.model_preset == "l3-full"
    for preset in runner.SMOKE_PRESETS:
        assert runner.parse_args(
            ["--data", "d.jsonl", "--out", "o", "--model-preset", preset]
        ).model_preset == preset
    with pytest.raises(SystemExit):
        runner.parse_args(["--data", "d.jsonl", "--out", "o", "--model-preset", "l3"])
