"""T-m5 — C-035: объявленная механика длинного контекста читается кодом, а не прозой.

Страж — `tools/check_declarative_context.py`: он сверяет декларацию
(`net/config.json`) с кодом (`net/config.py` — схема, `net/attn_sparse.py` —
читатель + `model`/`mla`/`kda` — потребители остальных механик AD-9) и обязан
краснеть при рассогласовании, а не только на отсутствии надписи (ADR-011).

Тесты фиксируют три состояния стража, как того требует договор контура:
``0`` PASS (декларация и код согласованы), ``1`` FAIL (рассогласование),
``2`` NOT-VERIFIED (сверять не с чем — ложный PASS запрещён).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

TOOLS_DIR = Path(__file__).resolve().parents[1]
CASE_DIR = TOOLS_DIR.parent

sys.path.insert(0, str(TOOLS_DIR))

import check_declarative_context as guard  # noqa: E402  (путь добавляется выше)

CONFIG = CASE_DIR / "net" / "config.json"
NET_DIR = CASE_DIR / "net"


def _declared() -> dict:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def _write_config(tmp_path: Path, payload: dict, name: str = "config.json") -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _run(config: Path | None = None, net_dir: Path | None = None) -> int:
    code, _ = guard.evaluate(config, net_dir)
    return code


# --- зелёный: рабочая декларация кейса --------------------------------------


def test_case_declaration_passes():
    """Декларация кейса и код net/ согласованы — страж зелёный."""
    code, lines = guard.evaluate()
    assert code == guard.EXIT_PASS, "\n".join(lines)


def test_report_names_the_declared_mechanisms():
    """Диагностика стража называет сверенные поля — доказательство, а не «OK»."""
    _, lines = guard.evaluate()
    text = "\n".join(lines)
    for field in guard.DECLARED_MECHANISMS:
        assert field in text, f"поле {field} не названо в диагностике"
    assert "mla_block_merge" in text and guard.BLOCK_MERGE_READER in text


def test_cli_runs_against_the_case_root():
    """CLI по умолчанию проверяет каталог кейса (правило C-035 без аргументов)."""
    assert guard.main([]) == guard.EXIT_PASS


# --- красный: рассогласование декларации и кода ------------------------------


def test_red_when_the_field_disappears(tmp_path):
    """Поле ушло из декларации, а код его читает — гейт красный."""
    payload = _declared()
    payload.pop("mla_block_merge")
    assert _run(_write_config(tmp_path, payload)) == guard.EXIT_FAIL


def test_red_when_the_block_is_not_a_positive_integer(tmp_path):
    """`block` вне {1, 2, …} — невалидная декларация, а не «механизм выключен»."""
    for block in (0, -4, "16", None):
        payload = dict(_declared(), mla_block_merge={"enabled": True, "block": block})
        assert (
            _run(_write_config(tmp_path, payload, f"block-{block}.json"))
            == guard.EXIT_FAIL
        ), f"block={block!r} прошёл как валидная декларация"


def test_red_when_the_window_is_narrower_than_the_block(tmp_path):
    """Включённый механизм при окне уже блока оставляет хвост блока без внимания."""
    payload = dict(
        _declared(), swa_window=8, mla_block_merge={"enabled": True, "block": 16}
    )
    assert _run(_write_config(tmp_path, payload)) == guard.EXIT_FAIL


def test_red_when_the_deviation_line_is_missing(tmp_path):
    """Механика без строки deviations со ссылкой на ADR-018 — молчаливый импорт."""
    payload = _declared()
    payload["deviations"] = [line for line in payload["deviations"] if "ADR-018" not in line]
    assert _run(_write_config(tmp_path, payload)) == guard.EXIT_FAIL


def test_red_when_the_code_stops_reading_the_declaration(tmp_path):
    """Схема/читатель пропали из кода — декларация стала прозой."""
    stub = tmp_path / "net"
    stub.mkdir()
    (stub / "config.py").write_text(
        "class ModelConfig:\n    swa_window: int = 128\n", encoding="utf-8"
    )
    (stub / "attn_sparse.py").write_text("def window_attention():\n    return 0\n", encoding="utf-8")
    assert _run(CONFIG, stub) == guard.EXIT_FAIL


# --- NOT-VERIFIED: сверять не с чем -----------------------------------------


def test_not_verified_when_the_config_is_missing(tmp_path):
    """Нет декларации — сверка невозможна; ложный PASS запрещён."""
    assert _run(tmp_path / "absent.json") == guard.EXIT_NOT_VERIFIED


def test_not_verified_when_the_net_directory_is_missing(tmp_path):
    """Нет каталога кода — сверять не с чем."""
    assert _run(CONFIG, tmp_path / "absent-net") == guard.EXIT_NOT_VERIFIED


def test_fail_is_never_reported_as_a_pass(tmp_path):
    """Вердикт несёт причину: красный вход не даёт PASS ни при какой подстановке."""
    payload = dict(_declared(), mla_block_merge={"enabled": True, "block": 0})
    code, lines = guard.evaluate(_write_config(tmp_path, payload), NET_DIR)
    assert code == guard.EXIT_FAIL
    assert any("mla_block_merge" in line for line in lines)
