"""CLI раннера: ``--ce-chunk-tokens`` (D-8 remainder) — дефолт, значение, fail-closed.

Проверяется контракт флага, а не прогон: не задан → ``None``, и ширина берётся из
конфига (``net/config.json`` пинит 1024, дефолт схемы 0); заданное положительное
значение замещает объявленное явным намерением прогона; ``0`` — явный выключатель
(наивный CE, граф прежний).

В отличие от ``--remat-policy`` ширина — это *мера*, а не перечисление, поэтому
``choices`` здесь неуместны: границей снизу служит ``int``-разбор (нечисловое
отсекается ``argparse``) и схема.  Отрицательное значение ``argparse`` пропускает
— его закрывает ``net.config.validate_config`` при сборке модели (fail-closed ниже
по стеку), а не молчаливый no-op.
"""

from __future__ import annotations

import dataclasses
import os
import sys
from pathlib import Path

import pytest

TOOLS_DIR = Path(__file__).resolve().parents[1]
CASE_DIR = TOOLS_DIR.parent

# Пиннинг бэкенда ДО первого импорта jax (ADR-010): тест остаётся файловым.
os.environ.setdefault("NET_JAX_BACKEND", "cpu")

for _path in (str(CASE_DIR), str(TOOLS_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)


def _cli():
    import pretrain_run

    return pretrain_run


def _cfg():
    from net.config import ModelConfig

    return ModelConfig()


def test_flag_is_absent_by_default_so_the_config_stays_primary():
    """Не задан → ``None``: ширина берётся из конфига (пин ``net/config.json``)."""
    assert _cli().parse_args([]).ce_chunk_tokens is None


@pytest.mark.parametrize("width", [1, 8, 1024, 4096])
def test_declared_width_is_accepted(width):
    assert _cli().parse_args(["--ce-chunk-tokens", str(width)]).ce_chunk_tokens == width


def test_zero_is_accepted_as_the_explicit_off_switch():
    """``0`` — наивный CE: механизм выключен, граф прежний (не ошибка)."""
    assert _cli().parse_args(["--ce-chunk-tokens", "0"]).ce_chunk_tokens == 0


def test_non_integer_is_rejected_by_argparse():
    with pytest.raises(SystemExit):  # type=int → fail-closed, не тихий дефолт
        _cli().parse_args(["--ce-chunk-tokens", "many"])


def test_negative_value_passes_argparse_but_the_schema_is_the_gate():
    """``-1`` ``argparse`` пропускает (это мера, не перечисление); его ловит
    ``validate_config`` при сборке модели — fail-closed ниже по стеку."""
    assert _cli().parse_args(["--ce-chunk-tokens=-1"]).ce_chunk_tokens == -1
    from net.config import validate_config

    with pytest.raises(AssertionError):
        validate_config(dataclasses.replace(_cfg(), ce_chunk_tokens=-1))


def test_apply_keeps_the_declared_value_when_the_flag_is_absent():
    cli = _cli()
    cfg = dataclasses.replace(_cfg(), ce_chunk_tokens=1024)
    assert cli.apply_ce_chunk_tokens(cfg, None) is cfg


def test_apply_overrides_the_declared_value_when_the_flag_is_given():
    cli = _cli()
    cfg = dataclasses.replace(_cfg(), ce_chunk_tokens=1024)
    out = cli.apply_ce_chunk_tokens(cfg, 4096)
    assert out.ce_chunk_tokens == 4096
    assert cfg.ce_chunk_tokens == 1024  # исходный конфиг не мутируется
