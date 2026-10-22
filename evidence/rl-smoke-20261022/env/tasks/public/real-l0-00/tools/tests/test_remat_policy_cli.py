"""CLI раннера: ``--remat-policy`` (ADR-049) — дефолт, список, fail-closed.

Проверяется контракт флага, а не прогон: значение по умолчанию (не задан →
конфиг первичен, а он ``none``), что объявленный список принимается целиком и
что неизвестное имя отсекается ``argparse``-ом (``choices`` из
``net.config.REMAT_POLICIES`` — один источник имён со схемой), а не превращается
молча в ``none``.
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

from net.config import REMAT_POLICIES  # noqa: E402


def _cli():
    import pretrain_run

    return pretrain_run


def test_flag_is_absent_by_default_so_the_config_stays_primary():
    """Не задан → ``None``: политика берётся из конфига (schema-дефолт ``none``)."""
    assert _cli().parse_args([]).remat_policy is None


@pytest.mark.parametrize("policy", REMAT_POLICIES)
def test_every_declared_policy_is_accepted(policy):
    """Выборка CLI — ровно ``REMAT_POLICIES``: каждая объявленная политика
    проходит (в т.ч. ``everything_saveable`` из расширения ADR-049)."""
    assert _cli().parse_args(["--remat-policy", policy]).remat_policy == policy


def test_unknown_policy_is_rejected_by_argparse():
    with pytest.raises(SystemExit):  # choices → fail-closed, не тихий дефолт
        _cli().parse_args(["--remat-policy", "save_everything_please"])


def test_choices_are_closed_to_other_jax_policy_names():
    """Список не «всё, что есть в JAX»: необъявленные политики отсекаются.

    ``nothing_saveable`` существует в JAX, но дефолт кейса — ``none`` (значит,
    в списке ему места нет: он бы включил поведение, которого никто не
    объявлял), а ``offload_dots_saveable``/``offload_dot_with_no_batch_dims`` в
    этой версии JAX отсутствуют как готовый callable — обе отвергаются.
    """
    for undeclared in (
        "nothing_saveable",
        "offload_dots_saveable",
        "offload_dot_with_no_batch_dims",
    ):
        with pytest.raises(SystemExit):
            _cli().parse_args(["--remat-policy", undeclared])


def test_apply_keeps_the_declared_value_when_the_flag_is_absent():
    cli = _cli()
    cfg = dataclasses.replace(_cfg(), remat_policy="dots_saveable")
    assert cli.apply_remat_policy(cfg, None) is cfg


def test_apply_overrides_the_declared_value_when_the_flag_is_given():
    cli = _cli()
    cfg = dataclasses.replace(_cfg(), remat_policy="dots_saveable")
    out = cli.apply_remat_policy(cfg, "none")
    assert out.remat_policy == "none"
    assert cfg.remat_policy == "dots_saveable"  # исходный конфиг не мутируется


def _cfg():
    from net.config import ModelConfig

    return ModelConfig()
