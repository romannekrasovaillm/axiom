"""Каталог свойств-шаблонов (ADR-039, дельта P).

Импорт пакета регистрирует все двенадцать шаблонов в реестре
:mod:`tools.properties.base`. Генераторы кандидатов, мутационный исполнитель и
матрица возмущений живут здесь же, но **не** импортируются путём вердикта: их
подтягивают только соответствующие режимы ``check_claims`` и тесты.
"""

from __future__ import annotations

from .base import (  # noqa: F401
    Facts,
    Mutant,
    Property,
    PropertyError,
    Verdict,
    all_properties,
    canonical_params,
    get_property,
    load_properties_registry,
    mutant_fingerprint,
    property_fingerprint,
    register,
    validate_params,
)

from . import (  # noqa: F401  (порядок не важен: регистрация по декоратору)
    bounds,
    conservation,
    declared_equals_actual,
    determinism,
    differential,
    freshness,
    identity,
    liveness,
    metamorphic,
    monotonic_trend,
    reversibility,
    safety,
)

#: Имена шаблонов каталога — ровно двенадцать (дельта P2).
TEMPLATE_NAMES: tuple[str, ...] = (
    "declared_equals_actual",
    "bounds",
    "conservation",
    "identity",
    "determinism",
    "reversibility",
    "differential",
    "metamorphic",
    "monotonic_trend",
    "liveness",
    "safety",
    "freshness",
)


def assert_catalog() -> None:
    """Каталог содержит ровно объявленные двенадцать шаблонов."""
    have = set(all_properties())
    want = set(TEMPLATE_NAMES)
    if have != want:
        raise PropertyError(f"каталог шаблонов расходится: лишние {have - want}, нет {want - have}")
