"""Пайплайн подготовки претрейн-датасета L3 (ADR-021): шарды W (веб), C (код), Q.

Модули:

* ``common`` — шард-запись с ротацией по размеру, потоковый sha256, манифест,
  дедуп с ограниченным окном, источники (локальный jsonl и HuggingFace stream);
* ``fineweb`` — шард W: FineWeb-Edu, ~17B токенов; шард Q: сужёный микс того же
  источника и кода с тестами, ~1B токенов (decay/annealing);
* ``stack`` — шард C: The Stack (языки/лицензии/длина), ~3B токенов;
* ``build`` — CLI (``prepare-w`` / ``prepare-c`` / ``prepare-q`` / ``probe`` / ``sources``).
"""

from __future__ import annotations

from . import common, fineweb, stack

__all__ = ["common", "fineweb", "stack"]
