"""Контракт и границы Mosaic-переноса KDA-решения (стадия 5 ADR-052).

Проверяется на CPU ровно то, что проверяемо без стенда:

* **контракт модуля** совпадает с Triton-версией (``CASES``/``TUNE_SPACE``/
  ``DEFAULT_PARAMS``/``TOLERANCE``/``make_inputs``/``reference``/``baseline``/
  ``reference_t``/``baseline_t``/``solve_jax``/``solve_jax_t``/``cost``/``valid_config``
  — тот же объект, а не копия);
* **backend-выбор**: ``AXIOM_KDA_SOLVE_KERNEL`` по умолчанию ``triton`` (прежнее
  поведение), ``mosaic`` принимается, неизвестное значение — ``ValueError`` без тихого
  отката;
* **Mosaic-ядро честно отсутствует**: вызов поднимает ``RuntimeError`` с причиной, а не
  подставляет CPU-путь или ``interpret`` (C-007);
* **Triton-линия не изменилась**: импорт Mosaic-модуля не подменяет ``kernel``/``solve``/
  ``solve_t`` Triton-версии;
* **паритет Mosaic — NOT RUN**: на хосте без GPU/Mosaic-API он не может быть объявлен
  пройденным; GPU-проверка — за архитектором в окружении 0112.
"""

from __future__ import annotations

import pytest

from net.kernels import kda_ut_solve as triton
from net.kernels import kda_ut_solve_mosaic as mosaic


def test_module_contract_is_the_triton_one():
    """Контракт совпадает структурно: это те же объекты Triton-модуля."""
    for name in ("CASES", "TUNE_SPACE", "DEFAULT_PARAMS", "TOLERANCE", "make_inputs",
                 "reference", "baseline", "reference_t", "baseline_t", "solve_jax",
                 "solve_jax_t", "cost", "valid_config"):
        assert hasattr(mosaic, name), f"в Mosaic-модуле нет элемента контракта {name}"
        assert getattr(mosaic, name) is getattr(triton, name), (
            f"{name} обязан быть тем же объектом, что в Triton-версии (не копия)")


def test_backend_default_is_triton(monkeypatch):
    monkeypatch.delenv("AXIOM_KDA_SOLVE_KERNEL", raising=False)
    assert mosaic.backend() == "triton"


def test_backend_mosaic_is_accepted(monkeypatch):
    monkeypatch.setenv("AXIOM_KDA_SOLVE_KERNEL", "mosaic")
    assert mosaic.backend() == "mosaic"


def test_unknown_backend_is_an_error(monkeypatch):
    monkeypatch.setenv("AXIOM_KDA_SOLVE_KERNEL", "banana")
    with pytest.raises(ValueError):
        mosaic.backend()


def test_neumann_steps_matches_log2():
    assert mosaic.neumann_steps(64) == 6
    assert mosaic.neumann_steps(2) == 1


def test_mosaic_kernel_is_honestly_missing():
    """Ядро не реализовано — вызов обязан упасть с причиной, а не «пройти» на CPU."""
    assert mosaic.KERNEL_IMPLEMENTED is False
    import jax.numpy as jnp

    a = jnp.eye(2, dtype=jnp.float32)
    b = jnp.zeros((2, 2), dtype=jnp.float32)
    with pytest.raises(RuntimeError) as exc:
        mosaic.kernel(a, b)
    assert str(exc.value), "у отказа обязана быть причина"


def test_parity_with_reference_is_not_run_on_this_host():
    """Паритет Mosaic с эталоном: NOT RUN без GPU/Mosaic-API (не «прошло»)."""
    if mosaic.mosaic_available() and mosaic.gpu_available():  # pragma: no cover — стенд
        pytest.skip("на этой машине есть GPU и Mosaic-API: паритет гоняет архитектор")

    assert not mosaic.mosaic_available() or not mosaic.gpu_available()
    # При этом CPU-эталон сравним сам с собой: reference и solve_jax считают одно и то же.
    import jax.numpy as jnp

    a = jnp.eye(8, dtype=jnp.float32) + jnp.tril(jnp.full((8, 8), 0.25, jnp.float32), -1)
    key = __import__("jax").random.PRNGKey(0)
    b = __import__("jax").random.normal(key, (8, 8))
    r = mosaic.reference(a[None], b[None])
    x = mosaic.solve_jax(a[None], b[None])
    assert jnp.allclose(r, x, rtol=1e-4, atol=1e-4), "CPU-эталон расходится с solve_jax"


def test_triton_line_is_untouched():
    """Импорт Mosaic-модуля не подменяет Triton-линию; её ядро и custom_vjp на месте."""
    assert callable(triton.kernel) and callable(triton.kernel_t)
    assert callable(triton.solve) and callable(triton.solve_t)
    assert triton.solve is not mosaic.__dict__.get("solve"), (
        "модуль Mosaic не должен переопределять solve Triton-версии")
