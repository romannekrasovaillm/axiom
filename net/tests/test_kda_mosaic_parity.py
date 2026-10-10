"""Контракт, структура и границы Mosaic-ядра KDA-решения (стадия 5 ADR-052).

Проверяется на CPU то, что проверяемо без стенда, и **не больше**:

* контракт модуля совпадает с Triton-версией (кроме ``valid_config`` — он расширен
  собственным SMEM-сеивом);
* ядро **реализовано** и требует GPU: на CPU — честный ``RuntimeError`` с причиной, а не
  подмена (C-007); паритет Mosaic остаётся NOT RUN;
* структура вызова соответствует проверенному рецепту цепочки (``scratch_types`` —
  список ``plgpu.SMEM``, Lane-семантика, SMEM-переходы между ``mma``);
* ``valid_config`` отсеивает конфигурации, не влезающие в SMEM блока;
* CPU-эталон ``reference`` сверен с ``solve_jax`` (единственная численная сверка,
  возможная без устройства).
"""

from __future__ import annotations

import pytest

from net.kernels import kda_ut_solve as triton
from net.kernels import kda_ut_solve_mosaic as mosaic

CONTRACT_SHARED = ("CASES", "TUNE_SPACE", "DEFAULT_PARAMS", "TOLERANCE", "make_inputs",
                   "reference", "baseline", "reference_t", "baseline_t", "solve_jax",
                   "solve_jax_t", "cost")


def test_module_contract_is_the_triton_one():
    """Общие элементы контракта — те же объекты Triton-модуля (не копипаста)."""
    for name in CONTRACT_SHARED:
        assert hasattr(mosaic, name), f"в Mosaic-модуле нет элемента контракта {name}"
        assert getattr(mosaic, name) is getattr(triton, name), (
            f"{name} обязан быть тем же объектом, что в Triton-версии")
    assert callable(mosaic.valid_config), "valid_config обязан быть на месте"


def test_cases_are_the_real_shapes():
    """CASES — реальные формы задачи: H=12, C=64, dk=dv=128."""
    assert mosaic.CASES, "CASES пуст"
    for case in mosaic.CASES:
        assert (case["H"], case["C"], case["dk"], case["dv"]) == (12, 64, 128, 128)


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


def test_neumann_steps_and_padding():
    assert mosaic.neumann_steps(64) == 6
    assert mosaic.packed_shape(64) == mosaic.MMA_BLOCK == 128
    assert mosaic.packed_shape(128) == 128


def test_kernel_is_implemented_and_requires_gpu():
    """Ядро реализовано; на CPU — честный отказ (не «прошло» без устройства)."""
    assert mosaic.KERNEL_IMPLEMENTED is True
    import jax.numpy as jnp

    a = jnp.eye(4, dtype=jnp.float32)
    b = jnp.zeros((4, 4), dtype=jnp.float32)
    if mosaic.mosaic_available() and mosaic.gpu_available():  # pragma: no cover — стенд
        pytest.skip("на этой машине есть GPU и Mosaic-API: ядро гоняет архитектор")
    with pytest.raises(RuntimeError) as exc:
        mosaic.kernel(a, b)
    assert str(exc.value), "у отказа обязана быть причина"
    assert ("GPU" in str(exc.value)) or ("Mosaic" in str(exc.value))


def test_recipe_structure_is_followed():
    """Структура ядра соответствует рецепту: список scratch из SMEM + Lane-семантика."""
    src = (mosaic.__file__ and open(mosaic.__file__, encoding="utf-8").read()) or ""
    assert "scratch_types=[plgpu.SMEM(" in src, "scratch_types обязан быть списком plgpu.SMEM"
    assert "LoweringSemantics.Lane" in src, "нужна Lane-семантика, не Warpgroup"
    assert "smem[...] =" in src, "между mma обязателен SMEM-переход (единственный путь смены раскладки)"
    assert "jax.jit(" in src, "у plgpu.kernel нет .lower: компиляция только через jax.jit"


def test_valid_config_sieves_smem_oversized():
    """Севив отсеивает то, что не влезает в SMEM блока (лимит ~99 КБ)."""
    ok = dict(H=12, C=64, dk=128, dv=128, dtype="float32")
    assert mosaic.valid_config(ok, dict(bn=64, num_warps=8, num_stages=1)) is True
    # Гигантский блок: (M, M) в f32 превышает лимит SMEM.
    big = dict(H=12, C=4096, dk=128, dv=128, dtype="float32")
    assert mosaic.valid_config(big, dict(bn=64, num_warps=8, num_stages=1)) is False
    assert mosaic.smem_bytes(128, 128, "float32") == 65536
    assert mosaic.smem_bytes(4096, 4096, "float32") > mosaic.SMEM_LIMIT_BYTES


def test_cpu_reference_matches_solve_jax():
    """Единственная численная сверка без устройства: reference жnp против solve_jax."""
    import jax
    import jax.numpy as jnp

    a = jnp.eye(8, dtype=jnp.float32) + jnp.tril(jnp.full((8, 8), 0.25, jnp.float32), -1)
    b = jax.random.normal(jax.random.PRNGKey(0), (8, 8))
    r = mosaic.reference(a[None], b[None])
    x = mosaic.solve_jax(a[None], b[None])
    assert jnp.allclose(r, x, rtol=1e-4, atol=1e-4)


def test_parity_with_mosaic_kernel_is_not_run_here():
    """Паритет Mosaic-ядра — NOT RUN без устройства (не «прошло»)."""
    if mosaic.mosaic_available() and mosaic.gpu_available():  # pragma: no cover — стенд
        pytest.skip("стенд: паритет гоняет архитектор")
    assert not mosaic.mosaic_available() or not mosaic.gpu_available()


def test_triton_line_is_untouched():
    assert callable(triton.kernel) and callable(triton.kernel_t)
    assert callable(triton.solve) and callable(triton.solve_t)
    assert triton.valid_config is not mosaic.valid_config, (
        "Mosaic-версия valid_config — расширенная; Triton-объект не подменяется")
