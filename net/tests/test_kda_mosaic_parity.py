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

import dataclasses
import sys
import types
from pathlib import Path

import jax.numpy as jnp
import numpy as np
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


# ---------------------------------------------------------------------------
# Класс `NameError` на GPU-пути: путь `_build_kernel`/`body` исполняется только при
# наличии устройства, поэтому CPU-парити его не покрывали. Три теста ниже ловят именно
# этот класс: (1) статически — неразрешённые имена; (2) юнит-тест ленивого геттера;
# (3) сборка ядра с подменённым Mosaic-модулем, включая исполнение тела `body`.
# ---------------------------------------------------------------------------


def test_no_undefined_names_in_module():
    """Статическая проверка: в модуле нет неразрешённых имён (будущий `NameError`).

    CPU-тесты раньше были зелёными именно потому, что `_build_kernel` не исполняется без
    GPU, а `NameError` (`_mosaic_gpu`, `_require_gpu`, `dataclasses`) всплывал только на
    стенде. pyflakes разбирает и путь сборки, и тело `body` — независимо от устройства.
    """
    pyflakes_api = pytest.importorskip("pyflakes.api")
    from pyflakes import messages as pyflakes_messages
    from pyflakes import reporter as pyflakes_reporter

    collected = []

    class _Collector(pyflakes_reporter.Reporter):
        def __init__(self):
            super().__init__(sys.stdout, sys.stderr)

        def flake(self, message):
            collected.append(message)

    src = Path(mosaic.__file__).read_text(encoding="utf-8")
    pyflakes_api.check(src, mosaic.__file__, _Collector())
    undefined = [
        m for m in collected
        if isinstance(m, (pyflakes_messages.UndefinedName, pyflakes_messages.UndefinedLocal))
    ]
    assert not undefined, (
        "неразрешённые имена в модуле (класс NameError на GPU-пути): "
        + "; ".join(f"строка {m.lineno}: {m.message}" for m in undefined)
    )


def test_mosaic_gpu_getter_is_lazy_cached_and_loud(monkeypatch):
    """`_mosaic_gpu` существует, импортирует Mosaic-API лениво, кэширует и падает с причиной."""
    assert callable(getattr(mosaic, "_mosaic_gpu", None)), (
        "ленивый геттер Mosaic-API `_mosaic_gpu` обязан быть определён")

    monkeypatch.setattr(mosaic, "_MOSAIC_GPU", None)
    first = mosaic._mosaic_gpu()
    second = mosaic._mosaic_gpu()
    assert first is second, "повторный вызов обязан отдать тот же объект (импорт ровно один раз)"

    # Отсутствующее API — RuntimeError с причиной, БЕЗ тихого отката на Triton.
    monkeypatch.setattr(mosaic, "_MOSAIC_GPU", None)
    import builtins

    real_import = builtins.__import__

    def _no_mosaic(name, *args, **kwargs):
        if name.startswith("jax.experimental.pallas"):
            raise ImportError("fake: mosaic_gpu отсутствует")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _no_mosaic)
    with pytest.raises(RuntimeError) as exc:
        mosaic._mosaic_gpu()
    assert "Mosaic" in str(exc.value), "у отказа обязана быть причина про Mosaic-API"
    assert mosaic._MOSAIC_GPU is None, "при ошибке импорта кэш не заполняется"


class _FakeMosaicGPU:
    """Минимальная подмена `mosaic_gpu`: гоняет сборку ядра без GPU.

    ``kernel`` вызывает ``body(...)`` на numpy-буферах, поэтому исполняются и ссылки
    внутри тела: любой неразрешённый или ошибочный ``plgpu.*``/локальное имя всплывает
    как `NameError`/`AttributeError` ещё на CPU.
    """

    def __init__(self):
        self.sentinel = object()
        self.calls = []
        self.body_ran = False

    class Layout:
        @staticmethod
        def MMA_ACC(dtype):
            return ("mma_acc", dtype)

        @staticmethod
        def MMA_LHS(dtype):
            return ("mma_lhs", dtype)

        @staticmethod
        def MMA_RHS(dtype):
            return ("mma_rhs", dtype)

    class LoweringSemantics:
        Lane = "Lane"

    @dataclasses.dataclass
    class CompilerParams:
        lowering_semantics: object = None

    class SMEM:
        def __init__(self, shape, dtype):
            self.shape = shape
            self.dtype = dtype

    @staticmethod
    def layout_cast(value, layout):
        return value

    @staticmethod
    def load(ref, layout=None, optimized=True):
        return jnp.asarray(ref)

    @staticmethod
    def mma(acc, a, b):
        # Как у Mosaic: (`MMA_LHS` m×k) @ (`MMA_RHS` n×k)ᵀ → m×n.
        return acc + a @ b.T

    def kernel(self, body, *, out_type, scratch_types, compiler_params, grid):
        self.calls.append(dict(out_type=out_type, scratch_types=scratch_types,
                               compiler_params=compiler_params, grid=grid))
        m, n = out_type.shape
        l_ref = np.zeros((m, m), np.float32)
        b_ref = np.zeros((m, n), np.float32)
        o_ref = np.zeros((m, n), np.float32)
        smem = np.zeros((m, m), np.float32)
        body(l_ref, b_ref, o_ref, smem)
        self.body_ran = True
        return self.sentinel


def test_build_kernel_runs_with_fake_mosaic(monkeypatch):
    """Путь сборки `_build_kernel` исполняется без GPU с подменённым Mosaic-модулем.

    Подмена идёт через ``sys.modules``, поэтому исполняется именно ленивый геттер
    ``_mosaic_gpu`` (а не заглушка вместо него), затем ``dataclasses.replace`` и тело
    ``body``. Это и есть тест на класс «NameError на GPU-пути».
    """
    fake = _FakeMosaicGPU()
    fake_mod = types.ModuleType("jax.experimental.pallas.mosaic_gpu")
    for name in ("Layout", "LoweringSemantics", "CompilerParams", "SMEM",
                 "layout_cast", "load", "mma", "kernel"):
        setattr(fake_mod, name, getattr(fake, name))

    import jax.experimental.pallas as pallas

    monkeypatch.setitem(sys.modules, "jax.experimental.pallas.mosaic_gpu", fake_mod)
    monkeypatch.setattr(pallas, "mosaic_gpu", fake_mod, raising=False)
    monkeypatch.setattr(mosaic, "_MOSAIC_GPU", None)          # сброс кэша ленивого геттера
    monkeypatch.setattr(mosaic, "gpu_available", lambda: True)  # обойти CPU-проверку

    fn = mosaic._build_kernel(64, 64, 1, jnp.float32, bn=64)

    assert fn is fake.sentinel, "сборка обязана дойти до plgpu.kernel без исключений"
    assert fake.body_ran, "тело ядра обязано исполниться (иначе NameError в body не пойман)"
    assert len(fake.calls) == 1
    call = fake.calls[0]
    assert tuple(call["out_type"].shape) == (mosaic.packed_shape(64), 64)
    assert call["grid"] == (1,)
    assert len(call["scratch_types"]) == 1
    assert isinstance(call["scratch_types"][0], _FakeMosaicGPU.SMEM)
    assert call["compiler_params"].lowering_semantics == _FakeMosaicGPU.LoweringSemantics.Lane
