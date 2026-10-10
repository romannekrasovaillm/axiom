"""Контракт, структура и границы Mosaic-ядра KDA-решения (стадия 5 ADR-052).

Проверяется на CPU то, что проверяемо без стенда, и **не больше**:

* контракт модуля совпадает с Triton-версией (кроме ``valid_config`` — он расширен
  собственным SMEM-сеивом);
* ядро **реализовано** и требует GPU: на CPU — честный ``RuntimeError`` с причиной, а не
  подмена (C-007); паритет Mosaic остаётся NOT RUN;
* структура вызова соответствует проверенному рецепту цепочки (``scratch_types`` —
  список ``plgpu.SMEM``, Lane-семантика, SMEM-переходы между ``mma``);
* раскладка RHS: ``kernel()`` подаёт правую часть транспонированной в памяти
  (``b.transpose(0, 2, 1)``), а ``body`` грузит её через ``.T``
  (``plgpu.load(<ref>.T, MMA_RHS)``) — k-контигуальность, которой требует
  ``plgpu.mma`` (рецепт ``REPORT-rhs-layout.md``); проверяется spy на ``_build_kernel``
  и статически по исходнику;
* ``valid_config`` отсеивает конфигурации, не влезающие в SMEM блока;
* CPU-эталон ``reference`` сверен с ``solve_jax`` (единственная численная сверка,
  возможная без устройства);
* **GPU-путевые дефекты сборки/трассировки** — класс, который CPU-парити не ловил:
  ``grid`` без ``grid_names`` (падение при создании ядра) и не-2D операнды
  ``plgpu.mma`` (падение ``_mma_abstract_eval`` при трассировке). Ловятся без GPU:
  трассировкой реальной машинерии с шимами примитивов 0.11.2 и AST-проверкой вызова.
"""

from __future__ import annotations

import ast
import collections
import dataclasses
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
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
# наличии устройства, поэтому CPU-парити его не покрывали. Два теста ниже ловят этот
# класс: (1) статически — неразрешённые имена; (2) юнит-тест ленивого геттера.
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


# ---------------------------------------------------------------------------
# Класс GPU-путевых дефектов сборки/трассировки, который CPU-парити не ловил:
# (1) `grid=(H,)` без `grid_names` падает при СОЗДАНИИ ядра (`Mesh`, чистый Python);
# (2) `plgpu.mma` требует 2D-операнды, а рефы в `body` — глобальные `(H, M, ...)`,
#     поэтому голова обязана срезаться `lax.axis_index("head")` + `ref.at[h]`.
# `test_build_and_trace_2d_operands` исполняет сборку и ТРАССИРОВКУ на CPU-буферах на
# реальной машинерии (`kernel`/mpmd/рефы/scratch) с шимами примитивов 0.11.2;
# AST-тесты ловят те же классы статически, независимо от трассировки.
# ---------------------------------------------------------------------------


class _TraceJournal:
    """Журнал трассировки: что ушло в ``plgpu.kernel`` и какие операнды увидел ``mma``."""

    def __init__(self):
        self.kernel_calls = []
        self.mma_operands = []


class _Layout0112:
    """Шим ``plgpu.Layout`` в контракте 0.11.2: фабрики раскладок по dtype.

    В локальном jax 0.10.2 ``Layout.MMA_LHS/MMA_RHS/MMA_ACC`` нет (другое поколение
    API); в 0.11.2 они есть и вызываются ровно так — проверенный рецепт GB10
    (``evidence/mfu-55/mosaic/REPORT-mosaic-chain.md``).
    """

    @staticmethod
    def MMA_ACC(dtype):
        return ("mma_acc", jnp.dtype(dtype))

    @staticmethod
    def MMA_LHS(dtype):
        return ("mma_lhs", jnp.dtype(dtype))

    @staticmethod
    def MMA_RHS(dtype):
        return ("mma_rhs", jnp.dtype(dtype))


def _install_0112_shims(monkeypatch, plgpu, journal):
    """Шимы примитивов с контрактом 0.11.2 поверх реального модуля ``mosaic_gpu``.

    Подменяются только те символы, которых нет в локальном venv или чья сигнатура
    там другая (``mma``/``Layout.MMA_*`` отсутствуют; у ``load`` в 0.10.2 обязателен
    ``idx``); ``kernel``, ``Mesh``, рефы, ``run_scoped`` и scratch — **реальные**:
    именно они дают глобальные формы рефов ``(H, M, M)``, на которых в 0.11.2 падает
    ``_mma_abstract_eval``.
    """

    def load(ref, idx=None, *, layout=None, optimized=True):
        # Контракт 0.11.2: источник — уже срез (``ref.at[h]``); форма сохраняется.
        return jnp.zeros(tuple(ref.shape), ref.dtype)

    def mma(acc, a, b, /):
        # Проверка как ``_mma_abstract_eval`` 0.11.2: ``a`` — логический ``(m, k)``,
        # ``b`` — логический ``(k, n)``, произведение — ``(m, n)``. Не-2D операнд падает
        # ``too many values to unpack``; несовпадение ``k`` — ``Incompatible shapes``
        # (ровно дефект №4 на стенде: ``rhs=(n, k)`` вместо ``(k, n)``).
        m2, k = a.shape
        k2, n2 = b.shape
        if k != k2:
            raise ValueError(f"plgpu.mma: k не совпал: {a.shape} vs {b.shape}")
        journal.mma_operands.append((tuple(a.shape), tuple(b.shape)))
        return jnp.zeros((m2, n2), acc.dtype)

    real_kernel = plgpu.kernel

    def kernel(body, **kwargs):
        journal.kernel_calls.append(kwargs)
        return real_kernel(body, **kwargs)

    monkeypatch.setattr(mosaic, "_MOSAIC_GPU", None)  # геттер вернёт тот же реальный модуль
    monkeypatch.setattr(plgpu, "Layout", _Layout0112)
    monkeypatch.setattr(plgpu, "load", load)
    monkeypatch.setattr(plgpu, "mma", mma, raising=False)  # в 0.10.2 символа нет — добавляем
    monkeypatch.setattr(plgpu, "layout_cast", lambda value, layout: value)
    monkeypatch.setattr(plgpu, "kernel", kernel)


def _iter_jaxprs(obj, _seen=None):
    """Рекурсивно обойти вложенные jaxpr-ы (``custom_vmap_call`` → ``mpmd_map`` → ...)."""
    _seen = _seen if _seen is not None else set()
    jaxpr = getattr(obj, "jaxpr", obj)
    if id(jaxpr) in _seen:
        return
    _seen.add(id(jaxpr))
    yield jaxpr
    for eqn in jaxpr.eqns:
        for value in eqn.params.values():
            for cand in (value if isinstance(value, (list, tuple)) else (value,)):
                if hasattr(cand, "eqns") or hasattr(cand, "jaxpr"):
                    yield from _iter_jaxprs(cand, _seen)


def _axis_index_names(closed_jaxpr):
    """Имена осей, читаемых ``lax.axis_index`` во всех вложенных jaxpr-ах трассировки."""
    names = []
    for jaxpr in _iter_jaxprs(closed_jaxpr):
        for eqn in jaxpr.eqns:
            if getattr(eqn.primitive, "name", "") == "axis_index":
                names.append(eqn.params.get("axis_name"))
    return names


def test_build_and_trace_2d_operands(monkeypatch):
    """(а) Сборка + ТРАССИРОВКА ядра на CPU-буферах: обе формы дефекта ловятся без GPU.

    Реальная машинерия ``plgpu.kernel``/mpmd строит рефы ГЛОБАЛЬНЫХ форм ``(H, M, ...)``,
    поэтому не-2D операнд ``plgpu.mma`` падает здесь так же, как ``_mma_abstract_eval``
    на стенде (``too many values to unpack``), а ``grid`` без ``grid_names`` падает ещё
    раньше — на создании ``Mesh`` внутри ``plgpu.kernel``.
    """
    if not mosaic.mosaic_available():
        pytest.skip("нет mosaic_gpu: трассировка Mosaic-ядра недостижима")
    plgpu = mosaic._mosaic_gpu()
    journal = _TraceJournal()
    _install_0112_shims(monkeypatch, plgpu, journal)
    monkeypatch.setattr(mosaic, "gpu_available", lambda: True)  # устройства нет — сборка его не требует

    H, C, N = 2, 64, 64          # N — ширина правой части (dk+dv в задаче)
    M = mosaic.packed_shape(C)   # M = 128: паддинг C=64 до блока MMA
    fn = mosaic._build_kernel(C, N, H, jnp.float32, bn=mosaic.DEFAULT_BN)

    # Контракт вызова: непустой grid несёт grid_names; выход — все головы (H, M, N).
    assert len(journal.kernel_calls) == 1, "сборка обязана позвать plgpu.kernel ровно раз"
    call = journal.kernel_calls[0]
    assert call["grid"] == (H,)
    assert call["grid_names"] == ("head",)
    assert tuple(call["out_type"].shape) == (H, M, N)
    assert len(call["scratch_types"]) == 1
    assert call["compiler_params"].lowering_semantics == plgpu.LoweringSemantics.Lane

    # Трассировка на CPU-буферах: здесь падает и форма без `grid_names`, и не-2D операнд.
    # Правая часть приходит ТРАНСПОНИРОВАННОЙ в памяти — (H, N, M) = (n, k), как её
    # подаёт kernel(); иначе последний RHS-операнд получится (n, k), а не (k, n), и
    # шим `mma` (как `_mma_abstract_eval`) упадёт на несовпадении k — дефект №4.
    a = jnp.zeros((H, M, M), jnp.float32)
    bt = jnp.zeros((H, N, M), jnp.float32)
    closed = jax.make_jaxpr(fn)(a, bt)
    assert tuple(closed.out_avals[0].shape) == (H, M, N)

    # Главное утверждение класса (а): все операнды mma при трассировке — 2D.
    assert journal.mma_operands, "цепочка mma обязана исполниться на трассировке"
    for a_shape, b_shape in journal.mma_operands:
        assert len(a_shape) == 2, f"LHS обязан быть (m, k), получено {a_shape}"
        assert len(b_shape) == 2, f"RHS обязан быть (k, n), получено {b_shape}"
    assert all(a_shape == (M, M) for a_shape, _ in journal.mma_operands)
    # X = T B: RHS — логический (k, n) = (M, N) с k-контигуальной памятью из (n, k)-источника.
    assert journal.mma_operands[-1] == ((M, M), (M, N)), "X = T B: (M,M) @ (M,N) [RHS (k,n)]"
    # Вся цепочка исполнена: -L → power → T → (steps-1) шагов → X = T B.
    assert len(journal.mma_operands) == 3 + max(mosaic.neumann_steps(M) - 1, 0) + 1

    # Голова — ось сетки, а не размерность массива: axis_index("head") реально в jaxpr.
    assert "head" in _axis_index_names(closed)


def test_trace_check_catches_missing_head_slicing(monkeypatch):
    """Различающая сила (а): тело без среза по `head` падает на трассировке — как на стенде."""
    if not mosaic.mosaic_available():
        pytest.skip("нет mosaic_gpu: трассировка Mosaic-ядра недостижима")
    plgpu = mosaic._mosaic_gpu()
    journal = _TraceJournal()
    _install_0112_shims(monkeypatch, plgpu, journal)

    H, C, N = 2, 64, 64
    M = mosaic.packed_shape(C)

    def broken_body(l_ref, b_ref, o_ref, smem):
        # Карикатура дефекта: операнды — глобальные (H, M, *) вместо 2D-срезов `ref.at[h]`.
        acc = jnp.zeros((M, M), jnp.float32)
        lhs = plgpu.load(l_ref, layout=plgpu.Layout.MMA_LHS(jnp.float32), optimized=False)
        rhs = plgpu.load(l_ref.T, layout=plgpu.Layout.MMA_RHS(jnp.float32), optimized=False)
        o_ref[...] = plgpu.mma(acc, lhs, rhs).astype(jnp.float32)

    fn = plgpu.kernel(
        broken_body,
        out_type=jax.ShapeDtypeStruct((H, M, N), jnp.float32),
        scratch_types=[plgpu.SMEM((M, M), jnp.float32)],
        compiler_params=dataclasses.replace(
            plgpu.CompilerParams(), lowering_semantics=plgpu.LoweringSemantics.Lane
        ),
        grid=(H,),
        grid_names=("head",),
    )
    with pytest.raises(ValueError, match=r"too many values to unpack \(expected 2\)"):
        jax.make_jaxpr(fn)(jnp.zeros((H, M, M), jnp.float32), jnp.zeros((H, M, N), jnp.float32))


def test_kernel_without_grid_names_fails_at_creation():
    """Дефект 1 динамически: `grid` без `grid_names` падает уже при создании ядра (Mesh).

    Тест фиксирует предпосылку AST-стражи (б): в этой линии jax непустой `grid` требует
    `grid_names` той же длины. Изменится предпосылка — покраснеет и стража, и этот тест.
    """
    if not mosaic.mosaic_available():
        pytest.skip("нет mosaic_gpu: создание Mosaic-ядра недостижимо")
    plgpu = mosaic._mosaic_gpu()
    with pytest.raises(ValueError, match="grid_names must have the same length as grid"):
        plgpu.kernel(
            lambda *refs: None,
            out_type=jax.ShapeDtypeStruct((4,), jnp.float32),
            grid=(2,),
        )


# ---------------------------------------------------------------------------
# (б) AST/статическая проверка вызова `plgpu.kernel`: непустой `grid` обязан нести
# `grid_names`; оси `jax.lax.axis_index(...)` обязаны быть объявлены в `grid_names`.
# Проверка не зависит от достижимости трассировки в данном окружении.
# ---------------------------------------------------------------------------


_PlgpuCall = collections.namedtuple("_PlgpuCall", ["lineno", "name", "args", "kwargs", "star_kwargs"])


def _dotted(node):
    """Точечное имя выражения: ``plgpu.Layout.MMA_LHS`` из цепочки Attribute/Name."""
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return ".".join(reversed(parts))


def _plgpu_calls(source):
    """Все вызовы ``plgpu.*`` из исходника: (строка, имя, позиционные, keyword-имена)."""
    calls = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Call):
            continue
        name = _dotted(node.func)
        if not name.startswith("plgpu."):
            continue
        kwargs = {}
        star_kwargs = False
        for kw in node.keywords:
            if kw.arg is None:
                star_kwargs = True
            else:
                kwargs[kw.arg] = kw.value
        calls.append(_PlgpuCall(node.lineno, name, list(node.args), kwargs, star_kwargs))
    return calls


def _literal_len(node):
    """Длина литеральной последовательности в узле AST; ``None`` — не разобрать статически."""
    try:
        value = ast.literal_eval(node)
    except (ValueError, TypeError, SyntaxError):
        return None
    return len(value) if isinstance(value, (tuple, list, str)) else None


def _grid_names_violations(source):
    """Нарушения контракта «непустой `grid` ⇒ `grid_names`» в вызовах ``plgpu.kernel``."""
    problems = []
    for call in _plgpu_calls(source):
        if call.name != "plgpu.kernel":
            continue
        if call.star_kwargs:
            problems.append(f"строка {call.lineno}: **kwargs в plgpu.kernel — grid/grid_names не проверяемы статически")
            continue
        grid = call.kwargs.get("grid")
        if grid is None:
            continue
        grid_len = _literal_len(grid)
        if grid_len == 0:
            continue
        names = call.kwargs.get("grid_names")
        if names is None:
            problems.append(f"строка {call.lineno}: `grid` без `grid_names`")
            continue
        names_len = _literal_len(names)
        if grid_len is not None and names_len is not None and names_len != grid_len:
            problems.append(
                f"строка {call.lineno}: len(grid_names)={names_len} != len(grid)={grid_len}")
    return problems


def test_kernel_call_declares_grid_names_for_nonempty_grid():
    """(б) Статически: непустой `grid` в вызове `plgpu.kernel` обязан нести `grid_names`."""
    src = Path(mosaic.__file__).read_text(encoding="utf-8")
    assert _plgpu_calls(src), "в модуле обязан быть хотя бы один вызов `plgpu.*`"
    assert _grid_names_violations(src) == [], (
        "вызов `plgpu.kernel` нарушает контракт grid/grid_names: "
        + "; ".join(_grid_names_violations(src)))

    # Различающая сила: мутанты без `grid_names` и с непарной длиной обязаны ловиться.
    assert _grid_names_violations("f = plgpu.kernel(body, out_type=t, grid=(12,))")
    assert _grid_names_violations('f = plgpu.kernel(body, out_type=t, grid=(12,), grid_names=("a", "b"))')
    assert _grid_names_violations("f = plgpu.kernel(body, out_type=t, grid=(12,), **kw)")
    # ...и контроль: пустой grid без имён легален; пары равной длины — легальны.
    assert _grid_names_violations("f = plgpu.kernel(body, out_type=t, grid=())") == []
    assert _grid_names_violations('f = plgpu.kernel(body, out_type=t, grid=(12,), grid_names=("head",))') == []


def test_axis_index_names_are_declared_in_grid_names():
    """Оси, читаемые `jax.lax.axis_index(name)`, объявлены в `grid_names` ядра.

    Контракт 0.11.2: координата программы берётся из именованной оси сетки
    (`program_id` в MGPU deprecated); ось без объявления — ошибка конфигурации ядра.
    """
    src = Path(mosaic.__file__).read_text(encoding="utf-8")
    declared = set()
    for call in _plgpu_calls(src):
        if call.name != "plgpu.kernel" or "grid_names" not in call.kwargs:
            continue
        try:
            names = ast.literal_eval(call.kwargs["grid_names"])
        except (ValueError, TypeError, SyntaxError):
            continue  # не литерал — сверка недостижима статически
        declared.update(str(name) for name in names)

    used = []
    for node in ast.walk(ast.parse(src)):
        if not isinstance(node, ast.Call) or _dotted(node.func) != "jax.lax.axis_index":
            continue
        if len(node.args) == 1 and isinstance(node.args[0], ast.Constant):
            used.append(str(node.args[0].value))

    assert used, "голова обязана выбираться осью `jax.lax.axis_index(...)`"
    assert set(used) <= declared, (
        f"оси {sorted(set(used) - declared)} не объявлены в grid_names")


# ---------------------------------------------------------------------------
# Сверка аргументов всех `plgpu.*`-вызовов модуля с контрактом 0.11.2.
# Источник: разведка стенда `evidence/mfu-55/env/mosaic-0112-probe.json` (подписи
# `kernel`/`mma`/`CompilerParams` из установленного jax 0.11.2) и рецепт, проверенный
# на GB10 (`evidence/mfu-55/mosaic/REPORT-mosaic-chain.md`): `load`/`layout_cast`/
# `SMEM`/`Layout.MMA_*`. Проверяются имена и позиционность; семантика — трассировкой.
# ---------------------------------------------------------------------------


#: name → (число позиционных аргументов, разрешённые keyword-имена) в контракте 0.11.2.
PLGPU_0112_CONTRACT = {
    "plgpu.kernel": (1, frozenset({
        "out_type", "scratch_types", "compiler_params", "grid", "grid_names",
        "cluster", "cluster_names", "num_threads", "thread_name", "interpret", "debug",
    })),
    "plgpu.mma": (3, frozenset()),  # (acc, a, b, /) — только позиционные
    "plgpu.load": (1, frozenset({"layout", "optimized", "idx"})),
    "plgpu.layout_cast": (2, frozenset()),
    "plgpu.SMEM": (2, frozenset({"transforms", "packed", "collective", "layout"})),
    "plgpu.CompilerParams": (0, frozenset()),
    "plgpu.Layout.MMA_ACC": (1, frozenset()),
    "plgpu.Layout.MMA_LHS": (1, frozenset()),
    "plgpu.Layout.MMA_RHS": (1, frozenset()),
}


def test_plgpu_call_arguments_match_the_0112_contract():
    """Аргументы всех `plgpu.*`-вызовов модуля соответствуют контракту 0.11.2 (таблица выше)."""
    src = Path(mosaic.__file__).read_text(encoding="utf-8")
    calls = _plgpu_calls(src)
    assert calls, "модуль обязан вызывать Mosaic-API"

    problems = []
    for call in calls:
        contract = PLGPU_0112_CONTRACT.get(call.name)
        if contract is None:
            problems.append(f"строка {call.lineno}: вызов {call.name} вне сверенной таблицы 0.11.2")
            continue
        n_pos, allowed = contract
        if call.star_kwargs:
            problems.append(f"строка {call.lineno}: {call.name}: **kwargs не сверить с подписью")
        if len(call.args) != n_pos:
            problems.append(
                f"строка {call.lineno}: {call.name}: позиционных {len(call.args)}, ожидается {n_pos}")
        extra = set(call.kwargs) - allowed
        if extra:
            problems.append(
                f"строка {call.lineno}: {call.name}: неизвестные keyword-имена {sorted(extra)}")
    assert not problems, "; ".join(problems)


# ---------------------------------------------------------------------------
# Раскладка RHS (дефект №5, решённый на GB10). ``plgpu.mma`` требует
# ``k``-контигуальную память RHS, поэтому правая часть ``B: (H, C, D) = (k, n)``
# подаётся в ядро транспонированной в памяти — ``b.transpose(0, 2, 1)``
# (``(H, D, C) = (n, k)``), и внутри грузится ``plgpu.load(<ref>.T, MMA_RHS)``.
# Источник: ``evidence/mfu-55/mosaic/REPORT-rhs-layout.md``.
# ---------------------------------------------------------------------------


def _callee(node):
    """Точечное имя вызываемого: для ``Call(func=Attribute(...))`` — имя ``func``."""
    return _dotted(node.func if isinstance(node, ast.Call) else node)


def _has_transpose_021(node):
    """Есть ли в поддереве вызов ``<expr>.transpose(0, 2, 1)`` (паттерн подачи RHS)."""
    for sub in ast.walk(node):
        if not isinstance(sub, ast.Call):
            continue
        func = sub.func
        if not (isinstance(func, ast.Attribute) and func.attr == "transpose"):
            continue
        if len(sub.args) == 3 and all(
            isinstance(arg, ast.Constant) and arg.value == value
            for arg, value in zip(sub.args, (0, 2, 1))
        ):
            return True
    return False


def _function_def(source, name):
    """Узел ``FunctionDef`` с данным именем (или ``None``)."""
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    return None


def test_kernel_transposes_rhs_in_memory(monkeypatch):
    """(1) ``kernel()`` подаёт правую часть транспонированной: ``(H, C, D) -> (H, D, C)``.

    Различающая сила — spy на ``_build_kernel``: фиксируется форма буфера, реально
    уходящего в ядро. До дефекта №5 правая часть уходила ``(H, M, N) = (k, n)``; по
    рецепту она уходит ``(H, D, C->M) = (n, k)`` (транспозиция в памяти), а паддинг C->M
    применён к оси k. Плюс статика: в теле ``kernel`` обязана быть буквальная
    ``b.transpose(0, 2, 1)``.
    """
    src = Path(mosaic.__file__).read_text(encoding="utf-8")
    kernel_def = _function_def(src, "kernel")
    assert kernel_def is not None, "в модуле обязана быть функция kernel()"
    assert _has_transpose_021(kernel_def), (
        "kernel() обязан транспонировать правую часть в памяти: b.transpose(0, 2, 1)")

    seen = {}

    def fake_build(C, N, H, dtype, *, bn=mosaic.DEFAULT_BN):
        M = mosaic.packed_shape(C)

        def fn(a_pad, bt_pad):
            seen["a"] = tuple(a_pad.shape)
            seen["bt"] = tuple(bt_pad.shape)
            return jnp.zeros((H, M, N), a_pad.dtype)

        return fn

    monkeypatch.setattr(mosaic, "_build_kernel", fake_build)
    monkeypatch.setattr(mosaic, "mosaic_available", lambda: True)
    monkeypatch.setattr(mosaic, "gpu_available", lambda: True)

    H, C, D = 2, 64, 96
    M = mosaic.packed_shape(C)
    out = mosaic.kernel(jnp.zeros((H, C, C), jnp.float32), jnp.zeros((H, C, D), jnp.float32))

    assert seen["a"] == (H, M, M), "LHS (I+L) паддится C->M без транспозиции"
    assert seen["bt"] == (H, D, M), (
        "RHS обязана уходить транспонированной: (H, D, C->M) = (n, k), а не (H, C->M, D)")
    assert tuple(out.shape) == (H, C, D), "наружу форма снимается обратно до (H, C, D)"


def test_body_rhs_operand_is_loaded_transposed():
    """(2) В ``body`` RHS грузится через ``.T``: ``plgpu.load(<ref>.T, MMA_RHS)``.

    ``k``-контигуальность памяти RHS достигается только ``.T`` от row-major источника
    (рецепт GB10). Статическая проверка исходника — не зависит от достижимости
    трассировки в этом окружении.
    """
    src = Path(mosaic.__file__).read_text(encoding="utf-8")
    transposed = []
    for call in _plgpu_calls(src):
        if call.name != "plgpu.load":
            continue
        layout = call.kwargs.get("layout")
        if layout is None or _callee(layout) != "plgpu.Layout.MMA_RHS":
            continue
        source = call.args[0] if call.args else None
        if isinstance(source, ast.Attribute) and source.attr == "T":
            transposed.append(call.lineno)
    assert transposed, (
        "RHS обязан грузиться через `.T` от источника (n, k): "
        "`plgpu.load(<ref>.T, layout=plgpu.Layout.MMA_RHS(...))`; иначе память RHS "
        "не k-контигуальна (UnsupportedTransferError)")
