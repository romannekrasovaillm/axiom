"""Fused dispatch/scatter маршрутизированных MoE-экспертов (ADR-054 Am.58).

Зачем один запуск (измеренный адрес)
------------------------------------
Am.56/57 разложили ``sec_bwd_moe`` = 7.144 с на fused-кернел экспертного
блока **3.55 с** и **XLA-часть ~3.6 с**: ``expert_of → stable-sort → take(z) →
ragged → scatter(dz)`` — это мелкие кернелы и латентность на 24 слоях, а не
арифметика и не память (~600 МБ при 417 ГБ/с = мс).  Этот модуль переносит
раскладку/сборку строк в Mosaic GPU:

* **dispatch** — по (token, top-k) парам собирает ``z_grouped`` в каноническом
  порядке (стабильно по эксперту, внутри эксперта — по порядку появления,
  ровно как ``jnp.argsort(expert_of, stable=True)`` forward'а), пишет
  ``group_sizes``, ``assign_rows`` (позиция каждой пары в группировке) и
  ``pair_of_row`` (обратная перестановка — пара каждой сгруппированной строки,
  нужна backward'у);
* **combine** — по котангенту ``du`` собирает ``de_grouped`` (та же
  перестановка, что forward; ``de = du·sel_p``) и ``dsel_p = <du, e_k>``;
* **scatter** — по ``dz_grouped`` собирает ``dz``-буфер ``(tokens, k, latent)``
  (каждая позиция пишется ровно один раз своим экспертом), сумма по ``k`` —
  фиксированного порядка в обёртке.

Граница ответственности
-----------------------
Кернел — только раскладка/сборка строк.  Экспертные GEMM'ы остаются за
``net/moe.py`` (ragged) и fused-backward'ом (``moe_experts_bwd``).  Гейт
``AXIOM_MOE_DISPATCH`` по умолчанию ``"jax"`` — прежняя XLA-последовательность
побитово; ``"fused"`` включает этот модуль.

Интерфейс и цепочка backward
----------------------------
* :func:`kernel` → ``(z_grouped, group_sizes, assign_rows, pair_of_row)``;
  ``row_of_token = assign_rows.reshape(tokens, k)`` — позиция каждой
  (token, slot)-пары (тот же канонический порядок, что forward);
* :func:`combine` → ``(de_grouped, dsel_p)``: ``de_grouped[r] =
  du[token]·sel_p[token, slot]`` — котангент сгруппированных выходов
  (``de = du·sel_p``, как в прежнем пути), ``dsel_p = <du, e_grouped>``.
  ``e_grouped`` и ``du`` — необходимые входы: градиент маршрута
  ``<du, e_k>`` без forward-выхода эксперта невыразим; в тексте задачи
  список входов combine называет только ``de_grouped``/``sel_p``;
* :func:`scatter` → ``dz = Σ_k dz_grouped[row]``; порядок слотов ``k``
  фиксирован.  Это эквивалент суммы ``Σ_k sel_p·de_raw``, где ``de_raw`` —
  котангент выхода эксперта до умножения на ``sel_p`` (в ``combine`` он уже
  домножен, поэтому повторное умножение дало бы ``sel_p²``).

Канонический порядок и детерминизм
----------------------------------
Позиция пары ``i`` (``i = token·k + слот``) в группировке —
``assign_rows[i] = prefix[expert_of[i]] + rank_i``, где ``rank_i`` — число более
ранних пар того же эксперта.  Это обратная перестановка к стабильному
``argsort``: внутри эксперта порядок — порядок появления.  Суммирование в
``combine``/``scatter`` идёт по фиксированным диапазонам эксперта, а сумма
``k``-вкладов — по фиксированному порядку слотов, поэтому два запуска дают
побитово один результат (никаких атомиков).

Пустая группа — пустой диапазон ``[start, start)``: ни одной итерации цикла,
никаких делений.  ``rows % bm == 0`` — fail-closed (:func:`valid_config`):
блочный скан пар не имеет клампа последнего блока, некратные ``rows`` дали бы
потерянные/дважды посчитанные пары.

Статус проверки (честно)
------------------------
* кернел **не исполнялся на устройстве**: узел CPU-only, interpret-режим
  jax 0.11.2 не имеет правил для Mosaic GPU;
* *lowering* для sm_121 проверяется на этом узле
  (``net/tests/moe_dispatch_lower.py`` + ``test_55_moe_dispatch.py``): живые
  ``CASES`` обязаны собраться, невалидные — честно скипнуться ситом;
* численная семантика пинуется jnp-эквивалентами :func:`reference_dispatch`,
  :func:`reference_combine`, :func:`reference_scatter` (CPU): тесты подменяют
  шов этими функциями и сверяют fused-путь с autodiff ``jax``-пути;
* структура выхода Pallas пинуется отдельно: ``out_type`` — дерево, поэтому
  один выход scatter-кернела приходит кортежем ``(dz_k,)``.  Обёртки
  :func:`combine`/:func:`scatter` распаковывают его явно (цикл-36: ``jnp.sum``
  на кортеже падал с "sum requires ndarray or scalar arguments"); модельный
  композит с настоящими обёртками и низкоуровневыми jnp-швами —
  ``net/tests/test_56_moe_compose.py``;
* паритет *исполненного* кернела, время и детерминизм на GB10 — за
  архитектором (``check_kernel.py``); до стенда паритет не заявляется.
"""

from __future__ import annotations

import dataclasses
import functools

import jax
import jax.numpy as jnp

# ---------------------------------------------------------------------------
# Контракт модуля (скилл pallas-gb10-kernel)
# ---------------------------------------------------------------------------

#: Формы случаев.  ``tokens·k = rows``; живая кампанийная геометрия —
#: ``latent=768``, ``k=2``, ``n_exp=12``.  Последний случай объявлен как
#: невалидный: ``rows=300`` не делится на ``bm=16`` — сито :func:`valid_config`
#: его честно скипает (fail-closed, а не молчаливый кламп).
CASES = [
    dict(tokens=1024, k=2, latent=768, n_exp=12, dtype="bfloat16"),
    dict(tokens=128, k=2, latent=768, n_exp=12, dtype="bfloat16"),
    dict(tokens=100, k=3, latent=768, n_exp=12, dtype="bfloat16"),  # rows % bm != 0
]

#: Операнд-путь — bf16; допуск паритета с autodiff — scale-relative 2e-2
#: (объявленный кампанийный допуск bf16, форма cyclic-30).
TOLERANCE = {"bfloat16": (2e-2, 2e-2)}

#: Статические ручки: блок пар ``bm`` (скан раскладки) и тайл колонок ``bc``
#: (копия строки ``latent`` кусками).  ``bm=16``/``bc=128`` — та же форма
#: тайлинга, что у живого ``moe_experts_bwd`` (16/128/32).
TUNE_SPACE = dict(bm=[16], bc=[128])

#: Бюджет SMEM одного блока GB10 (99 КиБ) минус эпизодический запас.
SMEM_LIMIT_BYTES = 101376
SMEM_EPILOGUE_BYTES = 4 * 1024

BM_DEFAULT = 16
BC_DEFAULT = 128


class MoeDispatchKernelError(RuntimeError):
    """Типизированный отказ fused dispatch/scatter.

    Возникает, когда (а) конфигурация вне живых случаев сита — форма/тип не
    опускаются на sm_121 либо нарушено ``rows % bm == 0``, либо (б) на узле
    нет GPU: Mosaic-кернел исполняется только на устройстве.  Тихий откат к
    ``jax`` запрещён — режим выбирается гейтом ``AXIOM_MOE_DISPATCH`` явно.
    """


def _smem_floor_bytes(bm, bc, itemsize=2, acc_itemsize=4):
    """Объявленный SMEM-пол одной программы.

    Кернел читает/пишет строку кусками ``bc``: буфер строки ``(1, bc)`` bf16,
    аккумулятор скалярного произведения ``(bc,)`` fp32 и запас эпилога.
    """
    smem = 2 * bc * itemsize + bc * acc_itemsize
    return smem + SMEM_EPILOGUE_BYTES


def valid_config(case, params):
    """Статическое сито: операнд-путь, кратности и ``rows % bm == 0``.

    Живые случаи — bf16/fp32 (движение строк), ``latent`` кратен ``bc``, ``bc``
    кратен 128 (загрузка строкового тайла), ``bm`` кратен 16 и
    ``tokens·k % bm == 0``.  Некратные
    ``rows`` — fail-closed: блочный скан не кламплит последний блок, и пары
    хвоста потерялись бы.  fp32 и некратная геометрия возвращают False — это
    честный скип lowering'а, а не молчаливый откат.
    """
    tokens = int(case["tokens"])
    k = int(case["k"])
    latent = int(case["latent"])
    n_exp = int(case["n_exp"])
    bm = int(params.get("bm", case.get("bm", BM_DEFAULT)))
    bc = int(params.get("bc", case.get("bc", BC_DEFAULT)))
    dt = jnp.dtype(case.get("dtype", "bfloat16"))
    rows = tokens * k
    # Диспетчеризация/сборка — движение данных: bf16 и fp32 допустимы.
    # fp32 сохраняет бит-в-бит forward на fp32-пути кампании; bf16 — объявленный
    # операнд-путь mma.  Аккумуляторы (dsel_p, dz) всегда fp32.
    if dt not in (jnp.bfloat16, jnp.float32):
        return False
    if tokens < 1 or k < 1 or n_exp < 1 or latent < 1:
        return False
    if bm < 16 or bm % 16 or bc < 128 or bc % 128:
        return False
    if latent % bc:
        return False
    if rows % bm:
        return False
    return _smem_floor_bytes(bm, bc) <= SMEM_LIMIT_BYTES


def cost(tokens, k, latent, n_exp, bm=BM_DEFAULT, bc=BC_DEFAULT,
         dtype="bfloat16", **kw):
    """Минимальные FLOPs/байты одного dispatch-запуска (roofline-учёт).

    Арифметики почти нет (сравнения эксперта), поэтому доминируют байты:
    чтение ``z`` и запись ``z_grouped`` (по ``rows·latent``), скан ``topk``
    каждой из ``n_exp`` программ (``n_exp·rows`` int32) и служебные
    ``assign_rows``/``pair_of_row``.
    """
    b = jnp.dtype(dtype).itemsize
    rows = tokens * k
    moves = 2 * rows * latent * b
    scan = n_exp * rows * 4
    bookkeeping = 2 * rows * 4
    flops = n_exp * rows  # сравнения «эксперт пары == e»
    return dict(flops=2 * flops, bytes=moves + scan + bookkeeping)


def make_inputs(key, tokens, k, latent, n_exp, dtype="bfloat16", **kw):
    """Входы одного dispatch-вызова: ``z``, ``topk_idx``, ``sel_p``.

    Раскладка намеренно содержит **пустую группу** (эксперт 1) и неровные
    группы: границы, которые кернел обязан пережить.  ``sel_p`` для dispatch
    не используется (веса маршрута нужны combine), но входит в контракт
    вызова — форма пинуется тестом.
    """
    dt = jnp.dtype(dtype)
    k1, k2 = jax.random.split(key, 2)
    rows = int(tokens) * int(k)
    z = (jax.random.normal(k1, (int(tokens), int(latent)), jnp.float32) * 0.1).astype(dt)
    idx = (jnp.arange(rows, dtype=jnp.int32) * 5 + 3) % int(n_exp)
    if n_exp > 1:
        idx = jnp.where(idx == 1, 0, idx)  # пустая группа эксперта 1
    topk = idx.reshape(int(tokens), int(k)).astype(jnp.int32)
    w = jax.random.uniform(k2, (int(tokens), int(k)), jnp.float32) + 0.1
    sel_p = w / jnp.sum(w, axis=-1, keepdims=True)
    return z, topk, sel_p


def device_available() -> bool:
    """True, когда JAX видит GPU-устройство (Mosaic требует устройство).

    Lowering устройства не требует (``lower_check`` работает ровно так), —
    поэтому проверка стоит в :func:`kernel`/``combine``/``scatter``, а не в
    построителе.
    """
    try:
        return any(d.platform == "gpu" for d in jax.devices())
    except Exception:  # noqa: BLE001 — сломанный плагин не должен читаться как GPU
        return False


# ---------------------------------------------------------------------------
# Адресация раскладки (единственный источник истины)
# ---------------------------------------------------------------------------


def group_offsets(group_sizes):
    """Префиксные суммы ``group_sizes``: начало блока каждого эксперта.

    ``offsets[e] = Σ_{j<e} group_sizes[j]``; пустая группа даёт
    ``start == end`` — пустой диапазон без итераций и делений.
    """
    gs = group_sizes.astype(jnp.int32)
    return jnp.cumsum(gs) - gs


def assignment_position(expert_of, group_sizes, n_exp, k):
    """Позиция каждой (token, slot)-пары в канонической группировке.

    ``pos[i] = offsets[expert_of[i]] + rank_i``; ``rank_i`` — число более
    ранних пар того же эксперта (порядок появления).  Это ровно обратная
    перестановка к ``jnp.argsort(expert_of, stable=True)``, поэтому fused-путь
    и ``jax``-путь дают один и тот же канонический порядок групп/строк.
    Реализация — CPU-эталон и проверка адресации; кернел считает ту же
    величину сканом.
    """
    expert_of = expert_of.reshape(-1)
    offsets = group_offsets(group_sizes)
    one_hot = jax.nn.one_hot(expert_of, n_exp, dtype=jnp.int32)
    ranks = jnp.cumsum(one_hot, axis=0) - 1
    rank_of = jnp.take_along_axis(ranks, expert_of[:, None], axis=1)[:, 0]
    return offsets[expert_of] + rank_of


# ---------------------------------------------------------------------------
# Mosaic GPU-кернелы
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class _KernelSpec:
    """Статические формы, под которые построен кернел (ключ jit-кеша)."""

    tokens: int
    k: int
    latent: int
    n_exp: int
    bm: int
    bc: int
    dtype: str


def _load_row(ref, row, c0, width):
    """Прочитать строку ``row`` в колонках ``[c0, c0+width)`` как вектор."""
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import mosaic_gpu as plgpu

    return plgpu.load(ref.at[row, pl.ds(c0, width)], optimized=False).reshape(width)


def _store_row(out, row, c0, value, width):
    """Записать вектор ``value`` в строку ``row``, колонки ``[c0, c0+width)``."""
    from jax.experimental import pallas as pl

    out[row, pl.ds(c0, width)] = value


@functools.lru_cache(maxsize=8)
def _build_kernel(tokens, k, latent, n_exp, bm, bc, dtype):
    """Построить dispatch-кернел: одна программа на эксперта.

    Программа эксперта ``e`` сканирует пары блоками ``bm`` и «уплотняет» свои
    пары в диапазон ``[offsets[e], offsets[e] + group_sizes[e])``.  Параметр
    ``cnt`` — число уже собранных пар эксперта (переносится между блоками);
    позиция пары — ``start + cnt``.  Чужие пары пишутся в dump-строку
    ``rows + e``: программа пишет только свои строки и собственный слот,
    гонки между программами нет.
    """
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import mosaic_gpu as plgpu

    dt = jnp.dtype(dtype)
    rows = tokens * k
    n_col = latent // bc

    def _body(zg, topk, sizes, offsets, zg_out, sizes_out, assign_out,
              pair_out):
        e = jax.lax.axis_index("expert")
        start = offsets[e]
        size = sizes[e]
        sizes_out[e] = size
        dump = rows + e

        def _pair(i, cnt):
            """Одна пара: своя — на позицию ``start+cnt``, чужая — в dump."""
            own = topk[i] == e
            pos = start + cnt
            dst = jnp.where(own, pos, dump)
            tok = i // k
            for c0 in range(n_col):
                _store_row(zg_out, dst, c0 * bc,
                           _load_row(zg, tok, c0 * bc, bc), bc)
            # Позиция пары в группировке.  Чужие пары уходят в свой dump-слот
            # (не в индекс i — иначе программы экспертов гонялись бы за одну
            # ячейку assign_rows).
            assign_out[jnp.where(own, i, dump)] = pos
            pair_out[dst] = i
            return cnt + own.astype(jnp.int32)

        def _block(b, cnt):
            base = b * bm
            for j in range(bm):
                cnt = _pair(base + jnp.int32(j), cnt)
            return cnt

        pl.loop(0, rows // bm, init_carry=jnp.int32(0))(_block)

    return plgpu.kernel(
        _body,
        out_type=(
            jax.ShapeDtypeStruct((rows + n_exp, latent), dt),  # z_grouped + dump
            jax.ShapeDtypeStruct((n_exp,), jnp.int32),          # group_sizes
            jax.ShapeDtypeStruct((rows + n_exp,), jnp.int32),   # assign_rows
            jax.ShapeDtypeStruct((rows + n_exp,), jnp.int32),   # pair_of_row
        ),
        scratch_types=[],
        compiler_params=dataclasses.replace(
            plgpu.CompilerParams(),
            lowering_semantics=plgpu.LoweringSemantics.Lane,
        ),
        grid=(n_exp,),
        grid_names=("expert",),
    )


@functools.lru_cache(maxsize=8)
def _build_combine_kernel(tokens, k, latent, n_exp, bm, bc, dtype):
    """Построить combine-кернел: ``de_grouped`` и ``dsel_p`` за один запуск.

    Программа эксперта ``e`` идёт по своим строкам ``r`` (диапазон
    ``[start, end)``): ``pair = pair_of_row[r]``, ``token = pair // k``,
    ``slot = pair % k``; ``de_grouped[r] = du[token]·sel_p[token, slot]`` и
    ``dsel_p[token, slot] = <du[token], e_grouped[r]>`` (fp32-аккумулятор,
    фиксированный порядок колонок).  Каждая строка и каждая (token, slot)-
    ячейка пишутся ровно одной программой — атомиков и гонок нет.
    """
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import mosaic_gpu as plgpu

    dt = jnp.dtype(dtype)
    rows = tokens * k
    f32 = jnp.float32
    n_col = latent // bc

    def _body(du, eg, pair, sel, sizes, offsets, de_out, dsel_out):
        e = jax.lax.axis_index("expert")
        start = offsets[e]
        end = start + sizes[e]

        def _row(r):
            p = pair[r]
            tok = p // k
            slot = p % k
            w = sel[tok, slot]
            acc = jnp.zeros((), f32)
            for c0 in range(n_col):
                duv = _load_row(du, tok, c0 * bc, bc).astype(f32)
                egv = _load_row(eg, r, c0 * bc, bc).astype(f32)
                _store_row(de_out, r, c0 * bc, (duv * w).astype(dt), bc)
                acc = acc + jnp.sum(duv * egv)
            dsel_out[tok, slot] = acc

        pl.loop(start, end)(_row)

    return plgpu.kernel(
        _body,
        out_type=(
            jax.ShapeDtypeStruct((rows, latent), dt),        # de_grouped
            jax.ShapeDtypeStruct((tokens, k), f32),          # dsel_p
        ),
        scratch_types=[],
        compiler_params=dataclasses.replace(
            plgpu.CompilerParams(),
            lowering_semantics=plgpu.LoweringSemantics.Lane,
        ),
        grid=(n_exp,),
        grid_names=("expert",),
    )


@functools.lru_cache(maxsize=8)
def _build_scatter_kernel(tokens, k, latent, n_exp, bm, bc, dtype):
    """Построить scatter-кернел: ``dz_grouped`` → буфер ``(tokens, k, latent)``.

    Программа эксперта ``e`` идёт по своим строкам ``r`` и кладёт строку в
    ``dz_k[token, slot]`` — каждая ячейка пишется ровно один раз своим
    экспертом.  Сумма по ``k`` выполняется в обёртке фиксированным порядком
    слотов, поэтому результат детерминирован.
    """
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import mosaic_gpu as plgpu

    dt = jnp.dtype(dtype)
    rows = tokens * k
    n_col = latent // bc

    def _body(dzg, pair, sizes, offsets, dzk_out):
        e = jax.lax.axis_index("expert")
        start = offsets[e]
        end = start + sizes[e]

        def _row(r):
            p = pair[r]
            tok = p // k
            slot = p % k
            for c0 in range(n_col):
                dzk_out[tok, slot, pl.ds(c0 * bc, bc)] = _load_row(
                    dzg, r, c0 * bc, bc).astype(jnp.float32)

        pl.loop(start, end)(_row)

    return plgpu.kernel(
        _body,
        out_type=(
            # dz — градиент: fp32-аккумулятор, как ``dz_grouped`` экспертного
            # backward'а (bf16 — путь операндов, не выхода).
            jax.ShapeDtypeStruct((tokens, k, latent), jnp.float32),  # dz_k
        ),
        scratch_types=[],
        compiler_params=dataclasses.replace(
            plgpu.CompilerParams(),
            lowering_semantics=plgpu.LoweringSemantics.Lane,
        ),
        grid=(n_exp,),
        grid_names=("expert",),
    )


@functools.lru_cache(maxsize=8)
def _kernel_run(tokens, k, latent, n_exp, bm, bc, dtype):
    """jit-обёртка построенного dispatch-кернела (кеш по спецификации)."""
    return jax.jit(_build_kernel(tokens, k, latent, n_exp, bm, bc, dtype))


@functools.lru_cache(maxsize=8)
def _combine_run(tokens, k, latent, n_exp, bm, bc, dtype):
    """jit-обёртка построенного combine-кернела (кеш по спецификации)."""
    return jax.jit(_build_combine_kernel(tokens, k, latent, n_exp, bm, bc, dtype))


@functools.lru_cache(maxsize=8)
def _scatter_run(tokens, k, latent, n_exp, bm, bc, dtype):
    """jit-обёртка построенного scatter-кернела (кеш по спецификации)."""
    return jax.jit(_build_scatter_kernel(tokens, k, latent, n_exp, bm, bc, dtype))


def _prepare(z, topk_idx, n_exp, bm, bc):
    """Общая проверка формы/сита и префиксные суммы ``group_sizes``."""
    tokens, latent = int(z.shape[0]), int(z.shape[1])
    k = int(topk_idx.shape[1])
    case = dict(tokens=tokens, k=k, latent=latent, n_exp=int(n_exp),
                dtype=jnp.dtype(z.dtype).name)
    params = dict(bm=int(bm), bc=int(bc))
    if not valid_config(case, params):
        raise MoeDispatchKernelError(
            f"fused MoE-dispatch: конфигурация {case} (bm={bm}, bc={bc}) вне "
            "живых случаев сита valid_config (bf16, кратности mma, "
            "rows % bm == 0); тихого отката нет")
    group_sizes = jnp.bincount(topk_idx.reshape(-1).astype(jnp.int32),
                               length=int(n_exp)).astype(jnp.int32)
    return tokens, k, latent, int(n_exp), group_sizes


def kernel(z, topk_idx, sel_p, *, n_exp, bm=BM_DEFAULT, bc=BC_DEFAULT):
    """Fused dispatch: ``(z, topk_idx, sel_p)`` → группировка строк.

    Возвращает ``(z_grouped, group_sizes, assign_rows, pair_of_row)``:

    * ``z_grouped (rows, latent)`` — строки в каноническом порядке (стабильно
      по эксперту, внутри — порядок появления пары);
    * ``group_sizes (n_exp,)`` — размеры групп (префиксные суммы — снаружи,
      :func:`group_offsets`);
    * ``assign_rows (rows,)`` — позиция каждой (token, slot)-пары в
      группировке (``row_of_token = assign_rows.reshape(tokens, k)``);
    * ``pair_of_row (rows,)`` — обратная перестановка (какая пара лежит в
      строке), нужна combine/scatter.

    ``rows % bm == 0`` — требование сита :func:`valid_config` (отказ —
    :class:`MoeDispatchKernelError`, без тихой ошибки).  Исполнение требует
    Mosaic GPU (GB10); на CPU-узле — типизированный отказ.  *Lowering*
    устройства не требует и проверяется отдельно
    (``net/tests/moe_dispatch_lower.py``).
    """
    tokens, k, latent, n_exp, group_sizes = _prepare(
        z, topk_idx, n_exp, bm, bc)
    if not device_available():
        raise MoeDispatchKernelError(
            "fused MoE-dispatch: Mosaic-кернел исполняется только на GPU "
            "(sm_121); на CPU-узле выберите AXIOM_MOE_DISPATCH=jax. "
            "Тихого отката нет — режим выбирается явно.")
    offsets = group_offsets(group_sizes)
    zg, sizes, assign, pair = _kernel_run(
        tokens, k, latent, n_exp, int(bm), int(bc),
        jnp.dtype(z.dtype).name)(z, topk_idx.reshape(-1), group_sizes, offsets)
    rows = tokens * k
    return zg[:rows], sizes, assign[:rows], pair[:rows]


def combine(du, e_grouped, pair_of_row, sel_p, group_sizes, *, n_exp,
            bm=BM_DEFAULT, bc=BC_DEFAULT):
    """Fused combine: ``du`` → ``(de_grouped, dsel_p)``.

    ``de_grouped[row] = du[token]·sel_p[token, slot]`` — котангент
    сгруппированных выходов экспертов в каноническом порядке (та же
    перестановка, что forward); ``dsel_p[token, slot] = <du[token],
    e_grouped[row]>`` — градиент маршрутных весов (fp32-аккумулятор,
    фиксированный порядок колонок).  Каждая строка/ячейка пишется ровно одной
    программой эксперта — атомиков нет, результат детерминирован.
    """
    tokens, k = int(sel_p.shape[0]), int(sel_p.shape[1])
    latent = int(du.shape[1])
    case = dict(tokens=tokens, k=k, latent=latent, n_exp=int(n_exp),
                dtype=jnp.dtype(du.dtype).name)
    if not valid_config(case, dict(bm=int(bm), bc=int(bc))):
        raise MoeDispatchKernelError(
            f"fused MoE-combine: конфигурация {case} (bm={bm}, bc={bc}) вне "
            "живых случаев сита valid_config (bf16/fp32, кратности mma, "
            "rows % bm == 0); тихого отката нет")
    group_sizes = group_sizes.astype(jnp.int32)
    if not device_available():
        raise MoeDispatchKernelError(
            "fused MoE-combine: Mosaic-кернел исполняется только на GPU "
            "(sm_121); на CPU-узле выберите AXIOM_MOE_DISPATCH=jax. "
            "Тихого отката нет — режим выбирается явно.")
    offsets = group_offsets(group_sizes)
    # ``_combine_run`` — Pallas-дерево из двух выходов: распаковка явная,
    # дальше ровно по одному ndarray на величину (цикл-36).
    de_grouped, dsel_p = _combine_run(
        tokens, k, latent, n_exp, int(bm), int(bc),
        jnp.dtype(du.dtype).name)(du, e_grouped, pair_of_row, sel_p,
                                  group_sizes, offsets)
    return de_grouped, dsel_p


def scatter(dz_grouped, pair_of_row, group_sizes, *, n_tokens, k, n_exp,
            bm=BM_DEFAULT, bc=BC_DEFAULT):
    """Fused scatter: ``dz_grouped (rows, latent)`` → ``dz (tokens, latent)``.

    Программа эксперта кладёт свои строки в ``(token, slot)``-буфер; сумма по
    ``slot`` — фиксированного порядка (``jnp.sum(axis=1)``), поэтому дерево
    суммирования не зависит от планировщика.  Пустая группа — пустой диапазон.
    """
    k = int(k)
    n_tokens = int(n_tokens)
    rows = n_tokens * k
    latent = int(dz_grouped.shape[1])
    case = dict(tokens=n_tokens, k=k, latent=latent, n_exp=int(n_exp),
                dtype=jnp.dtype(dz_grouped.dtype).name)
    if not valid_config(case, dict(bm=int(bm), bc=int(bc))):
        raise MoeDispatchKernelError(
            f"fused MoE-scatter: конфигурация {case} (bm={bm}, bc={bc}) вне "
            "живых случаев сита valid_config; тихого отката нет")
    if not device_available():
        raise MoeDispatchKernelError(
            "fused MoE-scatter: Mosaic-кернел исполняется только на GPU "
            "(sm_121); на CPU-узле выберите AXIOM_MOE_DISPATCH=jax. "
            "Тихого отката нет — режим выбирается явно.")
    offsets = group_offsets(group_sizes)
    # ``out_type`` scatter-кернела — дерево из ОДНОГО ``ShapeDtypeStruct``,
    # поэтому Pallas возвращает кортеж ``(dz_k,)``, а не голый массив
    # (цикл-36: ``jnp.sum`` на этом кортеже падал с "sum requires ndarray or
    # scalar arguments, got <class 'tuple'>"). Распаковка явная: дальше —
    # ровно один ndarray ``(tokens, k, latent)``.
    (dz_k,) = _scatter_run(
        n_tokens, k, latent, int(n_exp), int(bm), int(bc),
        jnp.dtype(dz_grouped.dtype).name)(
            dz_grouped, pair_of_row, group_sizes, offsets)
    return jnp.sum(dz_k, axis=1)


# ---------------------------------------------------------------------------
# jnp-эквиваленты (CPU-паритет; численный пин кернелов)
# ---------------------------------------------------------------------------


def reference_dispatch(z, topk_idx, sel_p=None, *, n_exp,
                       bm=BM_DEFAULT, bc=BC_DEFAULT):
    """jnp-эквивалент dispatch: та же каноническая раскладка, что forward.

    ``order = argsort(expert_of, stable=True)``; ``pair_of_row = order``;
    ``assign_rows = argsort(order)`` (обратная перестановка).  Сложность —
    ``O(rows log rows)``; это CPU-пин для малых форм паритета, не исполнитель
    кампанийной геометрии.
    """
    n_tokens, k = int(topk_idx.shape[0]), int(topk_idx.shape[1])
    rows = n_tokens * k
    expert_of = topk_idx.reshape(-1).astype(jnp.int32)
    group_sizes = jnp.bincount(expert_of, length=int(n_exp)).astype(jnp.int32)
    order = jnp.argsort(expert_of, stable=True).astype(jnp.int32)
    assign_rows = jnp.zeros((rows,), jnp.int32).at[order].set(
        jnp.arange(rows, dtype=jnp.int32))
    token_of = jnp.repeat(jnp.arange(n_tokens, dtype=jnp.int32), k)
    z_grouped = z[token_of[order]]
    return z_grouped, group_sizes, assign_rows, order


def reference_combine(du, e_grouped, pair_of_row, sel_p, group_sizes, *,
                      n_exp, bm=BM_DEFAULT, bc=BC_DEFAULT):
    """jnp-эквивалент combine: ``de_grouped`` и ``dsel_p`` без атомиков.

    Перестановка задана ``pair_of_row``: ``token = pair // k``,
    ``slot = pair % k``.  Каждая (token, slot) встречается ровно один раз,
    поэтому ``.at[].set`` — точная запись, а сумма по колонкам — тот же
    fp32-аккумулятор, что в кернеле.
    """
    n_tokens, k = int(sel_p.shape[0]), int(sel_p.shape[1])
    rows = n_tokens * k
    pair = pair_of_row.astype(jnp.int32).reshape(-1)
    tok = pair // k
    slot = pair % k
    f32 = jnp.float32
    de_grouped = du.astype(f32)[tok] * sel_p.astype(f32)[tok, slot][:, None]
    de_grouped = de_grouped.astype(du.dtype)
    dots = jnp.sum(du.astype(f32)[tok] * e_grouped.astype(f32), axis=-1)
    dsel_p = jnp.zeros((n_tokens, k), f32).at[tok, slot].set(dots)
    return de_grouped[:rows], dsel_p


def reference_scatter(dz_grouped, pair_of_row, group_sizes, *, n_tokens, k,
                      n_exp, bm=BM_DEFAULT, bc=BC_DEFAULT):
    """jnp-эквивалент scatter: буфер ``(tokens, k, latent)`` и сумма по ``k``."""
    n_tokens, k = int(n_tokens), int(k)
    latent = int(dz_grouped.shape[1])
    pair = pair_of_row.astype(jnp.int32).reshape(-1)
    tok = pair // k
    slot = pair % k
    dz_k = jnp.zeros((n_tokens, k, latent), jnp.float32).at[tok, slot].set(
        dz_grouped.astype(jnp.float32))
    return jnp.sum(dz_k, axis=1)
