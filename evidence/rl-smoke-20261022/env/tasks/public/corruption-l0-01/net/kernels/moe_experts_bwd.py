"""Fused backward экспертного блока маршрутизированных MoE-экспертов (ADR-054 Am.52).

Зачем один запуск (измеренный адрес)
------------------------------------
Пофазовый профиль цикла-29 отнёс ``sec_bwd_moe`` = 10.235 с (49 % полного
backward 20.98 с, 38 % шага 27.1 с) при ~0.6 TFLOP ≈ 0.06 TFLOPS.  Узкое
место — не арифметика экспертного блока (она мала), а *мелкие кернелы* и
submission overhead: autodiff трёх ``ragged_dot`` и ``siti_glu`` раскладывается
в десятки отдельных запусков на слой.  Этот модуль сворачивает **весь backward
экспертной части** в один ``plgpu.kernel``: ``grid=(n_exp,)``, один программа-
эксперт, все градиенты эксперта (``dwg``, ``dwu``, ``dwd``) и ``dz_grouped``
считаются внутри одного запуска.

Граница ответственности
-----------------------
Кернел — **только backward**.  Forward остаётся прежним ragged-путём
(``net/moe.py:_routed_experts_grouped``): fused-forward измерен минусом
(цикл-29), поэтому гейт ``AXIOM_MOE_BWD`` по умолчанию ``"jax"`` — прежний
autodiff побитово.  В режиме ``"fused"`` forward-правило ``custom_vjp``
дословно повторяет прежний путь (числа те же), а backward-правило зовёт
:func:`kernel` этого модуля; группировка/сборка строк и градиент маршрутизации
(``sel_p``/``topk``) остаются в XLA.

Алгебра, которую кернел транскрибирует (на эксперта ``e``)
----------------------------------------------------------
Входы — уже сгруппированные по экспертам строки (``rows = N * top_k``):
``z_grouped`` (rows, latent), ``h_g``/``h_u``/``a`` (rows, inter),
``de_grouped`` (rows, latent) — котангент выхода экспертного блока; веса
``wg``/``wu`` (n_exp, latent, inter), ``wd`` (n_exp, inter, latent),
``group_sizes`` (n_exp,).  Для каждой строки своего эксперта::

    dwd = a^T · de                                   # (inter, latent)
    da  = de · wd^T                                  # (rows, inter)
    gate = βg·tanh(hg/βg)·sigmoid(hg)
    up   = βu·tanh(hu/βu)
    d(hg) = sech²(hg/βg)·sigmoid(hg)
            + βg·tanh(hg/βg)·sigmoid(hg)·(1−sigmoid(hg))
    dhg = da·up·d(hg)                                # (rows, inter)
    dhu = da·gate·sech²(hu/βu)                       # (rows, inter)
    dwg = z^T · dhg                                  # (latent, inter)
    dwu = z^T · dhu                                  # (latent, inter)
    dz_grouped = dhg·wg^T + dhu·wu^T                 # (rows, latent)

``d(hg)`` — производная ``siti_glu`` по gate-ветви; множитель ``βg`` при
втором слагаемом — из ``d/dhg[βg·tanh(hg/βg)] = sech²(hg/βg)`` (в тексте
задачи он пропущен; здесь производная взята точно, иначе паритет с autodiff
не сходится).

Реализация: три последовательные фазы одной программы
------------------------------------------------------
* **A — строковая**: по блокам ``bm`` строк своего эксперта считает ``da``
  (K-поток по latent), siti-адъюнкт, ``dhg``/``dhu`` (в GMEM-скретч, bf16 —
  объявленный операнд-путь кернела) и ``dz`` (K-поток по inter, тайлы ``bt``
  по latent);
* **B/C/D — весовые**: ``dwd``, ``dwg``, ``dwu`` тайлами ``(bt, bt)`` с
  накоплением в регистрах по всем строкам эксперта (K = строки).

Строки читаются с маской принадлежности эксперту; строки вне группы в
строковых записях уходят в *dump-слот* ``rows + e`` (буферы ``dz``/``dhg``/
``dhu`` имеют ``rows + n_exp`` строк).  Это исключает гонку записи с соседним
экспертом: каждая программа пишет только свои строки и собственный слот.

Операнды — bf16, аккумулятор — fp32 (валидированный путь sm_121 mma; форма
``kda_cc_fused``).  SMEM ≤ 99 КиБ на программу: см. :func:`_smem_floor_bytes`.

Статус проверки (честно)
------------------------
* кернел **не исполнялся на устройстве**: узел CPU-only, а interpret-режим
  jax 0.11.2 не имеет правила для ``plgpu.mma``;
* *lowering* для sm_121 проверяется на этом узле (``lower_check.py`` скилла
  ``pallas-gb10-kernel`` и ``net/tests/moe_experts_bwd_lower.py``): все живые
  ``CASES`` обязаны собраться, oversize — честно скипнуться ситом;
* численная семантика кернела пинуется его jnp-адъюнктом
  :func:`reference_bwd` (CPU): ``net/tests/test_54_moe_bwd_fused.py`` сверяет
  его с autodiff прежнего пути на допуске 2e-2 (bf16);
* цикл-31: эпизод ``dz_grouped`` цикла-30 (стенд: rel 1.28 при исправных
  ``dwg``/``dwu``/``dwd``) локализован в **записи** dz — тайл latent писался
  в колонки ``[0, bt)`` вместо ``[tj·bt, tj·bt+bt)``, поэтому 5/6 колонок
  выхода оставались незаписанными, а первая — перезаписанной последним
  тайлом.  Адресация вынесена в единственный источник истины
  (:func:`dz_tile_columns`, :func:`expert_block_range`,
  :func:`expert_block_offset`, :func:`expert_row_slot`), который делят кернел
  и CPU-зеркало :func:`mirror_dz_path`; структурная проба
  ``test_54_moe_bwd_fused.py`` проверяет адресацию, которую исполняет **само
  тело кернела** (запись строк под регистрирующей обёрткой на lowering), а
  зеркало с исторической политикой «все тайлы в колонку 0» даёт
  ``‖Δdz‖/‖dz‖ ≈ 1.08`` (нулевой буфер) … ``1.41`` (остаточная память):
  наблюдённая на стенде 1.28 лежит внутри этого интервала;
* цикл-32: стенд дал ``dwg = nan`` на малой форме (rows=256, 12 экспертов) —
  у эксперта пустая группа, а весовая фаза читает скретч ``dhg``/``dhu``
  блоками строк: строки соседней группы к моменту чтения могут быть ещё не
  записаны (программы экспертов идут параллельно), и умножение на маску
  ``0 · NaN = NaN`` протаскивало nan в аккумулятор.  Фикс: маскированный
  операнд выбирается через :func:`masked_operand` (select, не умножение),
  а выход весового тайла пустой группы пишется нулями через
  :func:`empty_group_weight_guard` (аккумулятор не используется).  Плюс
  fail-closed ``rows % bm == 0`` в :func:`valid_config`: при некратных rows
  кламп последнего блока перекрывает строки, и весовые фазы посчитали бы их
  дважды;
* паритет *исполненного* кернела, время и детерминизм — замер архитектора на
  GB10 (``check_kernel.py``).
"""

from __future__ import annotations

import dataclasses
import functools

import jax
import jax.numpy as jnp

# ---------------------------------------------------------------------------
# Контракт модуля (скилл pallas-gb10-kernel)
# ---------------------------------------------------------------------------

#: Формы случаев.  ``rows`` — сгруппированные строки (``N * top_k``); живая
#: кампанийная геометрия — ``latent=768``, ``inter=384``, ``n_exp=12``.
#: Последний случай объявлен как oversize: его SMEM-пол по бюджету GB10 не
#: проходит (сито :func:`valid_config` его честно скипает, а не делает вид,
#: что он запускается).
CASES = [
    dict(rows=2048, latent=768, inter=384, n_exp=12, dtype="bfloat16"),
    dict(rows=256, latent=768, inter=384, n_exp=12, dtype="bfloat16"),
    dict(rows=2048, latent=1024, inter=768, n_exp=16, dtype="bfloat16"),  # oversize: skipped
]

#: Операнд-путь кернела — bf16; допуск паритета с autodiff — scale-relative
#: 2e-2 (объявленный кампанийный допуск bf16, форма cyclic-30).
TOLERANCE = {"bfloat16": (2e-2, 2e-2)}

#: Статические ручки :func:`kernel`: блок строк, тайл весового выхода и
#: K-тайл потоковых операндов.  ``bk=32`` — минимальная N-плитка
#: ``MMA_ACC`` (16, 32), на которой опускаются маски строк; он же держит
#: SMEM-пол под 99 КиБ (см. :func:`_smem_floor_bytes`).
TUNE_SPACE = dict(bm=[16], bt=[128], bk=[32])

#: Бюджет SMEM одного блока GB10 (99 КиБ) минус эпизодический запас эпилога
#: (4 КиБ) — та же величина, что в ``kda_cc_fused``.
SMEM_LIMIT_BYTES = 101376
SMEM_EPILOGUE_BYTES = 4 * 1024

BM_DEFAULT = 16
BT_DEFAULT = 128
BK_DEFAULT = 32

#: β SiTU-GLU (``net/config.py:siti_beta_gate/siti_beta_up``); передаются
#: статически — кернел не импортирует схему модели.
BETA_GATE_DEFAULT = 4.0
BETA_UP_DEFAULT = 25.0


class MoeBwdKernelError(RuntimeError):
    """Типизированный отказ fused-кернела MoE-backward.

    Возникает, когда (а) конфигурация вне живых случаев сита — форма/тип не
    опускаются на sm_121, либо (б) на узле нет GPU: Mosaic-кернел исполняется
    только на устройстве.  Тихий откат к ``jax`` запрещён — режим выбирается
    гейтом ``AXIOM_MOE_BWD`` явно.
    """


def _smem_floor_bytes(bm, bt, bk, latent, inter, itemsize=2, out_itemsize=4):
    """Объявленный SMEM-пол одной программы (все буферы живут одновременно).

    Буферы: маска строк ``(bm, inter)``, потоковый ``de`` ``(bm, bk)``,
    потоковый ``wd`` ``(inter, bk)``, скретч ``dhg``/``dhu`` ``(bm, inter)``
    (bf16), потоковый вес ``(bt, bk)``, эпилог ``dz`` ``(bm, bt)`` fp32, маска
    весовой фазы ``(bm, bt)`` и два транспонирующих буфера ``(bt, bm)``.
    """
    smem = (
        bm * inter * itemsize          # op_i: маска строк (фаза A)
        + bm * bk * itemsize           # de: потоковый операнд da
        + inter * bk * itemsize        # wd: потоковый вес da
        + 2 * bm * inter * itemsize    # dhg/dhu: скретч bf16
        + bt * bk * itemsize           # w: потоковый wg/wu
        + bm * bt * out_itemsize       # dz_s: эпилог dz (fp32)
        + bm * bt * itemsize           # op_t: маска весовой фазы
        + 2 * bt * bm * itemsize       # x_s/y_s: транспонирование операндов
    )
    return smem + SMEM_EPILOGUE_BYTES


def valid_config(case, params):
    """Статическое сито: формы mma-тайлинга, операнд-путь и SMEM-пол.

    Живые случаи — bf16, ``latent``/``inter`` кратны 128 (построчные записи
    читают из SMEM строку ``inter`` элементов: загрузка фрагментированного
    массива требует кратности 128), ``bm``/``bt``/``bk`` кратны 16, а
    ``rows % bm == 0`` — fail-closed: при некратных ``rows``
    :func:`expert_block_offset` кламплит последний блок к ``rows − bm``, крайние
    блоки перекрываются и весовые фазы (K-поток по строкам) дважды считают
    перекрытые строки.  Кампанийная геометрия (2048/256) кратна ``bm=16``;
    fp32 и oversize-геометрии возвращают False — это честный скип
    ``lower_check``, а не молчаливый откат.
    """
    rows = int(case["rows"])
    latent = int(case["latent"])
    inter = int(case["inter"])
    n_exp = int(case["n_exp"])
    bm = int(params.get("bm", case.get("bm", BM_DEFAULT)))
    bt = int(params.get("bt", case.get("bt", BT_DEFAULT)))
    bk = int(params.get("bk", case.get("bk", BK_DEFAULT)))
    dt = jnp.dtype(case.get("dtype", "bfloat16"))
    if dt != jnp.bfloat16:
        return False
    if n_exp < 1 or rows < bm or rows % bm:
        return False
    if min(bm, bt, bk) < 16 or bm % 16 or bt % 16 or bk % 16:
        return False
    if latent % 128 or inter % 128 or latent % bt or inter % bt:
        return False
    if latent % bk or inter % bk:
        return False
    return _smem_floor_bytes(bm, bt, bk, latent, inter) <= SMEM_LIMIT_BYTES


def cost(rows, latent, inter, n_exp, bm=BM_DEFAULT, bt=BT_DEFAULT,
         bk=BK_DEFAULT, dtype="bfloat16", **kw):
    """Минимальные FLOPs/байты одного запуска (roofline-учёт).

    Веса эксперта читаются по тайлам в каждой из весовых фаз; строковые
    операнды перечитываются ``latent/bt`` (dz) и ``inter/bt`` (веса) раз —
    это осознанная цена тайлинга аккумуляторов в регистрах.
    """
    b = jnp.dtype(dtype).itemsize
    rows_flops = 12 * rows * inter * latent          # da, dwd, dwg, dwu, 2·dz
    weight_tiles = (latent // bt) * (inter // bt)
    weight_bytes = weight_tiles * rows * (2 * latent + 2 * inter) * b
    row_bytes = rows * (2 * (latent + 3 * inter)) * b
    return dict(flops=2 * rows_flops,
                bytes=row_bytes + weight_bytes + 3 * n_exp * latent * inter * 4)


def make_inputs(key, rows, latent, inter, n_exp, dtype="bfloat16", **kw):
    """Входы одного backward-вызова кернела.

    ``z_grouped``/``de_grouped`` — ``(rows, latent)``; ``h_g``/``h_u``/``a`` —
    ``(rows, inter)``; веса — ``(n_exp, latent, inter)`` ×2 и
    ``(n_exp, inter, latent)``; ``group_sizes`` — разбиение ``rows`` по
    экспертам с одной **пустой** группой и неполной (не кратной ``bm``)
    последней группой: границы, которые кернел обязан пережить.
    """
    dt = jnp.dtype(dtype)
    k1, k2, k3, k4, k5 = jax.random.split(key, 5)
    scale = 0.1
    z = (jax.random.normal(k1, (rows, latent), jnp.float32) * scale).astype(dt)
    hg = (jax.random.normal(k2, (rows, inter), jnp.float32) * scale).astype(dt)
    hu = (jax.random.normal(k3, (rows, inter), jnp.float32) * scale).astype(dt)
    a = (jax.random.normal(k4, (rows, inter), jnp.float32) * scale).astype(dt)
    de = (jax.random.normal(k5, (rows, latent), jnp.float32) * scale).astype(dt)
    kk = jax.random.split(k1, 3)
    wg = (jax.random.normal(kk[0], (n_exp, latent, inter), jnp.float32) * 0.02).astype(dt)
    wu = (jax.random.normal(kk[1], (n_exp, latent, inter), jnp.float32) * 0.02).astype(dt)
    wd = (jax.random.normal(kk[2], (n_exp, inter, latent), jnp.float32) * 0.02).astype(dt)
    base = rows // n_exp
    sizes = jnp.full((n_exp,), base, jnp.int32).at[0].add(rows - base * n_exp)
    if n_exp > 1:
        # Пустая группа (граница, которую кернел обязан пережить) — её доля
        # уходит группе 0, сумма остаётся rows.
        sizes = sizes.at[1].set(0).at[0].add(base)
    return z, hg, hu, a, de, wg, wu, wd, sizes


def device_available() -> bool:
    """True, когда JAX видит GPU-устройство (Mosaic требует устройство).

    Lowering устройства не требует (``lower_check`` работает ровно так), —
    поэтому проверка стоит в :func:`kernel`, а не в :func:`_build_kernel`.
    """
    try:
        return any(d.platform == "gpu" for d in jax.devices())
    except Exception:  # noqa: BLE001 — сломанный плагин не должен читаться как GPU
        return False


# ---------------------------------------------------------------------------
# Адресация записей кернела (единственный источник истины)
# ---------------------------------------------------------------------------


def dz_tile_columns(latent, bt):
    """Колонки latent, которые обрабатывает тайл ``tj`` dz-эпилога: ``(c0, width)``.

    Кернел и его CPU-зеркало :func:`mirror_dz_path` берут базу колонок отсюда:
    тайл ``tj`` читает веса ``wg[e, tj·bt : tj·bt+bt, :]`` и пишет результат в
    колонки ``[tj·bt, tj·bt+bt)`` выхода ``dz``.  Дефект цикла-30 — запись
    каждого тайла в колонки ``[0, bt)`` — оставлял ``latent − bt`` колонок
    незаписанными и перезаписывал первую; проба
    ``net/tests/test_54_moe_bwd_fused.py`` проверяет покрытие ``[0, latent)``
    без перекрытий.
    """
    latent, bt = int(latent), int(bt)
    if bt <= 0 or latent % bt:
        raise ValueError(
            f"dz-эпилог: latent={latent} не делится на bt={bt} — тайлинг колонок "
            "не покрывает выход")
    return tuple((tj * bt, bt) for tj in range(latent // bt))


def expert_block_range(start, end, bm):
    """Границы блоков строк эксперта: ``(i0, i1)`` — полуинтервал номеров блоков.

    Блоки ``bm`` строк покрывают ``[start, end)`` с запасом по краям: первый
    начинается не позже ``start``, последний кончается не раньше ``end``;
    принадлежность строки группе решает маска :func:`expert_row_slot`.
    """
    return start // bm, (end + bm - 1) // bm


def expert_block_offset(i, rows, bm):
    """Смещение блока ``i`` в GMEM: ``i·bm``, клампленное к ``rows − bm``.

    Кламп держит чтения блока в границах буфера (последний блок не выходит за
    ``rows``).  При ``rows % bm ≠ 0`` крайние блоки перекрываются: построчные
    записи ``dz``/``dhg``/``dhu`` от этого не страдают (значения строки — те
    же), а весовые фазы с K-потоком по строкам посчитали бы перекрытые строки
    дважды.  Поэтому цикл-32 закрывает это ситом: :func:`valid_config` требует
    ``rows % bm == 0`` (fail-closed, типизированный отказ в :func:`kernel`).
    """
    return jnp.minimum(i * bm, rows - bm)


def expert_row_slot(gr, start, end, rows, e):
    """GMEM-строка записи строки ``gr``: своя строка группы либо dump-слот.

    Строки вне ``[start, end)`` уходят в слот ``rows + e`` (буферы ``dz``/
    ``dhg``/``dhu`` имеют ``rows + n_exp`` строк): программа эксперта пишет
    только свои строки и собственный слот, гонки с соседом нет.  Dump-строки
    не читаются (:func:`kernel` возвращает ``dz[:rows]``).
    """
    return jnp.where((gr >= start) & (gr < end), gr, rows + e)


def masked_operand(values, mask):
    """Операнд строки под маской принадлежности эксперту: вне группы — ровно 0.

    Это **select**, а не умножение на маску: ``0 · NaN = NaN``, а весовая фаза
    читает скретч ``dhg``/``dhu`` блоками строк, где соседние строки
    принадлежат другим экспертам и к моменту чтения могут быть ещё не записаны
    их программами (гонка программ в одном запуске).  ``jnp.where`` выбирает
    ноль в невыбранной строке и не протаскивает nan из невычисленной ветви;
    умножение на маску дало бы ``dwg = nan`` (цикл-32, стенд rows=256).

    ``mask`` — 0/1 в fp32 той же формы, что ``values`` (или broadcast-совместима
    с ней, например ``(bm, 1)``).
    """
    return jnp.where(mask > 0, values, jnp.zeros((), values.dtype))


def empty_group_weight_guard(acc, size):
    """Весовая фаза пустой группы: тайл — ровно нули, аккумулятор не читается.

    ``size == group_sizes[e]`` — динамический размер группы эксперта.  При
    ``size == 0`` (``start == end``) программа эксперта не исполняет ни одной
    строки группы, и её ``dwg[e]``/``dwu[e]``/``dwd[e]`` обязаны быть нулями;
    аккумулятор ``pl.loop`` (``init_carry`` — нули) в этом случае не является
    источником истины: неинициализированный/пустой регистровый carry не должен
    утекать в выход.  ``jnp.where`` выбирает нули, не читая значение carry.
    """
    return jnp.where(size > 0, acc, jnp.zeros((), acc.dtype))


def _store_row(out, idx, row_s, r, c0, width):
    """Записать строку ``r`` блока SMEM в строку ``idx`` GMEM, колонки ``[c0, c0+width)``.

    ``idx`` — динамическая строка записи (:func:`expert_row_slot`); ``c0`` —
    база колонок тайла (:func:`dz_tile_columns` для dz, ``0`` для скретча
    ``dhg``/``dhu``).  Регистровый путь — одна строка, поэтому ``bm`` построчных
    записей вместо скеттера: адрес записи и база колонок — параметры вызова, и
    структурная проба подменяет функцию на lowering, чтобы проверить адресацию,
    исполняемую самим телом кернела.
    """
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import mosaic_gpu as plgpu

    out[idx, pl.ds(c0, width)] = plgpu.load(
        row_s.at[pl.ds(r, 1), pl.ds(0, width)], optimized=False).reshape(width)


# ---------------------------------------------------------------------------
# Mosaic GPU-кернел
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class _KernelSpec:
    """Статические формы, под которые построен кернел (ключ jit-кеша)."""

    rows: int
    latent: int
    inter: int
    n_exp: int
    bm: int
    bt: int
    bk: int
    dtype: str
    beta_gate: float
    beta_up: float


@functools.lru_cache(maxsize=8)
def _build_kernel(rows, latent, inter, n_exp, bm, bt, bk, dtype,
                  beta_gate, beta_up):
    """Построить Pallas/Mosaic-кернел: одна программа на эксперта.

    Фаза A идёт по блокам строк группы (динамические границы ``pl.loop``, маска
    принадлежности эксперту), фазы B/C/D — по тайлам весовых выходов с
    накоплением в регистрах.  Скретч ``dhg``/``dhu`` — GMEM-выходы кернела
    (``plgpu.kernel`` не имеет GMEM-scratch, кроме ``out_type``; та же
    структура, что у персистентного KDA-кернела).
    """
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import mosaic_gpu as plgpu

    dt = jnp.dtype(dtype)
    f32 = jnp.float32
    bg = float(beta_gate)
    bu = float(beta_up)
    # Единственные знаменатели кернела — статические β SiTU-GLU (``hg/βg``,
    # ``hu/βu``); данных-зависимых делений (на счётчик группы/``inter``/``bm``)
    # в кернеле нет.  β < 1 сделал бы знаменатель усилителем сигнала и не
    # является живой геометрией — типизированный отказ до построения.
    if bg < 1.0 or bu < 1.0:
        raise MoeBwdKernelError(
            f"fused MoE-backward: β SiTU-GLU (gate={bg}, up={bu}) должны быть ≥ 1: "
            "это единственные знаменатели кернела; тихого 0/0 нет")

    def _mma_layouts(m):
        mw = 1 if m < 64 else 4
        return (
            plgpu.Layout.MMA_ACC(dt, m_warps=mw),
            plgpu.Layout.MMA_LHS(dt, m_warps=mw),
            plgpu.Layout.MMA_RHS(dt, m_warps=mw),
        )

    ACC_R, L_R, R_R = _mma_layouts(bm)      # (bm, .) — строковые аккумуляторы
    ACC_B, L_B, R_B = _mma_layouts(bt)      # (bt, bt) — весовые тайлы
    ACC_W, _, _ = _mma_layouts(inter)       # (inter, bk) — загрузка wd
    ACC_LT, _, _ = _mma_layouts(latent)     # (latent, bk) — загрузка wg/wu

    def _l(ref, layout):
        return plgpu.load(ref, layout=layout, optimized=False)

    def _zero(shape, layout):
        return plgpu.layout_cast(jnp.zeros(shape, f32), layout)

    def _iota(shape, axis):
        return plgpu.layout_cast(
            plgpu.broadcasted_iota(jnp.int32, shape, axis),
            plgpu.Layout.WG_STRIDED(shape, 2),
        )

    def _body(zg, hgg, hug, ag, deg, wgg, wug, wdg, gsg,
              dwg_o, dwu_o, dwd_o, dz_o, dhg_o, dhu_o,
              op_i_s, de_s, wd_s, dhg_s, dhu_s, w_s, dz_s, op_t_s, x_s, y_s):
        e = jax.lax.axis_index("expert")

        # Начало и размер группы эксперта e: статическая сумма по всем
        # экспертам с маской j < e (динамический срез group_sizes невозможен).
        start = jnp.zeros((), jnp.int32)
        for j in range(n_exp):
            start = start + jnp.where(jnp.int32(j) < e, gsg[j], jnp.int32(0))
        size = gsg[e]
        end = start + size
        i0, i1 = expert_block_range(start, end, bm)

        def _row_block(i):
            """Фаза A: da/dhg/dhu/dz одного блока строк эксперта."""
            off = expert_block_offset(i, rows, bm)
            gi = _iota((bm, inter), 0) + off
            valid = (gi >= start) & (gi < end)
            op_i_s[...] = valid.astype(dt)
            mask_i = _l(op_i_s, ACC_R).astype(f32)          # (bm, inter)
            mask_k = _l(op_i_s.at[pl.ds(0, bm), pl.ds(0, bk)], ACC_R).astype(f32)

            # da = de · wd^T (K-поток по latent, wd читается тайлами).
            da = _zero((bm, inter), ACC_R)
            for kk in range(latent // bk):
                de_v = _l(deg.at[pl.ds(off, bm), pl.ds(kk * bk, bk)], ACC_R).astype(f32)
                # select, не умножение: 0·NaN = NaN (см. :func:`masked_operand`).
                de_s[...] = masked_operand(de_v, mask_k).astype(dt)
                wd_s[...] = _l(wdg.at[e, pl.ds(0, inter), pl.ds(kk * bk, bk)], ACC_W)
                da = plgpu.mma(da, _l(de_s, L_R), _l(wd_s.T, R_R))

            # siti-адъюнкт: те же выражения, что net/mlp.py:siti_glu.
            hg_v = _l(hgg.at[pl.ds(off, bm), pl.ds(0, inter)], ACC_R).astype(f32) * mask_i
            hu_v = _l(hug.at[pl.ds(off, bm), pl.ds(0, inter)], ACC_R).astype(f32) * mask_i
            sig_g = jax.nn.sigmoid(hg_v)
            th_g = jnp.tanh(hg_v / bg)
            th_u = jnp.tanh(hu_v / bu)
            gate = bg * th_g * sig_g
            up = bu * th_u
            dgate = (1.0 - th_g * th_g) * sig_g + bg * th_g * sig_g * (1.0 - sig_g)
            dup = 1.0 - th_u * th_u
            dhg = da * up * dgate
            dhu = da * gate * dup
            dhg_s[...] = dhg.astype(dt)
            dhu_s[...] = dhu.astype(dt)

            # Скретч dhg/dhu для весовых фаз (bf16 — объявленный операнд-путь).
            for r in range(bm):
                gr = off + jnp.int32(r)
                idx = expert_row_slot(gr, start, end, rows, e)
                _store_row(dhg_o, idx, dhg_s, r, 0, inter)
                _store_row(dhu_o, idx, dhu_s, r, 0, inter)

            # dz = dhg·wg^T + dhu·wu^T, тайлы bt по latent: тайл tj читает
            # колонки wg/wu [tj·bt, (tj+1)·bt) и пишет в те же колонки dz
            # (база колонок — из dz_tile_columns, иначе 5/6 выхода теряется).
            for c0, width in dz_tile_columns(latent, bt):
                acc = _zero((bm, width), ACC_R)
                for kk in range(inter // bk):
                    dhg_k = _l(dhg_s.at[pl.ds(0, bm), pl.ds(kk * bk, bk)], L_R)
                    w_s[...] = _l(
                        wgg.at[e, pl.ds(c0, width), pl.ds(kk * bk, bk)], ACC_LT)
                    acc = plgpu.mma(acc, dhg_k, _l(w_s.T, R_R))
                    dhu_k = _l(dhu_s.at[pl.ds(0, bm), pl.ds(kk * bk, bk)], L_R)
                    w_s[...] = _l(
                        wug.at[e, pl.ds(c0, width), pl.ds(kk * bk, bk)], ACC_LT)
                    acc = plgpu.mma(acc, dhu_k, _l(w_s.T, R_R))
                dz_s[...] = acc
                for r in range(bm):
                    gr = off + jnp.int32(r)
                    idx = expert_row_slot(gr, start, end, rows, e)
                    _store_row(dz_o, idx, dz_s, r, c0, width)

        pl.loop(i0, i1)(_row_block)

        def _tile_mask(off):
            """Маска (bm, bt) текущего блока строк для весовых фаз."""
            gi = _iota((bm, bt), 0) + off
            op_t_s[...] = ((gi >= start) & (gi < end)).astype(dt)
            return _l(op_t_s, ACC_R).astype(f32)

        def _weight_tile(ti, tj, lhs_ref, rhs_ref, out_ref):
            """Один тайл ``(bt, bt)`` весового градиента: накопление по строкам.

            ``lhs_ref`` — строковый операнд левой матрицы (``a``/``z``),
            ``rhs_ref`` — правой (``de``/``dhg``); обе читаются с маской
            принадлежности эксперту и транспонируются через SMEM: mma ждёт
            ``k``-непрерывную память операндов.
            """

            def _inner(i, acc):
                off = jnp.minimum(i * bm, rows - bm)
                m = _tile_mask(off)
                lv = _l(lhs_ref.at[pl.ds(off, bm), pl.ds(ti * bt, bt)], ACC_R).astype(f32)
                rv = _l(rhs_ref.at[pl.ds(off, bm), pl.ds(tj * bt, bt)], ACC_R).astype(f32)
                # Строки вне группы выбираются нулём (select), а не умножаются
                # на маску: скретч соседней группы может быть ещё не записан.
                x_s.transpose((1, 0))[...] = masked_operand(lv, m).astype(dt)
                y_s.transpose((1, 0))[...] = masked_operand(rv, m).astype(dt)
                return plgpu.mma(acc, _l(x_s, L_B), _l(y_s.T, R_B))

            acc = pl.loop(i0, i1, init_carry=_zero((bt, bt), ACC_B))(_inner)
            # Пустая группа: выход — нули, carry не читается (цикл-32).
            acc = empty_group_weight_guard(acc, size)
            out_ref[e, pl.ds(ti * bt, bt), pl.ds(tj * bt, bt)] = acc

        def _weight_pass(lhs_ref, rhs_ref, out_ref, nt_m, nt_n):
            """Все тайлы одного весового градиента — динамический ``pl.loop``.

            Тайлы обходятся динамическим циклом (а не раскрываются Python-
            циклом): модуль остаётся компактным, компиляция — секунды, а не
            десятки; индексы выходного среза — динамические ``pl.ds``.
            """

            def _tile(t):
                _weight_tile(t // nt_n, t % nt_n, lhs_ref, rhs_ref, out_ref)

            pl.loop(0, nt_m * nt_n)(_tile)

        # B: dwd = a^T · de — (inter, latent).
        _weight_pass(ag, deg, dwd_o, inter // bt, latent // bt)
        # C: dwg = z^T · dhg — (latent, inter), скретч bf16.
        _weight_pass(zg, dhg_o, dwg_o, latent // bt, inter // bt)
        # D: dwu = z^T · dhu.
        _weight_pass(zg, dhu_o, dwu_o, latent // bt, inter // bt)

    return plgpu.kernel(
        _body,
        out_type=(
            jax.ShapeDtypeStruct((n_exp, latent, inter), jnp.float32),   # dwg
            jax.ShapeDtypeStruct((n_exp, latent, inter), jnp.float32),   # dwu
            jax.ShapeDtypeStruct((n_exp, inter, latent), jnp.float32),   # dwd
            jax.ShapeDtypeStruct((rows + n_exp, latent), jnp.float32),   # dz + dump
            jax.ShapeDtypeStruct((rows + n_exp, inter), dt),             # dhg скретч
            jax.ShapeDtypeStruct((rows + n_exp, inter), dt),             # dhu скретч
        ),
        scratch_types=[
            plgpu.SMEM((bm, inter), dt),     # op_i
            plgpu.SMEM((bm, bk), dt),        # de
            plgpu.SMEM((inter, bk), dt),     # wd
            plgpu.SMEM((bm, inter), dt),     # dhg
            plgpu.SMEM((bm, inter), dt),     # dhu
            plgpu.SMEM((bt, bk), dt),        # wg/wu
            plgpu.SMEM((bm, bt), f32),       # dz эпилог
            plgpu.SMEM((bm, bt), dt),        # op_t
            plgpu.SMEM((bt, bm), dt),        # x_s
            plgpu.SMEM((bt, bm), dt),        # y_s
        ],
        compiler_params=dataclasses.replace(
            plgpu.CompilerParams(), lowering_semantics=plgpu.LoweringSemantics.Lane
        ),
        grid=(n_exp,),
        grid_names=("expert",),
    )


@functools.lru_cache(maxsize=8)
def _kernel_run(rows, latent, inter, n_exp, bm, bt, bk, dtype,
                beta_gate, beta_up):
    """jit-обёртка построенного кернела (кеш по статической спецификации)."""
    return jax.jit(_build_kernel(rows, latent, inter, n_exp, bm, bt, bk,
                                 dtype, beta_gate, beta_up))


def kernel(z_grouped, h_g, h_u, a, de_grouped, wg, wu, wd, group_sizes, *,
           bm=BM_DEFAULT, bt=BT_DEFAULT, bk=BK_DEFAULT,
           beta_gate=BETA_GATE_DEFAULT, beta_up=BETA_UP_DEFAULT):
    """Fused backward экспертного блока: ``(z,hg,hu,a,de,wg,wu,wd,sizes)`` →
    ``(dwg, dwu, dwd, dz_grouped)``.

    Все операнды — bf16 (валидированный mma-путь sm_121), аккумуляторы — fp32;
    ``group_sizes`` — int32, сумма равна ``rows``; пустая группа и неполная
    группа — штатные границы (маска строк + dump-слоты): пустая группа
    получает ровно нулевые весовые градиенты, её строки не участвуют в
    аккумуляции.  ``rows % bm == 0`` — требование сита :func:`valid_config`
    (иначе крайние блоки перекрываются и весовые фазы дважды считают строки;
    отказ — :class:`MoeBwdKernelError`, без тихой ошибки).  Возвращает
    градиенты весов по экспертам (fp32) и ``dz_grouped`` (rows, latent) fp32.

    Исполнение требует Mosaic GPU (GB10); на CPU-узле — типизированный
    :class:`MoeBwdKernelError` (без тихого отката).  *Lowering* устройства не
    требует и проверяется отдельно (``net/tests/moe_experts_bwd_lower.py``).
    """
    rows, latent = z_grouped.shape
    inter = h_g.shape[1]
    n_exp = wg.shape[0]
    case = dict(rows=int(rows), latent=int(latent), inter=int(inter),
                n_exp=int(n_exp), dtype=jnp.dtype(z_grouped.dtype).name)
    params = dict(bm=int(bm), bt=int(bt), bk=int(bk))
    if not valid_config(case, params):
        raise MoeBwdKernelError(
            f"fused MoE-backward: конфигурация {case} (bm={bm}, bt={bt}, bk={bk}) "
            "вне живых случаев сита valid_config (формы mma/SMEM-пол/bf16/"
            "rows % bm == 0); "
            "тихого отката нет")
    if not device_available():
        raise MoeBwdKernelError(
            "fused MoE-backward: Mosaic-кернел исполняется только на GPU "
            "(sm_121); на CPU-узле выберите AXIOM_MOE_BWD=jax. "
            "Тихого отката нет — режим выбирается явно.")
    spec = _KernelSpec(rows, latent, inter, n_exp, int(bm), int(bt), int(bk),
                       jnp.dtype(z_grouped.dtype).name, float(beta_gate),
                       float(beta_up))
    run = _kernel_run(spec.rows, spec.latent, spec.inter, spec.n_exp, spec.bm,
                      spec.bt, spec.bk, spec.dtype, spec.beta_gate, spec.beta_up)
    dwg, dwu, dwd, dz, _dhg, _dhu = run(
        z_grouped, h_g, h_u, a, de_grouped, wg, wu, wd, group_sizes)
    return dwg, dwu, dwd, dz[:rows]


# ---------------------------------------------------------------------------
# jnp-адъюнкт экспертного блока (CPU-паритет; численный пин кернела)
# ---------------------------------------------------------------------------


def _q(x):
    """Квантование операнда в bf16 и обратно в fp32 (объявленный путь mma)."""
    return x.astype(jnp.bfloat16).astype(jnp.float32)


def reference_bwd(z_grouped, h_g, h_u, a, de_grouped, wg, wu, wd, group_sizes,
                  *, beta_gate=BETA_GATE_DEFAULT, beta_up=BETA_UP_DEFAULT):
    """Аналитический адъюнкт экспертного блока — точная семантика кернела.

    Повторяет алгебру :func:`kernel` в jnp: маска принадлежности строки
    эксперту (``eid`` из ``group_sizes``), те же формулы ``siti_glu`` и
    ``sech²``, те же точки квантования (все операнды и скретч ``dhg``/``dhu``
    округляются в bf16 — операнд-путь кернела), аккумулятор fp32.

    Сложность — ``O(n_exp · rows · latent · inter)``: это CPU-пин для малых
    форм паритета (``net/tests/test_54_moe_bwd_fused.py``), не исполнитель
    кампанийной геометрии.  Пустой эксперт (``group_size=0``) даёт нулевые
    градиенты своего среза; порядок групп/строк — входной (кернел его не
    переставляет).
    """
    rows, latent = z_grouped.shape
    inter = h_g.shape[1]
    n_exp = wg.shape[0]
    f32 = jnp.float32
    bg, bu = float(beta_gate), float(beta_up)

    # bf16-операнды (точки округления кернела) в fp32-арифметике.
    zq = _q(z_grouped)
    hgq = _q(h_g)
    huq = _q(h_u)
    aq = _q(a)
    deq = _q(de_grouped)
    wgq = _q(wg)
    wuq = _q(wu)
    wdq = _q(wd)

    gs = group_sizes.astype(jnp.int32)
    starts = jnp.cumsum(gs) - gs
    ridx = jnp.arange(rows, dtype=jnp.int32)
    eid = jnp.sum((ridx[:, None] >= starts[None, :]).astype(jnp.int32), axis=1) - 1

    dwg = jnp.zeros((n_exp, latent, inter), f32)
    dwu = jnp.zeros((n_exp, latent, inter), f32)
    dwd = jnp.zeros((n_exp, inter, latent), f32)
    dz = jnp.zeros((rows, latent), f32)

    for e in range(n_exp):
        m = (eid == e).astype(f32)[:, None]
        ze = zq * m
        hge = hgq * m
        hue = huq * m
        ae = aq * m
        dee = deq * m

        sig_g = jax.nn.sigmoid(hge)
        th_g = jnp.tanh(hge / bg)
        th_u = jnp.tanh(hue / bu)
        gate = bg * th_g * sig_g
        up = bu * th_u
        dgate = (1.0 - th_g * th_g) * sig_g + bg * th_g * sig_g * (1.0 - sig_g)
        dup = 1.0 - th_u * th_u

        da = dee @ wdq[e].T                       # (rows, inter)
        dhg = da * up * dgate
        dhu = da * gate * dup
        dwd = dwd.at[e].set(ae.T @ dee)           # (inter, latent)
        dhgq = _q(dhg)
        dhuq = _q(dhu)
        dwg = dwg.at[e].set(ze.T @ dhgq)          # (latent, inter)
        dwu = dwu.at[e].set(ze.T @ dhuq)
        dz = dz + dhgq @ wgq[e].T + dhuq @ wuq[e].T
    return dwg, dwu, dwd, dz


def reference(z_grouped, h_g, h_u, a, de_grouped, wg, wu, wd, group_sizes,
              **kwargs):
    """Псевдоним :func:`reference_bwd` для контракта скилла (``reference``)."""
    return reference_bwd(z_grouped, h_g, h_u, a, de_grouped, wg, wu, wd,
                         group_sizes, **kwargs)


def mirror_dz_path(h_g, h_u, de_grouped, wd, wg, wu, group_sizes, *,
                   bt=BT_DEFAULT, bk=BK_DEFAULT, tile_columns=None,
                   store_column=None, weights="tiled",
                   beta_gate=BETA_GATE_DEFAULT, beta_up=BETA_UP_DEFAULT):
    """CPU-зеркало dz-пути кернела: da → siti-адъюнкт → тайлы latent → запись строк.

    Повторяет последовательность фазы A кернела на строках своего эксперта:

    * ``da = de·wd^T`` — K-поток тайлами ``bk`` по latent, тайл веса
      ``(inter, bk)`` транспонируется в RHS ``(bk, inter)`` (тот же
      операнд-путь, что ``_l(wd_s.T, R_R)``);
    * siti-адъюнкт ``dhg``/``dhu`` и их квантование в bf16 — скретч кернела
      (``dhg_o``/``dhu_o``);
    * ``dz = dhg·wg^T + dhu·wu^T`` тайлами ``bt`` по latent с K-потоком ``bk``
      по inter: тайл ``(width, bk)`` веса транспонируется в RHS
      ``(bk, width)``;
    * запись: строки группы ``[start, end)`` — в свои строки, строки вне
      группы — в dump-слот ``rows + e`` (в зеркале dump-слот не заполняется:
      он скретч, его адресацию пинует целочисленная проба).

    ``tile_columns`` — политика колонок *операнда* (по умолчанию
    производственная :func:`dz_tile_columns`), ``store_column`` — политика
    колонок *записи* ``(tj, c0, width) → c0 записи``; по умолчанию тайл
    пишется в те же колонки, из которых прочитан.  Исторический дефект
    цикла-30 — ``store_column=lambda tj, c0, width: 0`` (все тайлы в колонку
    0): проба ``test_54`` показывает, что эта политика воспроизводит
    наблюдённую на стенде ошибку, а заведомо верная jnp-версия операнда
    (``weights="full"``) её **не** снимает — значит, дефект был в адресации
    записи, а не в ориентации ``wgᵀ``.

    ``weights="tiled"`` — операнд-путь кернела (тайлы ``bk``),
    ``"full"`` — заведомо верная jnp-версия одним matmul (``dhg·wg^T`` без
    тайлинга K): сверка режимов отделяет ориентацию операндов от адресации.

    Возвращает fp32-буфер ``(rows + n_exp, latent)`` — форму ``out_type``
    кернела; ``dz = out[:rows]``.  Это CPU-пин адресации и формул, а не
    исполнение Mosaic: семантику ``plgpu.mma`` на устройстве проверяет стенд
    (ADR-011).
    """
    rows, inter = h_g.shape
    latent = wg.shape[1]
    n_exp = wg.shape[0]
    gs = [int(v) for v in jnp.asarray(group_sizes).reshape(-1)]
    if len(gs) != n_exp:
        raise ValueError(f"group_sizes: {len(gs)} групп при n_exp={n_exp}")
    if sum(gs) != rows:
        raise ValueError(f"group_sizes: сумма {sum(gs)} ≠ rows={rows}")
    if inter % bk:
        raise ValueError(f"inter={inter} не делится на bk={bk}")
    if weights not in ("tiled", "full"):
        raise ValueError(f"weights: {weights!r} вне ('tiled', 'full')")
    f32 = jnp.float32
    bg, bu = float(beta_gate), float(beta_up)

    tiles = tuple(dz_tile_columns(latent, bt) if tile_columns is None
                  else tile_columns(latent, bt))
    store = (lambda tj, c0, width: c0) if store_column is None else store_column

    hgq, huq = _q(h_g), _q(h_u)
    deq, wdq = _q(de_grouped), _q(wd)
    wgq, wuq = _q(wg), _q(wu)
    out = jnp.zeros((rows + n_exp, latent), f32)

    start = 0
    for e in range(n_exp):
        end = start + gs[e]
        if end > start:
            hg_e = hgq[start:end]
            hu_e = huq[start:end]
            de_e = deq[start:end]
            # da = de·wd^T: K-поток по latent тайлами bk (операнд-путь кернела).
            da = jnp.zeros((end - start, inter), f32)
            for k0 in range(latent // bk):
                kk = k0 * bk
                da = da + de_e[:, kk:kk + bk] @ wdq[e, :, kk:kk + bk].T
            sig_g = jax.nn.sigmoid(hg_e)
            th_g = jnp.tanh(hg_e / bg)
            th_u = jnp.tanh(hu_e / bu)
            gate = bg * th_g * sig_g
            up = bu * th_u
            dgate = (1.0 - th_g * th_g) * sig_g + bg * th_g * sig_g * (1.0 - sig_g)
            dup = 1.0 - th_u * th_u
            dhg = _q(da * up * dgate)          # скретч ядра — bf16-операнд
            dhu = _q(da * gate * dup)
            for tj, (c0, width) in enumerate(tiles):
                if weights == "full":
                    # Заведомо верная jnp-версия: dhg·wg^T без тайлинга K.
                    acc = (dhg @ wgq[e, c0:c0 + width, :].T
                           + dhu @ wuq[e, c0:c0 + width, :].T)
                else:
                    acc = jnp.zeros((end - start, width), f32)
                    for k0 in range(inter // bk):
                        kk = k0 * bk
                        acc = acc + dhg[:, kk:kk + bk] @ \
                            wgq[e, c0:c0 + width, kk:kk + bk].T
                        acc = acc + dhu[:, kk:kk + bk] @ \
                            wuq[e, c0:c0 + width, kk:kk + bk].T
                c0_store = store(tj, c0, width)
                out = out.at[start:end, c0_store:c0_store + width].set(acc)
        start = end
    return out


def mirror_weight_path(z_grouped, h_g, h_u, a, de_grouped, wg, wu, wd,
                       group_sizes, *, bm=BM_DEFAULT, bt=BT_DEFAULT,
                       bk=BK_DEFAULT, beta_gate=BETA_GATE_DEFAULT,
                       beta_up=BETA_UP_DEFAULT, scratch_init="zeros"):
    """CPU-зеркало весовых фаз B/C/D кернела: guard'ы пустой/неполной группы.

    Повторяет структуру ``_weight_pass``/``_weight_tile`` кернела на jnp:

    * блоки строк эксперта — :func:`expert_block_range`/
      :func:`expert_block_offset`; строки вне ``[start, end)`` выбираются нулём
      через :func:`masked_operand` (select, а не умножение: ``0·NaN = NaN``);
    * весовой тайл ``(bt, bt)`` — K-поток по блокам строк с накоплением в fp32,
      как ``pl.loop(i0, i1)`` кернела;
    * выход тайла — :func:`empty_group_weight_guard` от ``group_sizes[e]``:
      пустая группа даёт ровно нули, аккумулятор не читается;
    * скретч ``dhg``/``dhu`` — bf16-буфер ``(rows + n_exp, inter)`` с
      dump-слотами (:func:`expert_row_slot`) — форма ``out_type`` кернела.

    ``scratch_init="nan"`` заполняет скретч nan перед работой: это
    воспроизводит стендовое состояние цикла-32, где строки соседней группы к
    моменту чтения ещё не записаны (GMEM не инициализирован).  С select-маской
    выход остаётся конечным и равным ``"zeros"``-прогону; умножение на маску
    дало бы ``dwg = nan``.

    ``rows % bm != 0`` — типизированный отказ (та же причина, что в
    :func:`valid_config`): кламп последнего блока перекрывает строки, и
    весовые фазы посчитали бы их дважды.  Возвращает
    ``(dwg, dwu, dwd, dhg_scratch, dhu_scratch)`` — формы ``out_type``
    кернела.  Это CPU-пин семантики guard'ов, а не исполнение Mosaic
    (ADR-011).
    """
    rows, inter = h_g.shape
    latent = z_grouped.shape[1]
    n_exp = wg.shape[0]
    gs = [int(v) for v in jnp.asarray(group_sizes).reshape(-1)]
    if len(gs) != n_exp:
        raise ValueError(f"group_sizes: {len(gs)} групп при n_exp={n_exp}")
    if sum(gs) != rows:
        raise ValueError(f"group_sizes: сумма {sum(gs)} ≠ rows={rows}")
    if rows % bm:
        raise ValueError(
            f"rows={rows} не делится на bm={bm}: крайние блоки перекрываются, "
            "весовые фазы дважды считают строки — форма вне сита valid_config")
    if latent % bt or inter % bt or latent % bk or inter % bk:
        raise ValueError(
            f"latent={latent}/inter={inter} не делятся на bt={bt}/bk={bk}")
    if scratch_init not in ("zeros", "nan"):
        raise ValueError(f"scratch_init: {scratch_init!r} вне ('zeros', 'nan')")
    f32 = jnp.float32
    bg, bu = float(beta_gate), float(beta_up)
    if bg < 1.0 or bu < 1.0:
        raise ValueError(f"β SiTU-GLU (gate={bg}, up={bu}) должны быть ≥ 1")

    dt = jnp.dtype(z_grouped.dtype)
    zq, hgq, huq = _q(z_grouped), _q(h_g), _q(h_u)
    aq, deq = _q(a), _q(de_grouped)
    wdq = _q(wd)
    if scratch_init == "nan":
        dhg_s = jnp.full((rows + n_exp, inter), jnp.nan, dt)
        dhu_s = jnp.full((rows + n_exp, inter), jnp.nan, dt)
    else:
        dhg_s = jnp.zeros((rows + n_exp, inter), dt)
        dhu_s = jnp.zeros((rows + n_exp, inter), dt)
    dwg = jnp.zeros((n_exp, latent, inter), f32)
    dwu = jnp.zeros((n_exp, latent, inter), f32)
    dwd = jnp.zeros((n_exp, inter, latent), f32)

    def _row_mask(off, start, end):
        gi = off + jnp.arange(bm, dtype=jnp.int32)
        return ((gi >= start) & (gi < end)).astype(f32)[:, None]

    start = 0
    for e in range(n_exp):
        size = gs[e]
        end = start + size
        i0, i1 = expert_block_range(start, end, bm)

        # Фаза A: dhg/dhu блоками; строки вне группы — в dump-слот эксперта.
        for i in range(int(i0), int(i1)):
            off = int(expert_block_offset(jnp.int32(i), rows, bm))
            m = _row_mask(off, start, end)
            de_b = masked_operand(deq[off:off + bm], m)
            da = jnp.zeros((bm, inter), f32)
            for k0 in range(latent // bk):
                kk = k0 * bk
                da = da + de_b[:, kk:kk + bk] @ wdq[e, :, kk:kk + bk].T
            hg_b = masked_operand(hgq[off:off + bm], m)
            hu_b = masked_operand(huq[off:off + bm], m)
            sig_g = jax.nn.sigmoid(hg_b)
            th_g = jnp.tanh(hg_b / bg)
            th_u = jnp.tanh(hu_b / bu)
            gate = bg * th_g * sig_g
            up = bu * th_u
            dgate = (1.0 - th_g * th_g) * sig_g + bg * th_g * sig_g * (1.0 - sig_g)
            dup = 1.0 - th_u * th_u
            dhg = _q(da * up * dgate)          # скретч ядра — bf16-операнд
            dhu = _q(da * gate * dup)
            idx = jnp.stack([
                expert_row_slot(off + r, start, end, rows, e) for r in range(bm)])
            dhg_s = dhg_s.at[idx].set(dhg.astype(dt))
            dhu_s = dhu_s.at[idx].set(dhu.astype(dt))

        def _weight_tiles(lhs, rhs, nt_m, nt_n):
            """Все тайлы одного весового градиента: K = блоки строк группы."""
            out = jnp.zeros((nt_m * bt, nt_n * bt), f32)
            for ti in range(nt_m):
                for tj in range(nt_n):
                    acc = jnp.zeros((bt, bt), f32)
                    for i in range(int(i0), int(i1)):
                        off = int(expert_block_offset(jnp.int32(i), rows, bm))
                        m = _row_mask(off, start, end)
                        lv = masked_operand(
                            lhs[off:off + bm, ti * bt:(ti + 1) * bt].astype(f32), m)
                        rv = masked_operand(
                            rhs[off:off + bm, tj * bt:(tj + 1) * bt].astype(f32), m)
                        acc = acc + lv.T @ rv
                    acc = empty_group_weight_guard(acc, jnp.int32(size))
                    out = out.at[ti * bt:(ti + 1) * bt,
                                 tj * bt:(tj + 1) * bt].set(acc)
            return out

        # B: dwd = a^T·de; C: dwg = z^T·dhg; D: dwu = z^T·dhu.
        dwd = dwd.at[e].set(_weight_tiles(aq, deq, inter // bt, latent // bt))
        dwg = dwg.at[e].set(_weight_tiles(zq, dhg_s, latent // bt, inter // bt))
        dwu = dwu.at[e].set(_weight_tiles(zq, dhu_s, latent // bt, inter // bt))
        start = end
    return dwg, dwu, dwd, dhg_s, dhu_s
