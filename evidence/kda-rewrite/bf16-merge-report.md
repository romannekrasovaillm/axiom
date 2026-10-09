# Перенос compute-dtype гейта (556cf08) в `arch/kda-rewrite` — отчёт о переносе

**Дата:** 2026-10-09
**Ветка:** `arch/kda-rewrite`, якорь откта — `edb83a0`
**Источник:** `arch/bf16-phase1-wip`, коммит `556cf08` («WIP bf16-phase1: salvage …»)
**Метод:** `git cherry-pick -n 556cf08` + ручное разрешение конфликта в `net/kda.py`

---

## 1. Что перенесено

Состав сверен с `git show --stat 556cf08`; для всех файлов ниже содержимое
**побайтово** совпадает с коммитом-источником (`git diff 556cf08 -- <файл>`
пуст), кроме `net/model.py` и `net/kda.py` — там перенос лёг поверх уже
принятых в ветке улучшений (см. §3).

| Файл | Роль |
| --- | --- |
| `net/compute_dtype.py` | гейт `AXIOM_COMPUTE_DTYPE` (`fp32` дефолт / `bf16` Beam-рецепт), fail-closed |
| `net/kda.py` | dtype-прокладка в проекциях и выходном гейте (+ разрешение конфликта) |
| `net/mla.py` | прокладка во всех GEMM/контракциях + перевод `_FLASH_DENSE` на `_flash_dense()` и собственно исправление layout flash-ветки (`(B,T,H,dq)` вместо ошибочного `(B,H,T,dq)`) |
| `net/attn_sparse.py` | `gemm_einsum` во всех score/out-контракциях |
| `net/attnres.py` | `gemm_einsum` в обеих ветках AttnRes |
| `net/mlp.py` | `gemm` в трёх проекциях SiTU-GLU |
| `net/moe.py` | `gemm`/`gemm_einsum` + bf16-спелл экспертного применения через `gemm_batched` |
| `net/mtp.py` | `gemm` в `W_f` |
| `net/model.py` | `gemm` в голове/MTP-логитах (+ сохранён `remat_policy`, см. §3) |
| `net/tests/test_29_compute_dtype.py` | приёмка (а)/(б)/(в): бит-точность gate-off против `773e389`, fp32-инварианты под bf16, «только bf16 и только на границах» |
| `net/tests/test_30_mla_flash_parity.py` | повторный замер `AXIOM_MLA_DENSE_FLASH` на текущем чекауте + мутационный тест на подмену оси |
| `tools/loss_parity_bf16.py`, `tools/mfu_bf16_protocol.py` | приборы loss-parity и MFU-протокола |
| `evidence/mfu-bf16/*.json`, `evidence/gemm_peak_0810.err` | артефакты WIP-прогона из `556cf08` (перенесены **как есть**, здесь не перемерялись) |

`evidence/facts/*` и `.arch-fleet*` не затронуты (как и в самом `556cf08`).

## 2. Как разрешён конфликт в `net/kda.py`

Конфликт был **один** — блок импортов (единственная содержательная коллизия
двух правок в одной строке):

```
<<<<<<< HEAD
from .remat import DEFAULT_REMAT_POLICY, remat_checkpoint
from . import attn_sparse, quant
=======
from . import attn_sparse, compute_dtype, quant
>>>>>>> 556cf08
```

Разрешено **объединением** (обе стороны несут нужное, ничего не выбирается):

```python
from .remat import DEFAULT_REMAT_POLICY, remat_checkpoint
from . import attn_sparse, compute_dtype, quant
```

Остальные ханки `556cf08` в `net/kda.py` (проекции `_project`, выходной гейт
`_output_gate`, оконная ветка `_window_projection`, комментарий про
алгебру состояния в `chunk_step`) легли автоматически и сохранены.

### Почему форма `chunked_cc` оказалась под гейтом

`556cf08` не гейтил ни один из *внутричанковых* einsum'ов KDA — гейту подлежат
только параметрические контракции, и это зафиксировано комментарием самого
коммита: «State algebra, deliberately NOT gated … the recurrent state is an
accumulator, and the recipe keeps accumulators fp32». Тот же принцип применялся
и к существовавшему уже тогда `wyut_chunk_step`.

`chunked_cc` (ADR-047) — не отдельная реализация арифметики, а переписывание
*той же* рекуррентности WY/UT: `cc_chunk_step` вызывает **те же** гейтированные
`_project` и `_output_gate` (и `_with_window` → `_window_projection`), поэтому
прокладка применяется внутри формы без второго её экземпляра. Внутричанковая
алгебра (`_cc_scores`, `t_mat`/`w`/`u`/`v_tilde`, перенос состояния) остаётся
fp32 — то же решение, что и для аналогичных einsum'ов `wyut`, иначе два плеча
ADR-047 стали бы несравнимы в кампании. Это задокументировано в docstring
`cc_chunk_step` (комментарий добавлен при переносе).

**Важно:** гейт-прокладка — единственное, что пришло из `556cf08` в `kda.py`.
Ни одна строка ADR-047 (`apply_chunked_cc`, `cc_chunk_step`, `_cc_scores`,
`_cc_tile`, диспетчер `apply_kda`) не откачена; `_remat_policy` и вызовы
`remat_checkpoint` в `apply_chunked`, `apply_wyut`, `apply_chunked_cc` на месте.

## 3. Что сохранено из уже принятых улучшений

Проверено диффом (`git diff 556cf08 -- <файл>` показывает только добавления
ветки, ни одного удаления):

* **`net/kda.py`** — форма `chunked_cc` целиком, `apply_kda` с тремя реализациями,
  `_remat_policy`, `remat_checkpoint` в трёх scan-телах; `body = jax.checkpoint(body)`
  к версии `556cf08` **не** откатился.
* **`net/model.py`** — `remat_policy` (ADR-049) сохранён во всех четырёх точках
  (`forward`, `_forward_group_scan` ×2, `compute_loss`) вместе с
  `validate_remat_policy` (fail-closed) и `remat_checkpoint`; слияние аддитивное:
  `compute_dtype.gemm` добавлен рядом, `remat_policy` не тронут.
* **`net/optimizer.py`, `net/remat.py`, `tools/pretrain_run.py`** — не затронуты
  cherry-pick'ом вовсе (в `git status` не появлялись); флаги `--phase-profile`,
  `--remat-policy` и «первый шаг вне KPI» на месте (`first_step_excluded`).
* **Дефолты не менялись**: `kda_impl="chunked"`, `remat_policy="none"`,
  `ns_steps`, `AXIOM_COMPUTE_DTYPE` (не задан → `fp32`) — как в `net/config.json`
  и `net/config.py` до переноса.

## 4. Семантика гейта

* **Дефолт — `fp32`, байт-точен по построению.** `compute_dtype.gemm` в режиме
  `fp32` возвращает `a @ b`, `gemm_einsum` — `jnp.einsum(...)`, `cast_in`/`cast_out` —
  тождества. Это буквально выражение вызывающего, поэтому gate-off граф
  совпадает с догейтовым, а не «близок» к нему.
* **`bf16`** включается только явным `AXIOM_COMPUTE_DTYPE=bf16`: операнды GEMM
  кастуются в bf16, произведение накапливается и возвращается в fp32
  (`preferred_element_type=jnp.float32`); мастер-копия весов, residual-поток,
  градиенты и слоты оптимизатора остаются fp32.
* **Fail-closed**: неизвестное значение → `ComputeDtypeError`, без молчаливого
  отката в fp32.
* Значение читается **на каждом вызове** (`mode()`), режим разрешается в момент
  трассировки; менять переменную под уже скомпилированным `jit` нельзя — это
  прямо задокументировано в `net/compute_dtype.py`.

## 5. Тесты

Окружение прогона (CPU): `XLA_PYTHON_CLIENT_MEM_FRACTION=0.5`,
`NET_JAX_BACKEND=cpu`, `NET_GATE_PROFILE=0`; Python 3.11.8, jax 0.10.2,
`jax.devices() == [CpuDevice(id=0)]`, `jax_default_matmul_precision=highest`
(пин ADR-010 из `net/tests/conftest.py`).

| Файл | Результат |
| --- | --- |
| `net/tests/test_31_remat_policy.py`, `net/tests/test_kda_chunked_cc.py`, `net/tests/test_kda_wyut.py`, `net/tests/test_02_kda_parity.py`, `net/tests/test_30_mla_flash_parity.py`, `tools/tests/test_remat_policy_cli.py`, `tools/tests/test_jax_preflight.py` (один прогон) | **160 passed, 1 skipped** (1039 с) |
| `net/tests/test_29_optimizer_groups.py` | **20 passed** (203 с) |
| `net/tests/test_29_compute_dtype.py` (новый из `556cf08`) | **23 passed, 1 skipped** (110 с) |
| `net/tests/test_kda_dtype_gate_parity.py` (**новый, добавлен при переносе**) | **8 passed** (24 с) |
| **Итого** | **211 passed, 2 skipped, 0 failed** |

Скипы — заявленные, не отказы:

* `test_02_kda_parity.py` — `pytest.importorskip("fla")` / объявленный скип
  «FLA реализует DeltaNet, не KDA» (внешний PyTorch-оракул не установлен);
* `test_29_compute_dtype.py` — числовое плечо bf16 на XLA:CPU неисполнимо
  («Unsupported element type for DotThunk::Execute: BF16 x BF16 = F32»);
  тест сам документирует, что этот замер — работа стенда GB10.

### Новый тест `test_kda_dtype_gate_parity.py` — зачем он

Требование «дефолт (fp32) побитово равен текущему поведению» в контексте
именно этого мержа проверялось отдельно. `test_29_compute_dtype.py` сравнивает
gate-off с `773e389` — коммитом, в котором **нет** форм `wyut`/`chunked_cc`
и нет ADR-049-границ, то есть скрытый дрейф в новой форме он увидеть не может.
Новый тест извлекает `net/` из **догейтового дерева этой ветки** (`edb83a0`,
`git archive` тем же идиомом, что и `test_29`) и сравнивает `apply_kda`
**побитово** (сырые fp32-паттерны, без допуска) для всех трёх `kda_impl`
(`chunked`, `wyut`, `chunked_cc`) × `kda_chunked_backward` ∈ {off, on}, плюс
отдельный случай оконной ветки с собственными проекциями
(`swa_share_kda_projections=False`) и проверку fail-closed на уровне слоя
(`AXIOM_COMPUTE_DTYPE=bogus` в `apply_kda` → `ComputeDtypeError`).

## 6. Границы

* Замер bf16-vs-fp32 на l3-full — **за архитектором** на GB10 (два прогона по
  20 шагов с `--phase-profile`); приборы перенесены, но здесь не запускались
  (нет GPU, нет сетевых вызовов).
* Включение bf16 в прод по-прежнему требует parity-вердикта (ADR-040).
* Артефакты `evidence/mfu-bf16/*` — из WIP-прогона `556cf08`, на этой ветке не
  перемерялись и не должны читаться как измерение текущего дерева.

## 7. Находка для архитектора (вне объёма переноса)

Перенесённые приборы `tools/mfu_bf16_protocol.py` и `tools/loss_parity_bf16.py`
**не несут префлайта ADR-041**: они не импортируют `tools/jax_preflight.py`, не
вызывают `ensure_mem_fraction()`/`gate_or_exit()` и не входят в
`_TOOLS_WITH_PREFLIGHT` (`tools/tests/test_jax_preflight.py`), так что регрессия
их не поймает. Каждую клетку они запускают в отдельном процессе, но
`XLA_PYTHON_CLIENT_MEM_FRACTION` наследуют из окружения вызывающего
(`env = dict(os.environ)`), а не выставляют сами.

Это ровно тот класс, из-за которого случился OOM-каскад 08.10 (ADR-041):
прибор запускается на **совмещённом** стенде GB10, где JAX по умолчанию
резервирует ~75 % памяти. Здесь приборы перенесены **как есть** из `556cf08`
(менять их — выходить за объём переноса и расходиться с источником); вопрос
подключения к префлайту вынесен архитектору.
