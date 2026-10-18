# Цикл-19 кампании MFU-55 — провенанс-карта хвоста (метки `AXIOM_TRACE_MARKERS`)

- Дата: 18.10.2026, стенд GB10, JAX 0.11.2, nsys 2025.3.2 (node-level).
- Основание: ADR-054 Am.31 (цикл-19 — провенанс-карта до выбора кернела).
- Код: ветка `arch/theseus-20261018113459` (`6efb816`, харнесс theseus, изолированный
  worktree, в main НЕ влита — owner-gate); прогон из worktree `/home/roman/axiom-c19`.
- Клетка: `l3-full` B=4 persist-on, `AXIOM_TRACE_MARKERS=1`, свежий кэш, HLO-дамп +
  nsys node-level, 2 шага (стационар 839.3 ток/с, шаг 39.04 с, nan 0).

## 1. Атрибуция стационарного шага (окно 1511–1551 с трейса)

GPU всего 39.04 с, 84 991 ядро. По `probe_*` из op_name:

| компонент | GPU | ядер | доля |
|---|---:|---:|---:|
| `probe_—` (вне размеченных границ) | 17.03 с | 23 761 | 43.6% |
| `probe_ce` (внешний loss-scope) | 14.62 с | 56 974 | 37.5% |
| `probe_moe` | 4.34 с | 2 415 | 11.1% |
| `probe_kda` | 1.75 с | 1 409 | 4.5% |
| `probe_mla` | 1.14 с | 382 | 2.9% |
| `probe_mlp` | 0.12 с | 16 | 0.3% |
| `probe_attnres` | 0.04 с | 34 | 0.1% |

Топ-спаны: `command_buffer_1226` — 7.11 с (98% `ce`), `_1070` — 3.10 с, `_985` —
2.64 с (90% `ce`), затем мелкие буферы. Неотнесённое — персистентные кернелы
(`_body_mosaic` 3.69 с, `batched_body_mosaic` 1.39 с) и оптимизатор
(`magma_sgemmEx` 2.08 с, `cutlass simt` 1.96 с) — их имена не из `jit_loss_fn`.

## 2. Главная находка: хвост — MTP KDA на jnp-пути `chunked`

Ядра хвоста (`command_buffer_1226/985`, 9.75 с GPU, 512 графов) несут op_name
`.../checkpoint/chuv,chu->chv/dot_general` и `.../reduce_sum`. Строка
`chuv,chu->chv` — это `net/kda.py:361`, функция **`chunk_step`** пути
**`apply_chunked`** (kda_impl="chunked"), и НЕ `cc_chunk_step` персистентного
пути. Единственный вызов `apply_chunked` в модели — **`net/mtp.py:64`**:

```python
attn = jax.vmap(lambda xb: kda_mod.apply_chunked(params.kda, cfg, xb, chunk_size))(z)
```

То есть **MTP-слой** (`mtp_layers=1`, вес 0.1) считает KDA по старому jnp-пути
(256 чанков C=32 × 1 слой, forward + backward/remat), в то время как основные
24 слоя идут через персистентный `chunked_cc` (`AXIOM_KDA_CC_IMPL=persistent`).
Это и есть 256-графовый хвост; на него не распространяется ни variant B, ни
KDA-персистентность — MTP просто не был переведён на кампанийную реализацию.

**Проверка**: `kda_impl=chunked_cc`, `AXIOM_KDA_CC_IMPL=persistent`; CPU-проба
маршрута (Am.31) — `cc_chunk_step` 0 трассировок; в HLO есть персистентные
KDA/MLA custom-call'ы (6 котангентов + chain), но `chuv,chu->chv` остаётся —
ровно у MTP-вызова.

## 3. Что это даёт для G1

G1 = 33.53 с/шаг. Хвост MTP — 9.75 с GPU (25% шага). Перевод MTP KDA на
персистентный `chunked_cc` (та же математика, другой исполнитель; паритет
`chunked_cc` vs `chunked` уже пиннут кампанией) ожидаемо снимает основную часть
хвоста: **−7…−9 с ⇒ 30–32 с/шаг ⇒ 1 020–1 090 ток/с (G1 взят)**.
Цена — одна строка маршрутизации в `net/mtp.py` + тесты; риск — паритет
MTP-выхода (вес 0.1) и CPU-ветка (персистентный кернел на CPU отказывает —
тесты должны выбирать `AXIOM_KDA_CC_IMPL=jax` или подменять швы).

## 4. Ограничения провенанс-карты

- `probe_ce` стоит на **всём `compute_loss`** (внешний loss-scope): через
  `transpose(jvp(probe_ce))` он захватывает и **backward**, поэтому 37.5% —
  это «loss-scope минус внутренние forward-метки», а не только CE-голова.
  Внутренние `probe_kda/mla/moe` видны в forward (4.5/2.9/11.1%), в backward —
  нет. Для точной backward-атрибуции нужны метки в backward-правилах
  (открытый вопрос харнесса).
- MTP-компонент **не размечен** (метки в MTP не ставились) — вывод §2 опирается
  на op_name/einsum и кратность 256, а не на `probe_mtp`; подтверждение —
  следующим прогоном с меткой `probe_mtp` вокруг `mtp.apply`.
- Неотнесённые 43.6% — оптимизатор (`jit_step`: magma/cutlass fp32 = Muon NS) и
  персистентные кернелы; их привязка к компонентам требует opmap по всем модулям
  дампа (сейчас карта строится по `jit_loss_fn`).

## 5. Артефакты

- Прогон: `evidence/mfu-55/cycle19-provenance/` (metrics, journal, summary,
  `prov-map.txt`, `cb1226.txt`, `cb985.txt`); трейс и дамп — на стенде
  (`prov.sqlite`, `~/axiom-run/xladump-c19`, ~1.5 ГБ).
- Код меток: ветка `arch/theseus-20261018113459` (`net/provenance.py`,
  `net/model.py`, `net/train_loop.py`, `net/tests/test_45_trace_markers.py`;
  25 тестов + регресс 15 passed/1 skip — проверено архитектором).
- Инструменты: `tools/c19-provenance.sh`, `tools/c19_prov_map.py`,
  `tools/c18_cmdbuf_kernels.py`.
