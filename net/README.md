# net/ — Kimi-Killer L3 walking skeleton (network from scratch, JAX)

Реализация сети с нуля по `docs/specs/MODEL-L3-SKELETON.md`.  Чистый JAX
(без Flax/NNX в ядре); MaxText-интеграция документирована в `maxtext/`.
Структура:

| Файл | Что |
|---|---|
| `config.json` | запиненный конфиг + фактический счётчик параметров + `deviations` |
| `config.py` | `ModelConfig`, загрузка/валидация |
| `kda-formulas.md` | формулы из `k3_tech_report.pdf` §2.1.1/§2.2/§2.3 (§2.5) со ссылками на строки извлечения |
| `kda.py` | Kimi Delta Attention: рекуррентная `lax.scan` + чанкированная (ассоциативный скан) формы; дельта v1.5: окно SWA на общих q/k/v (0 матричных параметров) со своей обучаемой долей смешивания |
| `mla.py` | Gated MLA (NoPE, латент KV, полноразмерный gate) + reference-оракул; дельта v1.5: SWA-ветка со своими K/V, sparse-выбор (indexer из латента, top-k), FP4 латента за `qat_kv_enabled` |
| `attn_sparse.py` | общие примитивы дельты: оконное внимание (D1), sparse-выбор ∪ окно с блочным сканом по запросам (D2), память O(top_k + n_win) на запрос, иерархический пул кандидатов (ADR-012: blockwise max → top-m блоков → выбор внутри пула) |
| `mlp.py` | SiTU-GLU MLP (Eq. 12) — dense-слой 0 и MTP-блок |
| `moe.py` | Stable LatentMoE (Eq. 11–14): 12 routed + 2 shared, top-2, latent-проекция, QB-балансировка |
| `attnres.py` | Attention Residuals (full form, аддитивная интеграция) |
| `mtp.py` | multi-token prediction слой |
| `vit.py` | ViT-S минимум (~22M, patch 14, проекция в hidden) |
| `norm.py` / `shortconv.py` | RMSNorm/L2Norm/Swish и каузальный depthwise ShortConv |
| `model.py` | сборка 24 слоёв (18 KDA + 6 MLA; 1 dense + 23 LatentMoE), forward, loss (NTP+MTP+QB), счётчики параметров total/active |
| `optimizer.py` | per-head Muon + weight clipping, cosine + 1% warmup, wd 0.1 |
| `quant.py` | QAT fake-quant: MXFP4 веса (block 32) и MXFP4 латента KV (E2M1 + E4M3 scale/16 каналов, без global scale, ADR-009 D3), FP8 E4M3 для SWA KV |
| `tokenizer.py` | byte-level BPE 160K, seed-пиннинг, хеш |
| `data.py` / `curriculum.py` | дата-пайплайн (4 домена + vision-min), карточка+хеш, куррикулум 8K→64K |
| `checkpoint.py` | Orbax save/load + хеш-манифест |
| `infer.py` | детерминированная инференс-петля `generate` (NTP-голова, greedy/temperature+top_p, seed-пиннинг; MTP-голова не используется) |
| `gpu_smoke.py` | раннер GPU-смоука §6.10: замеры ток/с (tiny/полный), отчёт `gpu-smoke-report.json` |
| `tests/` | 12 критериев приёмки §6 спеки + test_13 — end-to-end стыковка с env; критерии дельты v1.5: test_13_sparse_dense_equivalence (13), test_14_kv_fp4 (15), test_15_cost_8k_64k (16), test_16_indexer_determinism (17) |

Дельта v1.5 (ADR-009, gated A4) включается флагом `attn_dense_reference=false`
(по умолчанию `true` — плотный оракул D4, поведение тестов 01–12 не меняется).
Механики декларируются полями `net/config.json` (`swa_window`, `mla_top_k`,
`swa_share_kda_projections`, `qat_kv_enabled`, `indexer_seed`) и строками
`deviations` (spine AD-9 / C-035), а не правкой кода по месту.

Иерархический разреженный индексатор (ADR-012, импорт §2.3.2 источника)
декларируется там же: `mla_pool_block` (размер блока пула), `mla_pool_size`
(m — сколько блоков в пуле; `0` выключает пул) и `mla_layer_modes` (раскладка
режимов `full|reindex|reuse` по MLA-слоям).  Первый MLA-слой в режиме `full`
строит **общий пул кандидатов** (top-m блоков по максимуму скора в блоке) и
публикует его; остальные слои выбирают свои `top_k` внутри пула: `reindex` —
своей переоценкой скоров пула, `reuse` — готовым выбором строителя без
скоринга.  Слой-потребитель без опубликованного пула откатывается в `full`
(точный отбор по всему префиксу).  Критерий 13 сохраняется: при
`mla_top_k >= T` путь замкнут на плотное причинное внимание независимо от
раскладки, а при пуле, покрывающем префикс (`m * mla_pool_block >= T`),
`reindex` эквивалентен отбору по всему префиксу (тесты в
`test_13_sparse_dense_equivalence.py`); «пул строится ровно один раз за
forward» закреплено в `test_17_indexer_shortcut.py`.

Инвариант конфига: `num_heads * head_dim == hidden` и
`kda_dk == kda_dv == mla_head_dim == head_dim` (проверяется `validate_config`).

Данные/чекпойнты — только симлинки на `~/gb10-shared` (C-032/C-033);
веса инициализируются с нуля (REQ-001).  Запуск тестов:

```bash
python -m pytest net/tests/ -q
```

## Бэкенд и политика точности (ADR-010)

Вердикт сьюта не должен зависеть от оболочки, поэтому оба параметра пиннуются
в `net/tests/conftest.py` — в одном месте, а не в команде запуска:

* **политика точности** — `jax_default_matmul_precision="highest"`. Критерий 2
  (паритет форм KDA) на GPU при политике по умолчанию даёт 3.37e-04 против
  порога 1e-4; под `highest` — 1.2e-07 на CPU и GPU. Оракул обязан мерить
  алгоритм, а не политику точности железа; порог 1e-4 не меняется;
* **бэкенд** — выбирается явно: `nvidia-*` библиотеки венва предзагружаются
  до `import jax`, потому что с системной CUDA в `LD_LIBRARY_PATH` плагин
  подхватывает libcusparse, несовместимый по nvJitLink 12.9, и молча падает на
  CPU. Экспорт `LD_LIBRARY_PATH` вручную больше не нужен (и не должен
  переопределять venv).

Заголовок прогона печатает политику, `jax.devices()`, профиль
(`NET_GATE_PROFILE`) и режим детерминизма (ADR-013). Профили различают
требование GPU:

| Профиль | Как включить | Нет CUDA-устройства |
|---|---|---|
| gate (A4) | `NET_GATE_PROFILE=1` или `pytest --net-gate` | `test_10_gpu_smoke` — **FAIL** с внятным сообщением |
| local (предварительный) | по умолчанию | `test_10_gpu_smoke` — **SKIP** с явной причиной |

Переменные выбора: `NET_JAX_BACKEND=auto` (по умолчанию) \| `gpu` \| `cpu` —
`cpu` ставит `JAX_PLATFORMS=cpu`, чтобы явный выбор не переопределялся
окружением.

```bash
export LD_LIBRARY_PATH=$(ls -d ~/venv-axiom/lib/python3.11/site-packages/nvidia/*/lib | tr '\n' ':')
python -m pytest net/tests/test_10_gpu_smoke.py -q -s   # замер ток/с на GPU
python -m net.gpu_smoke                                  # отчёт net/gpu-smoke-report.json
python -m pytest net/tests/ -q                      # local: без GPU — SKIP, не PASS
NET_JAX_BACKEND=cpu python -m pytest net/tests/ -q   # предварительный прогон на CPU
NET_GATE_PROFILE=1 python -m pytest net/tests/ -q    # гейтовый профиль A4: GPU обязателен
python3 tools/check_precision_pinning.py --profile gate   # страж пиннинга (C-042)
python -m net.gpu_smoke                              # отчёт net/gpu-smoke-report.json
```

**Детерминизм (ADR-013).** В гейтовом профиле `conftest` добавляет
`--xla_gpu_deterministic_ops` в `XLA_FLAGS` — до `import jax`, потому что XLA
разбирает переменную один раз, при создании клиента. Без флага XLA не
гарантирует побитовое совпадение GPU-ядер, и A5 («повторный прогон → тот же
манифест», `workspace_sha256` считается по содержимому артефактов) держится
случайно. В локальном профиле флаг не ставится: скорость итераций важнее, а
заголовок честно печатает режим, в котором получен результат.

Страж `tools/check_precision_pinning.py` (правило C-042) проверяет не текст
conftest, а runtime-конфигурацию: применяет пиннинг в отдельном процессе и
читает обратно `jax_default_matmul_precision`, `jax.devices()` и наличие
`--xla_gpu_deterministic_ops` в `XLA_FLAGS` процесса (в гейтовом профиле его
отсутствие — FAIL). Режим `--measure-criterion2` подтверждает, что под
пиннингом критерий 2 укладывается в порог (без CUDA-устройства — SKIP с явной
причиной, не PASS).
