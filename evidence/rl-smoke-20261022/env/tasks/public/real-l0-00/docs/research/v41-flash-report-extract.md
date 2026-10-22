# Извлечение первоисточника: DeepSeek V4.1-Flash (2026-09-10)

**Назначение.** Доказательная база ADR-009. Все утверждения ниже проверены по PDF, а не по дайджесту и не по производным скиллам.

- **Файл:** `~/Документы/КОД/gigachat/РАЗБОРЫ/recipes_taxonomy/11_Техотчёты_лабораторий_LLM/2026/DeepSeek/2026-09-10_DeepSeek-V4.1-Flash-KV-Cache-Compression.pdf`
- **Проверка файла:** `pdfinfo` → 51 стр., A4, pdfTeX-1.40.27, CreationDate 2026-09-10 08:49 MSK, не шифрован. Лицензия MIT, чекпойнты: `huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash`.
- **Воспроизводимое извлечение:** `pdftotext -layout "<pdf>" /tmp/dsv41.txt` → 2881 строка. Ссылки ниже — номера строк этого извлечения (команда и файл воспроизводимы побитово у любого, у кого есть PDF).
- **Соседи по каталогу (для истории линии):** `2026-04-26_DeepSeek-V4-Million-Token-Context.pdf`, `2026-07-31_DeepSeek-V4-Flash-0731-Model-Card.pdf`, `2026-08-13_DeepSeek-V4-Pro-0813-Model-Card.docx`, `2026-01-12_Engram-Conditional-Memory-Scalable-Lookup.pdf`.

## 1. Общая форма (§2.1, :311–343)

| Параметр | Значение | Строка |
|---|---|---|
| Backbone | 40 причинных слоёв = 20 encoder + 20 decoder (CED) | :314–316 |
| Параметры | 552B backbone + 196B Engram; активация **8B prefill / 16B decode** | :331 |
| Внимание в слое | **глобальное + SWA в каждом слое**; первые два слоя — только SWA | :316–318 |
| SWA-окно | `n_win = 128` | §2.2, :394 |
| Мультимодальность | ViT + MLP-проектор, 3×3 pixel-unshuffle (÷9 токенов), до ~1344×1344, 2D-RoPE, patch-embedding → линейная проекция (совместимость с Muon) | :344–360 |
| MoE | DeepSeekMoE, shared + fine-grained routed; **modality-specific load balancing** (раздельные correction biases для text/image), aux-loss-free | :326–330, :363–375 |
| Претрейн | 45T токенов, мультимодальный корпус | §4 |
| MTP | **Исключён** из претрейна backbone; вместо него DSpark отдельной стадией | :333–336 |

## 2. CED — Causal Encoder–Decoder (§2.2, :380–415)

- Global KV декодера **не** выводится из собственных hidden: `C_l = H_{L/2}·W_l^KV`, `Z_l = H_{L/2}·W_l^Z` для `l > L/2` (уравнение 1).
- SWA остаётся **послойной** (local KV из `H_l`) — отсюда необходимость **SWA replay**.
- Сложность префилла: `O(NL) → O(NL/2 + n_win·L/2) ≈ O(NL/2)`.
- Мотивация — агентные нагрузки: «frequent tool calls generate extensive prefill requests».

## 3. CSA2 (§2.3, :416–491; кросс-слойное переиспользование :491–547)

- Три **мультипликативные** оси сжатия: entry size (GQA/MLA), sequence (m токенов → 1 запись), layer (переиспользование кэшей/выборов соседних слоёв). Предшественники (IndexCache, YOIO, HySparse) покрывают не все три — заявленное отличие CSA2 в совместном использовании.
- Упрощения против CSA: убраны **overlap** соседних сжатых записей и **absolute positional embedding**; indexer K теперь **проецируется из main KV**, а не отдельным путём из hidden.
- Каждый слой: main Q + **своя SWA KV** + выбранные записи main KV.
- Режимы: **Full** (свой main KV, свой indexer Q, свежие Top-K), **Reindex** (переиспользует main KV + indexer K, своими indexer Q переоценивает Top-K), **Reuse** (переиспользует main KV **и** Top-K, без скоринга). Раскладка — статически по слоям (encoder: 3 группы × 6 слоёв, 1 Full + 5 Reuse, m=2; decoder: 5 групп × 4 слоя, m=1).

## 4. Hierarchical Sparse Indexer (§2.3.2, :548–581)

- Первый CSA2-слой декодера в режиме Full выбирает свои **Top-512** и строит **общий пул кандидатов** из выбранных блоков (blockwise: максимум скора блока).
- Слои в режиме Reindex выбирают Top-512 **только из этого пула** → число скорируемых записей на запрос перестаёт расти с длиной контекста (первый Full-слой всё равно сканирует весь causal-диапазон).
- Механика **train-aware** (ограничение пула одинаково в train и inference).

## 5. FP4 main KV (§2.4.4, :678–707)

- Расширение QAT с indexer Q/K (V4) на **main KV**; FP4 здесь снижает **storage**, а не ускоряет matmul: значения де-квантуются перед attention.
- Формат **OCP MXFP4**: E2M1 + **один E4M3 scale на 16 каналов**, second-level global scale **опущен**.
- Обоснование: максимальная величина RMSNorm-веса ≈1; после нормировки L2-норма 512-канального латента ≤ √512 ≈ 22.6; наблюдаемый максимум ≈10; формат держит до 448×6 = 2688 → «no measurable decrease in accuracy».
- QAT введён в **пост-тренинге**; квантование **после RoPE** (до RoPE — лишь маргинальный прирост при доп. overhead); **SWA KV остаётся FP8** из-за чувствительности к квантованию.
- Против FP8-кэша V4 — почти двукратное сокращение footprint, в HBM и на SSD.

## 6. Persistent KV и SWA Bounded Replay (§3.2.1–3.2.2, :947–1028)

- Persistent KV = 1/8 от V4: (1) SWA KV больше не персистится (~вдвое), (2) global KV сжат до 1/4.
- SWA KV → распределённый **host-DRAM пул (10% DRAM на машину)**, TTL **минуты**, высокая оборачиваемость; **global KV — persistent, lifetime ≥ 72 ч**.
- **Encoder SWA Bounded Replay:** при промахе реплеятся последние `n_win` токенов кэш-префикса; регенерируется только SWA KV, cached global KV переиспользуется без перезаписи.
- **Decoder SWA Bounded Replay:** decoder-forward ограничивается `n_win` токенами, что **почти вдвое снижает общий prefill**; полученный decoder SWA KV используется только для декодинга, не для prefix-caching.
- Реконструкция **приближённая по построению**; вендор называет это границей робастности (§6).

## 7. Эффективные расширения (§2.4.1–2.4.3, :584–677)

- **Single-Pass mHC**: идеал трафика активаций `(2n+2)d`; multi-pass — `(4n+4)d`; сдвиг input-mixing на блок (`A_{l-1}`) + Mega-mHC fused-кернел дают `(2n+2)d`.
- **Engram**: 196B, 2 модуля, n-gram {2,3,4}, 8 hash-голов, dim 2048, ~16M записей/голова, FP8; слои 1 и 14; инференс — детерминированная адресация + фоновый RDMA-префетч из host-памяти.
- **DSpark**: драфтер 3 блока, окно 128, 5 draft-позиций за forward, Markov-head + confidence-head; вводится **отдельной стадией после претрейна** (бэкбон заморожен), продолжает учиться в пост-тренинге без градиентов в бэкбон.

## 8. Оптимизация и претрейн (§2.5, §4.2.1)

- **head-wise Muon** для весов Q/K; Muon momentum 0.95, wd 0.1, RMS-рескейл 0.18; AdamW (β1=0.9, β2=0.95, ε=1e-20, wd 0.1) для RMSNorm/прочих; **Sinkhorn-balanced** update для embed/prediction-head/Engram (K=11, τ=1e-3, γ=0.18).
- **Sparse attention учится с нуля на 64K, без dense-warmup**; расширение до 1M на 34T токенов; sample-level attention masking.
- Batch 100.6M токенов, LR warmup 2000 шагов → 2.6e-4 → cosine до 2.6e-5.
- Vision: **SigLIP** на ~47B пар @224, затем авторегрессионный до-тюн к 4B MoE на 236B токенов, разрешение 544–1344; после — LLM отбрасывается.

## 9. Оценка и агентный контур (§5.3)

- Анти-hacking контур: **отключение интернета, срез git-истории, пурж кэшей** (go/mod, node_modules, .jar, pycache); наблюдался **exploit-seeking** в кибер-оценке.
- Effort 25→100: reasoning 67.1→76.3; DeepSWE 66.0→74.2; TB2.1 82.4→90.6, цена ~2.5× токенов; прирост **фронт-лоадный** (60–80 даёт почти максимум при <½ бюджета).
- Переносимость между scaffold'ами: 8 конфигураций / 6 семейств без деградации.
- **Multi-agent (DSH Agent Team)**: ProgramBench Almost@1 30.04% (8 ч) против 20.39% single; FrontierSWE v2 Mean@5 32.90% (20 ч) против 28.20% single. Reward = task performance + collaboration bonus − **derived-latency** (critical path по token-cost и tool-time).

## 10. Расхождения и оговорки (честно)

| Пункт | Статус |
|---|---|
| Дайджест AINews (скилл `deepseek-v41-flash-architecture`) утверждает: «в multi-agent режиме качество может **падать** — субагенты генерируют slop» | **Противоречит** §5.3.5 отчёта, где Agent Team опережает single на всех дедлайнах. Источник дайджеста — сообщество/критика, не вендор. Для CMP-006 вывод: сравнивать solo vs team при **одинаковом wall-clock дедлайне и с учётом derived-latency**, иначе сравнение невалидно (spine AD-3) |
| «QAT для KV — inferred» (дайджест) | **Устарело/неверно**: §2.4.4 — QAT для main KV заявлен явно («we introduce QAT during post-training»). Inferred в отчёте — только устойчивость к формату без global scale |
| Счёт параметров (552B / 748B / 763B) | В отчёте: **552B backbone + 196B Engram**; остальные числа — из разбора safetensors сообществом, не из отчёта |
| Границы (§6) | Selection-ошибки CSA2 и приближение SWA replay названы предметом дальнейшего стресс-тестирования — для нас это **не гарантия**, а риск (см. ADR-009, Consequences) |
