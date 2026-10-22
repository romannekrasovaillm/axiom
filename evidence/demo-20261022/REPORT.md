# Axiom demo — pretrain + midtrain на DGX Spark (GB10)

- **Дата:** 2026-10-22, стенд `spark-44c3` (GB10, 121 ГБ unified, NVRM 580.173.02).
- **Директива владельца:** демо-обучение Axiom на претрейне и мидтрейне, бюджет времени ровно 24 часа.
- **Фактическое время:** старт 05:04:37Z, последний чекпоинт 05:59Z; ~55 минут, дедлайн не достигнут.
- **Репозиторий:** `/home/roman/axiom`, ветка `arch/bf16-phase1-wip`, HEAD `5b43e6f7`.
- **Данные:** претрейн — `tokens-v2` (поток W, токенизатор v2, hash `16e4a532…`); мидтрейн — доменный K/D/S (`axiom-domain-ds-v1`, `cpt-kds-v0.1`, 102 395 документов, 46.8M токенов после упаковки).
- **Бронь:** `~/gb10-shared/.locks/axiom-demo-20261022.lock`, `DEADLINE_EPOCH` = старт + 86400.

## 1. Префлайт стенда

- Остановлен только vLLM `llm-platform-vllm-qwen35-9b` (34.3 ГБ); gateway/supervisor не тронуты.
- После остановки: `nvidia-smi --query-compute-apps` пуст, доступно 106 ГБ, GPU 4.8 Вт / 44 °C.
- `XLA_PYTHON_CLIENT_MEM_FRACTION=0.96` (выделенный стенд), `XLA_PYTHON_CLIENT_PREALLOCATE=false`, `TF_GPU_ALLOCATOR=cuda_malloc_async`.
- Восстановление сервиса: `docker start llm-platform-vllm-qwen35-9b` (выполнено после прогонов).

## 2. Что запускалось

Кампанийный стек (тот же, что лидер MFU-55): `AXIOM_KDA_CC_IMPL=persistent`, `AXIOM_MTP_KDA_IMPL=campaign`, `AXIOM_GEMM_FP8=1`, `AXIOM_MOE_BWD=fused`, `AXIOM_MLA_PERSIST=1`, `AXIOM_MLA_BWD_BF16=1`, `AXIOM_CE_BF16=1`, `AXIOM_MASTER_DTYPE=bf16`, `AXIOM_GRAD_CKPT_POLICY=per_layer_block`, `--remat-policy none`, `--chunk-size 32`, `--ce-chunk-tokens 2048`.

| стадия | модель | данные | шаги | batch | LR | чекпоинт |
|---|---|---|---|---:|---:|---|
| pretrain | l3-full (1.01B / 502.4M active) | W, packed | 20 | 4 → 2 | 1e-2 | шаг 10 |
| midtrain (доменный CPT) | та же, resume с шага 10 | K/D/S, packed | 20 (шаги 11–30) | 2 | 2e-3 | шаги 10/20/30 |

Команда-эталон (претрейн):

```
tools/pretrain_run.py --model-preset l3-full --grad-checkpointing \
  --grad-checkpointing-policy full --seq-len 8192 --batch-size 4 --micro-batch 4 \
  --steps 20 --total-steps 20 --budget-limit-usd 5 --peak-tflops 98.2 \
  --ce-chunk-tokens 2048 --remat-policy none --chunk-size 32 \
  --shard-root ~/axiom-run/raw-v2 \
  --tokens-root ~/gb10-shared/datasets/axiom-pretrain-l3/tokens-v2 \
  --streams W --mix W=1.0 --checkpoint-every 10 --keep-last 3
```

Мидтрейн: тот же вызов с `--resume --lr 2e-3 --batch-size 2 --micro-batch 2`, `--streams K --mix K=1.0`, `--shard-root ~/axiom-run/domain-cpt`, `--tokens-root ~/gb10-shared/datasets/axiom-domain-ds-v1/tokens`.

## 3. Результаты

Стационарные медианы; первый шаг стадии исключён; nan = 0 везде.

| прогон | шаги | ток/с | с/шаг | MFU_98.2 | MFU_77.3 | loss |
|---|---|---:|---:|---:|---:|---|
| pretrain B=4 (чистые шаги 2–10) | 9 | **1 423.1** | **23.03** | 4.37 % | 5.55 % | 13.542 → 12.253 (шаг 10) |
| pretrain B=4, шаги 11–13 (после сейва) | 3 | 288.4 | 113.6 | 0.89 % | 1.13 % | деградация |
| pretrain B=2 (шаги 12–20) | 9 | 1 305.4 | 12.55 | 4.01 % | 5.09 % | 11.799 → 11.654 |
| **midtrain B=2 (шаги 12–30, без сейвов)** | 19 | **1 192.9** | **13.74** | 3.66 % | 4.65 % | **15.832 → 9.335** |
| midtrain B=2 (шаги 21–29, с сейвом на 20) | 9 | 1 085.2 | 15.10 | 3.33 % | 4.23 % | 10.102 → 9.350 |

- Претрейн: 20 шагов, tokens_seen 491 520, `status: executed`, cursor restored с шага 10.
- Мидтрейн: 20 шагов, 655 360 токенов, `status: executed`, loss упал на 6.50 нат за 20 шагов — домен для претрейн-чекпоинта OOD, поэтому стартовый loss 15.83.
- Финальный чекпоинт мидтрейна: шаг 30, `tree_hash=fb340814…`, `optimizer_state_hash=e590bf05…`, `master_dtype=bf16`, `master_comp=kahan`.

## 4. Инциденты и уроки (важно)

1. **Orbax-сейв на UMA удерживает host-память.** После сохранения чекпоинта шага 10 (B=4) RSS процесса вырос до 24.7 ГБ, доступная память упала с 23 до **5.9 ГБ**, шаги 11–13 деградировали до 44 / 167 / 114 с. Это не утечка JAX, а цена сейва: device 82.5 ГиБ + host-буфер. Прогон остановлен на чекпоинте шага 10; продолжение — B=2.
2. **Resume B=4 с загрузкой чекпоинта упирается в потолок.** Пик 98.6 ГиБ, свободно 2 ГБ — процесс остановлен до OOM. B=2 (66 ГиБ) безопасен.
3. **Смена batch — холодная компиляция.** Переход B=4 → B=2 при другом кэше дал ~15 минут компиляции (первый шаг 26.2 ток/с), только затем шаги 12.5 с.
4. **`--checkpoint-every` — абсолютный шаг, не относительный.** Cadence 20 сохранил шаг 20 и пропустил 30 (30 % 20 ≠ 0); для шага 30 нужен cadence 10. Это стоит зафиксировать в скилле.
5. **Журнал мидтрейна помечен `stage: pretrain`.** Отдельной стадии `midtrain`/`cpt` в `pretrain_run.py` нет; отличие видно по `streams: [K]`, `resume.from_step` и данным. Это пробел инструмента, а не данных.
6. **B=4/B=2 в одном демо — не controlled A/B.** tok/s стадий несопоставимы напрямую: у B=2 вдвое меньше токенов на шаг. Числа выше — измерение, не вердикт.
7. **Финальные веса мидтрейна сохранены, финальные веса претрейна шага 20 — нет.** Претрейн-чекпоинт шага 10 — точка передачи; шаги 11–20 претрейна измерены, но не сохранены (checkpoint-every 0 после инцидента).

## 5. Артефакты

```
~/gb10-shared/runs/axiom-demo-20261022/
  pretrain/
    metrics-attempt1.jsonl      # B=4, шаги 1-13 (включая деградацию после сейва)
    metrics-b2.jsonl            # B=2 resume, шаги 11-20
    journal.json                # resume с шага 10, status executed
    checkpoints/step-00000010/  # tree_hash 34d50e74…, 5.7 ГБ
  midtrain/
    metrics.jsonl               # чистый прогон, шаги 11-30
    metrics-ckpt.jsonl          # прогон с сейвом на шаге 20
    metrics-ckpt2.jsonl         # догон 21-30, сейв на шаге 30
    journal.json                # checkpoint step-30, tree_hash fb340814…
    checkpoints/step-00000010|20|30/  # по 5.7 ГБ
  summary.json                  # машиночитаемая сводка
```

Данные мидтрейна: `~/axiom-run/domain-cpt/K/{manifest-k.json,K-00000.jsonl.zst}` → `~/gb10-shared/datasets/axiom-domain-ds-v1/tokens/K/{manifest-k.json,K-00000.bin}` (46.8M токенов, 5 719 записей, токенизация 10.8 с на 8 воркерах).

## 6. Применённые скиллы и что они изменили

- `gb10-mfu-campaign-loop` — префлайт, бронь с дедлайном, дискретные вердикты, правило «атрибуция до кода».
- `gb10-jax-memory-arena` — MEM_FRACTION 0.96, бюджет B=4/B=2, распознавание OOM-риска по `MemAvailable`; именно она остановила B=4 resume на 2 ГБ.
- `gb10-experiment-launcher` — схема запуска через `setsid`/SSH, сбор логов и чекпоинтов.
- `gb10-parallel-gpu-oom` — правило «один vLLM за раз», остановка сервиса с командой восстановления.
- `gb10-jax-step-attribution` — не понадобился: демо не оптимизационное, адрес не искали.

## 7. Оценка чекпоинтов на holdout (добавлено после демо)

Протокол: NTP loss, seq 2048, до 131 072 токенов на набор; домен — `domain_eval_v3.txt` (61 окно), общий — `W-00051.bin` (63 окна, shard вне обучения; обучение прочитало только начало shard 0). Токенизатор v2. Абсолютные PPL высоки (скелет, 20 шагов), значимы дельты.

| checkpoint | domain loss | domain PPL | W loss | W PPL |
|---|---:|---:|---:|---:|
| pretrain step 10 | 15.632 | 6 147 598 | 10.925 | 55 562 |
| midtrain step 20 | **12.167** | **192 424** | **11.302** | **80 943** |
| midtrain step 30 | 12.294 | 218 356 | 11.896 | 146 679 |

- Домен: step10 → step20 **−3.464 nat**, step20 → step30 **+0.126 nat**.
- Общий корпус: step10 → step20 **+0.376 nat**, step20 → step30 **+0.594 nat**; итого **+0.971 nat** (PPL ×2.64).
- **Лучший баланс — step20, не последний step30.** Домен улучшился ценой forgetting; дальнейшие 10 шагов при LR 2e-3 ухудшили оба holdout (переобучение на узком домене / слишком высокий LR для 655k токенов).
- Артефакт: `~/gb10-shared/runs/axiom-demo-20261022/eval_ppl.json`; скрипт-зонд `~/axiom-run/demo-20261022/eval_checkpoint.py` (вне репозитория, не коммитился).

## 8. Политика мидтрейна: lower LR vs replay (clean holdout)

**Leak-check (AD-12/C-043).** Исходный `domain_eval_v3` дал `verdict: leak`: 0.27 % 12-граммовых окон, один документ (169) пересекается с корпусом K/D/S. После удаления этого документа набор — **199 документов, `verdict: clean`, ngram_overlap 0.0**. Все числа ниже — на чистом наборе (122 820 предсказанных токенов, 60 окон) и на W-00051 (128 961 токен, 63 окна).

| чекпоинт | domain loss | W loss | Δdomain vs pretrain | ΔW vs pretrain |
|---|---:|---:|---:|---:|
| pretrain step 10 | 15.640 | 10.924 | — | — |
| baseline step 20, LR 2e-3, K=1.0 | **12.177** | 11.302 | −3.464 | +0.378 |
| baseline step 30, LR 2e-3, K=1.0 | 12.307 | 11.896 | −3.333 | +0.971 |
| replay step 20, LR 2e-3, K=0.8/W=0.2 | 12.198 | 11.304 | −3.443 | +0.380 |
| **lowlr step 20, LR 1e-3, K=1.0** | **12.457** | **10.764** | **−3.183** | **−0.161** |

- **Критерий гейта** (domain ≤ 12.6, W ≤ 11.1) проходит только **lowlr step20**: domain −3.18 нат, W −0.16 нат (forgetting нет, общий корпус даже улучшился).
- **Replay 20 % при LR 2e-3 нейтрален**: +0.021 нат на домене и +0.002 нат на W относительно baseline step20. При высоком LR добавка данных не спасает от forgetting; рычаг — шаг обучения.
- **Baseline step20** — лучший домен (−3.46), но W +0.38 > +0.2 — гейт не проходит.
- **Step30** хуже step20 по обоим holdout: поздние шаги при LR 2e-3 переобучают.
- Рекомендация: мидтрейн с LR 1e-3, 10 шагов, отбор чекпоинта по holdout (step20); replay вводить только как отдельный A/B при уже снижённом LR.

## 9. Репликация seed 2027 и replay

| seed | pretrain (domain / W) | baseline step20, LR 2e-3 | lowlr step20, LR 1e-3 |
|---|---|---|---|
| 1337 | 15.641 / 10.925 | 12.176 / 11.302 | 12.457 / 10.763 |
| 2027 | 15.293 / 10.889 | 12.138 / 11.527 | **12.125 / 10.738** |

- **Lowlr воспроизведён:** domain −3.18 / −3.17 нат, W −0.16 / −0.15 нат; гейт (domain ≤12.6, W ≤11.1) проходит на обоих seed.
- **Baseline не проходит:** W +0.38 / +0.64 нат — forgetting устойчив, это не шум одного seed.
- **Replay 10 % при LR 1e-3** (seed 1337): 12.454 / 10.759 против lowlr 12.457 / 10.763 — Δdomain −0.003, ΔW −0.004, нейтрально. Replay не входит в дефолт.
- Политика принята владельцем и зафиксирована в `docs/adr/ADR-055-politika-midtreyna-axiom.md`.

## 10. Приёмка постоянного инструмента `tools/eval_checkpoint.py`

Постоянный инструмент (ветка `arch/claude-code-20261022082621`, коммит `20edba7`) проверен на реальных чекпоинтах GB10 против транзиентного зонда:

| чекпоинт | domain (tool) | domain (зонд) | W (tool) | W (зонд) |
|---|---:|---:|---:|---:|
| seed1337 pretrain step10 | 15.64095 | 15.64039 | 10.93667 | 10.92450 |
| seed1337 lowlr step20 | 12.45840 | 12.45780 | 10.77283 | 10.76330 |

Домен совпал до 0.001 нат (60 окон); W отличается на ~0.01 нат, потому что инструмент берёт 64 полных окна (131 008 предсказанных токенов) против 63 у зонда. `--plan-only` работает без JAX/GPU; 31/31 новых CPU-тестов зелёные. Артефакт: `eval_tool_check.json`.

## 11. Честные границы

Два seed (1337, 2027) для baseline/lowlr, один — для replay-плеч; один стенд, одна модель; B=4 и B=2 смешаны; мидтрейн-корпус — доменный K/D/S, не продолжение W; XLA-детерминизм не пиннился (`NET_GATE_PROFILE=1`), побитовая воспроизводимость не гарантируется. Демо доказывает провод стадий и работоспособность стека, но не является KPI-вердиктом.

## 12. Smoke интегрированного `--stage midtrain`

1 шаг от pretrain step-10 (B=2, K/D/S, `--resume`, без `--lr`): журнал
`stage=midtrain`, `policy={"name":"ADR-055","checkpoint_selection":"holdout","leak_check_required":true,"lr":0.001}`,
`resume.from_step=10`, `status=executed`; loss 15.8316, checkpoint не сохранялся
(`checkpoint-every 0`). Подтверждает, что политика ADR-055 исполняется на
интегрированном коде GB10, а не только в тестах.

## 13. Горизонт мидтрейна: 10 vs 20 шагов при LR 1e-3

| seed | 10 шагов (domain / W) | 20 шагов (domain / W) | Δdomain | ΔW (20 vs 10) | W(20 vs pretrain) |
|---|---|---|---|---|---|
| 1337 | 12.4571 / 10.7736 | **11.7685** / 10.8498 | −0.6886 | +0.0762 | −0.0860 |
| 2027 | 12.1245 / 10.7483 | **11.6653** / 11.0614 | −0.4592 | +0.3131 | +0.1594 |

20 шагов дают лучший домен, но на seed 2027 W упирается в верхнюю границу 11.1.
Дефолт — **10 шагов**; 20 — только domain-first с holdout-проверкой W ≤ +0.2.
Replay 10 % на 20 шагах (seed 2027, K=0.9/W=0.1): domain 11.6914, W 11.0546 против 11.6649 / 11.0610 без replay — нейтрально (Δdomain +0.0265, ΔW −0.0064); край W не снимает.
LR 5e-4 на 20 шагах (seed 2027): domain 12.6874, W 10.8459 — domain выше порога 12.6, W не лучше 10 шагов (10.7468); вариант отклонён.
Третий seed (42, 20 шагов LR1e-3): domain 11.8348, W 11.1209 — выше 11.1, гейт не пройден. На трёх seed 20 шагов W = 10.8498 / 11.0597 / 11.1209; дефолт 10 шагов подтверждён окончательно (domain 12.09–12.46, W 10.75–10.83).

## 14. SFT smoke (l3-full, 20 шагов)

- Данные: `sft_train_v12-normalized.jsonl` (симлинк на `~/gb10-shared`), карточка: 44 949 строк, 486 390 312 Б, sha256 `fd7908bc…`, 13 блоков.
- 20 шагов, QAT=on, C=32, MEM_FRACTION=0.85; **loss 7.5669 → 2.1787**, wall 387 с (14.2 с/шаг на финише).
- Чекпоинт: Orbax `tree_hash=7ae62134…`, roundtrip_ok, canonical `~/gb10-shared/checkpoints/sft-smoke` (2.7 ГБ); в репозитории — симлинк.
- Preflight CL-017 OVERRIDDEN с причиной «smoke-only: eval-набор не подключён; вердикт не выносится»; обход зафиксирован фактом S-033.
- Уроки: SFT l3-full требует ~82.5 ГиБ и MEM_FRACTION 0.85 (оценка ресурсов 15 ГиБ занижена); default `chunk_size=64` падает на persistent KDA — нужен 32.

## 15. RL smoke (small, 5 шагов)

- Цепочка: small SFT (10 шагов) → RL smoke: 5 шагов, K=2, inner_epochs=2, lr=1e-3, τ=0.2, β=1e-3.
- `status=executed`, wall **1966.6 с**; `reward_mean_first/last = 0.0` — калибровка K3 (pass-rate 10–90%) не проводилась, вырожденные группы ожидаемы; это smoke, не вердикт.
- `weights_changed=true`, `parameter_norm_delta=-0.0735`; checkpoint Orbax `tree_hash=6d1c31a4…`, 1.5 МБ, roundtrip ok.
- `arch-ml` обязан быть в PATH (`~/.cargo/bin`), иначе `ArchMlUnavailable`.
- **Дефект:** `run_sft_smoke` пишет canonical-чекпоинт в `~/gb10-shared/checkpoints/sft-smoke` независимо от `--out`; small SFT перезаписал l3-full SFT-чекпоинт (журнал l3-full с `tree_hash=7ae62134` остался, веса — уже small). Кандидат фикса: canonical-путь должен включать run/out.
