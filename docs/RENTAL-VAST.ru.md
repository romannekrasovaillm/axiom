# Runbook аренды H800 на vast.ai — претрейн L3 (20B → 17,84B)

Смета: `evidence/budget/pretreain-l3.json` — **compute $130** (116 ч interruptible) + storage/bw ~$15, **лимит $260**. Watchdog: `tools/vast_watchdog.py`.

## 0. Выбор инстанса (критерии поиска)

- GPU: **H800 или H100 SXM** (наши кернелы — CUDA 12+, bf16; H800-калибровка 343 TFLOP/s эфф. переносима на H100 с запасом);
- тип: **interruptible** (bid-кап ниже on-demand; преемпшн безопасен — см. §3);
- **reliability хоста ≥ 0.9** и R diversity-диск ≥ 30 ГБ/s (иначе IO станет бутылочным горлышком даталоадера);
- диск ≥ 120 ГБ (бины 71 ГБ + чекпойнты 2×12 ГБ + venv);
- сети ≥ 1 Гбит (загрузить 71 ГБ бинов за разумное время);
- образ: PyTorch/JAX CUDA 12.x docker-template.

## 0.5. Окружение инстанса (R4, ревью 01.10)

```bash
# venv + jax[cuda12] на инстансе — RS-шаблона нет, собираем:
python3 -m venv /root/venv && /root/venv/bin/pip install -U "jax[cuda12]"
/root/venv/bin/pip install zstandard datasets
# проверка: /root/venv/bin/python -c "import jax; print(jax.devices())" → CudaDevice(id=0)
```

Пакет `ormsgats` (опечатка в предыдущей редакции) не нужен и удалён: лупу
достаточно `zstandard` (чтение `.jsonl.zst`) и `datasets`; лишний пакет в
requirements аренды — лишняя поверхность установки.

## 1. Подготовка (хост → инстанс)

```bash
# после create: ssh-адрес в панели vast
rsync -a --partial --timeout 300 --exclude .git --exclude .arch-handoff ~/axiom/ root@<IP>:/root/axiom/
# бины претокенизации (71 ГБ) — только tokens/, сырые шарды не нужны
rsync -a --partial --timeout 300 ~/gb10-shared/datasets/axiom-pretrain-l3/tokens/ root@<IP>:/root/data/tokens/
rsync -a --partial --timeout 300 ~/gb10-shared/datasets/axiom-pretrain-l3/tokenizer/ root@<IP>:/root/data/tokenizer/
```

`--partial` — докачка оборванного 71-ГБ переноса вместо перезапуска с нуля
(канал до арендованного хоста рвётся; `--timeout 300` не даёт rsync висеть на
мёртвом соединении).

## 2. Старт лупа (смета уже в реестре — порядок AD-8 соблюдён)

Все флаги ниже существуют в CLI (`tools/pretrain_run.py --help`); команда
прогоняется как есть, подставляются только `<ШАГИ>`/`<ПИК>` и адреса хоста.

```bash
cd /root/axiom
export LD_LIBRARY_PATH=$(echo /usr/local/lib/python3*/dist-packages/nvidia/*/lib | tr ' ' ':')
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85   # на инстансе мы одни — но лимит явный
export NET_GATE_PROFILE=1                    # детерминизм ADR-013 запиннен
export STOP_FILE=/root/run/stop              # тот же путь, что у ватчдога (К5)
nohup python3 tools/pretrain_run.py --model-preset l3-full --grad-checkpointing \
  --run-ref pretreain-l3 --shard-root /root/data --tokens-root /root/data/tokens \
  --steps <ШАГИ> --total-steps <ШАГИ> \
  --decay-stream Q --decay-shard-root /root/data \
  --metrics /root/run/metrics.jsonl --ckpt-dir /root/run/ckpt \
  --ckpt-every-min 30 --journal /root/run/journal.json \
  --stop-file /root/run/stop \
  --usd-per-gpu-hour 1.1 \
  > /root/run/pretrain.log 2>&1 &
# ВАЖНО (R1): --journal явно — дефолт путя C-032 (~/gb10-shared) на инстансе отсутствует.
# --ckpt-every-min 30 — риск-каденс (рунбук §3); --stop-file читается лупом каждые
# --stop-check-every шагов (по умолчанию 1), поэтому стоп-файл ватчдога реально тормозит прогон.
# --peak-tflops/--peak-tflops-source — если нужен MFU; без объявленного пика MFU = None
# (не выдумывается). MFU считается по N_active (active_param_count), а не по N_total.
```

`--tokens-root` включает packed-путь (готовые `.bin` из §1) — по умолчанию
включён, если манифесты `tokens/` на месте; сырые `manifest-*.json` остаются
fallback-ом (`--no-packed` форсирует его). Перед стартом CLI сверяет decay-окно
с объёмом Q (H1) и при нехватке либо уменьшает `--decay-ratio` (запись в журнал),
либо отказывает, если доля задана явно.

### 2.5. Dry-run команды старта (обязателен до аренды)

На локальной 4080 те же флаги на малом пресете — команда обязана исполниться
до того, как платить за аренду (прогнано 02.10.2026, exit 0, `status=executed`,
3 шага, `kind=packed`, чекпойнт шага 3):

```bash
export LD_LIBRARY_PATH=$(ls -d ~/venv-axiom/lib/python3.11/site-packages/nvidia/*/lib | tr '\n' ':')
~/venv-axiom/bin/python tools/pretrain_run.py --model-preset small \
  --run-ref pretrain-dryrun --shard-root <корпус> --tokens-root <корпус>/tokens \
  --steps 3 --seq-len 64 --batch-size 1 --ckpt-every-min 30 --checkpoint-every 3 \
  --decay-stream Q --decay-shard-root <корпус> \
  --budget-limit-usd 5 --out ~/gb10-shared/runs/pretrain-dryrun \
  --stop-file ~/tmp/pretrain-dryrun-stop --json
```

Почему `--seq-len 64`, а не боевые 8192: на 4080 (16 ГБ) при словаре корпуса
160K лог-проекция `B·T·V` для `T=8192` не влезает (наблюдался OOM 21,5 ГиБ) —
это ограничение локальной карты, а не путей. Боевой `T=8192` считается на H800
(80 ГБ). Боевой runbook-старт (§2) гоняет **те же** флаги; dry-run меняет только
пресет и `T`, поэтому проверяет именно тот CLI, что пойдёт на аренду.

Dry-run обязан также проверить vocab: модель берёт словарь из манифеста `tokens/`
(фактические id в `.bin`), а не диапазон id канонического BPE — иначе embedding
читается по чужим индексам и лосс становится `NaN`.

Первые 2 часа боевого прогона — замер фактического tok/s: если прогноз по
фактической скорости выходит за 225 USD compute — **стоп-файл и пересмотр**
(потеря ≤$10, не всего прогона).

## 3. Против преемпшна (главный риск interruptible)

- чекпойнт-каденс: `--ckpt-every-min 30` (resume-механика лупа протестирована — продолжение с курсора без потерь и дублей);
- **каждый закрытый чекпойнт — наружу** (инстанс может исчезнуть с диском):
  `rclone`/`rsync` на домашний хост или S3-совместимое хранилище; держим ≥2 последних;
- re-старт после преемпшна: `vastai search offers` заново → rsync чекпойнта на новый инстанс → та же команда (resume по манифесту-курсору);
- если рынок interruptible пуст надолго — fallback on-demand (+50% цены, тот же лимит $260 корректируется вручную сметой, не молча).

## 4. Watchdog (слой №4, независим от лупа)

Два сторожа, две роли (D4: ключ vast — только на домашнем хосте):

```bash
# (а) на ИНСТАНСЕ — только stop-файл из локальных метрик, ключ vast не нужен:
nohup python3 tools/vast_watchdog.py --instance-id <id> \
  --metrics /root/run/metrics.jsonl --usd-per-gpu-hour 1.1 \
  --spend-cap 225 --stop-file /root/run/stop > /root/run/watchdog.log 2>&1 &
# (б) на ДОМАШНЕМ ХОСТЕ — жёсткие границы через vast CLI (синхр. копия метрик §4.5):
nohup python3 tools/vast_watchdog.py --host --instance-id <id> \
  --metrics ~/gb10-shared/runs/pretreain-l3/metrics.jsonl \
  --spend-cap 225 --hard-cap 260 --balance-min 20 > /tmp/watchdog.log 2>&1 &
```

- $225 (compute-лимит) → **stop-файл** — луп остановится сам на ближайшем чекпойнте
  (луп читает тот же `--stop-file`/`STOP_FILE`, см. §2 — без совпадения путей стоп-файл не сработает);
- $260 (полный лимит) → **vast stop** жёстко (хост, режим `--host`);
- баланс < $20 → stop — vast при нуле кредитов без карты **удаляет инстанс и данные** (docs: pricing → Billing Basics);
- инстанс исчез из `vastai show instances` (id сменился после пересоздания) → ватчдог
  **кричит STALE и выходит с кодом 4**: сторожить нечего, молчаливый «ноль spend» был
  бы ложным зелёным вердиктом. Метрики читаются **синхронизированной копией** (путь
  аргументом `--metrics`), а не путём на инстансе: хост не видит файловую систему инстанса.

`gpu_hours` в метриках лупа накопительный по прогону (К3: сумма ног при resume),
поэтому spend не «сбрасывается» на каждом перезапуске после преемпшна.

## 4.5. Возврат артефактов (R2, ревью 01.10 — pull-модель)

Инстанс vast может исчезнуть с диском в любой момент → **не push с инстанса, а pull с домашнего хоста** (cron каждые 30 мин):

```bash
# cron на домашнем хосте (инстанс имеет публичный IP+ssh-порт):
*/30 * * * * rsync -a --partial --timeout 300 root@<IP>:/root/run/ckpt/ ~/gb10-shared/runs/pretreain-l3/ckpt/ ;   rsync -a --partial --timeout 300 root@<IP>:/root/run/{metrics.jsonl,journal.json,pretrain.log} ~/gb10-shared/runs/pretreain-l3/
```

`--partial` докачивает оборванную копию чекпойнта/метрик (инстанс может
преемптнуться во время pull), `--timeout 300` не даёт rsync висеть на мёртвом соединении.
Именно эта копия `metrics.jsonl` — вход хост-ватчдога (§4): spend считается от неё.

Приёмка каждого чекпойнта: сверка tree_hash с журналом (побитовая целостность, как A5). Последний чекпойнт после завершения — той же командой; инстанс destroy только после сверки.

## 5. Финал

- последний чекпойнт → rsync домой → `tree_hash` сверить с журналом;
- фактический spend в evidence (смета vs факт — корм для следующей калибровки);
- инстанс **destroy** (не stop — storage тарифицируется в стопе!);
- манифест прогона → гейт закрытия претрейна.

## Ревью 01.10.2026 (второй проход) — исправления килл-свитча

- **D2 (критично):** spend считался из непроверенных полей vast (fallback `cur_state`/3600 — state-код, не секунды) → молчаливое занижение = килл-свитч не срабатывает никогда. Фикс: primary spend = `gpu_hours` из метрик лупа (schema pretrain-metrics/v1 — верифицированный источник) × ставка инстанса `dph_total`; unknown ставка → громкий крик, не молчание.
- **D3 (критично):** отсутствие поля `credit` трактовалось как 0 → ложная остановка инстанса. Фикс: unknown → пропуск проверки (fail-loud).
- **D4:** API-ключ vast на арендованной машине — риск угона: баланс/hard-cap проверяются на домашнем хосте, on-instance остаётся только spend→stop-файл из локальных метрик (ключ не нужен).
- **D1 (косметика):** остаток прототипа в f-string удалён.

Правило ревью: килл-свитч обязан отказываться ГРОМКО (unknown → крик в лог), никогда — молча и никогда — ложно.

## 6. Preflight: `unverified` не открывает расход (ADR-036)

До аренды — обязательный preflight сметы: каждый пункт `requires_verdicts`
(guard `performance-roofline` и/или утверждение `CL-NNN`) обязан дать `pass`:

```bash
python3 tools/check_budget_gate.py --preflight pretreain-l3
python3 tools/check_performance_roofline.py --run kda-wyut-delta \
  --metrics <metrics.jsonl> --require-verified     # neutral → exit 3 (unverified)
```

Отказ preflight («unverified/fail — расход не открыт») означает: KPI/вместимость не
доказаны фактами — аренда не открывается. Запуск сторожа `tools/vast_watchdog.py`
выполняется только после PASS preflight (точки старта инстанса в самом сторожe нет —
preflight живёт здесь, как шаг runbook).
