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
python3 -m venv /root/venv && /root/venv/bin/pip install -U "jax[cuda12]" ormsgats 2>/dev/null || /root/venv/bin/pip install -U "jax[cuda12]"
/root/venv/bin/pip install zstandard datasets
# проверка: /root/venv/bin/python -c "import jax; print(jax.devices())" → CudaDevice(id=0)
```

## 1. Подготовка (хост → инстанс)

```bash
# после create: ssh-адрес в панели vast
rsync -a --exclude .git --exclude .arch-handoff ~/axiom/ root@<IP>:/root/axiom/
# бины претокенизации (71 ГБ) — только tokens/, сырые шарды не нужны
rsync -a ~/gb10-shared/datasets/axiom-pretrain-l3/tokens/ root@<IP>:/root/data/tokens/
rsync -a ~/gb10-shared/datasets/axiom-pretrain-l3/tokenizer/ root@<IP>:/root/data/tokenizer/
```

## 2. Старт лупа (смета уже в реестре — порядок AD-8 соблюдён)

```bash
cd /root/axiom
export LD_LIBRARY_PATH=$(echo /usr/local/lib/python3*/dist-packages/nvidia/*/lib | tr ' ' ':')
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85   # на инстансе мы одни — но лимит явный
export NET_GATE_PROFILE=1                    # детерминизм ADR-013 запиннен
nohup python3 tools/pretrain_run.py --model-preset l3-full --grad-checkpointing \
  --run-ref pretreain-l3 --shard-root /root/data --metrics /root/run/metrics.jsonl \
  --ckpt-dir /root/run/ckpt --journal /root/run/journal.json > /root/run/pretrain.log 2>&1 &
# ВАЖНО (R1): --journal явно — дефолт путя C-032 (~/gb10-shared) на инстансе отсутствует
```

Первые 2 часа — замер фактического tok/s: если прогноз `20B-прогон` по фактической скорости выходит за 225 USD compute — **стоп-файл и пересмотр** (потеря ≤$10, не всего прогона).

## 3. Против преемпшна (главный риск interruptible)

- чекпойнт-каденс: `--ckpt-every-min 30` (resume-механика лупа протестирована — продолжение с курсора без потерь и дублей);
- **каждый закрытый чекпойнт — наружу** (инстанс может исчезнуть с диском):
  `rclone`/`rsync` на домашний хост или S3-совместимое хранилище; держим ≥2 последних;
- re-старт после преемпшна: `vastai search offers` заново → rsync чекпойнта на новый инстанс → та же команда (resume по манифесту-курсору);
- если рынок interruptible пуст надолго — fallback on-demand (+50% цены, тот же лимит $260 корректируется вручную сметой, не молча).

## 4. Watchdog (слой №4, независим от лупа)

```bash
nohup python3 tools/vast_watchdog.py --instance-id <id> \
  --spend-cap 225 --hard-cap 260 --balance-min 20 > /tmp/watchdog.log 2>&1 &
```

- $225 (compute-лимит) → **stop-файл** — луп остановится сам на ближайшем чекпойнте;
- $260 (полный лимит) → **vast stop** жёстко;
- баланс < $20 → stop — vast при нуле кредитов без карты **удаляет инстанс и данные** (docs: pricing → Billing Basics).

## 4.5. Возврат артефактов (R2, ревью 01.10 — pull-модель)

Инстанс vast может исчезнуть с диском в любой момент → **не push с инстанса, а pull с домашнего хоста** (cron каждые 30 мин):

```bash
# cron на домашнем хосте (инстанс имеет публичный IP+ssh-порт):
*/30 * * * * rsync -a --timeout 300 root@<IP>:/root/run/ckpt/ ~/gb10-shared/runs/pretreain-l3/ckpt/ ;   rsync -a root@<IP>:/root/run/{metrics.jsonl,journal.json,pretrain.log} ~/gb10-shared/runs/pretreain-l3/
```

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
