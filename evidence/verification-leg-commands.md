# VERIFICATION-LEG — команды прогонов (окно GB10 после очереди AD-7)

Предмет: dense-124M vs arch-124M при равном бюджете (~1B токенов, seq 8192, batch 1).

```bash
# 0. Лок-протокол (AD-7/C-040): аппенд в ~/gb10-shared/.locks/axiom-run.lock
#    «СТАТУС: ACTIVE — verification-leg dense-124M, владелец остановки: roman»

# 1. Dense-124M (~1B токенов)
python3 tools/run_sft_smoke.py \
  --config-path net/config-dense124m.json \
  --data /home/roman/gb10-shared/datasets/axiom-pretrain-l3/tokens-v2/W \
  --steps 122000 --seq-len 8192 --batch-size 1 \
  --ckpt-dir /home/roman/axiom-run/verification-leg/dense124m/ckpt \
  --out /home/roman/axiom-run/verification-leg/dense124m \
  --json
# steps: 1B токенов / 8192 = 122071 шагов при batch 1 × seq 8192

# 2. Arch-124M (та же ширина/глубина, композиция KDA/MLA 9+3)
python3 tools/run_sft_smoke.py \
  --config-path net/config-arch124m.json \
  --data /home/roman/gb10-shared/datasets/axiom-pretrain-l3/tokens-v2/W \
  --steps 122000 --seq-len 8192 --batch-size 1 \
  --ckpt-dir /home/roman/axiom-run/verification-leg/arch124m/ckpt \
  --out /home/roman/axiom-run/verification-leg/arch124m \
  --json

# 3. BPB-отчёт по каждой кривой
python3 tools/bpb_report.py --loss-jsonl <out>/metrics.jsonl \
  --sample-texts /home/roman/gb10-shared/datasets/axiom-pretrain-l3/tokens-v2 \
  --out evidence/verification-leg-<имя>/bpb.json
```

## Предостережения

1. **steps=122000 — проверка бюджета**: 122k шагов × ~1 с/шаг (dense-124M на GB10,
   грубая оценка 15–30 ток/с... расчёт: 1B токенов / 8192 = 122k шагов; при
   100% util dense-124M ~10-20 ток/с-реалистично? НЕ УТВЕРЖДАТЬ БЕЗ СМОУКА:
   сначала 50-шаговый смоук каждого конфига (5-10 мин), замер с/шаг, пересчёт
   реального времени ноги. Если 1B-нога >4 ч — сократить до 0.25B (30k шагов)
   и вердикт по наклону раннего участка (спека §Метрика п. (ii)).
2. **--data формат**: уточнить у раннера (tokens-v2/W/*.bin поддерживается ли
   напрямую; если раннер ждёт один .bin — собрать cat W-*.bin | head -c <бюджет>).
3. **arch-124M медленнее dense** (KDA/MLA машинерия) — паритет бюджета считать
   по токенам, wall-clock разница = тоже артефакт (ток/с сравнение = KPI-данные).
4. Стоп-протокол §3.1 действителен для обеих ног.
