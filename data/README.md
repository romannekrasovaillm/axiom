# data/ — монтирование наборов SFT/RL симлинками (C-032/C-033)

Канонически датасеты живут на сетевом диске `~/gb10-shared/datasets`; в рабочем
каталоге кейса лежат **только симлинки**. Копия крупного файла в рабочем
каталоге запрещена (решение владельца 12.09.2026) и проверяется механически:

```bash
find . -type f -size +500M -not -path './.git/*' -print -quit   # пусто = C-033 ok
find . -type f \( -name '*.safetensors' -o -name '*.gguf' -o -name '*.pt' \
    -o -name '*.pth' -o -name '*.ckpt' \) -print -quit          # пусто = C-032 ok
```

`find -type f` не разыменовывает симлинки, поэтому смонтированный набор под
запрет не попадает — это и есть смысл правила.

## Что смонтировано

| Симлинк | Цель | Назначение |
|---|---|---|
| `datasets/sft_train_v12.jsonl` | `~/gb10-shared/datasets/sft_train_v12.jsonl` | SFT-стадия A4 (`tools/run_sft_smoke.py`) |

`sft_train_v12.jsonl` = `sft_train_v10.1.jsonl` + `sft_train_v11.jsonl`
(`~/gb10-shared/build_sft_v12.sh`, 28.08.2026), 44 949 документов формата
`{"messages": [...]}`. Выбор набора по умолчанию и причина — в журнале стадии
(`evidence/a4-run-wire/sft/stage-journal.json`, поле `notes`).

Чекпойнты стадии монтируются так же: канонически —
`~/gb10-shared/checkpoints/sft-smoke`, в репозитории симлинк
`evidence/a4-run-wire/sft/checkpoint` (C-032: веса только симлинками).

## Восстановление монтирования

```bash
ln -sfn ~/gb10-shared/datasets/sft_train_v12.jsonl data/datasets/sft_train_v12.jsonl
```

Симлинк, ведущий на несуществующий файл или вне `~/gb10-shared`, — отказ
стадии до всякого обучения (`tools/run_sft_smoke.py:validate_data_path`).
