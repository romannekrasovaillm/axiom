# `prep_pretrain` — подготовка претрейн-датасета L3 (ADR-021)

Потоковая сборка шардов претрейн-микса: **W** (веб, FineWeb-Edu, ~17B токенов) и
**C** (код, The Stack, ~3B токенов). Шард Q (decay/annealing, ~1B) этим
пайплайном пока не собирается — он берётся сужённым фильтром из W+C.

Данные живут вне git: `~/gb10-shared/datasets/axiom-pretrain-l3/{W,C}/`
(C-032/C-033). В репозиторий попадают только код, тесты и карточка.

## Модули

| Модуль | Роль |
|---|---|
| `common.py` | `ShardWriter` (ротация по размеру сжатого файла, потоковый sha256, `.part` + rename), `Manifest` (шарды + курсор resume), `BoundedHashSet` (LRU-дедуп), `approx_tokens`, источники (локальный jsonl, HuggingFace stream), общий прогон шарда |
| `fineweb.py` | шард W: `HuggingFaceFW/fineweb-edu`, конфиг `sample-100BT`, нормализация записей |
| `stack.py` | шард C: реестр источников The Stack, фильтры языка/лицензии/длины, пофайловые метаданные |
| `build.py` | CLI: `prepare-w`, `prepare-c`, `probe`, `sources`, `verify-manifest` |

## Запуск

```bash
# проба на малом объёме (в /tmp, боевую загрузку не запускает)
python -m prep_pretrain.build probe --limit-mb 200 --shard-mb 100 --progress

# доступность источников кода и доля записей, проходящих фильтры
python -m prep_pretrain.build sources --report /tmp/sources.json

# боевые прогоны (запускает владелец: 17B и 3B — это десятки часов канала)
python -m prep_pretrain.build prepare-w --target-tokens 17e9 --progress
python -m prep_pretrain.build prepare-c --target-tokens 3e9 --source stack-dedup-v1 --progress

# проверка хешей шардов по манифесту (пересчёт с диска)
python -m prep_pretrain.build verify-manifest \
    --manifest ~/gb10-shared/datasets/axiom-pretrain-l3/W/manifest-w.json
```

Полную загрузку CLI сам не запускает: проба ограничена `--limit-mb` и пишет в
`/tmp/axiom-pretrain-probe`.

## Контракты, которые держит пайплайн

- **Потоковость.** Датасет не материализуется (`datasets` streaming); в памяти —
  одна запись и окно дедупа (`--dedup-window`, по умолчанию 5М хешей). Отчёт
  несёт `rss_at_start_mb` / `rss_growth_mb` / `max_rss_mb` — рост RSS за прогон
  виден и не маскируется импортом `datasets`.
- **Resume.** Манифест пишется по мере закрытия шардов (и периодически по ходу
  длинного шарда). `source_records` — сколько записей источника обработано
  полностью; повторный запуск проматывает ровно столько и продолжает. Запись, на
  которой прогон оборвался, перечитывается (не теряется). `--restart` начинает
  заново.
- **Целостность.** Шард пишется как `.part` и переименовывается только после
  успешного закрытия; sha256 считается по байтам, попавшим в файл, и потому
  равен хешу файла на диске (`verify-manifest` перепроверяет пересчётом).
- **Границы записи.** `ensure_output_allowed` пускает вывод только в
  `~/gb10-shared`, `/tmp`, `/var/tmp` (C-032/C-033); обход — явным
  `--allow-any-out`.
- **Сеть.** `sanitize_proxy_env` снимает `ALL_PROXY` с socks-схемой: httpx
  (huggingface_hub 1.x) не умеет socks без `socksio` и падает до запроса.
  Таймауты чтения подняты (`HF_HUB_DOWNLOAD_TIMEOUT=60`), потому что паркет
  источников — гигабайтный и дефолтный таймаут уходит в ретраи.

## Шард W (веб)

`HuggingFaceFW/fineweb-edu`, конфиг `sample-100BT` (100B токенов edu-выборки;
цели 17B хватает с запасом). Качество уже отфильтровано edu-классификатором;
порог `--min-int-score` выключен по умолчанию и оставлен зарезервированным.
Запись: `{"text": …, "meta": {source, id, dump, url, language, int_score}}`.

## Шард C (код)

Языки: `python, rust, go, javascript, shell`. Лицензии: `mit, apache-2.0,
bsd-3-clause, bsd-2-clause, isc, 0bsd` (лицензия проходит, только если ВСЕ
заявленные лицензии файла — из белого списка). Длина файла: 256 Б ≤ len ≤ 100 КБ
(микробные выбрасываются, длинные усекаются).

Реестр источников (`stack.py:SOURCES`) и порядок перебора `--source auto`:

| Источник | Гейт | Роль |
|---|---|---|
| `bigcode/the-stack-v2-dedup` | gated:auto | приоритет ADR-021; конфиги по языкам ровно нужные (`Python`/`Rust`/`Go`/`JavaScript`/`Shell`) |
| `bigcode/the-stack-dedup` | gated:auto | фолбэк v1: контент и поля `content`/`lang`/`license`/`size` в строке |
| `bigcode/the-stack-smol-xl` | нет | публичный срез Stack v1 по языкам — проверка пайплайна и приёмка, объёма на 3B нет |
| `codeparrot/codeparrot-clean` | нет | публичный python-корпус (лицензия и размер на файл) |
| `common-pile/stackv2` | нет | публичный Stack v2 с контентом, но языки — детектированные форматы (CSV/Futhark), объявленных языков в потоке почти нет |

Фактическая доступность и доля проходящих записей измеряются командой
`sources` (это и есть свидетельство для решения «v2 или v1»), а не берутся из
описания датасета.
