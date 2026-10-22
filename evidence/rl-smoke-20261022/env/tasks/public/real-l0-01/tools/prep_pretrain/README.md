# `prep_pretrain` — подготовка претрейн-датасета L3 (ADR-021)

Потоковая сборка шардов претрейн-микса: **W** (веб, FineWeb-Edu, ~17B токенов),
**C** (код, The Stack, ~3B токенов) и **Q** (decay/annealing, ~1B: сужёный
FineWeb-Edu + кодовые примеры с тестами).

Данные живут вне git: `~/gb10-shared/datasets/axiom-pretrain-l3/{W,C,Q}/`
(C-032/C-033). В репозиторий попадают только код, тесты и карточка.

## Модули

| Модуль | Роль |
|---|---|
| `common.py` | `ShardWriter` (ротация по размеру сжатого файла, потоковый sha256, `.part` + rename), `Manifest` (шарды + курсор resume), `BoundedHashSet` (LRU-дедуп), `approx_tokens`, источники (локальный jsonl, HuggingFace stream), общий прогон шарда |
| `fineweb.py` | шард W: `HuggingFaceFW/fineweb-edu`, конфиг `sample-100BT`; шард Q: сужёный фильтр того же источника + код с тестами, доля кода, near-dup дедуп против W/C |
| `stack.py` | шард C: реестр источников The Stack, фильтры языка/лицензии/длины, пофайловые метаданные |
| `build.py` | CLI: `prepare-w`, `prepare-c`, `prepare-q`, `probe`, `sources`, `verify-manifest` |

## Запуск

```bash
# проба на малом объёме (в /tmp, боевую загрузку не запускает)
python -m prep_pretrain.build probe --limit-mb 200 --shard-mb 100 --progress

# доступность источников кода и доля записей, проходящих фильтры
python -m prep_pretrain.build sources --report /tmp/sources.json

# боевые прогоны (запускает владелец: 17B, 3B и 1B — это десятки часов канала)
python -m prep_pretrain.build prepare-w --target-tokens 17e9 --progress
python -m prep_pretrain.build prepare-c --target-tokens 3e9 --source stack-dedup-v1 --progress
python -m prep_pretrain.build prepare-q --target-tokens 1e9 --progress

# проба шарда Q на 50 МБ выхода (без боевой загрузки; опора W/C — из каталога датасета)
python -m prep_pretrain.build prepare-q --target-tokens 1e9 --limit-mb 50 \
    --out /tmp/axiom-pretrain-probe/Q --shard-mb 50 --max-minutes 20 --no-prior-dedup

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

## Шард Q (decay/annealing)

Сужёный микс двух источников, уже покрытых шардами W и C (~1B токенов,
ADR-021; запуск — `prepare-q`):

* **веб-часть** — тот же FineWeb-Edu, но строже по качеству и длине:
  `--min-int-score 4` (по умолчанию; замер 30.09.2026 по W: score 4 — 14,3 %
  записей, score 3 — 85,7 %) и окно длины `--min-chars 800`…`--max-chars 50000`;
* **код-часть** — кодовые примеры **с тестами** из `codeparrot-clean`
  (тест-маркеры `def test_`, `unittest`, `pytest`; замер: 19,5 % записей
  источника), остальные фильтры — штатные фильтры шарда C;
* **доля кода** — `--code-share 0.15` от записанных approx-токенов. Поток
  смешивается :class:`MixGovernor`: он выравнивает **ожидаемые** токены
  источника (потянутые символы × `--web-yield`/`--code-yield`) — решение зависит
  только от позиций в исходных потоках, поэтому поток воспроизводим при resume.
  Фактическая доля измеряется на выходе (`mix` в отчёте, вместе с
  `measured_yield`); если она ушла от цели, отчёт даёт
  `mix.recommended_code_yield` для боевого прогона. Если источник кода исчерпан
  раньше цели, доля ниже целевой **не** из-за урожайности — отчёт это говорит,
  рекомендации не выдаёт;
* **дедуп против W/C** — опорный near-dup индекс (`PriorNearDupIndex`):
  MinHash-подписи из `axiom_ds.dedup` (точный дубль — sha256 нормализованного
  текста, near-dup — Jaccard ≥ порога). Опорой служат первые
  `--prior-max-records` (по умолчанию 50 000) записей шардов
  `~/gb10-shared/datasets/axiom-pretrain-l3/{W,C}`: полный W в память подписей не
  влезает, поэтому в отчёте это **окно**, а не «весь W» (`prior.reference_records`,
  `prior.truncated`). Проверяемый документ в индекс не добавляется — память не
  растёт с прогоном. Нет опорных шардов — прогон не стартует; отключение шага —
  только явным `--no-prior-dedup`, и в отчёте стоит `prior.enabled: false`;
* **запись**: `{"text": …, "meta": {component: "web"|"code", …}}` в шарды
  `Q-000NN.jsonl.zst`. У кодовой части в мете `tests` — найденные маркеры.

Отчёт несёт `rules` (границы веб- и код-фильтра, маркеры, опора) и `mix`
(доли, урожайности, исчерпание источников, рекомендация). Манифест
(`manifest-q.json`) хранит поток (`web`/`code`/доля/урожайности): смена любого из
этих полей в том же каталоге — отказ, а не тихая мешанина шардов; `--restart`
начинает заново.

Проба (без боевой загрузки, ~50 МБ выхода):

```bash
python -m prep_pretrain.build prepare-q --target-tokens 1e9 --limit-mb 50 \
    --out /tmp/axiom-pretrain-probe/Q --shard-mb 50 --max-minutes 20 --no-prior-dedup --progress
```
