# Задача для кодового харнесса

Реализовать контракт run manifest v2 (ADR-058) в кейсе axiom: ось победы и baseline в манифесте, запрет фиктивного пиннинга, три механических стража.

## Контекст
AD-1 («ось победы объявлена и одна») не проверялся механически. Найденные дефекты: (а) `env/main.py:76` писал `sha256_text("stub-model")` в поле пиннинга снапшота при отсутствии `--model-snapshot`, `--model-base` имел default="stub" — заглушка проходила как «запинненная модель»; (б) в схеме §8 манифеста не было ни `victory_axis`, ни baseline; (в) `victory_axis` лежит в `docs/DECISION-PACKAGE.json#/choice/victory_axis`, а не на верхнем уровне; (г) C-013 проверял подстроку.

ADR-058 уже принят (статус Proposed), спека `docs/specs/ENVIRONMENT-V1.md` §8 обновлена, в CONSTRAINTS.yaml добавлены C-055 (манифест v2, structural) и C-056 (RL-precondition, known-red), C-013 переведён в structural. Твоя задача — реализовать код и стражей так, чтобы эти правила стали исполняемыми.

## Что сделать

### 1. `env/axis.py` (новый модуль)
- `AXIS_PATH = ("choice", "victory_axis")`.
- `class AxisError(ValueError)`.
- `load_axis(package_path: Path) -> str`: читает JSON пакета; возвращает непустую строку по пути `choice.victory_axis`; ошибки (файл/ключ/пусто/не строка) — `AxisError` с путём.
- `normalize_axis(s: str) -> str` — нормализация пробелов (`" ".join(s.split())`); `axis_matches(a, b) -> bool` — по нормализованным строкам.
- `has_top_level_axis(doc: dict) -> bool` — есть ли ключ `victory_axis` на верхнем уровне (это ошибка двусмысленности).

### 2. `env/schemas.py`
- `MANIFEST_SCHEMA = "environment-v1/manifest-v2"`.
- `PINNING_PATHS` (v2): `schema_version`, `gates_version.{arch_ml_build,constraints_sha256,hidden_constraints_sha256,excluded_infra_rules}`, `workspace_sha256`, `victory_axis`, `model.{snapshot_sha256,base}`, `baseline.{model,snapshot_sha256}`.
- `PINNING_PATHS_LEGACY` — прежний набор (без schema_version/victory_axis/baseline/model.base).
- `validate_manifest(manifest, *, allow_legacy=False)`: требует `schema_version == MANIFEST_SCHEMA` (кроме allow_legacy и отсутствующего поля), `victory_axis` непустая строка, `model.is_stub` bool, `baseline` объект с `model` (непустая строка) и `snapshot_sha256` (hex), остальные прежние проверки сохраняются.
- `missing_pinning(manifest, *, allow_legacy=False) -> list[str]`: при `allow_legacy=True` и схеме не v2 — набор LEGACY; иначе — набор v2 и при несовпадении схемы добавить `"schema_version"`. При `model.is_stub is True` добавить `"model.is_stub"`. При `allow_legacy=True` и v2 — набор v2.

### 3. `env/manifest.py`
- `build_manifest(...)` получает keyword-аргументы `baseline_model: str`, `baseline_snapshot_sha256: str`, `victory_axis: Optional[str] = None`, `model_is_stub: bool = False`. Пишет `schema_version`, `victory_axis` (если None — `axis.load_axis(DEFAULT_PACKAGE_PATH)`), `baseline`, `model.is_stub`. `DEFAULT_PACKAGE_PATH = Path(__file__).resolve().parent.parent / "docs" / "DECISION-PACKAGE.json"`.
- `full_validate(manifest, *, package_path: Optional[Path] = None, allow_legacy: bool = False)`: структурная валидация + `missing_pinning` + (для v2 и заданного `package_path`) сверка оси: нормализованное значение манифеста обязано совпасть с `axis.load_axis(package_path)`; иначе ошибка `"victory_axis: не совпадает с пакетом решения (ADR-058)"`.
- `dump_manifest(manifest, path, *, package_path=None, allow_legacy=False)` — тот же набор проверок, `ManifestError` при ошибках.

### 4. `env/main.py`
- Убрать `sha256_text("stub-model")` и `default="stub"` у `--model-base`.
- `--model-snapshot`, `--model-base`, `--baseline-model`, `--baseline-snapshot`, `--victory-axis`, `--allow-stub`; для `verify --manifest-out` отсутствие `--model-snapshot`/`--baseline-*` — `SystemExit` с понятным сообщением и кодом 2, файл не пишется.
- `--allow-stub` пишет `is_stub=True`; `dump_manifest` для такого манифеста обязан упасть (`ManifestError`); CLI перехватывает и завершается кодом 2 с сообщением «манифест заглушки не является доказательством (ADR-058)».
- `dump_manifest` вызывать с `package_path=CASE_DIR/"docs"/"DECISION-PACKAGE.json"`.

### 5. `tools/check_victory_axis.py` (новый; C-013)
- `--root PATH` (default — корень репозитория от `__file__`), `--selftest`.
- Проверки: пакет `docs/DECISION-PACKAGE.json` существует и JSON-объект; `choice.victory_axis` непустая строка; top-level `victory_axis` отсутствует; `ARCHITECTURE-SPINE.md` содержит `choice.victory_axis`; `model/AD-1-os-pobedy-odna.md` содержит `choice.victory_axis`. Код 0/1, диагностика по каждой проверке.
- `--selftest`: во временном каталоге 4 фикстуры — валидная (зелёная), без поля (красная), с top-level дублем (красная), с пустой осью (красная). Код 0 только если ожидания совпали.

### 6. `tools/check_manifest_v2.py` (новый; C-055)
- `--selftest`, `--check PATH` (опционально).
- Selftest: собрать валидный v2 через `build_manifest` (фиктивные `snapshot_sha256` = "a"*64, baseline "kimi-k3" + "b"*64) и проверить: `full_validate` пусто; мутанты красные — удалить `victory_axis`; удалить `baseline.snapshot_sha256`; `model.is_stub=True`; `victory_axis` не совпала с пакетом; v1 без `allow_legacy` (красный), v1 с `allow_legacy=True` (зелёный по legacy-набору).

### 7. `tools/check_rl_precondition.py` (новый; C-056)
- `--evidence-dir` (default `evidence`), `--precondition` (default `evidence/rl-precondition.json`), `--selftest`.
- Ищет `calibration-*.json`: отчёт годен, если `model`/`model_name` != "stub", `pass_rate` (в `aggregate.pass_rate` или верхнем уровне) ∈ [0.10, 0.90], `ready` != false, пиннинг непуст; требует ссылку из precondition-файла (`calibration_report` = путь отчёта).
- Обычный прогон сейчас ОБЯЗАН быть красным (код 1) с классифицированным сообщением «K3-калибровка не найдена: known-red до прогона на GB10 (ADR-058)» — это ожидаемое состояние, а не баг.
- `--selftest`: фикстуры (валидный отчёт + ссылка → 0; stub → 1; pass_rate 0.95 → 1; без ссылки → 1).

### 8. Вызывающие
- `tools/run_rl_smoke.py`: добавить `--baseline-model`, `--baseline-snapshot`, `--victory-axis` (опционален), прокинуть в `build_manifest`; при отсутствии baseline — понятная ошибка.
- `tools/run_a4_pipeline.py`: то же для `dump_manifest` (или явный `allow_legacy=True` с предупреждением, если артефакт исторический).
- `net/tests/test_13_integration.py`: обновить фикстуру манифеста под v2 (axis из пакета, baseline-снапшот).

### 9. Тесты
- Обновить `env/tests/test_schemas.py`, `env/tests/test_manifest.py`: `valid_manifest()` → v2; добавить `valid_manifest_v1()` и тесты на `allow_legacy`, `is_stub`, отсутствие `victory_axis`/`baseline`.
- Добавить `tools/tests/test_check_victory_axis.py`, `test_check_manifest_v2.py`, `test_check_rl_precondition.py` (только `--selftest` + негативные кейсы, без сети и без jax).

## Инварианты (нарушение = отказ приёмки)
- I1. `environment-v1/manifest-v2` — единственная доказательная схема; v1 валиден только при явном `allow_legacy=True` и доказательством не является.
- I2. Заглушка не доказательство: `is_stub=True` даёт ошибку пиннинга; фиктивный хеш снапшота (`sha256_text("stub-model")` и подобные) запрещён.
- I3. Ось копируется из `docs/DECISION-PACKAGE.json#/choice/victory_axis` и сверяется по нормализованной строке; расхождение — ошибка.
- I4. Без сети; без абсолютных домашних путей в коде и артефактах; без новых зависимостей.
- I5. Архитекторские артефакты не менять: `CONSTRAINTS.yaml`, `ARCHITECTURE-SPINE.md`, `docs/**` (включая ADR-058 и ENVIRONMENT-V1), `model/**`, `.arch-handoff/**`. Нужна правка — фиксируй в `open_questions`, не правь.
- I6. C-056 остаётся красным (known-red) — не «чинить» правило и не подкладывать фиктивный отчёт калибровки в `evidence/`.

## Критерии приёмки (машинные)
1. `python3 tools/check_victory_axis.py --selftest` → код 0.
2. `python3 tools/check_manifest_v2.py --selftest` → код 0.
3. `python3 tools/check_rl_precondition.py --selftest` → код 0; обычный `python3 tools/check_rl_precondition.py` → код 1 с сообщением про known-red.
4. `python3 -m pytest env/tests -q` — зелёный.
5. `python3 -m pytest tools/tests -q -k "victory_axis or manifest_v2 or rl_precondition"` — зелёный.
6. Негативный CLI: `python3 -m env.main verify --task <spec> --base <dir> --state <dir> --manifest-out /tmp/m.json` без `--model-snapshot` → код 2 и файл не создан.
7. `python3 -c "from env import schemas; m={}; assert 'victory_axis' in schemas.missing_pinning(m)"` и аналогично для `baseline.snapshot_sha256`, `model.is_stub`, `schema_version`.

Финализация: финальный git-коммит с сообщением `feat(env): manifest v2 — axis/baseline/is_stub + стражи C-013/C-055/C-056 (ADR-058)` и отчёт в `status`/`summary`/`evidence`/`open_questions` по JSON-контракту.

## План отката

Откат дельты: `git reset --hard bfe5c0f` (архитектурная часть ADR-058 зафиксирована в этом коммите и остаётся). Сигналы отката: (1) сломаны существующие тесты env/tests или net/tests/test_13_integration.py; (2) манифест v2 нельзя собрать без ручной правки схемы; (3) selftest-мутанты не краснеют (страж фиктивен). Владелец решения об откате — архитектор; частичный откат кода допустим без отката ADR (правила C-055/C-056 остаются красными и объявлены долгом).

## Финализация (обязательно)

Результат забирается из git, поэтому перед финальным ответом зафиксируй работу коммитом:

```bash
git add -A -- . ':!.arch-handoff'
git commit -m "<кратко: что реализовано>"
git status --short   # пусто, кроме .arch-handoff/
```

- Коммитится код и тесты; служебный каталог `.arch-handoff/` в коммит не входит.
- Работа без коммита считается невыполненной: оркестратор увидит её только через git log.

## Контракт результата

Финальный ответ обязан завершаться JSON-объектом (после него — ни символа):

```json
{"status": "complete|partial|blocked", "assumptions": [], "open_questions": [], "conflicts_with_prior_decisions": []}
```

- `status`: `complete` — выполнено полностью; `partial` — частично; `blocked` — заблокировано.
- `assumptions`: допущения, принятые при реализации.
- `open_questions`: вопросы к архитектору.
- `conflicts_with_prior_decisions`: расхождения с принятыми ранее решениями (ADR, spine).

Архитектурный контекст — `ARCHITECTURE.md`, ограничения — `CONSTRAINTS.yaml`, рубрика приёмки — `RUBRIC.yaml` (при наличии).

## Чеклист перед финальным ответом

- [ ] `SPEC.md` (контракты интерфейсов: входы/выходы, структуры данных, границы ошибок, критерии верификации) заполнен архитектором — сверь реализацию с ним; расхождения фиксируй в `conflicts_with_prior_decisions`, а не молчаливым отступлением.
