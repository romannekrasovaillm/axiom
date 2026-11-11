# Дельта: axiom-manifest-v2

- Route: Standard (ADR-058, 3 триггера: api_contract_change, data_contract_change, significant_nfr)
- Created: 2026-11-11
- Решение: `docs/adr/ADR-058-manifest-progona-environment-v2-os-pobedy.md`, карточка `model/ADR-058-manifest-v2.md`

## Проблема

AD-1 (ось победы объявлена и одна) не проверялся механически: run manifest среды не нёс ни
`victory_axis`, ни `baseline`, а поле пиннинга снапшота заполнялось `sha256_text("stub-model")`
при отсутствии `--model-snapshot` — заглушка проходила как «запинненная модель». `victory_axis`
лежит в `docs/DECISION-PACKAGE.json#/choice/victory_axis` (не top-level), а C-013 проверял
подстроку. Заявление о превосходстве было непроверяемым, K3-калибровка не исполнялась.

## ADDED

- When манифест пишется (`env/main.py verify --manifest-out`), the loader shall отклонять
  манифест без `victory_axis`, `baseline.snapshot_sha256`, `model.base` и без совпадения оси
  с `docs/DECISION-PACKAGE.json#/choice/victory_axis` (`env/schemas.py::PINNING_PATHS`,
  `env/manifest.py::full_validate`).
- When модель прогона — заглушка (`model.is_stub: true`), the loader shall отклонять манифест
  как доказательство (`missing_pinning` → `model.is_stub`).
- When манифест имеет схему v1/без `schema_version`, the loader shall принимать его только при
  явном `allow_legacy=True`, и такой манифест доказательством не является.
- When гейт исполняется, the страж C-013 shall проверять путь и единственность оси в пакете
  (`tools/check_victory_axis.py --selftest`, 5 мутантов), C-055 — схему v2 и отклонение
  заглушки (`tools/check_manifest_v2.py --selftest`, 7 мутантов), C-056 — наличие
  K3-калибровки в коридоре [10%, 90%] со ссылкой из RL-precondition
  (`tools/check_rl_precondition.py`; сейчас known-red, ADR-056).

## MODIFIED

- `env/schemas.py`: `validate_manifest`/`missing_pinning` — схема v2, `victory_axis`,
  `baseline.*`, `model.is_stub`, `allow_legacy`; было — одна модель и v1-набор пиннинга.
- `env/manifest.py`: `build_manifest` требует `baseline_model`/`baseline_snapshot_sha256`,
  ось по умолчанию читается из пакета (`env/axis.py`); `full_validate`/`dump_manifest`
  сверяют ось с пакетом.
- `env/main.py`: удалён дефолт `sha256_text("stub-model")` и `--model-base default="stub"`;
  добавлены `--model-snapshot`, `--baseline-model`, `--baseline-snapshot`, `--victory-axis`,
  `--allow-stub` (fail-closed, exit 2, файл не пишется).
- `docs/specs/ENVIRONMENT-V1.md` §8: схема `environment-v1/manifest-v2` (было — без оси и
  baseline), правила пиннинга и legacy.
- `ARCHITECTURE-SPINE.md` AD-1, `model/AD-1-os-pobedy-odna.md`: путь `choice.victory_axis`,
  запрет заявлений вне оси, `PENDING-EVIDENCE` с двумя structural-стражами.
- `CONSTRAINTS.yaml`: C-013 (must_contain → structural `infra: false`), добавлены C-055
  (манифест v2, инфраструктурное) и C-056 (RL-precondition, known-red).
- Вызывающие: `tools/run_rl_smoke.py`, `tools/run_a4_pipeline.py`, `net/tests/test_13_integration.py`,
  `net/tests/test_21_rl_stage.py` — baseline обязателен, v2 по умолчанию, legacy явно.

## REMOVED

- `env/main.py`: фиктивный `sha256_text("stub-model")` как пиннинг снапшота. Замена —
  обязательный `--model-snapshot`; заглушка — только `--allow-stub` без записи манифеста.
- C-013: `must_contain` по подстроке `"victory_axis"` (заменён structural-стражем пути).

## План отката

`git reset --hard c7e7b4f` — архитектурная часть (ADR-058, спека, спайн, модель, правила)
остаётся; откат кода возвращает v1-манифест, а C-055/C-056 остаются объявленным known-red
долгом. Манифесты v1 не переписываются: старые артефакты `evidence/**` остаются как есть.

## Критерии приёмки

- [ ] `python3 tools/check_victory_axis.py --selftest` → 0; plain — PASS на пакете.
- [ ] `python3 tools/check_manifest_v2.py --selftest` → 0 (мутанты: нет оси, нет baseline, `is_stub`, расхождение оси — красные).
- [ ] `python3 tools/check_rl_precondition.py --selftest` → 0; plain → 1 с known-red сообщением.
- [ ] `schemas.missing_pinning({})` содержит `victory_axis`, `baseline.snapshot_sha256`, `model.is_stub`, `schema_version`.
- [ ] `env.main verify --manifest-out` без `--model-snapshot` → exit 2, файл не создан.
- [ ] `EXCLUDED_INFRA_RULES` = динамический набор (C-055/C-056 в исключаемых, C-013 — в вердикте); `pytest env/tests -q -k "r1_workspace_ruleset or manifest or schemas"` зелёный.
- [ ] Зафиксированный stub-baseline калибровки перегенерирован под новый реестр (или причина дрейфа зафиксирована отдельным ADR).
