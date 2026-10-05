# Журнал публикаций HF (ADR-030)

Каждая строка — upload артефакта Axiom во внешний канал (репо/коллекция под Rob1234567, private по умолчанию). Канон артефактов — на `~/gb10-shared` (C-032); HF-копия не канон.

| Дата (UTC+3) | Репо | Тип | Источник (канон) | Файлы / sha256-манифест | Коммит контура | Примечание |
|---|---|---|---|---|---|---|
| 05.10.2026 ~14:3x | `Rob1234567/axiom-pilot-l3full-step300` | model (**public** с 05.10 ~18:1x, решение владельца; provenance подтверждён манифестами: W=fineweb-edu, C=codeparrot-clean, Q=mix thereof — приватного домена в претрейне нет) | `~/gb10-shared/axiom-run/pilot-l3full-endurance/checkpoints/step-00000300` (6.9 ГБ, orbax: params+opt_state+meta) | 60 файлов (58 + README + UPLOAD_MANIFEST), sha256 в `UPLOAD_MANIFEST.json` (репо) | `9a38cdb` (ADR-030) | Первый артефакт: база полного SFT; wow-карточка (история + facts.png + loss-curve.png) — коммит `bc683f86`; public-ревизия `73bcb6c3` |

Коллекция: `Rob1234567/axiom-6ac384a6e1b090371595ce90` (private; slug с суффиксом — поведение HF; переименование при переводе в public).

Правила: публикация — только по вердикту приёмки стадии; private→public — решение владельца; каждый upload — строка здесь до/сразу после upload.

Решение владельца 05.10: staging-копия чекпойнта на ПК (`/home/roman/hf-staging/axiom-pilot-l3full-step300/`, 6.9 ГБ) — **оставить** (страховка перепубликации; канон по-прежнему на стенде, C-032).
