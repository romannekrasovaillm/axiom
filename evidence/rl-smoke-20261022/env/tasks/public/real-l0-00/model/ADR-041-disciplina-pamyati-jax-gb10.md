---
id: ADR-041
type: adr
title: "Дисциплина памяти JAX-прогонов на GB10: XLA_PYTHON_CLIENT_MEM_FRACTION обязателен, префлайт-гейт перед стартом (инцидент OOM 08.10)"
status: "accepted"
affects: [CMP-002, NFR-003]
source: "docs/adr/ADR-041-disciplina-pamyati-jax-progonov-na-gb10-xla-python-client-mem-fraction-obyazatelen-preflayt-geyt-pered-startom-incident-oom-08-10.md"
---

Любой JAX-прогон на GB10 обязан выставлять `XLA_PYTHON_CLIENT_MEM_FRACTION` (никогда дефолтные ~75%): дефолтная резервация на 128 ГБ унифицированной памяти вызывала OOM-каскады. Перед стартом обязателен префлайт-гейт `tools/jax_preflight.py`: проверка чужих compute-процессов на устройстве и запрет совмещения без явного разрешения владельца (текстовый файл allow|permit|co-locate в `~/gb10-shared/.locks`). Инцидент-источник: OOM 08.10.2026 — совмещённый прогон убил смоук, SFT владельца и llama-server. Гейт на маршрутах enforce только GB10; прочие рантаймы — SKIP (ADR-046).
