---
id: ADR-049
type: adr
title: "Политика ремaтериализации как рычаг backward: jax.checkpoint с policy вместо полного пересчёта слоя — 73% шага в backward"
status: "accepted"
affects: [CMP-002, NFR-003]
source: "docs/adr/ADR-049-politika-rematerializacii-kak-rychag-backward-jax-checkpoint-s-policy-vmesto-polnogo-perescheta-sloya-73-shaga-v-backward.md"
---

Полная раскладка шага получена (l3-full, `chunked_cc`, jit-оптимизатор, 20 шагов, сверка 100.4%):
`sec_backward` **14.698 с = 73%**, `sec_forward` 4.292 (21%), `sec_backopt` 1.223 (6%), `sec_loader` 0.001.
Внутри forward: KDA 1.453, CE 1.127, MLA 0.463, MoE 0.163. KPI — 407 ток/с.

Отключение grad-checkpointing запрещено контуром (память активаций; урок OOM), но текущий
`jax.checkpoint` работает **без политики** — пересчитывается всё (save-nothing). Решение:
параметризовать `remat_policy` (`none` — дефолт, побитово прежний; `dots_saveable`;
`dots_with_no_batch_dims_saveable`) в `net/model.py` и телах scan `net/kda.py`.
Паритет обязателен (политика меняет, *что* пересчитывается, не математику); память — жёсткий
ограничитель (ADR-041), «не влезло» фиксируется как неприменимость, а не подъёмом лимита.

Матрица GB10 (20 шагов, l3-full): `none` **407.2** ток/с (20.14 с), `dots_with_no_batch_dims_saveable`
**468.6** (17.49 с, лучшая), `dots_saveable` — OOM 106.94 ГиБ. Дефолт `none` сохранён; политика
включается явно (`--remat-policy`). Артефакты: `evidence/kda-rewrite/mfu-remat-*.jsonl`,
`remat-policy-smoke.json`. Продолжение — ADR-050 (кампания MFU-55, гейт G1).
