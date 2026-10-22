# Карточка претрейн-корпуса L3 (финал, 01.10.2026)

Состав (претокенизация BPE 160K, hash `500f80237b3bd0fb…`, пакеты T=8192 uint32):

| Шард | Источник | Токенов | Бинов |
|---|---|---|---|
| W | FineWeb-Edu sample-100BT | 13,87B | 52 |
| C | codeparrot-clean (python, лицензии per-file) | 3,12B | 6 |
| Q | сужёный W + 15% код-с-тестами (decay-фаза) | 0,85B | 5 |
| **Σ** | | **17,84B** | **63** |

Пайплайн: `tools/prep_pretrain/` (стрим → дедуп → фильтры → шарды 500 МБ → sha256) → `tools/bpe_train.py` → `tools/pretokenize.py`.
Верификации: манифесты шардов `bad: []` (W, C), токен-манифесты `tokens/{W,C,Q}/`.
Луп: `net/train_loop.py` (детерминизм 0/138 GPU под ADR-013; смоук на бинах 30 шагов, `evidence/pretrain-smoke-bins/`).
Смета: `evidence/budget/pretreain-l3.json` — 116 ч / $240, лимит $350 (H800, калибровка 343 TFLOP/s, k=1.33 checkpoint-recompute).
Приватное в претрейне отсутствует (AD-6): состав 100% публичный (FineWeb-Edu, codeparrot, permissive-лицензии).
