# D-8 закрыт 01.10.2026: «молчаливый segfault» l3-full = RESOURCE_EXHAUSTED 973 ГиБ

Изоляция ступенями на GB10 (tools/d8_isolate.py, T мал/8192, JAX_PLATFORMS=cuda):
1. forward — OK 23,5 с; 2. loss — OK 29 с; 3. **value_and_grad на T=8192: RESOURCE_EXHAUSTED, аллокация 973,70 ГиБ** (граф без чекпойнтинга материализует пул 64×8192 полного графа).
2. Событие 29.09 («лог обрывается без traceback» после строки «цикл») — тот же OOM: процесс убивался до печати ресурсной ошибки. Не дефект sm_121/кернелов: KDA-проекция и полный граф на малом T компилируются чисто.
3. Грубый jax.checkpoint(loss_fn) НЕ лечит (980,71 ГиБ — чекпойнт целой функции не режет внутренние аллокации). Рабочее лекарство — послойная политика внутри compute_loss: реализована в net/train_loop.py (_checkpoint_policy), претрейн-луп обязателен с grad_checkpointing=true на l3-full (уже в ТЗ лупа).
4. Побочный факт: пиннинг бэкенда в net/config.py отдаёт 'rocm' на GB10 при пустом NET_JAX_BACKEND — workaround JAX_PLATFORMS=cuda; кандидат в микро-фикс.

Итог: D-8 из blockers → closed (misdiagnosed OOM). Претрейн-луп на l3-full разрешён к запуску при grad_checkpointing=true.
