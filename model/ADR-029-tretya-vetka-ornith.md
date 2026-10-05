---
id: ADR-029
type: adr
title: "Третья ветка RL ornith — самоскаффолдинг (DeepReinforce/Ornith-1.5)"
status: "accepted"
affects: [CMP-006]
source: "docs/adr/ADR-029-tretya-vetka-ornith-samoscaffolding.md"
---

Одна политика — три роли (proposer / harness-generator / solver), три
reward-стрима в одном GRPO: R_task = V×D×N (V — жёсткий гейт двусторонней
валидации; D — frontier p*=0.2 по текущим роллаутам; N — новизна против
буфера), R_harness = C×F×H (H = анти-hacking контур), R_rollout = вердикт
сгенерированного харнесса. Инференс генерации пиннут (AD-4). Fail-closed:
3 раунда без валидных задач — ветка честно останавливается. Вердикт —
попарные МакНемары с Бонферрони (3 сравнения). Решение владельца 05.10
(«третья ветка сравнения»).
