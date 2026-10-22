---
id: ADR-034
type: adr
title: "Мульти-GPU претрейна: data-parallel через jax.sharding; FSDP отклонён; MaxText отложен"
status: "accepted"
affects: [CMP-002, CMP-004]
source: "docs/adr/ADR-034-multi-gpu-dp-jax-sharding.md"
---

Батч шардится NamedSharding P("dp"), модель-реплика на каждом GPU
(1,01B: реплика ~16 ГБ ≪ 80 ГБ H100 — FSDP отклонён математикой,
окупается на 10B+), градиенты psum(axis_name="dp"), KPI — глобальные
токены. MoE без EP (реплика несёт всех экспертов). MaxText-миграция
(план ADR-008) отложена отдельным решением: триггеры — модель >8B,
TP/EP/CP, мультиузельность. Аренда 8×H100 — после посадки DP и
подтверждения KPI WY/UT (20B ≈ 7–21 ч счёта, $300–600).
Приёмка: dp_size=1 бит-в-бит = текущий путь; weak-scaling ≥1,8×.
Решение владельца 05.10 («DP через jax.sharding»).
