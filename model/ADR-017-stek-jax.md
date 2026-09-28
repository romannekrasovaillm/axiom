---
id: ADR-017
type: adr
title: "Вычислительный стек остаётся JAX; Rust только точечно через FFI и на инференсе"
status: "accepted"
affects: [CMP-002, CMP-004]
source: "docs/adr/ADR-017-vychislitelnyy-stek-ostayotsya-jax-rust-tolko-tochechno-cherez-ffi-i-na-inferense.md"
---

Стек axiom остаётся JAX (подтверждение границы ADR-008). Полный перенос на Rust отклонён
(нет официального порта JAX/Rust, нет аналогов Pallas/Tunix). Rust — точечно: JAX FFI-ops
и инференс-сервинг. Триггеры пересмотра формализованы (p99 шага, tok/s на GB10, зрелость Rust-XLA).
Ратифицировано владельцем.
