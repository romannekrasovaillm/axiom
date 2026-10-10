# Кандидатная линия JAX 0.11.2: статус capability gate (10.10.2026)

Стенд GB10 (spark-44c3), окружение `/home/roman/venv-axiom-0112` (legacy `venv-axiom` 0.10.2 не тронут, ADR-052).

## Статусы (раздельно, только по evidence)

| Что | Статус | Доказательство |
|---|---|---|
| **JAX 0.11.2 GPU** | **DONE** | `jax 0.11.2 / jaxlib 0.11.2`, `jax-cuda13-plugin/pjrt 0.11.2`; `devices: [CudaDevice(id=0)]`; JIT-операция `bf16 256x256 @ 256x256 → 256.0` с `block_until_ready`; `pip check` — no broken requirements |
| **Mosaic ALU (non-MMA)** | **DONE** | `mosaic-0112-kernel/scripts/smoke.py` → **`correctness-pass`** (тесты `fill`, `iota`) на `cuda:0`, GB10 |
| **Mosaic MMA** | **present-unverified** | Публичный API найден: `plgpu.mma(acc, a, b)` — «Computes `acc + a @ b` synchronously using **Ampere MMA instructions**»; существуют layout'ы `Layout.MMA_ACC / MMA_LHS / MMA_RHS` (**на 0.10.2 их не было** — там капы давали FAIL). Smoke-кернел доходит до верификатора, но падает: `VerificationError: 'mosaic_gpu.mma' op operand #1 must be vector of A type … got vector<128x128xf32>` — `plgpu.load(..., layout=MMA_LHS(dtype))` при загрузке из GMEM не применяет layout (нужен путь через SMEM/оптимизированные передачи: `copy_gmem_to_smem` + `wait_gmem_to_smem`, как в апстрим-тесте `tests/pallas/mosaic_gpu_test.py::test_mma`) |
| Mosaic transforms (vmap/grad) | NOT RUN | — |

## Провенанс

- probe: `mosaic-0112-probe.json` (`jax 0.11.2`, `jax-cuda13-plugin 0.11.2`, устройства: GB10 CC **12.1**, 48 SM; `ptxas` не в PATH этого venv).
- ALU smoke: `mosaic-0112-alu-smoke.json` (`status: correctness-pass`).
- MMA smoke: `mosaic-0112-mma-smoke.json` (`status: blocked`, `error_type: VerificationError`), скрипт `mosaic-mma-smoke.py` (написан по эталону апстрим-теста, входы bf16).

## Что это значит для кампании

1. **Линия 0.11.2 рабочая** — Mosaic на CC12.1 компилирует и исполняет (ALU), то есть кандидат не BLOCKED по железу.
2. **MMA — не «отсутствует», а требует корректного кернела**: API и layout'ы есть; нужен путь загрузки через SMEM и проверка компиляции под sm_121. Это следующий шаг (см. задачу).
3. Перемер чисел кампании на 0.11.2 (ток/с, MFU, число `cuGraphLaunch`) ещё не делался — он впереди и только после parity-переноса (ADR-052, стадия 5–6).
