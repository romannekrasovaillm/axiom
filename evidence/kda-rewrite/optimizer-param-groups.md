# Классификация параметров по группам оптимизатора (ADR-048)

- конфиг: `net/config.json` (vocab 160000, hidden 1536, 24 слоёв)
- режим: `legacy_muon_all_2d=False`
- источник предиката: `net.optimizer.classify_leaf` (тот же, что читают `init_state` и `make_step`)
- дерево: абстрактные формы (`jax.eval_shape`), модель не аллоцируется

| группа | листьев | параметров | имена |
|---|---:|---:|---|
| muon_matrix | 322 | 280078848 | W_a_down, W_a_up, W_beta, W_c, W_down, W_f, W_g, W_idx_k, … (+18) |
| muon_per_head | 75 | 158072832 | W_k, W_k_up, W_q, W_v, W_v_up |
| muon_batched | 138 | 325582848 | expert_d, expert_g, expert_u, shared_d, shared_g, shared_u |
| adamw_vector | 187 | 138385 | A, b_alpha, norm, norm1, norm2, norm_attn, norm_final, norm_in, … (+4) |
| adamw_embed | 1 | 245760000 | embedding |
| unclassified | 0 | 0 | — |

всего: листьев 723, параметров 1009632913, legacy_muon_all_2d=False

Честная граница: отчёт описывает **геометрию дерева и маршрутизацию**,
а не измеренное время или память шага.  Время шага и loss-динамика дают
прогоны на GB10 (`--ns-steps` / `--legacy-muon-all-2d` у раннера).
