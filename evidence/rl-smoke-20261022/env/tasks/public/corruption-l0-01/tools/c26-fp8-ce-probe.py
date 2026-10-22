#!/usr/bin/env python3
"""c26-fp8-ce-probe.py — численность FP8 (e4m3) для CE/lm_head на реальной форме.

Сравнивает CE-голову (features (N,1536) @ embedding.T (160000)) в bf16 и fp8:
  * forward loss и backward dW/dfeatures;
  * fp8 без скейла и с per-tensor скейлом (макс. по тензору -> ±448).
Это предусловие FP8-пути к 55 % MFU (Am.38): проба 8192^3 дала fp8 131.7 TFLOPS,
но lm_head-форма в bf16 патологически медленная (1.85 TFLOPS в изоляции; в модели
~18) — fp8 даёт ×4.7 на реальном пути. Здесь проверяем, выдерживает ли численность
допуск (loss |d|<=2e-2 scale-relative).

Запуск на стенде (GPU): /home/roman/venv-axiom-0112/bin/python tools/c26-fp8-ce-probe.py
"""
import jax
import jax.numpy as jnp

N, K, V = 8192, 1536, 160000
FP8_MAX = 448.0


def q(x, scale):
    return jnp.clip(x / scale, -FP8_MAX, FP8_MAX).astype(jnp.float8_e4m3fn)


def ce_bf16(feat, emb, tgt):
    logits = jnp.matmul(feat.astype(jnp.bfloat16), emb.astype(jnp.bfloat16).T,
                        preferred_element_type=jnp.float32)
    lse = jax.nn.logsumexp(logits, axis=-1)
    nll = lse - jnp.take_along_axis(logits, tgt[:, None], axis=-1)[:, 0]
    return nll.mean()


def ce_fp8(feat, emb, tgt, scale_f=1.0, scale_e=1.0):
    a = q(feat, scale_f)
    b = q(emb.T, scale_e)
    logits = jnp.matmul(a, b, preferred_element_type=jnp.float32) * (scale_f * scale_e)
    lse = jax.nn.logsumexp(logits, axis=-1)
    nll = lse - jnp.take_along_axis(logits, tgt[:, None], axis=-1)[:, 0]
    return nll.mean()


def main():
    key = jax.random.PRNGKey(0)
    feat = jax.random.normal(key, (N, K), dtype=jnp.float32)
    emb = jax.random.normal(jax.random.PRNGKey(1), (V, K), dtype=jnp.float32) * 0.02
    tgt = jax.random.randint(jax.random.PRNGKey(2), (N,), 0, V)
    loss_ref = ce_bf16(feat, emb, tgt)
    g_ref = jax.grad(ce_bf16, argnums=(0, 1))(feat, emb, tgt)
    print(f"loss bf16 {float(loss_ref):.6f}")
    sf = float(jnp.max(jnp.abs(feat))) / FP8_MAX
    se = float(jnp.max(jnp.abs(emb))) / FP8_MAX
    for name, kw in [("fp8 raw", {}),
                     ("fp8 scaled", {"scale_f": sf, "scale_e": se})]:
        try:
            l = ce_fp8(feat, emb, tgt, **kw)
            gl, ge = jax.grad(ce_fp8, argnums=(0, 1))(feat, emb, tgt, **kw)
            dl = abs(float(l) - float(loss_ref))
            rel_l = dl / max(1e-9, abs(float(loss_ref)))
            rel_f = float(jnp.max(jnp.abs(gl - g_ref[0])) / jnp.max(jnp.abs(g_ref[0])))
            rel_e = float(jnp.max(jnp.abs(ge - g_ref[1])) / jnp.max(jnp.abs(g_ref[1])))
            print(f"{name:10s} loss {float(l):.6f} d={dl:.3e} rel={rel_l:.2e} | "
                  f"dFeat rel={rel_f:.2e} dEmb rel={rel_e:.2e}")
        except Exception as exc:  # noqa: BLE001
            print(f"{name:10s} FAIL {type(exc).__name__}: {str(exc)[:140]}")


if __name__ == "__main__":
    main()
