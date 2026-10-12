"""TEMP diagnostic (deleted before commit): piecewise adjoint dump, one chunk.

Re-derives the emulation's per-chunk quantities next to the reference's on the
SAME recompute inputs (T = C = 32, S = 0, dS = 0), so chain/state terms are out
of the equation and only the chunk adjoint appears.
"""

import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

import jax.numpy as jnp  # noqa: E402
import jax.random as jr  # noqa: E402

from net.tests.test_kda_cc_fused_persist import _bwd_inputs  # noqa: E402

f32, bf = jnp.float32, jnp.bfloat16


def main():
    T, Cc, dk, dv, H, tile = 32, 32, 128, 128, 1, 16
    q, k, v, alpha, beta32, S = _bwd_inputs(T, dk, dv, H, 0)
    do = jr.normal(jr.PRNGKey(50), (T, H * dv))
    qv, kv, vv = (t.reshape(Cc, -1).astype(bf) for t in (q, k, v))
    ah = alpha.reshape(Cc, -1).astype(f32)
    beta = jnp.sum(beta32.reshape(Cc, -1).astype(f32), axis=1) * (1.0 / 32.0)
    dov = do.reshape(Cc, -1).astype(bf)
    ar = jnp.arange(Cc)
    tril = (ar[:, None] >= ar[None, :]).astype(f32)
    strict = (ar[:, None] > ar[None, :]).astype(f32)

    # reference-side recompute (fp32, direct E)
    la = jnp.log(ah)
    G = jnp.cumsum(la, axis=0)
    E = jnp.exp(G[:, None, :] - G[None, :, :])
    kh, qh = kv.astype(f32), qv.astype(f32)
    gamma = jnp.exp(G)
    aqk_ref = tril * jnp.einsum("cid,cd,id->ci", E, qh, kh)
    L_ref = strict * jnp.einsum("cid,cd,id->ci", E, kh * beta[:, None], kh)
    xw = gamma * kh * beta[:, None]
    vw = vv.astype(f32) * beta[:, None]
    SM = jnp.linalg.inv(jnp.eye(Cc, dtype=f32) + L_ref)
    W = SM @ xw
    U = SM @ vw
    Sh = jnp.zeros((dk, dv), f32)
    vt = U - W @ Sh
    lam_ref = jnp.exp(jnp.sum(la, axis=0)[None, :] - G)
    y = lam_ref * kh
    daqk = (dov.astype(f32) @ vt.T) * tril
    dr_a_ref = jnp.einsum("ci,cid,id->cd", daqk, E, kh)
    dc_a_ref = jnp.einsum("ci,cid,cd->id", daqk, E, qh)
    dL_ref = -(jnp.zeros((Cc, dk)) @ W.T
               + (SM.T @ jnp.zeros((Cc, dv))) @ U.T)  # dW = -(dvt Sh^T) = 0 here
    dvt_ref = y @ jnp.zeros((dk, dv)) + aqk_ref.T @ dov.astype(f32)

    # emulation-side recompute (bf16 storages, hi/lo cumsum, tile form)
    hi = la.astype(bf)
    lo = (la - hi.astype(f32)).astype(bf)
    Ge = tril @ hi.astype(f32) + tril @ lo.astype(f32)
    gam_e = jnp.exp(Ge).astype(bf).astype(f32)

    def scores(row, kk, mask):
        acc = jnp.zeros((Cc, Cc), f32)
        for p in range(0, Cc, tile):
            pre = (ar[None, :] <= p).astype(f32)
            anchor = pre @ hi.astype(f32) + pre @ lo.astype(f32)
            rowmask = ((ar >= p) & (ar < p + tile)).astype(f32)[:, None]
            rf = jnp.exp(jnp.minimum(Ge - anchor, 0.0))
            rq = (row.astype(f32) * (rf.astype(bf).astype(f32) * rowmask)).astype(bf)
            colmask = (ar < p + tile).astype(f32)[:, None]
            cf = jnp.exp(anchor - Ge)
            ck = (kk.astype(f32) * (cf * colmask).astype(bf).astype(f32)).astype(bf)
            acc = acc + rq.astype(f32) @ ck.astype(f32).T
        return acc * mask

    akk = scores((kv.astype(f32) * beta[:, None]).astype(bf), kv, strict)
    aqk_e = scores(qv, kv, tril)
    kb = (kv.astype(f32) * beta[:, None]).astype(bf)
    Xt = (gam_e * kb.astype(f32)).astype(bf).astype(f32).T
    Yt = (vv.astype(f32) * beta[:, None]).astype(bf).astype(f32).T
    pw = akk.astype(bf); pf = pw.astype(f32)
    Xt = (Xt - Xt @ pf.T).astype(bf).astype(f32)
    Yt = (Yt - Yt @ pf.T).astype(bf).astype(f32)
    for _ in range(4):
        pw = (pf @ pf).astype(bf); pf = pw.astype(f32)
        Xt = (Xt + Xt @ pf.T).astype(bf).astype(f32)
        Yt = (Yt + Yt @ pf.T).astype(bf).astype(f32)
    Sbf = jnp.zeros((dk, dv), bf)
    ws_t = Sbf.T.astype(f32) @ Xt.astype(f32)
    vt_t = (Yt - ws_t).astype(bf)
    g_e = jnp.exp(jnp.sum(la, axis=0)).astype(bf).astype(f32)
    ones = jnp.ones((Cc, Cc), f32)
    glast = ones @ hi.astype(f32) + ones @ lo.astype(f32)
    lam_e = jnp.exp(jnp.minimum(glast - Ge, 0.0))
    y_e = (lam_e.astype(bf).astype(f32) * kv.astype(f32)).astype(bf)
    dvt_e = (y_e.astype(f32) @ Sbf.astype(f32).T.T
             + aqk_e.astype(f32).T @ dov.astype(f32))
    daqk_e = ((dov.astype(f32) @ vt_t.astype(f32)) * tril).astype(bf).astype(f32)
    dr_a = jnp.zeros((Cc, dk), f32); dc_a = jnp.zeros((Cc, dk), f32)
    for p in range(0, Cc, tile):
        rm = ((ar[:, None] >= p) & (ar[:, None] < p + tile)).astype(f32)
        dT = (daqk_e * rm).astype(bf).astype(f32)
        anchor = jnp.sum(Ge * (ar == p)[:, None].astype(f32), axis=0)
        rf = jnp.exp(jnp.minimum(Ge - anchor[None, :], 0.0))
        cf = jnp.exp(anchor[None, :] - Ge)
        dr_a = dr_a + rf * (dT @ (kv.astype(f32) * cf).astype(bf).astype(f32))
        dc_a = dc_a + cf * (dT.T @ (qv.astype(f32) * rf).astype(bf).astype(f32))

    def rep(nm, a, b):
        a = jnp.asarray(a, f32); b = jnp.asarray(b, f32)
        print(f"{nm}: ref_max={float(jnp.max(jnp.abs(b))):.3e} "
              f"err={float(jnp.max(jnp.abs(a - b))):.3e}")

    rep("aqk", aqk_e.astype(f32), aqk_ref)
    rep("L", akk.astype(f32), L_ref)
    rep("vt", vt_t.astype(f32).T, vt)
    rep("lam", lam_e, lam_ref)
    rep("dvt", dvt_e, dvt_ref)
    rep("daqk", daqk_e, daqk)
    rep("dr_a", dr_a, dr_a_ref)
    rep("dc_a", dc_a, dc_a_ref)


if __name__ == "__main__":
    main()
