"""TEMP diagnostic (deleted before commit): single-chunk reference vs emu pieces.

Both sides computed from the SAME single-chunk inputs, S = 0, dS = 0:
reference = the fp32 adjoint of kda_cc_fused_bwd.reference_bwd, re-derived
piece by piece; emu = the emulation's declared-numerics pieces.
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


def rep(nm, a, b):
    a = jnp.asarray(a, f32); b = jnp.asarray(b, f32)
    err = float(jnp.max(jnp.abs(a - b)))
    sc = float(jnp.max(jnp.abs(b)))
    print(f"  {nm}: ref_max={sc:.3e} err={err:.3e} rel={err / max(sc, 1e-30):.2e}")


def main():
    C, dk, dv, tile = 32, 128, 128, 16
    q, k, v, alpha, beta32, S = _bwd_inputs(C, dk, dv, 1, 0)
    do = jr.normal(jr.PRNGKey(50), (C, dv))
    qh = q.astype(f32); kh = k.astype(f32); vh = v.astype(f32)
    ah = alpha.astype(f32)
    beta = (jnp.sum(beta32.astype(f32), axis=1) * (1.0 / 32.0))[:, None]
    doh = do.astype(f32)
    Sh = jnp.zeros((dk, dv), f32)
    ar = jnp.arange(C)
    tril = (ar[:, None] >= ar[None, :]).astype(f32)
    strict = (ar[:, None] > ar[None, :]).astype(f32)

    # ---------------- reference side (fp32, direct E) ----------------------
    la = jnp.log(ah)
    G = jnp.cumsum(la, axis=0)
    E = jnp.exp(G[:, None, :] - G[None, :, :])
    gamma = jnp.exp(G)
    aqk = tril * jnp.einsum("cid,cd,id->ci", E, qh, kh)
    L = strict * jnp.einsum("cid,cd,id->ci", E, kh * beta, kh)
    xw = gamma * kh * beta
    vw = vh * beta
    SM = jnp.linalg.inv(jnp.eye(C, dtype=f32) + L)
    W = SM @ xw; U = SM @ vw
    vt = U - W @ Sh
    lam = jnp.exp(jnp.sum(la, axis=0)[None, :] - G)
    y = lam * kh
    dvt = aqk.T @ doh
    dU = dvt; dW = -(dvt @ Sh.T)
    Yw = SM.T @ dW; Yu = SM.T @ dU
    dL = -(Yw @ W.T + Yu @ U.T) * strict
    dr_a_r = jnp.einsum("ci,cid,id->cd", (doh @ vt.T) * tril, E, kh)
    dc_a_r = jnp.einsum("ci,cid,cd->id", (doh @ vt.T) * tril, E, qh)
    dr_k_r = jnp.einsum("ci,cid,id->cd", dL, E, kh)
    dc_k_r = jnp.einsum("ci,cid,cd->id", dL, E, kh * beta)
    dG_r = (dr_a_r * qh - dc_a_r * kh) + (dr_k_r * (kh * beta) - dc_k_r * kh)

    # ---------------- emu side (declared storages) --------------------------
    qv = qh.astype(bf); kv = kh.astype(bf); vv = vh.astype(bf)
    dov = doh.astype(bf)
    hi = la.astype(bf); lo = (la - hi.astype(f32)).astype(bf)
    Ge = tril @ hi.astype(f32) + tril @ lo.astype(f32)

    def scores(row, kk, mask):
        acc = jnp.zeros((C, C), f32)
        for p in range(0, C, tile):
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

    kb = (kv.astype(f32) * beta).astype(bf)
    akk = scores(kb, kv, strict)
    aqk_e = scores(qv, kv, tril)
    gam = jnp.exp(Ge).astype(bf).astype(f32)
    Xt = (gam * kb.astype(f32)).astype(bf).astype(f32).T
    Yt = (vv.astype(f32) * beta).astype(bf).astype(f32).T
    pw = akk.astype(bf); pf = pw.astype(f32)
    Xt = (Xt - Xt @ pf.T).astype(bf).astype(f32)
    Yt = (Yt - Yt @ pf.T).astype(bf).astype(f32)
    for _ in range(4):
        pw = (pf @ pf).astype(bf); pf = pw.astype(f32)
        Xt = (Xt + Xt @ pf.T).astype(bf).astype(f32)
        Yt = (Yt + Yt @ pf.T).astype(bf).astype(f32)
    Sbf = Sh.astype(bf)
    ws_t = Sbf.T.astype(f32) @ Xt.astype(f32)
    vt_t = (Yt - ws_t).astype(bf)
    g_e = jnp.exp(jnp.sum(la, axis=0)).astype(bf).astype(f32)
    ones = jnp.ones((C, C), f32)
    glast = ones @ hi.astype(f32) + ones @ lo.astype(f32)
    lam_e = jnp.exp(jnp.minimum(glast - Ge, 0.0))
    y_bf = (lam_e * kv.astype(f32)).astype(bf)
    dvt_e = y_bf.astype(f32) @ Sbf.astype(f32) + aqk_e.astype(f32).T @ dov.astype(f32)
    dvt_op = dvt_e.astype(bf).astype(f32)
    smt = jnp.eye(C, dtype=f32)
    smt = (smt - akk.astype(bf).astype(f32).T @ smt).astype(bf).astype(f32)
    pf2 = akk.astype(bf).astype(f32)
    for _ in range(4):
        pf2 = (pf2 @ pf2).astype(bf).astype(f32)
        smt = (smt + pf2.T @ smt).astype(bf).astype(f32)
    yuv = (smt @ dvt_op).astype(bf).astype(f32)
    ywv = (-(yuv @ Sbf.T.astype(f32))).astype(bf).astype(f32)
    dlv = (-(yuv @ vt_t.astype(f32)) * strict).astype(bf).astype(f32)
    daqk_e = ((dov.astype(f32) @ vt_t.astype(f32)) * tril).astype(bf).astype(f32)
    dr_a = jnp.zeros((C, dk), f32); dc_a = jnp.zeros((C, dk), f32)
    for p in range(0, C, tile):
        rm = ((ar[:, None] >= p) & (ar[:, None] < p + tile)).astype(f32)
        dT = (daqk_e * rm).astype(bf).astype(f32)
        anchor = jnp.sum(Ge * (ar == p)[:, None].astype(f32), axis=0)
        rf = jnp.exp(jnp.minimum(Ge - anchor[None, :], 0.0))
        cf = jnp.exp(anchor[None, :] - Ge)
        dr_a = dr_a + rf * (dT @ (kv.astype(f32) * cf).astype(bf).astype(f32))
        dc_a = dc_a + cf * (dT.T @ (qv.astype(f32) * rf).astype(bf).astype(f32))
    kbf = kv.astype(f32) * beta
    dr_k = jnp.zeros((C, dk), f32); dc_k = jnp.zeros((C, dk), f32)
    for p in range(0, C, tile):
        rm = ((ar[:, None] >= p) & (ar[:, None] < p + tile)).astype(f32)
        dT = (dlv * rm).astype(bf).astype(f32)
        anchor = jnp.sum(Ge * (ar == p)[:, None].astype(f32), axis=0)
        rf = jnp.exp(jnp.minimum(Ge - anchor[None, :], 0.0))
        cf = jnp.exp(anchor[None, :] - Ge)
        dr_k = dr_k + rf * (dT @ (kv.astype(f32) * cf).astype(bf).astype(f32))
        dc_k = dc_k + cf * (dT.T @ (kbf * rf).astype(bf).astype(f32))
    dG_e = dr_a * kv.astype(f32) - dc_a * kv.astype(f32)
    dG_e = (dr_a * qv.astype(f32) - dc_a * kv.astype(f32)) + (
        dr_k * kbf - dc_k * kv.astype(f32))

    print("pieces:")
    rep("aqk", aqk_e.astype(f32), aqk)
    rep("L", akk.astype(f32), L)
    rep("vt", vt_t.astype(f32).T, vt)
    rep("lam", lam_e, lam)
    rep("dvt", dvt_e, dvt)
    rep("Yu", yuv, Yu)
    rep("dL", dlv, dL)
    rep("dr_a", dr_a, dr_a_r)
    rep("dc_a", dc_a, dc_a_r)
    rep("dr_k", dr_k, dr_k_r)
    rep("dc_k", dc_k, dc_k_r)
    rep("dG", dG_e, dG_r)
    # dbeta-ish and dalpha terms
    dbv_r = (jnp.sum(Yw * gamma * kh, axis=1) + jnp.sum(Yu * vh, axis=1)
             + jnp.sum(dr_k_r * kh, axis=1))
    dbv_e = (jnp.sum(ywv * gam * kv.astype(f32), axis=1)
             + jnp.sum(yuv * vv.astype(f32), axis=1)
             + jnp.sum(dr_k * kv.astype(f32), axis=1))
    rep("dbv", dbv_e, dbv_r)
    dlam_r = (vt @ jnp.zeros((dk, dv))) * kh  # dY = vt @ dS^T = 0
    dz_r = dlam_r * lam
    dsum_r = jnp.sum(dz_r, axis=0)
    dla_r = tril.T @ dG_r
    dalpha_r = dla_r / ah
    dz_e = jnp.zeros_like(dz_r)
    dsum_e = jnp.sum(dz_e, axis=0)
    dla_e = tril.T @ dG_e
    dalpha_e = (dla_e + jnp.broadcast_to(dsum_e, (C, dk))) / ah
    rep("dla", dla_e, dla_r)
    rep("dalpha", dalpha_e, dalpha_r)


if __name__ == "__main__":
    main()
