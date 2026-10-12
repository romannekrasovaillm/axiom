"""TEMP diagnostic (deleted before commit): full single-chunk adjoint, S != 0, dS != 0.

Reference-style (fp32 direct) vs emulation-style (declared storages) chunk
adjoint on identical inputs, piece by piece to the final cotangents.
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
    dSh = jr.normal(jr.PRNGKey(51), (dk, dv))
    Sh = S[0] * 100
    qh = q.astype(f32); kh = k.astype(f32); vh = v.astype(f32)
    ah = alpha.astype(f32)
    beta = jnp.sum(beta32.astype(f32), axis=1) * (1.0 / 32.0)
    doh = do.astype(f32)
    ar = jnp.arange(C)
    tril = (ar[:, None] >= ar[None, :]).astype(f32)
    strict = (ar[:, None] > ar[None, :]).astype(f32)

    # ---------------- reference side (fp32, direct E) ----------------------
    la = jnp.log(ah)
    G = jnp.cumsum(la, axis=0)
    E = jnp.exp(G[:, None, :] - G[None, :, :])
    gamma = jnp.exp(G)
    aqk = tril * jnp.einsum("cid,cd,id->ci", E, qh, kh)
    L = strict * jnp.einsum("cid,cd,id->ci", E, kh * beta[:, None], kh)
    xw = gamma * kh * beta[:, None]
    vw = vh * beta[:, None]
    SM = jnp.linalg.inv(jnp.eye(C, dtype=f32) + L)
    W = SM @ xw; U = SM @ vw
    vt = U - W @ Sh
    gc = jnp.exp(jnp.sum(la, axis=0))
    lam = jnp.exp(jnp.sum(la, axis=0)[None, :] - G)
    y = lam * kh
    dS_acc = gc[:, None] * dSh
    dgc = jnp.sum(Sh * dSh, axis=1)
    dY_r = vt @ dSh.T
    dvt_r = y @ dSh + aqk.T @ doh
    daqk = (doh @ vt.T) * tril
    dS_acc = dS_acc + (gamma * qh).T @ doh
    dgq = doh @ Sh.T
    dgamma = dgq * qh
    dqh_r = dgq * gamma
    dU_r = dvt_r; dW_r = -(dvt_r @ Sh.T)
    dS_acc = dS_acc - W.T @ dvt_r
    Yw = SM.T @ dW_r; Yu = SM.T @ dU_r
    dL = -(Yw @ W.T + Yu @ U.T) * strict
    dgamma = dgamma + Yw * (kh * beta[:, None])
    dkh_r = Yw * (gamma * beta[:, None])
    dvh = Yu * beta[:, None]
    db = jnp.sum(Yw * gamma * kh, axis=1) + jnp.sum(Yu * vh, axis=1)
    dr_a_r = jnp.einsum("ci,cid,id->cd", daqk, E, kh)
    dc_a_r = jnp.einsum("ci,cid,cd->id", daqk, E, qh)
    dqh_r = dqh_r + dr_a_r
    dkh_r = dkh_r + dc_a_r
    dr_k_r = jnp.einsum("ci,cid,id->cd", dL, E, kh)
    dc_k_r = jnp.einsum("ci,cid,cd->id", dL, E, kh * beta[:, None])
    dkh_r = dkh_r + dr_k_r * beta[:, None] + dc_k_r
    db = db + jnp.sum(dr_k_r * kh, axis=1)
    dG_r = (dr_a_r * qh - dc_a_r * kh) + (dr_k_r * (kh * beta[:, None]) - dc_k_r * kh)
    dG_r = dG_r + dgamma * gamma
    dlam_r = dY_r * kh
    dkh_r = dkh_r + dY_r * lam
    dz_r = dlam_r * lam
    dG_r = dG_r - dz_r
    dsum_r = jnp.sum(dz_r, axis=0)
    dla = (tril.T @ dG_r) + jnp.broadcast_to((dsum_r + gc * dgc)[None, :], (C, dk))
    dalpha_r = dla / ah
    db_r = db / 32.0

    # ---------------- emu side (declared storages) --------------------------
    qv = qh.astype(bf); kv = kh.astype(bf); vv = vh.astype(bf)
    dov = doh.astype(bf)
    dsh = dSh.astype(bf).astype(f32)
    Sbf = Sh.astype(bf)
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

    kb = (kv.astype(f32) * beta[:, None]).astype(bf)
    akk = scores(kb, kv, strict)
    aqk_e = scores(qv, kv, tril)
    gam = jnp.exp(Ge).astype(bf).astype(f32)
    Xt = (gam * kb.astype(f32)).astype(bf).astype(f32).T
    Yt = (vv.astype(f32) * beta[:, None]).astype(bf).astype(f32).T
    pw = akk.astype(bf); pf = pw.astype(f32)
    Xt = (Xt - Xt @ pf.T).astype(bf).astype(f32)
    Yt = (Yt - Yt @ pf.T).astype(bf).astype(f32)
    for _ in range(4):
        pw = (pf @ pf).astype(bf); pf = pw.astype(f32)
        Xt = (Xt + Xt @ pf.T).astype(bf).astype(f32)
        Yt = (Yt + Yt @ pf.T).astype(bf).astype(f32)
    ws_t = Sbf.T.astype(f32) @ Xt.astype(f32)
    vt_t = (Yt - ws_t).astype(bf)
    g_e = jnp.exp(jnp.sum(la, axis=0)).astype(bf).astype(f32)
    ones = jnp.ones((C, C), f32)
    glast = ones @ hi.astype(f32) + ones @ lo.astype(f32)
    lam_e = jnp.exp(jnp.minimum(glast - Ge, 0.0))
    dgc_e = jnp.sum(Sbf.astype(f32) * dsh, axis=1)
    gq = (gam * qv.astype(f32)).astype(bf)
    acc1 = g_e[:, None] * dsh + gq.astype(f32).T @ dov.astype(f32)
    daqk_e = ((dov.astype(f32) @ vt_t.astype(f32)) * tril).astype(bf).astype(f32)
    y_bf = (lam_e * kv.astype(f32)).astype(bf)
    dvt = y_bf.astype(f32) @ dsh + aqk_e.astype(f32).T @ dov.astype(f32)
    dvt_op = dvt.astype(bf).astype(f32)
    dgq = dov.astype(f32) @ Sbf.T.astype(f32)
    dr_a = jnp.zeros((C, dk), f32); dc_a = jnp.zeros((C, dk), f32)
    for p in range(0, C, tile):
        rm = ((ar[:, None] >= p) & (ar[:, None] < p + tile)).astype(f32)
        dT = (daqk_e * rm).astype(bf).astype(f32)
        anchor = jnp.sum(Ge * (ar == p)[:, None].astype(f32), axis=0)
        rf = jnp.exp(jnp.minimum(Ge - anchor[None, :], 0.0))
        cf = jnp.exp(anchor[None, :] - Ge)
        dr_a = dr_a + rf * (dT @ (kv.astype(f32) * cf).astype(bf).astype(f32))
        dc_a = dc_a + cf * (dT.T @ (qv.astype(f32) * rf).astype(bf).astype(f32))
    dqh = (dgq * gam + dr_a).astype(bf)
    dkh = dc_a
    dG_e = (dr_a * qv.astype(f32) - dc_a * kv.astype(f32)
            + (dgq * qv.astype(f32)) * gam)
    dyv = vt_t.astype(f32).T @ dsh.T
    acc1_bf = acc1.astype(bf).astype(f32)
    acc2 = acc1_bf - Xt.astype(f32) @ dvt_op
    dsh_out = acc2.astype(bf).astype(f32)
    smt = jnp.eye(C, dtype=f32)
    smt = (smt - akk.astype(bf).astype(f32).T @ smt).astype(bf).astype(f32)
    pf2 = akk.astype(bf).astype(f32)
    for _ in range(4):
        pf2 = (pf2 @ pf2).astype(bf).astype(f32)
        smt = (smt + pf2.T @ smt).astype(bf).astype(f32)
    yuv = (smt @ dvt_op).astype(bf).astype(f32)
    ywv = (-(yuv @ Sbf.T.astype(f32))).astype(bf).astype(f32)
    dlv = (-(yuv @ vt_t.astype(f32)) * strict).astype(bf).astype(f32)
    kbf = kv.astype(f32) * beta[:, None]
    dG2 = dG_e + (ywv * kbf) * gam
    dkh2 = dkh + ywv * (gam * jnp.broadcast_to(beta[:, None], (C, dk)))
    dbv = (jnp.sum(ywv * gam * kv.astype(f32), axis=1)
           + jnp.sum(yuv * vv.astype(f32), axis=1))
    dvh_e = (yuv * beta[:, None]).astype(bf)
    dr_k = jnp.zeros((C, dk), f32); dc_k = jnp.zeros((C, dk), f32)
    for p in range(0, C, tile):
        rm = ((ar[:, None] >= p) & (ar[:, None] < p + tile)).astype(f32)
        dT = (dlv * rm).astype(bf).astype(f32)
        anchor = jnp.sum(Ge * (ar == p)[:, None].astype(f32), axis=0)
        rf = jnp.exp(jnp.minimum(Ge - anchor[None, :], 0.0))
        cf = jnp.exp(anchor[None, :] - Ge)
        dr_k = dr_k + rf * (dT @ (kv.astype(f32) * cf).astype(bf).astype(f32))
        dc_k = dc_k + cf * (dT.T @ (kbf * rf).astype(bf).astype(f32))
    dlam_e = dyv * kv.astype(f32)
    dz = dlam_e * lam_e
    dkh2 = dkh2 + dr_k * beta[:, None] + dc_k + dyv * lam_e
    dbv = dbv + jnp.sum(dr_k * kv.astype(f32), axis=1)
    dG2 = dG2 + dr_k * kbf - dc_k * kv.astype(f32) - dz
    dsum_e = jnp.sum(dz, axis=0)
    gcv_c = g_e
    dgc_c = dgc_e  # the (C,dk)-reduce load of the bf16 bs_s store
    bc_b = jnp.broadcast_to(gcv_c * dgc_c + dsum_e, (C, dk))
    dla_e = tril.T @ dG2
    dalpha_e = (dla_e + bc_b) / ah

    print("pieces:")
    rep("lam", lam_e, lam)
    rep("vt", vt_t.astype(f32).T, vt)
    rep("dvt", dvt, dvt_r)
    rep("dgc", dgc_e, dgc)
    rep("Yw", ywv, Yw)
    rep("Yu", yuv, Yu)
    rep("dL", dlv, dL)
    rep("dr_a", dr_a, dr_a_r)
    rep("dG2", dG2, dG_r)
    rep("dyv", dyv, dY_r)
    rep("dz", dz, dz_r)
    rep("dsh_out", dsh_out, dS_acc)
    rep("dq", dqh.astype(f32), dqh_r)
    rep("dk", (dkh2).astype(bf).astype(f32), dkh_r)
    rep("dv", dvh_e.astype(f32), dvh)
    rep("dbeta", dbv / 32.0, db_r)
    rep("dc_a", dc_a, dc_a_r)
    rep("dr_k", dr_k, dr_k_r)
    rep("dc_k", dc_k, dc_k_r)

    # term-wise dk composition
    te = [("dc_a", dc_a, dc_a_r),
          ("Ywgb", ywv * (gam * jnp.broadcast_to(beta[:, None], (C, dk))), Yw * (gamma * beta[:, None])),
          ("dr_k b", dr_k * beta[:, None], dr_k_r * beta[:, None]),
          ("dc_k", dc_k, dc_k_r),
          ("dY lam", dyv * lam_e, dY_r * lam)]
    tot_e = sum(t[1] for t in te)
    tot_r = sum(t[2] for t in te)
    for nm, a, b in te:
        rep("  tile " + nm, a, b)
    rep("  sum", tot_e, tot_r)
    rep("  sum bf16", tot_e.astype(bf).astype(f32), tot_r)

    a = ywv * (gam * kbf); b = Yw * (gamma * beta[:, None])
    d = jnp.abs(a - b); ij = jnp.unravel_index(jnp.argmax(d), d.shape)
    i, j = int(ij[0]), int(ij[1])
    print("worst Ywgb entry", (i, j), "emu", float(a[i, j]), "ref", float(b[i, j]))
    print("  ywv", float(ywv[i, j]), "Yw", float(Yw[i, j]),
          "gam", float(gam[i, j]), "gamma", float(gamma[i, j]),
          "kbf", float(kbf[i, j]), "beta*", float(beta[i] if beta.ndim == 1 else beta[i, 0]))
    print("  yuv row max", float(jnp.max(jnp.abs(yuv[i]))),
          "Yu row max", float(jnp.max(jnp.abs(Yu[i]))))
    print("  smt vs SM.T:", float(jnp.max(jnp.abs(smt - SM.T))),
          "|SM.T| max", float(jnp.max(jnp.abs(SM.T))))
    rep("dalpha", dalpha_e, dalpha_r)


if __name__ == "__main__":
    main()
