"""TEMP diagnostic (deleted before commit): two-chunk attribution.

For both chunks of a T=64 emulation run:
* internal: tile dr/dc vs direct-E dr/dc on the SAME emulated inputs;
* external: the emulated pieces (aqk, L, vt, dr, dc) vs the fp32 reference
  pieces computed from the SAME chunk inputs (no chain).
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
    print(f"  {nm}: ref_max={sc:.3e} err={err:.3e}")


def main():
    T, Cc, dk, dv, H, tile = 64, 32, 128, 128, 1, 16
    q, k, v, alpha, beta32, S = _bwd_inputs(T, dk, dv, H, 0)
    do = jr.normal(jr.PRNGKey(50), (T, H * dv))
    n = T // Cc
    qc, kc, vc = (t.reshape(n, Cc, -1) for t in (q, k, v))
    ac = alpha.reshape(n, Cc, -1)
    bc = beta32.reshape(n, Cc, -1)
    doc = do.reshape(n, Cc, -1)
    ar = jnp.arange(Cc)
    tril = (ar[:, None] >= ar[None, :]).astype(f32)
    strict = (ar[:, None] > ar[None, :]).astype(f32)

    def emu_chunk(i, S_bf):
        kv = kc[i].astype(bf); qv = qc[i].astype(bf); vv = vc[i].astype(bf)
        ah = ac[i].astype(f32)
        beta = jnp.sum(bc[i].astype(f32), axis=1) * (1.0 / 32.0)
        dov = doc[i].astype(bf)
        la = jnp.log(ah)
        hi = la.astype(bf); lo = (la - hi.astype(f32)).astype(bf)
        G = tril @ hi.astype(f32) + tril @ lo.astype(f32)

        def scores(row, kk, mask):
            acc = jnp.zeros((Cc, Cc), f32)
            for p in range(0, Cc, tile):
                pre = (ar[None, :] <= p).astype(f32)
                anchor = pre @ hi.astype(f32) + pre @ lo.astype(f32)
                rowmask = ((ar >= p) & (ar < p + tile)).astype(f32)[:, None]
                rf = jnp.exp(jnp.minimum(G - anchor, 0.0))
                rq = (row.astype(f32) * (rf.astype(bf).astype(f32) * rowmask)).astype(bf)
                colmask = (ar < p + tile).astype(f32)[:, None]
                cf = jnp.exp(anchor - G)
                ck_v = (kk.astype(f32) * (cf * colmask).astype(bf).astype(f32)).astype(bf)
                acc = acc + rq.astype(f32) @ ck_v.astype(f32).T
            return acc * mask

        kb = (kv.astype(f32) * beta[:, None]).astype(bf)
        akk = scores(kb, kv, strict)
        aqk = scores(qv, kv, tril)
        gam = jnp.exp(G).astype(bf).astype(f32)
        Xt = (gam * kb.astype(f32)).astype(bf).astype(f32).T
        Yt = (vv.astype(f32) * beta[:, None]).astype(bf).astype(f32).T
        pw = akk.astype(bf); pf = pw.astype(f32)
        Xt = (Xt - Xt @ pf.T).astype(bf).astype(f32)
        Yt = (Yt - Yt @ pf.T).astype(bf).astype(f32)
        for _ in range(4):
            pw = (pf @ pf).astype(bf); pf = pw.astype(f32)
            Xt = (Xt + Xt @ pf.T).astype(bf).astype(f32)
            Yt = (Yt + Yt @ pf.T).astype(bf).astype(f32)
        Sbf = S_bf.astype(bf)
        ws_t = Sbf.T.astype(f32) @ Xt.astype(f32)
        vt_t = (Yt - ws_t).astype(bf)
        g = jnp.exp(jnp.sum(la, axis=0)).astype(bf).astype(f32)
        ones = jnp.ones((Cc, Cc), f32)
        glast = ones @ hi.astype(f32) + ones @ lo.astype(f32)
        lam = jnp.exp(jnp.minimum(glast - G, 0.0))
        y_a = (lam.astype(bf).astype(f32) * kv.astype(f32)).astype(bf)
        s_new = (Sbf.astype(f32) * g[:, None] + y_a.astype(f32).T @ vt_t.astype(f32).T)
        return dict(kv=kv, qv=qv, vv=vv, ah=ah, beta=beta, dov=dov, la=la, G=G,
                    hi=hi, lo=lo, kb=kb, akk=akk, aqk=aqk, gam=gam, Xt=Xt,
                    Yt=Yt, vt_t=vt_t, g=g, lam=lam, Sbf=Sbf, s_new=s_new)

    # forward chain (pass A)
    chain = []
    s_cur = jnp.zeros((dk, dv), f32)
    for i in range(n):
        chain.append(s_cur.copy())
        d = emu_chunk(i, s_cur)
        s_cur = d["s_new"].astype(bf).astype(f32)

    for i in range(n):
        print(f"--- chunk {i} ---")
        d = emu_chunk(i, chain[i])
        print("internal tile vs direct-E:")
        # direct-E (fp32) from the emulated recompute inputs
        E = jnp.exp(d["G"][:, None, :] - d["G"][None, :, :])
        kh, qh = d["kv"].astype(f32), d["qv"].astype(f32)
        daqk = ((d["dov"].astype(f32) @ d["vt_t"].astype(f32)) * tril).astype(bf).astype(f32)
        dr_ref = jnp.einsum("ci,cid,id->cd", daqk, E, kh)
        dc_ref = jnp.einsum("ci,cid,cd->id", daqk, E, qh)
        dr = jnp.zeros((Cc, dk), f32); dc = jnp.zeros((Cc, dk), f32)
        for p in range(0, Cc, tile):
            rm = ((ar[:, None] >= p) & (ar[:, None] < p + tile)).astype(f32)
            dT = (daqk * rm).astype(bf).astype(f32)
            anchor = jnp.sum(d["G"] * (ar == p)[:, None].astype(f32), axis=0)
            rf = jnp.exp(jnp.minimum(d["G"] - anchor[None, :], 0.0))
            cf = jnp.exp(anchor[None, :] - d["G"])
            dr = dr + rf * (dT @ (kh * cf).astype(bf).astype(f32))
            dc = dc + cf * (dT.T @ (qh * rf).astype(bf).astype(f32))
        rep("dr_a", dr, dr_ref)
        rep("dc_a", dc, dc_ref)
        # reference pieces from the same inputs (fp32 forward recompute)
        la = d["la"]
        Gr = jnp.cumsum(la, axis=0)
        Er = jnp.exp(Gr[:, None, :] - Gr[None, :, :])
        gamma = jnp.exp(Gr)
        kh, qh, vv = d["kv"].astype(f32), d["qv"].astype(f32), d["vv"].astype(f32)
        beta = d["beta"]
        aqk_r = tril * jnp.einsum("cid,cd,id->ci", Er, qh, kh)
        L_r = strict * jnp.einsum("cid,cd,id->ci", Er, kh * beta[:, None], kh)
        xw = gamma * kh * beta[:, None]
        vw = vv * beta[:, None]
        SM = jnp.linalg.inv(jnp.eye(Cc, dtype=f32) + L_r)
        W = SM @ xw; U = SM @ vw
        Sh = chain[i]
        vt = U - W @ Sh
        lam_r = jnp.exp(jnp.sum(la, axis=0)[None, :] - Gr)
        y = lam_r * kh
        print("external vs reference (same inputs):")
        rep("aqk", d["aqk"].astype(f32), aqk_r)
        rep("lam", d["lam"], lam_r)
        rep("vt", d["vt_t"].astype(f32).T, vt)
        rep("G", d["G"], Gr)


if __name__ == "__main__":
    main()
