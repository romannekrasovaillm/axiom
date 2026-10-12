"""TEMP diagnostic (deleted before commit): stage-by-stage first inf/nan.

A verbatim copy of ``_emulate_bf16_bwd`` (the declared numerics of the
variant-B backward kernel) with a checker after every named stage.
"""

import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

import jax  # noqa: E402  (eager, no jit)
import jax.numpy as jnp  # noqa: E402
import jax.random as jr  # noqa: E402

from net.tests._tmp_probe_nan import make  # noqa: E402

FIRST = {"inf": None, "nan": None}
LOGS = []


def chk(stage, x, ctx):
    x = jnp.asarray(x, jnp.float32)
    has_nan = bool(jnp.any(jnp.isnan(x)))
    has_inf = bool(jnp.any(jnp.isinf(x)))
    if has_inf and FIRST["inf"] is None:
        FIRST["inf"] = (ctx, stage)
    if has_nan and FIRST["nan"] is None:
        FIRST["nan"] = (ctx, stage)
    return x


def maxabs(x):
    x = jnp.asarray(x, jnp.float32)
    return float(jnp.max(jnp.where(jnp.isfinite(x), jnp.abs(x), 0.0)))


def emulate(q, k, v, alpha, beta32, S, do, dS, *, tile=16, chunk=32):
    T = q.shape[0]
    Cc = int(chunk)
    n = T // Cc
    H, dk, dv = S.shape
    f32, bf = jnp.float32, jnp.bfloat16
    ar = jnp.arange(Cc)
    ridx, cidx = ar[:, None], ar[None, :]
    tril = (ridx >= cidx).astype(f32)
    strict = (ridx > cidx).astype(f32)
    steps = 0
    while (1 << steps) < Cc:
        steps += 1

    qc, kc, vc = q.reshape(n, Cc, -1), k.reshape(n, Cc, -1), v.reshape(n, Cc, -1)
    ac, bc = alpha.reshape(n, Cc, -1), beta32.reshape(n, Cc, -1)
    doc = do.reshape(n, Cc, -1)

    def forward_chunk(i):
        out = {"G": [], "gam": [], "aqk": [], "L": [], "WT": [], "UT": [],
               "vtT": [], "g": [], "lamA": [], "yA": [], "Snew": []}
        for h in range(H):
            ctx = (i, h)
            ko, vo, bo = h * dk, h * dv, h * 32
            kv = kc[i, :, ko:ko + dk].astype(bf)
            qv = qc[i, :, ko:ko + dk].astype(bf)
            vv = vc[i, :, vo:vo + dv].astype(bf)
            ah = ac[i, :, ko:ko + dk].astype(f32)
            beta = jnp.sum(bc[i, :, bo:bo + 32].astype(f32), axis=1) * (1.0 / 32.0)
            bcol = beta[:, None]
            la = jnp.log(ah)
            chk("la", la, ctx)
            hi = la.astype(bf)
            lo = (la - hi.astype(f32)).astype(bf)
            G = tril @ hi.astype(f32) + tril @ lo.astype(f32)
            chk("G", G, ctx)

            def scores(row, kk, mask):
                acc = jnp.zeros((Cc, Cc), f32)
                for p in range(0, Cc, tile):
                    pre = (cidx <= p).astype(f32)
                    anchor = pre @ hi.astype(f32) + pre @ lo.astype(f32)
                    rowmask = ((ar >= p) & (ar < p + tile)).astype(f32)[:, None]
                    rf = jnp.exp(jnp.minimum(G - anchor, 0.0))
                    rq = (row.astype(f32)
                          * (rf.astype(bf).astype(f32) * rowmask)).astype(bf)
                    colmask = (ar < p + tile).astype(f32)[:, None]
                    cf = jnp.exp(anchor - G)
                    ck = (kk.astype(f32)
                          * (cf * colmask).astype(bf).astype(f32)).astype(bf)
                    acc = acc + rq.astype(f32) @ ck.astype(f32).T
                return acc * mask

            akk = scores((kv.astype(f32) * bcol).astype(bf), kv, strict)
            chk("akk", akk, ctx)
            aqk = scores(qv, kv, tril)
            chk("aqk", aqk, ctx)
            gam_f = jnp.exp(G)
            chk("gam_fp32", gam_f, ctx)
            gam = gam_f.astype(bf).astype(f32)
            kb = (kv.astype(f32) * bcol).astype(bf)
            Xt = (gam * kb.astype(f32)).astype(bf).astype(f32).T
            Yt = (vv.astype(f32) * bcol).astype(bf).astype(f32).T
            pw = akk.astype(bf)
            pf = pw.astype(f32)
            Xt = (Xt - Xt @ pf.T).astype(bf).astype(f32)
            Yt = (Yt - Yt @ pf.T).astype(bf).astype(f32)
            for _ in range(steps - 1):
                pw = (pf @ pf).astype(bf)
                pf = pw.astype(f32)
                Xt = (Xt + Xt @ pf.T).astype(bf).astype(f32)
                Yt = (Yt + Yt @ pf.T).astype(bf).astype(f32)
            chk("Xt_solve", Xt, ctx)
            chk("Yt_solve", Yt, ctx)
            Sbf = S[h].astype(bf)
            ws_t = Sbf.T.astype(f32) @ Xt.astype(f32)
            vt_t = (Yt - ws_t).astype(bf)
            chk("vt_t", vt_t, ctx)
            g = jnp.exp(jnp.sum(la, axis=0)).astype(bf).astype(f32)
            chk("gc", g, ctx)
            ones = jnp.ones((Cc, Cc), f32)
            glast = ones @ hi.astype(f32) + ones @ lo.astype(f32)
            lam_a = jnp.exp(jnp.minimum(glast - G, 0.0))
            chk("lam_fwd", lam_a, ctx)
            y_a = (lam_a.astype(bf).astype(f32) * kv.astype(f32)).astype(bf)
            s_new = (Sbf.astype(f32) * g[:, None]
                     + y_a.astype(f32).T @ vt_t.astype(f32).T)
            out["G"].append(G); out["gam"].append(gam); out["aqk"].append(aqk)
            out["L"].append(akk.astype(bf).astype(f32))
            out["WT"].append(Xt); out["UT"].append(Yt)
            out["vtT"].append(vt_t); out["g"].append(g)
            out["lamA"].append(lam_a); out["yA"].append(y_a)
            out["Snew"].append(s_new)
        return {key: jnp.stack(val) for key, val in out.items()}

    chain_nat = []
    chain_t = []
    s_cur = jnp.zeros((H, dk, dv), f32)
    for i in range(n):
        sv = s_cur.astype(bf)
        chain_nat.append(sv.astype(f32))
        chain_t.append(jnp.swapaxes(sv, -1, -2).astype(f32))
        fwd = forward_chunk(i)
        s_cur = fwd["Snew"].astype(bf).astype(f32)

    dq = jnp.zeros((n, Cc, H * dk), f32)
    dk_out = jnp.zeros((n, Cc, H * dk), f32)
    dv_out = jnp.zeros((n, Cc, H * dv), f32)
    da = jnp.zeros((n, Cc, H * dk), f32)
    db = jnp.zeros((n, Cc, H * 32), f32)
    dsh_cur = dS.astype(bf).astype(f32)
    for i in reversed(range(n)):
        fwd = forward_chunk(i)
        for h in range(H):
            ctx = (i, h)
            ko, vo, bo = h * dk, h * dv, h * 32
            Sbf = chain_nat[i][h]
            kv = kc[i, :, ko:ko + dk].astype(bf)
            qv = qc[i, :, ko:ko + dk].astype(bf)
            vv = vc[i, :, vo:vo + dv].astype(bf)
            ah = ac[i, :, ko:ko + dk].astype(f32)
            beta = jnp.sum(bc[i, :, bo:bo + 32].astype(f32), axis=1) * (1.0 / 32.0)
            dov = doc[i, :, vo:vo + dv].astype(bf)
            G = fwd["G"][h]; gam = fwd["gam"][h]; aqk = fwd["aqk"][h]
            Lbf = fwd["L"][h]; Xt = fwd["WT"][h]; vt_t = fwd["vtT"][h]
            g = fwd["g"][h]
            dsh = dsh_cur[h]
            dgc = jnp.sum(Sbf.astype(f32) * dsh, axis=1)
            gq = (gam * qv.astype(f32)).astype(bf)
            acc1 = g[:, None] * dsh + gq.astype(f32).T @ dov.astype(f32)
            chk("acc1", acc1, ctx)
            daqk = ((dov.astype(f32) @ vt_t.astype(f32)) * tril).astype(bf).astype(f32)
            emg = 1.0 / gam
            chk("emg", emg, ctx)
            lam_b = jnp.broadcast_to(g[None, :], (Cc, dk))
            lam = lam_b / gam
            chk("lam_div", lam, ctx)
            y_bf = (lam * kv.astype(f32)).astype(bf)
            dvt = ((y_bf.astype(f32) @ dsh) + aqk.astype(f32).T @ dov.astype(f32))
            chk("dvt", dvt, ctx)
            dvt_bf = dvt.astype(bf).astype(f32)
            dgq = dov.astype(f32) @ Sbf.T.astype(f32)
            y_s = (kv.astype(f32) * emg).astype(bf)
            chk("y_s_emg_bf16", y_s, ctx)
            dr_a = (daqk.astype(f32) @ y_s.astype(f32)) * gam
            chk("dr_a", dr_a, ctx)
            y_s = (qv.astype(f32) * gam).astype(bf)
            dc_a = (daqk.T.astype(f32) @ y_s.astype(f32)) * emg
            chk("dc_a", dc_a, ctx)
            dqh = (dgq * gam + dr_a).astype(bf)
            chk("dqh_bf", dqh, ctx)
            dkh = dc_a
            dG = (dr_a * qv.astype(f32) - dc_a * kv.astype(f32)
                  + (dgq * qv.astype(f32)) * gam)
            chk("dG", dG, ctx)
            dyv = vt_t.astype(f32).T @ dsh.T
            acc1_bf = acc1.astype(bf).astype(f32)
            acc2 = acc1_bf - Xt.astype(f32) @ dvt_bf
            chk("acc2", acc2, ctx)
            dsh_cur = dsh_cur.at[h].set(acc2.astype(bf).astype(f32))
            smt = jnp.eye(Cc, dtype=f32)
            smt = (smt - Lbf.T @ smt).astype(bf).astype(f32)
            pf = Lbf
            for _ in range(steps - 1):
                pf = (pf @ pf).astype(bf).astype(f32)
                smt = (smt + pf.T @ smt).astype(bf).astype(f32)
            chk("smt", smt, ctx)
            yuv = (smt @ dvt_bf).astype(bf).astype(f32)
            ywv = (-(yuv.astype(f32) @ Sbf.T.astype(f32))).astype(bf).astype(f32)
            dlv = (-(yuv.astype(f32) @ vt_t.astype(f32)) * strict).astype(bf).astype(f32)
            kbf = kv.astype(f32) * beta[:, None]
            dG2 = dG + (ywv * kbf) * gam
            dkh2 = dkh + ywv * (gam * kbf)
            dbv = (jnp.sum(ywv * gam * kv.astype(f32), axis=1)
                   + jnp.sum(yuv * vv.astype(f32), axis=1))
            dvh = (yuv * beta[:, None]).astype(bf)
            y_s = (kv.astype(f32) * emg).astype(bf)
            dr_k = (dlv.astype(f32) @ y_s.astype(f32)) * gam
            chk("dr_k", dr_k, ctx)
            y_s = (kbf * gam).astype(bf)
            dc_k = (dlv.T.astype(f32) @ y_s.astype(f32)) * emg
            chk("dc_k", dc_k, ctx)
            dlam = dyv * kv.astype(f32)
            dz = dlam * lam
            chk("dz", dz, ctx)
            dkh2 = dkh2 + dr_k * beta[:, None] + dc_k + dyv * lam
            dbv = dbv + jnp.sum(dr_k * kv.astype(f32), axis=1)
            dG2 = dG2 + dr_k * kbf - dc_k * kv.astype(f32) - dz
            chk("dG2", dG2, ctx)
            dsum = jnp.sum(dz, axis=0)
            bc_bf = jnp.broadcast_to((g[None, :] * dgc + dsum), (Cc, dk))
            dla = tril.T @ dG2
            dalpha = (dla + bc_bf) / ah
            chk("dalpha", dalpha, ctx)
            dq = dq.at[i, :, ko:ko + dk].set(dqh.astype(f32))
            dk_out = dk_out.at[i, :, ko:ko + dk].set(dkh2.astype(bf).astype(f32))
            dv_out = dv_out.at[i, :, vo:vo + dv].set(dvh.astype(f32))
            da = da.at[i, :, ko:ko + dk].set(dalpha)
            db = db.at[i, :, bo:bo + 32].set(
                jnp.broadcast_to((dbv * (1.0 / 32.0))[:, None], (Cc, 32)))
    return (dq.reshape(T, H * dk), dk_out.reshape(T, H * dk),
            dv_out.reshape(T, H * dv), da.reshape(T, H * dk),
            db.reshape(T, H * 32), dsh_cur)


def main():
    args = make(5.0, 64, 32, 128, 128, 1, 0, "ramp")
    q, k, v, alpha, beta32, S, do, dS = args
    # per-chunk diagnostics: max |G| at the chunk end and max finite emg
    la = jnp.log(alpha.reshape(2, 32, -1))
    G = jnp.cumsum(la, axis=1)
    print("per-chunk max |G|:", [float(jnp.max(jnp.abs(G[i]))) for i in range(2)])
    out = emulate(q, k, v, alpha, beta32, S, do, dS, tile=16, chunk=32)
    print("FIRST inf:", FIRST["inf"])
    print("FIRST nan:", FIRST["nan"])
    for name, t in zip(("dq", "dk", "dv", "dalpha", "dbeta32", "dS_in"), out):
        t = jnp.asarray(t, jnp.float32)
        print(f"{name}: nan={bool(jnp.any(jnp.isnan(t)))} "
              f"inf={bool(jnp.any(jnp.isinf(t)))}")


if __name__ == "__main__":
    main()
