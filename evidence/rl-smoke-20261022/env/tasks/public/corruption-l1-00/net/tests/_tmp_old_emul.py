import jax
import jax.numpy as jnp

C = 32

def _emulate_bf16_bwd(q, k, v, alpha, beta32, S, do, dS, *, tile=16, chunk=C):
    """The variant-B kernel's declared numerics: a jnp mirror, chunk by chunk.

    The two passes of :func:`net.kernels.kda_cc_fused_persist.kernel_bwd` at
    the level of its *declared rounding points*: the bf16 state chain (natural
    and transposed copies), the forward recompute with the milestone-1 operand
    storages, the bf16 carried ``dS`` state (two throws per chunk), the bf16
    ``dvt`` crossing, every adjoint operand buffer, and the cotangent casts
    (bf16 ``dq``/``dk``/``dv``).  The two exactly-equal reformulations of the
    kernel are mirrored too: the factored decay ratios in the score adjoints,
    ``dL = -Yu vt^T``, and ``lam = gc / gamma`` in the recompute pass (the
    state pass keeps the forward kernel's full-row-sum ``lam``).

    This is the seam the CPU tests substitute for the kernel execution; it is
    pinned against the fp32 adjoint (:func:`persist.reference_bwd_scan`) at the
    declared 2e-2 operand budget.
    """
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

    # ---- one chunk of the forward body (declared storages), per head --------
    def forward_chunk(i):
        out = {"G": [], "gam": [], "aqk": [], "L": [], "WT": [], "UT": [],
               "vtT": [], "g": [], "lamA": [], "yA": [], "Snew": []}
        for h in range(H):
            ko, vo, bo = h * dk, h * dv, h * 32
            kv = kc[i, :, ko:ko + dk].astype(bf)
            qv = qc[i, :, ko:ko + dk].astype(bf)
            vv = vc[i, :, vo:vo + dv].astype(bf)
            ah = ac[i, :, ko:ko + dk].astype(f32)
            beta = jnp.sum(bc[i, :, bo:bo + 32].astype(f32), axis=1) * (1.0 / 32.0)
            bcol = beta[:, None]
            la = jnp.log(ah)
            hi = la.astype(bf)
            lo = (la - hi.astype(f32)).astype(bf)
            G = tril @ hi.astype(f32) + tril @ lo.astype(f32)

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

            akk = scores((kv.astype(f32) * bcol).astype(bf), kv,
                         strict)
            aqk = scores(qv, kv, tril)
            gam = jnp.exp(G).astype(bf).astype(f32)
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
            Sbf = S[h].astype(bf)
            ws_t = Sbf.T.astype(f32) @ Xt.astype(f32)
            vt_t = (Yt - ws_t).astype(bf)                       # (dv, C)
            g = jnp.exp(jnp.sum(la, axis=0)).astype(bf).astype(f32)
            # the state pass keeps the forward kernel's full-row-sum lam ...
            ones = jnp.ones((Cc, Cc), f32)
            glast = ones @ hi.astype(f32) + ones @ lo.astype(f32)
            lam_a = jnp.exp(jnp.minimum(glast - G, 0.0))
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

    # ---- pass A: the bf16 chain of incoming states --------------------------
    chain_nat = []
    chain_t = []
    s_cur = jnp.zeros((H, dk, dv), f32)
    for i in range(n):
        sv = s_cur.astype(bf)                       # the SMEM state is bf16
        chain_nat.append(sv.astype(f32))
        chain_t.append(jnp.swapaxes(sv, -1, -2).astype(f32))
        fwd = forward_chunk(i)
        s_cur = fwd["Snew"].astype(bf).astype(f32)

    # ---- pass B: the reverse adjoint ---------------------------------------
    dq = jnp.zeros((n, Cc, H * dk), f32)
    dk_out = jnp.zeros((n, Cc, H * dk), f32)
    dv_out = jnp.zeros((n, Cc, H * dv), f32)
    da = jnp.zeros((n, Cc, H * dk), f32)
    db = jnp.zeros((n, Cc, H * 32), f32)
    dsh_cur = dS.astype(bf).astype(f32)             # the SMEM state init (bf16)
    for i in reversed(range(n)):
        fwd = forward_chunk(i)
        for h in range(H):
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
            dsh = dsh_cur[h]                         # f32 of the bf16 state
            dgc = jnp.sum(Sbf.astype(f32) * dsh, axis=1)
            # adjoint stage 1 (the declared bf16 operand storages)
            gq = (gam * qv.astype(f32)).astype(bf)
            acc1 = g[:, None] * dsh + gq.astype(f32).T @ dov.astype(f32)
            daqk = ((dov.astype(f32) @ vt_t.astype(f32)) * tril).astype(bf).astype(f32)
            emg = 1.0 / gam
            lam_b = jnp.broadcast_to(g[None, :], (Cc, dk))
            lam = lam_b / gam
            y_bf = (lam * kv.astype(f32)).astype(bf)
            dvt = ((y_bf.astype(f32) @ dsh)
                   + aqk.astype(f32).T @ dov.astype(f32))
            dvt_bf = dvt.astype(bf).astype(f32)      # the GMEM crossing
            dgq = dov.astype(f32) @ Sbf.T.astype(f32)
            y_s = (kv.astype(f32) * emg).astype(bf)
            dr_a = (daqk.astype(f32) @ y_s.astype(f32)) * gam
            y_s = (qv.astype(f32) * gam).astype(bf)
            dc_a = (daqk.T.astype(f32) @ y_s.astype(f32)) * emg
            dqh = (dgq * gam + dr_a).astype(bf)
            dkh = dc_a
            dG = (dr_a * qv.astype(f32) - dc_a * kv.astype(f32)
                  + (dgq * qv.astype(f32)) * gam)
            # adjoint stage 2
            dyv = vt_t.astype(f32).T @ dsh.T         # dY (fp32)
            acc1_bf = acc1.astype(bf).astype(f32)
            acc2 = acc1_bf - Xt.astype(f32) @ dvt_bf
            dsh_cur = dsh_cur.at[h].set(acc2.astype(bf).astype(f32))
            smt = jnp.eye(Cc, dtype=f32)
            smt = (smt - Lbf.T @ smt).astype(bf).astype(f32)
            pf = Lbf
            for _ in range(steps - 1):
                pf = (pf @ pf).astype(bf).astype(f32)
                smt = (smt + pf.T @ smt).astype(bf).astype(f32)
            yuv = (smt @ dvt_bf).astype(bf).astype(f32)
            ywv = (-(yuv.astype(f32) @ Sbf.T.astype(f32))).astype(bf).astype(f32)
            dlv = (-(yuv.astype(f32) @ vt_t.astype(f32))
                   * strict).astype(bf).astype(f32)
            kbf = kv.astype(f32) * beta[:, None]
            dG2 = dG + (ywv * kbf) * gam
            dkh2 = dkh + ywv * (gam * kbf)
            dbv = (jnp.sum(ywv * gam * kv.astype(f32), axis=1)
                   + jnp.sum(yuv * vv.astype(f32), axis=1))
            dvh = (yuv * beta[:, None]).astype(bf)
            y_s = (kv.astype(f32) * emg).astype(bf)          # ccol / gamma
            dr_k = (dlv.astype(f32) @ y_s.astype(f32)) * gam
            y_s = (kbf * gam).astype(bf)                     # ccol * beta * gamma
            dc_k = (dlv.T.astype(f32) @ y_s.astype(f32)) * emg
            dlam = dyv * kv.astype(f32)
            dz = dlam * lam
            dkh2 = dkh2 + dr_k * beta[:, None] + dc_k + dyv * lam
            dbv = dbv + jnp.sum(dr_k * kv.astype(f32), axis=1)
            dG2 = dG2 + dr_k * kbf - dc_k * kv.astype(f32) - dz
            dsum = jnp.sum(dz, axis=0)
            bc_bf = jnp.broadcast_to((g[None, :] * dgc + dsum), (Cc, dk))
            dla = tril.T @ dG2
            dalpha = (dla + bc_bf) / ah
            dq = dq.at[i, :, ko:ko + dk].set(dqh.astype(f32))
            dk_out = dk_out.at[i, :, ko:ko + dk].set(dkh2.astype(bf).astype(f32))
            dv_out = dv_out.at[i, :, vo:vo + dv].set(dvh.astype(f32))
            da = da.at[i, :, ko:ko + dk].set(dalpha)
            db = db.at[i, :, bo:bo + 32].set(
                jnp.broadcast_to((dbv * (1.0 / 32.0))[:, None], (Cc, 32)))
    return (dq.reshape(T, H * dk), dk_out.reshape(T, H * dk),
            dv_out.reshape(T, H * dv), da.reshape(T, H * dk),
            db.reshape(T, H * 32), dsh_cur)
