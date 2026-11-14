"""Analytic adjoint of the fused ``chunked_cc`` chunk body (ADR-054 stage-2b, milestone 2).

What this module is
-------------------
:func:`reference_bwd` is the reverse-mode adjoint of
:func:`net.kernels.kda_cc_fused.reference` (the fp32 replica of the fused
forward kernel's algebra).  It is the backward half of the differentiable
entry point :func:`net.kernels.kda_cc_fused.cc_step` (``jax.custom_vjp``):
the forward rule runs the Mosaic kernel on the device, the backward rule runs
this jnp adjoint.  That is **variant B** of the milestone-2 delta: the forward
pass *and* the rematerialised forward pass collapse into one kernel per chunk,
the backward pass runs as fused XLA/jnp (it does **not** collapse to one
launch — the honest boundary; see "Variant A" below and ``open_questions``).

Verification status (honest)
----------------------------
* ``reference_bwd`` is pinned on CPU against ``jax.grad`` of
  :func:`net.kernels.kda_cc_fused.reference` — the same object the milestone-1
  tests pin against the committed :func:`net.kda.cc_chunk_step` at 2e-5.  Both
  fp32 and bf16-operand inputs; observed worst |diff| ~1e-6 (fp32) and the exact
  bf16 cotangent rounding for bf16 inputs (see
  ``net/tests/test_kda_cc_fused_bwd.py``);
* gradient parity of the *composite* path (bf16 operand cast + custom_vjp body
  vs the committed jnp body) is pinned at the declared operand budget — the
  observed numbers are in the test module's docstring;
* the module is pure jnp: on this CPU node it is *executed*, not only traced.

The adjoint, per head (the equations the code transcribes)
----------------------------------------------------------
Forward (see :func:`net.kernels.kda_cc_fused.reference`)::

    la = log(alpha);  G = cumsum_c la;  gamma = exp(G)
    E[c,i,d] = exp(G[c,d] - G[i,d])
    aqk = tril(  sum_d q[c,d] k[i,d] E[c,i,d] )
    L   = strict( sum_d (k*beta_c)[c,d] k[i,d] E[c,i,d] )        # WY matrix
    xw  = gamma*k*beta;  vw = v*beta
    M   = I + L;   W = M^{-1} xw;   U = M^{-1} vw
    vt  = U - W S
    o   = (gamma*q) S + aqk vt
    gc  = exp(sum_c la);  lam = exp(sum_c la - G);  y = lam*k
    S'  = gc[:,None]*S + y^T vt

Backward, given ``(do, dS')`` (the code accumulates in this order)::

    dS  = gc[:,None]*dS'                                   # into the S cotangent
    dgc = sum_u S[d,u] dS'[d,u]
    dY  = vt dS'^T                       dvt  = y dS' + aqk^T do
    daqk = tril(do vt^T)                 dS  += (gamma*q)^T do
    dgq = do S^T;  dgamma = dgq*q;  dq = dgq*gamma
    dU  = dvt;     dW = -dvt S^T;        dS  -= W^T dvt
    Yw = M^{-T} dW;  Yu = M^{-T} dU      # transposed (adjoint) solves
    dL  = strict(-(Yw W^T + Yu U^T))
    dgamma += Yw*(k*beta);  dk += Yw*(gamma*beta);  dv = Yu*beta
    dbeta  = rowsum(Yw*gamma*k) + rowsum(Yu*v) + rowsum(dr_k*k)
    # score adjoints, for a mask * dT (T = m * sum_d r[c,d] ccol[i,d] E[c,i,d]):
    dr[c,d] = sum_i dT[c,i] ccol[i,d] E[c,i,d]
    dc[i,d] = sum_c dT[c,i]  r[c,d]  E[c,i,d]
    dG[c,d] +=  r[c,d]*dr[c,d];   dG[i,d] -= ccol[i,d]*dc[i,d]
    dG += dgamma*gamma                     # through gamma = exp(G)
    dlam = dY*k;  dk += dY*lam;  dz = dlam*lam
    dG -= dz                               # z = sum_c la - G
    dla = tril^T dG + broadcast(sum_c dz + gc*dgc)   # through G = cumsum(la)
    dalpha = dla / alpha                   # la = log(alpha)

The two solve adjoints are the "transposed kernel" of the forward chain: the
forward solves right-multiply by ``(I+L)^{-1}`` in the transposed storage
(``X^T <- X^T (I-L)^T (I+L^2)^T ...``); the adjoint needs ``M^{-T} dW``, whose
transpose ``dW^T M^{-1}`` is the *same chain* with the transposed multipliers —
the exact recipe :func:`net.kernels.kda_ut_solve._backward` applies with its
sibling kernel.  Here the chain is written directly (``SM.T @ dW`` with
``SM = (I+L)^{-1}``): the matrices are 32x32 and this is the CPU-verified form;
a future kernel (variant A) transcribes the same two solves as the transposed
Neumann chain on the transposed cotangents.

Variant A (the fused backward kernel) — why it is not shipped in this round
--------------------------------------------------------------------------
Variant A (one Mosaic kernel per chunk in the backward pass too, so all three
passes — fwd, remat-fwd, bwd — collapse) was derived and prototyped on paper
during this delta and has a concrete blocker plus a concrete risk, both stated
for the architect:

* **``dot_general`` does not lower in the Mosaic GPU backend for Warpgroup
  semantics in jax 0.11.2** (verified experimentally on this node:
  ``NotImplementedError: Unimplemented primitive in Pallas Mosaic GPU lowering:
  dot_general``).  A fused backward therefore has to be hand-written with
  explicit ``plgpu.mma`` sites and layout/memory-orientation juggling exactly
  like the forward kernel — the jnp adjoint here cannot simply be lifted into a
  ``plgpu.kernel`` body;
* the backward's dataflow needs ~20+ mma sites and both the natural and the
  transposed memory orientation of several operands (e.g. the dS cotangent
  enters as ``dS'`` (RHS of ``dY``) and as ``dS'^T`` (RHS of ``dvt``), and the
  two adjoint solves rerun the transposed chain), i.e. the same SMEM-etiquette
  class as the forward;
* **this node cannot verify such a kernel beyond lowering** (no GPU; milestone
  1's own GB10 probe measured launches only, not values — and the
  declared-algebra audit performed in this delta found two forward-kernel
  transcription errors that no CPU test had caught: the phase-4 ``lam`` pattern
  and the ``gamma^C`` broadcast axis; both are fixed in
  :mod:`net.kernels.kda_cc_fused`, pinned by static tests).  A shadow-emulation
  of a hand-written backward would carry the same epistemic limit.  Shipping
  variant B first keeps the backward *executed and gradchecked* on CPU and the
  fused backward kernel as a well-scoped follow-up whose numbers can be taken
  against a verified adjoint.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

#: The forward module carries the shared contract (CASES, TOLERANCE, ...).
from . import kda_cc_fused  # noqa: F401 — re-exported for the sibling contract

__all__ = ["reference_bwd"]


def reference_bwd(q, k, v, alpha, beta32, S, do, dS, *, tile=16):
    """Analytic adjoint of :func:`kda_cc_fused.reference`.

    Inputs mirror the forward exactly: ``q``/``k`` ``(C, H*dk)``, ``v``
    ``(C, H*dv)``, ``alpha`` ``(C, H*dk)`` fp32, ``beta32`` ``(C, H*32)`` fp32
    (the write gate replicated over 32 columns), ``S`` ``(H, dk, dv)`` fp32;
    the cotangents of the forward outputs are ``do`` ``(C, H*dv)`` and ``dS``
    ``(H, dk, dv)``.  Returns ``(dq, dk, dv, dalpha, dbeta32, dS_in)`` with the
    *primal dtypes* (bf16 q/k/v get bf16 cotangents, exactly like
    ``jax.grad``), so the function can serve directly as the ``custom_vjp``
    backward rule.

    ``tile`` is the forward's decay-factor tile width; it must match the value
    the forward ran with (the score adjoints below use the *direct* form
    ``E[c,i,d] = exp(G_c - G_i)``, which the forward's tile decomposition
    evaluates up to fp32 rounding — hence the tolerance class of the pins, not
    exactness; see the module docstring).
    """
    C = q.shape[0]
    H, dk, dv = S.shape
    f32 = jnp.float32
    ar = jnp.arange(C)
    ridx, cidx = ar[:, None], ar[None, :]
    tril = (ridx >= cidx).astype(f32)
    strict = (ridx > cidx).astype(f32)

    dq = jnp.zeros((C, H * dk), f32)
    dk_out = jnp.zeros((C, H * dk), f32)
    dv_out = jnp.zeros((C, H * dv), f32)
    dalpha = jnp.zeros((C, H * dk), f32)
    dbeta32 = jnp.zeros_like(beta32, dtype=f32)
    dS_in = jnp.zeros((H, dk, dv), f32)

    for h in range(H):
        ko, vo, bo = h * dk, h * dv, h * 32
        qh = q[:, ko:ko + dk].astype(f32)
        kh = k[:, ko:ko + dk].astype(f32)
        vh = v[:, vo:vo + dv].astype(f32)
        alpha_h = alpha[:, ko:ko + dk].astype(f32)
        # beta32 replicates beta over 32 columns (the loadable shape of a
        # per-row vector in the kernel); the recovery is the exact mean.
        beta = (jnp.sum(beta32[:, bo:bo + 32].astype(f32), axis=1) * (1.0 / 32.0))[:, None]
        Sh = S[h]

        la = jnp.log(alpha_h)
        G = tril @ la
        gamma = jnp.exp(G)
        # Clamp the exponent at 0: the upper triangle (i < j) is masked out
        # later by tril/strict, but inf * 0 == nan inside the einsums.
        # The reference (net.kda._cc_scores) uses the same exp(min(diff, 0)).
        E = jnp.exp(jnp.minimum(G[:, None, :] - G[None, :, :], 0.0))  # (C, C, dk)

        aqk = tril * jnp.einsum("cid,cd,id->ci", E, qh, kh)
        L = strict * jnp.einsum("cid,cd,id->ci", E, kh * beta, kh)
        xw = gamma * kh * beta
        vw = vh * beta
        SM = jnp.linalg.inv(jnp.eye(C, dtype=f32) + L)      # (I+L)^{-1}
        W = SM @ xw
        U = SM @ vw
        vt = U - W @ Sh
        gc = jnp.exp(jnp.sum(la, axis=0))                   # (dk,)
        lam = jnp.exp(jnp.sum(la, axis=0)[None, :] - G)     # gamma^{c+1->C}
        y = lam * kh

        doh = do[:, vo:vo + dv]
        dSh = dS[h]

        # --- state transfer + pseudo-value/output adjoints -----------------
        dS_acc = gc[:, None] * dSh
        dgc = jnp.sum(Sh * dSh, axis=1)                     # (dk,)
        dY = vt @ dSh.T                                     # (C, dk)
        dvt = y @ dSh + aqk.T @ doh                         # (C, dv)
        daqk = (doh @ vt.T) * tril
        dS_acc = dS_acc + (gamma * qh).T @ doh
        dgq = doh @ Sh.T                                    # cotangent of gamma*q
        dgamma = dgq * qh
        dqh = dgq * gamma

        # --- solve / pseudo-value adjoint ----------------------------------
        dU = dvt
        dW = -(dvt @ Sh.T)
        dS_acc = dS_acc - W.T @ dvt
        Yw = SM.T @ dW                                      # M^{-T} dW
        Yu = SM.T @ dU
        dL = -(Yw @ W.T + Yu @ U.T) * strict
        dgamma = dgamma + Yw * (kh * beta)
        dkh = Yw * (gamma * beta)
        dvh = Yu * beta
        db = jnp.sum(Yw * gamma * kh, axis=1) + jnp.sum(Yu * vh, axis=1)

        # --- score adjoints (aqk and L share the column operand k) ---------
        dr_a = jnp.einsum("ci,cid,id->cd", daqk, E, kh)
        dc_a = jnp.einsum("ci,cid,cd->id", daqk, E, qh)
        dqh = dqh + dr_a
        dkh = dkh + dc_a
        dr_k = jnp.einsum("ci,cid,id->cd", dL, E, kh)
        dc_k = jnp.einsum("ci,cid,cd->id", dL, E, kh * beta)
        dkh = dkh + dr_k * beta + dc_k
        db = db + jnp.sum(dr_k * kh, axis=1)
        dG = (dr_a * qh - dc_a * kh) + (dr_k * (kh * beta) - dc_k * kh)
        dG = dG + dgamma * gamma                            # through exp(G)

        # --- lam / gc / cumsum adjoints ------------------------------------
        dlam = dY * kh
        dkh = dkh + dY * lam
        dz = dlam * lam                                     # z = sum_c la - G
        dG = dG - dz
        dsum = jnp.sum(dz, axis=0)                          # (dk,)
        dla = (tril.T @ dG) + jnp.broadcast_to(
            (dsum + gc * dgc)[None, :], (C, dk))

        dq = dq.at[:, ko:ko + dk].add(dqh)
        dk_out = dk_out.at[:, ko:ko + dk].add(dkh)
        dv_out = dv_out.at[:, vo:vo + dv].add(dvh)
        dalpha = dalpha.at[:, ko:ko + dk].add(dla / alpha_h)  # la = log(alpha)
        dS_in = dS_in.at[h].add(dS_acc)
        dbeta32 = dbeta32.at[:, bo:bo + 32].add(
            jnp.broadcast_to((db / 32.0)[:, None], (C, 32)))

    # Cotangents must carry the primal dtypes for the custom_vjp rule (bf16
    # q/k/v get bf16 cotangents — the same rounding jax.grad performs).
    return (dq.astype(q.dtype), dk_out.astype(k.dtype), dv_out.astype(v.dtype),
            dalpha.astype(alpha.dtype), dbeta32.astype(beta32.dtype), dS_in)
