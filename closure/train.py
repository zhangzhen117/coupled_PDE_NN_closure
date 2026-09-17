#!/usr/bin/env python3
"""Train state-dependent channel closures with Adam and L-BFGS.

See README.md for the five reported families and reference-state refinement.
Checkpoint metadata records stopping criteria; a saved model alone is not a
certificate of optimizer convergence or global tangent dissipation.
"""
import argparse
import os
import sys

import numpy as np
import torch

torch.set_num_threads(1)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from chan.cases import CASES, MODELS
from closure.nets import (ClosureDiv, ClosureDivMono, ClosureHybrid,
                          ClosureHybrid2, ClosureNuEff, ClosureNuEffS,
                          ClosureTau)
from data import lm_channel as LM

FAMILIES = ("force", "force_pen", "hybrid",
            "force_pen2",   # A1 with the FULL symmetric-part coefficient F_U - 1/2 D1 F_S
                            # (paper sec. 3.3); force_pen penalised F_U alone
            "force_pen2w",  # same objective, warm-started from the trained A0 (`force`)
                            # checkpoint: from scratch the penalty gradient dominated and
                            # L-BFGS plateaued at |g| ~ 0.1 after 17-57 cycles
            # Stage 13c -- the two structural knobs applied together
            "force_mm",     # force form  + row-sum + off-diagonal, off-manifold
            "nueff",        # nu_eff >= 0 by construction, no penalty (baseline)
            "nueff_mm",     # nu_eff >= 0  + row-sum + off-diagonal, off-manifold
            "tau",          # FORM C: predict tau, differentiate it.  Unconstrained.
            "tau_mm",       # form C + row-sum + off-diagonal, off-manifold
            "nueffs",       # FORM C++: nu_eff>=0 AND no U input -> M-matrix by
                            # construction, no penalty of any kind
            "force_mono",   # A4: form A with dg/dU >= 0 BY CONSTRUCTION (U-integral)
            "hybrid2")      # B2: C+ flux (+) A4 residual -- both branches architectural
ROW_LAMBDA = 30.0       # same weight the force_pen reaction penalty already uses
OFF_LAMBDA = 3.0        # the Peclet term is a conditioning knob, not an existence one
REACT_LAMBDA = 30.0
WIDTH, DEPTH = 48, 2


def per_case_data(cases=CASES):
    """One tensor bundle per case.  The hybrid differentiates on a grid, so it must be fed
    per case -- differencing across a pooled concatenation of five Reynolds numbers would
    straddle the boundary."""
    out = []
    for r in cases:
        d = LM.load(r)
        y = d["y"]
        w = np.gradient(y)
        w = w / w.sum() / len(cases)
        t = lambda a: torch.tensor(np.asarray(a, float), dtype=torch.float64)
        out.append(dict(U=t(d["U"]), S=t(d["dUdy"]), y=t(y), W=t(w),
                        G=t(np.gradient(d["tau_true"], y)),
                        nu=d["nu"], rt=d["retau"], retau=r))
    return out


def predict(model, family, c):
    if family.startswith(("hybrid", "nueff", "tau")):
        return model.g_total(c["U"], c["S"], c["y"], c["nu"], c["rt"])
    return model.dtau_dy(c["U"], c["S"], c["nu"], c["rt"])


def data_loss(model, family, data):
    """Globally-normalised term + pointwise term normalised by (|G| + 1).

    The global term alone is blind to Re_tau = 5200, y+ 100-300, where |dtau/dy| ~ 0.26
    sits beside a near-wall spike of 345; the local term alone lets the near wall drift.
    The sum holds both ends.
    """
    ng = nl = dg = dw = 0.0
    for c in data:
        g = predict(model, family, c)
        ng = ng + torch.sum(c["W"] * (g - c["G"]) ** 2)
        dg = dg + torch.sum(c["W"] * c["G"] ** 2)
        nl = nl + torch.sum(c["W"] * ((g - c["G"]) / (torch.abs(c["G"]) + 1.0)) ** 2)
        dw = dw + torch.sum(c["W"])
    return ng / dg + nl / dw


def reaction_penalty(model, data):
    """mean( max(-G_U, 0)^2 ), G_U = d(dtau/dy)/dU.

    Only the negative part: B = -diag(dV) dg/dU, so G_U < 0 puts a POSITIVE diagonal onto
    A (negative definite) and pushes eigenvalues right.  G_U > 0 is harmless.
    """
    num = den = 0.0
    for c in data:
        Ug = c["U"].detach().requires_grad_(True)
        g = model.dtau_dy(Ug, c["S"], c["nu"], c["rt"])
        G_U, = torch.autograd.grad(g.sum(), Ug, create_graph=True)
        num = num + torch.sum(c["W"] * torch.clamp(-G_U, min=0.0) ** 2)
        den = den + torch.sum(c["W"])
    return num / den


def reaction_penalty_full(model, data):
    """mean( max(-(G_U - 1/2 D1 G_S), 0)^2 ): the positive part of the reaction coefficient
    of sym B in the force form.

    Continuous form (paper, Lemma 1 with c0 = F_U, c1 = F_S, c2 = 0):
        sym B = F_U - 1/2 D1 F_S ,
    and it is this coefficient, not F_U alone, that a first-order operator contributes to
    the symmetric part -- the advection term F_S D1 is skew up to the derivative of its
    coefficient.  In the code's sign (g = dtau_code/dy = -F, tau_code = <u'v'>):
        F_U - 1/2 D1 F_S = -(G_U - 1/2 D1 G_S) ,
    so the destabilising part is clamp(-(G_U - 1/2 D1 G_S), 0).  D1 is a centred difference
    on the DNS grid, one-sided at the ends.  `reaction_penalty` above is the special case
    that drops the D1 G_S term.
    """
    num = den = 0.0
    for c in data:
        Ug = c["U"].detach().requires_grad_(True)
        Sg = c["S"].detach().requires_grad_(True)
        g = model.dtau_dy(Ug, Sg, c["nu"], c["rt"])
        G_U, G_S = torch.autograd.grad(g.sum(), (Ug, Sg), create_graph=True)
        y = c["y"]
        d_int = (G_S[2:] - G_S[:-2]) / (y[2:] - y[:-2])
        d_lo = ((G_S[1] - G_S[0]) / (y[1] - y[0])).reshape(1)
        d_hi = ((G_S[-1] - G_S[-2]) / (y[-1] - y[-2])).reshape(1)
        coef = G_U - 0.5 * torch.cat([d_lo, d_int, d_hi])
        num = num + torch.sum(c["W"] * torch.clamp(-coef, min=0.0) ** 2)
        den = den + torch.sum(c["W"])
    return num / den


# ---------------------------------------------------------------------------
# Stage 13c -- structural penalties evaluated on the OPERATOR mesh.
#
# B at a reference state is algebraic: B = B(U_ref, theta, d/dy).  No PDE solve.  So every
# structural property of J = A + B is available per sample at training time, for free.
# Two of them decide whether the problem is solvable at all:
#
#   ROW SUM.  In the force form B = -diag(dV)[diag(G_U) + diag(G_S) D], and D (the cell
#     d/dy stencil) differentiates a constant to zero, so rowsum(B) = -dV*G_U EXACTLY
#     (verified to 1e-14 on interior rows).  The S-branch cannot touch the row sum.  A
#     positive row sum means J maps a positive vector to a positive one -- the M-matrix
#     property is lost, det can change sign, and a FOLD becomes possible.  Stage 13a
#     measured exactly that: the only family with rowsum ~ 0 is the only one whose
#     solution branch never disconnects.
#
#   OFF-DIAGONALS.  J_{i,i+-1} >= 0 is a cell-Peclet condition: the closure's advection
#     must not outrun the viscous coupling nu/d.  Advection has zero row sum so it cannot
#     create a fold by itself -- this one buys conditioning, not existence.
#
# The two published knobs are precisely these: the reaction penalty is a ROW-SUM
# constraint, and nu_t >= 0 / nu_eff >= 0 is an OFF-DIAGONAL constraint.  Neither family
# had both.  These functions supply both, to either form.
# ---------------------------------------------------------------------------
N_PERTURB = 4           # off-manifold states per case, fixed so the objective is
PERTURB_AMP = 0.25      # deterministic -- L-BFGS needs that


def operator_data(cases=CASES, n_perturb=N_PERTURB, amp=PERTURB_AMP):
    """Per-case operator-mesh bundle, plus off-manifold states.

    The reference state alone constrains J only where the solution already is; a fold can
    happen anywhere on the path.  So carry a fixed set of smooth perturbations of U_ref
    and apply the penalties at all of them.
    """
    from chan.cases import case_setup
    from closure import jacobian as JAC
    out = []
    for r in cases:
        c = case_setup(r)
        m = c["m"]
        rt = LM.load(r)["retau"]
        cl, cc, cr = JAC._cell_gradient_coeffs(m)
        A = c["A"]
        i = np.arange(m.n)
        t = lambda a: torch.tensor(np.asarray(a, float), dtype=torch.float64)
        g = np.random.default_rng(20260825 + r)
        states = [c["U_dns"]]
        for k in range(n_perturb):                      # smooth multiplicative modes
            ph, co = g.uniform(0, 2 * np.pi, 3), g.uniform(-1, 1, 3)
            xi = sum(co[j] * np.sin((j + 1) * np.pi * m.yc / 2.0 + ph[j]) for j in range(3))
            states.append(c["U_dns"] * (1.0 + amp * xi / np.abs(xi).max()))
        out.append(dict(U=[t(u) for u in states], cl=t(cl), cc=t(cc), cr=t(cr),
                        dV=t(m.dV), w=t(JAC.face_weights(m)), d=t(m.d),
                        A_sub=t(np.r_[0.0, A[i[1:], i[1:] - 1]]),
                        A_sup=t(np.r_[A[i[:-1], i[:-1] + 1], 0.0]),
                        A_row=t(A.sum(1)), A_diag=t(np.abs(np.diag(A))),
                        row_scale=t(np.median(np.abs(np.diag(A)) / m.dV)),
                        nu=1.0 / rt, rt=rt, n=m.n))
    return out


def _stencil_S(U, o):
    """S_i = cl U_{i-1} + cc U_i + cr U_{i+1}, ghost cells at the walls hold U = 0."""
    Um = torch.cat([torch.zeros(1, dtype=U.dtype), U[:-1]])
    Up = torch.cat([U[1:], torch.zeros(1, dtype=U.dtype)])
    return o["cl"] * Um + o["cc"] * U + o["cr"] * Up


def structure_penalty(model, family, odata, want_off=True):
    """(L_row, L_off), both dimensionless, averaged over cases and off-manifold states.

    Both terms are the DIAGONAL-DOMINANCE DEFICIT of J, normalised by the viscous entry
    they have to beat:

      L_row = mean( clamp( rowsum(J) / |A_ii| , 0 )^2 )    -- the existence condition
      L_off = mean( clamp( -J_offdiag / A_offdiag, 0 )^2 ) -- the cell-Peclet condition

    The normalisation is not cosmetic.  An unnormalised row-sum penalty has different
    magnitudes in the two closure forms -- ~6 for the force form, ~400 for the flux form
    -- so one weight cannot serve both.  At ROW_LAMBDA=30 the flux-form penalty reached
    1.2e4 against a data loss of 4e-5, and L-BFGS correctly threw the fit away and drove
    tau to zero, which satisfies every constraint and models nothing.
    """
    if not family.startswith(("nueff", "tau", "force")):
        raise ValueError(f"structure_penalty: no stencil route for family {family!r} "
                         "(hybrid mixes a flux branch and a cell-divergence branch; "
                         "route it explicitly before using this)")
    row = off = cnt = 0.0
    flux = family.startswith(("nueff", "tau"))     # tau at faces -> B = -E T
    for o in odata:
        for Uref in o["U"]:
            if flux:
                Ue = torch.cat([torch.zeros(1, dtype=Uref.dtype), Uref,
                                torch.zeros(1, dtype=Uref.dtype)])
                Ul = Ue[:-1].detach().requires_grad_(True)
                Ur = Ue[1:].detach().requires_grad_(True)
                tau = model.tau_uv((1 - o["w"]) * Ul + o["w"] * Ur,
                                   (Ur - Ul) / o["d"], o["nu"], o["rt"])
                a, = torch.autograd.grad(tau.sum(), Ul, create_graph=True)
                b, = torch.autograd.grad(tau.sum(), Ur, create_graph=True)
                # B_{i,i-1} = a_i ,  B_{i,i+1} = -b_{i+1} ,  rowsum(B)_i = -(N_U,i+1 - N_U,i)
                N_U = a + b
                G_U = (N_U[1:] - N_U[:-1]) / o["dV"]
                J_sub = o["A_sub"] + torch.cat([torch.zeros(1, dtype=a.dtype), a[1:-1]])
                J_sup = o["A_sup"] - torch.cat([b[1:-1], torch.zeros(1, dtype=b.dtype)])
            else:
                Ua = Uref.detach().requires_grad_(True)
                Sd = _stencil_S(Uref, o).detach().requires_grad_(True)
                g = model.dtau_dy(Ua, Sd, o["nu"], o["rt"])
                G_U, = torch.autograd.grad(g.sum(), Ua, create_graph=True)
                G_S, = torch.autograd.grad(g.sum(), Sd, create_graph=True)
                J_sub = o["A_sub"] - o["dV"] * G_S * o["cl"]
                J_sup = o["A_sup"] - o["dV"] * G_S * o["cr"]
            rs = o["A_row"] - o["dV"] * G_U          # = rowsum(J), exactly
            # Normalise by a per-case SCALAR, not per cell.  Dividing by |A_ii| looked
            # principled -- it is the relative diagonal-dominance margin -- but |A_ii| ~
            # nu/d spans 340x across the channel, so the term collapsed to ~1e-7 while
            # the data loss sat at ~1e-4: at lambda=30 the penalty was ~1% of the
            # objective and exerted almost no pressure.  Measured consequence: force_mm
            # left a near-wall violation of 152 at Re_tau=5200 (force_pen: exactly 0)
            # while its own loss read 3e-7, and it lost convergence at high Re.
            # rs/dV recovers force_pen's per-cell quantity (= -G_U in the interior);
            # the scalar only sets the overall size so one weight serves both forms.
            row = row + torch.mean(torch.clamp(rs / o["dV"] / o["row_scale"],
                                               min=0.0) ** 2)
            if want_off:
                s_ = torch.clamp(-J_sub[1:] / o["A_sub"][1:].abs().clamp(min=1e-30),
                                 min=0.0) ** 2
                p_ = torch.clamp(-J_sup[:-1] / o["A_sup"][:-1].abs().clamp(min=1e-30),
                                 min=0.0) ** 2
                off = off + 0.5 * (s_.mean() + p_.mean())
            cnt += 1.0
    return row / cnt, (off / cnt if want_off else torch.zeros((), dtype=torch.float64))


def _save(path, model, info, state=None):
    """Checkpoint after every cycle so a killed call costs one cycle, not the model."""
    if path is None:
        return
    torch.save({"state": state if state is not None else model.state_dict(),
                "info": info, "family": info["family"],
                "width": WIDTH, "depth": DEPTH}, path)


def build(family, seed):
    if family == "hybrid2":
        return ClosureHybrid2(WIDTH, DEPTH, seed).double()
    if family == "hybrid":
        return ClosureHybrid(WIDTH, DEPTH, seed).double()
    if family == "force_mono":              # before the generic "force" test
        return ClosureDivMono(WIDTH, DEPTH, seed).double()
    if family.startswith("nueffs"):          # must precede the "nueff" test
        return ClosureNuEffS(WIDTH, DEPTH, seed).double()
    if family.startswith("nueff"):
        return ClosureNuEff(WIDTH, DEPTH, seed).double()
    if family.startswith("tau"):
        return ClosureTau(WIDTH, DEPTH, seed).double()
    return ClosureDiv(WIDTH, DEPTH, seed).double()


def train(family, seed, adam_epochs=3000, lr=3e-3, lb_iter=500, max_cycles=2000,
          gtol=1e-5, ftol=1e-5, patience=12, grel=0.01, verbose=True,
          resume=None, save_path=None, row_lambda=None, off_lambda=None):
    """`resume` is a prior checkpoint dict: skip the Adam warm start and continue L-BFGS.

    Training is resumable because a full run exceeds the ~2 min this environment allows a
    foreground call, and background shells here get suspended between turns.  Each call
    does up to `max_cycles` cycles; repeat until `converged` is reported.

    State is written to `save_path` after the Adam warm start and after EVERY L-BFGS
    cycle, so a kill mid-run costs one cycle rather than the whole model.

    STOPPING.  First-order optimality ||grad L||_2 <= `gtol`, with a GRADIENT-STAGNATION
    fallback: stop when ||grad L|| has failed to improve on its running best by more than
    `grel` for `patience` consecutive cycles.  `stop_reason` and the achieved `grad_norm`
    are recorded, so "converged" is verifiable rather than asserted.

    The fallback keys on the GRADIENT, not the loss, because a loss-based test does not
    terminate here: L-BFGS reaches a floor where ||grad L|| is frozen to four digits while
    the loss still creeps by ~1e-5 relative per cycle.  A loss-plateau test with any
    tolerance tight enough to mean convergence then never fires, and the run crawls to the
    walltime -- observed at cycle 714 with |g| pinned at 4.263e-4.

    An earlier version stopped on a SINGLE slow cycle.  That fired at 29, 76, 144 and 200
    cycles across seeds of one family, leaving gradient norms spread over 124x -- so the
    seed-to-seed spread in every downstream figure was partly TRAINING-BUDGET variation
    rather than seed variation.  Since these closures are the foundation of the whole
    conditioning analysis, that contamination is not acceptable.
    """
    torch.manual_seed(seed)
    model = build(family, seed)
    data = per_case_data()
    lam = (ROW_LAMBDA if family.endswith("_mm")
           else REACT_LAMBDA if family in ("force_pen", "force_pen2", "force_pen2w") else 0.0)
    lam_off = OFF_LAMBDA if family.endswith("_mm") else 0.0
    if row_lambda is not None:      # penalty-weight continuation: the constraint is hard
        lam = float(row_lambda)     # (L_row ~ 170x the data loss for the flux form), so
    if off_lambda is not None:      # ramp lam over stages and let the model track the
        lam_off = float(off_lambda) # constrained optimum instead of jumping to tau = 0
    if family.endswith("_mm"):
        # Both structural knobs, on the operator mesh, at the reference state AND at
        # fixed off-manifold perturbations of it.  The old reaction_penalty saw only the
        # DNS state on the training grid, so it constrained J where the solution already
        # is -- but a fold can happen anywhere on the path.
        odata = operator_data()

        def obj():
            row, off = structure_penalty(model, family, odata)   # one pass, both terms
            return data_loss(model, family, data) + lam * row + lam_off * off
    elif family == "force_pen":
        obj = lambda: (data_loss(model, family, data)
                       + REACT_LAMBDA * reaction_penalty(model, data))
    elif family in ("force_pen2", "force_pen2w"):
        obj = lambda: (data_loss(model, family, data)
                       + REACT_LAMBDA * reaction_penalty_full(model, data))
    else:
        obj = lambda: data_loss(model, family, data)

    if resume is not None:
        model.load_state_dict(resume["state"])
        adam_loss = resume["info"].get("adam_loss", float("nan"))
        cycles_done = resume["info"].get("cycles_total", 0)
    else:
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, adam_epochs)
        for _ in range(adam_epochs):
            opt.zero_grad()
            obj().backward()
            opt.step()
            sch.step()
        adam_loss = float(obj())
        cycles_done = 0
        _save(save_path, model, dict(family=family, seed=seed, loss=adam_loss,
                                     adam_loss=adam_loss, cycles=0, cycles_total=0,
                                     converged=False, per_re={}, width=WIDTH,
                                     depth=DEPTH, lam=lam, lam_off=lam_off))

    lb = torch.optim.LBFGS(model.parameters(), lr=1.0, max_iter=lb_iter,
                           history_size=100, tolerance_grad=1e-14,
                           tolerance_change=1e-16, line_search_fn="strong_wolfe")

    def closure():
        lb.zero_grad()
        L = obj()
        L.backward()
        return L

    def grad_norm():
        model.zero_grad()
        L = obj()
        g = torch.autograd.grad(L, list(model.parameters()))
        return float(torch.sqrt(sum((x ** 2).sum() for x in g)))

    best = float(obj())
    state = {k: t.detach().clone() for k, t in model.state_dict().items()}
    gn = best_g = grad_norm()
    cycles, converged, stop_reason, stale = 0, False, "running", 0
    for cycles in range(1, max_cycles + 1):
        lb.step(closure)
        v = float(obj())
        if np.isfinite(v) and v < best:
            best = v
            state = {k: t.detach().clone() for k, t in model.state_dict().items()}
        gn = grad_norm()
        if gn < best_g * (1.0 - grel):          # a real improvement in optimality
            best_g, stale = gn, 0
        else:
            stale += 1
        if gn <= gtol:
            converged, stop_reason = True, "grad"
        elif stale >= patience:
            converged, stop_reason = True, "grad_plateau"
        if verbose:
            print(f"    cycle {cycles:>3}  loss {v:.6e}  best {best:.6e}  "
                  f"|g| {gn:.3e}  best|g| {best_g:.3e}  stale {stale}", flush=True)
        _save(save_path, model, dict(family=family, seed=seed, loss=best,
                                     adam_loss=adam_loss, cycles=cycles,
                                     cycles_total=cycles_done + cycles,
                                     converged=converged, stop_reason=stop_reason,
                                     grad_norm=gn, best_grad=best_g, gtol=gtol,
                                     per_re={}, width=WIDTH,
                                     depth=DEPTH, lam=lam, lam_off=lam_off), state=state)
        if converged:
            break
    model.load_state_dict(state)

    with torch.no_grad():
        per_re = {}
        for c in data:
            g = predict(model, family, c)
            per_re[c["retau"]] = float(torch.sqrt(
                torch.sum(c["W"] * (g - c["G"]) ** 2)
                / torch.sum(c["W"] * c["G"] ** 2)) * 100)
    return model, dict(family=family, seed=seed, loss=best, adam_loss=adam_loss,
                       cycles=cycles, cycles_total=cycles_done + cycles,
                       converged=converged, stop_reason=stop_reason, grad_norm=gn,
                       best_grad=best_g, gtol=gtol, per_re=per_re, width=WIDTH,
                       depth=DEPTH, lam=lam, lam_off=lam_off)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True, choices=FAMILIES)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--adam-epochs", type=int, default=3000)
    ap.add_argument("--max-cycles", type=int, default=2000)
    ap.add_argument("--gtol", type=float, default=1e-5,
                    help="stop when ||grad L||_2 falls below this")
    ap.add_argument("--ftol", type=float, default=1e-5)
    ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--grel", type=float, default=0.01,
                    help="fractional gradient improvement that counts as progress")
    ap.add_argument("--row-lambda", type=float, default=None,
                    help="override ROW_LAMBDA; used to ramp the constraint over stages")
    ap.add_argument("--off-lambda", type=float, default=None)
    ap.add_argument("--force-continue", action="store_true",
                    help="ignore the 'already converged' skip -- the next ramp stage has "
                         "a different objective, so a converged previous stage is the "
                         "starting point, not a reason to stop")
    ap.add_argument("--warm-from", default=None,
                    help="family whose trained seed-matched checkpoint seeds this run. "
                         "Two-stage training: fit the data FIRST with no penalty, then "
                         "turn the structural constraint on and continue.  Training a "
                         "_mm family from scratch let the penalty gradient dominate "
                         "while the model was still wildly wrong, and nueff_mm collapsed "
                         "to the trivial tau = 0, which satisfies every constraint and "
                         "fits nothing (a priori error 68-100%).")
    ap.add_argument("--lb-iter", type=int, default=500,
                    help="L-BFGS iterations per cycle; lower = cheaper "
                         "cycles and more frequent tolerance checks")
    a = ap.parse_args()
    os.makedirs(MODELS, exist_ok=True)
    path = os.path.join(MODELS, f"{a.family}_s{a.seed}.pt")
    prev = None
    if a.warm_from and not os.path.exists(path):
        w = os.path.join(MODELS, f"{a.warm_from}_s{a.seed}.pt")
        if not os.path.exists(w):
            raise SystemExit(f"--warm-from: no checkpoint {w}")
        wk = torch.load(w, weights_only=False)
        prev = {"state": wk["state"], "info": dict(wk["info"], cycles_total=0,
                                                   warm_from=a.warm_from)}
        print(f"warm start {a.family} s{a.seed} from {a.warm_from} "
              f"(|g| was {wk['info'].get('grad_norm', float('nan')):.3e})")
    if prev is None and os.path.exists(path):
        prev = torch.load(path, weights_only=False)
        if (prev["info"].get("converged") and prev["info"].get("stop_reason")
                and not a.force_continue):
            print(f"{a.family:>10} s{a.seed}  already converged "
                  f"(loss {prev['info']['loss']:.6e}, |g| "
                  f"{prev['info'].get('grad_norm', float('nan')):.3e}, "
                  f"{prev['info'].get('cycles_total')} cycles, "
                  f"{prev['info'].get('stop_reason')}) -- skipping")
            return
    model, info = train(a.family, a.seed, adam_epochs=a.adam_epochs,
                        max_cycles=a.max_cycles, gtol=a.gtol, ftol=a.ftol,
                        patience=a.patience, grel=a.grel, resume=prev, save_path=path,
                        row_lambda=a.row_lambda, off_lambda=a.off_lambda,
                        lb_iter=a.lb_iter)
    torch.save({"state": model.state_dict(), "info": info,
                "family": a.family, "width": WIDTH, "depth": DEPTH}, path)
    print(f"{a.family:>10} s{a.seed}  loss {info['loss']:.6e}  |g| {info['grad_norm']:.3e}  "
          f"cycles {info['cycles_total']}  {info['stop_reason']}  "
          f"dtau% " + " ".join(f"{info['per_re'][r]:.2f}" for r in CASES))


if __name__ == "__main__":
    main()
