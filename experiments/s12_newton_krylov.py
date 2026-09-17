#!/usr/bin/env python3
"""Problem-level vs solution-level conditioning, solved by Newton-Krylov.

PROBLEM LEVEL -- properties of J = A + B alone.  sigma_min(J), Wu et al.'s local K_j(y)
and K_x.  These belong to the discretised problem; no solver can change them.

SOLUTION LEVEL -- properties of P^-1 J, where P is the matrix actually inverted:

    e^{k+1} = (I - P^-1 J) e^k     for a STATIONARY (Picard) iteration

so rho(I - P^-1 J) is the analysis/solve mismatch.  But GMRES converges for any
nonsingular system -- rho > 1 does NOT predict Krylov failure -- so the measured GMRES
iteration count is reported alongside, and they are allowed to disagree.

  P = I            unpreconditioned
  P = A            explicit Picard
  P = A + B_diff   implicit analogue for the force closure (diffusive part of B folded
                   in, reaction left explicit)
  P = A + D(nu_t)  implicit Picard, hybrid only
  P = J            Newton (exact)

J is exactly tridiagonal here, so a direct solve is O(n) and the Krylov study is a PROXY
for 2D/3D where direct solves are infeasible: iteration counts are the transferable
quantity, not wall-clock.
"""
import argparse
import os
import sys

import numpy as np
import scipy.sparse.linalg as sla
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
torch.set_num_threads(1)

from chan import mesh as M, metrics as MET, operator as OP
from chan.cases import CASES, CACHE, MODELS, case_setup
from closure import jacobian as JAC
from closure.nets import (ClosureDiv, ClosureDivMono, ClosureHybrid,
                          ClosureHybrid2, ClosureNuEff, ClosureNuEffS,
                          ClosureTau)
from data import lm_channel as LM

CONFIGS = [("frozen", "FROZEN  (B=0)"),
           ("force", "A0  force, unpenalised"),
           ("force_pen", "A1(old)  force + penalty on [F_U]_+"),
           ("force_pen2", "A1  force + penalty on [F_U - 1/2 D1 F_S]_+, from scratch"),
           ("force_pen2w", "A1  force + penalty on [F_U - 1/2 D1 F_S]_+, warm from A0"),
           ("force_mm", "A2  force + structural, off-manifold"),
           ("hybrid", "B   hybrid nu_t + residual"),
           ("tau", "C   tau, differentiated"),
           ("tau_mm", "C1  tau + structural"),
           ("nueff", "C+  tau = -int nu_eff,  nu_eff >= 0"),
           ("nueff_mm", "C+1 nu_eff >= 0 + structural"),
           ("nueffs", "C++ nu_eff >= 0, no U input (M-matrix by construction)"),
           ("force_mono", "A4  force, dg/dU >= 0 by construction"),
           ("hybrid2", "B2  C+ flux (+) A4 residual, both architectural")]
NSEED = 10

# residual level below which a stalled iteration counts as converged; the solution
# error it implies is rn/sigma_min, so eps_U is still reported and remains the
# arbiter of whether the root found is any good.
STALL_LEVEL = 1e-8


def load(family, seed):
    ck = torch.load(os.path.join(MODELS, f"{family}_s{seed}.pt"), weights_only=False)
    cls = (ClosureHybrid2 if family == "hybrid2" else
           ClosureHybrid if family == "hybrid" else
           ClosureDivMono if family == "force_mono" else
           ClosureNuEffS if family.startswith("nueffs") else   # before "nueff"
           ClosureNuEff if family.startswith("nueff") else
           ClosureTau if family.startswith("tau") else ClosureDiv)
    net = cls(ck["width"], ck["depth"]).double()
    net.load_state_dict(ck["state"])
    net.eval()
    return net, ck["info"]


def residual(family, model, m, U, rt, A, src):
    if family.startswith("hybrid"):
        return JAC.hybrid_residual(m, U, model, rt, A=A)
    if family.startswith(("nueff", "tau")):   # flux form: R = src + A U - div(tau)
        return JAC.residual_uv(m, U, model, rt, A=A, src=src)
    return JAC.residual_div(m, U, model, rt, A=A, src=src)


def operators(family, model, c, rt, U):
    """(J, preconditioners) at state U.  Every P here is tridiagonal."""
    m, A = c["m"], c["A"]
    P = {"A": A}
    if family.startswith(("nueff", "tau")):
        # for nueff, nu_eff >= 0 by construction, so P = A + D(nu_eff) is the implicit-Picard
        # analogue here -- and unlike the hybrid's nu_t it is the coefficient that
        # actually reaches the Jacobian.
        _, a, b = JAC.tau_jacobian_faces(model, m, U, rt)
        J = A + JAC.assemble_B_uv(m, a, b)
        ne = JAC.nu_t_eff_uv(m, a, b)
        ne[0] = ne[-1] = 0.0
        P["A+nu_eff"] = A + OP.diffusion(m, ne)
        P["J"] = J
        return J, P
    if family.startswith("hybrid"):
        J, B_nut, B_res = JAC.hybrid_jacobian(m, U, model, rt, A=A)
        S = torch.tensor(JAC.dUdy_faces(m, U), dtype=torch.float64)
        Uf = torch.tensor(OP.face_interp(m, U), dtype=torch.float64)
        with torch.no_grad():
            # B2 exposes the TANGENT nu_eff, which is the coefficient that actually
            # reaches the Jacobian; B exposes only the secant nu_t.  Using each family's
            # own object is the point of the comparison.
            coef = model.nu_eff if hasattr(model, "nu_eff") else model.nu_t
            nut = coef(Uf, S, 1.0 / rt, rt).numpy().copy()
        nut[0] = nut[-1] = 0.0
        P["A+nu_t"] = A + OP.diffusion(m, nut)
    else:
        _, dl, dc, dr = JAC.dtau_dy_jacobian(model, m, U, rt)
        B_full, B_diff, _, _, _ = MET.split_B(m, dl, dc, dr)
        J = A + B_full
        P["A+B_diff"] = A + B_diff
    P["J"] = J
    return J, P


def initial_guess(kind, c):
    """Starting field for a solve.  `dns` is NOT available in deployment.

    Every solve in this study originally started from U_DNS, which is exactly the field a
    closure is supposed to predict.  That makes the reported convergence optimistic in a
    way that cannot be reproduced in practice, so we also measure two guesses that need no
    high-fidelity data at all:

      zero     : U = 0.
      laminar  : A U = -src, i.e. the closure switched off entirely.  This is the profile
                 y(2-y) Re_tau/2, whose centreline value is Re_tau/2 -- about 2600 at
                 Re_tau = 5200 against a true U_max near 25.  It is a POOR guess by
                 construction, and deliberately so: it is what an honest cold start looks
                 like, and it is the correct lambda = 0 anchor for a continuation that uses
                 no DNS (see s17).
    """
    if not isinstance(kind, str):          # an explicit field, e.g. from a coarser Re
        return np.asarray(kind, float).copy()
    if kind == "dns":
        return c["U_dns"].copy()
    if kind == "zero":
        return np.zeros_like(c["U_dns"])
    if kind == "laminar":
        return np.linalg.solve(c["A"], -OP.source(c["m"]))
    raise ValueError(f"unknown initial guess {kind!r}")


def newton_krylov(family, model, c, rt, precond="J", itmax=60, tol=1e-11,
                  gtol=1e-10, u0="dns"):
    """Damped Newton with a GMRES inner solve preconditioned by `precond`.

    Returns (converged, newton_steps, gmres_iters_per_step, U).
    """
    m, A, src = c["m"], c["A"], OP.source(c["m"])
    U = initial_guess(u0, c)
    ref = M.wnorm(src / m.dV, m.dV)
    counts = []
    best_rn, stall = np.inf, 0
    for _ in range(itmax):
        R = residual(family, model, m, U, rt, A, src)
        rn = M.wnorm(R / m.dV, m.dV) / ref
        if rn < tol:
            return True, len(counts), counts, U
        # A FIXED absolute tolerance is wrong: the attainable residual floor is set by
        # round-off in forming src + A U - F(U) and grows with the mesh -- measured
        # 4e-13 at n=110 and 4.2e-11 at n=1040.  Testing rn < 1e-11 therefore marked
        # every one of ten B1 seeds at Re_tau=5200 as FAILED while they sat at 4.1e-11
        # with eps_U = 0.019%, the most accurate solutions in the study.  Modelling the
        # floor does not work either (eps*||AU||/ref is off by 363x to 13000x, and the
        # ratio itself varies 36x across meshes), so detect the stall instead: a residual
        # that is already negligible and has stopped decreasing IS converged.
        if rn < STALL_LEVEL and rn > 0.9 * best_rn:
            stall += 1
            if stall >= 2:
                return True, len(counts), counts, U
        else:
            stall = 0
        best_rn = min(best_rn, rn)
        J, P = operators(family, model, c, rt, U)
        Pm = P[precond]
        lu = None if precond == "I" else np.linalg.inv(Pm)
        Mop = None if lu is None else sla.LinearOperator((m.n, m.n), matvec=lu.dot)
        it = [0]
        dU, _ = sla.gmres(J, -R, M=Mop, rtol=gtol, restart=min(m.n, 200), maxiter=5,
                          callback=lambda *_a: it.__setitem__(0, it[0] + 1),
                          callback_type="pr_norm")
        counts.append(it[0])
        if not np.all(np.isfinite(dU)):
            return False, len(counts), counts, U
        for d in (1.0, 0.5, 0.25, 0.1, 0.03, 1e-2, 3e-3):
            Un = U + d * dU
            if not np.all(np.isfinite(Un)):
                continue
            if M.wnorm(residual(family, model, m, Un, rt, A, src) / m.dV, m.dV) / ref < rn:
                U = Un
                break
        else:
            return False, len(counts), counts, U
    return False, len(counts), counts, U


def fixed_point(family, model, c, rt, itmax=200, tol=1e-12):
    """Direct-solve Newton to the fixed point.  J is tridiagonal, so this is exact and
    cheap; every rho and every operator property below is evaluated HERE, because those
    are LOCAL quantities and the DNS field is several percent away."""
    m, A, src = c["m"], c["A"], OP.source(c["m"])
    U = c["U_dns"].copy()
    ref = M.wnorm(src / m.dV, m.dV)
    for _ in range(itmax):
        R = residual(family, model, m, U, rt, A, src)
        rn = M.wnorm(R / m.dV, m.dV) / ref
        if rn < tol:
            return U, True
        J, _ = operators(family, model, c, rt, U)
        try:
            dU = np.linalg.solve(J, -R)
        except np.linalg.LinAlgError:
            return U, False
        if M.wnorm(dU, m.dV) < 1e-14 * M.wnorm(U, m.dV):
            return U, rn < tol
        for d in (1.0, 0.5, 0.25, 0.1, 0.03, 1e-2, 3e-3, 1e-3):
            Un = U + d * dU
            if not np.all(np.isfinite(Un)):
                continue
            if M.wnorm(residual(family, model, m, Un, rt, A, src) / m.dV, m.dV) / ref < rn:
                U = Un
                break
        else:
            return U, False
    return U, False


def analyse(family, seed, retau):
    c = case_setup(retau)
    m = c["m"]
    rt = LM.load(retau)["retau"]
    if family == "frozen":
        J, P, model, Us, fok = c["A"], {"A": c["A"], "J": c["A"]}, \
            None, c["U_dns"], True
    else:
        model, _ = load(family, seed)
        Us, fok = fixed_point(family, model, c, rt)
        J, P = operators(family, model, c, rt, Us)

    Jt = MET.sym_weighted(J, m.dV)
    sv = np.linalg.svd(Jt, compute_uv=False)
    Kj = MET.K_local(J, m.dV, c["div_tau"], c["U_ref"])
    rec = dict(family=family, seed=seed, retau=retau, fp_ok=fok,
               fp_err_U=float(M.wnorm(Us - c["U_dns"], m.dV) /
                              M.wnorm(c["U_dns"], m.dV)),
               sigma_min=float(sv.min()), cond=float(sv.max() / sv.min()),
               re_max=float(np.linalg.eigvals(Jt).real.max()),
               symmetric=bool(np.allclose(J, J.T, atol=1e-12 * np.abs(J).max())),
               Kj=Kj, Kx=MET.K_volume(Kj, m.dV), y=m.yc, n=m.n, precond={})

    for name, Pm in P.items():
        M_ = np.linalg.solve(Pm, J)
        lo, hi, nrm = MET.fov_bounds(M_)
        entry = dict(rho=float(np.abs(np.linalg.eigvals(np.eye(m.n) - M_)).max()),
                     cond=float(np.linalg.cond(M_)), fov_lo=lo, fov_hi=hi, norm=nrm,
                     elman10=MET.elman_bound(M_, 10))
        if family != "frozen":
            ok, steps, counts, Uk = newton_krylov(family, model, c, rt, precond=name)
            # a posteriori error of the SOLVED profile against DNS, in the same
            # sqrt(dV)-weighted norm used everywhere else.  Only meaningful when the
            # solve converged; recorded regardless so the filter stays in the figure.
            entry.update(nk_ok=ok, nk_steps=steps,
                         err_U=float(M.wnorm(Uk - c["U_dns"], m.dV) /
                                     M.wnorm(c["U_dns"], m.dV)),
                         gmres_mean=float(np.mean(counts)) if counts else np.nan,
                         gmres_max=int(np.max(counts)) if counts else -1)
        rec["precond"][name] = entry
    return rec


def main(cases, seeds=NSEED, families=None, tag=""):
    out = []
    cfg = [(f, l) for f, l in CONFIGS if families is None or f in families]
    for family, lab in cfg:
        ns = 1 if family == "frozen" else seeds
        print(f"\n{'='*100}\n{lab}\n{'='*100}")
        print("PROBLEM LEVEL (at the fixed point)")
        print(f"{'Re':>6} {'sigma_min(J)':>13} {'cond(J)':>11} {'max Re(lam)':>12} "
              f"{'K_x':>11} {'sym':>5} {'fp ok':>6}")
        recs = {}
        for r in cases:
            rs = [analyse(family, s, r) for s in range(ns)]
            recs[r] = rs
            out.extend(rs)
            g = lambda k: float(np.median([x[k] for x in rs]))
            print(f"{r:>6} {g('sigma_min'):>13.4e} {g('cond'):>11.3e} "
                  f"{g('re_max'):>+12.3e} {g('Kx'):>11.4e} "
                  f"{str(rs[0]['symmetric']):>5} {str(all(x['fp_ok'] for x in rs)):>6}")

        print("\nSOLUTION LEVEL   rho = rho(I - P^-1 J);  GMRES = mean iters / Newton step")
        print(f"{'Re':>6} {'P':>10} {'rho(I-P^-1J)':>13} {'cond(P^-1J)':>12} "
              f"{'ReW(P^-1J)':>22} {'Elman@10':>10} {'GMRES':>7} {'NK steps':>9} {'NK ok':>6}")
        for r in cases:
            for name in recs[r][0]["precond"]:
                e = [x["precond"][name] for x in recs[r]]
                g = lambda k: float(np.median([x[k] for x in e]))
                gm = f"{g('gmres_mean'):>7.1f}" if "gmres_mean" in e[0] else f"{'-':>7}"
                nk = f"{int(np.median([x['nk_steps'] for x in e])):>9}" \
                    if "nk_steps" in e[0] else f"{'-':>9}"
                ok = f"{sum(x['nk_ok'] for x in e)}/{len(e)}" if "nk_ok" in e[0] else "-"
                print(f"{r:>6} {name:>10} {g('rho'):>13.4e} {g('cond'):>12.3e} "
                      f"[{g('fov_lo'):>9.2e},{g('fov_hi'):>9.2e}] {g('elman10'):>10.4f} "
                      f"{gm} {nk} {ok:>6}")
    np.save(os.path.join(CACHE, f"s12{tag}_{'_'.join(map(str, cases))}.npy"),
            np.array(out, dtype=object))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cases", nargs="?", default="180,550,1000,2000,5200")
    ap.add_argument("--seeds", type=int, default=NSEED)
    ap.add_argument("--families", default=None,
                    help="comma list; restrict the run so a newly added closure can be "
                         "measured without recomputing the families already cached")
    ap.add_argument("--tag", default="",
                    help="suffix for the cache file, e.g. --tag b -> s12b_1000.npy")
    a = ap.parse_args()
    fams = a.families.split(",") if a.families else None
    main(tuple(int(x) for x in a.cases.split(",")), seeds=a.seeds,
         families=fams, tag=a.tag)
