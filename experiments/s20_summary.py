"""The whole study, reduced to what we actually report.

Seven closures x ten seeds x five Reynolds numbers, and for each:

  PROBLEM LEVEL, at the reference state, no solve
      omega(J)      = lambda_max(sym J~)   the numerical abscissa; omega < 0 is coercivity,
                      and sigma_min >= -omega, so this is a computable distance-to-singular
      sigma_min(J~) = the distance to the nearest singular operator, and 1/sigma_min is the
                      worst-case amplification of a priori into a posteriori error
      both on J~ = D^-1/2 J D^-1/2, which is similar to D^-1 J and mesh-convergent

  SOLUTION LEVEL, with P = A (explicit Picard) and NO high-fidelity data
      cold       damped Newton from U = 0
      Re-ladder  ascending in Re_tau, each solution interpolated onto the next mesh

Everything else has been dropped and the reasons are recorded in make_tables.FREE: the DNS
start (unavailable, and measured WORSE than a cold start in several cells), the laminar
start (77x from the answer at Re_tau=5200, erratic), lambda-continuation (10/10 at
Re_tau=180 and 0-2/10 above it) and psi-tc+SER (rescued nothing Newton failed, and needs a
problem-dependent cfl0).

A closure is credited at a given Re_tau if EITHER surviving strategy converges; only if
both fail on all ten seeds is the cell marked as no root found -- and even then that is a
statement about two searches, not about the equation.
"""

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from chan import mesh as M, metrics as MET, operator as OP
from chan.cases import CASES, CACHE, REPORTED, SHORT, case_setup
from closure import jacobian as JAC
from data import lm_channel as LM
from s12_newton_krylov import load, newton_krylov, operators


def problem_level(family, model, c, rt):
    """omega and sigma_min of J at the REFERENCE state.  No solve, no DNS beyond u_ref."""
    m = c["m"]
    J, _ = operators(family, model, c, rt, c["U_dns"])
    Jt = MET.sym_weighted(J, m.dV)
    sv = np.linalg.svd(Jt, compute_uv=False)
    return dict(omega=JAC.numerical_abscissa(Jt),
                spec=float(np.linalg.eigvals(Jt).real.max()),
                sigma_min=float(sv.min()),
                rowsum=float(J.sum(1).max()),
                offdiag=float((J - np.diag(np.diag(J)))[
                    np.abs(J - np.diag(np.diag(J))) > 0].min()))


def ladder(family, seeds, cases):
    """Reynolds continuation: ascending, interpolating each solution onto the next mesh."""
    out = {}
    for s in range(seeds):
        carried = None
        for retau in sorted(cases):
            c = case_setup(retau)
            m = c["m"]
            rt = LM.load(retau)["retau"]
            u0 = "zero" if carried is None else np.interp(m.yc, carried[1], carried[0])
            model, _ = load(family, s)
            ok, st, _, U = newton_krylov(family, model, c, rt, precond="A", u0=u0)
            e = (M.wnorm(U - c["U_dns"], m.dV) / M.wnorm(c["U_dns"], m.dV)
                 if ok else np.nan)
            out[(retau, s)] = (bool(ok), float(e), int(st))
            carried = (U, m.yc, retau) if ok else None   # a failed rung breaks the chain
    return out


def main(families, seeds, cases):
    out = []
    for family in families:
        t_fam = time.time()
        lad = ladder(family, seeds, cases)
        for retau in cases:
            c = case_setup(retau)
            m = c["m"]
            rt = LM.load(retau)["retau"]
            Un = M.wnorm(c["U_dns"], m.dV)
            for s in range(seeds):
                model, _ = load(family, s)
                rec = dict(family=family, retau=retau, seed=s,
                           **problem_level(family, model, c, rt))
                ok, st, _, U = newton_krylov(family, model, c, rt, precond="A", u0="zero")
                rec.update(cold_ok=bool(ok), cold_steps=int(st),
                           cold_err=float(M.wnorm(U - c["U_dns"], m.dV) / Un
                                          if ok else np.nan))
                lo, le, ls = lad[(retau, s)]
                rec.update(lad_ok=lo, lad_err=le, lad_steps=ls)
                any_ok = rec["cold_ok"] or rec["lad_ok"]
                errs = [e for o, e in ((rec["cold_ok"], rec["cold_err"]),
                                       (rec["lad_ok"], rec["lad_err"])) if o]
                rec.update(any_ok=bool(any_ok),
                           best_err=float(min(errs)) if errs else float("nan"))
                out.append(rec)
            sub = [o for o in out if o["family"] == family and o["retau"] == retau]
            g = lambda k: sum(o[k] for o in sub)
            e = [o["best_err"] for o in sub if o["any_ok"]]
            print(f"  {SHORT.get(family, family):>4} {family:>11} Re={retau:>5}  "
                  f"cold {g('cold_ok'):>2}/10  ladder {g('lad_ok'):>2}/10  "
                  f"union {g('any_ok'):>2}/10   "
                  f"eps_U {100*np.median(e):7.3f}%" if e else
                  f"  {SHORT.get(family, family):>4} {family:>11} Re={retau:>5}  "
                  f"cold {g('cold_ok'):>2}/10  ladder {g('lad_ok'):>2}/10  "
                  f"union {g('any_ok'):>2}/10   NO ROOT FOUND", flush=True)
        print(f"  ... {family} took {time.time()-t_fam:.0f}s\n", flush=True)
    q = os.path.join(CACHE, f"s20_{'_'.join(families)}.npy")
    np.save(q, np.array(out, dtype=object), allow_pickle=True)
    print("wrote", q)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--families", default=",".join(f for f in REPORTED if f != "frozen"))
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--cases", default=",".join(str(c) for c in CASES))
    a = ap.parse_args()
    import torch
    torch.set_num_threads(1)
    main(a.families.split(","), a.seeds, [int(x) for x in a.cases.split(",")])
