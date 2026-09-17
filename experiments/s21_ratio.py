"""How much of the a priori error the operator amplifies, against how much it is ALLOWED to.

The chain is an identity.  The true solution satisfies the true equation, so the residual of
the MODELLED equation evaluated at the true solution is the a priori error itself:

    r := R_model(u_ref) = -dV ( g_NN(u_ref) - g_true ) .

R_model(U_hat) = 0 by definition, so to first order  e = -J^-1 r,  and in the weighted space
Jt = D^-1/2 J D^-1/2  (with  ehat = D^1/2 e,  rhat = D^-1/2 r)  the bound is exact linear
algebra:

    ||ehat||  <=  ||rhat|| / sigma_min(Jt) .

This script measures three things per (closure, Re_tau, seed):

    BOUND      1 / sigma_min(Jt)                    the worst case, attained only if r aligns
                                                    with the smallest singular direction
    LINEAR     ||J^-1 r|| / ||r||                   what the operator actually does to THIS r
    REALISED   ||U_hat - u_ref|| / ||r||            the same ratio from the converged solve,
                                                    which includes the O(|e|^2) the bound drops

and the LOOSENESS = BOUND / LINEAR, i.e. the alignment factor.  Reporting the bound alone
invites the reading that better conditioning means a smaller error; it does not, because a
smooth physical residual is nearly orthogonal to the smallest singular direction and the
looseness runs to five orders of magnitude.

REALISED is read from the s20 cache (best_err, relative to ||u_ref||) rather than re-solved,
so this script needs no Newton solve at all and runs in about a minute.
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from chan import mesh as M, metrics as MET, operator as OP
from chan.cases import CASES, CACHE, REPORTED, SHORT, case_setup
from data import lm_channel as LM
from s12_newton_krylov import load, residual, operators


def frozen_sigma_min(c):
    """sigma_min of the frozen operator, B = 0.  The reference every ratio is taken against."""
    return float(np.linalg.svd(MET.sym_weighted(c["A"], c["m"].dV), compute_uv=False).min())


def main(families, seeds, cases, tag=""):
    s20 = {}
    for f in families:
        q = os.path.join(CACHE, f"s20_{f}.npy")
        if os.path.exists(q):
            for o in np.load(q, allow_pickle=True):
                s20[(o["family"], o["retau"], o["seed"])] = o
    out = []
    for retau in cases:
        c = case_setup(retau)
        m, A = c["m"], c["A"]
        dV, src = m.dV, OP.source(m)
        rt = LM.load(retau)["retau"]
        sf = frozen_sigma_min(c)
        Un = M.wnorm(c["U_dns"], dV)
        print(f"\nRe_tau={retau}  n={m.n}  sigma_min(frozen)={sf:.4e}")
        print(f"{'':>4} {'seed':>4} {'||rhat||':>10} {'sigma_min':>10} {'x frozen':>9} "
              f"{'BOUND':>10} {'LINEAR':>10} {'REALISED':>10} {'loose':>10}")
        for f in families:
            for s in range(seeds):
                model, _ = load(f, s)
                r = residual(f, model, m, c["U_dns"], rt, A, src)
                J, _ = operators(f, model, c, rt, c["U_dns"])
                Jt = MET.sym_weighted(J, dV)
                sv = np.linalg.svd(Jt, compute_uv=False)
                sig = float(sv.min())
                rhat = MET.to_hat(r, dV)
                rn = float(np.linalg.norm(rhat))
                elin = float(np.linalg.norm(np.linalg.solve(Jt, rhat)))
                rec = s20.get((f, retau, s))
                real = (float(rec["best_err"]) * Un / rn
                        if rec is not None and rec["any_ok"]
                        and np.isfinite(rec["best_err"]) else np.nan)
                out.append(dict(family=f, retau=retau, seed=s, rnorm=rn, sigma_min=sig,
                                sigma_frozen=sf, bound=1.0 / sig, linear=elin / rn,
                                realized=real, loose=(1.0 / sig) / (elin / rn),
                                apriori=rn / Un))
                print(f"{SHORT.get(f,f):>4} {s:>4} {rn:10.3e} {sig:10.3e} "
                      f"{sig/sf:9.2f} {1/sig:10.3e} {elin/rn:10.3e} "
                      f"{real:10.3e} {(1/sig)/(elin/rn):10.3e}", flush=True)
    q = os.path.join(CACHE, f"s21{tag}_ratio.npy")
    np.save(q, np.array(out, dtype=object), allow_pickle=True)
    n = sum(1 for o in out if np.isfinite(o["realized"]))
    held = sum(1 for o in out if np.isfinite(o["realized"]) and o["realized"] <= o["bound"])
    print(f"\nbound held on {held}/{n} converged solves")
    print("wrote", q)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--families", default=",".join(f for f in REPORTED if f != "frozen"))
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--cases", default=",".join(str(c) for c in CASES))
    ap.add_argument("--tag", default="", help="cache suffix, e.g. --tag _a1 -> s21_a1_ratio.npy")
    a = ap.parse_args()
    import torch
    torch.set_num_threads(1)
    main(a.families.split(","), a.seeds, [int(x) for x in a.cases.split(",")], tag=a.tag)
