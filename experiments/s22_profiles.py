"""Fields for the two figures that need more than a scalar per run.

MULTIMODALITY.  A closure that is not constrained can be solved to a clean residual and
still land on a root that is not the one wanted.  Convergence is then not evidence of
anything: it is the solver reporting, correctly, that it has found A root of the equation
the network wrote down.  This saves the converged U(y) for every (closure, seed, start) so
the spurious branches can be drawn next to the physical one.

SECANT AGAINST TANGENT.  C0 enforces nu_t >= 0 by softplus.  The operator does not contain
nu_t; it contains the tangent

    nu_eff = -d(tau)/dS = nu_t + S d(nu_t)/dS ,

and nothing constrains the second term.  This saves both profiles at the reference state so
the gap can be shown rather than asserted.
"""

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from chan import mesh as M, operator as OP
from chan.cases import CASES, CACHE, REPORTED, SHORT, case_setup
from closure import jacobian as JAC
from data import lm_channel as LM
from s12_newton_krylov import load, newton_krylov


def solved_fields(families, seeds, cases):
    """Converged U for cold start and for the Re-ladder, both DNS-free."""
    out = []
    lad = {}
    for f in families:
        for s in range(seeds):
            carried = None
            for retau in sorted(cases):
                c = case_setup(retau)
                rt = LM.load(retau)["retau"]
                u0 = "zero" if carried is None else np.interp(c["m"].yc, carried[1],
                                                              carried[0])
                model, _ = load(f, s)
                ok, st, _, U = newton_krylov(f, model, c, rt, precond="A", u0=u0)
                lad[(f, retau, s)] = (bool(ok), U.copy())
                carried = (U, c["m"].yc) if ok else None
    for retau in cases:
        c = case_setup(retau)
        rt = LM.load(retau)["retau"]
        m = c["m"]
        Un = M.wnorm(c["U_dns"], m.dV)
        for f in families:
            for s in range(seeds):
                model, _ = load(f, s)
                ok, st, _, U = newton_krylov(f, model, c, rt, precond="A", u0="zero")
                lo, Ul = lad[(f, retau, s)]
                for tag, o, Uk in (("cold", ok, U), ("ladder", lo, Ul)):
                    e = M.wnorm(Uk - c["U_dns"], m.dV) / Un
                    out.append(dict(family=f, retau=retau, seed=s, start=tag,
                                    ok=bool(o), err=float(e) if o else float("nan"),
                                    y=m.yc.copy(), U=Uk.copy()))
                print(f"  {SHORT.get(f,f):>4} Re={retau:>5} seed {s} "
                      f"cold {'ok' if ok else '--'} "
                      f"{100*M.wnorm(U-c['U_dns'],m.dV)/Un:8.2f}%   "
                      f"ladder {'ok' if lo else '--'} "
                      f"{100*M.wnorm(Ul-c['U_dns'],m.dV)/Un:8.2f}%", flush=True)
    return out


def secant_vs_tangent(seeds, cases, family="hybrid"):
    """nu_t (constrained, >= 0) and nu_eff = nu_t + S d(nu_t)/dS (free) at the faces."""
    out = []
    for retau in cases:
        c = case_setup(retau)
        m = c["m"]
        rt = LM.load(retau)["retau"]
        S = torch.tensor(JAC.dUdy_faces(m, c["U_dns"]), dtype=torch.float64,
                         requires_grad=True)
        Uf = torch.tensor(OP.face_interp(m, c["U_dns"]), dtype=torch.float64)
        for s in range(seeds):
            model, _ = load(family, s)
            nut = model.nu_t(Uf, S, 1.0 / rt, rt)
            g, = torch.autograd.grad(nut.sum(), S)
            nue = (nut + S * g).detach().numpy()
            out.append(dict(family=family, retau=retau, seed=s, yf=m.yf.copy(),
                            nu_t=nut.detach().numpy().copy(), nu_eff=nue.copy(),
                            nu=c["nu"]))
            print(f"  {SHORT.get(family,family)} Re={retau:>5} seed {s}  "
                  f"min nu_t {nut.min().item():11.3e}   min nu_eff {nue.min():11.3e}",
                  flush=True)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--families", default=",".join(f for f in REPORTED if f != "frozen"))
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--cases", default=",".join(str(c) for c in CASES))
    ap.add_argument("--tag", default="", help="cache suffix, e.g. --tag _a1 -> s22_a1_fields.npy")
    a = ap.parse_args()
    torch.set_num_threads(1)
    fams = a.families.split(",")
    cs = [int(x) for x in a.cases.split(",")]
    print("=== solved fields ===")
    np.save(os.path.join(CACHE, f"s22{a.tag}_fields.npy"),
            np.array(solved_fields(fams, a.seeds, cs), dtype=object), allow_pickle=True)
    print("\n=== secant vs tangent ===")
    np.save(os.path.join(CACHE, f"s22{a.tag}_nueff.npy"),
            np.array(secant_vs_tangent(a.seeds, cs), dtype=object), allow_pickle=True)
    print(f"\nwrote s22{a.tag}_fields.npy, s22{a.tag}_nueff.npy")
