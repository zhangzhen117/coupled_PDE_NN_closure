#!/usr/bin/env python3
"""Gates for the Newton-Krylov study.

Every number in the solution-level tables is a function of the preconditioned operator or
of GMRES, so these run before the study does.
"""
import os
import sys

import numpy as np
import scipy.sparse.linalg as sla
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
torch.set_num_threads(1)
from chan import mesh as M, metrics as MET, operator as OP
from chan.cases import case_setup
from closure import jacobian as JAC
from closure.nets import ClosureDiv
from data import lm_channel as LM
from tests.test_operator import _run

RETAU = 1000.0


def _case(seed=1):
    c = case_setup(1000)
    mdl = ClosureDiv(48, 2, seed).double()
    rt = LM.load(1000)["retau"]
    _, dl, dc, dr = JAC.dtau_dy_jacobian(mdl, c["m"], c["U_dns"], rt)
    J = c["A"] + JAC.assemble_B_div(c["m"], dl, dc, dr)
    return c, mdl, rt, J


def test_exact_preconditioner_converges_in_one_iteration():
    """P = J is the exact preconditioner, so GMRES must terminate immediately.  If this
    fails, the preconditioner is not being applied the way the tables assume."""
    c, _, _, J = _case()
    n = c["m"].n
    b = np.random.default_rng(0).standard_normal(n)
    Minv = sla.LinearOperator((n, n), matvec=np.linalg.inv(J).dot)
    it = [0]
    x, _ = sla.gmres(J, b, M=Minv, rtol=1e-12, restart=n, maxiter=50,
                     callback=lambda *_a: it.__setitem__(0, it[0] + 1),
                     callback_type="pr_norm")
    assert it[0] <= 1, f"exact preconditioner took {it[0]} GMRES iterations"
    assert np.linalg.norm(J @ x - b) / np.linalg.norm(b) < 1e-10


def test_krylov_solution_matches_direct():
    """J is tridiagonal, so the direct solve is exact -- any disagreement is a Krylov bug,
    not a conditioning effect."""
    c, _, _, J = _case()
    n = c["m"].n
    b = np.random.default_rng(1).standard_normal(n)
    Minv = sla.LinearOperator((n, n), matvec=np.linalg.inv(c["A"]).dot)
    x, _ = sla.gmres(J, b, M=Minv, rtol=1e-13, restart=n, maxiter=2000)
    assert np.linalg.norm(x - np.linalg.solve(J, b)) / np.linalg.norm(x) < 1e-9


def test_rho_identity_matches_stage3_criterion():
    """rho(I - A^-1 J) == rho(A^-1 B), tying the general splitting identity to the
    original explicit-Picard criterion."""
    c, _, _, J = _case()
    A = c["A"]
    B = J - A
    lhs = np.abs(np.linalg.eigvals(np.eye(c["m"].n) - np.linalg.solve(A, J))).max()
    rhs = np.abs(np.linalg.eigvals(np.linalg.solve(A, B))).max()
    assert abs(lhs - rhs) <= 1e-12 * max(1.0, rhs), (lhs, rhs)


def test_elman_bounds_measured_gmres():
    """Where 0 is not in the field of values, the Elman estimate must UPPER-bound the
    measured GMRES residual at every step.

    Uses a controlled operator with a definite Hermitian part, because for the real
    preconditioned closure operators 0 usually IS in W -- see
    `test_real_case_field_of_values_straddles_zero`, which records that as a fact rather
    than working around it.
    """
    rng = np.random.default_rng(3)
    n = 60
    Mmat = np.eye(n) + 0.25 * rng.standard_normal((n, n)) / np.sqrt(n)
    lo, hi, _ = MET.fov_bounds(Mmat)
    assert lo > 0, "constructed operator should have a positive-definite Hermitian part"
    b = rng.standard_normal(n)
    hist = []
    sla.gmres(Mmat, b, rtol=1e-14, restart=n, maxiter=40,
              callback=lambda r: hist.append(r), callback_type="pr_norm")
    r0 = hist[0]
    for k, rk in enumerate(hist[1:], start=1):
        bound = MET.elman_bound(Mmat, k)
        assert rk / r0 <= bound * (1 + 1e-8), (k, rk / r0, bound)


def test_real_case_field_of_values_straddles_zero():
    """A property of the actual problem, asserted so it cannot silently change: for the
    real preconditioned operator A^-1 J, the field of values contains 0, so the Elman
    bound does NOT apply and `elman_bound` must report inf rather than a finite number.
    """
    c, _, _, J = _case()
    Mmat = np.linalg.solve(c["A"], J)
    lo, hi, _ = MET.fov_bounds(Mmat)
    assert lo * hi <= 0, (lo, hi)
    assert MET.elman_bound(Mmat, 10) == np.inf


def test_split_B_reconstructs():
    """B_diff + B_react == B, and B_diff is what the implicit-analogue preconditioner
    folds in."""
    c, mdl, rt, J = _case()
    _, dl, dc, dr = JAC.dtau_dy_jacobian(mdl, c["m"], c["U_dns"], rt)
    Bf, Bd, Br, G_U, G_S = MET.split_B(c["m"], dl, dc, dr)
    assert np.allclose(Bf, Bd + Br, atol=1e-13)
    assert np.allclose(Bf, J - c["A"], atol=1e-13)
    assert np.abs(Br - np.diag(np.diag(Br))).max() < 1e-13, "reaction must be diagonal"


if __name__ == "__main__":
    sys.exit(min(_run(globals()), 1))
