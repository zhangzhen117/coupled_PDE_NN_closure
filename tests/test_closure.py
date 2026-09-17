#!/usr/bin/env python3
"""Stage 3 contract: nu_t_eff, B, and the segregated iteration matrix."""
import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
torch.set_num_threads(1)
from chan import mesh as M, operator as OP, metrics as MET
from closure import jacobian as JAC
from closure.nets import Closure
from data import lm_channel as LM
from tests.test_operator import _run


class ConstNut(torch.nn.Module):
    """tau = -c * dU/dy with c constant: nu_t_eff must come back exactly c."""
    def __init__(self, c):
        super().__init__()
        self.c = c
        self.arm = "const"

    def nu_t(self, feat):
        return torch.full(feat.shape[:-1], self.c, dtype=feat.dtype)

    def tau(self, dUdy, d, nu):
        return -self.c * dUdy

    def double(self):
        return self


def test_nu_t_eff_of_constant_eddy_viscosity():
    """The calibration case.  nu_t_eff = -dtau/dS = c exactly, for every face."""
    m = M.for_case(1000)
    U = np.sin(np.pi * m.yc / 2.0)
    for c in (0.0, 1e-3, 0.05):
        ne, nt, _ = JAC.nu_t_eff_faces(ConstNut(c), m, U, 1e-3, wall_zero=False)
        assert np.allclose(ne, c, atol=1e-12), (c, ne.min(), ne.max())
        assert np.allclose(nt, c, atol=1e-12)


def test_nu_t_eff_matches_finite_difference():
    """Autodiff nu_t_eff against a central difference of the arm's own tau()."""
    m = M.stretched(80, 1000, 0.7)
    d = LM.load(1000)
    from scipy.interpolate import PchipInterpolator
    U = PchipInterpolator(d["y"], d["U"])(m.yc)
    for arm in ("A1", "A3"):
        mod = Closure(arm, seed=3).double()
        ne, _, _ = JAC.nu_t_eff_faces(mod, m, U, d["nu"], wall_zero=False)
        S = torch.tensor(JAC.dUdy_faces(m, U), dtype=torch.float64)
        dd = torch.tensor(np.minimum(m.yf, 2 - m.yf), dtype=torch.float64)
        h = 1e-6 * torch.clamp(torch.abs(S), min=1.0)
        with torch.no_grad():
            fd = -(mod.tau(S + h, dd, d["nu"]) - mod.tau(S - h, dd, d["nu"])) / (2 * h)
        rel = np.abs(ne - fd.numpy()) / (np.abs(ne) + 1e-12)
        assert rel.max() < 1e-5, (arm, rel.max())


def test_A_plus_B_from_closure_is_the_same_diffusion_operator():
    """CONVENTIONS.md again, but now with B coming from a real network."""
    m = M.for_case(550)
    nu = 1 / 550.0
    U = np.sin(np.pi * m.yc / 2.0) * 10
    mod = Closure("A1", seed=1).double()
    ne, _, _ = JAC.nu_t_eff_faces(mod, m, U, nu, wall_zero=False)
    A = OP.assemble_A(m, nu)
    B = JAC.assemble_B(m, ne)
    assert np.allclose(A + B, OP.diffusion(m, nu + ne), atol=1e-14)
    assert np.allclose(B, B.T, atol=0), "B must be symmetric (frozen face field)"


def test_constant_nut_raises_sigma_min():
    """Sec 6.4: negative-definite sym(B) (i.e. nu_t_eff > 0) must RAISE sigma_min."""
    m = M.for_case(1000)
    nu = 1e-3
    At = MET.sym_weighted(OP.assemble_A(m, nu), m.dV)
    s0 = np.linalg.svd(At, compute_uv=False).min()
    for c in (1e-4, 1e-3, 1e-2):
        ABt = MET.sym_weighted(OP.diffusion(m, np.full(m.n + 1, nu + c)), m.dV)
        s1 = np.linalg.svd(ABt, compute_uv=False).min()
        assert s1 > s0
        assert abs(s1 / s0 - (nu + c) / nu) < 1e-3     # sigma_min scales with total nu


def _segregated(m, A, nu_t_f, itmax=80):
    """Run the segregated iteration driven by exactly the face field B is built from."""
    B = OP.diffusion(m, nu_t_f)
    Uf = np.linalg.solve(A + B, -OP.source(m))
    scale = M.wnorm(Uf, m.dV)
    U, errs = np.zeros(m.n), []
    for _ in range(itmax):
        tf = -nu_t_f * JAC.dUdy_faces(m, U)
        U = np.linalg.solve(A, -(OP.source(m) - np.diff(tf)))
        errs.append(M.wnorm(U - Uf, m.dV))
        if errs[-1] > 1e12:
            break
    return B, np.array(errs), scale


def test_iteration_matrix_matches_measured_contraction():
    """Sec 1.3: the segregated error map is -A^-1 B, so the observed contraction rate
    must equal rho(A^-1 B).  The criterion, verified directly.

    For a constant nu_t across ALL faces the answer is analytic: A^-1 B = (nu_t/nu) I, so
    rho = nu_t/nu exactly and the rate is that from the very first iterate.  c straddles
    rho = 1 without landing on it (rho = 1 is marginal: neither converges nor diverges).
    """
    m = M.for_case(1000)
    nu = 1e-3
    A = OP.assemble_A(m, nu)
    for c, expect_conv in ((2e-4, True), (8e-4, True), (2e-3, False)):
        B, errs, scale = _segregated(m, A, np.full(m.n + 1, c))
        rho = JAC.spectral_radius_AinvB(A, B)
        assert abs(rho - c / nu) < 1e-9, (c, rho)
        good = np.where(errs > 1e-11 * scale)[0]
        good = good[good < len(errs) - 1]
        assert len(good) >= 4, (c, errs[:5])
        obs = float(np.exp(np.mean(np.diff(np.log(errs[good[:8]])))))
        assert abs(obs / rho - 1) < 0.02, (c, rho, obs)
        assert (errs[-1] < errs[0]) == expect_conv == (rho < 1), (c, rho)


def test_wall_zeroing_keeps_B_the_true_iteration_matrix():
    """The consistency that wall_zero=True buys: with tau and nu_t_eff pinned together at
    the walls, rho(A^-1 B) still predicts the observed rate for a NON-constant nu_t.
    If only tau were pinned, B would not be the linearisation of the iteration."""
    m = M.for_case(550)
    nu = 1 / 550.0
    A = OP.assemble_A(m, nu)
    prof = 0.4 * nu * (1.0 + np.sin(3 * np.pi * m.yf / 2.0) ** 2)
    prof[0] = prof[-1] = 0.0                      # the wall_zero convention
    B, errs, scale = _segregated(m, A, prof, itmax=200)
    rho = JAC.spectral_radius_AinvB(A, B)
    good = np.where(errs > 1e-11 * scale)[0]
    good = good[good < len(errs) - 1]
    # rho is the ASYMPTOTIC rate; with a non-constant nu_t the early iterates carry a
    # mixture of modes, so the rate is measured over the LAST window above the floor.
    obs = float(np.exp(np.mean(np.diff(np.log(errs[good[-15:]])))))
    assert abs(obs / rho - 1) < 0.05, (rho, obs)


if __name__ == "__main__":
    sys.exit(min(_run(globals()), 1))
