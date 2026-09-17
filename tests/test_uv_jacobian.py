#!/usr/bin/env python3
"""Gating tests for the velocity-dependent closure's Jacobian.

Every downstream number is a function of B, and a wrong B is silently plausible, so these
run before anything else.
"""
import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
torch.set_num_threads(1)
from chan import mesh as M, operator as OP
from closure import jacobian as JAC
from closure.nets import ClosureUV
from tests.test_operator import _run

RETAU = 1000.0


def _setup(n=60, seed=2):
    m = M.stretched(n, RETAU, 0.7)
    mdl = ClosureUV(seed=seed).double()
    rng = np.random.default_rng(0)
    U = 15.0 * np.sin(np.pi * m.yc / 2.0) + 0.4 * rng.standard_normal(m.n)
    return m, mdl, U


def test_full_residual_finite_difference():
    """THE gating test: finite-difference the whole coupled residual R(U; tau(U)) and
    compare column by column to A + B.  Catches a wrong interpolation weight, a wrong
    stencil index, or a sign in E -- none of which any weaker test would see."""
    m, mdl, U = _setup()
    A = OP.assemble_A(m, 1.0 / RETAU)
    tau, a, b = JAC.tau_jacobian_faces(mdl, m, U, RETAU)
    J = A + JAC.assemble_B_uv(m, a, b)
    Jfd = np.zeros_like(J)
    for j in range(m.n):
        h = 1e-6 * max(abs(U[j]), 1.0)
        Up, Um = U.copy(), U.copy()
        Up[j] += h
        Um[j] -= h
        Jfd[:, j] = (JAC.residual_uv(m, Up, mdl, RETAU)
                     - JAC.residual_uv(m, Um, mdl, RETAU)) / (2 * h)
    rel = np.abs(J - Jfd).max() / np.abs(Jfd).max()
    assert rel < 1e-6, f"A+B disagrees with the finite-differenced residual: {rel:.2e}"


def test_B_is_not_symmetric_and_the_asymmetry_is_dtau_dU():
    """B[i,i+1] - B[i+1,i] = -(d tau/dU) at the shared face, exactly."""
    m, mdl, U = _setup()
    tau, a, b = JAC.tau_jacobian_faces(mdl, m, U, RETAU)
    B = JAC.assemble_B_uv(m, a, b)
    c = JAC.dtau_dU_faces(m, a, b)
    assert not np.allclose(B, B.T, atol=1e-12), "B should NOT be symmetric here"
    i = np.arange(m.n - 1)
    lhs = B[i, i + 1] - B[i + 1, i]
    assert np.allclose(lhs, -c[1:-1], rtol=1e-9, atol=1e-12), (lhs[:3], -c[1:4])


def test_reduces_to_the_strain_only_operator_when_dtau_dU_is_zero():
    """Zero the U-input weight so d(tau)/dU == 0.  Then B must be EXACTLY symmetric and
    equal D(nu_t_eff) -- recovering Stages 3-7 as a special case of the new code."""
    m, mdl, U = _setup()
    with torch.no_grad():
        mdl.g.net[0].weight[:, 0] = 0.0          # kill the u_hat input
    tau, a, b = JAC.tau_jacobian_faces(mdl, m, U, RETAU)
    c = JAC.dtau_dU_faces(m, a, b)
    assert np.abs(c).max() < 1e-12, f"d(tau)/dU should vanish, got {np.abs(c).max():.2e}"
    B = JAC.assemble_B_uv(m, a, b)
    assert np.allclose(B, B.T, atol=1e-14), "B must be symmetric when d(tau)/dU = 0"
    ne = JAC.nu_t_eff_uv(m, a, b)
    ne_w = ne.copy()
    ne_w[0] = ne_w[-1] = 0.0
    assert np.allclose(B, OP.diffusion(m, ne_w), atol=1e-12, rtol=0)


def test_stencil_split_matches_direct_autodiff():
    """split_stencil must recover (N_U, N_S) including the (2w-1) term, which is nonzero
    on a stretched mesh.  Checked against a direct AD of tau(U_f, S_f)."""
    m, mdl, U = _setup()
    _, a, b = JAC.tau_jacobian_faces(mdl, m, U, RETAU, wall_zero=False)
    N_U, N_S = JAC.split_stencil(m, a, b)
    Uext = np.concatenate([[0.0], U, [0.0]])
    w = JAC.face_weights(m)
    Uf = torch.tensor((1 - w) * Uext[:-1] + w * Uext[1:], dtype=torch.float64)
    Sf = torch.tensor(np.diff(Uext) / m.d, dtype=torch.float64)
    g = torch.vmap(torch.func.jacrev(
        lambda u, s: mdl.tau_uv(u.reshape(1), s.reshape(1), 1.0 / RETAU,
                                RETAU).squeeze(0), argnums=(0, 1)))(Uf, Sf)
    assert np.allclose(N_U, g[0].detach().numpy(), rtol=1e-8, atol=1e-12)
    assert np.allclose(N_S, g[1].detach().numpy(), rtol=1e-8, atol=1e-12)
    assert np.abs(2 * JAC.face_weights(m)[1:-1] - 1).max() > 1e-3, \
        "mesh too uniform for this test to discriminate the (2w-1) term"


def test_wall_rows_pinned_together():
    """tau and its Jacobian row are pinned at the walls TOGETHER, or B is not the
    Jacobian of the system actually being solved."""
    m, mdl, U = _setup()
    tau, a, b = JAC.tau_jacobian_faces(mdl, m, U, RETAU, wall_zero=True)
    assert tau[0] == 0.0 and tau[-1] == 0.0
    assert a[0] == b[0] == a[-1] == b[-1] == 0.0




# --------------------------------------------------------------------------- div form
def _setup_div(n=60, seed=3):
    from closure.nets import ClosureDiv
    m = M.stretched(n, RETAU, 0.7)
    mdl = ClosureDiv(seed=seed).double()
    rng = np.random.default_rng(1)
    U = 15.0 * np.sin(np.pi * m.yc / 2.0) + 0.4 * rng.standard_normal(m.n)
    return m, mdl, U


def test_cell_gradient_is_exact_for_linear_fields():
    """The FV cell gradient must reproduce d/dy of a linear profile exactly, or the
    strain feature is wrong everywhere.

    Interior cells only: a linear field does NOT vanish at the walls, so the Dirichlet
    ghosts make the two wall cells legitimately different.  The wall cells are covered by
    the next test, with a field that does vanish there.
    """
    m = M.stretched(60, RETAU, 0.7)
    cl, cc, cr = JAC._cell_gradient_coeffs(m)
    for slope in (1.0, -2.5):
        U = slope * m.yc
        Ue = np.concatenate([[0.0], U, [0.0]])
        S = cl * Ue[:-2] + cc * Ue[1:-1] + cr * Ue[2:]
        assert np.abs(S[1:-1] - slope).max() < 1e-12, slope


def test_wall_faces_honour_dirichlet_and_gradient_is_symmetric():
    """BOTH wall faces must interpolate to zero, and the cell gradient must be
    antisymmetric under y -> 2-y for a symmetric profile.

    This is the regression test for the `w[-1] = 0` bug: with it, the upper wall face
    evaluated to U_{n-1} instead of 0 and the last cell's gradient was half its true
    value, silently breaking channel symmetry.
    """
    m = M.stretched(60, RETAU, 0.7)
    w = JAC.face_weights(m)
    U = 1.5 * m.yc * (2.0 - m.yc)                 # vanishes at BOTH walls
    Ue = np.concatenate([[0.0], U, [0.0]])
    Uf = (1.0 - w) * Ue[:-1] + w * Ue[1:]
    assert abs(Uf[0]) < 1e-14, Uf[0]
    assert abs(Uf[-1]) < 1e-14, Uf[-1]

    cl, cc, cr = JAC._cell_gradient_coeffs(m)
    S = cl * Ue[:-2] + cc * Ue[1:-1] + cr * Ue[2:]
    exact = 3.0 * (1.0 - m.yc)
    rel = np.abs((S - exact) / exact)
    assert rel.max() < 2e-2, rel.max()            # wall cells included
    assert np.abs(S + S[::-1]).max() < 1e-12      # antisymmetric on a symmetric mesh


def test_div_full_residual_finite_difference():
    """THE gating test for the divergence-form closure: finite-difference R(U) and
    compare column by column to A + B."""
    m, mdl, U = _setup_div()
    A = OP.assemble_A(m, 1.0 / RETAU)
    g, dl, dc, dr = JAC.dtau_dy_jacobian(mdl, m, U, RETAU)
    J = A + JAC.assemble_B_div(m, dl, dc, dr)
    Jfd = np.zeros_like(J)
    for j in range(m.n):
        hh = 1e-6 * max(abs(U[j]), 1.0)
        Up, Um = U.copy(), U.copy()
        Up[j] += hh
        Um[j] -= hh
        Jfd[:, j] = (JAC.residual_div(m, Up, mdl, RETAU)
                     - JAC.residual_div(m, Um, mdl, RETAU)) / (2 * hh)
    rel = np.abs(J - Jfd).max() / np.abs(Jfd).max()
    assert rel < 1e-6, f"A+B disagrees with the finite-differenced residual: {rel:.2e}"


def test_div_closure_is_not_the_momentum_identity():
    """d(tau)/dy = 1 + nu U'' would make R vanish for EVERY U and the system singular.
    The closure must NOT reproduce that: A+B has to stay far from singular, and the
    residual must actually depend on U."""
    m, mdl, U = _setup_div()
    A = OP.assemble_A(m, 1.0 / RETAU)
    g, dl, dc, dr = JAC.dtau_dy_jacobian(mdl, m, U, RETAU)
    J = A + JAC.assemble_B_div(m, dl, dc, dr)
    assert np.linalg.matrix_rank(J) == m.n, "A+B is singular: the closure is degenerate"
    R1 = JAC.residual_div(m, U, mdl, RETAU)
    R2 = JAC.residual_div(m, 1.3 * U, mdl, RETAU)
    assert np.linalg.norm(R1 - R2) > 1e-6 * np.linalg.norm(R1)


def test_div_g_matches_between_jacobian_and_cheap_paths():
    m, mdl, U = _setup_div()
    g1, _, _, _ = JAC.dtau_dy_jacobian(mdl, m, U, RETAU)
    g2 = JAC.dtau_dy_cells(mdl, m, U, RETAU)
    assert np.allclose(g1, g2, rtol=1e-12, atol=1e-14)




def test_hybrid_full_residual_finite_difference():
    """Gate for the hybrid closure: finite-difference the combined residual and compare
    column by column to A + B_nut + B_res."""
    from closure.nets import ClosureHybrid
    m = M.stretched(60, RETAU, 0.7)
    mdl = ClosureHybrid(seed=5).double()
    rng = np.random.default_rng(4)
    U = 15.0 * np.sin(np.pi * m.yc / 2.0) + 0.4 * rng.standard_normal(m.n)
    J, Bn, Br = JAC.hybrid_jacobian(m, U, mdl, RETAU)
    Jfd = np.zeros_like(J)
    for j in range(m.n):
        hh = 1e-6 * max(abs(U[j]), 1.0)
        Up, Um = U.copy(), U.copy()
        Up[j] += hh
        Um[j] -= hh
        Jfd[:, j] = (JAC.hybrid_residual(m, Up, mdl, RETAU)
                     - JAC.hybrid_residual(m, Um, mdl, RETAU)) / (2 * hh)
    rel = np.abs(J - Jfd).max() / np.abs(Jfd).max()
    assert rel < 1e-6, f"hybrid A+B disagrees with finite difference: {rel:.2e}"


def test_lipschitz_bound_actually_binds():
    """The spectral-norm construction must give a HARD bound on |G_U|:
        |G_U| <= Re_tau * G_SCALE * lip / U_SCALE
    Checked against the measured autodiff Jacobian, for two values of lip."""
    from closure.nets import ClosureDivLip
    from closure.features import U_SCALE
    m = M.stretched(80, RETAU, 0.7)
    rng = np.random.default_rng(7)
    U = 15.0 * np.sin(np.pi * m.yc / 2.0) + 0.4 * rng.standard_normal(m.n)
    for lip in (0.25, 1.0):
        mdl = ClosureDivLip(seed=2, lip=lip).double()
        _, dl, dc, dr = JAC.dtau_dy_jacobian(mdl, m, U, RETAU)
        _, _, _, G_U, _ = _split(m, dl, dc, dr)
        bound = RETAU * mdl.G_SCALE * lip / U_SCALE
        assert np.abs(G_U).max() <= bound * 1.02, (lip, np.abs(G_U).max(), bound)


def _split(mesh, dl, dc, dr):
    cl, cc, cr = JAC._cell_gradient_coeffs(mesh)
    with np.errstate(divide="ignore", invalid="ignore"):
        G_S = np.where(np.abs(cr) > np.abs(cl), dr / cr, dl / cl)
    G_S = np.nan_to_num(G_S)
    return None, None, None, dc - G_S * cc, G_S


if __name__ == "__main__":
    sys.exit(min(_run(globals()), 1))
