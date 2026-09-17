#!/usr/bin/env python3
"""Conditioning metrics: the global K_A, the Wu et al. local K_j / K_x, and the SVD.

CONVENTIONS.md "K_j".  A is the FLUX-form operator, so the discrete Green's function
sampled at the cell centres is exactly a row of A^-1 and the sqrt(dV) weighting is the
quadrature of the continuous L2 norm.  No multigrid needed: n <= 1040, so form A^-1 once
by a dense solve against I and every row norm falls out of one reduction.
"""
import numpy as np

from . import mesh as M


def K_A(A):
    """Global condition number ||A|| ||A^-1|| in the 2-norm.  Wu et al. Sec 2.1."""
    s = np.linalg.svd(A, compute_uv=False)
    return float(s.max() / s.min())


def K_A_uniform_exact(n):
    """Exact K_A of the flux-form Dirichlet Laplacian on a UNIFORM n-cell mesh.

    Cell-centred FV puts HALF distances at the two wall faces, so the matrix is
    tridiag(1, -2, 1) with the two corner diagonals at -3, not the textbook
    tridiag(1, -2, 1).  Its condition number still tends to 4 n^2 / pi^2 (the operator
    is the same Dirichlet Laplacian on [0, L]); this returns the exact finite-n value
    so checkpoint 1 can separate "wrong discretisation" from "not yet asymptotic".
    """
    T = np.diag(-2.0 * np.ones(n)) + np.diag(np.ones(n - 1), 1) + np.diag(np.ones(n - 1), -1)
    T[0, 0] = T[-1, -1] = -3.0
    w = np.abs(np.linalg.eigvalsh(T))
    return float(w.max() / w.min())


def green_rows(A, dV, weighted=True):
    """r_tilde: rows of A^-1 scaled by sqrt(dV) (or unscaled, for the ckpt-3 gate).

    Returns (n, n); row j is the discrete Green's function for output cell j.
    """
    Ainv = np.linalg.solve(A, np.eye(A.shape[0]))
    return Ainv * np.sqrt(dV)[None, :] if weighted else Ainv


def K_local(A, dV, div_tau, U_ref, weighted=True):
    """K_j for every output cell.  Wu et al. Eq. for the local condition number.

    div_tau : (n,) POINTWISE stress divergence d(tau)/dy at the cell centres.
    Returns (n,) K_j.
    """
    R = green_rows(A, dV, weighted)
    gnorm = np.linalg.norm(R, axis=1)                 # ||G_j||_Omega
    return gnorm * M.wnorm(div_tau, dV) / U_ref


def K_volume(K_j, dV):
    """Volume-averaged scalar K_x (research plan Sec 2.1)."""
    return float(np.sum(K_j * dV) / np.sum(dV))


def svd_cache(A, assume_symmetric=True):
    """Full SVD of A, returned with sigma ASCENDING.

    Index 0 is the SMALLEST singular value, i.e. the most dangerous direction -- and by
    the direction reversal of research plan Sec 3.1 that is the SMOOTHEST, largest-scale
    mode, not a high-wavenumber one (verified: the leading modes are sin(k pi y / 2) to
    5 decimal places).

    A is symmetric here, so this is an eigendecomposition in disguise; eigh is ~10x
    faster than svd at n = 1040 and is what actually runs.  For A = Q L Q^T with L < 0
    (A is negative definite): sigma = -L, V = Q, U = -Q.  Kept in SVD form because
    Stage 2 indexes by singular value.  `test_eigh_matches_svd` guards the equivalence.
    """
    if assume_symmetric and np.allclose(A, A.T, atol=0, rtol=0):
        lam, Q = np.linalg.eigh(A)
        sig = np.abs(lam)
        o = np.argsort(sig)
        sig, Q, lam = sig[o], Q[:, o], lam[o]
        return dict(sigma=sig, U=Q * np.sign(lam)[None, :], V=Q)
    U, s, Vt = np.linalg.svd(A)
    o = np.argsort(s)
    return dict(sigma=s[o], U=U[:, o], V=Vt.T[:, o])



# ---------------------------------------------------------------------------
# The weighted-norm form of the operator.
#
# The bound we actually want is in the dV-weighted field norm.  With D = diag(dV),
#
#     ||dU||_dV = ||D^{1/2} A^-1 r||_2 = || Atil^-1 rhat ||_2,
#     Atil := D^{-1/2} A D^{-1/2},        rhat := D^{-1/2} r,
#
# and ||rhat||_2 = ||r/dV||_dV is the weighted norm of the POINTWISE residual.  Atil is
# the operator that is self-adjoint in the weighted inner product; it is the discrete
# analogue of nu d2/dy2, so sigma_min(Atil) CONVERGES under mesh refinement (to nu pi^2/4
# on y in [0,2]) whereas sigma_min(A) does not.  Stage 2 works entirely in this form so
# every predictor is mesh independent.
# ---------------------------------------------------------------------------

def sym_weighted(A, dV):
    """Atil = D^{-1/2} A D^{-1/2}.  Symmetric; sigma_min is mesh-convergent."""
    w = 1.0 / np.sqrt(dV)
    return A * w[:, None] * w[None, :]


def to_hat(r, dV):
    """rhat = D^{-1/2} r, so ||rhat||_2 == wnorm(r/dV, dV)."""
    return np.asarray(r, float) / np.sqrt(dV)


def from_hat(uhat, dV):
    """Inverse of to_hat applied to a solution: dU = D^{-1/2} uhat."""
    return np.asarray(uhat, float) / np.sqrt(dV)


def sigma_min_continuum(nu, L=2.0):
    """Continuum limit of sigma_min(Atil): nu (pi/L)^2 for Dirichlet on [0, L]."""
    return nu * (np.pi / L) ** 2


def split_B(mesh, dl, dc, dr):
    """Split the divergence-form closure Jacobian into DIFFUSION and REACTION parts.

    With g_i = G(U_i, S_i) and S the FV cell gradient,

        dg_i/dU_{i-1} = G_S c_l          diffusive
        dg_i/dU_i     = G_U + G_S c_c    G_U is the REACTION (zeroth-order) term
        dg_i/dU_{i+1} = G_S c_r          diffusive

    so G_S is recoverable from an off-diagonal and G_U by subtraction.  Note G_S DOES
    reach the diagonal through c_c -- the split is reaction vs diffusion, not diagonal vs
    off-diagonal.

    Returns (B_full, B_diff, B_react, G_U, G_S).  `B_diff` is the preconditioner that
    mimics implicit treatment for a closure with no explicit eddy viscosity: it folds the
    diffusive part in and leaves only the reaction explicit.
    """
    from closure import jacobian as JAC
    cl, cc, cr = JAC._cell_gradient_coeffs(mesh)
    with np.errstate(divide="ignore", invalid="ignore"):
        G_S = np.where(np.abs(cr) > np.abs(cl), dr / cr, dl / cl)
    G_S = np.nan_to_num(G_S)
    G_U = dc - G_S * cc
    B_full = JAC.assemble_B_div(mesh, dl, dc, dr)
    B_diff = JAC.assemble_B_div(mesh, dl, dc - G_U, dr)      # reaction removed
    return B_full, B_diff, B_full - B_diff, G_U, G_S


# ---------------------------------------------------------------------------
# Non-normal diagnostics.
#
# P^-1 J is non-normal here, so eigenvalue clustering does NOT bound GMRES: a clustered
# spectrum can accompany stagnation.  The quantities below are the ones that do bound it.
# ---------------------------------------------------------------------------

def fov_bounds(M):
    """(lam_min, lam_max, norm) for the Hermitian part H = (M + M^T)/2 and ||M||_2.

    Re W(M) = [lam_min(H), lam_max(H)] exactly, where W is the field of values.  So both
    endpoints having the same sign certifies 0 not in W, which is what the Elman bound
    needs.
    """
    H = 0.5 * (np.asarray(M, float) + np.asarray(M, float).T)
    w = np.linalg.eigvalsh(H)
    return float(w.min()), float(w.max()), float(np.linalg.norm(M, 2))


def elman_bound(M, k):
    """Elman's GMRES bound  ||r_k||/||r_0|| <= (1 - nu^2 / Lam^2)^(k/2),
    nu = dist(0, W(M)), Lam = ||M||_2.  Returns np.inf when 0 is in W(M) (no bound).

    nu is taken as min(|lam_min(H)|, |lam_max(H)|), valid only when the two share a sign;
    W is convex and its real part spans [lam_min, lam_max], so a sign change means the
    imaginary axis cuts W and the bound does not apply.
    """
    lo, hi, nrm = fov_bounds(M)
    if lo * hi <= 0:
        return np.inf
    nu = min(abs(lo), abs(hi))
    r = 1.0 - (nu / nrm) ** 2
    return float(np.inf if r < 0 else r ** (0.5 * k))


def pseudospectrum(M, re, im):
    """sigma_min(zI - M) on the grid re x im.  O(n^3) per point -- coarse grids only."""
    M = np.asarray(M, float)
    n = M.shape[0]
    out = np.empty((len(im), len(re)))
    for j, y in enumerate(im):
        for i, x in enumerate(re):
            out[j, i] = np.linalg.svd((x + 1j * y) * np.eye(n) - M,
                                      compute_uv=False)[-1]
    return out
