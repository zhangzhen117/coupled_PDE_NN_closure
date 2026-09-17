#!/usr/bin/env python3
"""nu_t_eff = -dtau/d(dU/dy) and the closure Jacobian B.

CONVENTIONS.md: with tau = N(dU/dy, d),

    B      = d/dy( nu_t_eff d/dy . )          (same flux form as A)
    A + B  = d/dy( (nu + nu_t_eff) d/dy . )

In 1D nu_t_eff(y) is the COMPLETE information content of B -- it is just a curve, and
its sign is the headline (research plan Sec 6.4).  Because B is assembled from a frozen
face field, A + B is symmetric, so sym(B) analysis is the whole of B here.

nu_t_eff is taken by automatic differentiation of the arm's own tau(), so every chain-rule
term is included.  That matters for arm A3: nu_t >= 0 by softplus, but

    nu_t_eff = nu_t + S dnu_t/dS

and the second term can dominate.  Nothing here assumes it does not.
"""
import numpy as np
import torch

from chan import operator as OP
from .features import wall_distance


def dUdy_faces(mesh, U):
    """dU/dy at the faces from cell values, with U = 0 at both walls."""
    Uf = np.concatenate([[0.0], np.asarray(U, float), [0.0]])
    return np.diff(Uf) / mesh.d


def nu_t_eff_faces(model, mesh, U, nu, wall_zero=True):
    """(nu_t_eff, nu_t, tau) at the faces, by autodiff of the arm's tau(S, d).

    nu_t_eff := -d(tau)/dS evaluated at the current S = dU/dy.

    wall_zero pins tau = 0 AND nu_t_eff = 0 at the two wall faces.  Both, together --
    that consistency is load-bearing.  The Reynolds stress vanishes at a no-slip wall, so
    the deployed iteration pins tau_wall = 0; if B were then assembled with a nonzero
    wall-face nu_t_eff it would no longer be the linearisation of the iteration actually
    being run, and rho(A^-1 B) would not predict its convergence.  With both zeroed the
    segregated error map is exactly -A^-1 B (asserted in tests/test_closure.py).
    """
    S = torch.tensor(dUdy_faces(mesh, U), dtype=torch.float64)
    d = torch.tensor(wall_distance(mesh.yf), dtype=torch.float64)
    m = model.double()

    def tau_of_S(s_scalar, d_scalar):
        s = s_scalar.reshape(1)
        return m.tau(s, d_scalar.reshape(1), nu).squeeze(0)

    dtau_dS = torch.vmap(torch.func.grad(tau_of_S))(S, d)
    with torch.no_grad():
        tau = m.tau(S, d, nu)
        from .features import features_torch
        nut = m.nu_t(features_torch(S, d, nu))
    ne = (-dtau_dS).detach().numpy().copy()
    nt = nut.detach().numpy().copy()
    tf = tau.detach().numpy().copy()
    if wall_zero:
        ne[0] = ne[-1] = 0.0
        tf[0] = tf[-1] = 0.0
    return ne, nt, tf


def assemble_B(mesh, nu_t_eff_f):
    """B from the face-wise effective eddy viscosity."""
    return OP.diffusion(mesh, nu_t_eff_f)


def spectral_radius_AinvB(A, B):
    """rho(A^-1 B) -- the sharp criterion for the segregated iteration (Sec 1.3).

    Segregated / deferred correction: freeze tau from u^k, solve A u^{k+1} = -(b - div tau^k).
    Linearising, e^{k+1} = -A^-1 B e^k, so the iteration converges iff rho(A^-1 B) < 1,
    independently of the condition number.
    """
    return float(np.abs(np.linalg.eigvals(np.linalg.solve(A, B))).max())


def ellipticity(nu, nu_t_eff_f):
    """Faces where nu + nu_t_eff < 0: LOSS OF ELLIPTICITY.

    A sharper failure than a raised condition number, and reported separately rather than
    folded into sigma_min (which is not even meaningful once the operator is indefinite).
    Returns (n_bad, min_total_viscosity).
    """
    tot = nu + np.asarray(nu_t_eff_f, float)
    return int(np.sum(tot < 0)), float(tot.min())


# ---------------------------------------------------------------------------
# Jacobian of a VELOCITY-DEPENDENT closure  tau = N(U, dU/dy; Re).
#
# On the face i+1/2 the stress sees only the two adjacent cells:
#
#     U_{i+1/2} = (1-w_i) U_i + w_i U_{i+1}        (linear face interpolation)
#     S_{i+1/2} = (U_{i+1} - U_i) / d_{i+1/2}
#
# so the face-to-cell Jacobian T (shape (n+1, n)) is BIDIAGONAL:
#
#     T[i, i]   = a_i = d(tau_{i+1/2})/d(U_i)
#     T[i, i+1] = b_i = d(tau_{i+1/2})/d(U_{i+1})
#
# The residual is R_i = source_i + (A U)_i - (tau_{i+1} - tau_i)  (face indices), hence
#
#     B = -E T,     E[i, i] = -1, E[i, i+1] = +1        (n x (n+1) face difference)
#
# B is symmetric IFF d(tau)/dU = 0: working the algebra through gives exactly
#
#     B[i, i+1] - B[i+1, i] = -(d tau / d U)_{i+1/2}
#
# In the continuum B = d/dy(nu_t_eff d/dy .) - d/dy[(d tau/dU) .]: a diffusion term plus a
# FIRST-ORDER, advection-like term with speed c = d(tau)/dU.  So A+B is non-normal, which
# a strain-only closure can never make it in plane channel.
# ---------------------------------------------------------------------------

def face_weights(mesh):
    """Linear interpolation weights w_f with U_f = (1-w_f) U_left + w_f U_right.

    Face f sits between the extended cells f (left) and f+1 (right), where the extended
    array carries the Dirichlet ghosts U = 0 outside both walls.  So

        lower wall, f = 0 :  U_f = (1-w_0)*0 + w_0*U_0        =>  w_0 = 0
        upper wall, f = n :  U_f = (1-w_n)*U_{n-1} + w_n*0     =>  w_n = 1

    and BOTH wall faces then evaluate to zero, as no-slip requires.  The two ends are not
    symmetric in `w` because `w` multiplies the RIGHT cell in both cases.

    This asymmetry was a bug until it was found: `w_n` was left at 0, so the upper wall
    face took the value U_{n-1} instead of 0 and the FV gradient in the last cell came out
    a factor of two too small (-1.4876 against an exact -2.9883 on a field vanishing at
    both walls).  It survived because `assemble_B_div` is CONSISTENT with `residual_div`
    -- both used the same wrong stencil -- so every finite-difference Jacobian test passed,
    and the gradient test checked interior cells only.  Measured effect on the reported
    diagnostics was below the fifth significant figure, but it broke the y <-> 2-y symmetry
    of the whole cell-gradient pipeline.
    """
    yc, yf = mesh.yc, mesh.yf
    w = np.zeros(mesh.n + 1)
    w[1:-1] = (yf[1:-1] - yc[:-1]) / (yc[1:] - yc[:-1])
    w[-1] = 1.0                                   # Dirichlet at the upper wall face
    return w


def tau_jacobian_faces(model, mesh, U, retau, wall_zero=True):
    """(tau, a, b) at the faces for a velocity-dependent closure, by AD on the 2-cell
    stencil.  a_i = d(tau_i)/d(U_left), b_i = d(tau_i)/d(U_right); O(n).

    The ghost values outside the two walls are the Dirichlet U = 0, so their derivative
    contribution is dropped.  wall_zero pins tau AND its Jacobian row to zero at both wall
    faces -- both together, or B is not the Jacobian of the system being solved.
    """
    nu = 1.0 / retau
    n = mesh.n
    Uext = np.concatenate([[0.0], np.asarray(U, float), [0.0]])   # ghost cells at the walls
    Ul = torch.tensor(Uext[:-1], dtype=torch.float64)             # (n+1,)
    Ur = torch.tensor(Uext[1:], dtype=torch.float64)
    w = torch.tensor(face_weights(mesh), dtype=torch.float64)
    dd = torch.tensor(mesh.d, dtype=torch.float64)
    m = model.double()

    def tau_of(ul, ur, wi, di):
        Uf = (1.0 - wi) * ul + wi * ur
        Sf = (ur - ul) / di
        return m.tau_uv(Uf.reshape(1), Sf.reshape(1), nu, retau).squeeze(0)

    jac = torch.vmap(torch.func.jacrev(tau_of, argnums=(0, 1)))(Ul, Ur, w, dd)
    a = jac[0].detach().numpy().copy()
    b = jac[1].detach().numpy().copy()
    with torch.no_grad():
        Uf = (1.0 - w) * Ul + w * Ur
        Sf = (Ur - Ul) / dd
        tau = m.tau_uv(Uf, Sf, nu, retau).numpy().copy()
    if wall_zero:
        tau[0] = tau[-1] = 0.0
        a[0] = b[0] = a[-1] = b[-1] = 0.0
    return tau, a, b


def assemble_B_uv(mesh, a, b):
    """B = -E T from the bidiagonal face-to-cell Jacobian.  NOT symmetric in general.

    Face f spans cells f-1 (left) and f (right), so T[f, f-1] = a_f and T[f, f] = b_f.
    """
    n = mesh.n
    T = np.zeros((n + 1, n))
    f = np.arange(n + 1)
    left, right = f[1:], f[:-1]                 # faces that have a left / right cell
    T[left, left - 1] = np.asarray(a, float)[1:]
    T[right, right] = np.asarray(b, float)[:-1]
    E = np.zeros((n, n + 1))
    i = np.arange(n)
    E[i, i] = -1.0
    E[i, i + 1] = 1.0
    return -(E @ T)


def split_stencil(mesh, a, b):
    """Recover (N_U, N_S) = (d tau/dU, d tau/dS) at the faces from the stencil pair.

    With U_f = (1-w)Ul + w Ur and S_f = (Ur - Ul)/d,

        a = N_U (1-w) - N_S / d
        b = N_U w     + N_S / d

    so  N_U = a + b   and   N_S = [ (b - a) - (a + b)(2w - 1) ] * d / 2.
    The (2w-1) term vanishes only on a uniform mesh; dropping it is a real error on the
    stretched meshes used here.
    """
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    w = face_weights(mesh)
    N_U = a + b
    N_S = 0.5 * mesh.d * ((b - a) - N_U * (2.0 * w - 1.0))
    return N_U, N_S


def dtau_dU_faces(mesh, a, b):
    """c = d(tau)/dU at the faces: the ADVECTION SPEED that makes A+B non-normal."""
    return split_stencil(mesh, a, b)[0]


def nu_t_eff_uv(mesh, a, b):
    """nu_t_eff = -d(tau)/dS at the faces: the DIFFUSIVE part of B."""
    return -split_stencil(mesh, a, b)[1]


def tau_faces_only(model, mesh, U, retau, wall_zero=True):
    """tau at the faces WITHOUT the Jacobian -- the AD pass dominates the cost, and a
    line search only needs the residual."""
    nu = 1.0 / retau
    Uext = np.concatenate([[0.0], np.asarray(U, float), [0.0]])
    w = face_weights(mesh)
    Uf = torch.tensor((1.0 - w) * Uext[:-1] + w * Uext[1:], dtype=torch.float64)
    Sf = torch.tensor(np.diff(Uext) / mesh.d, dtype=torch.float64)
    with torch.no_grad():
        tau = model.double().tau_uv(Uf, Sf, nu, retau).numpy().copy()
    if wall_zero:
        tau[0] = tau[-1] = 0.0
    return tau


def residual_uv(mesh, U, model, retau, A=None, src=None):
    """Cell-integrated R(U) = source + A U - div tau(U), for the coupled closure."""
    from chan import operator as OP
    if A is None:
        A = OP.assemble_A(mesh, 1.0 / retau)
    if src is None:
        src = OP.source(mesh)
    return src + A @ U - np.diff(tau_faces_only(model, mesh, U, retau))


def numerical_abscissa(J):
    """omega = lambda_max(sym(J)).  omega > 0 => transient growth is possible even when
    every eigenvalue is stable.  Zero for a symmetric negative-definite operator."""
    return float(np.linalg.eigvalsh(0.5 * (J + J.T)).max())


def henrici(J):
    """Henrici departure from normality, normalised: sqrt(||J||_F^2 - sum|lam|^2)/||J||_F.
    Exactly 0 for a normal (here: symmetric) operator."""
    fro = np.linalg.norm(J, "fro")
    lam = np.linalg.eigvals(J)
    val = fro ** 2 - float(np.sum(np.abs(lam) ** 2))
    return float(np.sqrt(max(val, 0.0)) / fro)


# ---------------------------------------------------------------------------
# Jacobian of a DIVERGENCE-form closure  g = d(tau)/dy = G(U, dU/dy; Re).
#
# The cell-integrated residual is
#
#     R_i = dV_i + (A U)_i - dV_i * g_i
#
# and g_i is evaluated at the CELL CENTRE from U_i and the FV cell gradient
#
#     S_i = ( Uf_{i+1/2} - Uf_{i-1/2} ) / dV_i ,   Uf = linear face interpolation
#
# so g_i sees the 3-point stencil (U_{i-1}, U_i, U_{i+1}) and
#
#     B = - diag(dV) * dg/dU        (tridiagonal, NOT symmetric)
# ---------------------------------------------------------------------------

def _cell_gradient_coeffs(mesh):
    """S_i = cl_i U_{i-1} + cc_i U_i + cr_i U_{i+1}, the FV cell gradient."""
    w = face_weights(mesh)                      # (n+1,)
    dV = mesh.dV
    cl = -(1.0 - w[:-1]) / dV                   # left face contributes -(1-w_l) U_{i-1}
    cc = ((1.0 - w[1:]) - w[:-1]) / dV
    cr = w[1:] / dV                             # right face contributes +w_r U_{i+1}
    return cl, cc, cr


def dtau_dy_cells(model, mesh, U, retau):
    """g = d(tau)/dy at the cell centres, no Jacobian."""
    nu = 1.0 / retau
    cl, cc, cr = _cell_gradient_coeffs(mesh)
    Uext = np.concatenate([[0.0], np.asarray(U, float), [0.0]])
    S = cl * Uext[:-2] + cc * Uext[1:-1] + cr * Uext[2:]
    with torch.no_grad():
        g = model.double().dtau_dy(torch.tensor(np.asarray(U, float), dtype=torch.float64),
                                   torch.tensor(S, dtype=torch.float64), nu, retau)
    return g.numpy().copy()


def dtau_dy_jacobian(model, mesh, U, retau):
    """(g, dl, dc, dr): g at the cell centres and dg_i/dU_{i-1,i,i+1}, by AD."""
    nu = 1.0 / retau
    cl, cc, cr = _cell_gradient_coeffs(mesh)
    Uext = np.concatenate([[0.0], np.asarray(U, float), [0.0]])
    Um1 = torch.tensor(Uext[:-2], dtype=torch.float64)
    U0 = torch.tensor(Uext[1:-1], dtype=torch.float64)
    Up1 = torch.tensor(Uext[2:], dtype=torch.float64)
    tcl = torch.tensor(cl, dtype=torch.float64)
    tcc = torch.tensor(cc, dtype=torch.float64)
    tcr = torch.tensor(cr, dtype=torch.float64)
    m = model.double()

    def g_of(um1, u0, up1, a, b, c):
        S = a * um1 + b * u0 + c * up1
        return m.dtau_dy(u0.reshape(1), S.reshape(1), nu, retau).squeeze(0)

    jac = torch.vmap(torch.func.jacrev(g_of, argnums=(0, 1, 2)))(
        Um1, U0, Up1, tcl, tcc, tcr)
    dl = jac[0].detach().numpy().copy()
    dc = jac[1].detach().numpy().copy()
    dr = jac[2].detach().numpy().copy()
    with torch.no_grad():
        S = tcl * Um1 + tcc * U0 + tcr * Up1
        g = m.dtau_dy(U0, S, nu, retau).numpy().copy()
    return g, dl, dc, dr


def assemble_B_div(mesh, dl, dc, dr):
    """B = -diag(dV) dg/dU.  Tridiagonal, non-symmetric in general."""
    n = mesh.n
    B = np.zeros((n, n))
    i = np.arange(n)
    B[i, i] = -mesh.dV * dc
    B[i[1:], i[1:] - 1] = -mesh.dV[1:] * dl[1:]
    B[i[:-1], i[:-1] + 1] = -mesh.dV[:-1] * dr[:-1]
    return B


def residual_div(mesh, U, model, retau, A=None, src=None):
    """R = dV + A U - dV * g(U)."""
    from chan import operator as OP
    if A is None:
        A = OP.assemble_A(mesh, 1.0 / retau)
    if src is None:
        src = OP.source(mesh)
    return src + A @ U - mesh.dV * dtau_dy_cells(model, mesh, U, retau)


# ---------------------------------------------------------------------------
# Hybrid closure:  d(tau)/dy = d/dy(-nu_t dU/dy) + g_res
#
#   R = source + A U + D(nu_t) U - dV * g_res
#   J = A + B_uv(tau_B)  +  B_div(g_res)
#
# Both pieces reuse machinery that is already finite-difference verified: the Boussinesq
# branch is a face-based stress (tau_jacobian_faces / assemble_B_uv) and the residual
# branch is a cell-based divergence (dtau_dy_jacobian / assemble_B_div).
# ---------------------------------------------------------------------------

def _g_res_jacobian(model, mesh, U, retau):
    """(g_res, dl, dc, dr) at the cell centres -- same 3-point stencil as the div form."""
    nu = 1.0 / retau
    cl, cc, cr = _cell_gradient_coeffs(mesh)
    Uext = np.concatenate([[0.0], np.asarray(U, float), [0.0]])
    Um1 = torch.tensor(Uext[:-2], dtype=torch.float64)
    U0 = torch.tensor(Uext[1:-1], dtype=torch.float64)
    Up1 = torch.tensor(Uext[2:], dtype=torch.float64)
    tcl, tcc, tcr = (torch.tensor(v, dtype=torch.float64) for v in (cl, cc, cr))
    m = model.double()

    def f(um1, u0, up1, a, b, c):
        S = a * um1 + b * u0 + c * up1
        return m.g_res(u0.reshape(1), S.reshape(1), nu, retau).squeeze(0)

    j = torch.vmap(torch.func.jacrev(f, argnums=(0, 1, 2)))(Um1, U0, Up1, tcl, tcc, tcr)
    with torch.no_grad():
        S = tcl * Um1 + tcc * U0 + tcr * Up1
        g = m.g_res(U0, S, nu, retau).numpy().copy()
    return (g, j[0].detach().numpy().copy(), j[1].detach().numpy().copy(),
            j[2].detach().numpy().copy())


def hybrid_residual(mesh, U, model, retau, A=None, src=None):
    from chan import operator as OP
    if A is None:
        A = OP.assemble_A(mesh, 1.0 / retau)
    if src is None:
        src = OP.source(mesh)
    tau_B, _, _ = tau_jacobian_faces(model, mesh, U, retau)
    g_res, _, _, _ = _g_res_jacobian(model, mesh, U, retau)
    return src + A @ U - np.diff(tau_B) - mesh.dV * g_res


def hybrid_jacobian(mesh, U, model, retau, A=None):
    """J = A + B_uv(tau_B) + B_div(g_res), plus the two pieces separately so the
    destabilising contribution can be attributed."""
    from chan import operator as OP
    if A is None:
        A = OP.assemble_A(mesh, 1.0 / retau)
    _, a, b = tau_jacobian_faces(model, mesh, U, retau)
    B_nut = assemble_B_uv(mesh, a, b)
    _, dl, dc, dr = _g_res_jacobian(model, mesh, U, retau)
    B_res = assemble_B_div(mesh, dl, dc, dr)
    return A + B_nut + B_res, B_nut, B_res
