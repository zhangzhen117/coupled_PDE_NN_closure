#!/usr/bin/env python3
"""Flux-form discretisation of the steady 1D channel momentum operator.

CONVENTIONS.md:  R(U; tau) = 1 + d/dy(nu dU/dy) - d(tau)/dy,  tau = <u'v'>,
Dirichlet U = 0 at both walls, full channel y in [0, 2], u_tau = h = 1, nu = 1/Re_tau.

Everything is CELL-INTEGRATED (not divided by dV) so the matrices are symmetric on a
non-uniform mesh -- see CONVENTIONS.md "Operators".  A is symmetric negative definite.
"""
import numpy as np


def diffusion(mesh, nu_face):
    """Flux-form d/dy( nu d/dy . ) with Dirichlet 0 at both walls.

    nu_face : (n+1,) diffusivity at the faces (walls included).
    Returns a dense (n, n) symmetric negative-definite array.

    This single routine builds A (nu_face = nu), B (nu_face = nu_t_eff), and A + B
    (nu_face = nu + nu_t_eff), which is exactly the identity asserted in tests.
    """
    nu_face = np.asarray(nu_face, float)
    n = mesh.n
    if nu_face.shape != (n + 1,):
        raise ValueError(f"nu_face must be ({n + 1},), got {nu_face.shape}")
    k = nu_face / mesh.d                      # (n+1,) face conductances
    M = np.zeros((n, n))
    idx = np.arange(n)
    M[idx, idx] = -(k[:-1] + k[1:])           # wall faces contribute via the half d
    M[idx[:-1], idx[1:]] = k[1:-1]
    M[idx[1:], idx[:-1]] = k[1:-1]
    return M


def assemble_A(mesh, nu):
    """A = dR/dU with tau frozen.  Symmetric negative definite."""
    return diffusion(mesh, np.full(mesh.n + 1, float(nu)))


def face_interp(mesh, fc):
    """Cell-centred field -> faces, linear in y; one-sided at the two walls."""
    fc = np.asarray(fc, float)
    yc, yf = mesh.yc, mesh.yf
    out = np.empty(mesh.n + 1)
    w = (yf[1:-1] - yc[:-1]) / (yc[1:] - yc[:-1])
    out[1:-1] = (1.0 - w) * fc[:-1] + w * fc[1:]
    out[0], out[-1] = fc[0], fc[-1]
    return out


def div_face(mesh, q_face):
    """POINTWISE divergence d(q)/dy from face values: (q[i+1]-q[i]) / dV[i]."""
    return np.diff(np.asarray(q_face, float)) / mesh.dV


def source(mesh):
    """Cell-integrated driving term, -(1/rho) dP/dx = 1  =>  b_src = dV."""
    return mesh.dV.copy()


def residual(mesh, U, tau_face, nu):
    """Cell-integrated R(U; tau) = 1 + d/dy(nu dU/dy) - d(tau)/dy.

    tau_face : (n+1,) stress at the faces (tau = 0 at both walls by no-slip).
    Returns (n,) -- the cell-integrated residual, in the same units as A @ U.
    """
    A = assemble_A(mesh, nu)
    return source(mesh) + A @ U - np.diff(np.asarray(tau_face, float))


def solve_frozen(mesh, tau_face, nu):
    """Exact solve of R(U; tau) = 0 with tau frozen.  R is LINEAR in U here."""
    A = assemble_A(mesh, nu)
    rhs = -(source(mesh) - np.diff(np.asarray(tau_face, float)))
    return np.linalg.solve(A, rhs)


def tau_face_from_profile(mesh, y_dns, tau_dns):
    """Interpolate a DNS stress profile onto the mesh faces, pinning tau = 0 at walls.

    Monotone cubic (PCHIP): the DNS grid is finer than the coarse meshes and a linear
    interpolant pollutes the near-wall stress gradient.
    """
    from scipy.interpolate import PchipInterpolator
    tf = PchipInterpolator(y_dns, tau_dns)(mesh.yf)
    tf[0] = 0.0
    tf[-1] = 0.0
    return tf


def face_interp_inv(mesh, ff):
    """Face field -> cell centres (simple face average).  Used only for taking norms of
    a face-defined quantity in the cell-volume-weighted norm."""
    ff = np.asarray(ff, float)
    return 0.5 * (ff[1:] + ff[:-1])
