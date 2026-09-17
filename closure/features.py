#!/usr/bin/env python3
"""Closure inputs.

Research plan Sec 6.1: the NN input must NOT be y alone, or dtau/du = 0, B = 0, and
Stage 3 degenerates into Stage 2.  We split the inputs deliberately:

  s = |nu dU/dy| / u_tau^2   FIELD input   -- depends on U, so it CREATES B
  d = min(y, 2h - y) / h     STATIC input  -- wall distance, dd/du = 0, contributes
                                              nothing to B

Wall distance is a standard closure input in every RANS model, and it is legitimately
B-free, so this split is honest rather than a dodge.  It also makes "which inputs create
closure-velocity coupling" a design knob rather than an accident.

Both features are SYMMETRIC about the centreline.  The closure predicts an eddy
viscosity nu_t(s, d) and the stress is assembled as

    tau = -nu_t(s, d) * dU/dy

so tau is automatically ANTISYMMETRIC and vanishes where dU/dy does, matching the
physical structure without the network having to learn it.  This is also what makes arms
A1/A3 differ in exactly one thing (the output activation), as Sec 6.3 requires.
"""
import numpy as np

LY = 2.0


def wall_distance(y):
    """d = distance to the NEAREST wall, in [0, 1]."""
    return np.minimum(np.asarray(y, float), LY - np.asarray(y, float))


def features_np(dUdy, y, nu):
    """(s, d) for numpy arrays.  s = |nu dU/dy| in [0, ~1]."""
    s = np.abs(nu * np.asarray(dUdy, float))
    return np.stack([s, wall_distance(y)], axis=-1)


def features_torch(dUdy, d, nu):
    """(s, d) for torch tensors; `d` precomputed since it does not depend on U."""
    import torch
    return torch.stack([torch.abs(nu * dUdy), d], dim=-1)


# ---------------------------------------------------------------------------
# Velocity-dependent closure:  tau = N(U, dU/dy; Re_tau)
#
# Both U and dU/dy are FIELD inputs, so BOTH contribute to B.  That is the whole
# difference from the strain-only closure above: d(tau)/dU is nonzero, so B picks up a
# first-order (advection-like) term and A+B is NO LONGER SYMMETRIC.  See
# closure/jacobian.py::tau_jacobian_faces.
#
#   u_hat = U / 30        wall units; U+ reaches ~24 at Re_tau = 5200
#   s_hat = nu dU/dy      SIGNED, in [-1, 1] -- not |.| as in the strain-only closure,
#                         because tau must be ODD in dU/dy (it flips sign across the
#                         centreline while U does not) and a direct-tau head has to learn
#                         that from seeing both halves of the channel
#   r_hat = log10(Re)/4   STATIC: d(r_hat)/dU = 0, contributes nothing to B
# ---------------------------------------------------------------------------

U_SCALE = 30.0


def features_uv_np(U, dUdy, nu, retau):
    """(u_hat, s_hat, r_hat) for numpy arrays."""
    U = np.asarray(U, float)
    return np.stack([U / U_SCALE,
                     nu * np.asarray(dUdy, float),
                     np.full_like(U, np.log10(retau) / 4.0)], axis=-1)


def features_uv_torch(U, dUdy, nu, retau):
    """(u_hat, s_hat, r_hat) for torch tensors.  `retau` may be a tensor or a float."""
    import torch
    if not torch.is_tensor(retau):
        retau = torch.full_like(U, float(retau))
    return torch.stack([U / U_SCALE, nu * dUdy, torch.log10(retau) / 4.0], dim=-1)


# ---------------------------------------------------------------------------
# Features for the divergence-form closure.  Same three inputs, but Re is rescaled to
# span the FULL unit interval over the training range instead of [0.57, 0.93].
#
# This matters at Re_tau = 5200, y+ ~ 100-300.  There the features (U+, nu dU/dy) are
# nearly identical to those at Re_tau = 1000, y+ ~ 100, while d(tau)/dy is -0.26 against
# +0.76 -- the network has to separate them on the Re input alone, and log10(Re)/4 gave it
# only a third of the available dynamic range to do it with.
# ---------------------------------------------------------------------------

RE_LO, RE_HI = 180.0, 5200.0


def features_div_torch(U, dUdy, nu, retau):
    import torch
    if not torch.is_tensor(retau):
        retau = torch.full_like(U, float(retau))
    r = (torch.log10(retau) - np.log10(RE_LO)) / (np.log10(RE_HI) - np.log10(RE_LO))
    return torch.stack([U / U_SCALE, nu * dUdy, 2.0 * r - 1.0], dim=-1)
