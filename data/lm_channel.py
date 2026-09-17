#!/usr/bin/env python3
"""Loader for the Lee & Moser (2015) channel profiles.

Returns everything on the FULL channel y in [0, 2] (the half-channel database profile
mirrored about y = 1), in the wall units fixed by CONVENTIONS.md: u_tau = 1, h = 1,
nu = 1/Re_tau.

The reference stress is the ANALYTIC one (research plan Sec 4.2):

    tau_true(y) = nu dU/dy - (1 - y)        [lower half; mirrored antisymmetrically]

built from column 4 of the mean file, which is already dU+/dy+ = nu dU/dy / u_tau^2.
The database <u'v'> is loaded too, but ONLY so the gap between the two can be reported;
it is never the reference. See CONVENTIONS.md "Stress sign".
"""
import os
import re
import numpy as np

RAW = os.path.join(os.path.dirname(os.path.abspath(__file__)), "raw")
CASES = {180: "0180", 550: "0550", 1000: "1000", 2000: "2000", 5200: "5200"}


def _header_scalars(path):
    """Pull nu, u_tau, Re_tau out of the %-comment header."""
    out = {}
    pat = re.compile(r"^%.*?\b(nu|u_tau|Re_tau)\s*=\s*([-+0-9.eE]+)")
    with open(path) as f:
        for line in f:
            if not line.startswith("%"):
                break
            m = pat.match(line)
            if m:
                out[m.group(1)] = float(m.group(2))
    return out


def _mirror(y_half, f_half, parity):
    """Half-channel profile (y/delta in [0,1], y[0]=0 wall) -> full channel y in [0,2].

    parity=+1 mirrors symmetrically (U, k), parity=-1 antisymmetrically (dU/dy, tau).
    The centreline point is kept once.
    """
    yc = 1.0
    up = 2.0 * yc - y_half[::-1]        # (1,2], excluding the shared centreline below
    fu = parity * f_half[::-1]
    if np.isclose(y_half[-1], yc, atol=1e-6):
        up, fu = up[1:], fu[1:]         # centreline already present in the half profile
    return np.concatenate([y_half, up]), np.concatenate([f_half, fu])


def load(retau):
    """Load one case.  Returns a dict of full-channel arrays on the DNS grid.

    keys: retau_nominal, retau, nu, y, U, dUdy, visc_frac, tau_true, tau_dns, y_plus
      y          full-channel wall-normal coordinate in [0, 2], h = 1
      U          mean velocity in wall units (u_tau = 1)
      dUdy       dU/dy in wall units (= visc_frac / nu)
      visc_frac  nu dU/dy / u_tau^2 = column 4 = the viscous fraction of total shear
      tau_true   ANALYTIC <u'v'> = nu dU/dy - (1 - y)          <- the reference
      tau_dns    database <u'v'>                               <- diagnostic only
    """
    if retau not in CASES:
        raise KeyError(f"no case {retau}; have {sorted(CASES)}")
    tok = CASES[retau]
    mp = os.path.join(RAW, f"LM_Channel_{tok}_mean_prof.dat")
    fp = os.path.join(RAW, f"LM_Channel_{tok}_vel_fluc_prof.dat")
    if not os.path.exists(mp):
        raise FileNotFoundError(f"{mp}: run data/fetch_lm.py first")

    hdr = _header_scalars(mp)
    nu = 1.0 / hdr["Re_tau"]            # wall units: nu = 1/Re_tau exactly

    m = np.loadtxt(mp, comments="%")
    yh, ypl, Uh, visch = m[:, 0], m[:, 1], m[:, 2], m[:, 3]
    yh = np.clip(yh, 0.0, 1.0)          # first point is 1.1e-16, not exactly 0

    # analytic reference stress on the half channel, then mirror antisymmetrically
    tau_h = visch - (1.0 - yh)

    y, U = _mirror(yh, Uh, +1)
    _, visc = _mirror(yh, visch, -1)
    _, tau_true = _mirror(yh, tau_h, -1)
    _, y_plus = _mirror(yh, ypl, +1)

    # database <u'v'> (col 6), for the a-priori "irreducible residual" diagnostic only
    fl = np.loadtxt(fp, comments="%")
    tau_dns_h = np.interp(yh, np.clip(fl[:, 0], 0.0, 1.0), fl[:, 5])
    _, tau_dns = _mirror(yh, tau_dns_h, -1)

    return dict(retau_nominal=retau, retau=hdr["Re_tau"], nu=nu,
                y=y, y_plus=y_plus, U=U, dUdy=visc / nu, visc_frac=visc,
                tau_true=tau_true, tau_dns=tau_dns)


def u_bulk(d):
    """Bulk velocity, the U_ref of CONVENTIONS.md, by trapezoid over the full channel."""
    return np.trapezoid(d["U"], d["y"]) / (d["y"][-1] - d["y"][0])


if __name__ == "__main__":
    print(f"{'Re_tau':>8} {'exact':>10} {'nu':>10} {'npts':>6} "
          f"{'U_bulk':>8} {'|tau|max':>9} {'irred%':>7}")
    for r in CASES:
        d = load(r)
        rel = (np.abs(d["tau_true"] - d["tau_dns"]).max()
               / np.abs(d["tau_true"]).max() * 100)
        print(f"{r:>8} {d['retau']:>10.3f} {d['nu']:>10.3e} {len(d['y']):>6} "
              f"{u_bulk(d):>8.3f} {np.abs(d['tau_true']).max():>9.4f} {rel:>7.3f}")
