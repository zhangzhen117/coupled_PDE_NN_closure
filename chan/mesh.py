#!/usr/bin/env python3
"""Cell-centred 1D meshes on the full channel y in [0, 2].

See CONVENTIONS.md: full channel, Dirichlet at both walls, cell-centred finite volume.
A mesh carries faces `yf` (n+1), centres `yc` (n), cell volumes `dV` (n), and the
face-normal distances `d` (n+1) used by the flux-form operator -- note d[0] and d[-1] are
HALF distances (centre to wall face), which is the correct Dirichlet treatment.
"""
import numpy as np

H = 1.0          # half width
LY = 2.0 * H     # full channel

# Wu et al. (2019) mesh sizes, research plan Sec 4
N_PRESET = {180: 36, 550: 110, 1000: 200, 2000: 400, 5200: 1040}


class Mesh:
    __slots__ = ("yf", "yc", "dV", "d", "n", "kind", "stretch")

    def __init__(self, yf, kind="?", stretch=0.0):
        yf = np.asarray(yf, float)
        if yf.ndim != 1 or np.any(np.diff(yf) <= 0):
            raise ValueError("faces must be strictly increasing 1D")
        self.yf = yf
        self.yc = 0.5 * (yf[1:] + yf[:-1])
        self.dV = np.diff(yf)
        # centre-to-centre distances, with half cells at the two walls
        self.d = np.concatenate([[self.yc[0] - yf[0]],
                                 np.diff(self.yc),
                                 [yf[-1] - self.yc[-1]]])
        self.n = self.yc.size
        self.kind = kind
        self.stretch = stretch

    def y_plus(self, retau):
        """Wall distance in viscous units, measured to the NEAREST wall."""
        return np.minimum(self.yc, LY - self.yc) * retau

    @property
    def first_yplus_at(self):
        return self.yc[0]      # multiply by retau for y+

    def __repr__(self):
        return (f"Mesh({self.kind}, n={self.n}, s={self.stretch:.3f}, "
                f"y_c[0]={self.yc[0]:.3e}, max/min dV="
                f"{self.dV.max() / self.dV.min():.1f})")


def wnorm(v, dV):
    """THE field norm: cell-volume-weighted discrete L2.  CONVENTIONS.md "Norms".

    Never use np.linalg.norm on a field -- the sqrt(dV) weighting is what makes K_j
    mesh independent, and its absence is the Stage-1 checkpoint-3 failure mode.
    """
    v = np.asarray(v, float)
    return float(np.sqrt(np.sum(v * v * dV)))


def winner(a, b, dV):
    """Matching weighted inner product."""
    return float(np.sum(np.asarray(a, float) * np.asarray(b, float) * dV))


def uniform(n):
    """Uniform mesh -- used by Stage-1 checkpoint 1 (K_A -> 4 n^2 / pi^2)."""
    return Mesh(np.linspace(0.0, LY, n + 1), kind="uniform")


def _tanh_faces(n, s):
    """Symmetric two-sided tanh stretching, clustered at both walls."""
    if s <= 1e-9:
        return np.linspace(0.0, LY, n + 1)
    xi = np.linspace(-1.0, 1.0, n + 1)
    return H * (1.0 + np.tanh(s * xi) / np.tanh(s))


def stretched(n, retau, target_yplus=0.7, smax=6.0):
    """Non-uniform mesh with the first CELL CENTRE at y+ = target_yplus.

    Research plan Sec 4 asks for first-cell y+ < 1.  Bisection on the tanh parameter;
    monotone because yc[0] decreases with s.
    """
    want = target_yplus / retau
    lo, hi = 0.0, smax
    if _tanh_faces(n, hi)[1] * 0.5 > want:
        raise ValueError(f"n={n} too coarse for y+={target_yplus} at Re_tau={retau}")
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        yc0 = 0.5 * _tanh_faces(n, mid)[1]
        if yc0 > want:
            lo = mid
        else:
            hi = mid
    s = 0.5 * (lo + hi)
    return Mesh(_tanh_faces(n, s), kind="tanh", stretch=s)


def for_case(retau, n=None, target_yplus=0.7):
    """The Wu et al. mesh for a nominal Re_tau."""
    return stretched(N_PRESET[retau] if n is None else n, retau, target_yplus)


if __name__ == "__main__":
    print("uniform:", uniform(64))
    print(f"\n{'Re_tau':>7} {'n':>5} {'s':>6} {'y+_1':>7} {'dVmax/dVmin':>12} {'y+_ctr':>8}")
    for r, n in N_PRESET.items():
        m = for_case(r)
        print(f"{r:>7} {n:>5} {m.stretch:>6.3f} {m.yc[0] * r:>7.3f} "
              f"{m.dV.max() / m.dV.min():>12.1f} {m.y_plus(r).max():>8.1f}")
