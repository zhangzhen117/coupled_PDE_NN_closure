#!/usr/bin/env python3
"""Conventions contract.  Every claim in CONVENTIONS.md that code can violate."""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from chan import mesh as M, operator as OP
from data import lm_channel as LM


def test_A_symmetric_on_stretched_mesh():
    m = M.for_case(1000)
    A = OP.assemble_A(m, 1e-3)
    assert np.allclose(A, A.T, atol=0, rtol=0), "flux form must be EXACTLY symmetric"


def test_A_negative_definite():
    m = M.for_case(550)
    w = np.linalg.eigvalsh(OP.assemble_A(m, 1 / 550))
    assert w.max() < 0


def test_eigh_matches_svd():
    """Sec 7.1: A symmetric => singular values are |eigenvalues|, zero non-normality."""
    m = M.for_case(550)
    A = OP.assemble_A(m, 1 / 550)
    assert np.allclose(np.sort(np.abs(np.linalg.eigvalsh(A))),
                       np.sort(np.linalg.svd(A, compute_uv=False)), rtol=1e-10)


def test_A_plus_B_is_one_diffusion_operator():
    """CONVENTIONS.md "The closure Jacobian B": A + B == d/dy((nu+nu_t_eff) d/dy .)."""
    m = M.for_case(1000)
    nu = 1 / 1000.0
    rng = np.random.default_rng(0)
    nut = 0.01 * rng.random(m.n + 1)                 # arbitrary face-wise nu_t_eff
    A = OP.diffusion(m, np.full(m.n + 1, nu))
    B = OP.diffusion(m, nut)
    AB = OP.diffusion(m, nu + nut)
    assert np.allclose(A + B, AB, atol=1e-14, rtol=0)


def test_total_shear_is_linear():
    """CONVENTIONS.md: nu dU/dy - tau = 1 - y, exactly, from the analytic tau_true."""
    for r in (180, 1000, 5200):
        d = LM.load(r)
        tot = d["nu"] * d["dUdy"] - d["tau_true"]
        assert np.abs(tot - (1.0 - d["y"])).max() < 1e-12, r


def test_tau_true_vanishes_at_walls_and_is_antisymmetric():
    for r in (180, 5200):
        d = LM.load(r)
        assert abs(d["tau_true"][0]) < 1e-12 and abs(d["tau_true"][-1]) < 1e-12
        n = len(d["y"]) // 2
        assert np.allclose(d["tau_true"][:n], -d["tau_true"][::-1][:n], atol=1e-12)


def test_reference_solve_recovers_dns_velocity():
    """End-to-end sign check: feeding tau_true back through the operator must return the
    DNS mean velocity.  This is the one test that catches a global sign error.

    Asserts CONVERGENCE, not a magic tolerance: on Wu et al.'s meshes the error is real
    discretisation error (0.9% on the 36-cell Re_tau=180 mesh), so the meaningful
    statement is that it falls under refinement and reaches discretisation level.
    """
    from scipy.interpolate import PchipInterpolator
    for r in (180, 1000, 5200):
        d = LM.load(r)
        errs = []
        for n in (M.N_PRESET[r], 2 * M.N_PRESET[r], 4 * M.N_PRESET[r]):
            m = M.stretched(n, r, 0.7)
            tf = OP.tau_face_from_profile(m, d["y"], d["tau_true"])
            U = OP.solve_frozen(m, tf, d["nu"])
            Uex = PchipInterpolator(d["y"], d["U"])(m.yc)
            errs.append(M.wnorm(U - Uex, m.dV) / M.wnorm(Uex, m.dV))
        assert all(b < a for a, b in zip(errs, errs[1:])), f"Re_tau={r}: {errs} not falling"
        assert errs[0] < 1e-2 and errs[-1] < 5e-4, f"Re_tau={r}: {errs}"


def test_residual_of_reference_pair_is_small():
    """r := R(u_ref; tau_true) should be at discretisation level, not O(1)."""
    from scipy.interpolate import PchipInterpolator
    d = LM.load(1000)
    m = M.for_case(1000)
    U = PchipInterpolator(d["y"], d["U"])(m.yc)
    tf = OP.tau_face_from_profile(m, d["y"], d["tau_true"])
    r = OP.residual(m, U, tf, d["nu"])
    assert M.wnorm(r / m.dV, m.dV) / M.wnorm(OP.source(m) / m.dV, m.dV) < 0.05


def test_svd_cache_reconstructs_and_is_ascending():
    """The eigh fast path must reproduce A exactly and order sigma smallest-first."""
    from chan import metrics as MET
    m = M.for_case(1000)
    At = MET.sym_weighted(OP.assemble_A(m, 1 / 1000.0), m.dV)
    c = MET.svd_cache(At)
    R = c["U"] @ np.diag(c["sigma"]) @ c["V"].T
    assert np.abs(R - At).max() / np.abs(At).max() < 1e-12
    assert np.all(np.diff(c["sigma"]) >= 0)
    ref = MET.svd_cache(At, assume_symmetric=False)
    assert np.allclose(c["sigma"], ref["sigma"], rtol=1e-9)


def test_smallest_sigma_modes_are_smooth():
    """Research plan Sec 3.1 direction reversal: small sigma <-> SMOOTH, large scale.
    If these come out oscillatory the operator assembly has a sign error, and the whole
    'bias is worse than noise' argument is inverted."""
    from chan import metrics as MET
    m = M.stretched(400, 1000, 0.7)
    At = MET.sym_weighted(OP.assemble_A(m, 1 / 1000.0), m.dV)
    c = MET.svd_cache(At)
    for i in range(4):
        v = c["V"][:, i]
        ana = np.sin((i + 1) * np.pi * m.yc / 2.0) * np.sqrt(m.dV)
        ana /= np.linalg.norm(ana)
        assert abs(float(v @ ana)) > 0.999, f"mode {i} is not sin({i+1} pi y/2)"
        assert int(np.sum(np.diff(np.sign(v)) != 0)) == i


def test_sigma_min_weighted_is_mesh_convergent():
    """sigma_min(Atil) -> nu (pi/2)^2; sigma_min(A) does not.  This is why every Stage-2
    predictor is computed in the weighted form."""
    from chan import metrics as MET
    nu = 1 / 1000.0
    target = MET.sigma_min_continuum(nu)
    for n in (200, 800):
        m = M.stretched(n, 1000, 0.7)
        s = np.linalg.svd(MET.sym_weighted(OP.assemble_A(m, nu), m.dV), compute_uv=False)
        assert abs(s.min() / target - 1) < 1e-3, n


def test_P2_equals_P3_identically():
    """The plan's correction #1, asserted: sum_i (u_i.r)^2/sigma_i^2 == ||A^-1 r||^2.
    They are the same number, so they cannot be two rungs of a predictor ladder."""
    from chan import metrics as MET
    m = M.for_case(550)
    At = MET.sym_weighted(OP.assemble_A(m, 1 / 550.0), m.dV)
    c = MET.svd_cache(At)
    r = np.random.default_rng(1).standard_normal(m.n)
    spectral = np.sqrt(np.sum((c["U"].T @ r / c["sigma"]) ** 2))
    direct = np.linalg.norm(np.linalg.solve(At, r))
    assert abs(spectral / direct - 1) < 1e-9, (spectral, direct)


def test_closed_form_modal_response():
    """FORMULATION.md Sec 6: for dtau = sin(k pi y / 2) the response is exactly

        dU = (2 / (k pi nu)) (c_k - cos(k pi y / 2)),   c_k = 1 (k even), 1 - y (k odd)

    Both parities, against a direct solve.  This is the closed form the 1/k amplification
    law and the k=1 cancellation are derived from, so it is asserted rather than quoted.
    """
    nu = 1e-3
    m = M.stretched(2000, 1000, 0.7)
    A = OP.assemble_A(m, nu)
    for k in (1, 2, 3, 4, 8, 16, 32):
        dt = np.sin(k * np.pi * m.yf / 2.0)
        dt[0] = dt[-1] = 0.0
        dU = np.linalg.solve(A, np.diff(dt))
        ck = 1.0 if k % 2 == 0 else 1.0 - m.yc
        ana = (2.0 / (k * np.pi * nu)) * (ck - np.cos(k * np.pi * m.yc / 2.0))
        rel = M.wnorm(dU - ana, m.dV) / M.wnorm(ana, m.dV)
        assert rel < 2e-3, (k, rel)


def _run(mod=None):
    """Self-contained runner -- no pytest in any of the local conda envs."""
    import traceback
    g = mod if mod is not None else globals()
    names = sorted(k for k in g if k.startswith("test_"))
    bad = 0
    for k in names:
        try:
            g[k]()
            print(f"  PASS  {k}")
        except Exception as e:
            bad += 1
            print(f"  FAIL  {k}: {e}")
            traceback.print_exc()
    print(f"\n{len(names) - bad}/{len(names)} passed")
    return bad


if __name__ == "__main__":
    sys.exit(min(_run(), 1))
