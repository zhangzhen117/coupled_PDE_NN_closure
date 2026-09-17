#!/usr/bin/env python3
"""One channel case: mesh + DNS fields + the frozen-stress operator.

`case_setup` is the unit every driver builds on.  It lived in
`experiments/s1_reproduce.py`, which made an experiment script a load-bearing dependency
of the whole package; it belongs here.

Paths for computed artefacts are also fixed here so nothing has to guess them:
    data/models/   trained checkpoints
    data/cache/    computed .npz / .npy
    figures/       final figures
    debug/         scratch figures and run logs
"""
import os

from scipy.interpolate import PchipInterpolator

from data import lm_channel as LM
from . import mesh as M, operator as OP

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = os.path.join(ROOT, "data", "models")
CACHE = os.path.join(ROOT, "data", "cache")
FIGURES = os.path.join(ROOT, "figures")
DEBUG = os.path.join(ROOT, "debug")

CASES = (180, 550, 1000, 2000, 5200)


def case_setup(retau, n=None, target_yplus=0.7):
    """Mesh, DNS fields and the frozen operator A for one Reynolds number.

    keys: d (the DNS record), m (mesh), A, tau_f, div_tau, U_dns, U_ref, nu
    """
    d = LM.load(retau)
    m = M.for_case(retau, n=n, target_yplus=target_yplus)
    A = OP.assemble_A(m, d["nu"])
    tau_f = OP.tau_face_from_profile(m, d["y"], d["tau_true"])
    div_tau = OP.div_face(m, tau_f)
    U_dns = PchipInterpolator(d["y"], d["U"])(m.yc)
    return dict(d=d, m=m, A=A, tau_f=tau_f, div_tau=div_tau, U_dns=U_dns,
                U_ref=LM.u_bulk(d), nu=d["nu"])


# Reported taxonomy. Internal checkpoint family names remain unchanged.
# The stress sign used internally is opposite to the manuscript's convention.
# See CONVENTIONS.md for the mapping and the full tangent reaction contribution.

TAXONOMY = [
    ("frozen",     "FROZEN",  r"FROZEN  ($B=0$)",                       "#0b0b0b", "-"),
    ("force",      "A0",      r"A0  force",                             "#eb6834", "--"),
    ("force_pen2w", "A1",     r"A1  force $+$ penalty",                 "#2a78d6", "-."),
    ("tau",        "B0",      r"B0  stress",                            "#d4a017", "--"),
    ("nueff",      "B1",      r"B1  stress, $\int\!\nu_{\rm tan}$",  "#00a5c4", "-"),
    ("hybrid",     "C0",      r"C0  $\nu_t$ $+$ residual",              "#1baf7a", ":"),
]

REPORTED = [t[0] for t in TAXONOMY]
SHORT = {t[0]: t[1] for t in TAXONOMY}
LABEL = {t[0]: t[2] for t in TAXONOMY}
COLOR = {t[0]: t[3] for t in TAXONOMY}
STYLE = {t[0]: t[4] for t in TAXONOMY}
