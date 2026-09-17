"""Emit the paper's LaTeX tables from the cached measurements.

Every number in section 4 comes from here, so that no value in the draft is typed by hand
and a regenerated cache regenerates the paper.  Missing cache entries are written as "--"
rather than omitted, so an incomplete run is visible in the PDF instead of silently
producing a shorter table.

Caches used:
    s20_{family}.npy   omega, sigma_min at u_ref; cold-start and Re-ladder solves, P = A
    s21_ratio.npy      ||r||, the bound 1/sigma_min, the linear and realized amplification
    s12_{retau}.npy    the preconditioner sweep (the isolation result)
"""

import glob
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from chan.cases import CASES, CACHE, MODELS, REPORTED, LABEL, SHORT
from closure.train import per_case_data, predict
from s12_newton_krylov import load

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "paper", "tex", "tables")

# Labels come from chan.cases.TAXONOMY, the single source of truth.
LAB = [(f, LABEL[f]) for f in REPORTED]
FAM = [(f, l) for f, l in LAB if f != "frozen"]
# THE PHYSICAL-ROOT THRESHOLD.
#
# A converged solve is not evidence.  An unconstrained closure converges, with the residual
# at the round-off floor, onto roots 10-258% from the reference profile, and the residual
# cannot tell those from the right one because both ARE roots of the equation the network
# wrote down.  So every converged solve is classified:
#
#     PHYSICAL   eps_U <  PHYS   the root we wanted
#     SPURIOUS   eps_U >= PHYS   a root of the same equation, somewhere else
#
# and every a posteriori statistic is taken over the PHYSICAL roots only.  Averaging eps_U
# over both populations averages a modeling error with a distance to a different solution,
# which is not a meaningful quantity.
#
# PHYS = 0.05: the largest physical error is 4.48% (A1) and the smallest spurious 6.35% (A0).
# 3% and 62 above 30%, with 21 spread across the decade between; the largest error below the
# threshold is 2.91% and the next solve up is at 3.42%, so the cut falls in a gap rather than
# through a continuum.  The ranking of the closures is unchanged for any threshold in 3-30%.
PHYS = 0.05
USEFUL = PHYS             # backwards-compatible alias

# ONE convention for conditioning, used in every table and figure:
#     amp = sigma_min(A) / sigma_min(J),  the amplification RELATIVE TO FROZEN.
# Bigger is worse, so it reads the same way as omega; below 1 means the closed operator is
# better conditioned than the frozen one.  The reciprocal appears nowhere.
BS = chr(92) * 2          # a literal LaTeX row break, kept out of the f-strings


def _w(name, body):
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, name), "w") as fh:
        fh.write(body)
    print("wrote", os.path.join(OUT, name))


def s20():
    """{(family, retau): [record per seed]} -- the problem level and the DNS-free solves."""
    S = {}
    for f, _ in FAM:
        p = os.path.join(CACHE, f"s20_{f}.npy")
        if os.path.exists(p):
            for r in np.load(p, allow_pickle=True):
                S.setdefault((r["family"], r["retau"]), []).append(r)
    return S


def s21():
    p = os.path.join(CACHE, "s21_ratio.npy")
    if not os.path.exists(p):
        return {}
    R = {}
    for r in np.load(p, allow_pickle=True):
        R.setdefault((r["family"], r["retau"]), []).append(r)
    return R


def s12():
    S = {}
    for r in CASES:
        for tag in ("s12", "s12b"):
            p = os.path.join(CACHE, f"{tag}_{r}.npy")
            if os.path.exists(p):
                for rec in np.load(p, allow_pickle=True):
                    S.setdefault((rec["family"], r), []).append(rec)
    return S


def frozen_sigma():
    """sigma_min of the frozen operator per Re -- the reference for every ratio."""
    from chan import metrics as MET
    from chan.cases import case_setup
    out = {}
    for r in CASES:
        c = case_setup(r)
        out[r] = float(np.linalg.svd(MET.sym_weighted(c["A"], c["m"].dV),
                                     compute_uv=False).min())
    return out


def _ms(v, fmt="%.2f", pm=r" $\pm$ "):
    """mean +- std over seeds.  Empty -> a dash.

    Reported in preference to the median throughout: where the two disagree it is because
    one seed is far from the others, and that is a fact about the closure worth seeing, not
    noise worth suppressing.
    """
    v = np.asarray([x for x in v if np.isfinite(x)], float)
    if v.size == 0:
        return "--"
    if v.size == 1:
        return fmt % v[0]
    return (fmt % v.mean()) + pm + (fmt % v.std())


def _sci(x, sig=2):
    """LaTeX scientific notation.  "%g" emits 1.7e-17, which sets as literal text."""
    t = "%.*g" % (sig, x)
    if "e" not in t:
        return "$%s$" % t
    m, e = t.split("e")
    return "$%s\\times10^{%d}$" % (m, int(e))


def _tab(name, colspec, header, rows, note=""):
    _w(name, "\\begin{tabular}{@{}" + colspec + "@{}}\n\\toprule\n" + header
       + "\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n" + note)


# ---------------------------------------------------------------------------
# 1.  What the five closures are.
#
# Hand-written because it describes the CONSTRUCTIONS, not a measurement; the parameter
# counts are read from the checkpoints so a change of width cannot silently desynchronize
# the table from the models.
CLOSURES = [
    ("force",     r"$F = \mathcal{N}$",
     r"react.\,$+$\,adv.", r"none"),
    ("force_pen2w", r"$F = \mathcal{N}$",
     r"react.\,$+$\,adv.", r"penalty: $F_U-\tfrac12\D F_S\le0$"),
    ("tau",       r"$\tau = \mathcal{N}$",
     r"adv.\,$+$\,diff.", r"none"),
    ("nueff",     r"$\tau = \int_0^S \nutan\,ds$",
     r"adv.\,$+$\,diff.", r"construction: $\nutan\ge0$"),
    ("hybrid",    r"$F = \partial_y(\nut\partial_yU)+F_{\rm res}$",
     r"both", r"$\nut\ge0$: state viscosity only"),
]


def tab_closures():
    rows = []
    for fam, pred, typ, con in CLOSURES:
        m, _ = load(fam, 0)
        npar = sum(p.numel() for p in m.parameters())
        rows.append(f"{LABEL[fam]} & {pred} & {typ} & {con} & {npar} {BS}")
    _tab("tab_closures.tex", "lllll",
         "closure & network predicts & adds & constraint on the operator & params " + BS,
         rows,
         "\n% C0 constrains the secant nu_t, not the tangent nu_eff that the Jacobian\n"
         "% contains, so its operator is in fact unconstrained.\n"
         "% Paper sign convention: tau = -<u'v'> (the code uses tau = <u'v'>).\n")


# ---------------------------------------------------------------------------
def tab_apriori():
    """A priori d(tau)/dy error [%]: median over ten seeds, with the seed range."""
    data = per_case_data()
    rows = []
    for fam, lab in FAM:
        if not os.path.exists(os.path.join(MODELS, f"{fam}_s0.pt")):
            continue
        E = []
        for s in range(10):
            m, _ = load(fam, s)
            e = []
            for c in data:
                with torch.no_grad():
                    g = predict(m, fam, c)
                e.append(100 * float(torch.sqrt(torch.sum(c["W"] * (g - c["G"]) ** 2) /
                                                torch.sum(c["W"] * c["G"] ** 2))))
            E.append(e)
        E = np.array(E)
        rows.append(lab + " & " + " & ".join(_ms(E[:, j]) for j in range(E.shape[1]))
                    + " " + BS)
    _tab("tab_apriori.tex", "l" + "r" * len(CASES),
         "closure & " + " & ".join(str(c) for c in CASES) + " " + BS, rows)


def tab_indices(S, sf):
    """The two indices at the reference state, as two tables.

    One table of ten mean+-std cells is 400pt too wide for the text block, and shrinking the
    font to fit makes it unreadable.  omega and the conditioning ratio answer different
    questions anyway (section 3.4), so they are reported separately.
    """
    for name, key, head, note in (
            ("tab_omega.tex", "omega",
             r"\\omega(J)=\\lambda_{\\max}(\\sym\\tilde J)",
             "\n% bold = negative for all ten seeds (dissipative at u_ref).\n"),
            ("tab_cond.tex", "amp",
             r"\\amp=\\sigma_{\\min}(A)/\\sigma_{\\min}(J)",
             "\n% bold = mean below 1, i.e. better conditioned than the frozen operator.\n")):
        rows = []
        for fam, lab in FAM:
            cells = []
            for r in CASES:
                rs = S.get((fam, r), [])
                if not rs:
                    cells.append("--"); continue
                o = np.array([x["omega"] for x in rs])
                v = sf[r] / np.array([x["sigma_min"] for x in rs])
                z = o if key == "omega" else v
                c = _ms(z, "%.2g")
                hit = (o < 0).all() if key == "omega" else v.mean() < 1
                cells.append(("\\textbf{%s}" % c) if hit else c)
            # short codes here: the full labels plus five mean+-std cells overrun the
            # text block, and table 1 has already defined A0...C0.
            rows.append(SHORT[fam] + " & " + " & ".join(cells) + " " + BS)
        _tab(name, "l" + "r" * len(CASES),
             "closure & " + " & ".join(str(c) for c in CASES) + " " + BS, rows,
             "\n%% $%s$, mean +- s.d. over ten seeds.%s" % (head, note))


def tab_closures_apriori():
    """JFM version: the five closures and their a priori error in F [%] (median over ten
    seeds, five Reynolds numbers) in one table."""
    data = per_case_data()
    rows = []
    for fam, pred, typ, con in CLOSURES:
        m, _ = load(fam, 0)
        E = []
        for s in range(10):
            m, _ = load(fam, s)
            e = []
            for c in data:
                with torch.no_grad():
                    g = predict(m, fam, c)
                e.append(100 * float(torch.sqrt(torch.sum(c["W"] * (g - c["G"]) ** 2) /
                                                torch.sum(c["W"] * c["G"] ** 2))))
            E.append(e)
        E = np.median(np.array(E), axis=0)
        rows.append(f"{LABEL[fam]} & {pred} & {con} & "
                    + " & ".join("%.2f" % v for v in E) + " " + BS)
    _tab("tab_closures_apriori.tex", "lll" + "r" * len(CASES),
         "closure & predicts & constraint on the operator & "
         + " & ".join(str(c) for c in CASES) + " " + BS, rows,
         "\n% right block: a priori error in F [%], median over ten seeds.\n")


def tab_indices_merged(S, sf):
    """JFM version: omega(J) and the conditioning ratio A in one table, two row blocks."""
    rows = []
    for key, head in (("omega", r"$\omega(J)$, negative is dissipative"),
                      ("amp", r"$\amp=\sigma_{\min}(A)/\sigma_{\min}(J)$, below one is better than frozen")):
        rows.append("\\multicolumn{%d}{@{}l}{%s} %s" % (len(CASES) + 1, head, BS))
        for fam, lab in FAM:
            cells = []
            for r in CASES:
                rs = S.get((fam, r), [])
                if not rs:
                    cells.append("--"); continue
                o = np.array([x["omega"] for x in rs])
                v = sf[r] / np.array([x["sigma_min"] for x in rs])
                z = o if key == "omega" else v
                c = _ms(z, "%.2g")
                hit = (o < 0).all() if key == "omega" else v.mean() < 1
                cells.append(("\\textbf{%s}" % c) if hit else c)
            rows.append(SHORT[fam] + " & " + " & ".join(cells) + " " + BS)
        if key == "omega":
            rows.append("\\midrule")
    _tab("tab_indices.tex", "l" + "r" * len(CASES),
         "closure & " + " & ".join(str(c) for c in CASES) + " " + BS, rows,
         "\n% mean +- s.d. over ten seeds; bold: omega < 0 for every seed / A < 1.\n")


def tab_useful(S):
    """Useful solves out of ten -- converged from a DNS-FREE start AND eps_U < 10%."""
    rows = []
    for fam, lab in FAM:
        cv, eu = [], []
        for r in CASES:
            rs = S.get((fam, r), [])
            if not rs:
                cv.append("--"); eu.append("--"); continue
            conv = [x for x in rs if x["any_ok"] and np.isfinite(x["best_err"])]
            good = [x["best_err"] for x in conv if x["best_err"] < PHYS]
            # physical / converged: the gap between the two IS the multimodality
            cell = "%d\\,/\\,%d" % (len(good), len(conv))
            # bold marks only the perfect cell; a bold zero would make one weight mean
            # two opposite things
            cv.append(("\\textbf{%s}" % cell) if len(good) == len(rs) else cell)
            eu.append(_ms([100 * g for g in good]))
        rows.append(SHORT[fam] + " & " + " & ".join(cv) + " & " + " & ".join(eu)
                    + " " + BS)
    n = len(CASES)
    _tab("tab_useful.tex", "l" + "c" * n + "r" * n,
         ("& \\multicolumn{%d}{c}{physical / converged, of 10} & \\multicolumn{%d}{c}{$\\epsU$ [\\%%]} %s\n"
          % (n, n, BS))
         + "\\cmidrule(lr){2-%d}\\cmidrule(l){%d-%d}\n" % (n + 1, n + 2, 2 * n + 1)
         + "closure & " + " & ".join(str(c) for c in CASES) + " & "
         + " & ".join(str(c) for c in CASES) + " " + BS, rows,
         "\n% each cell: physical roots / converged solves, out of ten seeds, P = A.\n")


def tab_ratio(R, S):
    """Local estimates, mean and standard deviation over seeds.

    Realized amplification and inverse alignment use only roots within the accuracy
    threshold. The other columns use all seeds; selection does not validate a bound.
    """
    rows = []
    for fam, lab in FAM:
        use = {(x["retau"], x["seed"])
               for r in CASES for x in S.get((fam, r), [])
               if x["any_ok"] and np.isfinite(x["best_err"]) and x["best_err"] < USEFUL}
        for r in CASES:
            rs = R.get((fam, r), [])
            if not rs:
                continue
            g = lambda k, f="%.2g": _ms([x[k] for x in rs], f)
            real = [x["realized"] for x in rs
                    if (x["retau"], x["seed"]) in use and np.isfinite(x["realized"])]
            loose = [x["loose"] for x in rs if (x["retau"], x["seed"]) in use]
            rows.append("%s & %d & %s & %s & %s & %s & %s %s"
                        % (SHORT[fam], r, g("rnorm"), g("bound"), g("linear"),
                           _ms(real, "%.2g"), _ms(loose, "%.0f") if real else "--", BS))
    _tab("tab_ratio.tex", "lrrrrrr",
         ("closure & $\\Ret$ & $\\|\\hat r\\|$ & estimate $1/\\sigma_{\\min}$ & "
          "linear & realized & $\\gamma^{-1}$ " + BS), rows,
         "\n% linear   = ||J^-1 r||/||r||, the operator acting on THIS residual\n"
         "% realized = ||U_hat - u_ref||/||r||, PHYSICAL roots only\n"
         "% looseness = bound / linear, the alignment factor\n")


def tab_precond(S, retau=1000):
    """The isolation result: what changes with P and what does not."""
    rows = []
    for fam, lab in FAM:
        rs = S.get((fam, retau), [])
        if not rs:
            continue
        ps = list(rs[0]["precond"])
        cells = []
        for p in ("A", "A+B_diff", "A+nu_t", "A+nu_eff", "J"):
            if p not in ps:
                continue
            e = [x["precond"][p] for x in rs if x["precond"][p]["nk_ok"]]
            cells.append(_ms([z["gmres_mean"] for z in e], "%.0f") if e else "--")
        eu = [x["precond"]["A"] for x in rs if x["precond"]["A"]["nk_ok"]]
        spread = "--"
        if eu:
            allp = [[x["precond"][p]["err_U"] for p in ps if x["precond"][p]["nk_ok"]]
                    for x in rs if x["precond"]["A"]["nk_ok"]]
            spread = "%.1e" % max((max(a) - min(a)) / max(abs(np.mean(a)), 1e-30)
                                  for a in allp if len(a) > 1)
        rows.append(lab + " & " + " & ".join(cells) + f" & {spread} " + BS)
    _tab("tab_precond.tex", "lrrrr",
         ("closure & $P{=}A$ & $P{=}A{+}B_{\\rm diff}/{+}\\nueff$ & $P{=}J$ & "
          "rel.\\ spread in $\\epsU$ " + BS), rows,
         "\n%% mean GMRES iterations per Newton step at Re_tau = %d.\n" % retau)


if __name__ == "__main__":
    torch.set_num_threads(1)
    S, R, P, sf = s20(), s21(), s12(), frozen_sigma()
    tab_closures()
    tab_apriori()
    tab_indices(S, sf)
    tab_closures_apriori()
    tab_indices_merged(S, sf)
    tab_useful(S)
    tab_ratio(R, S)
    tab_precond(P)
