#!/usr/bin/env python3
"""Figures for the Newton-Krylov conditioning study.

  fig1  a priori test -- the three trained closure families against the DNS reference
  fig2  the two indices -- omega(J) and sigma_min(J), both at the reference state
  fig3  the error bound 1/sigma_min against the amplification actually realized
  fig4  multimodality -- converged solutions, physical roots against spurious ones

  fig_secant  C0's secant nu_t against its tangent nu_eff.  NOT in the paper: the result is
              reported as numbers in section 4.2, the figure was cut for space.  Kept because
              it is the evidence behind those numbers.

Output files are numbered by the order the figures appear in the paper, so fig5_*.png is
Figure 5 in the PDF.  Function names follow the content, not the number.

Exploratory output belongs in debug/, not here.
"""
import glob
import os
import sys

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
torch.set_num_threads(1)

from chan import mesh as M, metrics as MET
from chan.cases import (CASES, CACHE, FIGURES, case_setup,
                        COLOR as CFG_C, LABEL as CFG_LAB, STYLE as CFG_LS,
                        SHORT as CFG_SH, REPORTED)
from data import lm_channel as LM
from s12_newton_krylov import CONFIGS as _ALL_CONFIGS, load, operators, fixed_point
# reported set, in taxonomy order (A0 A1 A2 B0 B1 C0 C1), not discovery order
# labels come from chan.cases.TAXONOMY, the single source of truth -- s12's own CONFIGS
# still carries the pre-taxonomy names (C, C+, B) and using them relabelled the panels
_D = dict(_ALL_CONFIGS)
CONFIGS = [(f, CFG_LAB[f]) for f in REPORTED if f in _D]

os.makedirs(FIGURES, exist_ok=True)

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#4a3aa7"]
RE_C = dict(zip(CASES, SERIES))
P_C = {"A": "#eb6834", "A+B_diff": "#2a78d6", "A+nu_t": "#1baf7a",
       "A+nu_eff": "#1baf7a",
       "J": "#0b0b0b"}
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#b9b8b2"

# Serif text and math (STIX, shipped with matplotlib) so the figures set in the same
# family as the running text; DejaVu Sans at equal nominal size reads a full step larger.
# Figures are drawn at \textwidth, so these sizes are the printed sizes: one step below
# the 11 pt body, at the size of the captions.
plt.rcParams.update({
    "font.family": "STIXGeneral", "mathtext.fontset": "stix",
    "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9.5,
    "axes.edgecolor": INK2, "axes.linewidth": 0.7, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "figure.facecolor": "white", "axes.facecolor": "white",
    "savefig.facecolor": "white", "legend.frameon": False,
    "lines.linewidth": 1.6, "grid.color": MUTED, "grid.linewidth": 0.5,
})


def _tidy(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.35, linewidth=0.5)
    ax.set_axisbelow(True)


NSEED_TXT = 10
TW = 5.3            # JFM \textwidth (32 pc) in inches; figures are drawn at printed size

# The physical-root threshold, matching make_tables.PHYS.  A converged solve with
# eps_U below this is the root we wanted; above it, a root of the same equation
# somewhere else.  Figures 3 and 4 both classify by it.
PHYS = 0.05


def _band_legend(ax, n=NSEED_TXT):
    """Key for the two shaded bands, one column in the upper left of panel B.

    Median, interquartile range and min-max envelope -- see _band for why this figure uses
    order statistics where the tables use mean +- s.d.  A light frame keeps it legible where
    it overlaps the B0 envelope.
    """
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    h = [Line2D([], [], color=INK, lw=1.8, marker="o", ms=4.5, mew=0,
                label=f"median of {n} seeds"),
         Patch(facecolor=INK, alpha=.30, lw=0, label="interquartile range"),
         Patch(facecolor=INK, alpha=.10, lw=0, label="min-max over seeds")]
    lg = ax.legend(handles=h, loc="upper right", ncol=1, fontsize=7.5, frameon=True,
                   framealpha=.85, edgecolor="none", handlelength=1.8)
    return lg


def study():
    """All s12 records, keyed (family, retau) -> list over seeds."""
    recs = {}
    # s12_*  = families measured together; s12b_* = closures added later and measured
    # separately.  A bare "s12_*" glob silently drops the latter.
    for f in sorted(glob.glob(os.path.join(CACHE, "s12_*.npy"))
                    + glob.glob(os.path.join(CACHE, "s12b_*.npy"))):
        for r in np.load(f, allow_pickle=True):
            recs.setdefault((r["family"], r["retau"]), []).append(r)
    if not recs:
        sys.exit("no s12_*.npy in data/cache -- run experiments/s12_newton_krylov.py")
    return recs


def med(rs, key):
    return float(np.median([r[key] for r in rs]))


# ------------------------------------------------------------------ figure 1
def fig_apriori(seed=0):
    """A priori: true DNS features in, no solve anywhere."""
    fams = [(f, CFG_LAB[f], CFG_C[f], CFG_LS[f])
            for f, _ in CONFIGS if f != "frozen"]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(TW, 2.55), constrained_layout=True)
    err = {}
    for k, (fam, lab, col, ls) in enumerate(fams):
        mdl, _ = load(fam, seed)
        et, eg = [], []
        for r in CASES:
            d = LM.load(r)
            y, U = d["y"], torch.tensor(d["U"], dtype=torch.float64)
            S = torch.tensor(d["dUdy"], dtype=torch.float64)
            with torch.no_grad():
                # same dispatch as closure.train.predict: the flux forms expose
                # g_total, the force forms dtau_dy.  Hard-coding "hybrid" here silently
                # excluded B0 and B1 from the figure.
                gp = (mdl.g_total(U, S, torch.tensor(y, dtype=torch.float64),
                                  d["nu"], d["retau"])
                      if fam.startswith(("hybrid", "nueff", "tau"))
                      else mdl.dtau_dy(U, S, d["nu"], d["retau"])).numpy()
            # the paper writes tau = -<u'v'> while the code carries tau = <u'v'>, so the
            # displayed stress and its divergence are the negatives of the code's.
            gp = -gp
            gt, tt = -np.gradient(d["tau_true"], y), -d["tau_true"]
            tp = np.concatenate([[0.], np.cumsum(.5 * (gp[1:] + gp[:-1]) * np.diff(y))])
            h = (y <= 1.0) & (y > 0)
            yp, c = y[h] * d["retau"], RE_C[r]
            if k == 0:
                ax.semilogx(yp, tt[h], "-", color=c, lw=2.8, alpha=.25, zorder=1)
                ax2.semilogx(yp, gt[h], "-", color=c, lw=2.8, alpha=.25, zorder=1)
            ax.semilogx(yp, tp[h], ls, color=c, lw=1.3, zorder=2)
            ax2.semilogx(yp, gp[h], ls, color=c, lw=1.3, zorder=2)
            w = np.gradient(y)
            et.append(np.sqrt(np.sum(w * (tp - tt) ** 2) / np.sum(w * tt ** 2)) * 100)
            eg.append(np.sqrt(np.sum(w * (gp - gt) ** 2) / np.sum(w * gt ** 2)) * 100)
        err[fam] = (np.mean(et), np.mean(eg))
    from matplotlib.lines import Line2D
    # the per-closure errors used to sit in the panel titles, where five of them ran off the
    # right edge of the page; they belong with the line styles they describe
    hc = [Line2D([], [], ls=CFG_LS[f], color=INK, lw=1.4,
                 label=f"{CFG_SH[f]}   {err[f][0]:.2f}% / {err[f][1]:.2f}%")
          for f, *_ in fams]
    hc.append(Line2D([], [], color=INK, lw=2.8, alpha=.35, label="DNS"))
    hr = [Line2D([], [], color=RE_C[r], lw=1.6, label=f"$Re_\\tau$={r}") for r in CASES]
    ax.set_xlabel("$y^+$"); ax.set_xlim(.1, 6e3); ax.set_ylim(0, 1.25)
    ax.set_ylabel(r"$\tau/u_\tau^2$")
    ax.set_title(r"A.  Reynolds shear stress $\tau=-\langle u'v'\rangle$", loc="left")
    _tidy(ax)
    # the closure key sits above both panels: inside panel A it covered the near-wall rise
    fig.legend(handles=hc, loc="outside upper center", ncol=3, fontsize=8, frameon=False,
               handlelength=2.2, columnspacing=1.6,
               title=r"closure: error in $\tau$ / in $F$", title_fontsize=8)
    ax2.set_xlabel("$y^+$"); ax2.set_xlim(.5, 6e3)
    ax2.set_yscale("symlog", linthresh=1.0); ax2.axhline(0, color=MUTED, lw=.8)
    ax2.set_ylabel(r"$F=d\tau/dy$")
    ax2.set_title(r"B.  Reynolds force $F=d\tau/dy$", loc="left")
    _tidy(ax2); ax2.legend(handles=hr, loc="upper right", fontsize=8)
    q = os.path.join(FIGURES, "fig1_apriori.png"); fig.savefig(q, dpi=200)
    fig.savefig(q[:-4] + ".pdf"); plt.close(fig)
    print("  a priori (tau %, div %):", {f: (round(v[0], 3), round(v[1], 3))
                                         for f, v in err.items()})
    return q




# ------------------------------------------------------------------ caches
def s20():
    """{(family, retau): [record per seed]}: the two indices and the DNS-free solves."""
    R = {}
    for f, _ in CONFIGS:
        p = os.path.join(CACHE, f"s20_{f}.npy")
        if os.path.exists(p):
            for r in np.load(p, allow_pickle=True):
                R.setdefault((r["family"], r["retau"]), []).append(r)
    if not R:
        sys.exit("no s20_*.npy in data/cache -- run experiments/s20_summary.py")
    return R


def _load(name, hint):
    p = os.path.join(CACHE, name)
    if not os.path.exists(p):
        sys.exit(f"missing {p} -- run experiments/{hint}")
    return list(np.load(p, allow_pickle=True))


def frozen_sigma():
    """sigma_min of A alone, per Re: the reference every conditioning ratio is taken to."""
    return {r: float(np.linalg.svd(MET.sym_weighted(case_setup(r)["A"],
                                                    case_setup(r)["m"].dV),
                                   compute_uv=False).min()) for r in CASES}


def _band(a, x, vals, col, ls, lab=None, jit=0.0):
    """Median line, interquartile band, min-max envelope.

    This figure alone departs from the mean +- s.d. reported in every table, and does so
    deliberately.  Both panels are log or symlog and both quantities are heavy-tailed across
    seeds, which breaks mean +- s.d. as a band in two ways:

      * in panel A it CLOSES OVER outliers -- B1 at Re_tau = 5200 has mean + s.d. = -0.009
        while one of its ten seeds sits at +0.085, so the band asserted uniform dissipation
        that section 3.5 explicitly denies;
      * in panel B the quantity is strictly positive and mean - s.d. is NEGATIVE in 11 of 25
        cells, and fill_between clips a negative lower edge to the bottom of a log axis, so
        those bands appeared to reach zero.

    Order statistics have neither problem: the quartiles and the extremes are positive
    whenever the data are, and the min-max envelope shows the outlying seed instead of
    swallowing it.  `jit` is unused here and kept for signature compatibility.
    """
    v = np.array([np.median(z) for z in vals])
    a.plot(x, v, ls, color=col, lw=1.9, marker="o", ms=4.8, mew=0, label=lab, zorder=4)
    a.fill_between(x, [np.percentile(z, 25) for z in vals],
                   [np.percentile(z, 75) for z in vals], color=col, alpha=.30, lw=0,
                   zorder=2)
    a.fill_between(x, [z.min() for z in vals], [z.max() for z in vals],
                   color=col, alpha=.10, lw=0, zorder=1)


# ------------------------------------------------------------------ figure 2
def fig_indices():
    """The two indices, both measured at the reference state and neither solver-dependent.

    LEFT, omega(J) = lambda_max(sym Jtil).  Negative is dissipative; if it were negative at
    every state the root would be unique (Browder-Minty).  Only B1 is negative at every
    Reynolds number, and it is the only closure constructed to be.

    RIGHT, sigma_min(A) divided by sigma_min(J), so that both panels read the same way:
    UP IS WORSE.  Below one means the closed operator is BETTER conditioned than the frozen
    one -- the constraint does not merely avoid harm, it buys a tighter error bound, and for
    B1 it buys up to 373x (a ratio of 0.0027).
    """
    R, sf = s20(), frozen_sigma()
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(TW, 2.65), constrained_layout=True)
    for k, (fam, _) in enumerate(CONFIGS):
        re = [r for r in CASES if (fam, r) in R]
        if not re:
            continue
        col, ls = CFG_C[fam], CFG_LS[fam]
        jit = 0.055 * (k - 2)
        _band(ax, re, [np.array([x["omega"] for x in R[(fam, r)]]) for r in re],
              col, ls, CFG_LAB[fam], jit)
        _band(ax2, re, [sf[r] / np.array([x["sigma_min"] for x in R[(fam, r)]])
                        for r in re], col, ls, None, jit)
    ax.axhline(0, color=INK, lw=1.2, zorder=1)
    ax.set_yscale("symlog", linthresh=1e-2)
    # the negative values reach only -0.12, so a symmetric range wastes four decades and
    # squashes the band that decides every outcome in the paper
    ax.set_ylim(-1.0, 3e5)
    ax.set_ylabel(r"$\omega(J)=\lambda_{\max}(\mathrm{sym}\,\tilde J)$")
    ax.set_title(r"A.  Dissipation index $\omega$", loc="left")
    ax2.axhline(1, color=INK, lw=1.2, zorder=1)
    ax2.set_yscale("log")
    ax2.set_ylabel(r"$\sigma_{\min}(A)\,/\,\sigma_{\min}(J)$")
    ax2.set_title(r"B.  Conditioning against frozen", loc="left")
    for a in (ax, ax2):
        a.set_xscale("log"); a.set_xticks(CASES, [str(c) for c in CASES])
        a.set_xlabel(r"$Re_\tau$"); _tidy(a)
    from matplotlib.lines import Line2D
    hh, ll = ax.get_legend_handles_labels()
    hh.append(Line2D([], [], color=INK, lw=1.2)); ll.append(r"$\omega=0$ (A), frozen $\mathcal{A}=1$ (B)")
    fig.legend(hh, ll, loc="outside upper center", ncol=3, fontsize=8, frameon=False,
               handlelength=2.2, columnspacing=1.6)
    _band_legend(ax2)
    q = os.path.join(FIGURES, "fig2_indices.png"); fig.savefig(q, dpi=200)
    fig.savefig(q[:-4] + ".pdf"); plt.close(fig)
    return q


# ------------------------------------------------------------------ figure 3
def fig_multimodal(retau=5200, scatter=False):
    """MULTIPLICITY: the solver converges, and not always to the solution that was wanted.

    Every curve here is a converged solve from a DNS-free start, at a residual reduced to
    the round-off floor.  A convergence count alone would credit A0 with 8 of 10 at
    Re_tau = 1000.  The profiles show what that hides: the solutions split into a tight
    cluster on the DNS profile and a scatter at 30-130% error, with nothing in between, and
    the residual cannot tell them apart.  Multiplicity is a property of the closed
    EQUATION, so no solver, tolerance or globalization can remove it.

    Rows: the mean velocity; the Reynolds shear stress tau = (1 - y) - nu S recovered from
    the converged field through the momentum balance (paper sign, tau = -<u'v'>), which is
    exact at a root for every closure form and so needs no per-form stress evaluation; and
    eps_U for every converged solve.
    """
    from closure.jacobian import _cell_gradient_coeffs
    F = _load("s22_fields.npy", "s22_profiles.py")
    fams = [(f, l) for f, l in CONFIGS if f != "frozen"]
    nf = len(fams)
    small = {"font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
             "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7}
    plt.rcParams.update(small)
    # the eps_U scatter row duplicates table 4 and is off by default (JFM page budget)
    fig = plt.figure(figsize=(TW, 3.5 if not scatter else 4.8), constrained_layout=True)
    gs = fig.add_gridspec(3 if scatter else 2, nf,
                          height_ratios=(1.0, 1.0, 0.85) if scatter else (1.0, 1.0))
    c = case_setup(retau)
    m, nu = c["m"], c["nu"]
    cl, cc, cr = _cell_gradient_coeffs(m)

    def stress(U):
        Ue = np.concatenate([[0.0], U, [0.0]])
        S = cl * Ue[:-2] + cc * Ue[1:-1] + cr * Ue[2:]
        return (1.0 - m.yc) - nu * S

    tau_dns = stress(c["U_dns"])

    def distinct(fam, r, tol=1e-6):
        """One record per seed and per DISTINCT converged solution.  The cold start and the
        Reynolds-number continuation reach the same solution to ~1e-11 wherever both
        converge, except in four (closure, Re_tau, seed) cases of A0 and B0 where they
        reach two different roots; those keep both."""
        out = []
        for s in range(NSEED_TXT):
            rs = [o for o in F if o["family"] == fam and o["retau"] == r
                  and o["seed"] == s and o["ok"]]
            keep = []
            for o in rs:
                if all(np.linalg.norm(o["U"] - q["U"]) > tol * np.linalg.norm(q["U"])
                       for q in keep):
                    keep.append(o)
            out.extend(keep)
        return out

    for k, (fam, lab) in enumerate(fams):
        axU = fig.add_subplot(gs[0, k]); axT = fig.add_subplot(gs[1, k])
        rs = distinct(fam, retau)
        ng = nb = 0
        for o in rs:
            good = o["err"] < PHYS
            ng, nb = ng + good, nb + (not good)
            col, lw = ("#2a78d6", 1.3) if good else ("#b3411a", 1.1)
            axU.semilogx(o["y"], o["U"], "-", lw=lw, color=col, alpha=.85)
            axT.semilogx(o["y"], stress(o["U"]), "-", lw=lw, color=col, alpha=.85)
        axU.semilogx(m.yc, c["U_dns"], "--", color=INK, lw=2.0, zorder=5)
        axT.semilogx(m.yc, tau_dns, "--", color=INK, lw=2.0, zorder=5)
        # counts as text in the empty upper-left corner; the line key is one figure-level
        # legend above the panels, so nothing sits on the DNS curve
        axU.text(.04, .96, f"physical {ng}\nspurious {nb}", transform=axU.transAxes,
                 ha="left", va="top", fontsize=7, color=INK2).set_in_layout(False)
        for ax in (axU, axT):
            ax.set_xlim(m.yc[0] * .8, 1.05); ax.set_xlabel("$y/h$"); _tidy(ax)
            ax.set_xticks([1e-3, 1e-2, 1e-1, 1e0], ["$10^{-3}$", "", "$10^{-1}$", "$1$"])
            ax.tick_params(axis="x", which="minor", bottom=False)
        if k == 0:
            axU.set_ylabel("$U^+$"); axT.set_ylabel(r"$\tau/u_\tau^2$")
        else:
            axU.set_yticklabels([]); axT.set_yticklabels([])
        # one velocity range for all panels so the closures compare; a spurious solution
        # of B0 reaches U^+ = -50 and would otherwise squash the DNS curve into a line
        axU.set_ylim(-6, 1.15 * c["U_dns"].max())
        axT.set_ylim(-0.6, 1.6); axT.axhline(0, color=MUTED, lw=.8)
        axU.set_title(lab, loc="left", fontsize=8)
        axT.set_title(r"$\tau=(1-y)-\nu S$", loc="left", fontsize=8, color=INK2)
        if k == 0:
            from matplotlib.lines import Line2D
            fig.legend(handles=[Line2D([], [], color="#2a78d6", lw=1.3, label="physical solution"),
                                Line2D([], [], color="#b3411a", lw=1.1, label="spurious solution"),
                                Line2D([], [], color=INK, lw=2.0, ls="--", label="DNS")],
                       loc="outside upper center", ncol=3, fontsize=8, frameon=False,
                       handlelength=2.0, columnspacing=2.0)

    if not scatter:
        q = os.path.join(FIGURES, "fig4_multimodal.png"); fig.savefig(q, dpi=200)
        fig.savefig(q[:-4] + ".pdf"); plt.close(fig)
        plt.rcParams.update({"font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9.5,
                             "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8})
        return q
    ax = fig.add_subplot(gs[2, :])
    for i, (fam, lab) in enumerate(fams):
        for j, r in enumerate(CASES):
            e = np.array([o["err"] for o in distinct(fam, r)]) * 100
            if not e.size:
                continue
            x = i + np.linspace(-.30, .30, len(CASES))[j]
            ax.scatter(np.full(e.shape, x), np.clip(e, 1e-2, None), s=13,
                       facecolor=np.where(e < 100 * PHYS, "#2a78d6", "#b3411a"),
                       edgecolor="none", alpha=.8, zorder=3)
    ax.axhline(100 * PHYS, color=INK, lw=1.1)
    ax.annotate(r"physical / spurious threshold, $\epsilon_{\rm phys}=%g\%%$" % (100 * PHYS),
                (.99, .53), xycoords="axes fraction", fontsize=8, ha="right",
                color=INK).set_in_layout(False)
    ax.set_yscale("log"); ax.set_ylim(1e-2, 3e2)
    ax.set_xticks(range(nf), [CFG_SH[f] for f, _ in fams])
    ax.set_xlim(-.6, nf - .4)
    ax.set_ylabel(r"$\epsilon_U$  [%]")
    ax.set_title("every converged seed, all five $Re_\\tau$", loc="left", fontsize=8)
    _tidy(ax)
    q = os.path.join(FIGURES, "fig4_multimodal.png"); fig.savefig(q, dpi=200)
    fig.savefig(q[:-4] + ".pdf"); plt.close(fig)
    plt.rcParams.update({"font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9.5,
                         "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8})
    return q


# ------------------------------------------------------------------ figure 4
def fig_bound():
    """Compare nonlinear error with a reference-state linear estimate.

    Restricting the display to accurate roots is an empirical selection, not a proof
    that the Taylor remainder is controlled. Panel B uses estimate / realized.
    """
    R = _load("s21_ratio.npy", "s21_ratio.py")
    S = s20()
    use = {(x["family"], x["retau"], x["seed"]) for r in CASES for f, _ in CONFIGS
           for x in S.get((f, r), [])
           if x["any_ok"] and np.isfinite(x["best_err"]) and x["best_err"] < PHYS}
    key = lambda o: (o["family"], o["retau"], o["seed"])
    ok = [o for o in R if np.isfinite(o["realized"]) and key(o) in use]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(TW, 2.75))
    lo = min(min(o["bound"], o["realized"]) for o in ok) * .4
    hi = max(max(o["bound"], o["realized"]) for o in ok) * 2.5
    for fam, lab in CONFIGS:
        rs = [o for o in ok if o["family"] == fam]
        if not rs:
            continue
        ax.loglog([o["bound"] for o in rs], [o["realized"] for o in rs], "o",
                  color=CFG_C[fam], ms=5.5, mew=0, ls="none", zorder=3, label=lab)
        ax2.scatter([o["retau"] for o in rs], [o["bound"] / o["realized"] for o in rs], s=20,
                    facecolor=CFG_C[fam], edgecolor="none", alpha=.8, zorder=3)
    ax.plot([lo, hi], [lo, hi], "-", color=INK, lw=1.2, zorder=1)
    ax.annotate("equal values", (.60, .66), xycoords="axes fraction", rotation=41,
                fontsize=8, color=INK)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel(r"linear estimate  $1/\sigma_{\min}(J)$")
    ax.set_ylabel(r"realized  $\|\hat e\|/\|\hat r\|$")
    ax.set_title("A.  Realized against local estimate", loc="left")
    ax.legend(loc="upper left", bbox_to_anchor=(0.015, 0.97), fontsize=8,
              borderaxespad=0)
    _tidy(ax)

    ax2.set_xscale("log"); ax2.set_yscale("log")
    ax2.set_xticks(CASES, [str(c) for c in CASES]); ax2.set_xlabel(r"$Re_\tau$")
    ax2.set_ylabel(r"$(1/\sigma_{\min})\,/\,(\|\hat e\|/\|\hat r\|)$")
    ax2.set_title("B.  Estimate over realized", loc="left")
    ax2.axhline(1, color=INK, lw=1.1)
    ax2.annotate("equal values", xy=(CASES[0], 1.0), xytext=(2, 5),
                 textcoords="offset points", fontsize=8, color=INK)
    _tidy(ax2)
    fig.tight_layout()
    q = os.path.join(FIGURES, "fig3_error_bound.png"); fig.savefig(q, dpi=200)
    fig.savefig(q[:-4] + ".pdf"); plt.close(fig)
    return q


# ------------------------------------------------------------------ figure 5
def fig_secant(retau=180):
    """The secant is constrained; the tangent is what the operator contains.

    C0 writes nu_t = nu_scale * softplus(.), so nu_t >= 0 holds for any parameters -- panel A
    is non-negative at every face, every seed and every Reynolds number.  The Jacobian does
    not contain nu_t.  It contains

        nu_eff = -d(tau)/dS = nu_t + S d(nu_t)/dS ,

    and the second term is free.  Panel B is the same networks at the same state, and it
    goes negative.

    The panels cover the FULL channel, y/h in [0, 2], and they must.  The violation is
    entirely in the upper half: nu_t is even in S because softplus is applied to a network
    output, but nu_eff carries an explicit factor S, so the closure's asymmetry in S -- which
    nothing penalises -- lands wholly on the tangent.  Plotting the lower half alone, as one
    naturally would for a symmetric channel, shows nothing wrong.

    Panel C is the scope of the claim: one to five seeds of ten depending on Reynolds number,
    but reaching -0.35 nu where it happens.  A guarantee that holds for eight seeds of ten is
    not a guarantee.
    """
    D = _load("s22_nueff.npy", "s22_profiles.py")
    fig, (ax, ax2, ax3) = plt.subplots(1, 3, figsize=(13.2, 4.0))
    d0 = [o for o in D if o["retau"] == retau]
    nu = d0[0]["nu"]
    for o in d0:
        ax.plot(o["yf"], o["nu_t"] / nu, "-", color="#2a78d6", lw=1.1, alpha=.7)
        ax2.plot(o["yf"], o["nu_eff"] / nu, "-", color="#b3411a", lw=1.1, alpha=.7)
    for a, t, k in ((ax, r"A.  $\nu_t$, the secant — constrained", "nu_t"),
                    (ax2, r"B.  $\nu_{\mathrm{eff}}=\nu_t+S\,\partial_S\nu_t$,"
                          r" the tangent — free", "nu_eff")):
        a.axhline(0, color=INK, lw=1.1)
        a.set_yscale("symlog", linthresh=1e-3)
        a.set_ylim(-1.0, 3.0)
        a.set_xlabel("$y/h$"); a.set_xlim(0, 2)
        a.set_title(t, loc="left", fontsize=9.5); _tidy(a)
        v = min(o[k].min() for o in d0) / nu
        a.annotate(r"$\min = %.2g\,\nu$" % v, (.03, .06), xycoords="axes fraction",
                   fontsize=8, color=INK if v >= 0 else "#b3411a")
    ax.set_ylabel(r"$/\,\nu$"); ax2.set_ylabel(r"$/\,\nu$")
    ax2.annotate("every violation is in the\nupper half of the channel", (.34, .12),
                 xycoords="axes fraction", fontsize=7.2, color="#b3411a")

    # Panel C as a count with the worst magnitude annotated.  Plotting min(nu_eff) itself
    # is unreadable: the violations span -1e-1 to -3e-43, and forty decades of a quantity
    # that is zero to within round-off is not information.
    xs = np.arange(len(CASES))
    cnt, worst = [], []
    for r in CASES:
        v = np.array([o["nu_eff"].min() / o["nu"] for o in D if o["retau"] == r])
        cnt.append(int((v < 0).sum()))
        worst.append(v.min())
    ax3.bar(xs, cnt, width=.55, color="#b3411a", alpha=.85, lw=0)
    for x, c, w in zip(xs, cnt, worst):
        ax3.annotate(r"$%.1g\,\nu$" % w, (x, c + .25), ha="center", fontsize=8,
                     color="#b3411a")
    ax3.set_xticks(xs, [str(c) for c in CASES]); ax3.set_xlabel(r"$Re_\tau$")
    ax3.set_ylim(0, 10); ax3.set_yticks(range(0, 11, 2))
    ax3.set_ylabel(r"seeds of ten with $\nu_{\mathrm{eff}}<0$ somewhere")
    ax3.set_title("C.  How often, and how badly", loc="left", fontsize=9.5)
    ax3.annotate("annotation: the worst $\\min_y\\nu_{\\mathrm{eff}}$ over the ten seeds",
                 (.03, .94), xycoords="axes fraction", fontsize=7, color=INK2)
    _tidy(ax3)
    fig.suptitle(r"C0 constrains $\nu_t\geq 0$; the operator sees $\nu_{\mathrm{eff}}$ — "
                 rf"ten seeds, panels A--B at $Re_\tau$={retau}", x=.01, ha="left",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, .94))
    q = os.path.join(FIGURES, "fig3_secant_tangent.png")
    fig.savefig(q, dpi=200); plt.close(fig)
    return q


if __name__ == "__main__":
    for f in (fig_apriori, fig_indices, fig_bound, fig_multimodal):
        print("wrote", f())
