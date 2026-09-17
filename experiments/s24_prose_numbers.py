"""Every number quoted in the prose of sec. 5-6 that is not read from a table, recomputed
from the caches so the text can be checked against them after any change to the reported
set.  Run after make_tables; prints only."""
import os
import sys

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from chan.cases import CASES, CACHE, REPORTED, SHORT

PHYS = 0.05
FAMS = [f for f in REPORTED if f != "frozen"]
L = lambda p: list(np.load(os.path.join(CACHE, p), allow_pickle=True))

s20 = [o for f in FAMS for o in L(f"s20_{f}.npy")]
s21 = [o for o in L("s21_ratio.npy") if o["family"] in FAMS]
conv = [o for o in s20 if o["any_ok"] and np.isfinite(o["best_err"])]
phys = [o for o in conv if o["best_err"] < PHYS]
e = np.array([o["best_err"] for o in conv])
print(f"cases {len(s20)}   converged {len(conv)}   physical {len(phys)}   spurious {len(conv)-len(phys)}")
print(f"  below {100*PHYS:g}%: {np.sum(e<PHYS)}   above 30%: {np.sum(e>0.30)}   between: {np.sum((e>=PHYS)&(e<=0.30))}")
lo = np.sort(e[e < PHYS]); hi = np.sort(e[e >= PHYS])
print(f"  largest physical {100*lo[-1]:.2f}%   next above threshold {100*hi[0]:.2f}%")
print(f"  spurious errors range {100*hi.min():.0f}-{100*hi.max():.0f}%")
for f in FAMS:
    c = [o for o in conv if o["family"] == f]
    p3 = sum(o["best_err"] < PHYS for o in c); p30 = sum(o["best_err"] < 0.30 for o in c)
    om = sum(o["omega"] < 0 for o in s20 if o["family"] == f)
    pe = [100 * o["best_err"] for o in c if o["best_err"] < PHYS]
    print(f"  {SHORT[f]:>3} converged {len(c):>2}/50  physical {p3:>2}  (at 30%: {p30:>2})  spurious {len(c)-p3:>2}   omega<0 {om:>2}/50"
          f"   eps_U phys {np.mean(pe) if pe else float('nan'):.2f} +- {np.std(pe) if pe else float('nan'):.2f}%"
          f"   |omega| where neg: median {np.median([abs(o['omega']) for o in s20 if o['family']==f and o['omega']<0]) if om else float('nan'):.1e}"
          f"   median omega {np.median([o['omega'] for o in s20 if o['family']==f]):.1e}")
# rank correlations over all cases
isphys = np.array([1.0 if (o["any_ok"] and o["best_err"] < PHYS) else 0.0 for o in s20])
isconv = np.array([1.0 if o["any_ok"] else 0.0 for o in s20])
om = np.array([o["omega"] for o in s20])
sf = {}
for o in s21:
    sf[(o["family"], o["retau"], o["seed"])] = o["sigma_frozen"] / o["sigma_min"]
amp = np.array([np.log10(sf[(o["family"], o["retau"], o["seed"])]) for o in s20])
for name, x in (("omega", om), ("log10 amp", amp)):
    r1, p1 = spearmanr(x, isphys); r2, p2 = spearmanr(x, isconv)
    print(f"  spearman {name}: vs physical rho={r1:+.2f} (p={p1:.0e})   vs converged rho={r2:+.2f} (p={p2:.0e})")
# a priori orderings: from table values -> skip; ordering by physical roots
order = sorted(FAMS, key=lambda f: -sum(1 for o in phys if o["family"] == f))
print("  ordering by physical roots:", ", ".join(SHORT[f] for f in order))
# the bound (s21)
physkey = {(o["family"], o["retau"], o["seed"]) for o in phys}
ph = [o for o in s21 if np.isfinite(o["realized"])
      and (o["family"], o["retau"], o["seed"]) in physkey]
held = sum(o["realized"] <= o["bound"] for o in ph)
loose = np.array([o["bound"] / o["realized"] for o in ph])
print(f"bound: held on {held}/{len(ph)} physical solutions; looseness bound/realized {loose.mean():.0f} +- {loose.std():.0f}, range {loose.min():.0f} to {loose.max():.1e}")
for f in FAMS:
    a = [100 * o["apriori"] for o in s21 if o["family"] == f]
    byre = [np.median([100 * o["apriori"] for o in s21 if o["family"] == f and o["retau"] == r]) for r in CASES]
    print(f"  {SHORT[f]:>3} ||rhat||/||uref|| by Re (median): " + " ".join(f"{v:.2f}" for v in byre) + "%")
# amp per family at 180 and 5200
for f in FAMS:
    for r in (180, 5200):
        v = [sf[(f, r, s)] for s in range(10) if (f, r, s) in sf]
        print(f"  {SHORT[f]:>3} amp at {r}: {np.mean(v):.3g} +- {np.std(v):.2g}   sigma_min improvement x{1/np.mean(v):.0f}")
# preconditioner invariance (sec. 5.4): worst |eps_U(P1) - eps_U(P2)| over converged pairs
import glob, itertools
recs = [o for f in sorted(glob.glob(os.path.join(CACHE, "s12_*.npy")))
        + sorted(glob.glob(os.path.join(CACHE, "s12b_*.npy")))
        for o in np.load(f, allow_pickle=True) if o["family"] in FAMS]
w = 0.0
for o in recs:
    P = o["precond"]; ok = [p for p in P if P[p]["nk_ok"]]
    for a, b in itertools.combinations(ok, 2):
        w = max(w, abs(P[a]["err_U"] - P[b]["err_U"]))
print(f"preconditioner invariance: worst |eps_U(P1)-eps_U(P2)| = {w:.2e} over {len(recs)} s12 records")
