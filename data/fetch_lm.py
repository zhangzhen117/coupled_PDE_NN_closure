#!/usr/bin/env python3
"""Download the Lee & Moser (2015) plane-channel mean profiles.

Source: https://turbulence.oden.utexas.edu/channel2015/
Cite:   M. K. Lee and R. D. Moser, JFM 774:395-415 (2015).

Two files per Reynolds number:
  *_mean_prof.dat      y/delta, y+, U+, dU+/dy+, W+, P+     <- tau_true comes from here
  *_vel_fluc_prof.dat  y/delta, y+, uu, vv, ww, uv, ...     <- reference only, see below

The mean file is the one the study uses. The fluctuation file is fetched so that the gap
between the analytic tau_true and the database <u'v'> can be reported as an "irreducible"
residual scale (research plan Sec 4.2: that ~0.5% gap IS the quantity being measured, so
it must never be used as the reference stress).
"""
import os
import sys
import urllib.request

BASE = "https://turbulence.oden.utexas.edu/channel2015/data"
RAW = os.path.join(os.path.dirname(os.path.abspath(__file__)), "raw")

# nominal Re_tau -> filename token
CASES = {180: "0180", 550: "0550", 1000: "1000", 2000: "2000", 5200: "5200"}
KINDS = ("mean_prof", "vel_fluc_prof")


def fetch(force=False):
    os.makedirs(RAW, exist_ok=True)
    for retau, tok in CASES.items():
        for kind in KINDS:
            name = f"LM_Channel_{tok}_{kind}.dat"
            dst = os.path.join(RAW, name)
            if os.path.exists(dst) and not force:
                print(f"  have {name} ({os.path.getsize(dst)} B)")
                continue
            url = f"{BASE}/{name}"
            print(f"  get  {name} ... ", end="", flush=True)
            with urllib.request.urlopen(url, timeout=60) as r:
                blob = r.read()
            if not blob.lstrip().startswith(b"%"):
                raise RuntimeError(f"{url} did not return a %-commented data file")
            with open(dst, "wb") as f:
                f.write(blob)
            print(f"{len(blob)} B")


if __name__ == "__main__":
    fetch(force="--force" in sys.argv)
    print(f"\nraw data in {RAW}")
