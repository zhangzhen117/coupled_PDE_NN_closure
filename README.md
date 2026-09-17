# Coupled PDE–neural closure

Research code accompanying **Well-posedness of neural turbulence closures and tangent dissipation**, by Zhen Zhang and George Em Karniadakis.

This repository implements the paper's one-dimensional plane-channel experiments: learned closures coupled to a finite-volume momentum equation, full closure Jacobians, conditioning diagnostics, Newton–Krylov solves, and searches for multiple solutions. It is not a general three-dimensional RANS solver.

## Models and files

| Paper | Checkpoint family | Closure |
|---|---|---|
| A0 | `force` | Unconstrained force |
| A1 | `force_pen2w` | Force with tangent-reaction penalty |
| B0 | `tau` | Learned stress |
| B1 | `nueff` | Tangent diffusion by construction |
| C0 | `hybrid` | Non-negative secant eddy viscosity |

- `chan/`: stretched meshes, finite-volume operators, weighted norms and conditioning.
- `closure/`: networks, automatic-differentiation Jacobians and training.
- `experiments/`: ensemble solves, error measurements and figure/table generators.
- `experiments/multistart/`: common initial-profile pool, bounded Newton searches, clustering and independent root checks.
- `data/models/`: 50 reported checkpoints (ten per closure).
- `data/cache/`: frozen measurements used for the paper's figures and tables.
- `data/multistart/`: five frozen search checkpoints, 50,000 trial records and accepted roots.
- `tests/`: operator, closure, Jacobian and Krylov checks.

Additional model classes remain for compatibility and numerical tests; only the five families above correspond to the reported comparison. `ARTIFACTS.sha256` records checksums of the distributed scientific artifacts.

## Setup

Tested with Python 3.11.14 on Linux. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python data/fetch_lm.py
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
```

The downloader obtains the mean and fluctuation profiles from the official [Lee–Moser channel-flow database](https://turbulence.oden.utexas.edu/channel2015/data/). DNS files are not redistributed here. Please cite Lee & Moser (2015), *Direct numerical simulation of turbulent channel flow up to Re_tau approximately 5200*, JFM 774, 395–415, when using these data. Five Reynolds numbers are used: 180, 550, 1000, 2000 and 5200; the code reads the actual Reynolds number from each DNS file.

## Reproduce figures and tables

```bash
python experiments/reproduce_paper.py
```

This uses the distributed checkpoints and cached measurements, writes figures to `figures/` and LaTeX tables to `paper/tex/tables/`, and does not retrain networks. Figure 3 uses the recorded multistart roots; the error-bound plot is saved as `figS1_error_bound`. The manuscript itself is not included.

Run the numerical checks with:

```bash
python tests/test_operator.py
python tests/test_closure.py
python tests/test_uv_jacobian.py
python tests/test_krylov.py
```

## Training and fresh measurements

For example:

```bash
python closure/train.py --family force --seed 0
python closure/train.py --family force_pen2w --seed 0 --warm-from force
python experiments/refine_penalty.py --checkpoint data/models/force_pen2w_s0.pt --output results/penalty.pt --seed 0
```

The penalty refinement adds the zero and intermediate reference profiles `(U,S) = alpha (U_DNS,S_DNS)` with alpha = 0, 1/4, 1/2, 3/4 to the original DNS reference. It uses weights 30 for each of the original-reference and averaged additional-reference penalties. Refinement writes a separate checkpoint and reports the actual gradient criterion. Frozen checkpoints preserve the states used for the reported experiments; do not infer optimizer convergence from the existence of a checkpoint. Retraining need not reproduce floating-point results or optimizer stopping points exactly.

The base trainer writes to `data/models/` and may skip existing completed runs. Use a separate working copy for new ensembles. Inspect `--help` before launching expensive training or overwriting cached measurements. The measurement entry points are `s20_summary.py` (convergence/accuracy), `s21_ratio.py` (inverse sensitivity), `s22_profiles.py` (profiles), and `s12_newton_krylov.py` (preconditioning). Their `--families`, `--seeds` and case options select the experiment.

## Search for multiple roots

At Re_tau = 1000, one frozen model per closure was tested against the **same 10,000 initial profiles**. The pool contains zero and laminar profiles plus randomized 16-mode sine series, with or without a parabolic base. Screening is limited to 20 Newton steps and 15 seconds per profile; candidate refinement has a 30-second cap. See `protocol.json` and the source for residual, correction and clustering tolerances.

| Closure | Distinct accepted roots | Starts reaching accepted roots |
|---|---:|---:|
| A0 | 7 | 910 |
| A1 | 1 | 5617 |
| B0 | 93 | 230 |
| B1 | 1 | 4024 |
| C0 | 4 | 2993 |

To regenerate the initial pool and check all saved roots:

```bash
python experiments/multistart/run.py prepare
python experiments/multistart/validate_plot.py
```

For a new search, use a fresh output directory (Linux/POSIX is required for the wall-clock alarms):

```bash
export MULTISTART_OUTPUT=results/multistart
python experiments/multistart/run.py prepare
# Run task IDs 0 through 49, sequentially or as independent jobs:
python experiments/multistart/run.py 0
# After all 50 tasks finish:
python experiments/multistart/aggregate.py
python experiments/multistart/validate_plot.py
python experiments/multistart/plot_velocity_stress.py
```

Each task handles 1,000 starts. Existing trial records are protected against accidental duplicate appends. A single task does not reproduce the full search.

## Interpretation

A single observed root provides empirical evidence, not a proof of uniqueness; failed starts do not prove nonexistence. The theorem's global full-tangent condition is stronger than sampled penalties. B1's exact-integral construction controls tangent diffusion only: reaction still needs control, and the implemented eight-point quadrature is not an all-state certificate for the exact integral. Negative closure tangent diffusion alone does not imply loss of ellipticity of the total operator.

The code uses the stress sign `<u'v'>`; the paper uses its negative. See `CONVENTIONS.md`. Figure 3 reconstructs stress as `1-y-nu*S`; for asymmetric roots this is a fixed-integration-constant diagnostic, not necessarily the network's stress. Blue marks the selected physical profiles; the secondary near-DNS C0 root remains brown as in the paper.
