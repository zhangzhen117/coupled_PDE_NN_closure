# Release validation

Validated on 16 September 2026 with the versions in `requirements.txt` (Python 3.11.14, Linux, one BLAS/OpenMP thread).

- 37/37 checks passed: 13 operator, 6 closure, 12 state-dependent Jacobian, 6 Krylov.
- Current main figures, supplementary error-bound figure, and five LaTeX tables regenerated from the distributed artifacts.
- All 106 saved root representatives passed independent residual and dense-Jacobian Newton-correction checks.
- Regenerated 10,000-profile pool matched the archived SHA-256 hash.
- All 50 distributed ensemble checkpoints matched the source experiment files byte for byte.
- One-iteration smoke test of the portable additional-reference penalty refinement completed and correctly reported that the gradient convergence criterion was not met. No reported checkpoint was changed.

This verifies artifact consistency and numerical implementation, not global uniqueness or convergence of fresh training runs.
