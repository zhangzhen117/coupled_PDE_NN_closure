# Numerical conventions

The channel spans y/h in [0,2], with homogeneous no-slip wall values and cell-centred velocity unknowns. Pressure forcing is positive. The code's stress is tau_code = <u'v'>, opposite to the paper's Reynolds shear stress. The strong residual is 1 + nu U'' - d(tau_code)/dy.

`A` is the integrated finite-volume molecular-diffusion matrix: it is symmetric negative definite with Dirichlet walls. `D = diag(dV)` contains cell volumes. Residual and Jacobian arrays are integrated quantities unless explicitly divided by dV. Weighted velocity norms use sum(dV * U**2); comparisons must retain this weighting on stretched meshes.

The full coupled Jacobian includes derivatives through the state-dependent closure, including velocity and shear dependence. Holding a closure fixed during a linear solve changes the iteration/preconditioner, not the nonlinear root equation. Secant viscosity and tangent viscosity are different: positivity of the former does not ensure positivity of the latter. Full tangent dissipation includes reaction as well as diffusion. Numerical checks at DNS/reference or converged states are local measurements, not global monotonicity certificates.

The archived checkpoint and multistart artifact hashes identify the reported numerical states. Optimizer metadata and nonlinear-solver success are separate diagnostics. The tests also retain simpler analytical closures to check signs, assembly and derivatives.
