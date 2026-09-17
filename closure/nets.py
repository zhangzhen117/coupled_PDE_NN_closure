#!/usr/bin/env python3
"""The four closure arms (research plan Sec 6.3).

Same architecture, same data, same features; the arms differ ONLY in how the network
output becomes an eddy viscosity:

  A0  tau frozen at a trained profile, never updated       -> B = 0   (Wu et al. explicit)
  A1  nu_t = g(s, d)                     unconstrained     -> sym(B) indefinite
  A2  A1 + spectral-norm penalty on g, swept              -> ||B|| controlled
  A3  nu_t = softplus(g(s, d))           nu_t >= 0        -> "dissipative" by intent

A3 is deliberately NOT monotone-by-construction.  nu_t >= 0 does NOT imply dtau/dS <= 0:
with tau = -nu_t(s(S), d) S the actual effective viscosity is

    nu_t_eff = -dtau/dS = nu_t + S dnu_t/dS

so a positive-eddy-viscosity network can still produce backscatter in the operator sense.
Measuring where that happens is the point of the arm (see closure/jacobian.py).
"""
import torch
import torch.nn as nn


class MLP(nn.Module):
    """Trunk plus `nout` linear heads.

    Multi-head rather than multi-network, because a two-branch closure built from two
    separate MLPs has TWICE the parameters of a one-branch closure and the form-vs-form
    comparison is then meaningless.  Sharing the trunk costs only `width + 1` parameters
    per extra head: at width 48 one head is 2593 and two heads are 2642, a 1.9% difference,
    against 5186 for two separate networks.
    """

    def __init__(self, nin=2, width=32, depth=2, seed=0, nout=1):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        layers, n = [], nin
        for _ in range(depth):
            lin = nn.Linear(n, width)
            with torch.no_grad():
                lin.weight.copy_(torch.randn(width, n, generator=g) * (2.0 / n) ** 0.5)
                lin.bias.zero_()
            layers += [lin, nn.Tanh()]
            n = width
        out = nn.Linear(n, nout)
        with torch.no_grad():
            out.weight.copy_(torch.randn(nout, n, generator=g) * 0.1)
            out.bias.zero_()
        layers.append(out)
        self.net = nn.Sequential(*layers)
        self.nout = int(nout)

    def forward(self, x):
        y = self.net(x)
        return y.squeeze(-1) if self.nout == 1 else y


class Closure(nn.Module):
    """nu_t(s, d) with the arm's output constraint, and tau = -nu_t * dU/dy.

    nu_scale sets the magnitude of nu_t so the raw network output is O(1); the DNS eddy
    viscosity peaks near 0.08 u_tau h in the channel.
    """

    def __init__(self, arm="A1", width=32, depth=2, seed=0, nu_scale=0.1):
        super().__init__()
        if arm not in ("A1", "A2", "A3"):
            raise ValueError(f"arm {arm} is not a trainable arm (A0 freezes A1's tau)")
        self.arm = arm
        self.nu_scale = float(nu_scale)
        self.g = MLP(2, width, depth, seed)

    def nu_t(self, feat):
        raw = self.g(feat)
        if self.arm == "A3":
            return self.nu_scale * torch.nn.functional.softplus(raw)
        return self.nu_scale * raw                       # A1, A2: unconstrained

    def tau(self, dUdy, d, nu):
        from .features import features_torch
        return -self.nu_t(features_torch(dUdy, d, nu)) * dUdy

    def spectral_penalty(self):
        """Sum of squared spectral norms of the linear layers -- the A2 knob.
        Penalises the Lipschitz constant of g, hence ||B||."""
        p = 0.0
        for m in self.g.net:
            if isinstance(m, torch.nn.Linear):
                p = p + torch.linalg.matrix_norm(m.weight, ord=2) ** 2
        return p


class ClosureUV(nn.Module):
    """tau = N(U, dU/dy; Re_tau) -- a direct-tau head with TWO field inputs.

    Unlike `Closure` above there is no eddy-viscosity parametrisation and no output
    constraint: the network maps (U, dU/dy, Re) straight to tau_xy.  Consequences:

      * d(tau)/dU is nonzero, so B gains a first-order term and A+B is NOT symmetric.
      * Oddness of tau in dU/dy is NOT enforced -- it has to be learned from seeing both
        halves of the channel.  `antisymmetry_report` measures how well that worked; a
        poor number is a result, not something to paper over.
      * tau does not vanish at the walls by construction; it is pinned there by the
        no-slip convention, together with its Jacobian rows (see closure/jacobian.py).
    """

    def __init__(self, width=32, depth=2, seed=0, tau_scale=1.0):
        super().__init__()
        self.arm = "NU"
        self.tau_scale = float(tau_scale)
        self.g = MLP(3, width, depth, seed)

    def tau_uv(self, U, dUdy, nu, retau):
        from .features import features_uv_torch
        return self.tau_scale * self.g(features_uv_torch(U, dUdy, nu, retau))

    def spectral_penalty(self):
        p = 0.0
        for m in self.g.net:
            if isinstance(m, torch.nn.Linear):
                p = p + torch.linalg.matrix_norm(m.weight, ord=2) ** 2
        return p


class ClosureDiv(nn.Module):
    """Predicts d(tau)/dy DIRECTLY -- the quantity the momentum equation consumes.

        d(tau)/dy = 1 + Re_tau * G_SCALE * net(U, dU/dy; Re_tau)

    Why this output form:

      * The raw target spans [-345, +1] at Re_tau = 5200 and [-10, +1] at 180, but
        (d(tau)/dy - 1) / Re_tau collapses onto [-0.067, 0] at EVERY Reynolds number.
        Factoring out Re_tau turns a two-scale regression into a bounded one.
      * The additive 1 is a CONSTANT, not nu*d2U/dy2.  The identity
        d(tau)/dy = 1 + nu U'' is the momentum equation rearranged, so a network given
        U'' would drive the residual to zero for EVERY U and make the system singular.
        The features here are (U, dU/dy) only, so h stays a genuine closure and the
        momentum equation reads   nu U'' = Re_tau * h(U, dU/dy)  -- a proper second-order
        nonlinear ODE.
    """

    G_SCALE = 0.1

    def __init__(self, width=48, depth=2, seed=0):
        super().__init__()
        self.arm = "DIV"
        self.g = MLP(3, width, depth, seed)

    def h(self, U, dUdy, nu, retau):
        from .features import features_div_torch
        return self.G_SCALE * self.g(features_div_torch(U, dUdy, nu, retau))

    def dtau_dy(self, U, dUdy, nu, retau):
        rt = retau if torch.is_tensor(retau) else torch.full_like(U, float(retau))
        return 1.0 + rt * self.h(U, dUdy, nu, retau)

    def spectral_penalty(self):
        p = 0.0
        for m in self.g.net:
            if isinstance(m, torch.nn.Linear):
                p = p + torch.linalg.matrix_norm(m.weight, ord=2) ** 2
        return p


class ClosureDivLip(ClosureDiv):
    """Divergence-form closure with a HARD Lipschitz bound.

    Every linear layer is spectrally normalised to ||W||_2 = 1 and tanh is 1-Lipschitz,
    so the raw network is 1-Lipschitz in its inputs; the output is then scaled by `lip`.
    Hence, exactly and by construction,

        |d h / d x| <= G_SCALE * lip          =>   |G_U| <= Re_tau * G_SCALE * lip / U_SCALE

    `G_U = d(dtau/dy)/dU` is the reaction coefficient identified in the Stage-9 diagnosis
    as the thing that pushes Re(lambda) positive, so `lip` is a direct handle on it -- a
    hard bound, not a penalty that may or may not bind.
    """

    def __init__(self, width=48, depth=2, seed=0, lip=1.0):
        super().__init__(width, depth, seed)
        self.lip = float(lip)
        from torch.nn.utils.parametrizations import spectral_norm
        for i, m in enumerate(self.g.net):
            if isinstance(m, nn.Linear):
                self.g.net[i] = spectral_norm(m)

    def h(self, U, dUdy, nu, retau):
        from .features import features_div_torch
        return self.G_SCALE * self.lip * self.g(features_div_torch(U, dUdy, nu, retau))


class ClosureHybrid(nn.Module):
    """Split closure:  d(tau)/dy  =  [ Boussinesq nu_t part ]  +  [ learned residual ].

        tau_B = -nu_t(U, S; Re) * dU/dy ,   nu_t = nu_scale * softplus(...) >= 0
        g_res = res_scale * net_b(U, S; Re)          (a cell-centred divergence)

    The point of the split is structural, not cosmetic.  The nu_t part enters the operator
    as D(nu_t), a SYMMETRIC NEGATIVE-DEFINITE diffusion whenever nu_t >= 0 -- it can only
    stabilise.  Every destabilising contribution is therefore confined to the residual
    branch, whose reaction term d(g_res)/dU is the only thing that can push Re(lambda)
    positive.  If the eddy-viscosity branch carries most of the physics, the residual is
    small and so is its reaction term.

    Note this does NOT make the operator unconditionally stable: nu_t depends on U, so
    the nu_t branch also contributes a d(nu_t)/dU term (the nu_t_eff - nu_t gap of
    Stages 3-7).  The claim is only that the dangerous part is isolated and small.
    """

    def __init__(self, width=48, depth=2, seed=0, nu_scale=0.1, res_scale=1.0):
        super().__init__()
        self.arm = "HYB"
        self.nu_scale = float(nu_scale)
        self.res_scale = float(res_scale)
        # one trunk, two heads: 2642 parameters against 5186 for two networks, so this
        # is capacity-matched to the single-branch forms and the comparison is fair.
        self.t = MLP(3, width, depth, seed, nout=2)

    def _feat(self, U, dUdy, nu, retau):
        from .features import features_div_torch
        return features_div_torch(U, dUdy, nu, retau)

    def _heads(self, U, dUdy, nu, retau):
        y = self.t(self._feat(U, dUdy, nu, retau))
        return y[..., 0], y[..., 1]

    def nu_t(self, U, dUdy, nu, retau):
        return self.nu_scale * torch.nn.functional.softplus(
            self._heads(U, dUdy, nu, retau)[0])

    def tau_uv(self, U, dUdy, nu, retau):
        """Boussinesq stress at the faces -- the `tau_uv` interface that
        closure.jacobian.tau_jacobian_faces already knows how to differentiate."""
        return -self.nu_t(U, dUdy, nu, retau) * dUdy

    def g_res(self, U, dUdy, nu, retau):
        """Residual divergence at the cell centres."""
        return self.res_scale * self._heads(U, dUdy, nu, retau)[1]

    def g_total(self, U, dUdy, y, nu, retau):
        """Total d(tau)/dy on ONE case's grid:  d/dy(-nu_t dU/dy) + g_res.

        Takes `y` explicitly and differentiates with torch.gradient on that grid, so it
        must be called per case -- differencing across a pooled concatenation of several
        Reynolds numbers would straddle the boundary and be wrong.

        The deployed operator uses the FV form in closure/jacobian.py, not this; the two
        agree to discretisation order, and every operator statement (Jacobian, spectrum)
        comes from the FV path, which is finite-difference verified.
        """
        tau_B = -self.nu_t(U, dUdy, nu, retau) * dUdy
        return torch.gradient(tau_B, spacing=(y,))[0] + self.g_res(U, dUdy, nu, retau)


class ClosureNuEff(nn.Module):
    """Stage 13b -- parameterise the EFFECTIVE viscosity, not the eddy viscosity.

    Stage 9 established the identity that governs the operator's stability:

        nu_t_eff := -d(tau)/dS  =  nu_t + S * d(nu_t)/dS          (S = dU/dy)

    and Stage 12 showed why it matters: ClosureHybrid enforces nu_t >= 0 by softplus and
    STILL drifts to max Re(lambda) > 0, because the diffusion coefficient that actually
    reaches the Jacobian is nu_t_eff, and the S*d(nu_t)/dS term is unconstrained.  Bolting
    a non-negativity constraint onto nu_t constrains the wrong object.

    So constrain the right one.  Model nu_eff >= 0 directly and recover the stress by
    integrating it, rather than modelling nu_t and multiplying:

        tau(U, S; Re) = - integral_0^S nu_eff(U, s; Re) ds
        =>  d(tau)/dS = -nu_eff(U, S; Re) <= 0                    by construction

    The S-branch of B is then a diffusion with a non-negative coefficient -- symmetric
    negative semi-definite, unable to destabilise -- for ANY parameter values, at ANY
    state, trained or untrained.  No penalty, no tuning constant.

    tau(U, 0) = 0 falls out of the construction and is physically right: uv vanishes at
    the centreline where S = 0, and the wall faces are pinned separately.

    The U-branch d(tau)/dU is deliberately left free.  It is the reaction term, it is a
    genuinely different mechanism, and it is what the off-manifold penalty (Stage 13c)
    addresses.  This class fixes the diffusion branch only, and claims nothing more.

    Caveat, stated because it is a real one.  The integral is a fixed Gauss-Legendre
    quadrature, so AD returns the derivative OF THE QUADRATURE, which equals -nu_eff(U,S)
    only up to quadrature error rather than identically.  For a smooth integrand that
    error is geometric in the node count and saturates early: 4e-9 at 4 nodes, 3e-13 at 6,
    and 4e-16 -- machine precision -- from 8 on, which is why 8 is the default.  The
    guarantee is numerical rather than algebraic, but it is tighter than every other error
    in the problem by twelve orders of magnitude.
    """

    def __init__(self, width=48, depth=2, seed=0, nu_scale=0.1, nq=8):
        super().__init__()
        self.arm = "NUEFF"
        self.nu_scale = float(nu_scale)
        self.a = MLP(3, width, depth, seed)
        import numpy as _np
        x, w = _np.polynomial.legendre.leggauss(int(nq))   # nodes on [-1, 1]
        self.register_buffer("qt", torch.tensor((x + 1.0) / 2.0, dtype=torch.float64))
        self.register_buffer("qw", torch.tensor(w / 2.0, dtype=torch.float64))

    def _feat(self, U, dUdy, nu, retau):
        from .features import features_div_torch
        return features_div_torch(U, dUdy, nu, retau)

    def nu_eff(self, U, dUdy, nu, retau):
        """The effective viscosity -nu_t_eff = d(tau)/dS.  Non-negative by softplus."""
        return self.nu_scale * torch.nn.functional.softplus(
            self.a(self._feat(U, dUdy, nu, retau)))

    def tau_uv(self, U, dUdy, nu, retau):
        """tau = -S * sum_k w_k nu_eff(U, S t_k) -- the `tau_uv` interface that
        closure.jacobian.tau_jacobian_faces already knows how to differentiate."""
        t = self.qt.to(dUdy.dtype)
        w = self.qw.to(dUdy.dtype)
        Sq = dUdy.unsqueeze(-1) * t                       # (..., nq) nodes on [0, S]
        Uq = U.unsqueeze(-1).expand_as(Sq)
        rq = retau if not torch.is_tensor(retau) else retau.unsqueeze(-1).expand_as(Sq)
        return -dUdy * (self.nu_eff(Uq, Sq, nu, rq) * w).sum(-1)

    def g_total(self, U, dUdy, y, nu, retau):
        """d(tau)/dy on ONE case's grid -- for the a priori loss only.  Must be called
        per case: differencing a pooled multi-Re concatenation straddles the boundary."""
        return torch.gradient(self.tau_uv(U, dUdy, nu, retau), spacing=(y,))[0]


class ClosureTau(nn.Module):
    """FORM C -- predict the stress, then differentiate it.

        tau(U, S; Re) = tau_scale * N(U, S; Re)          S = dU/dy
        d(tau)/dy     = the FV difference of tau across the cell

    This is the textbook arrangement: model the Reynolds stress, let the discretisation
    supply the divergence.  It completes the set of three ways to place the network
    relative to the derivative:

        A  ClosureDiv     model d(tau)/dy directly           (no diffusion term at all)
        B  ClosureHybrid  model nu_t, plus a residual force
        C  ClosureTau     model tau, differentiate it        (this class)

    C is deliberately UNCONSTRAINED, which is the point of having it: ClosureNuEff is C
    with d(tau)/dS forced non-positive by construction, so the pair isolates what that
    single structural constraint is worth, at matched form, features, width and seeds.

    Because tau is a flux, B = -E T is a divergence: the closure cannot manufacture a
    pointwise reaction the way form A can.  Its row sum is -grad(N_U) rather than
    -dV*G_U, so the existence condition constrains a GRADIENT here, not a value -- a
    structurally weaker requirement.  Whether that makes C easier to constrain in
    practice is exactly what the penalised variant is for.

    Features are `features_div_torch`, the same set A and B use, so the comparison is
    about where the network sits relative to d/dy and nothing else.
    """

    def __init__(self, width=48, depth=2, seed=0, tau_scale=1.0):
        super().__init__()
        self.arm = "TAU"
        self.tau_scale = float(tau_scale)
        self.g = MLP(3, width, depth, seed)

    def _feat(self, U, dUdy, nu, retau):
        from .features import features_div_torch
        return features_div_torch(U, dUdy, nu, retau)

    def tau_uv(self, U, dUdy, nu, retau):
        """The `tau_uv` interface closure.jacobian.tau_jacobian_faces differentiates."""
        return self.tau_scale * self.g(self._feat(U, dUdy, nu, retau))

    def nu_eff(self, U, dUdy, nu, retau):
        """-d(tau)/dS, for reporting only.  Nothing constrains its sign here -- that is
        the difference from ClosureNuEff, and the reason to measure it."""
        S = dUdy.detach().requires_grad_(True)
        t = self.tau_uv(U, S, nu, retau)
        d, = torch.autograd.grad(t.sum(), S, create_graph=torch.is_grad_enabled())
        return -d

    def g_total(self, U, dUdy, y, nu, retau):
        """d(tau)/dy on ONE case's grid -- a priori loss only.  Must be called per case."""
        return torch.gradient(self.tau_uv(U, dUdy, nu, retau), spacing=(y,))[0]


class ClosureNuEffS(ClosureNuEff):
    """FORM C++ -- the full M-matrix certificate, by construction, with no penalty.

        tau(S; Re) = - integral_0^S nu_eff(s; Re) ds ,   nu_eff = 0.1*softplus(net) >= 0

    Identical to ClosureNuEff except the U channel of the feature vector is ZEROED, so tau
    has no U dependence at all and the gradient path from tau to U is severed.

    Why that is the whole point.  For any FLUX closure the Jacobian block is a divergence,
    B = -E T, and its row sum is exactly

        rowsum(B)_i = -( N_U,i+1 - N_U,i ) ,      N_U = d(tau)/dU

    verified to machine precision.  Nothing else contributes.  With no U input N_U == 0, so

        rowsum(B) == 0     identically -- every state, every Re, trained or untrained.

    Combine that with nu_eff >= 0, which adds positively to BOTH off-diagonals, and A's
    negative diagonal, and all four conditions for -J to be an irreducibly diagonally
    dominant M-matrix hold BY CONSTRUCTION:

        (a) J_ii < 0                (b) J_offdiag >= 0
        (c) rowsum(J) <= 0          (d) irreducible (tridiagonal, nonzero off-diagonals)

    Hence -J is nonsingular, (-J)^-1 >= 0 entrywise, and every eigenvalue has Re < 0 --
    at every lambda on the homotopy.  A fold requires det J_lambda = 0, so there can be
    none: the branch from the frozen problem must reach lambda = 1.  This is the only
    closure here that earns the certificate structurally rather than by penalty, and the
    prediction (lambda* >= 1 at every Re_tau, no folds) is directly falsifiable.

    The cost is a real modelling restriction, not a free lunch: this is a generalised-
    Newtonian closure tau = tau(dU/dy; Re).  It gives up the U dependence that carries
    wall-distance and inhomogeneity information, and the trained ClosureNuEff does use
    that channel -- |N_U| reaches 0.56.  Whether the accuracy loss is acceptable is the
    experiment.

    Parameter count is unchanged at 2593 so the comparison against ClosureNuEff is exactly
    matched in architecture; the 48 first-layer weights on the dead U channel simply carry
    no gradient.
    """

    def __init__(self, width=48, depth=2, seed=0, nu_scale=0.1, nq=8):
        super().__init__(width, depth, seed, nu_scale, nq)
        self.arm = "NUEFFS"

    def _feat(self, U, dUdy, nu, retau):
        f = super()._feat(U, dUdy, nu, retau)
        # zeros_like carries no gradient, so d(tau)/dU is identically zero -- not small
        return torch.cat([torch.zeros_like(f[..., :1]), f[..., 1:]], dim=-1)


def _matched_half_width(width):
    """Width `w2` such that TWO MLP(3, w2, 2) carry about as many parameters as ONE
    MLP(3, width, 2).  A two-branch closure is then capacity-matched to a one-branch one,
    which is the only way a form-vs-form comparison means anything.  48 -> 33 (2576 vs
    2593 parameters, 0.7% fewer)."""
    target = (width ** 2 + 6 * width + 1) / 2.0
    w = int(round((-6.0 + (36.0 + 4.0 * (target - 1.0)) ** 0.5) / 2.0))
    return max(w, 4)


class ClosureDivMono(ClosureDiv):
    """FORM A4 -- the force form with a NON-NEGATIVE REACTION BY CONSTRUCTION.

    Form A contributes a reaction and an advection and, structurally, no diffusion at all:

        B = -diag(dV) [ diag(G_U) + diag(G_S) D ]

    so it can never ADD dissipation, only avoid subtracting it, and its whole defence is
    the sign of the reaction G_U.  The existence condition is rowsum(J) <= 0, and because
    the S-stencil annihilates constants this is exactly G_U >= 0 (verified to 1e-14 on
    interior rows).  A1 enforces that with a penalty, which holds only where it is trained:
    measured, A1's violation is 1.1e-8 at U_ref and three orders larger one perturbation
    away.

    Here it is enforced by construction instead, using the same device that worked for
    nu_eff but integrated along U rather than S:

        h(U,S;Re) = h0(S;Re) + integral_0^{U/U_SCALE} m(u,S;Re) du ,   m = softplus(.) >= 0
        =>  dh/dU = m / U_SCALE >= 0    =>   G_U = Re_tau * G_SCALE * dh/dU >= 0

    for ANY parameters, at ANY state, trained or untrained.  h0 has the U channel zeroed
    so it cannot smuggle a U dependence past the constraint; the gradient path is severed,
    not merely small.

    What this does NOT fix: the advection coefficient b = dg/dS enters the coercivity
    condition as a - b'/2, so the energy route wants G_U >= b'/2, strictly stronger than
    the row-sum condition whenever b' > 0.  There is no cheap architectural route to that
    and it remains a penalty.  This class closes the existence gap, not the coercivity gap.
    """

    def __init__(self, width=48, depth=2, seed=0, nq=8, m_scale=0.1):
        nn.Module.__init__(self)
        self.arm = "DIVMONO"
        self.m_scale = float(m_scale)
        # one trunk, two heads at FULL width.  The first version used two networks at
        # width 33 to keep the parameter count matched, which also halved the capacity;
        # a shared trunk gives full width at 2642 parameters.
        self.t = MLP(3, width, depth, seed, nout=2)
        import numpy as _np
        x, wq = _np.polynomial.legendre.leggauss(int(nq))
        self.register_buffer("qt", torch.tensor((x + 1.0) / 2.0, dtype=torch.float64))
        self.register_buffer("qw", torch.tensor(wq / 2.0, dtype=torch.float64))

    def _feat(self, U, dUdy, nu, retau):
        from .features import features_div_torch
        return features_div_torch(U, dUdy, nu, retau)

    def m(self, U, dUdy, nu, retau):
        """dh/du >= 0.  Non-negative by softplus, for any parameters at any state."""
        return self.m_scale * torch.nn.functional.softplus(
            self.t(self._feat(U, dUdy, nu, retau))[..., 1])

    def h(self, U, dUdy, nu, retau):
        from .features import U_SCALE
        f = self._feat(U, dUdy, nu, retau)
        # zeros_like severs the gradient path, so h0 has NO U dependence at all
        f0 = torch.cat([torch.zeros_like(f[..., :1]), f[..., 1:]], dim=-1)
        h0 = self.t(f0)[..., 0]
        t, wq = self.qt.to(U.dtype), self.qw.to(U.dtype)
        Uq = U.unsqueeze(-1) * t                      # nodes on [0, U]
        Sq = dUdy.unsqueeze(-1).expand_as(Uq)
        rq = retau if not torch.is_tensor(retau) else retau.unsqueeze(-1).expand_as(Uq)
        integ = (U / U_SCALE) * (self.m(Uq, Sq, nu, rq) * wq).sum(-1)
        return self.G_SCALE * (h0 + integ)


class ClosureHybrid2(nn.Module):
    """FORM B2 -- the hybrid, repaired on both branches.

    The classical nu_t + residual split is wrong twice over, and B2 fixes both.

    (i) The eddy-viscosity branch constrains the SECANT.  Enforcing nu_t >= 0 by softplus
        guarantees nothing about the operator, which sees the TANGENT

            nu_t_eff = -d(tau)/dS = nu_t + S d(nu_t)/dS ,

        and the second term is left free.  B2 models the tangent directly and integrates,
        so d(tau)/dS = -nu_eff <= 0 by construction (measured 4e-16).

    (ii) The residual branch is a form-A closure, so it inherits form A's reaction
        pathology in full.  B2 gives it the A4 monotone construction, so its reaction is
        non-negative by construction too.

    Together: tau_B = -integral_0^S nu_eff ds  and  g_res = integral_0^{U/U_SCALE} m du,
    with nu_eff >= 0 and m >= 0.  Both branches are architecturally controlled, which the
    classical split never was.

    Note the taxonomy collapses here: tau_B is just a parameterisation of tau(U,S), so the
    flux branch IS a form-C closure and the residual branch IS a form-A closure -- B = C + A.
    B2 is therefore C+ (+) A4, and needs no theory of its own.

    Two MLPs at the given width, matching ClosureHybrid exactly, so B2 vs B is a controlled
    comparison.
    """

    def __init__(self, width=48, depth=2, seed=0, nu_scale=0.1, res_scale=0.1, nq=8):
        super().__init__()
        self.arm = "HYB2"
        self.nu_scale = float(nu_scale)
        self.res_scale = float(res_scale)
        # nu_eff keeps a full-width net (matching ClosureHybrid's nu_t net); the residual
        # branch splits ClosureHybrid's second net's budget between an OFFSET and the
        # non-negative integrand.  The offset is not optional: without it
        # g_res = int_0^U m du with m >= 0 would force g_res >= 0 for all U >= 0, which is
        # a constraint we never intended and which the data cannot satisfy.
        # one trunk, three heads: nu_eff, the non-negative residual reaction, and the
        # residual offset.  2691 parameters against 5169 for three networks.
        self.t = MLP(3, width, depth, seed, nout=3)
        import numpy as _np
        x, wq = _np.polynomial.legendre.leggauss(int(nq))
        self.register_buffer("qt", torch.tensor((x + 1.0) / 2.0, dtype=torch.float64))
        self.register_buffer("qw", torch.tensor(wq / 2.0, dtype=torch.float64))

    def _feat(self, U, dUdy, nu, retau):
        from .features import features_div_torch
        return features_div_torch(U, dUdy, nu, retau)

    def nu_eff(self, U, dUdy, nu, retau):
        """The tangent -d(tau)/dS, non-negative by construction."""
        return self.nu_scale * torch.nn.functional.softplus(
            self.t(self._feat(U, dUdy, nu, retau))[..., 0])

    def tau_uv(self, U, dUdy, nu, retau):
        """tau = -integral_0^S nu_eff ds -- the `tau_uv` interface that
        closure.jacobian.tau_jacobian_faces already differentiates."""
        t, wq = self.qt.to(dUdy.dtype), self.qw.to(dUdy.dtype)
        Sq = dUdy.unsqueeze(-1) * t
        Uq = U.unsqueeze(-1).expand_as(Sq)
        rq = retau if not torch.is_tensor(retau) else retau.unsqueeze(-1).expand_as(Sq)
        return -dUdy * (self.nu_eff(Uq, Sq, nu, rq) * wq).sum(-1)

    def m(self, U, dUdy, nu, retau):
        """d(g_res)/du >= 0."""
        return self.res_scale * torch.nn.functional.softplus(
            self.t(self._feat(U, dUdy, nu, retau))[..., 1])

    def g_res(self, U, dUdy, nu, retau):
        """The residual force, with a non-negative reaction by construction -- the
        `g_res` interface that closure.jacobian._g_res_jacobian differentiates."""
        from .features import U_SCALE
        f = self._feat(U, dUdy, nu, retau)
        f0 = torch.cat([torch.zeros_like(f[..., :1]), f[..., 1:]], dim=-1)
        t, wq = self.qt.to(U.dtype), self.qw.to(U.dtype)
        Uq = U.unsqueeze(-1) * t
        Sq = dUdy.unsqueeze(-1).expand_as(Uq)
        rq = retau if not torch.is_tensor(retau) else retau.unsqueeze(-1).expand_as(Uq)
        integ = (U / U_SCALE) * (self.m(Uq, Sq, nu, rq) * wq).sum(-1)
        return self.res_scale * self.t(f0)[..., 2] + integ

    def g_total(self, U, dUdy, y, nu, retau):
        """d(tau)/dy on ONE case's grid -- a priori loss only, must be called per case."""
        tau = self.tau_uv(U, dUdy, nu, retau)
        return torch.gradient(tau, spacing=(y,))[0] + self.g_res(U, dUdy, nu, retau)
