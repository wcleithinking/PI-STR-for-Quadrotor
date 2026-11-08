"""Self-tuning regulator (RLS + pole placement) for one SISO channel.

Normative source: ``docs/CONTROL_SPEC.md`` §3 (discrete ARX model), §4 (RLS with
forgetting), §5 (pole placement / Diophantine / control law), §6 (numerical-safety
policy).  The constants of §6 are reproduced verbatim below; they are part of the
Python↔C++ golden-vector contract and must not be "tuned".

Design notes
------------
* The Diophantine system is solved with Gaussian elimination **with partial
  pivoting** (:func:`gaussian_solve`).  The legacy 3x3/5x5 adjugate expansions are
  forbidden (``CONTROL_SPEC.md`` §5) because that is where the historical
  ``s0 = x[2]`` indexing bug lived.
* All signals are SI: angle in rad, angular rate in rad/s, altitude in m,
  vertical speed in m/s.  The analytic initial guess is therefore built from
  ``tf_alpha`` converted to rad/s per PWM unit for the attitude channels (see
  :func:`analytic_theta0_mode_ii`).  The legacy firmware/MATLAB worked in degrees;
  the estimator/controller mathematics is identical, only the initial guess is
  scaled (``y_unit="deg"`` reproduces the original numbers).
* Update order per controller tick (identical to MATLAB ``hattheta(:,t+1)`` usage):
  regressor -> RLS -> projection -> gains -> control law -> saturation.  The
  controller then waits for the realised command (:meth:`STR.ack`), which is what
  the regressor uses on the next tick (§5, §6.4).

Order-3 parameter projection (normative for the C++ port)
---------------------------------------------------------
§6.1 only specifies the second-order model.  The third-order extension is:

1. Box clamps::

       a1 <- clamp(a1, -4, 4)
       a2 <- clamp(a2, -4, 4)          # not the 2nd-order [-2,2]: (z-1)^2 (z-p)
       a3 <- clamp(a3, -1, 1)          #   has a2 = 1 + 2p, up to 3
       b0, b1, b2 <- clamp(., -10, 10)

2. Relative b-sum lower bound (units matter: the Mode-I SI b coefficients are
   ~1e-4, so the absolute ``BSUM_MIN = 1e-3`` would systematically bias them)::

       bsum_min_3 = 0.01 * |b0(0) + b1(0) + b2(0)|      (theta(0) = analytic guess)
       if |b0 + b1 + b2| < bsum_min_3:  b0 <- bsum_min_3 - b1 - b2

   (``max(..., 1e-12)`` guards a zero analytic guess.  The Lead's shorthand
   ``|b0+b1|_initial`` denotes this initial b-sum for the 3rd-order model.)

3. Schur stability projection.  Let ``p(z) = 1 + a1 z^-1 + a2 z^-2 + a3 z^-3``
   and ``R(a) = max|roots(p)|``.  A marginally stable model (``R = 1``, the true
   Mode-I channel has a double integrator at ``z = 1``) is **accepted**; only a
   clearly unstable one is corrected::

       if R(a) > 1 + 1e-3:
           rho <- largest value in (0, 1) with R(rho*a) <= 1
           a   <- rho * a
           # rho found by 30 bisection steps on [0, 1]

Idempotency: after step 1 the box clamps are inactive; after step 2
``|sum b| >= bsum_min_3``; after step 3 ``R(a) <= 1 <= 1 + 1e-3`` (or the model was
already inside), so a second application changes nothing.  The b-sum fix is
applied after the b clamps (the §6.1 order), so ``b0`` can exceed ``B_MAX`` by at
most ``bsum_min_3``; that is intentional and must be reproduced bit-for-bit.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Sequence

import numpy as np

# ---------------------------------------------------------------------------
# Numerical-safety constants, docs/CONTROL_SPEC.md §6.  verbatim.
# ---------------------------------------------------------------------------
BSUM_MIN: float = 1e-3          # §6.1  (double / float firmware are both 1e-3)
#: Order-3 relative b-sum bound: ``0.01 * |sum b(theta0)|`` (see module docstring).
BSUM_MIN_REL: float = 0.01
GUARD_BSUM: float = 1e-6        # §6.2
GUARD_DET: float = 1e-9         # §6.2
GUARD_U: float = 1e6            # §6.2
FAIL_LIMIT: int = 50            # §6.3

#: Parameter clamps from §6.1.
A1_MIN, A1_MAX = -4.0, 4.0
A2_MIN, A2_MAX = -2.0, 2.0
#: Order-3 extension: the 2nd-order clamps are not wide enough for a 3rd-order
#: denominator (e.g. ``(z-1)^2 (z-p)`` has ``a2 = 1 + 2p`` up to 3), so a wider
#: box plus the Schur projection below is used.  §6.1 defines only the 2nd-order
#: projection; this is the documented extension.
A2_MIN_O3, A2_MAX_O3 = -4.0, 4.0
A3_MIN, A3_MAX = -1.0, 1.0
B_MIN, B_MAX = -10.0, 10.0

#: Only used by the order-3 stability projection: a model is considered unstable
#: when a pole leaves the unit disc by more than this tolerance.  A marginally
#: stable pole at ``z = 1`` (a pure integrator, which the true channel has) is
#: *accepted* so that identification is not biased.
STABILITY_TOL: float = 1e-3
_RAD2DEG = 180.0 / np.pi


# ---------------------------------------------------------------------------
# Reference model (§5)
# ---------------------------------------------------------------------------
def am_coeffs(order: int, zeta: float, wn: float, h: float, cc: float = 50.0) -> np.ndarray:
    """Denominator of the reference model ``Am(z)``, ``[am1, am2]`` or ``[1..3]``-indexed.

    * ``order == 2``: pole-matched discretisation of ``s^2+2ζωn s+ωn²`` (§5):
      ``am1 = -2 e^{-ζωn h} cos(ωn sqrt(1-ζ²) h)``, ``am2 = e^{-2ζωn h}``.
    * ``order == 3``: port of the Mode-I MATLAB script,
      ``Gdc(s) = wn² cc / ((s+cc)(s²+2ζwn s+wn²))`` discretised with a ZOH.
    """
    if order == 2:
        wd = wn * np.sqrt(max(1.0 - zeta * zeta, 0.0))
        am1 = -2.0 * np.exp(-zeta * wn * h) * np.cos(wd * h)
        am2 = np.exp(-2.0 * zeta * wn * h)
        return np.array([am1, am2])
    if order == 3:
        from scipy import signal

        num = np.array([wn * wn * cc], dtype=float)
        den = np.array(
            [1.0, 2.0 * zeta * wn + cc, wn * wn + 2.0 * zeta * wn * cc, cc * wn * wn],
            dtype=float,
        )
        _, d, _ = signal.cont2discrete((num, den), h, method="zoh")
        d = np.ravel(d)
        return np.array([d[1], d[2], d[3]], dtype=float)
    raise ValueError(f"order must be 2 or 3, got {order!r}")


def ao_coeffs(order: int, mu: float) -> np.ndarray:
    """Integral extra pole(s) ``Ao``: ``[ao1]`` for order 2, ``[ao1, ao2]`` for order 3.

    §5 defines ``Ao = 1 + mu z^-1`` for the 2nd-order case.  For the 3rd-order
    port (Mode I MATLAB) ``Ao = (1 - mu z^-1)² = 1 - 2 mu z^-1 + mu² z^-2``;
    with the paper's ``mu = 0`` both reduce to ``Ao = 1``.
    """
    if order == 2:
        return np.array([mu], dtype=float)
    if order == 3:
        return np.array([-2.0 * mu, mu * mu], dtype=float)
    raise ValueError(f"order must be 2 or 3, got {order!r}")


# ---------------------------------------------------------------------------
# Analytic initial guess (§3)
# ---------------------------------------------------------------------------
def analytic_theta0_mode_ii(
    tf_alpha: float | Sequence[float],
    tf_beta: float | Sequence[float],
    h: float,
    percent: float = 1.0,
) -> np.ndarray:
    """``theta0 = percent * [a1, a2, b0, b1]`` exactly as written in §3."""
    alpha = np.asarray(tf_alpha, dtype=float)
    beta = np.asarray(tf_beta, dtype=float)
    a2 = np.exp(-beta * h)
    a1 = -(1.0 + a2)
    b0 = alpha / beta * h - alpha / beta**2 * (1.0 - a2)
    b1 = (1.0 / beta * (1.0 - a2)) * (alpha / beta * (1.0 - a2)) - a2 * b0
    return percent * np.stack([a1, a2, b0, b1], axis=-1)


def analytic_theta0_mode_i(
    tf_alpha: float | Sequence[float],
    tf_beta: float | Sequence[float],
    h: float,
    percent: float = 1.0,
) -> np.ndarray:
    """3rd-order guess: ZOH discretisation of ``G(s) = tf_beta/(s^2 (s + tf_alpha))``.

    Port of the Mode-I MATLAB setup (``Gc = tf(tf_beta, [1, tf_alpha, 0, 0])``,
    ``Gd = c2d(Gc, h_sample)``).  ``theta = [a1, a2, a3, b0, b1, b2]``.
    """
    from scipy import signal

    alpha = np.atleast_1d(np.asarray(tf_alpha, dtype=float))
    beta = np.atleast_1d(np.asarray(tf_beta, dtype=float))
    if alpha.size == 1 and beta.size > 1:
        alpha = np.full_like(beta, alpha[0])
    if beta.size == 1 and alpha.size > 1:
        beta = np.full_like(alpha, beta[0])

    out = np.empty((alpha.size, 6), dtype=float)
    for i in range(alpha.size):
        num, den, _ = signal.cont2discrete(
            (np.array([beta[i]]), np.array([1.0, alpha[i], 0.0, 0.0])), h, method="zoh"
        )
        num, den = np.ravel(num), np.ravel(den)
        out[i] = [den[1], den[2], den[3], num[1], num[2], num[3]]
    return percent * (out[0] if out.shape[0] == 1 else out)


# ---------------------------------------------------------------------------
# Linear algebra: Gaussian elimination with partial pivoting (§5)
# ---------------------------------------------------------------------------
class NumericalSingularity(ArithmeticError):
    """Raised when the Diophantine matrix cannot be factorised at all."""


def gaussian_solve(A: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, float]:
    """Solve ``A x = b`` by Gaussian elimination with partial pivoting.

    Returns ``(x, det)``.  Raises :class:`NumericalSingularity` when a pivot is
    exactly zero or a non-finite value appears; the caller decides whether the
    determinant is small enough to freeze the gains (§6.2 uses ``|det| < 1e-9``).
    """
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float).reshape(-1)
    n = A.shape[0]
    if A.shape != (n, n) or b.size != n:
        raise ValueError("gaussian_solve expects a square A and a matching b")

    M = np.empty((n, n + 1), dtype=float)
    M[:, :n] = A
    M[:, n] = b
    sign = 1.0
    det = 1.0
    for k in range(n):
        col = np.abs(M[k:, k])
        p = int(np.argmax(col))
        pivot = M[k + p, k]
        if not np.isfinite(pivot) or pivot == 0.0:
            raise NumericalSingularity(f"zero/non-finite pivot in column {k}")
        if p:
            M[[k, k + p], :] = M[[k + p, k], :]
            sign = -sign
        pivot = M[k, k]
        det *= pivot
        # Eliminate below the pivot.  The loop is written explicitly (instead of
        # a vectorised rank-1 update) so that the floating-point operation order
        # is obvious and reproducible in the C++ golden-vector core.
        for i in range(k + 1, n):
            factor = M[i, k] / pivot
            if factor == 0.0:
                continue
            for j in range(k, n + 1):
                M[i, j] -= factor * M[k, j]
            M[i, k] = 0.0

    det *= sign
    if not np.isfinite(det):
        raise NumericalSingularity("non-finite determinant")

    x = np.empty(n, dtype=float)
    for i in range(n - 1, -1, -1):
        acc = M[i, n]
        for j in range(i + 1, n):
            acc -= M[i, j] * x[j]
        x[i] = acc / M[i, i]
    if not np.all(np.isfinite(x)):
        raise NumericalSingularity("non-finite solution")
    return x, det


# ---------------------------------------------------------------------------
# Diophantine system (§5)
# ---------------------------------------------------------------------------
def diophantine_system(
    theta: Sequence[float],
    order: int,
    am: Sequence[float],
    ao: Sequence[float],
) -> tuple[np.ndarray, np.ndarray]:
    """Build ``A x = b`` for ``A(z) R(z) + B(z) S(z) = Am(z) Ao(z)``.

    ``theta = [a1, a2, b0, b1]`` (order 2) or ``[a1, a2, a3, b0, b1, b2]``
    (order 3); unknown order is ``[r1, s0, s1]`` / ``[r1, r2, s0, s1, s2]``
    exactly as required by §5 (this is the mapping the legacy firmware got wrong).
    """
    t = np.asarray(theta, dtype=float)
    if order == 2:
        a1, a2, b0, b1 = t
        am1, am2 = am
        ao1 = ao[0]
        A = np.array(
            [
                [1.0, b0, 0.0],
                [a1, b1, b0],
                [a2, 0.0, b1],
            ]
        )
        rhs = np.array(
            [
                am1 + ao1 - a1,
                am2 + ao1 * am1 - a2,
                ao1 * am2,
            ]
        )
        return A, rhs

    if order == 3:
        a1, a2, a3, b0, b1, b2 = t
        am1, am2, am3 = am
        ao1, ao2 = ao
        A = np.array(
            [
                [1.0, 0.0, b0, 0.0, 0.0],
                [a1, 1.0, b1, b0, 0.0],
                [a2, a1, b2, b1, b0],
                [a3, a2, 0.0, b2, b1],
                [0.0, a3, 0.0, 0.0, b2],
            ]
        )
        rhs = np.array(
            [
                am1 + ao1 - a1,
                am2 + ao1 * am1 + ao2 - a2,
                am3 + ao1 * am2 + ao2 * am1 - a3,
                ao1 * am3 + ao2 * am2,
                ao2 * am3,
            ]
        )
        return A, rhs

    raise ValueError(f"order must be 2 or 3, got {order!r}")


# ---------------------------------------------------------------------------
# Parameter projection (§6.1)
# ---------------------------------------------------------------------------
def _schur_max_radius(a: np.ndarray) -> float:
    """Largest pole modulus of ``1 + a1 z^-1 + a2 z^-2 + a3 z^-3``."""
    return float(np.max(np.abs(np.roots(np.concatenate(([1.0], a)))))) if a.size else 0.0


def _shrink_into_unit_disc(a: np.ndarray) -> np.ndarray:
    """Scale ``a`` towards the origin until the polynomial is Schur.

    Only used for the 3rd-order model (the 2nd-order case has the closed-form
    triangle of §6.1).  Scaling is used instead of an ad-hoc projection so that
    the map stays idempotent: once stable, a second application is a no-op.
    """
    lo, hi = 0.0, 1.0
    for _ in range(30):
        mid = 0.5 * (lo + hi)
        if _schur_max_radius(mid * a) <= 1.0:
            lo = mid
        else:
            hi = mid
    return lo * a


def project_theta(theta: Sequence[float], order: int, bsum_min: float = BSUM_MIN) -> np.ndarray:
    """Project identified parameters into the admissible set of §6.1.

    The projection is idempotent and is applied after every RLS update and
    **before** the gains are recomputed (§6.1).

    The closure of the 2nd-order stability triangle is used (``a2 >= -1``,
    ``-1-a2 <= a1 <= 1+a2``).  This is deliberate: the true identified rate
    channel sits exactly on ``a1 + a2 = -1`` (its pole is the pure integrator of
    the motor), so an open-set projection with a margin would bias the estimate
    by construction.  The boundary is marginal, not unstable, and the projection
    is a fixed point on it.
    """
    t = np.asarray(theta, dtype=float).copy()
    if t.size not in (4, 6):
        raise ValueError(f"unexpected parameter vector length {t.size}")

    if order == 2:
        a1, a2, b0, b1 = t
        a1 = float(np.clip(a1, A1_MIN, A1_MAX))
        a2 = float(np.clip(a2, A2_MIN, A2_MAX))
        a2 = max(a2, -1.0)
        a1 = float(np.clip(a1, -1.0 - a2, 1.0 + a2))
        b0 = float(np.clip(b0, B_MIN, B_MAX))
        b1 = float(np.clip(b1, B_MIN, B_MAX))
        if abs(b0 + b1) < bsum_min:
            b0 = bsum_min - b1
        return np.array([a1, a2, b0, b1])

    a1, a2, a3 = t[:3]
    a1 = float(np.clip(a1, A1_MIN, A1_MAX))
    a2 = float(np.clip(a2, A2_MIN_O3, A2_MAX_O3))
    a3 = float(np.clip(a3, A3_MIN, A3_MAX))
    a = np.array([a1, a2, a3])
    if _schur_max_radius(a) > 1.0 + STABILITY_TOL:
        a = _shrink_into_unit_disc(a)
    b = np.clip(t[3:], B_MIN, B_MAX)
    if abs(b.sum()) < bsum_min:
        b[0] = bsum_min - b[1] - b[2]
    return np.concatenate([a, b])


# ---------------------------------------------------------------------------
# RLS (§4)
# ---------------------------------------------------------------------------
@dataclass
class RlsResult:
    theta: np.ndarray
    P: np.ndarray
    error: float
    alpha: float
    gain: np.ndarray
    symmetric: bool
    clamped: bool


def rls_update(
    theta: np.ndarray,
    P: np.ndarray,
    phi: np.ndarray,
    y: float,
    forgetting: float,
    p_max: float,
) -> RlsResult:
    """One RLS step, update order exactly as §4.

    ``P = (P - K (phi' P)) / lambda`` is the rank-1 form (no ``I - K phi'``),
    followed by symmetrisation and diagonal clamping to ``[0, p_max]``.
    """
    theta = np.asarray(theta, dtype=float)
    P = np.asarray(P, dtype=float)
    phi = np.asarray(phi, dtype=float)

    error = float(y - theta @ phi)
    Pphi = P @ phi
    alpha = float(forgetting + phi @ Pphi)
    if not np.isfinite(alpha) or alpha <= 0.0:
        raise NumericalSingularity("RLS alpha is not positive/finite")
    K = Pphi / alpha

    theta_new = theta + K * error
    P_new = (P - np.outer(K, phi @ P)) / forgetting
    P_new = 0.5 * (P_new + P_new.T)

    diag = np.diag(P_new).copy()
    diag_clamped = np.clip(diag, 0.0, p_max)
    clamped = bool(np.any(diag_clamped != diag))
    if clamped:
        # §4.3: rebuild from the clamped diagonal, keep the off-diagonals.
        idx = np.diag_indices_from(P_new)
        P_new[idx] = diag_clamped

    return RlsResult(
        theta=theta_new,
        P=P_new,
        error=error,
        alpha=alpha,
        gain=K,
        symmetric=bool(np.allclose(P_new, P_new.T, rtol=0.0, atol=0.0)),
        clamped=clamped,
    )


# ---------------------------------------------------------------------------
# Gains and status
# ---------------------------------------------------------------------------
@dataclass
class Gains:
    """Pole-placement gains for one channel (all fields are per-channel scalars)."""

    t0: float
    t1: float
    t2: float
    r1: float
    r2: float
    s0: float
    s1: float
    s2: float
    det: float

    @classmethod
    def zeros(cls) -> "Gains":
        return cls(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    def as_dict(self) -> dict[str, float]:
        return {
            "t0": self.t0,
            "t1": self.t1,
            "t2": self.t2,
            "r1": self.r1,
            "r2": self.r2,
            "s0": self.s0,
            "s1": self.s1,
            "s2": self.s2,
            "det": self.det,
        }


@dataclass
class GuardStatus:
    """Diagnostics for the current tick (reset at the start of every tick)."""

    frozen_gains: bool = False
    safe_mode: bool = False
    fail_streak: int = 0
    events: dict[str, int] = field(default_factory=dict)

    def fire(self, name: str) -> None:
        self.events[name] = self.events.get(name, 0) + 1


@dataclass(frozen=True)
class STRConfig:
    """Per-channel configuration (order, reference model, RLS, limits, guards)."""

    order: int = 2
    h: float = 0.04
    zeta: float = 0.7
    wn: float = 5.0
    mu: float = 0.0
    cc: float = 50.0               # only used by the 3rd-order reference model
    forgetting: float = 0.99
    p_max: float = 5.0
    init_percent: float = 0.75
    u_min: float = -80.0
    u_max: float = 80.0
    theta0: tuple[float, ...] = ()
    bsum_min: float = BSUM_MIN
    bsum_min_order3: float | None = None
    guard_bsum: float = GUARD_BSUM
    guard_det: float = GUARD_DET
    guard_u: float = GUARD_U
    fail_limit: int = FAIL_LIMIT

    def __post_init__(self) -> None:
        if self.order not in (2, 3):
            raise ValueError(f"order must be 2 or 3, got {self.order}")
        if not (0.0 < self.forgetting <= 1.0):
            raise ValueError("forgetting factor must be in (0, 1]")
        if self.p_max <= 0.0:
            raise ValueError("p_max must be positive")
        if self.u_max <= 0.0 or self.u_min >= 0.0:
            raise ValueError("u_min/u_max must straddle zero")
        n = 2 * self.order
        if self.theta0 and len(self.theta0) != n:
            raise ValueError(f"theta0 must have {n} entries for order {self.order}")
        if self.order == 3 and self.bsum_min_order3 is None:
            b_sum0 = abs(float(np.sum(np.asarray(self.theta0, dtype=float)[3:]))) if self.theta0 else 0.0
            object.__setattr__(self, "bsum_min_order3", max(BSUM_MIN_REL * b_sum0, 1e-12))

    @property
    def n_params(self) -> int:
        return 2 * self.order

    @property
    def bsum_threshold(self) -> float:
        """Projection threshold actually used (absolute for order 2, relative for 3)."""
        if self.order == 3:
            return float(self.bsum_min_order3 if self.bsum_min_order3 is not None else BSUM_MIN)
        return float(self.bsum_min)

    def with_(self, **kwargs) -> "STRConfig":
        return replace(self, **kwargs)

    # -- factories ---------------------------------------------------------
    @classmethod
    def from_control_params(
        cls,
        control,
        theta0: Sequence[float] | None = None,
        **overrides,
    ) -> "STRConfig":
        """Build from :class:`pistr_sim.params.ControlParams` (contract layer)."""
        u_max = float(np.max(np.abs(np.asarray(control.u_max, dtype=float))))
        base = dict(
            order=int(control.order),
            h=float(control.h),
            zeta=float(control.zeta),
            wn=float(control.wn),
            mu=float(control.mu),
            forgetting=float(control.forgetting),
            p_max=float(control.p_max),
            init_percent=float(control.init_percent),
            u_min=-u_max,
            u_max=u_max,
            bsum_min=float(control.bsum_min),
            guard_bsum=float(control.guard_bsum),
            guard_det=float(control.guard_det),
            guard_u=float(control.guard_u),
            fail_limit=int(control.fail_limit),
        )
        base.update(overrides)
        if theta0 is not None:
            base["theta0"] = tuple(float(v) for v in theta0)
        return cls(**base)

    @classmethod
    def mode_ii(
        cls,
        params,
        channel: int = 0,
        y_unit: str = "rad",
        percent: float | None = None,
        **overrides,
    ) -> "STRConfig":
        """Mode-II (2nd order) configuration with the analytic guess of §3.

        ``y_unit="deg"`` reproduces the legacy firmware/MATLAB scaling (``tf_alpha``
        is given in deg/s per PWM unit); the Python stack uses ``"rad"``.
        """
        control = params.control
        pct = control.init_percent if percent is None else percent
        alpha = float(params.tf_alpha[channel])
        if y_unit == "rad" and channel < 3:
            alpha /= _RAD2DEG
        elif y_unit not in ("rad", "deg"):
            raise ValueError("y_unit must be 'rad' or 'deg'")
        theta0 = analytic_theta0_mode_ii(alpha, float(params.tf_beta[channel]), control.h, pct)
        channel_max = float(np.asarray(control.u_max, dtype=float).reshape(-1)[channel])
        overrides.setdefault("u_max", channel_max)
        overrides.setdefault("u_min", -channel_max)
        return cls.from_control_params(control, theta0=theta0, order=2, **overrides)

    @classmethod
    def mode_i(
        cls,
        params,
        channel: int = 0,
        percent: float | None = None,
        **overrides,
    ) -> "STRConfig":
        """Mode-I (3rd order) configuration; the plant guess is the MATLAB ZOH port."""
        control = params.control
        pct = control.init_percent if percent is None else percent
        alpha = float(params.tf_alpha[channel])
        if channel < 3:
            alpha /= _RAD2DEG
        theta0 = analytic_theta0_mode_i(alpha, float(params.tf_beta[channel]), control.h, pct)
        channel_max = float(np.asarray(control.u_max, dtype=float).reshape(-1)[channel])
        # Mode-I reference model, ported from the MATLAB script: wn=7.9 rad/s,
        # zeta=0.707, cc=50 (the 2nd-order Mode-II values are intentionally not reused).
        overrides.setdefault("u_max", channel_max)
        overrides.setdefault("u_min", -channel_max)
        overrides.setdefault("wn", 7.9)
        overrides.setdefault("zeta", 0.707)
        overrides.setdefault("cc", 50.0)
        return cls.from_control_params(control, theta0=theta0, order=3, **overrides)


# ---------------------------------------------------------------------------
# The regulator
# ---------------------------------------------------------------------------
class STR:
    """One self-tuning regulator channel.

    Typical use::

        u = reg.step(y_des, y_meas)       # design command for tick k
        ...                               # mixer + PWM saturation
        reg.ack(u_actual)                 # realised command, fed back at k+1

    If :meth:`ack` is not called, the saturated design command is used as the
    "realised" value (identical to the legacy firmware when nothing saturates).
    """

    def __init__(self, config: STRConfig):
        self.cfg = config
        theta0 = np.asarray(config.theta0, dtype=float) if config.theta0 else np.zeros(config.n_params)
        if theta0.size != config.n_params:
            raise ValueError("theta0 size does not match order")
        self.am = am_coeffs(config.order, config.zeta, config.wn, config.h, config.cc)
        self.ao = ao_coeffs(config.order, config.mu)
        self._theta0 = theta0.copy()
        self.resetdata()

    # -- lifecycle ---------------------------------------------------------
    def resetdata(self) -> None:
        """Restore exactly ``theta0`` and ``P = p_max I``; clear all history.

        Mirrors the firmware ``STR::resetdata()`` contract (§4.5): no stale
        history survives, the fail counter and the safe-mode latch are cleared.
        """
        n = self.cfg.order
        self._theta = self._theta0.copy()
        self._P = np.eye(self.cfg.n_params) * self.cfg.p_max
        self._y: list[float] = [0.0] * n          # y(k-1), y(k-2), ...
        self._u: list[float] = [0.0] * n          # u_actual(k-1), u_actual(k-2), ...
        self._yd: list[float] = [0.0] * (n - 1)   # y_des(k-1), ...
        self._v: list[float] = [0.0] * (n - 1)    # v(k-1), ...
        self._theta_valid = self._theta.copy()
        self._P_valid = self._P.copy()
        self._gains = self._design_gains(self._theta)
        self._gains_valid = replace(self._gains)
        self._last_cmd = 0.0
        self._pending_ack = False
        self._fail_streak = 0
        self._safe_mode = False
        self._ticks = 0
        self.status = GuardStatus()

    @property
    def theta(self) -> np.ndarray:
        return self._theta.copy()

    @property
    def theta_hat(self) -> np.ndarray:
        return self._theta.copy()

    @property
    def P(self) -> np.ndarray:
        return self._P.copy()

    @property
    def gains(self) -> Gains:
        return replace(self._gains)

    @property
    def gains_valid(self) -> Gains:
        return replace(self._gains_valid)

    @property
    def safe_mode(self) -> bool:
        return self._safe_mode

    @property
    def frozen(self) -> bool:
        return self.status.frozen_gains

    @property
    def ticks(self) -> int:
        return self._ticks

    # -- design ------------------------------------------------------------
    def _design_gains(self, theta: np.ndarray) -> Gains:
        """Solve the Diophantine equation; on any guard, return zero gains.

        The caller (:meth:`step`) is responsible for replacing the result with
        the last valid gains when the model is singular (§6.2).
        """
        cfg = self.cfg
        t = np.asarray(theta, dtype=float)
        bsum = float(t[cfg.order:].sum())
        if not np.isfinite(bsum) or abs(bsum) < cfg.guard_bsum:
            return Gains.zeros()
        t0 = float((1.0 + float(self.am.sum())) / bsum)
        if cfg.order == 2:
            t1, t2 = t0 * float(self.ao[0]), 0.0
        else:
            t1, t2 = t0 * float(self.ao[0]), t0 * float(self.ao[1])
        A, rhs = diophantine_system(t, cfg.order, self.am, self.ao)
        try:
            x, det = gaussian_solve(A, rhs)
        except NumericalSingularity:
            return Gains.zeros()
        if not np.isfinite(det) or abs(det) < cfg.guard_det:
            return Gains.zeros()
        if cfg.order == 2:
            r1, s0, s1 = float(x[0]), float(x[1]), float(x[2])
            r2 = s2 = 0.0
        else:
            r1, r2, s0, s1, s2 = (float(x[i]) for i in range(5))
        g = Gains(t0, t1, t2, r1, r2, s0, s1, s2, float(det))
        vals = np.array([g.t0, g.t1, g.t2, g.r1, g.r2, g.s0, g.s1, g.s2, g.det])
        if not np.all(np.isfinite(vals)):
            return Gains.zeros()
        return g

    @staticmethod
    def _gains_are_valid(g: Gains) -> bool:
        return (
            g.det != 0.0
            and np.isfinite(g.det)
            and np.all(
                np.isfinite([g.t0, g.t1, g.t2, g.r1, g.r2, g.s0, g.s1, g.s2])
            )
        )

    # -- per-tick helpers ---------------------------------------------------
    def _commit_history(self, y_des: float, y_meas: float, u: float, v: float) -> None:
        n = self.cfg.order
        self._y.insert(0, float(y_meas))
        del self._y[n:]
        self._u.insert(0, float(u))
        del self._u[n:]
        self._yd.insert(0, float(y_des))
        del self._yd[n - 1:]
        self._v.insert(0, float(v))
        del self._v[n - 1:]
        self._pending_ack = True

    def _fail_tick(self, reason: str) -> float:
        """§6.3: restore the last valid snapshot, hold the command, count failures."""
        self.status.fire(reason)
        self._theta = self._theta_valid.copy()
        self._P = self._P_valid.copy()
        self._gains = replace(self._gains_valid)
        self._fail_streak += 1
        self.status.fail_streak = self._fail_streak
        if self._fail_streak >= self.cfg.fail_limit:
            self._safe_mode = True
            self.status.safe_mode = True
            self._last_cmd = 0.0
            return 0.0
        return float(self._last_cmd)

    def _control_law(self, y_des: float, y_meas: float, g: Gains) -> float:
        ao1 = float(self.ao[0])
        if self.cfg.order == 2:
            return (
                -ao1 * float(self._v[0])
                + g.t0 * y_des
                + g.t1 * float(self._yd[0])
                - g.s0 * y_meas
                - g.s1 * float(self._y[0])
                + (ao1 - g.r1) * float(self._u[0])
            )
        ao2 = float(self.ao[1])
        return (
            -ao1 * float(self._v[0])
            - ao2 * float(self._v[1])
            + g.t0 * y_des
            + g.t1 * float(self._yd[0])
            + g.t2 * float(self._yd[1])
            - g.s0 * y_meas
            - g.s1 * float(self._y[0])
            - g.s2 * float(self._y[1])
            + (ao1 - g.r1) * float(self._u[0])
            + (ao2 - g.r2) * float(self._u[1])
        )

    # -- main entry point ---------------------------------------------------
    def step(self, y_des: float, y_meas: float) -> float:
        """Run one controller tick and return the designed command ``u(k)``.

        The caller must saturate/mix, then call :meth:`ack` with the command that
        was actually realised so that the identifier sees the truth (§5/§6.4).
        """
        self._ticks += 1
        self.status = GuardStatus()
        cfg = self.cfg
        n = cfg.order

        if self._safe_mode:
            self.status.safe_mode = True
            self.status.frozen_gains = True
            self._last_cmd = 0.0
            return 0.0

        y_des = float(y_des)
        y_meas = float(y_meas)
        if not (np.isfinite(y_des) and np.isfinite(y_meas)):
            return self._fail_tick("nonfinite_input")

        # -- §4: RLS on the previous estimate, current sample -----------------
        phi = np.empty(2 * n, dtype=float)
        phi[:n] = -np.asarray(self._y, dtype=float)
        phi[n:] = np.asarray(self._u, dtype=float)
        try:
            rls = rls_update(
                self._theta, self._P, phi, y_meas, cfg.forgetting, cfg.p_max
            )
        except (NumericalSingularity, FloatingPointError):
            return self._fail_tick("rls_singular")
        if not (
            np.all(np.isfinite(rls.theta))
            and np.all(np.isfinite(rls.P))
            and np.isfinite(rls.error)
            and np.isfinite(rls.alpha)
        ):
            return self._fail_tick("nonfinite_rls")

        # -- §6.1: projection before the gains are recomputed -----------------
        theta_proj = project_theta(rls.theta, n, cfg.bsum_threshold)
        if not np.all(np.isfinite(theta_proj)):
            return self._fail_tick("nonfinite_projection")

        # -- §5/§6.2: Diophantine with singularity guards ---------------------
        gains = self._design_gains(theta_proj)
        if not self._gains_are_valid(gains):
            self.status.frozen_gains = True
            self.status.fire(
                "bsum" if abs(float(theta_proj[n:].sum())) < cfg.guard_bsum else "det"
            )
            gains = replace(self._gains_valid)
        else:
            self._gains_valid = replace(gains)
        self._gains = gains

        self._theta = theta_proj
        self._P = rls.P
        self.status.fail_streak = self._fail_streak

        # -- §5: control law --------------------------------------------------
        v = self._control_law(y_des, y_meas, gains)
        if not np.isfinite(v):
            return self._fail_tick("nonfinite_command")

        # -- §6.2: |v| > 1e6 is treated as singular -> freeze the gains -------
        if abs(v) > cfg.guard_u:
            self.status.fire("large_command")
            self.status.frozen_gains = True
            gains = replace(self._gains_valid)
            self._gains = gains
            v = self._control_law(y_des, y_meas, gains)
            if not np.isfinite(v):
                return self._fail_tick("nonfinite_command")

        # -- §6.4: saturate; ack() supplies the realised value later ----------
        u = float(np.clip(v, cfg.u_min, cfg.u_max))
        self._commit_history(y_des, y_meas, u, v)
        self._last_cmd = u
        self._fail_streak = 0
        self._theta_valid = self._theta.copy()
        self._P_valid = self._P.copy()
        self._gains_valid = replace(gains)
        return u

    def ack(self, u_actual: float) -> None:
        """Feed back the physically realised command (§5/§6.4).

        Called after the mixer/PWM saturation stage.  Ignored when the tick was
        rejected, so a stale acknowledgement cannot corrupt the history.
        """
        if not self._pending_ack:
            return
        u_actual = float(u_actual)
        if not np.isfinite(u_actual):
            return
        self._u[0] = u_actual
        self._pending_ack = False

    # convenience aliases matching the firmware vocabulary -----------------
    reset = resetdata
    reset_data = resetdata


class STRBank:
    """A bank of independent :class:`STR` channels (one RLS state per channel)."""

    def __init__(self, configs: Sequence[STRConfig]):
        if not configs:
            raise ValueError("at least one channel is required")
        self.channels = [STR(c) for c in configs]

    def __len__(self) -> int:
        return len(self.channels)

    def reset(self) -> None:
        for ch in self.channels:
            ch.resetdata()

    def step(self, y_des: Sequence[float], y_meas: Sequence[float]) -> np.ndarray:
        yd = np.asarray(y_des, dtype=float).reshape(-1)
        ym = np.asarray(y_meas, dtype=float).reshape(-1)
        if yd.size != len(self.channels) or ym.size != len(self.channels):
            raise ValueError("y_des/y_meas must match the number of channels")
        return np.array([ch.step(float(yd[i]), float(ym[i])) for i, ch in enumerate(self.channels)])

    def ack(self, u_actual: Sequence[float]) -> None:
        ua = np.asarray(u_actual, dtype=float).reshape(-1)
        if ua.size != len(self.channels):
            raise ValueError("u_actual must match the number of channels")
        for i, ch in enumerate(self.channels):
            ch.ack(float(ua[i]))

    @property
    def theta(self) -> np.ndarray:
        return np.stack([ch.theta for ch in self.channels]) if self.channels else np.empty((0, 0))

    def gains(self) -> list[Gains]:
        return [ch.gains for ch in self.channels]

    def safe_mode(self) -> bool:
        return any(ch.safe_mode for ch in self.channels)


__all__ = [
    "BSUM_MIN",
    "BSUM_MIN_REL",
    "GUARD_BSUM",
    "GUARD_DET",
    "GUARD_U",
    "FAIL_LIMIT",
    "A1_MIN",
    "A1_MAX",
    "A2_MIN",
    "A2_MAX",
    "A2_MIN_O3",
    "A2_MAX_O3",
    "A3_MIN",
    "A3_MAX",
    "B_MIN",
    "B_MAX",
    "STABILITY_TOL",
    "NumericalSingularity",
    "Gains",
    "GuardStatus",
    "STRConfig",
    "STR",
    "STRBank",
    "RlsResult",
    "am_coeffs",
    "ao_coeffs",
    "analytic_theta0_mode_ii",
    "analytic_theta0_mode_i",
    "diophantine_system",
    "gaussian_solve",
    "project_theta",
    "rls_update",
]
