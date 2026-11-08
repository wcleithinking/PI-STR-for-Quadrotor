"""Wind field: steady offset plus Dryden turbulence.

Normative text: ``docs/PLANT_MODEL.md`` §6.

The gust is three independent first-order shaping filters driven by white
noise, as required by the specification::

    H_u(s) = sigma_u * sqrt(2 L_u / (pi Vbar)) / (1 + (L_u / Vbar) s)

Normalisation (important, and the reason the numbers below look unusual)
-----------------------------------------------------------------------
Two conventions have to be reconciled:

1. the *turbulence* one, in which the one-sided Dryden spectrum

       Phi(omega) = K^2 / (1 + (omega/omega_c)^2),   K = sigma sqrt(2L/(pi Vbar)),
       omega_c = Vbar / L,   integral over [0, inf) = sigma^2

   has peak ``K^2`` (this is ``|H(0)|^2`` of the specification's ``H_u``); and
2. the *discrete simulation* one, in which the one-pole recursion

       y[k] = a y[k-1] + b n[k],   a = exp(-Vbar dt / L)

   driven by discrete white noise of variance ``q = 1/dt`` (i.e. continuous
   white noise of unit intensity) has variance ``q b^2/(1-a^2)``.

Matching the *time-domain* variance to ``sigma^2`` (what P11 checks, and what
"sigma_u = 1.5 m/s" means to a reader) fixes

    b = sigma sqrt(2 L / Vbar) (1 - a) = K sqrt(pi) (1 - a).

The ``sqrt(pi)`` is exactly the one-sided-PSD/Fourier convention factor, so
``b`` is *not* ``K(1-a)``.  Using ``K(1-a)`` — the naive continuous-limit
reading — makes the gusts a factor ``sqrt(pi)`` too weak (0.56 sigma), which is
easy to miss in a time-domain plot and impossible to miss in P11.

With ``sigma = 1.5 m/s``, ``L = 200 m``, ``Vbar = 5 m/s`` the corner is at
``omega_c = 0.025 rad/s`` (period 250 s), which is why the validation test (P11)
checks the *filter* frequency response rather than a Welch estimate of a short
time series.

Determinism
-----------
The filters are evaluated in blocks with ``scipy.signal.lfilter`` (as specified)
but served sample by sample, so a 120 s / 1 kHz run does not pay the per-sample
``lfilter`` call overhead.  The RNG is seeded from ``WindParams.seed`` (or an
explicit override) and the filter state is carried across blocks, so the
generated series is identical to a single ``lfilter`` call over the same noise.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import lfilter

from .params import WindParams

__all__ = ["DrydenTurbulence", "WindModel"]

#: Block length used when materialising the gust series.  Purely a performance
#: knob: the filter state is carried across blocks, so results do not depend on it.
_BLOCK = 4096


class DrydenTurbulence:
    """Three-axis Dryden gust generator (one first-order filter per axis)."""

    def __init__(self, params: WindParams, dt: float) -> None:
        if dt <= 0.0:
            raise ValueError("dt must be positive")
        self.params = params
        self.dt = float(dt)

        sigma = np.asarray(params.sigma, dtype=float)
        length = np.asarray(params.length_scale, dtype=float)
        v_ref = float(params.v_ref)
        if np.any(length <= 0.0) or v_ref <= 0.0:
            raise ValueError("Dryden length scales and v_ref must be positive")

        # one-pole coefficients: a = exp(-Vbar*dt/L), b = sigma*sqrt(2L/Vbar)*(1-a)
        self.omega_c = v_ref / length                       # rad/s, corner
        self.a = np.exp(-self.omega_c * self.dt)            # (3,)
        # Peak of the *one-sided* Dryden PSD (the spec's |H(0)|):
        #   Phi(omega) = K_psd^2 / (1 + (omega/omega_c)^2),  int_0^inf Phi = sigma^2
        self.K_psd = sigma * np.sqrt(2.0 * length / (np.pi * v_ref))
        # Filter gain that makes the *time-domain* standard deviation equal sigma.
        # The extra sqrt(pi) against K_psd is the one-sided-PSD vs Fourier
        # convention factor; see the module docstring.
        self.b = sigma * np.sqrt(2.0 * length / v_ref) * (1.0 - self.a)
        # driving noise std so that the discrete filter matches the continuous PSD
        self.noise_std = 1.0 / np.sqrt(self.dt)

        self.reset()

    @property
    def variance(self) -> np.ndarray:
        """Analytic steady-state variance of each gust channel (== ``sigma**2``)."""
        q = self.noise_std**2
        return q * self.b**2 / (1.0 - self.a**2)

    # -- public ----------------------------------------------------------
    def reset(self, seed: int | None = None) -> None:
        """Restart the gust series from a (deterministic) seed."""
        self.seed = self.params.seed if seed is None else int(seed)
        self._rng = np.random.default_rng(self.seed)
        self._zi = np.zeros((3, 1))
        self._buf = np.zeros((0, 3))
        self.n_generated = 0

    def value_at(self, index: int) -> np.ndarray:
        """Gust ``[u, v, w]`` (m/s, inertial frame) at discrete step ``index``."""
        index = int(index)
        if index < 0:
            raise ValueError("index must be non-negative")
        if index >= self._buf.shape[0]:
            self._extend(index + 1)
        return self._buf[index].copy()

    def series(self, n: int) -> np.ndarray:
        """First ``n`` gust samples, shape ``(n, 3)``."""
        n = int(n)
        if n > self._buf.shape[0]:
            self._extend(n)
        return self._buf[:n].copy()

    def psd(self, omega: np.ndarray) -> np.ndarray:
        """Analytic one-sided PSD of the *continuous* process, ``Phi(omega)``."""
        omega = np.asarray(omega, dtype=float)
        return np.stack(
            [self.K[i] ** 2 / (1.0 + (omega / self.omega_c[i]) ** 2) for i in range(3)],
            axis=-1,
        )

    # -- internals -------------------------------------------------------
    def _extend(self, need: int) -> None:
        while self._buf.shape[0] < need:
            noise = self._rng.normal(0.0, self.noise_std, size=(_BLOCK, 3))
            out = np.empty_like(noise)
            for i in range(3):
                out[:, i], self._zi[i] = lfilter(
                    [self.b[i]], [1.0, -self.a[i]], noise[:, i], zi=self._zi[i]
                )
            self._buf = np.concatenate([self._buf, out])
            self.n_generated += _BLOCK


class WindModel:
    """``v_wind(xi, t) = v_steady + v_gust(t)`` in the inertial frame."""

    def __init__(self, params: WindParams | None = None, dt: float = 1e-3) -> None:
        self.params = params if params is not None else WindParams()
        self.dt = float(dt)
        self.turbulence = DrydenTurbulence(self.params, self.dt)
        self._steady = np.zeros(3)
        self._turbulence_on = bool(self.params.enabled)

    # -- configuration ---------------------------------------------------
    def reset(self, seed: int | None = None) -> None:
        self._steady = np.zeros(3)
        self._turbulence_on = bool(self.params.enabled)
        self.turbulence.reset(seed)

    def set_steady(self, steady: np.ndarray) -> None:
        steady = np.asarray(steady, dtype=float).reshape(3)
        if not np.all(np.isfinite(steady)):
            raise ValueError("steady wind must be finite")
        self._steady = steady

    def set_turbulence(self, enabled: bool) -> None:
        self._turbulence_on = bool(enabled)

    # -- evaluation ------------------------------------------------------
    @property
    def steady(self) -> np.ndarray:
        return self._steady.copy()

    def at(self, t: float) -> np.ndarray:
        """Wind vector at time ``t`` (inertial frame, m/s)."""
        v = self._steady.copy()
        if self._turbulence_on:
            index = int(round(t / self.dt))
            v += self.turbulence.value_at(index)
        return v
