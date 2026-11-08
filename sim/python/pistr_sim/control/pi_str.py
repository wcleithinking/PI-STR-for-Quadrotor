"""PI-STR cascade (Mode II) and single-loop STR (Mode I).

``docs/CONTROL_SPEC.md`` §7:

* **Mode II** — outer PI on the angle error produces a rate command; the inner
  STR (2nd-order ARX) tracks that rate command.  The firmware and the paper use
  different outer gains on purpose (§7, §9).
* **Mode I** — no outer loop: the 3rd-order STR is fed the angle reference
  directly and absorbs the motor lag in its model.

All controllers expose the same four-channel interface used by the simulation
bench: ``update(y_ref, y_meas, rate_ref, rate_meas) -> u_design`` followed by
``ack(u_actual)`` once the realised command is known.  Channel order is
``[roll, pitch, yaw, altitude]`` and the outer variable is an angle for the first
three channels and altitude for the fourth.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Sequence

import numpy as np

from .pid import CascadePIDBank, RobustPIDBank, robust_pid_from_params
from .str import STRBank, STRConfig

N_CHANNELS = 4


# ---------------------------------------------------------------------------
# Outer loop
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class OuterPIConfig:
    """Outer PI on the angle error (§7).

    ``form="firmware"``  — ``rate_des = Kp e + Ki ∫e + Kd (rate_des_prev - rate_meas)``
    (``Controller_STR.ino``; ``Kp=1.5, Ki=0.01, Kd=0``).

    ``form="paper"``     — ``rate_des = rate_ff + Kp e + Ki ∫e`` with the reference
    rate feed-forward of the MATLAB trajectory study (``Kp=1.0, Ki=0.02``); the
    ESO-derived ``rate_ff`` is supplied by the caller.
    """

    kp: float = 1.5
    ki: float = 0.01
    kd: float = 0.0
    h: float = 0.04
    form: str = "firmware"
    rate_min: float = -6.0
    rate_max: float = 6.0

    def __post_init__(self) -> None:
        if self.form not in ("firmware", "paper"):
            raise ValueError("form must be 'firmware' or 'paper'")
        if self.rate_min >= self.rate_max:
            raise ValueError("rate_min must be smaller than rate_max")


class OuterPI:
    """One outer-loop channel; output is a rate (or vertical-speed) command."""

    def __init__(self, config: OuterPIConfig):
        self.cfg = config
        self.reset()

    def reset(self) -> None:
        self._integ = 0.0
        self._rate_des = 0.0

    @property
    def integrator(self) -> float:
        return self._integ

    def _int_limit(self) -> float:
        # Keep the integral contribution inside the output range (anti-windup).
        if self.cfg.ki <= 0.0:
            return np.inf
        return self.cfg.rate_max / self.cfg.ki

    def update(
        self,
        ref: float,
        meas: float,
        rate_meas: float = 0.0,
        rate_ff: float = 0.0,
    ) -> float:
        cfg = self.cfg
        ref = float(ref)
        meas = float(meas)
        if not (np.isfinite(ref) and np.isfinite(meas) and np.isfinite(rate_meas)):
            return float(self._rate_des)

        error = ref - meas
        lim = self._int_limit()
        self._integ = float(np.clip(self._integ + cfg.ki * error * cfg.h, -lim, lim))

        if cfg.form == "firmware":
            raw = cfg.kp * error + self._integ + cfg.kd * (self._rate_des - float(rate_meas))
        else:
            raw = float(rate_ff) + cfg.kp * error + self._integ

        self._rate_des = float(np.clip(raw, cfg.rate_min, cfg.rate_max))
        return self._rate_des


# ---------------------------------------------------------------------------
# Controller interface
# ---------------------------------------------------------------------------
class ControllerBase:
    """Common interface for every controller used by the bench/CLI."""

    n_channels: int = N_CHANNELS
    name: str = "controller"

    def reset(self) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    def update(
        self,
        y_ref: Sequence[float],
        y_meas: Sequence[float],
        rate_ref: Sequence[float] | None = None,
        rate_meas: Sequence[float] | None = None,
    ) -> np.ndarray:  # pragma: no cover - interface
        raise NotImplementedError

    def ack(self, u_actual: Sequence[float]) -> None:
        """Feed the realised command back (no-op for controllers without RLS)."""

    @property
    def theta(self) -> np.ndarray:
        return np.empty((self.n_channels, 0))

    @property
    def safe_mode(self) -> bool:
        return False

    def gains(self) -> list | None:
        return None


def _as4(value: Sequence[float] | np.ndarray | None) -> np.ndarray:
    if value is None:
        return np.zeros(N_CHANNELS)
    arr = np.asarray(value, dtype=float).reshape(-1)
    if arr.size != N_CHANNELS:
        raise ValueError("expected four channel values")
    return arr


# ---------------------------------------------------------------------------
# Mode II: outer PI + inner STR
# ---------------------------------------------------------------------------
class PISTRController(ControllerBase):
    """``Mode II``: PI outer loop on angle/altitude, 2nd-order STR inner loop."""

    name = "pi_str"

    def __init__(
        self,
        params,
        str_configs: Sequence[STRConfig] | None = None,
        outer: OuterPIConfig | None = None,
        percent: float | None = None,
        order: int = 2,
    ):
        control = params.control
        self.order = int(order)
        self.outer_cfg = outer or OuterPIConfig(
            kp=float(control.kp_outer),
            ki=float(control.ki_outer),
            kd=float(control.kd_outer),
            h=float(control.h),
            form="firmware",
        )
        if str_configs is None:
            factory = STRConfig.mode_ii if self.order == 2 else STRConfig.mode_i
            str_configs = [factory(params, ch, percent=percent) for ch in range(N_CHANNELS)]
        self.inner = STRBank(str_configs)
        self.outer = [OuterPI(self.outer_cfg) for _ in range(N_CHANNELS)]
        self.n_channels = len(str_configs)

    @property
    def theta(self) -> np.ndarray:
        return self.inner.theta

    @property
    def safe_mode(self) -> bool:
        return self.inner.safe_mode()

    def gains(self):
        return self.inner.gains()

    def reset(self) -> None:
        self.inner.reset()
        for pi in self.outer:
            pi.reset()

    def resetdata(self) -> None:
        self.reset()

    def update(
        self,
        y_ref: Sequence[float],
        y_meas: Sequence[float],
        rate_ref: Sequence[float] | None = None,
        rate_meas: Sequence[float] | None = None,
    ) -> np.ndarray:
        y_ref = _as4(y_ref)
        y_meas = _as4(y_meas)
        rate_ref = _as4(rate_ref)
        rate_meas = _as4(rate_meas)

        rate_des = np.empty(self.n_channels)
        for i in range(self.n_channels):
            rate_des[i] = self.outer[i].update(
                y_ref[i], y_meas[i], rate_meas=rate_meas[i], rate_ff=rate_ref[i]
            )
        return self.inner.step(rate_des, rate_meas)

    def ack(self, u_actual: Sequence[float]) -> None:
        self.inner.ack(u_actual)


# ---------------------------------------------------------------------------
# Mode I: single-loop 3rd-order STR
# ---------------------------------------------------------------------------
class SingleLoopSTRController(ControllerBase):
    """``Mode I``: the STR is fed the outer reference directly (order 3)."""

    name = "str"

    def __init__(
        self,
        params,
        str_configs: Sequence[STRConfig] | None = None,
        percent: float | None = None,
    ):
        if str_configs is None:
            str_configs = [
                STRConfig.mode_i(params, ch, percent=percent) for ch in range(N_CHANNELS)
            ]
        self.inner = STRBank(str_configs)
        self.n_channels = len(str_configs)
        self.order = int(str_configs[0].order)

    @property
    def theta(self) -> np.ndarray:
        return self.inner.theta

    @property
    def safe_mode(self) -> bool:
        return self.inner.safe_mode()

    def gains(self):
        return self.inner.gains()

    def reset(self) -> None:
        self.inner.reset()

    resetdata = reset

    def update(
        self,
        y_ref: Sequence[float],
        y_meas: Sequence[float],
        rate_ref: Sequence[float] | None = None,
        rate_meas: Sequence[float] | None = None,
    ) -> np.ndarray:
        return self.inner.step(_as4(y_ref), _as4(y_meas))

    def ack(self, u_actual: Sequence[float]) -> None:
        self.inner.ack(u_actual)


# ---------------------------------------------------------------------------
# PID adapters (baseline)
# ---------------------------------------------------------------------------
class RobustPIDController(ControllerBase):
    """Paper ``c1`` robust PID in the common interface."""

    name = "pid"

    def __init__(self, params):
        self.bank = robust_pid_from_params(params)
        self.n_channels = N_CHANNELS

    def reset(self) -> None:
        self.bank.reset()

    def update(self, y_ref, y_meas, rate_ref=None, rate_meas=None) -> np.ndarray:
        return self.bank.update(y_ref, y_meas, rate_ref, rate_meas)


class CascadePIDController(ControllerBase):
    """Firmware ``PID_v2`` cascade in the common interface."""

    name = "cascade_pid"

    def __init__(self, params, **kwargs):
        control = params.control
        self.bank = CascadePIDBank(u_max=control.u_max, h=control.h, **kwargs)
        self.n_channels = N_CHANNELS

    def reset(self) -> None:
        self.bank.reset()

    def update(self, y_ref, y_meas, rate_ref=None, rate_meas=None) -> np.ndarray:
        return self.bank.update(y_ref, y_meas, rate_meas if rate_meas is not None else np.zeros(4))


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
CONTROLLER_NAMES = ("pid", "str", "pi_str", "cascade_pid")


def make_controller(
    name: str,
    params,
    order: int | None = None,
    outer_form: str = "firmware",
    percent: float | None = None,
    str_configs: Sequence[STRConfig] | None = None,
    rate_max: float | None = None,
):
    """Build a controller by name.

    ``name="str"`` is Mode I (order 3).  ``name="pi_str"`` is Mode II; passing
    ``order=3`` builds the cascade on a 3rd-order inner model (used only for the
    cross-checks; the paper's Mode II is order 2).
    """
    key = name.lower()
    if key == "pid":
        return RobustPIDController(params)
    if key == "cascade_pid":
        return CascadePIDController(params)
    if key in ("str", "mode_i", "mode-i"):
        return SingleLoopSTRController(params, str_configs=str_configs, percent=percent)
    if key in ("pi_str", "pi-str", "mode_ii", "mode-ii"):
        control = params.control
        kp, ki, kd = (
            (1.0, 0.02, 0.0) if outer_form == "paper"
            else (float(control.kp_outer), float(control.ki_outer), float(control.kd_outer))
        )
        outer = OuterPIConfig(
            kp=kp,
            ki=ki,
            kd=kd,
            h=float(control.h),
            form=outer_form,
            rate_max=float(rate_max if rate_max is not None else 6.0),
            rate_min=-float(rate_max if rate_max is not None else 6.0),
        )
        return PISTRController(
            params,
            str_configs=str_configs,
            outer=outer,
            percent=percent,
            order=2 if order is None else int(order),
        )
    raise ValueError(f"unknown controller {name!r}; choose from {CONTROLLER_NAMES}")


__all__ = [
    "N_CHANNELS",
    "OuterPIConfig",
    "OuterPI",
    "ControllerBase",
    "PISTRController",
    "SingleLoopSTRController",
    "RobustPIDController",
    "CascadePIDController",
    "make_controller",
    "CONTROLLER_NAMES",
]
