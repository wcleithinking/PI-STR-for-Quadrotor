"""Controller registry for the D-phase matrix.

All entries implement the four-channel interface used by the experiment loop
(``update(y_ref, y_meas, rate_ref, rate_meas)`` + ``ack``), exactly like the
controllers from task-2/task-6.

The hybrid entries are the point of task-5 item 1: the paper/firmware runs STR
on one axis (firmware: yaw) and PID elsewhere; ``pi_str_3axis`` runs STR on the
three attitude axes and PID on altitude; ``pi_str`` runs STR on all four.
"""

from __future__ import annotations

import numpy as np

from ..control import (
    STRConfig,
    ControllerBase,
    RobustPIDController,
    SingleLoopSTRController,
    make_controller,
)
from .lqr import cached_lqr

CONTROLLERS = ("pid", "str", "pi_str", "pi_str_1axis", "pi_str_3axis", "lqr")


def _pi_str_configs(params, percent: float = 0.95):
    return [
        STRConfig.mode_ii(params, ch, percent=percent, p_max=100.0, forgetting=0.998)
        for ch in range(4)
    ]


def _mode_i_configs(params):
    return [
        STRConfig.mode_i(params, ch, percent=1.0, p_max=100.0, forgetting=0.998)
        for ch in range(4)
    ]


class HybridController(ControllerBase):
    """Run a STR cascade and a robust PID in parallel, select channels per axis."""

    def __init__(self, str_controller, pid_controller, str_channels):
        self.str = str_controller
        self.pid = pid_controller
        self.str_channels = tuple(int(c) for c in str_channels)
        self.n_channels = 4
        self.name = "hybrid"

    def reset(self) -> None:
        self.str.reset()
        self.pid.reset()

    def update(self, y_ref, y_meas, rate_ref=None, rate_meas=None) -> np.ndarray:
        u_str = np.asarray(self.str.update(y_ref, y_meas, rate_ref, rate_meas), dtype=float)
        u_pid = np.asarray(self.pid.update(y_ref, y_meas, rate_ref, rate_meas), dtype=float)
        u = u_pid.copy()
        for j in self.str_channels:
            u[j] = u_str[j]
        return u

    def ack(self, u_actual) -> None:
        self.str.ack(u_actual)

    @property
    def theta(self) -> np.ndarray:
        return self.str.theta

    @property
    def safe_mode(self) -> bool:
        return self.str.safe_mode

    def gains(self):
        return self.str.gains()


def build_controller(name: str, params, fidelity: str) -> ControllerBase:
    key = name.lower()
    if key == "pid":
        return RobustPIDController(params)
    if key == "str":
        return SingleLoopSTRController(params, str_configs=_mode_i_configs(params))
    if key == "pi_str":
        return make_controller(
            "pi_str",
            params,
            outer_form="paper",
            str_configs=_pi_str_configs(params),
            rate_max=2.0,
        )
    if key == "pi_str_1axis":
        inner = make_controller(
            "pi_str",
            params,
            outer_form="paper",
            str_configs=_pi_str_configs(params),
            rate_max=2.0,
        )
        hybrid = HybridController(inner, RobustPIDController(params), str_channels=(2,))
        hybrid.name = "pi_str_1axis"
        return hybrid
    if key == "pi_str_3axis":
        inner = make_controller(
            "pi_str",
            params,
            outer_form="paper",
            str_configs=_pi_str_configs(params),
            rate_max=2.0,
        )
        hybrid = HybridController(inner, RobustPIDController(params), str_channels=(0, 1, 2))
        hybrid.name = "pi_str_3axis"
        return hybrid
    if key == "lqr":
        return cached_lqr(fidelity)
    raise ValueError(f"unknown controller {name!r}; choose from {CONTROLLERS}")


def controller_names() -> tuple[str, ...]:
    return CONTROLLERS


__all__ = ["CONTROLLERS", "HybridController", "build_controller", "controller_names"]
