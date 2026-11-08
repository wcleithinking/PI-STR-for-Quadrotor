"""Analytical CPU budget for the controller variants.

The Arduino Due runs at 84 MHz with a 40 ms control period, i.e. 3.36e6 cycles
per tick.  The estimates below count floating-point operations per controller
tick from the implemented algorithms (RLS, projection, Diophantine solve, control
law, outer PI), add the common stack (attitude observer, mixer, 3-D planner) and
convert with a deliberately conservative 4 cycles per FLOP.  They are an
order-of-magnitude design budget, not a measurement on hardware.
"""

from __future__ import annotations

#: Conservative cycles per floating-point operation on the Cortex-M3 FPU-less Due
#: (software float): the true figure for a mix of mul/add/div is ~2-6.
CYCLES_PER_FLOP = 4.0
#: Due clock and control period.
CLOCK_HZ = 84e6
PERIOD_S = 0.04
BUDGET_CYCLES = CLOCK_HZ * PERIOD_S


def _str_flops(order: int) -> float:
    n = 2 * order
    rls = 6.0 * n * n + 8.0 * n + 4.0
    projection = 2.0 * n + 8.0
    diophantine = 2.0 / 3.0 * (n + 1) ** 3 + 6.0 * (n + 1)
    control_law = 3.0 * n + 6.0
    outer_pi = 12.0
    return rls + projection + diophantine + control_law + outer_pi


STR2 = _str_flops(2)   # ~218 FLOPs per channel
STR3 = _str_flops(3)   # ~400 FLOPs per channel
PID_CHANNEL = 14.0
LQR_CHANNEL = 10.0
ESTIMATOR = 180.0      # quaternion propagation + mag correction + altitude KF
MIXER = 25.0
PLANNER_3D = 60.0      # position PD + attitude planning + 2 ESOs


def estimate(name: str) -> dict:
    key = name.lower()
    if key == "pid":
        controller = 4 * PID_CHANNEL
        str_channels = 0
    elif key == "str":
        controller = 4 * STR3
        str_channels = 4
    elif key == "pi_str":
        controller = 4 * STR2
        str_channels = 4
    elif key == "pi_str_3axis":
        controller = 3 * STR2 + PID_CHANNEL
        str_channels = 3
    elif key == "pi_str_1axis":
        controller = STR2 + 3 * PID_CHANNEL
        str_channels = 1
    elif key == "lqr":
        controller = 4 * LQR_CHANNEL
        str_channels = 0
    else:
        raise ValueError(f"unknown controller {name!r}")
    total = controller + ESTIMATOR + MIXER + PLANNER_3D
    cycles = total * CYCLES_PER_FLOP
    return {
        "controller": name,
        "str_channels": str_channels,
        "controller_flops": round(controller, 1),
        "total_flops": round(total, 1),
        "cycles_per_tick": int(round(cycles)),
        "budget_pct": round(100.0 * cycles / BUDGET_CYCLES, 3),
    }


def table() -> list[dict]:
    return [estimate(name) for name in ("pid", "str", "pi_str_1axis", "pi_str_3axis", "pi_str", "lqr")]


__all__ = ["CYCLES_PER_FLOP", "CLOCK_HZ", "PERIOD_S", "BUDGET_CYCLES", "estimate", "table"]
