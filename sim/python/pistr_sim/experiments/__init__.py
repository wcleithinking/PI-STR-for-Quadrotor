"""D-phase experiment harness: 3-axis STR, LQR baseline, matrix and report.

The package is deliberately additive: it only *uses* ``pistr_sim.plant``,
``pistr_sim.control`` and ``pistr_sim.estimator``; none of their algorithms are
modified.  Entry point::

    python -m pistr_sim.experiments.matrix --quick
    python -m pistr_sim.experiments.matrix --full
"""

from .controllers import CONTROLLERS, build_controller, controller_names
from .faults import FAULT_NAMES, apply_fault, build_plant
from .lqr import LQRController, design_lqr
from .loop import RunTrace, run_step, run_trajectory

__all__ = [
    "CONTROLLERS",
    "controller_names",
    "build_controller",
    "FAULT_NAMES",
    "apply_fault",
    "build_plant",
    "LQRController",
    "design_lqr",
    "RunTrace",
    "run_step",
    "run_trajectory",
]
