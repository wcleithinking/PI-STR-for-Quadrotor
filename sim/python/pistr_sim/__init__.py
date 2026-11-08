"""PI-STR quadrotor: a modern, testable re-implementation.

Layout
------
``types``    shared state/actuator contracts (no dependencies on anything else)
``params``   physical and controller parameters (faithful MATLAB port)
``plant``    two-fidelity physics: ``paper`` (parity) and ``high`` (realistic)
``control``  PID / STR / PI-STR controllers and the attitude estimator
``metrics``  tracking-quality metrics used by every experiment
``cli``      one-command entry point for scenarios, figures and CSV output
"""

from .params import Params, default_params
from .types import Estimate, SensorSample, VehicleState

__version__ = "0.1.0"

__all__ = [
    "Params",
    "default_params",
    "Estimate",
    "SensorSample",
    "VehicleState",
    "__version__",
]
