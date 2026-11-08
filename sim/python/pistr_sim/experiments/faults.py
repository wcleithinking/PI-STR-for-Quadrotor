"""Fault scenarios for the D-phase matrix.

The controller always uses the *nominal* parameters; a fault changes the plant
only.  This is what makes the matrix a robustness test rather than a retuning
exercise.

Faults
------
``none``            nominal plant.
``aero_density``    rotor coefficients scaled by +20 % (``kT, kH, kQ`` ∝ rho),
                    matching the paper's ``kT=(1+0.2 sin t)kT_N`` style disturbance.
``wind_gust``       steady 3 m/s inertial wind + Dryden turbulence.  The ``paper``
                    fidelity has no aero/wind path, so there the fault is a no-op
                    and is reported as such (not silently dropped).
``mass_change``     +0.25 kg CoG mass (inertia follows from ``Airframe.inertia``).
``voltage_decay``   -15 % pack voltage: ``motor.u_max`` and the LiPo OCV curve are
                    scaled, which reproduces a discharged pack in both fidelities.

Every scenario record is a frozen dataclass so a matrix row is reproducible from
its configuration alone.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from ..params import Airframe, Battery, Motor, Params, default_params
from ..plant import QuadrotorPlant


@dataclass(frozen=True)
class FaultSpec:
    name: str
    description: str
    rho_scale: float = 1.0
    mass_add: float = 0.0
    voltage_scale: float = 1.0
    wind: tuple[float, float, float] = (0.0, 0.0, 0.0)
    turbulence: bool = True


FAULTS: dict[str, FaultSpec] = {
    "none": FaultSpec("none", "nominal plant"),
    "aero_density": FaultSpec(
        "aero_density", "rotor coefficients +20 % (kT, kH, kQ scale with rho)", rho_scale=1.2
    ),
    "wind_gust": FaultSpec(
        "wind_gust",
        "3 m/s steady wind + Dryden turbulence (no effect in paper fidelity)",
        wind=(3.0, 0.0, 0.0),
    ),
    "mass_change": FaultSpec("mass_change", "+0.25 kg CoG mass", mass_add=0.25),
    "voltage_decay": FaultSpec(
        "voltage_decay", "-15 % pack voltage (motor u_max and LiPo OCV)", voltage_scale=0.85
    ),
}
FAULT_NAMES = tuple(FAULTS)


def apply_fault(params: Params, fault: str) -> Params:
    """Return a modified copy of ``params`` for the plant side of the fault."""
    spec = FAULTS[fault]
    if spec.rho_scale != 1.0:
        rho = params.airframe.rho * spec.rho_scale
        rotor = replace(
            params.rotor,
            kt=params.rotor.kt * spec.rho_scale,
            kh=params.rotor.kh * spec.rho_scale,
            kq=params.rotor.kq * spec.rho_scale,
        )
        airframe = replace(params.airframe, rho=rho)
        params = replace(params, airframe=airframe, rotor=rotor)
    if spec.mass_add:
        params = replace(
            params, airframe=replace(params.airframe, m_cog=params.airframe.m_cog + spec.mass_add)
        )
    if spec.voltage_scale != 1.0:
        motor = replace(params.motor, u_max=params.motor.u_max * spec.voltage_scale)
        battery = replace(
            params.battery,
            cell_ocv=tuple(v * spec.voltage_scale for v in params.battery.cell_ocv),
        )
        params = replace(params, motor=motor, battery=battery)
    return params


def build_plant(
    fidelity: str,
    fault: str = "none",
    *,
    seed: int = 0,
    extras: dict[str, bool] | None = None,
) -> QuadrotorPlant:
    """Build the plant for ``(fidelity, fault)``; wind is armed after reset."""
    if fault not in FAULTS:
        raise ValueError(f"unknown fault {fault!r}; choose from {FAULT_NAMES}")
    spec = FAULTS[fault]
    nominal = default_params(fidelity)
    plant_params = apply_fault(nominal, fault)
    plant = QuadrotorPlant(
        params=plant_params, fidelity=fidelity, extras=extras, seed=int(seed)
    )
    plant._fault_wind = spec.wind  # consumed by loop._arm_wind after reset
    plant._fault_turbulence = spec.turbulence
    return plant


def arm_wind(plant: QuadrotorPlant) -> None:
    """Apply the fault's steady wind/turbulence switch after ``plant.reset``."""
    wind = getattr(plant, "_fault_wind", (0.0, 0.0, 0.0))
    turbulence = getattr(plant, "_fault_turbulence", True)
    if np.any(np.asarray(wind) != 0.0) or turbulence:
        plant.set_wind(np.asarray(wind, dtype=float), turbulence=bool(turbulence))


__all__ = [
    "FaultSpec",
    "FAULTS",
    "FAULT_NAMES",
    "apply_fault",
    "build_plant",
    "arm_wind",
]
