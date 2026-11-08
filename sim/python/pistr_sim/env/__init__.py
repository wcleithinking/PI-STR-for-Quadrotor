"""MuJoCo-based 3-D environment (optional dependency).

The rest of ``pistr_sim`` is a headless time-series integrator; this package adds a
rendered world with ground contact, a camera and multi-vehicle support. Importing
``pistr_sim`` must not require MuJoCo, so every public name is resolved lazily.

See ``docs/3D_ENVIRONMENT.md`` for the contract and the frame mapping.
"""

from __future__ import annotations

__all__ = [
    "build_model",
    "build_xml",
    "SceneSpec",
    "RotorPack",
    "MuJoCoPlant",
    "run_flight",
]


def __getattr__(name: str):  # pragma: no cover - thin lazy loader
    if name in ("build_model", "build_xml"):
        from . import mjcf

        return getattr(mjcf, name)
    if name == "SceneSpec":
        from .scene import SceneSpec

        return SceneSpec
    if name == "RotorPack":
        from .actuation import RotorPack

        return RotorPack
    if name == "MuJoCoPlant":
        from .bridge import MuJoCoPlant

        return MuJoCoPlant
    if name == "run_flight":
        from .loop import run_flight

        return run_flight
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
