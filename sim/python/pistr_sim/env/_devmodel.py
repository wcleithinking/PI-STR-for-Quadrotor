"""Throw-away development model so M3 (render) can start before M1 lands.

Not part of the contract: it has the right mass/inertia/arm geometry and a hover-height
level pose, but no actuation, no scene, no contact tuning. Replaced by
``pistr_sim.env.mjcf.build_model`` as soon as M1 exists. Kept afterwards only as a
render unit-test fixture (a scene that is trivially known to be valid).
"""

from __future__ import annotations

import numpy as np

from ..params import default_params


def _inertia_diag(J: np.ndarray) -> str:
    return " ".join(f"{v:.6f}" for v in (J[0, 0], J[1, 1], J[2, 2]))


def dev_xml(altitude: float = 1.5, *, params=None) -> str:
    p = params or default_params("high")
    a = p.airframe
    J = a.inertia
    rotors = "\n".join(
        f'      <geom type="cylinder" pos="{x:.4f} {y:.4f} 0" size="0.012 0.012 0.02" '
        f'rgba="{c}"/>'
        for (x, y, _), c in zip(
            a.arm_positions, [".9 .3 .2 1", ".3 .9 .5 1", ".9 .8 .2 1", ".3 .6 .95 1"]
        )
    )
    return f"""<mujoco model="pistr-dev">
  <compiler angle="radian" inertiafromgeom="false"/>
  <option gravity="0 0 -9.81" timestep="0.002"/>
  <visual><global offwidth="640" offheight="480"/><quality shadowsize="2048"/></visual>
  <asset>
    <texture name="grid" type="2d" builtin="checker" rgb1=".22 .30 .38" rgb2=".45 .55 .65" width="512" height="512"/>
    <material name="grid" texture="grid" texrepeat="10 10" reflectance="0.05"/>
  </asset>
  <worldbody>
    <light pos="6 -6 12" dir="-0.4 0.4 -1" diffuse=".8 .8 .8"/>
    <geom name="floor" type="plane" size="40 40 .05" material="grid"/>
    <body name="drone" pos="0 0 {altitude}">
      <freejoint/>
      <inertial pos="0 0 0" mass="{a.mass:.6f}" diaginertia="{_inertia_diag(J)}"/>
      <geom type="box" size="0.16 0.16 0.04" rgba=".18 .20 .24 1"/>
{rotors}
    </body>
  </worldbody>
</mujoco>
"""


def build_dev_model(altitude: float = 1.5, *, params=None):
    import mujoco

    return mujoco.MjModel.from_xml_string(dev_xml(altitude, params=params))
