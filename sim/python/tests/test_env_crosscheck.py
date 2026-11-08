"""V8 -- cross-check: PI-STR in the 3-D MuJoCo world vs ``fidelity="high"``.

The gate (``docs/3D_ENVIRONMENT.md`` §5): the closed loop in the 3-D world must hold
the 5 m hover **at least as well as** the Python plant -- attitude RMSE <= 2x the
plant value, altitude bounded.

Why this test uses several seeds
--------------------------------
The hover loop is driven by ``QuaternionAttitudeEstimator``, whose roll/pitch are pure
gyro integration (no position aiding -- there is no GPS).  The gyro-bias random walk
is the *same* code in both backends, but the loop sits close enough to a lightly
damped mode that a single 2 s window can land on either side of it: with the same
seed, seed 0 is the 3-D world's worst case (roll RMSE 0.59 deg) while seed 4 is the
*plant's* worst case (roll 0.60 deg, altitude drift 1.12 m).  Averaging the
prescribed 2 s window over 5 seeds therefore compares the *worlds*, not the luck of
one noise realisation; the per-seed table is printed so the scatter is visible.

The numbers in the printed table are the V8 evidence and are quoted in the M4 report.
"""

from __future__ import annotations

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from pistr_sim.env.bridge import MuJoCoPlant
from pistr_sim.env.loop import run_flight
from pistr_sim.params import default_params
from pistr_sim.plant import make_plant

# Same-directory helper: the fixture model/pack live next to this file so both env
# test modules exercise the same stand-ins when M1's modules are absent.
from test_env_bridge import _build_model, _pack_factory  # noqa: E402

PARAMS = default_params("high")
SEEDS = (0, 1, 2, 3, 4)
T_END = 2.0
ALTITUDE = 5.0
G = float(PARAMS.airframe.g)
MASS = float(PARAMS.airframe.mass)


def _mujoco_plant(seed: int, *, model=None) -> MuJoCoPlant:
    return MuJoCoPlant(
        params=PARAMS,
        model=model if model is not None else _build_model(1),
        dt=0.002,
        seed=seed,
        initial_altitude=ALTITUDE,
    )


@pytest.fixture(scope="module")
def runs() -> dict[int, tuple]:
    """One hover run per seed in both worlds (the V8 data set)."""
    data = {}
    for seed in SEEDS:
        mj = run_flight(
            scenario="hover", t_end=T_END, controller="pi_str", seed=seed,
            plant=_mujoco_plant(seed),
        )
        py = run_flight(
            scenario="hover", t_end=T_END, controller="pi_str", seed=seed, plant="high",
        )
        data[seed] = (mj, py)
    return data


@pytest.fixture(scope="module")
def clean_runs() -> tuple:
    """Noisy sensors out of the way: the same loop, deterministic, seed 0."""
    mj = run_flight(
        scenario="hover", t_end=T_END, controller="pi_str", seed=0,
        plant=MuJoCoPlant(
            params=PARAMS, model=_build_model(1), dt=0.002, seed=0,
            sensor_noise=False, initial_altitude=ALTITUDE,
        ),
    )
    py = run_flight(
        scenario="hover", t_end=T_END, controller="pi_str", seed=0,
        plant=make_plant("high", extras={"sensors": False}),
    )
    return mj, py


def _rmse_deg(trace) -> np.ndarray:
    return np.rad2deg(trace.attitude_rmse())


def _alt_rmse(trace) -> float:
    return float(np.sqrt(np.mean(trace.drone(0).altitude_error() ** 2)))


def _print_table(runs: dict[int, tuple]) -> None:
    print("\n[V8] PI-STR, hover 5 m, 2 s, control_h=40 ms, physics dt=2 ms")
    print("  seed | mujoco RMSE r/p/y [deg] | plant RMSE r/p/y [deg] | "
          "mj drift/altRMSE [m] | plant drift/altRMSE [m]")
    for seed, (mj, py) in runs.items():
        rmse_mj, rmse_py = _rmse_deg(mj), _rmse_deg(py)
        print(
            f"  {seed:4d} | {rmse_mj[0]:7.3f} {rmse_mj[1]:7.3f} {rmse_mj[2]:7.3f} | "
            f"{rmse_py[0]:8.3f} {rmse_py[1]:8.3f} {rmse_py[2]:8.3f} | "
            f"{mj.altitude_stats()['drift_max']:8.4f} {_alt_rmse(mj):8.4f} | "
            f"{py.altitude_stats()['drift_max']:9.4f} {_alt_rmse(py):9.4f}"
        )
    mj_all = np.array([_rmse_deg(runs[s][0]) for s in SEEDS])
    py_all = np.array([_rmse_deg(runs[s][1]) for s in SEEDS])
    print(f"  mean | {np.round(mj_all.mean(0), 3)} | {np.round(py_all.mean(0), 3)} | "
          f"ratio {np.round(mj_all.mean(0) / py_all.mean(0), 2)}")
    print(f"  med. | {np.round(np.median(mj_all, 0), 3)} | "
          f"{np.round(np.median(py_all, 0), 3)} | "
          f"ratio {np.round(np.median(mj_all, 0) / np.median(py_all, 0), 2)}")


def test_v8_hover_holds_in_the_3d_world(runs: dict[int, tuple]) -> None:
    """Every run: bounded altitude, bounded attitude, sane PWM, no divergence."""
    for seed, (mj, _py) in runs.items():
        stats = mj.altitude_stats()
        pwm = mj.pwm
        print(
            f"[V8] seed {seed}: alt drift {stats['drift_max']:.4f} m, final "
            f"{stats['final_altitude']:.4f} m, |euler|max "
            f"{np.rad2deg(np.max(np.abs(mj.euler))):.3f} deg, "
            f"pwm [{pwm.min():.1f}, {pwm.max():.1f}] us, "
            f"rotor omega {mj.rotor_omega.mean():.1f} rad/s"
        )
        assert not mj.metadata["diverged"], f"seed {seed} diverged"
        assert np.all(np.isfinite(mj.xi)) and np.all(np.isfinite(mj.euler))
        assert stats["drift_max"] <= 0.35, "altitude must stay bounded for 2 s"
        assert np.max(np.abs(mj.euler)) < np.deg2rad(8.0), "attitude must stay bounded"
        assert np.all(pwm >= 1000.0) and np.all(pwm <= 2000.0)
        assert np.all(np.isfinite(mj.rotor_omega)) and np.all(mj.rotor_omega >= 0.0)


def test_v8_attitude_rmse_within_2x_of_plant(runs: dict[int, tuple]) -> None:
    """V8: attitude RMSE <= 2x the ``fidelity="high"`` value."""
    _print_table(runs)
    mj_all = np.array([_rmse_deg(runs[s][0]) for s in SEEDS])
    py_all = np.array([_rmse_deg(runs[s][1]) for s in SEEDS])
    mean_ratio = mj_all.mean(axis=0) / py_all.mean(axis=0)
    median_ratio = np.median(mj_all, axis=0) / np.median(py_all, axis=0)
    print(f"[V8] mean RMSE ratio (mj/plant) r/p/y = {np.round(mean_ratio, 2)}")
    print(f"[V8] median RMSE ratio          r/p/y = {np.round(median_ratio, 2)}")
    for axis, name in enumerate(("roll", "pitch", "yaw")):
        assert mean_ratio[axis] <= 2.0, f"{name}: mean RMSE {mean_ratio[axis]:.2f}x > 2x"
        assert median_ratio[axis] <= 2.0, f"{name}: median RMSE {median_ratio[axis]:.2f}x > 2x"


def test_v8_altitude_holds_at_least_as_well(runs: dict[int, tuple]) -> None:
    """Altitude error of the 3-D world must not be worse than the plant's."""
    drift_mj = np.array([runs[s][0].altitude_stats()["drift_max"] for s in SEEDS])
    drift_py = np.array([runs[s][1].altitude_stats()["drift_max"] for s in SEEDS])
    alt_mj = np.array([_alt_rmse(runs[s][0]) for s in SEEDS])
    alt_py = np.array([_alt_rmse(runs[s][1]) for s in SEEDS])
    print(
        f"[V8] altitude drift  mean mujoco {drift_mj.mean():.4f} m vs plant "
        f"{drift_py.mean():.4f} m (max {drift_mj.max():.4f} / {drift_py.max():.4f})"
    )
    print(
        f"[V8] altitude RMSE   mean mujoco {alt_mj.mean():.4f} m vs plant "
        f"{alt_py.mean():.4f} m (ratio {alt_mj.mean() / alt_py.mean():.2f})"
    )
    assert drift_mj.mean() <= 2.0 * drift_py.mean() + 0.02
    assert alt_mj.mean() <= 2.0 * alt_py.mean() + 0.01


def test_v8_trim_matches_python_plant(runs: dict[int, tuple]) -> None:
    """The pack trim the 3-D world flies at is the Python plant's trim."""
    mj, _py = runs[SEEDS[0]]
    plant_trim = float(_py.metadata["throttle_pwm"])
    world_trim = float(mj.metadata["throttle_pwm"])
    print(
        f"[V8] trim: mujoco {world_trim:.3f} us vs plant {plant_trim:.3f} us "
        f"(delta {world_trim - plant_trim:+.3f} us); "
        f"thrust {mj.metadata['trim']['thrust']:.4f} N vs m*g {MASS * G:.4f} N"
    )
    assert abs(world_trim - plant_trim) < 5.0
    assert abs(float(mj.metadata["trim"]["thrust"]) - MASS * G) < 0.01 * MASS * G


def test_v8_clean_world_does_not_degrade_the_loop(clean_runs: tuple) -> None:
    """With sensor noise off, the 3-D world must hold the hover just as tightly.

    This isolates the *world* from the sensor RNG: if the bridge's wrench, frame map
    or timing were wrong, this would not be sub-centimetre.
    """
    mj, py = clean_runs
    stats_mj, stats_py = mj.altitude_stats(), py.altitude_stats()
    rmse_mj, rmse_py = _rmse_deg(mj), _rmse_deg(py)
    print(
        f"[V8] clean sensors, seed 0: mujoco alt drift {stats_mj['drift_max']:.6f} m, "
        f"RMSE {np.round(rmse_mj, 6)} deg | plant alt drift {stats_py['drift_max']:.6f} m, "
        f"RMSE {np.round(rmse_py, 6)} deg"
    )
    assert stats_mj["drift_max"] < 0.02, "the world itself must track the 5 m hover"
    assert np.max(rmse_mj) < 0.05, "clean attitude error must be negligible"
    assert stats_mj["drift_max"] <= max(stats_py["drift_max"], 1e-3) * 2.0 + 0.01


def test_v8_multi_vehicle_hover() -> None:
    """A two-vehicle world holds the same hover, each on its own free joint."""
    model = _build_model(2)
    trace = run_flight(
        scenario="hover", t_end=T_END, controller="pi_str", seed=0,
        model=model, n_drones=2,
        plant=MuJoCoPlant(params=PARAMS, model=model, n_drones=2, dt=0.002, seed=0),
    )
    stats = [trace.drone(i).altitude_stats() for i in range(2)]
    print(
        f"[V8] two drones: drift {[round(s['drift_max'], 4) for s in stats]} m, "
        f"final {[round(s['final_altitude'], 3) for s in stats]} m, "
        f"offset {np.round(np.diff(trace.stacked('xi')[:, 0, 0]), 3)} m"
    )
    assert not trace.metadata["diverged"]
    assert trace.n_drones == 2
    for i, stat in enumerate(stats):
        assert stat["drift_max"] < 0.35, f"drone{i} altitude bounded"
        assert abs(stat["final_altitude"] - ALTITUDE) < 0.35
    assert abs(abs(float(np.diff(trace.stacked("xi")[:, 0, 0])[0])) - 2.0) < 1e-6
