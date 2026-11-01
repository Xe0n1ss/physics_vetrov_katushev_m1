from dataclasses import replace

import numpy as np
import pytest

from src.models import DragModel, LaunchParameters
from src.solver import solve_trajectory
from src.sweeps import sweep_angle, sweep_drag, sweep_speed


def base_parameters() -> LaunchParameters:
    return LaunchParameters(
        initial_speed=20.0,
        angle_deg=45.0,
        mass=0.2,
    )


def test_sweep_angle_returns_consistent_array_lengths() -> None:
    angles = np.linspace(10.0, 80.0, 8)

    result = sweep_angle(
        parameters=base_parameters(),
        angles=angles,
    )

    assert result.values == pytest.approx(angles)
    assert len(result.flight_time) == len(angles)
    assert len(result.horizontal_range) == len(angles)
    assert len(result.max_height) == len(angles)
    assert len(result.impact_speed) == len(angles)


def test_sweep_angle_point_matches_direct_solve() -> None:
    parameters = base_parameters()

    result = sweep_angle(
        parameters=parameters,
        angles=[30.0],
        model=DragModel.QUADRATIC,
        drag_coeff=0.001,
    )

    direct = solve_trajectory(
        parameters=replace(parameters, angle_deg=30.0),
        model=DragModel.QUADRATIC,
        drag_coeff=0.001,
    )

    assert result.horizontal_range[0] == pytest.approx(direct.horizontal_range)
    assert result.flight_time[0] == pytest.approx(direct.flight_time)
    assert result.max_height[0] == pytest.approx(direct.max_height)
    assert result.impact_speed[0] == pytest.approx(direct.impact_speed)


def test_sweep_angle_reproduces_vacuum_range_formula() -> None:
    parameters = base_parameters()
    angles = np.array([15.0, 30.0, 45.0, 60.0, 75.0])

    result = sweep_angle(
        parameters=parameters,
        angles=angles,
    )

    expected = (
        parameters.initial_speed ** 2
        * np.sin(2 * np.radians(angles))
        / parameters.gravity
    )

    assert result.horizontal_range == pytest.approx(expected, rel=1e-6)


def test_sweep_speed_reproduces_vacuum_range_formula() -> None:
    parameters = base_parameters()
    speeds = np.array([10.0, 20.0, 30.0])

    result = sweep_speed(
        parameters=parameters,
        speeds=speeds,
    )

    expected = speeds ** 2 / parameters.gravity

    assert result.horizontal_range == pytest.approx(expected, rel=1e-6)


def test_sweep_drag_range_decreases_monotonically() -> None:
    result = sweep_drag(
        parameters=base_parameters(),
        drag_coeffs=[0.0, 0.001, 0.002, 0.005, 0.01],
    )

    assert np.all(np.diff(result.horizontal_range) < 0)
