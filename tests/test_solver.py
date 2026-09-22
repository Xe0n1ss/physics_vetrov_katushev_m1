import pytest

from src.analytical import (
    vacuum_flight_time,
    vacuum_max_height,
    vacuum_range,
    linear_position_x,
    linear_position_y,
    linear_velocity_x,
    linear_velocity_y,

)
from src.models import DragModel, LaunchParameters
from src.solver import solve_trajectory


def test_numerical_solution_matches_vacuum_theory() -> None:
    parameters = LaunchParameters(
        initial_speed=20.0,
        angle_deg=45.0,
        mass=0.2,
    )

    result = solve_trajectory(
        parameters=parameters,
        model=DragModel.NONE,
    )

    assert result.flight_time == pytest.approx(
        vacuum_flight_time(parameters),
        rel=1e-9,
    )

    assert result.horizontal_range == pytest.approx(
        vacuum_range(parameters),
        rel=1e-9,
    )

    assert result.max_height == pytest.approx(
        vacuum_max_height(parameters),
        abs=5e-4,
    )

    assert result.y[-1] == pytest.approx(
        0.0,
        abs=1e-10,
    )

    assert result.impact_speed == pytest.approx(
        parameters.initial_speed,
        rel=1e-9,
    )


def test_zero_drag_matches_vacuum() -> None:
    parameters = LaunchParameters(
        initial_speed=20.0,
        angle_deg=45.0,
        mass=0.2,
    )

    vacuum_result = solve_trajectory(
        parameters=parameters,
        model=DragModel.NONE,
    )

    linear_result = solve_trajectory(
        parameters=parameters,
        model=DragModel.LINEAR,
        drag_coeff=0.0,
    )

    quadratic_result = solve_trajectory(
        parameters=parameters,
        model=DragModel.QUADRATIC,
        drag_coeff=0.0,
    )

    assert linear_result.flight_time == pytest.approx(
        vacuum_result.flight_time
    )

    assert linear_result.horizontal_range == pytest.approx(
        vacuum_result.horizontal_range
    )

    assert quadratic_result.flight_time == pytest.approx(
        vacuum_result.flight_time
    )

    assert quadratic_result.horizontal_range == pytest.approx(
        vacuum_result.horizontal_range
    )

def test_numerical_solution_matches_linear_theory() -> None:
    parameters = LaunchParameters(
        initial_speed=20.0,
        angle_deg=45.0,
        mass=0.2,
    )
    drag_coeff = 0.02

    result = solve_trajectory(
        parameters=parameters,
        model=DragModel.LINEAR,
        drag_coeff=drag_coeff,
    )

    expected_x = [
        linear_position_x(t, parameters, drag_coeff)
        for t in result.time
    ]
    expected_y = [
        linear_position_y(t, parameters, drag_coeff)
        for t in result.time
    ]
    expected_vx = [
        linear_velocity_x(t, parameters, drag_coeff)
        for t in result.time
    ]
    expected_vy = [
        linear_velocity_y(t, parameters, drag_coeff)
        for t in result.time
    ]

    assert result.x == pytest.approx(expected_x, rel=0, abs=1e-9)
    assert result.y == pytest.approx(expected_y, rel=0, abs=1e-9)
    assert result.vx == pytest.approx(expected_vx, rel=0, abs=1e-9)
    assert result.vy == pytest.approx(expected_vy, rel=0, abs=1e-9)

def test_apex_event_finds_max_height() -> None:
    parameters = LaunchParameters(
        initial_speed=20.0,
        angle_deg=45.0,
        mass=0.2,
    )

    result = solve_trajectory(
        parameters=parameters,
        model=DragModel.NONE,
        max_step=1.0,
        rtol=1e-3,
        atol=1e-6,
    )

    assert result.apex_height is not None
    assert result.max_height == pytest.approx(
        vacuum_max_height(parameters),
        abs=1e-9,
    )

def test_quadratic_solution_converges() -> None:
    parameters = LaunchParameters(
        initial_speed=20.0,
        angle_deg=45.0,
        mass=0.2,
    )

    coarse = solve_trajectory(
        parameters,
        DragModel.QUADRATIC,
        drag_coeff=0.001,
        max_step=1.0,
        rtol=1e-3,
        atol=1e-6,
    )

    medium = solve_trajectory(
        parameters,
        DragModel.QUADRATIC,
        drag_coeff=0.001,
        max_step=0.2,
        rtol=1e-5,
        atol=1e-8,
    )

    fine = solve_trajectory(
        parameters,
        DragModel.QUADRATIC,
        drag_coeff=0.001,
        max_step=0.02,
        rtol=1e-9,
        atol=1e-11,
    )

    assert abs(medium.flight_time - fine.flight_time) < abs(
        coarse.flight_time - medium.flight_time
    )

    assert abs(medium.horizontal_range - fine.horizontal_range) < abs(
        coarse.horizontal_range - medium.horizontal_range
    )

    assert abs(medium.max_height - fine.max_height) < abs(
        coarse.max_height - medium.max_height
    )