import pytest

from src.analytical import (
    vacuum_flight_time,
    vacuum_max_height,
    vacuum_position,
    vacuum_range,
    vacuum_velocity,
)
from src.models import LaunchParameters


def test_vacuum_characteristics() -> None:
    parameters = LaunchParameters(
        initial_speed=20.0,
        angle_deg=45.0,
        mass=0.2,
    )

    _, vy0 = parameters.initial_velocity()

    expected_flight_time = (
        2 * vy0 / parameters.gravity
    )

    expected_max_height = (
        vy0 ** 2
        / (2 * parameters.gravity)
    )

    expected_range = (
        parameters.initial_speed ** 2
        / parameters.gravity
    )

    assert vacuum_flight_time(parameters) == pytest.approx(
        expected_flight_time
    )

    assert vacuum_max_height(parameters) == pytest.approx(
        expected_max_height
    )

    assert vacuum_range(parameters) == pytest.approx(
        expected_range
    )


def test_vacuum_landing_state() -> None:
    parameters = LaunchParameters(
        initial_speed=20.0,
        angle_deg=45.0,
        mass=0.2,
    )

    vx0, vy0 = parameters.initial_velocity()
    flight_time = vacuum_flight_time(parameters)

    landing_x, landing_y = vacuum_position(
        time=flight_time,
        parameters=parameters,
    )

    landing_vx, landing_vy = vacuum_velocity(
        time=flight_time,
        parameters=parameters,
    )

    assert landing_x == pytest.approx(
        vacuum_range(parameters)
    )

    assert landing_y == pytest.approx(
        0.0,
        abs=1e-12,
    )

    assert landing_vx == pytest.approx(vx0)
    assert landing_vy == pytest.approx(-vy0)

