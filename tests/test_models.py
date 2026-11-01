from math import sqrt

import pytest

from src.models import (
    DragModel,
    LaunchParameters,
    calculate_acceleration,
)


def test_initial_velocity_at_45_degrees() -> None:
    parameters = LaunchParameters(
        initial_speed=20.0,
        angle_deg=45.0,
        mass=0.2,
    )

    vx0, vy0 = parameters.initial_velocity()

    expected_component = 20.0 / sqrt(2)

    assert vx0 == pytest.approx(expected_component)
    assert vy0 == pytest.approx(expected_component)


def test_acceleration_without_drag() -> None:
    parameters = LaunchParameters(
        initial_speed=20.0,
        angle_deg=45.0,
        mass=0.2,
    )

    ax, ay = calculate_acceleration(
        vx=10.0,
        vy=5.0,
        parameters=parameters,
        model=DragModel.NONE,
    )

    assert ax == pytest.approx(0.0)
    assert ay == pytest.approx(-9.81)