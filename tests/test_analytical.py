from math import radians

import pytest

from src.analytical import (
    exact_frictionless_period,
    small_angle_damped_angle,
    small_angle_damped_period,
    small_angle_period,
)
from src.models import PendulumParameters


@pytest.fixture
def parameters() -> PendulumParameters:
    return PendulumParameters(0.8, 0.2, 0.08, 0.5)


def test_exact_period_tends_to_small_angle_limit(parameters: PendulumParameters) -> None:
    assert exact_frictionless_period(parameters, 0) == pytest.approx(small_angle_period(parameters))
    assert exact_frictionless_period(parameters, 5) < exact_frictionless_period(parameters, 60)
    assert exact_frictionless_period(parameters, 60) < exact_frictionless_period(parameters, 150)


def test_damped_small_angle_solution_starts_from_rest() -> None:
    parameters = PendulumParameters(0.8, 0.2, 0.08, 0.5, damping=0.015)
    angle = small_angle_damped_angle(0.0, parameters, 5)
    numerical_initial_slope = (
        small_angle_damped_angle(1e-5, parameters, 5)
        - small_angle_damped_angle(-0.0, parameters, 5)
    ) / 1e-5
    assert angle == pytest.approx(radians(5))
    assert numerical_initial_slope == pytest.approx(0.0, abs=1e-5)
    assert small_angle_damped_period(parameters) > small_angle_period(parameters)


def test_invalid_amplitude_and_overdamping(parameters: PendulumParameters) -> None:
    with pytest.raises(ValueError):
        exact_frictionless_period(parameters, 180)
    with pytest.raises(ValueError):
        exact_frictionless_period(parameters, -1)
    overdamped = PendulumParameters(0.8, 0.2, 0.08, 0.5, damping=10)
    with pytest.raises(ValueError):
        small_angle_damped_period(overdamped)
    with pytest.raises(ValueError):
        small_angle_damped_angle(1, overdamped, 5)
