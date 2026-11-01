from math import pi

import pytest

from src.models import InitialConditions, PendulumParameters, motion_equations


@pytest.fixture
def parameters() -> PendulumParameters:
    return PendulumParameters(rod_length=0.8, rod_mass=0.2, sphere_radius=0.08, sphere_mass=0.5, gravity=9.81)


def test_geometry_and_inertia(parameters: PendulumParameters) -> None:
    assert parameters.sphere_center_distance == pytest.approx(0.88)
    assert parameters.center_of_mass_distance == pytest.approx((0.2 * 0.4 + 0.5 * 0.88) / 0.7)
    expected_inertia = 0.2 * 0.8**2 / 3 + 2 * 0.5 * 0.08**2 / 5 + 0.5 * 0.88**2
    assert parameters.moment_of_inertia == pytest.approx(expected_inertia)


def test_equation_and_energy(parameters: PendulumParameters) -> None:
    _, angular_acceleration = motion_equations(0.0, (pi / 2, 0.0), parameters)
    assert angular_acceleration == pytest.approx(
        -parameters.restoring_torque_scale / parameters.moment_of_inertia
    )
    assert parameters.energy(0.0, 0.0) == 0.0
    assert parameters.energy(pi, 0.0) == pytest.approx(2 * parameters.restoring_torque_scale)


def test_viscous_torque_changes_acceleration(parameters: PendulumParameters) -> None:
    damped = PendulumParameters(0.8, 0.2, 0.08, 0.5, 9.81, damping=0.015)
    ideal_acceleration = motion_equations(0.0, (0.0, 2.0), parameters)[1]
    damped_acceleration = motion_equations(0.0, (0.0, 2.0), damped)[1]
    assert ideal_acceleration == 0.0
    assert damped_acceleration == pytest.approx(-0.015 * 2 / damped.moment_of_inertia)


def test_invalid_inputs() -> None:
    with pytest.raises(ValueError):
        PendulumParameters(0.0, 0.2, 0.08, 0.5, 9.81)
    with pytest.raises(ValueError):
        PendulumParameters(0.8, 0.2, 0.08, 0.5, 9.81, damping=-0.01)
    with pytest.raises(ValueError):
        PendulumParameters(1e308, 0.2, 0.08, 0.5, 9.81)
    with pytest.raises(ValueError):
        InitialConditions(180)
    assert InitialConditions(60).angle_rad == pytest.approx(pi / 3)
