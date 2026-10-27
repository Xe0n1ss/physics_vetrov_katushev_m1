import numpy as np
import pytest

from src.analytical import exact_frictionless_period, small_angle_damped_angle
from src.models import InitialConditions, PendulumParameters
from src.solver import solve_pendulum


@pytest.fixture
def ideal_parameters() -> PendulumParameters:
    return PendulumParameters(0.8, 0.2, 0.08, 0.5, 9.81)


@pytest.mark.parametrize("amplitude", [5, 60, 150])
def test_numerical_period_matches_exact_theory(
    ideal_parameters: PendulumParameters, amplitude: float
) -> None:
    result = solve_pendulum(ideal_parameters, InitialConditions(amplitude), duration=8)
    theory = exact_frictionless_period(ideal_parameters, amplitude)
    assert result.period == pytest.approx(theory, rel=2e-8)


def test_energy_is_conserved_without_friction(ideal_parameters: PendulumParameters) -> None:
    result = solve_pendulum(ideal_parameters, InitialConditions(60))
    assert result.relative_energy_drift < 1e-8


def test_damped_energy_obeys_dissipation_law() -> None:
    parameters = PendulumParameters(0.8, 0.2, 0.08, 0.5, 9.81, damping=0.015)
    result = solve_pendulum(parameters, InitialConditions(60))
    assert result.energy[-1] < result.energy[0]
    assert np.max(np.diff(result.energy)) < 1e-9
    dissipated = parameters.damping * np.trapezoid(result.angular_velocity**2, result.time)
    assert result.energy[0] - result.energy[-1] == pytest.approx(dissipated, rel=2e-4)


def test_small_angle_damped_motion_matches_linear_solution() -> None:
    parameters = PendulumParameters(0.8, 0.2, 0.08, 0.5, 9.81, damping=0.015)
    result = solve_pendulum(parameters, InitialConditions(1), duration=8)
    theory = np.array([small_angle_damped_angle(t, parameters, 1) for t in result.time])
    assert np.max(np.abs(result.angle - theory)) < 2e-5


def test_short_simulation_cannot_measure_period(ideal_parameters: PendulumParameters) -> None:
    result = solve_pendulum(ideal_parameters, InitialConditions(60), duration=0.1)
    with pytest.raises(ValueError):
        _ = result.period
