"""Independent theoretical results used to check the numerical solution."""

from math import cos, exp, pi, radians, sin, sqrt

from scipy.special import ellipk

from src.models import PendulumParameters


def small_angle_period(parameters: PendulumParameters) -> float:
    """T₀ = 2π sqrt(I / Mgd), valid when |angle| is much smaller than 1 rad."""
    return 2 * pi * sqrt(parameters.moment_of_inertia / parameters.restoring_torque_scale)


def exact_frictionless_period(parameters: PendulumParameters, amplitude_deg: float) -> float:
    """Exact nonlinear period for release from rest at an amplitude below 180°."""
    if not 0 <= amplitude_deg < 180:
        raise ValueError("Амплитуда должна лежать в диапазоне [0°, 180°)")
    modulus_squared = sin(radians(amplitude_deg) / 2) ** 2
    return 4 * sqrt(parameters.moment_of_inertia / parameters.restoring_torque_scale) * float(
        ellipk(modulus_squared)
    )


def small_angle_damped_period(parameters: PendulumParameters) -> float:
    """Linearized damped period; meaningful only in the underdamped regime."""
    omega_0_squared = parameters.restoring_torque_scale / parameters.moment_of_inertia
    gamma = parameters.damping / (2 * parameters.moment_of_inertia)
    if gamma**2 >= omega_0_squared:
        raise ValueError("При таком трении малые колебания не являются затухающими колебаниями")
    return 2 * pi / sqrt(omega_0_squared - gamma**2)


def small_angle_damped_angle(
    time: float, parameters: PendulumParameters, initial_angle_deg: float
) -> float:
    """Linear approximation for release from rest, in radians."""
    if time < 0:
        raise ValueError("Время не может быть отрицательным")
    omega_0_squared = parameters.restoring_torque_scale / parameters.moment_of_inertia
    gamma = parameters.damping / (2 * parameters.moment_of_inertia)
    if gamma**2 >= omega_0_squared:
        raise ValueError("Аналитическая формула задана для слабого трения")
    omega_d = sqrt(omega_0_squared - gamma**2)
    return radians(initial_angle_deg) * exp(-gamma * time) * (
        cos(omega_d * time) + gamma / omega_d * sin(omega_d * time)
    )
