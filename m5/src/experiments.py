from dataclasses import dataclass, replace
from math import isfinite, sqrt
from typing import Mapping

import numpy as np

from src.analytical import exact_frictionless_period, small_angle_damped_period, small_angle_period
from src.models import InitialConditions, PendulumParameters
from src.solver import PendulumResult, solve_pendulum


GRAVITY = 9.81
DAMPING_LIMIT_FRACTION = 0.1
LABORATORY_LIMITS = {
    "rod_length": 3.0,
    "rod_mass": 50.0,
    "sphere_radius": 0.5,
    "sphere_mass": 100.0,
}
AMPLITUDES = (5.0, 15.0, 30.0, 60.0, 90.0, 120.0, 150.0)
COMPARISON_ANGLE = 60.0
SMALL_ANGLE = 1.0
SAMPLES = 2001
RTOL = 1e-10
ATOL = 1e-12


def _positive_number(fields: Mapping[str, str], name: str, label: str) -> float:
    raw = fields.get(name, "").strip().replace(",", ".")
    try:
        value = float(raw)
    except ValueError as error:
        raise ValueError(f"{label}: введите число") from error
    if not isfinite(value) or value <= 0:
        raise ValueError(f"{label}: нужно положительное конечное число")
    maximum = LABORATORY_LIMITS.get(name)
    if maximum is not None and value > maximum:
        raise ValueError(f"{label}: для лабораторной установки значение не должно превышать {maximum:g}")
    return value


@dataclass(frozen=True)
class ExperimentConfig:
    parameters: PendulumParameters
    damping: float
    amplitudes: tuple[float, ...]
    comparison_angle: float
    small_angle: float
    duration: float
    samples: int
    max_step: float
    rtol: float
    atol: float

    @classmethod
    def from_fields(cls, fields: Mapping[str, str]) -> "ExperimentConfig":
        parameters = PendulumParameters(
            rod_length=_positive_number(fields, "rod_length", "Длина стержня"),
            rod_mass=_positive_number(fields, "rod_mass", "Масса стержня"),
            sphere_radius=_positive_number(fields, "sphere_radius", "Радиус шара"),
            sphere_mass=_positive_number(fields, "sphere_mass", "Масса шара"),
            gravity=GRAVITY,
        )
        damping = _positive_number(fields, "damping", "Коэффициент трения")
        critical = 2 * sqrt(parameters.moment_of_inertia) * sqrt(
            parameters.restoring_torque_scale
        )
        damping_limit = DAMPING_LIMIT_FRACTION * critical
        if damping > damping_limit:
            raise ValueError(
                f"Для режима малого трения коэффициент должен быть не больше "
                f"{damping_limit:.6g} для выбранного маятника"
            )

        damped = replace(parameters, damping=damping)
        longest_period = max(
            exact_frictionless_period(parameters, max(AMPLITUDES)),
            small_angle_damped_period(damped),
        )
        if not isfinite(longest_period) or longest_period <= 0:
            raise ValueError("Для этих параметров период не удалось вычислить")
        duration = 6 * longest_period
        max_step = small_angle_period(parameters) / 100
        if not isfinite(duration) or not isfinite(max_step) or max_step <= 0:
            raise ValueError("Параметры выходят за вычислимый диапазон")

        return cls(
            parameters=parameters,
            damping=damping,
            amplitudes=AMPLITUDES,
            comparison_angle=COMPARISON_ANGLE,
            small_angle=SMALL_ANGLE,
            duration=duration,
            samples=SAMPLES,
            max_step=max_step,
            rtol=RTOL,
            atol=ATOL,
        )


@dataclass(frozen=True)
class ExperimentResult:
    config: ExperimentConfig
    numerical_periods: tuple[float, ...]
    theoretical_periods: tuple[float, ...]
    ideal: PendulumResult
    friction: PendulumResult
    small_friction: PendulumResult
    dissipated_energy: float


def run_experiment(config: ExperimentConfig) -> ExperimentResult:
    base = config.parameters
    damped = replace(base, damping=config.damping)
    options = {
        "duration": config.duration,
        "samples": config.samples,
        "max_step": config.max_step,
        "rtol": config.rtol,
        "atol": config.atol,
    }
    numerical_periods = []
    theoretical_periods = []
    for amplitude in config.amplitudes:
        result = solve_pendulum(base, InitialConditions(amplitude), **options)
        numerical_periods.append(result.period)
        theoretical_periods.append(exact_frictionless_period(base, amplitude))

    initial = InitialConditions(config.comparison_angle)
    ideal = solve_pendulum(base, initial, **options)
    friction = solve_pendulum(damped, initial, **options)
    small_friction = solve_pendulum(
        damped, InitialConditions(config.small_angle), **options
    )
    dissipated_energy = config.damping * float(
        np.trapezoid(friction.angular_velocity**2, friction.time)
    )
    return ExperimentResult(
        config=config,
        numerical_periods=tuple(numerical_periods),
        theoretical_periods=tuple(theoretical_periods),
        ideal=ideal,
        friction=friction,
        small_friction=small_friction,
        dissipated_energy=dissipated_energy,
    )
