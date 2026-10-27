"""Numerical integration and period measurement for the physical pendulum."""

from dataclasses import dataclass
from math import isfinite

import numpy as np
from scipy.integrate import solve_ivp

from src.models import InitialConditions, PendulumParameters, motion_equations


def _downward_crossing(time: float, state: np.ndarray, parameters: PendulumParameters) -> float:
    return float(state[0])


_downward_crossing.direction = -1
_downward_crossing.terminal = False


@dataclass(frozen=True)
class PendulumResult:
    time: np.ndarray
    angle: np.ndarray  # rad
    angular_velocity: np.ndarray  # rad/s
    energy: np.ndarray  # J
    downward_crossings: np.ndarray  # s; one crossing per full oscillation

    @property
    def period(self) -> float:
        if len(self.downward_crossings) < 2:
            raise ValueError("Для измерения периода нужны два прохода через нижнюю точку")
        return float(self.downward_crossings[1] - self.downward_crossings[0])

    @property
    def relative_energy_drift(self) -> float:
        initial = float(self.energy[0])
        if initial == 0:
            raise ValueError("При нулевой начальной энергии относительное изменение не определено")
        return float(np.max(np.abs(self.energy - initial)) / initial)


def solve_pendulum(
    parameters: PendulumParameters,
    initial: InitialConditions,
    duration: float = 20.0,
    samples: int = 2001,
    max_step: float = 0.02,
    rtol: float = 1e-10,
    atol: float = 1e-12,
) -> PendulumResult:
    if not isfinite(duration) or duration <= 0:
        raise ValueError("Длительность расчёта должна быть положительной и конечной")
    if samples < 2:
        raise ValueError("Нужно как минимум две точки графика")
    if not isfinite(max_step) or max_step <= 0:
        raise ValueError("Шаг интегрирования должен быть положительным и конечным")

    times = np.linspace(0.0, duration, samples)
    solution = solve_ivp(
        motion_equations,
        (0.0, duration),
        (initial.angle_rad, initial.angular_velocity),
        args=(parameters,),
        events=_downward_crossing,
        t_eval=times,
        max_step=max_step,
        rtol=rtol,
        atol=atol,
    )
    if not solution.success:
        raise RuntimeError(f"Численное интегрирование не удалось: {solution.message}")
    angle, angular_velocity = solution.y
    energy = (
        0.5 * parameters.moment_of_inertia * angular_velocity**2
        + parameters.restoring_torque_scale * (1 - np.cos(angle))
    )
    return PendulumResult(
        time=solution.t,
        angle=angle,
        angular_velocity=angular_velocity,
        energy=energy,
        downward_crossings=solution.t_events[0],
    )
