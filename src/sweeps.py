from collections.abc import Iterable
from dataclasses import dataclass, replace

import numpy as np

from src.models import DragModel, LaunchParameters
from src.solver import TrajectoryResult, solve_trajectory


@dataclass(frozen=True)
class SweepResult:
    """Характеристики полёта по серии расчётов: i-й элемент каждого
    массива относится к значению параметра values[i]."""

    values: np.ndarray
    flight_time: np.ndarray
    horizontal_range: np.ndarray
    max_height: np.ndarray
    impact_speed: np.ndarray


def _collect(
        values: np.ndarray,
        results: list[TrajectoryResult],
) -> SweepResult:
    return SweepResult(
        values=values,
        flight_time=np.array([r.flight_time for r in results]),
        horizontal_range=np.array([r.horizontal_range for r in results]),
        max_height=np.array([r.max_height for r in results]),
        impact_speed=np.array([r.impact_speed for r in results]),
    )


def sweep_angle(
        parameters: LaunchParameters,
        angles: Iterable[float],
        model: DragModel = DragModel.NONE,
        drag_coeff: float = 0.0,
) -> SweepResult:
    angles = np.asarray(angles, dtype=float)

    results = [
        solve_trajectory(
            parameters=replace(parameters, angle_deg=float(angle)),
            model=model,
            drag_coeff=drag_coeff,
        )
        for angle in angles
    ]

    return _collect(angles, results)


def sweep_speed(
        parameters: LaunchParameters,
        speeds: Iterable[float],
        model: DragModel = DragModel.NONE,
        drag_coeff: float = 0.0,
) -> SweepResult:
    speeds = np.asarray(speeds, dtype=float)

    results = [
        solve_trajectory(
            parameters=replace(parameters, initial_speed=float(speed)),
            model=model,
            drag_coeff=drag_coeff,
        )
        for speed in speeds
    ]

    return _collect(speeds, results)


def sweep_drag(
        parameters: LaunchParameters,
        drag_coeffs: Iterable[float],
        model: DragModel = DragModel.QUADRATIC,
) -> SweepResult:
    drag_coeffs = np.asarray(drag_coeffs, dtype=float)

    results = [
        solve_trajectory(
            parameters=parameters,
            model=model,
            drag_coeff=float(drag_coeff),
        )
        for drag_coeff in drag_coeffs
    ]

    return _collect(drag_coeffs, results)
