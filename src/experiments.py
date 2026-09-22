from dataclasses import replace
from math import hypot
from pathlib import Path

import numpy as np

from src.analytical import (
    linear_position_x,
    linear_position_y,
    linear_velocity_x,
    linear_velocity_y,
    vacuum_flight_time,
    vacuum_max_height,
    vacuum_position,
    vacuum_range,
    vacuum_velocity,
)
from src.models import DragModel, LaunchParameters
from src.plotting import (
    plot_characteristics,
    plot_curves,
    plot_numeric_vs_analytic,
    plot_trajectories,
)
from src.solver import solve_trajectory
from src.sweeps import SweepResult, sweep_angle, sweep_drag, sweep_speed

PLOTS_DIR = Path(__file__).resolve().parent.parent / "docs" / "plots"

LINEAR_COEFF = 0.02
QUADRATIC_COEFF = 0.001

BASE = LaunchParameters(
    initial_speed=20.0,
    angle_deg=45.0,
    mass=0.2,
)

MODELS = (
    ("Без сопротивления", DragModel.NONE, 0.0),
    ("Линейное", DragModel.LINEAR, LINEAR_COEFF),
    ("Квадратичное", DragModel.QUADRATIC, QUADRATIC_COEFF),
)

ANGLES = np.arange(5.0, 85.5, 1.0)
SPEEDS = np.arange(5.0, 60.5, 2.5)
DRAG_COEFFS = np.linspace(0.0, 0.01, 21)

EXPECTED_FIGURES = (
    "01_numeric_vs_analytic.png",
    "02_angle_trajectories.png",
    "03_range_vs_angle.png",
    "04_speed_trajectories.png",
    "05_range_vs_speed.png",
    "06_drag_trajectories.png",
    "07_drag_characteristics.png",
)


def print_table(title: str, headers: list[str], rows: list[list[str]]) -> None:
    print(f"## {title}\n")
    print("| " + " | ".join(headers) + " |")
    print("|" + "---|" * len(headers))

    for row in rows:
        print("| " + " | ".join(row) + " |")

    print()


def _at(sweep: SweepResult, value: float) -> int:
    return int(np.argmin(np.abs(sweep.values - value)))


def verification() -> None:
    vacuum = solve_trajectory(BASE)
    linear = solve_trajectory(BASE, DragModel.LINEAR, LINEAR_COEFF)

    vacuum_times = np.linspace(0.0, vacuum_flight_time(BASE), 25)
    vacuum_points = np.array([vacuum_position(t, BASE) for t in vacuum_times])
    linear_times = np.linspace(0.0, linear.flight_time, 25)

    plot_numeric_vs_analytic(
        panels=[
            (
                "Без сопротивления",
                vacuum,
                vacuum_points[:, 0],
                vacuum_points[:, 1],
            ),
            (
                f"Линейное сопротивление, b = {LINEAR_COEFF} кг/с",
                linear,
                np.array([linear_position_x(t, BASE, LINEAR_COEFF) for t in linear_times]),
                np.array([linear_position_y(t, BASE, LINEAR_COEFF) for t in linear_times]),
            ),
        ],
        title="Численное решение и аналитическое, $v_0$ = 20 м/с, α = 45°",
        path=PLOTS_DIR / "01_numeric_vs_analytic.png",
    )

    impact_vx, impact_vy = vacuum_velocity(vacuum_flight_time(BASE), BASE)

    checks = [
        ("Время полёта T, с", vacuum_flight_time(BASE), vacuum.flight_time),
        ("Дальность L, м", vacuum_range(BASE), vacuum.horizontal_range),
        ("Высота H, м", vacuum_max_height(BASE), vacuum.max_height),
        ("Скорость удара, м/с", hypot(impact_vx, impact_vy), vacuum.impact_speed),
    ]

    print_table(
        "Сверка с аналитикой: без сопротивления",
        ["Величина", "Формула", "Численно", "Отн. погрешность"],
        [
            [name, f"{exact:.6f}", f"{numeric:.6f}", f"{abs(numeric / exact - 1):.1e}"]
            for name, exact, numeric in checks
        ],
    )

    x_exact = np.array([linear_position_x(t, BASE, LINEAR_COEFF) for t in linear.time])
    y_exact = np.array([linear_position_y(t, BASE, LINEAR_COEFF) for t in linear.time])
    vx_exact = np.array([linear_velocity_x(t, BASE, LINEAR_COEFF) for t in linear.time])
    vy_exact = np.array([linear_velocity_y(t, BASE, LINEAR_COEFF) for t in linear.time])

    range_by_angle = sweep_angle(BASE, ANGLES).horizontal_range
    range_by_angle_exact = BASE.initial_speed ** 2 * np.sin(2 * np.radians(ANGLES)) / BASE.gravity
    range_by_speed = sweep_speed(BASE, SPEEDS).horizontal_range
    range_by_speed_exact = SPEEDS ** 2 / BASE.gravity

    print_table(
        "Сверка с аналитикой: вдоль траектории и по сериям",
        ["Проверка", "Макс. отклонение"],
        [
            ["Линейная модель, x(t)", f"{np.max(np.abs(linear.x - x_exact)):.1e} м"],
            ["Линейная модель, y(t)", f"{np.max(np.abs(linear.y - y_exact)):.1e} м"],
            [
                "Линейная модель, v(t)",
                f"{np.max(np.hypot(linear.vx - vx_exact, linear.vy - vy_exact)):.1e} м/с",
            ],
            [
                "Вакуум, L(α) при α = 5…85° против $v_0^2 \\sin 2α / g$",
                f"{np.max(np.abs(range_by_angle / range_by_angle_exact - 1)):.1e} (отн.)",
            ],
            [
                "Вакуум, L(v₀) при v₀ = 5…60 м/с против $v_0^2 / g$",
                f"{np.max(np.abs(range_by_speed / range_by_speed_exact - 1)):.1e} (отн.)",
            ],
        ],
    )


def base_launch() -> None:
    rows = []

    for title, model, coeff in MODELS:
        result = solve_trajectory(BASE, model, coeff)

        rows.append([
            title,
            f"{result.flight_time:.3f}",
            f"{result.horizontal_range:.2f}",
            f"{result.max_height:.2f}",
            f"{result.impact_speed:.2f}",
        ])

    print_table(
        "Базовый бросок: v₀ = 20 м/с, α = 45°",
        ["Модель", "T, с", "L, м", "H, м", "Скорость удара, м/с"],
        rows,
    )


def angle_study() -> None:
    family = [15.0, 30.0, 45.0, 60.0, 75.0]

    plot_trajectories(
        panels=[
            (
                title,
                [solve_trajectory(replace(BASE, angle_deg=a), model, coeff) for a in family],
                [f"{a:.0f}°" for a in family],
            )
            for title, model, coeff in MODELS
        ],
        title="Траектории при разных углах броска, $v_0$ = 20 м/с",
        path=PLOTS_DIR / "02_angle_trajectories.png",
    )

    sweeps = [sweep_angle(BASE, ANGLES, model, coeff) for _, model, coeff in MODELS]
    exact = BASE.initial_speed ** 2 * np.sin(2 * np.radians(ANGLES)) / BASE.gravity

    plot_curves(
        curves=[
            (sweep.values, sweep.horizontal_range, title)
            for sweep, (title, _, _) in zip(sweeps, MODELS)
        ],
        xlabel="Угол броска α, °",
        ylabel="Дальность L, м",
        title="Дальность от угла броска, $v_0$ = 20 м/с",
        path=PLOTS_DIR / "03_range_vs_angle.png",
        reference=(ANGLES, exact, "формула $v_0^2 \\sin 2\\alpha / g$"),
    )

    rows = [
        [f"{a:.0f}°", f"{exact[_at(sweeps[0], a)]:.2f}"]
        + [f"{sweep.horizontal_range[_at(sweep, a)]:.2f}" for sweep in sweeps]
        for a in family
    ]
    rows.append(
        ["max L при α", "45°"]
        + [f"{sweep.values[np.argmax(sweep.horizontal_range)]:.0f}°" for sweep in sweeps]
    )

    print_table(
        "Дальность L (м) от угла броска",
        ["α", "Формула", "Без сопротивления", "Линейное", "Квадратичное"],
        rows,
    )


def speed_study() -> None:
    family = [10.0, 20.0, 30.0, 40.0, 60.0]

    plot_trajectories(
        panels=[
            (
                title,
                [solve_trajectory(replace(BASE, initial_speed=v), model, coeff) for v in family],
                [f"{v:.0f} м/с" for v in family],
            )
            for title, model, coeff in MODELS[1:]
        ],
        title="Траектории при разной начальной скорости, α = 45°",
        path=PLOTS_DIR / "04_speed_trajectories.png",
    )

    linear = sweep_speed(BASE, SPEEDS, DragModel.LINEAR, LINEAR_COEFF)
    quadratic = sweep_speed(BASE, SPEEDS, DragModel.QUADRATIC, QUADRATIC_COEFF)
    exact = SPEEDS ** 2 / BASE.gravity

    plot_curves(
        curves=[
            (SPEEDS, linear.horizontal_range, "Линейное"),
            (SPEEDS, quadratic.horizontal_range, "Квадратичное"),
        ],
        xlabel="Начальная скорость $v_0$, м/с",
        ylabel="Дальность L, м",
        title="Дальность от начальной скорости, α = 45°",
        path=PLOTS_DIR / "05_range_vs_speed.png",
        reference=(SPEEDS, exact, "без сопротивления: $v_0^2 / g$"),
    )

    rows = []

    for v in family:
        i = _at(linear, v)
        rows.append([
            f"{v:.0f}",
            f"{exact[i]:.2f}",
            f"{linear.horizontal_range[i]:.2f}",
            f"{quadratic.horizontal_range[i]:.2f}",
            f"{100 * (1 - linear.horizontal_range[i] / exact[i]):.1f}",
            f"{100 * (1 - quadratic.horizontal_range[i] / exact[i]):.1f}",
        ])

    print_table(
        "Дальность от начальной скорости, α = 45°",
        ["v₀, м/с", "v₀²/g, м", "L лин., м", "L квадр., м", "Потеря лин., %", "Потеря квадр., %"],
        rows,
    )


def drag_study() -> None:
    family = [0.0, 0.001, 0.002, 0.005, 0.01]

    plot_trajectories(
        panels=[
            (
                "Квадратичное сопротивление",
                [solve_trajectory(BASE, DragModel.QUADRATIC, c) for c in family],
                [f"c = {c:g} кг/м" for c in family],
            )
        ],
        title="Траектории при разном c, $v_0$ = 20 м/с, α = 45°",
        path=PLOTS_DIR / "06_drag_trajectories.png",
    )

    sweep = sweep_drag(BASE, DRAG_COEFFS)

    plot_characteristics(
        sweep=sweep,
        parameter_label="Коэффициент c, кг/м",
        title="Характеристики полёта от коэффициента сопротивления",
        path=PLOTS_DIR / "07_drag_characteristics.png",
    )

    weight = BASE.mass * BASE.gravity
    rows = []

    for c in family:
        i = _at(sweep, c)
        rows.append([
            f"{c:g}",
            f"{c * BASE.initial_speed ** 2 / weight:.2f}",
            f"{sweep.flight_time[i]:.3f}",
            f"{sweep.horizontal_range[i]:.2f}",
            f"{sweep.max_height[i]:.2f}",
            f"{sweep.impact_speed[i]:.2f}",
        ])

    print_table(
        "Квадратичное сопротивление: характеристики от c",
        ["c, кг/м", "cv₀²/mg", "T, с", "L, м", "H, м", "Скорость удара, м/с"],
        rows,
    )


def main() -> None:
    verification()
    base_launch()
    angle_study()
    speed_study()
    drag_study()


if __name__ == "__main__":
    main()
