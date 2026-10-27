from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.analytical import exact_frictionless_period, small_angle_damped_period, small_angle_period
from src.models import InitialConditions, PendulumParameters
from src.solver import solve_pendulum


def main() -> None:
    parser = argparse.ArgumentParser(description="Эксперименты с физическим маятником")
    parser.add_argument("--output-dir", type=Path, default=Path("docs/figures"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    base = PendulumParameters(
        rod_length=0.8, rod_mass=0.2, sphere_radius=0.08, sphere_mass=0.5
    )
    damped = PendulumParameters(
        rod_length=0.8, rod_mass=0.2, sphere_radius=0.08,
        sphere_mass=0.5, damping=0.015
    )
    amplitudes = [5, 15, 30, 60, 90, 120, 150]
    numerical_periods = []
    theoretical_periods = []
    for amplitude in amplitudes:
        result = solve_pendulum(base, InitialConditions(amplitude))
        numerical_periods.append(result.period)
        theoretical_periods.append(exact_frictionless_period(base, amplitude))

    print(f"Момент инерции: {base.moment_of_inertia:.6f} кг·м²")
    print(f"Расстояние до центра масс: {base.center_of_mass_distance:.6f} м")
    print(f"Период при малом угле: {small_angle_period(base):.6f} с")
    print("Амплитуда | Численно | Теория | Отклонение")
    for amplitude, numerical, theoretical in zip(
        amplitudes, numerical_periods, theoretical_periods, strict=True
    ):
        print(
            f"{amplitude:>8}° | {numerical:>7.4f} с | {theoretical:>6.4f} с | "
            f"{abs(numerical / theoretical - 1):.2e}"
        )

    fig, ax = plt.subplots()
    ax.plot(amplitudes, numerical_periods, "o-", label="Численное решение")
    ax.plot(amplitudes, theoretical_periods, "--", label="Точная формула без трения")
    ax.axhline(small_angle_period(base), color="gray", linestyle=":", label="Малые углы")
    ax.set(xlabel="Начальная амплитуда, градусы", ylabel="Период, с",
           title="Период растёт с амплитудой")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.output_dir / "period_vs_amplitude.png", dpi=160)
    plt.close(fig)

    initial = InitialConditions(60)
    ideal = solve_pendulum(base, initial)
    friction = solve_pendulum(damped, initial)
    print(
        "Максимальный относительный дрейф энергии без трения: "
        f"{ideal.relative_energy_drift:.2e}"
    )
    print(f"Период при слабом трении, численно: {friction.period:.6f} с")
    print(
        "Период при слабом трении, теория малых углов: "
        f"{small_angle_damped_period(damped):.6f} с (для 60° неприменима)"
    )
    small_friction = solve_pendulum(damped, InitialConditions(1))
    print(
        "При амплитуде 1°: период с трением численно/по теории малых углов: "
        f"{small_friction.period:.6f} / {small_angle_damped_period(damped):.6f} с"
    )
    dissipated = damped.damping * np.trapezoid(friction.angular_velocity**2, friction.time)
    print(
        "Потеря энергии при 60°: по траектории / по интегралу b·ω²: "
        f"{friction.energy[0] - friction.energy[-1]:.6f} / {dissipated:.6f} Дж"
    )

    fig, ax = plt.subplots()
    ax.plot(ideal.time, np.rad2deg(ideal.angle), label="Без трения")
    ax.plot(friction.time, np.rad2deg(friction.angle), label="С трением")
    ax.set(xlabel="Время, с", ylabel="Угол, градусы", title="Свободные колебания")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.output_dir / "angle_vs_time.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots()
    ax.plot(ideal.time, ideal.energy, label="Без трения")
    ax.plot(friction.time, friction.energy, label="С трением")
    ax.set(xlabel="Время, с", ylabel="Полная энергия, Дж", title="Энергия маятника")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.output_dir / "energy_vs_time.png", dpi=160)
    plt.close(fig)
    print(f"Графики сохранены в {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
