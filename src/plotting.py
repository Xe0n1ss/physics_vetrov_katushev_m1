from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from src.solver import TrajectoryResult
from src.sweeps import SweepResult

Curve = tuple[np.ndarray, np.ndarray, str]


def _save(figure: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def _trajectory_axes(axes: plt.Axes, title: str) -> None:
    axes.set_xlabel("x, м")
    axes.set_ylabel("y, м")
    axes.set_title(title)
    axes.grid(True, alpha=0.3)
    axes.axhline(0.0, color="black", linewidth=0.8)
    axes.legend(fontsize=8)


def plot_trajectories(
        panels: list[tuple[str, list[TrajectoryResult], list[str]]],
        title: str,
        path: Path,
) -> None:
    """Семейства траекторий y(x): одна панель на каждый элемент panels."""
    figure, axes_row = plt.subplots(
        1,
        len(panels),
        figsize=(5 * len(panels) + 2, 4.5),
        squeeze=False,
    )

    for axes, (panel_title, results, labels) in zip(axes_row[0], panels):
        for result, label in zip(results, labels):
            axes.plot(result.x, result.y, label=label)

        _trajectory_axes(axes, panel_title)

    figure.suptitle(title)
    _save(figure, path)


def plot_numeric_vs_analytic(
        panels: list[tuple[str, TrajectoryResult, np.ndarray, np.ndarray]],
        title: str,
        path: Path,
) -> None:
    """Численная траектория линией, аналитическая — точками поверх неё."""
    figure, axes_row = plt.subplots(
        1,
        len(panels),
        figsize=(5 * len(panels) + 2, 4.5),
        squeeze=False,
    )

    for axes, (panel_title, numeric, x, y) in zip(axes_row[0], panels):
        axes.plot(numeric.x, numeric.y, label="численно")
        axes.plot(x, y, "o", markersize=4, fillstyle="none", label="аналитически")

        _trajectory_axes(axes, panel_title)

    figure.suptitle(title)
    _save(figure, path)


def plot_curves(
        curves: list[Curve],
        xlabel: str,
        ylabel: str,
        title: str,
        path: Path,
        reference: Curve | None = None,
) -> None:
    """Несколько зависимостей на одних осях; reference — аналитическая
    кривая, рисуется чёрным пунктиром."""
    figure, axes = plt.subplots(figsize=(8, 5))

    for x, y, label in curves:
        axes.plot(x, y, label=label)

    if reference is not None:
        x, y, label = reference
        axes.plot(x, y, "k--", alpha=0.7, label=label)

    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.set_title(title)
    axes.grid(True, alpha=0.3)
    axes.legend()

    _save(figure, path)


def plot_characteristics(
        sweep: SweepResult,
        parameter_label: str,
        title: str,
        path: Path,
) -> None:
    """Четыре характеристики полёта в зависимости от параметра серии."""
    figure, axes_grid = plt.subplots(2, 2, figsize=(11, 8))

    panels = [
        (sweep.flight_time, "Время полёта T, с"),
        (sweep.horizontal_range, "Дальность L, м"),
        (sweep.max_height, "Максимальная высота H, м"),
        (sweep.impact_speed, "Скорость удара, м/с"),
    ]

    for axes, (data, ylabel) in zip(axes_grid.flat, panels):
        axes.plot(sweep.values, data, marker="o", markersize=3)
        axes.set_xlabel(parameter_label)
        axes.set_ylabel(ylabel)
        axes.grid(True, alpha=0.3)

    figure.suptitle(title)
    figure.tight_layout()
    _save(figure, path)
