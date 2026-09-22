from src.models import DragModel, LaunchParameters
from src.solver import TrajectoryResult, solve_trajectory


def print_result(
        title: str,
        result: TrajectoryResult,
) -> None:
    print(title)
    print(f"  Время полёта:       {result.flight_time:.3f} с")
    print(f"  Дальность:          {result.horizontal_range:.3f} м")
    print(f"  Максимальная высота: {result.max_height:.3f} м")
    print(f"  Скорость падения:   {result.impact_speed:.3f} м/с")
    print()


def main() -> None:
    parameters = LaunchParameters(
        initial_speed=20.0,
        angle_deg=45.0,
        mass=0.2,
    )

    vacuum_result = solve_trajectory(
        parameters=parameters,
        model=DragModel.NONE,
    )

    linear_result = solve_trajectory(
        parameters=parameters,
        model=DragModel.LINEAR,
        drag_coeff=0.02,
    )

    quadratic_result = solve_trajectory(
        parameters=parameters,
        model=DragModel.QUADRATIC,
        drag_coeff=0.001,
    )

    print_result(
        title="Без сопротивления",
        result=vacuum_result,
    )

    print_result(
        title="Линейное сопротивление",
        result=linear_result,
    )

    print_result(
        title="Квадратичное сопротивление",
        result=quadratic_result,
    )


if __name__ == "__main__":
    main()