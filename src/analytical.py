from src.models import LaunchParameters
from math import exp, sqrt

def vacuum_position(
        time: float,
        parameters: LaunchParameters,
) -> tuple[float, float]:
    vx0, vy0 = parameters.initial_velocity()

    x = vx0 * time

    y = (
        parameters.initial_height
        + vy0 * time
        - parameters.gravity * time ** 2 / 2
    )

    return x, y

def vacuum_velocity(
        time: float,
        parameters: LaunchParameters,
) -> tuple[float, float]:
    vx0, vy0 = parameters.initial_velocity()

    vx = vx0
    vy = vy0 - parameters.gravity * time

    return vx, vy

def vacuum_time_to_apex(
        parameters: LaunchParameters,
) -> float:
    _, vy0 = parameters.initial_velocity()

    time_to_apex = vy0 / parameters.gravity

    return time_to_apex


def vacuum_max_height(
        parameters: LaunchParameters,
) -> float:
    time_to_apex = vacuum_time_to_apex(parameters)

    _, max_height = vacuum_position(
        time=time_to_apex,
        parameters=parameters,
    )

    return max_height

def vacuum_flight_time(
        parameters: LaunchParameters,
) -> float:
    _, vy0 = parameters.initial_velocity()

    discriminant = (
        vy0 ** 2
        + 2
        * parameters.gravity
        * parameters.initial_height
    )

    flight_time = (
        vy0 + sqrt(discriminant)
    ) / parameters.gravity

    return flight_time


def vacuum_range(
        parameters: LaunchParameters,
) -> float:
    flight_time = vacuum_flight_time(parameters)

    horizontal_range, _ = vacuum_position(
        time=flight_time,
        parameters=parameters,
    )

    return horizontal_range

def linear_velocity_x(
        time: float,
        parameters: LaunchParameters,
        drag_coeff: float,
) -> float:
    vx0, _ = parameters.initial_velocity()

    vx = vx0 * exp(
        -drag_coeff * time / parameters.mass
    )

    return vx


def linear_velocity_y(
        time: float,
        parameters: LaunchParameters,
        drag_coeff: float,
) -> float:
    _, vy0 = parameters.initial_velocity()

    if drag_coeff == 0:
        return vy0 - parameters.gravity * time

    gamma = drag_coeff / parameters.mass

    vy = (
        (vy0 + parameters.gravity / gamma)
        * exp(-gamma * time)
        - parameters.gravity / gamma
    )

    return vy


def linear_position_x(
        time: float,
        parameters: LaunchParameters,
        drag_coeff: float,
) -> float:
    vx0, _ = parameters.initial_velocity()

    if drag_coeff == 0:
        return vx0 * time

    gamma = drag_coeff / parameters.mass

    x = vx0 * (1 - exp(-gamma * time)) / gamma

    return x

def linear_position_y(
        time: float,
        parameters: LaunchParameters,
        drag_coeff: float,
) -> float:
    _, vy0 = parameters.initial_velocity()

    if drag_coeff == 0:
        _, y = vacuum_position(time, parameters)
        return y

    gamma = drag_coeff / parameters.mass

    y = (
        parameters.initial_height
        + (vy0 + parameters.gravity / gamma)
        * (1 - exp(-gamma * time))
        / gamma
        - parameters.gravity * time / gamma
    )

    return y