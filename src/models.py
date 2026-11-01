from enum import Enum
from dataclasses import dataclass
from math import cos, hypot, radians, sin

class DragModel(Enum):
    NONE = "none"
    LINEAR = "linear"
    QUADRATIC = "quadratic"

@dataclass(frozen = True)
class LaunchParameters:
    initial_speed: float
    angle_deg: float
    mass: float
    initial_height: float = 0.0
    gravity: float = 9.81

    def __post_init__(self) -> None:
        if self.initial_speed <= 0:raise ValueError("Начальная скорость должна быть больше нуля")

        if not 0 <= self.angle_deg <= 90:
            raise ValueError("Угол должен находиться в диапазоне от 0 до 90 градусов")

        if self.mass <= 0:
            raise ValueError("Масса должна быть больше нуля")

        if self.initial_height < 0:
            raise ValueError("Начальная высота не может быть отрицательной")

    def initial_velocity(self) -> tuple[float,float]:
        angle_rad = radians(self.angle_deg)

        vx = self.initial_speed * cos(angle_rad)
        vy = self.initial_speed * sin(angle_rad)

        return vx,vy

def calculate_acceleration(
        vx: float,
        vy: float,
        parameters: LaunchParameters,
        model: DragModel,
        drag_coeff: float = 0.0,
) -> tuple[float,float]:
    if model is DragModel.NONE:
        ax = 0.0
        ay = -parameters.gravity
        return ax,ay

    elif model is DragModel.LINEAR:
        drag_factor = drag_coeff / parameters.mass
        ax = -drag_factor * vx
        ay = -parameters.gravity - drag_factor * vy
        return ax,ay

    elif model is DragModel.QUADRATIC:
        speed = hypot(vx,vy)
        drag_factor = drag_coeff * speed / parameters.mass
        ax = -drag_factor * vx
        ay = -parameters.gravity - drag_factor * vy
        return ax,ay

def motion_equations(
        time: float,
        state: tuple[float, float, float, float],
        parameters: LaunchParameters,
        model: DragModel,
        drag_coeff: float = 0.0,
) -> tuple[float, float, float, float]:
    x, y, vx, vy = state

    ax, ay = calculate_acceleration(
        vx=vx,
        vy=vy,
        parameters=parameters,
        model=model,
        drag_coeff=drag_coeff,
    )

    dx_dt = vx
    dy_dt = vy
    dvx_dt = ax
    dvy_dt = ay

    return dx_dt, dy_dt, dvx_dt, dvy_dt


