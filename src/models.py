from dataclasses import dataclass
from math import cos, isfinite, radians, sin


@dataclass(frozen=True)
class PendulumParameters:
    rod_length: float
    rod_mass: float
    sphere_radius: float
    sphere_mass: float
    gravity: float = 9.81
    damping: float = 0.0

    def __post_init__(self) -> None:
        positive = (
            self.rod_length,
            self.rod_mass,
            self.sphere_radius,
            self.sphere_mass,
            self.gravity,
        )
        if not all(isfinite(value) and value > 0 for value in positive):
            raise ValueError("Длины, массы и ускорение свободного падения должны быть положительными и конечными")
        if not isfinite(self.damping) or self.damping < 0:
            raise ValueError("Коэффициент трения должен быть неотрицательным и конечным")

    @property
    def total_mass(self) -> float:
        return self.rod_mass + self.sphere_mass

    @property
    def sphere_center_distance(self) -> float:
        return self.rod_length + self.sphere_radius

    @property
    def center_of_mass_distance(self) -> float:
        return (
            self.rod_mass * self.rod_length / 2
            + self.sphere_mass * self.sphere_center_distance
        ) / self.total_mass

    @property
    def moment_of_inertia(self) -> float:
        rod = self.rod_mass * self.rod_length**2 / 3
        sphere = (
            2 * self.sphere_mass * self.sphere_radius**2 / 5
            + self.sphere_mass * self.sphere_center_distance**2
        )
        return rod + sphere

    @property
    def restoring_torque_scale(self) -> float:
        return self.total_mass * self.gravity * self.center_of_mass_distance

    def energy(self, angle_rad: float, angular_velocity: float) -> float:
        return (
            0.5 * self.moment_of_inertia * angular_velocity**2
            + self.restoring_torque_scale * (1 - cos(angle_rad))
        )


@dataclass(frozen=True)
class InitialConditions:
    angle_deg: float
    angular_velocity: float = 0.0

    def __post_init__(self) -> None:
        if not isfinite(self.angle_deg) or not -180 < self.angle_deg < 180:
            raise ValueError("Начальный угол должен быть конечным и лежать между -180° и 180°")
        if not isfinite(self.angular_velocity):
            raise ValueError("Начальная угловая скорость должна быть конечной")

    @property
    def angle_rad(self) -> float:
        return radians(self.angle_deg)


def motion_equations(
    time: float,
    state: tuple[float, float],
    parameters: PendulumParameters,
) -> tuple[float, float]:
    angle, angular_velocity = state
    angular_acceleration = (
        -parameters.restoring_torque_scale * sin(angle)
        - parameters.damping * angular_velocity
    ) / parameters.moment_of_inertia
    return angular_velocity, angular_acceleration
