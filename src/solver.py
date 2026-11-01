from dataclasses import dataclass
from src.models import (
    DragModel,
    LaunchParameters,
    motion_equations,
)
import numpy as np
from scipy.integrate import solve_ivp

@dataclass(frozen=True)
class TrajectoryResult:
    time: np.ndarray
    x: np.ndarray
    y: np.ndarray
    vx: np.ndarray
    vy: np.ndarray
    apex_height: float | None = None

    @property
    def flight_time(self) -> float:
        return float(self.time[-1])

    @property
    def horizontal_range(self) -> float:
        return float(self.x[-1] - self.x[0])

    @property
    def max_height(self) -> float:
        if self.apex_height is not None:
            return self.apex_height

        return float(np.max(self.y))

    @property
    def impact_speed(self) -> float:
        final_vx = self.vx[-1]
        final_vy = self.vy[-1]

        return float(np.hypot(final_vx, final_vy))

def create_initial_state(
        parameters: LaunchParameters,
) -> np.ndarray:
    vx0, vy0 = parameters.initial_velocity()

    initial_state = np.array(
        [
            0.0,
            parameters.initial_height,
            vx0,
            vy0,
        ],
        dtype=float,
    )

    return initial_state

def ground_event(
        time: float,
        state: np.ndarray,
        parameters: LaunchParameters,
        model: DragModel,
        drag_coeff: float,
) -> float:
    return float(state[1])


ground_event.terminal = True
ground_event.direction = -1
def apex_event(
        time: float,
        state: np.ndarray,
        parameters: LaunchParameters,
        model: DragModel,
        drag_coeff: float,
) -> float:
    return float(state[3])


apex_event.terminal = False
apex_event.direction = -1


def solve_trajectory(
        parameters: LaunchParameters,
        model: DragModel = DragModel.NONE,
        drag_coeff: float = 0.0,
        max_step: float = 0.02,
        rtol: float = 1e-9,
        atol: float = 1e-11,
) -> TrajectoryResult:
    initial_state = create_initial_state(parameters)

    solution = solve_ivp(
        fun=motion_equations,
        t_span=(0.0, 100.0),
        y0=initial_state,
        args=(parameters, model, drag_coeff),
        events=(ground_event, apex_event),
        max_step=max_step,
        rtol=rtol,
        atol=atol,
    )
    if len(solution.y_events[1]) > 0:
        apex_height = float(solution.y_events[1][0][1])
    else:
        apex_height = None
    return TrajectoryResult(
        time=solution.t,
        x=solution.y[0],
        y=solution.y[1],
        vx=solution.y[2],
        vy=solution.y[3],
        apex_height=apex_height,
    )
