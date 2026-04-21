"""Estimate-driven formation control and static 3D scene geometry."""

from __future__ import annotations

from dataclasses import dataclass
from math import atan2, cos, sin
from typing import Protocol

import numpy as np

import sim_env
from agent_classes import AgentClass, normalize_agent_class
from robot import Robot


def wrap_to_pi(angle: float) -> float:
    """Wrap an angle to [-pi, pi)."""

    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


def default_render_altitude(agent_class: AgentClass | str) -> float:
    """Return the nominal render altitude for an agent class."""

    agent_class = normalize_agent_class(agent_class)
    if agent_class == AgentClass.CLASS_A_UGV:
        return 0.10
    return 1.60


@dataclass(frozen=True)
class FormationTarget3D:
    """Desired 3D formation slot for one agent."""

    agent_id: int
    position_xyz: tuple[float, float, float]
    label: str

    def to_dict(self) -> dict[str, object]:
        return {
            "agent_id": int(self.agent_id),
            "position_xyz": [float(value) for value in self.position_xyz],
            "label": self.label,
        }


@dataclass(frozen=True)
class StaticObstacle3D:
    """Static obstacle geometry rendered in the scene."""

    obstacle_id: str
    primitive: str
    position_xyz: tuple[float, float, float]
    half_extents_xyz: tuple[float, float, float] | None = None
    radius: float | None = None
    height: float | None = None
    color_rgba: tuple[float, float, float, float] = (0.40, 0.45, 0.52, 0.95)

    def to_dict(self) -> dict[str, object]:
        return {
            "obstacle_id": self.obstacle_id,
            "primitive": self.primitive,
            "position_xyz": [float(value) for value in self.position_xyz],
            "half_extents_xyz": (
                None
                if self.half_extents_xyz is None
                else [float(value) for value in self.half_extents_xyz]
            ),
            "radius": None if self.radius is None else float(self.radius),
            "height": None if self.height is None else float(self.height),
            "color_rgba": [float(value) for value in self.color_rgba],
        }


class SafetyFilter(Protocol):
    """Hook for future CBF-style safety filtering."""

    def filter_control(
        self,
        *,
        agent_id: int,
        nominal_input: np.ndarray,
        estimated_position_xy: np.ndarray,
        target_position_xyz: np.ndarray,
        current_theta: float,
        obstacles: list[StaticObstacle3D],
        robot: Robot,
        dt: float,
    ) -> np.ndarray: ...


class IdentitySafetyFilter:
    """Pass-through placeholder until a CBF is added."""

    def filter_control(
        self,
        *,
        agent_id: int,
        nominal_input: np.ndarray,
        estimated_position_xy: np.ndarray,
        target_position_xyz: np.ndarray,
        current_theta: float,
        obstacles: list[StaticObstacle3D],
        robot: Robot,
        dt: float,
    ) -> np.ndarray:
        return np.asarray(nominal_input, dtype=float)


@dataclass(frozen=True)
class FormationControlGains:
    linear_gain: float = 0.75
    angular_gain: float = 2.20
    position_tolerance: float = 0.16
    heading_slowdown_power: float = 1.5
    vertical_rate_limit: float = 0.75
    angular_noise_scale: float = 1.0


def default_formation_targets(
    agent_classes: list[AgentClass | str],
    center_xy: tuple[float, float] = (0.0, 1.5),
    ground_radius: float = 4.0,
    air_radius: float = 1.9,
    air_base_altitude: float = 2.45,
    air_altitude_step: float = 0.60,
) -> list[FormationTarget3D]:
    """Build a default 3D stacked formation for the current team composition."""

    cx, cy = (float(center_xy[0]), float(center_xy[1]))
    normalized_classes = [normalize_agent_class(agent_class) for agent_class in agent_classes]
    ugv_ids = [agent_id for agent_id, agent_class in enumerate(normalized_classes) if agent_class == AgentClass.CLASS_A_UGV]
    uav_ids = [agent_id for agent_id, agent_class in enumerate(normalized_classes) if agent_class == AgentClass.CLASS_B_UAV]

    targets: dict[int, FormationTarget3D] = {}

    if ugv_ids:
        angles = np.linspace(0.0, 2.0 * np.pi, len(ugv_ids), endpoint=False) + (0.5 * np.pi)
        for slot_id, (agent_id, angle) in enumerate(zip(ugv_ids, angles, strict=False)):
            position_xyz = (
                cx + ground_radius * cos(angle),
                cy + ground_radius * sin(angle),
                default_render_altitude(AgentClass.CLASS_A_UGV),
            )
            targets[agent_id] = FormationTarget3D(
                agent_id=agent_id,
                position_xyz=position_xyz,
                label=f"UGV slot {slot_id}",
            )

    if uav_ids:
        if len(uav_ids) == 1:
            air_angles = np.asarray([0.5 * np.pi], dtype=float)
        else:
            air_angles = np.linspace(0.0, 2.0 * np.pi, len(uav_ids), endpoint=False) + np.pi
        for slot_id, (agent_id, angle) in enumerate(zip(uav_ids, air_angles, strict=False)):
            position_xyz = (
                cx + air_radius * cos(angle),
                cy + air_radius * sin(angle),
                air_base_altitude + air_altitude_step * slot_id,
            )
            targets[agent_id] = FormationTarget3D(
                agent_id=agent_id,
                position_xyz=position_xyz,
                label=f"UAV slot {slot_id}",
            )

    return [targets[agent_id] for agent_id in range(len(agent_classes))]


def default_static_obstacles() -> list[StaticObstacle3D]:
    """Scene obstacles that later safety filters can reason about."""

    return [
        StaticObstacle3D(
            obstacle_id="center_box",
            primitive="box",
            position_xyz=(0.0, 1.6, 0.90),
            half_extents_xyz=(0.55, 0.55, 0.90),
            color_rgba=(0.44, 0.46, 0.50, 0.92),
        ),
        StaticObstacle3D(
            obstacle_id="left_column",
            primitive="cylinder",
            position_xyz=(-2.25, 2.75, 1.10),
            radius=0.45,
            height=2.20,
            color_rgba=(0.58, 0.42, 0.30, 0.94),
        ),
        StaticObstacle3D(
            obstacle_id="right_column",
            primitive="cylinder",
            position_xyz=(2.25, 2.55, 1.35),
            radius=0.55,
            height=2.70,
            color_rgba=(0.30, 0.48, 0.62, 0.94),
        ),
    ]


class EstimateDrivenFormationController:
    """Drive each robot toward a fixed 3D formation slot using its local estimate."""

    def __init__(
        self,
        formation_targets: list[FormationTarget3D],
        obstacles: list[StaticObstacle3D] | None = None,
        gains: FormationControlGains | None = None,
        safety_filter: SafetyFilter | None = None,
    ):
        if not formation_targets:
            raise ValueError("formation_targets must be non-empty")

        self.targets = {
            int(target.agent_id): np.asarray(target.position_xyz, dtype=float)
            for target in formation_targets
        }
        self.formation_targets = list(formation_targets)
        self.obstacles = list(obstacles or default_static_obstacles())
        self.gains = gains or FormationControlGains()
        self.safety_filter = safety_filter or IdentitySafetyFilter()
        self.name = "estimate_driven_formation_controller"

    def target_position_xyz(self, agent_id: int) -> np.ndarray:
        return self.targets[int(agent_id)].copy()

    def update_render_altitude(
        self,
        agent_id: int,
        current_altitude: float,
        dt: float,
    ) -> float:
        target_altitude = float(self.target_position_xyz(agent_id)[2])
        delta = target_altitude - float(current_altitude)
        max_step = self.gains.vertical_rate_limit * float(dt)
        return float(current_altitude + np.clip(delta, -max_step, max_step))

    def compute_nominal_input(
        self,
        *,
        agent_id: int,
        estimated_position_xy: np.ndarray,
        current_theta: float,
        robot: Robot,
        dt: float,
    ) -> list[float]:
        estimated_position_xy = np.asarray(estimated_position_xy, dtype=float).reshape(2)
        target_position_xyz = self.target_position_xyz(agent_id)
        delta_xy = target_position_xyz[:2] - estimated_position_xy
        distance = float(np.linalg.norm(delta_xy))

        if distance <= self.gains.position_tolerance:
            nominal_input = np.zeros(2, dtype=float)
        else:
            heading = atan2(delta_xy[1], delta_xy[0])
            heading_error = wrap_to_pi(heading - float(current_theta))
            max_speed = sim_env.max_v * robot.class_profile.max_v_scale
            max_turn_rate = sim_env.max_omega * robot.class_profile.max_omega_scale

            heading_alignment = max(0.0, np.cos(heading_error))
            heading_scale = heading_alignment ** self.gains.heading_slowdown_power
            linear_velocity = min(self.gains.linear_gain * distance, max_speed) * heading_scale
            angular_velocity = np.clip(
                self.gains.angular_gain * heading_error,
                -max_turn_rate,
                max_turn_rate,
            )
            nominal_input = np.array([linear_velocity, angular_velocity], dtype=float)

        filtered_input = self.safety_filter.filter_control(
            agent_id=agent_id,
            nominal_input=nominal_input,
            estimated_position_xy=estimated_position_xy,
            target_position_xyz=target_position_xyz,
            current_theta=float(current_theta),
            obstacles=self.obstacles,
            robot=robot,
            dt=float(dt),
        )
        return np.asarray(filtered_input, dtype=float).reshape(2).tolist()

    def sample_realized_input(
        self,
        *,
        robot: Robot,
        nominal_input: list[float],
        dt: float,
        rng: np.random.Generator,
    ) -> list[float]:
        nominal_velocity, nominal_omega = nominal_input
        velocity_noise = rng.normal(
            0.0,
            np.sqrt(sim_env.var_u_v * robot.class_profile.process_var_scale),
        )
        omega_noise = rng.normal(
            0.0,
            np.sqrt(sim_env.var_u_theta) * self.gains.angular_noise_scale,
        )
        realized_velocity = float(nominal_velocity + velocity_noise)
        realized_omega = float(nominal_omega + omega_noise)
        future_theta = float(robot.theta + realized_omega * dt)
        proposal = np.array(
            [
                robot.position[0] + cos(future_theta) * realized_velocity * dt,
                robot.position[1] + sin(future_theta) * realized_velocity * dt,
            ],
            dtype=float,
        )
        if not sim_env.inRange(proposal, sim_env.origin):
            return [0.0, float(realized_omega)]
        return [realized_velocity, realized_omega]
