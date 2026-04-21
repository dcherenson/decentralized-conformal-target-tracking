"""Shared heterogeneous scenario helpers for UGV/UAV rollouts."""

from __future__ import annotations

from math import atan2, cos, sin

import numpy as np

import sim_env
from agent_classes import AgentClass, normalize_agent_class
from algorithm.gs_ci import GS_CI
from formation_control import (
    EstimateDrivenFormationController,
    default_formation_targets,
    default_render_altitude,
)
from robot import Robot
from topology import sample_topologies


def default_epsilon_by_class(
    epsilon_ugv: float = 0.10,
    epsilon_uav: float = 0.05,
) -> dict[AgentClass, float]:
    return {
        AgentClass.CLASS_A_UGV: float(epsilon_ugv),
        AgentClass.CLASS_B_UAV: float(epsilon_uav),
    }


def default_class_quantiles(
    ugv_quantile: float = 1.0,
    uav_quantile: float = 1.0,
) -> dict[AgentClass, float]:
    return {
        AgentClass.CLASS_A_UGV: float(ugv_quantile),
        AgentClass.CLASS_B_UAV: float(uav_quantile),
    }


def alternating_agent_classes(num_agents: int) -> list[AgentClass]:
    classes = [AgentClass.CLASS_A_UGV, AgentClass.CLASS_B_UAV]
    return [classes[idx % len(classes)] for idx in range(num_agents)]


def sample_initial_state(
    rng: np.random.Generator,
    jitter_std: float = 0.25,
) -> np.ndarray:
    base_state = np.asarray(sim_env.initial_position, dtype=float).reshape(-1)
    jitter = rng.normal(0.0, jitter_std, size=base_state.shape)
    return base_state + jitter


def build_ground_truth_team(
    initial_state: np.ndarray,
    agent_classes: list[AgentClass],
    epsilon_by_class: dict[AgentClass, float] | None = None,
) -> list[Robot]:
    epsilon_by_class = epsilon_by_class or default_epsilon_by_class()
    robots = []
    for agent_id, agent_class in enumerate(agent_classes):
        ii = 2 * agent_id
        robots.append(
            Robot(
                _position=initial_state[ii:ii + 2],
                agent_class=agent_class,
                epsilon=epsilon_by_class[agent_class],
            )
        )
    return robots


def build_local_filters(
    initial_state: np.ndarray,
    agent_classes: list[AgentClass],
    epsilon_by_class: dict[AgentClass, float] | None = None,
    class_quantiles: dict[AgentClass | str, float] | None = None,
    ci_coeff: float = 0.8,
) -> list[GS_CI]:
    epsilon_by_class = epsilon_by_class or default_epsilon_by_class()
    initial_state_column = np.matrix(initial_state, dtype=float).reshape((-1, 1))
    return [
        GS_CI(
            _index=agent_id,
            _initial_s=initial_state_column.copy(),
            agent_class=agent_class,
            epsilon=epsilon_by_class[agent_class],
            class_quantiles=class_quantiles,
            ci_coeff=ci_coeff,
        )
        for agent_id, agent_class in enumerate(agent_classes)
    ]


def sample_odometry_input(
    robot: Robot,
    dt: float,
    rng: np.random.Generator,
    max_attempts: int = 128,
) -> tuple[list[float], list[float]]:
    """Sample nominal and realized odometry inputs for one robot."""

    profile = robot.class_profile
    for _ in range(max_attempts):
        nominal_v = sim_env.max_v * profile.max_v_scale * rng.uniform(-1.0, 1.0)
        nominal_omega = sim_env.max_omega * profile.max_omega_scale * rng.uniform(-1.0, 1.0)
        realized_v = nominal_v + rng.normal(
            0.0,
            np.sqrt(sim_env.var_u_v * profile.process_var_scale),
        )
        proposal = np.array(
            [
                robot.position[0] + cos(robot.theta) * realized_v * dt,
                robot.position[1] + sin(robot.theta) * realized_v * dt,
            ]
        )
        if sim_env.inRange(proposal, sim_env.origin):
            return [float(nominal_v), float(nominal_omega)], [float(realized_v), float(nominal_omega)]

    return [0.0, 0.0], [0.0, 0.0]


def noisy_relative_measurement(
    observer_position: np.ndarray,
    observer_theta: float,
    target_position,
    range_var: float,
    bearing_var: float,
    rng: np.random.Generator,
) -> list[float]:
    target_position = np.asarray(target_position, dtype=float).reshape(-1)
    delta = target_position - np.asarray(observer_position, dtype=float).reshape(-1)
    distance = max(1.0e-6, float(np.linalg.norm(delta) + rng.normal(0.0, np.sqrt(range_var))))
    bearing = atan2(delta[1], delta[0]) + rng.normal(0.0, np.sqrt(bearing_var)) - observer_theta
    return [distance, bearing]


def capture_team_state(robots: list[Robot]) -> np.ndarray:
    state = np.zeros(2 * len(robots), dtype=float)
    for robot_id, robot in enumerate(robots):
        ii = 2 * robot_id
        state[ii:ii + 2] = robot.position
    return state


def _normalized_quantiles(class_quantiles: dict[AgentClass | str, float] | None) -> dict[AgentClass, float]:
    if class_quantiles is None:
        return default_class_quantiles()
    return {
        normalize_agent_class(agent_class): float(quantile)
        for agent_class, quantile in class_quantiles.items()
    }


def rollout_ground_truth_episode(
    num_steps: int,
    seed: int = 7,
    initial_jitter_std: float = 0.25,
    agent_classes: list[AgentClass] | None = None,
    epsilon_by_class: dict[AgentClass, float] | None = None,
    dt: float | None = None,
) -> tuple[list[Robot], list[dict[str, object]]]:
    """Generate a purely kinematic heterogeneous rollout for visualization."""

    rng = np.random.default_rng(seed)
    dt = float(sim_env.dt if dt is None else dt)
    agent_classes = agent_classes or alternating_agent_classes(sim_env.N)
    epsilon_by_class = epsilon_by_class or default_epsilon_by_class()

    initial_state = sample_initial_state(rng, jitter_std=initial_jitter_std)
    robots = build_ground_truth_team(initial_state, agent_classes, epsilon_by_class)
    frames: list[dict[str, object]] = []

    for step_id in range(num_steps):
        nominal_inputs = []
        realized_inputs = []
        for robot in robots:
            nominal, realized = sample_odometry_input(robot=robot, dt=dt, rng=rng)
            nominal_inputs.append(nominal)
            realized_inputs.append(realized)

        for robot, realized in zip(robots, realized_inputs):
            robot.motion_propagation(realized, dt)

        frames.append(
            {
                "step": int(step_id),
                "team_state": capture_team_state(robots).tolist(),
                "poses": [
                    {
                        "agent_id": int(agent_id),
                        "agent_class": robot.agent_class.value,
                        "position_xy": robot.position.tolist(),
                        "theta": float(robot.theta),
                        "nominal_input": nominal_inputs[agent_id],
                        "realized_input": realized_inputs[agent_id],
                    }
                    for agent_id, robot in enumerate(robots)
                ],
            }
        )

    return robots, frames


def simulate_class_conditional_gs_ci_rollout(
    num_steps: int,
    seed: int = 7,
    initial_jitter_std: float = 0.25,
    agent_classes: list[AgentClass] | None = None,
    epsilon_by_class: dict[AgentClass, float] | None = None,
    class_quantiles: dict[AgentClass | str, float] | None = None,
    dt: float | None = None,
    observ_prob: float | None = None,
    comm_prob: float | None = None,
    ci_coeff: float = 0.8,
    motion_mode: str = "random",
    formation_controller: EstimateDrivenFormationController | None = None,
) -> dict[str, object]:
    """Run one heterogeneous GS-CI rollout and collect rendering diagnostics."""

    rng = np.random.default_rng(seed)
    dt = float(sim_env.dt if dt is None else dt)
    observ_prob = float(sim_env.observ_prob if observ_prob is None else observ_prob)
    comm_prob = float(sim_env.comm_prob if comm_prob is None else comm_prob)
    epsilon_by_class = epsilon_by_class or default_epsilon_by_class()
    class_quantiles = _normalized_quantiles(class_quantiles)
    agent_classes = agent_classes or alternating_agent_classes(sim_env.N)
    if motion_mode not in {"random", "formation"}:
        raise ValueError("motion_mode must be 'random' or 'formation'")

    initial_state = sample_initial_state(rng, jitter_std=initial_jitter_std)
    truth_robots = build_ground_truth_team(initial_state, agent_classes, epsilon_by_class)
    gs_ci_robots = build_local_filters(
        initial_state=initial_state,
        agent_classes=agent_classes,
        epsilon_by_class=epsilon_by_class,
        class_quantiles=class_quantiles,
        ci_coeff=ci_coeff,
    )

    landmarks = [
        sim_env.Landmark(
            0,
            np.matrix(sim_env.landmark_position, dtype=float).reshape((2, 1)),
        )
    ]
    landmark_position_xy = np.asarray(landmarks[0].position, dtype=float).reshape(-1)
    observ_topology, comm_topology = sample_topologies(
        node_num=sim_env.N,
        observ_prob=observ_prob,
        comm_prob=comm_prob,
        rng=rng,
    )

    frames: list[dict[str, object]] = []
    position_error = np.zeros((sim_env.N, num_steps), dtype=float)
    calibrated_cov_trace = np.zeros((sim_env.N, num_steps), dtype=float)
    raw_cov_trace = np.zeros((sim_env.N, num_steps), dtype=float)
    truth_positions = np.zeros((sim_env.N, num_steps, 2), dtype=float)
    estimated_positions = np.zeros((sim_env.N, num_steps, 2), dtype=float)
    raw_covariances = np.zeros((sim_env.N, num_steps, 2, 2), dtype=float)
    formation_position_error = np.full((sim_env.N, num_steps), np.nan, dtype=float)
    render_altitude = np.zeros((sim_env.N, num_steps), dtype=float)
    current_render_altitude = np.asarray(
        [default_render_altitude(agent_class) for agent_class in agent_classes],
        dtype=float,
    )

    controller_name = "random_odometry"
    formation_targets: list[dict[str, object]] = []
    static_obstacles: list[dict[str, object]] = []
    target_positions_xyz = np.zeros((sim_env.N, 3), dtype=float)
    controller = None
    if motion_mode == "formation":
        controller = formation_controller or EstimateDrivenFormationController(
            formation_targets=default_formation_targets(agent_classes),
        )
        controller_name = controller.name
        formation_targets = [target.to_dict() for target in controller.formation_targets]
        static_obstacles = [obstacle.to_dict() for obstacle in controller.obstacles]
        target_positions_xyz = np.asarray(
            [controller.target_position_xyz(agent_id) for agent_id in range(sim_env.N)],
            dtype=float,
        )

    for step_id in range(num_steps):
        for agent_id in range(sim_env.N):
            gs_ci_robots[agent_id].theta = truth_robots[agent_id].theta

        nominal_inputs = [None] * sim_env.N
        realized_inputs = [None] * sim_env.N
        for agent_id, robot in enumerate(truth_robots):
            if controller is None:
                nominal_inputs[agent_id], realized_inputs[agent_id] = sample_odometry_input(
                    robot=robot,
                    dt=dt,
                    rng=rng,
                )
            else:
                ii = 2 * agent_id
                estimated_self_position = np.asarray(
                    gs_ci_robots[agent_id].s[ii:ii + 2],
                    dtype=float,
                ).reshape(-1)
                nominal_inputs[agent_id] = controller.compute_nominal_input(
                    agent_id=agent_id,
                    estimated_position_xy=estimated_self_position,
                    current_theta=gs_ci_robots[agent_id].theta,
                    robot=robot,
                    dt=dt,
                )
                realized_inputs[agent_id] = controller.sample_realized_input(
                    robot=robot,
                    nominal_input=nominal_inputs[agent_id],
                    dt=dt,
                    rng=rng,
                )
                current_render_altitude[agent_id] = controller.update_render_altitude(
                    agent_id=agent_id,
                    current_altitude=current_render_altitude[agent_id],
                    dt=dt,
                )

        for agent_id in range(sim_env.N):
            truth_robots[agent_id].motion_propagation(realized_inputs[agent_id], dt)
            gs_ci_robots[agent_id].motion_propagation_update(nominal_inputs[agent_id], dt)

        for observer_idx, observed_idx in observ_topology.edges:
            observer = truth_robots[observer_idx]
            local_filter = gs_ci_robots[observer_idx]

            if observed_idx == sim_env.N:
                if not sim_env.inRange(observer.position, landmark_position_xy):
                    continue
                measurement = noisy_relative_measurement(
                    observer_position=observer.position,
                    observer_theta=observer.theta,
                    target_position=landmark_position_xy,
                    range_var=local_filter.var_dis,
                    bearing_var=local_filter.var_phi,
                    rng=rng,
                )
                local_filter.ablt_obsv_update(measurement, landmarks[0])
            else:
                target_robot = truth_robots[observed_idx]
                if not sim_env.inRange(observer.position, target_robot.position):
                    continue
                measurement = noisy_relative_measurement(
                    observer_position=observer.position,
                    observer_theta=observer.theta,
                    target_position=target_robot.position,
                    range_var=local_filter.var_dis,
                    bearing_var=local_filter.var_phi,
                    rng=rng,
                )
                local_filter.rela_obsv_update(observed_idx, measurement)

        for sender_idx, receiver_idx in comm_topology.edges:
            gs_ci_robots[receiver_idx].comm_update(
                gs_ci_robots[sender_idx].s,
                gs_ci_robots[sender_idx].sigma,
                gs_ci_robots[sender_idx].th_sigma,
                comm_robot_class=gs_ci_robots[sender_idx].agent_class,
                class_quantiles=class_quantiles,
            )

        frame_poses = []
        for agent_id in range(sim_env.N):
            ii = 2 * agent_id
            truth_position = truth_robots[agent_id].position.copy()
            estimate = np.asarray(gs_ci_robots[agent_id].s[ii:ii + 2], dtype=float).reshape(-1)
            sigma_self = np.asarray(gs_ci_robots[agent_id].sigma[ii:ii + 2, ii:ii + 2], dtype=float)
            quantile = class_quantiles[agent_classes[agent_id]]
            calibrated_sigma_self = float(quantile) * sigma_self

            truth_positions[agent_id, step_id] = truth_position
            estimated_positions[agent_id, step_id] = estimate
            raw_covariances[agent_id, step_id] = sigma_self
            position_error[agent_id, step_id] = float(np.linalg.norm(estimate - truth_position))
            raw_cov_trace[agent_id, step_id] = float(np.trace(sigma_self))
            calibrated_cov_trace[agent_id, step_id] = float(np.trace(calibrated_sigma_self))
            render_altitude[agent_id, step_id] = float(current_render_altitude[agent_id])
            if controller is not None:
                formation_position_error[agent_id, step_id] = float(
                    np.linalg.norm(estimate - target_positions_xyz[agent_id, :2])
                )

            frame_poses.append(
                {
                    "agent_id": int(agent_id),
                    "agent_class": truth_robots[agent_id].agent_class.value,
                    "position_xy": truth_position.tolist(),
                    "theta": float(truth_robots[agent_id].theta),
                    "estimated_position_xy": estimate.tolist(),
                    "target_position_xyz": target_positions_xyz[agent_id].tolist(),
                    "formation_position_error": float(formation_position_error[agent_id, step_id]),
                    "position_error": float(position_error[agent_id, step_id]),
                    "raw_cov_trace": float(raw_cov_trace[agent_id, step_id]),
                    "calibrated_cov_trace": float(calibrated_cov_trace[agent_id, step_id]),
                    "quantile": float(quantile),
                    "render_z": float(current_render_altitude[agent_id]),
                    "nominal_input": [float(value) for value in nominal_inputs[agent_id]],
                    "realized_input": [float(value) for value in realized_inputs[agent_id]],
                }
            )

        if controller is not None:
            focus_points_xy = np.vstack((truth_positions[:, step_id, :], target_positions_xyz[:, :2]))
        else:
            focus_points_xy = truth_positions[:, step_id, :]

        frames.append(
            {
                "step": int(step_id),
                "team_state": capture_team_state(truth_robots).tolist(),
                "camera_focus_xy": focus_points_xy.mean(axis=0).tolist(),
                "poses": frame_poses,
            }
        )

    return {
        "frames": frames,
        "truth_robots": truth_robots,
        "filters": gs_ci_robots,
        "agent_classes": [agent_class.value for agent_class in agent_classes],
        "class_quantiles": {agent_class.value: float(q) for agent_class, q in class_quantiles.items()},
        "motion_mode": motion_mode,
        "controller_name": controller_name,
        "formation_targets": formation_targets,
        "static_obstacles": static_obstacles,
        "target_positions_xyz": target_positions_xyz,
        "time": (np.arange(num_steps, dtype=float) * dt),
        "truth_positions": truth_positions,
        "estimated_positions": estimated_positions,
        "raw_covariances": raw_covariances,
        "render_altitude": render_altitude,
        "formation_position_error": formation_position_error,
        "position_error": position_error,
        "raw_cov_trace": raw_cov_trace,
        "calibrated_cov_trace": calibrated_cov_trace,
        "observ_topology_edges": [edge.copy() for edge in observ_topology.edges],
        "comm_topology_edges": [edge.copy() for edge in comm_topology.edges],
    }
