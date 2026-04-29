"""Shared heterogeneous scenario helpers for UGV/UAV rollouts."""

from __future__ import annotations

from math import atan2, cos, sin
from pathlib import Path

import numpy as np
from scipy.stats import chi2

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


RANGE_OUTLIER_PROB = 0.05
RANGE_OUTLIER_SCALE = 6.0
BEARING_OUTLIER_PROB = 0.03
BEARING_OUTLIER_SCALE = 4.0


def default_epsilon_by_class(
    epsilon_ugv: float = 0.10,
    epsilon_uav: float = 0.10,
) -> dict[AgentClass, float]:
    # Class-conditioned nominal error rates used by downstream calibration logic.
    return {
        AgentClass.CLASS_A_UGV: float(epsilon_ugv),
        AgentClass.CLASS_B_UAV: float(epsilon_uav),
    }


def default_class_quantiles(
    ugv_quantile: float = 1.0,
    uav_quantile: float = 1.0,
) -> dict[AgentClass, float]:
    # Class-conditioned covariance inflation factors (default: neutral scaling).
    return {
        AgentClass.CLASS_A_UGV: float(ugv_quantile),
        AgentClass.CLASS_B_UAV: float(uav_quantile),
    }


def alternating_agent_classes(num_agents: int) -> list[AgentClass]:
    # Deterministic class assignment for mixed-team experiments.
    classes = [AgentClass.CLASS_A_UGV, AgentClass.CLASS_B_UAV]
    return [classes[idx % len(classes)] for idx in range(num_agents)]


def sample_initial_state(
    rng: np.random.Generator,
    jitter_std: float = 0.25,
) -> np.ndarray:
    # Perturb the common initial stacked state to diversify episode starts.
    base_state = np.asarray(sim_env.initial_position, dtype=float).reshape(-1)
    jitter = rng.normal(0.0, jitter_std, size=base_state.shape)
    return base_state + jitter


def build_ground_truth_team(
    initial_state: np.ndarray,
    agent_classes: list[AgentClass],
    epsilon_by_class: dict[AgentClass, float] | None = None,
) -> list[Robot]:
    # Instantiate truth robots with per-class epsilon settings.
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
    # Instantiate one local GS-CI filter per robot.
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

    # Re-sample until the realized control keeps the robot inside the arena.
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
    # Shared noisy range-bearing observation model used in simulation and calibration.
    def contaminated_gaussian(std: float, outlier_prob: float, outlier_scale: float) -> float:
        if std <= 0.0:
            return 0.0
        scale = float(outlier_scale) if rng.random() < float(outlier_prob) else 1.0
        return float(rng.normal(0.0, std * scale))

    target_position = np.asarray(target_position, dtype=float).reshape(-1)
    delta = target_position - np.asarray(observer_position, dtype=float).reshape(-1)
    range_std = float(np.sqrt(max(float(range_var), 0.0)))
    bearing_std = float(np.sqrt(max(float(bearing_var), 0.0)))
    distance_noise = contaminated_gaussian(
        std=range_std,
        outlier_prob=RANGE_OUTLIER_PROB,
        outlier_scale=RANGE_OUTLIER_SCALE,
    )
    bearing_noise = contaminated_gaussian(
        std=bearing_std,
        outlier_prob=BEARING_OUTLIER_PROB,
        outlier_scale=BEARING_OUTLIER_SCALE,
    )
    distance = max(1.0e-6, float(np.linalg.norm(delta) + distance_noise))
    bearing = atan2(delta[1], delta[0]) + bearing_noise - observer_theta
    bearing = (float(bearing) + np.pi) % (2.0 * np.pi) - np.pi
    return [distance, bearing]


def capture_team_state(robots: list[Robot]) -> np.ndarray:
    # Flatten current team positions into the stacked 2N state convention.
    state = np.zeros(2 * len(robots), dtype=float)
    for robot_id, robot in enumerate(robots):
        ii = 2 * robot_id
        state[ii:ii + 2] = robot.position
    return state


def _normalized_quantiles(class_quantiles: dict[AgentClass | str, float] | None) -> dict[AgentClass, float]:
    # Accept enum or string keys and normalize for internal use.
    if class_quantiles is None:
        return default_class_quantiles()
    return {
        normalize_agent_class(agent_class): float(quantile)
        for agent_class, quantile in class_quantiles.items()
    }


def _split_conformal_quantile(scores: np.ndarray, alpha: float) -> tuple[float, float]:
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 1 or scores.size == 0:
        raise ValueError("scores must be a non-empty 1D array")
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be in (0, 1)")
    n = scores.size
    tau = float(np.ceil((n + 1) * (1.0 - alpha)) / n)
    tau = min(max(tau, 0.0), 1.0)
    return float(np.quantile(scores, tau, method="higher")), tau


def _mahalanobis_radius(
    mean: np.ndarray,
    covariance: np.ndarray,
    truth: np.ndarray,
    epsilon: float | None = None,
) -> float:
    mean = np.asarray(mean, dtype=float).reshape(-1)
    truth = np.asarray(truth, dtype=float).reshape(-1)
    covariance = np.asarray(covariance, dtype=float)
    covariance = 0.5 * (covariance + covariance.T)
    covariance += 1.0e-12 * np.eye(covariance.shape[0], dtype=float)
    diff = truth - mean
    inv_covariance = np.linalg.pinv(covariance)
    distance_sq = float(diff.T @ inv_covariance @ diff)
    radius = float(np.sqrt(max(distance_sq, 0.0)))
    if epsilon is None:
        return radius

    epsilon = min(max(float(epsilon), 1.0e-12), 1.0 - 1.0e-12)
    chi2_cutoff = float(chi2.ppf(1.0 - epsilon, df=diff.size))
    if not np.isfinite(chi2_cutoff) or chi2_cutoff <= 0.0:
        return radius
    return float(radius / np.sqrt(chi2_cutoff))


def _doubly_stochastic_weights_from_comm_edges(
    agent_ids: list[int],
    comm_edges: list[list[int]],
) -> np.ndarray:
    # Build doubly stochastic Metropolis weights from communication-neighbor relations.
    # Directed edges are symmetrized into an undirected neighbor graph.
    num_agents = len(agent_ids)
    if num_agents <= 0:
        raise ValueError("agent_ids must be non-empty")

    id_to_local = {int(agent_id): local_idx for local_idx, agent_id in enumerate(agent_ids)}
    neighbors = [set() for _ in range(num_agents)]
    for edge in comm_edges:
        if len(edge) != 2:
            continue
        sender = int(edge[0])
        receiver = int(edge[1])
        if sender == receiver:
            continue
        sender_local = id_to_local.get(sender)
        receiver_local = id_to_local.get(receiver)
        if sender_local is None or receiver_local is None:
            continue
        neighbors[sender_local].add(receiver_local)
        neighbors[receiver_local].add(sender_local)

    weights = np.zeros((num_agents, num_agents), dtype=float)
    degrees = np.array([len(nbrs) for nbrs in neighbors], dtype=float)
    for i in range(num_agents):
        for j in sorted(neighbors[i]):
            weights[i, j] = 1.0 / (1.0 + max(degrees[i], degrees[j]))
        weights[i, i] = 1.0 - weights[i].sum()
    return weights


def _augment_comm_edges_with_joiners(
    base_comm_edges: list[list[int]],
    base_agent_count: int,
    joiner_agent_ids: list[int],
    comm_prob: float,
    rng: np.random.Generator,
) -> list[list[int]]:
    # Preserve existing rollout comm edges and sample joiner-related links with same Bernoulli model.
    edge_set: set[tuple[int, int]] = set()
    for edge in base_comm_edges:
        if len(edge) != 2:
            continue
        sender = int(edge[0])
        receiver = int(edge[1])
        if 0 <= sender < base_agent_count and 0 <= receiver < base_agent_count and sender != receiver:
            edge_set.add((sender, receiver))

    all_agent_ids = list(range(base_agent_count)) + [int(agent_id) for agent_id in joiner_agent_ids]
    for sender in all_agent_ids:
        for receiver in all_agent_ids:
            if sender == receiver:
                continue
            if sender < base_agent_count and receiver < base_agent_count:
                continue
            if rng.random() < float(comm_prob):
                edge_set.add((sender, receiver))

    return [[sender, receiver] for sender, receiver in sorted(edge_set)]


def _load_classwise_calibration_score_pools(
    dataset_path: Path,
) -> tuple[dict[AgentClass, np.ndarray], dict[AgentClass, float]]:
    # Local import prevents top-level circular dependency with data collector.
    from collect_calibration_data import load_calibration_dataset

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Calibration dataset not found at {dataset_path}. "
            "Run multirobot_localization/collect_calibration_data.py first."
        )

    dataset = load_calibration_dataset(dataset_path)
    class_scores: dict[AgentClass, list[float]] = {}
    alpha_by_class: dict[AgentClass, float] = {}

    for class_name, samples in dataset["samples_by_class"].items():
        agent_class = normalize_agent_class(class_name)
        class_scores.setdefault(agent_class, [])

        for sample in samples:
            agent_id = int(sample["agent_id"])
            truth_state = np.asarray(sample["ground_truth_state"], dtype=float).reshape(-1)
            posterior_mean = np.asarray(sample["posterior_mean"], dtype=float).reshape(-1)
            posterior_covariance = np.asarray(sample["posterior_covariance"], dtype=float)
            state_index = 2 * agent_id

            score = _mahalanobis_radius(
                mean=posterior_mean[state_index:state_index + 2],
                covariance=posterior_covariance[state_index:state_index + 2, state_index:state_index + 2],
                truth=truth_state[state_index:state_index + 2],
                epsilon=float(sample["epsilon"]),
            )
            class_scores[agent_class].append(score)
            alpha_by_class[agent_class] = float(sample["epsilon"])

    class_score_arrays = {
        agent_class: np.asarray(scores, dtype=float)
        for agent_class, scores in class_scores.items()
    }
    return class_score_arrays, alpha_by_class


def _load_online_dcp_calibration_scores(
    dataset_path: Path,
    agent_classes: list[AgentClass],
) -> tuple[dict[int, np.ndarray], dict[AgentClass, float]]:
    class_score_arrays, alpha_by_class = _load_classwise_calibration_score_pools(dataset_path)
    scores_by_agent: dict[int, list[float]] = {agent_id: [] for agent_id in range(len(agent_classes))}

    for agent_class, class_scores in class_score_arrays.items():
        if class_scores.size == 0:
            continue
        class_agent_ids = [
            agent_id
            for agent_id, class_name in enumerate(agent_classes)
            if normalize_agent_class(class_name) == agent_class
        ]
        if not class_agent_ids:
            continue
        assignments = np.array_split(class_scores, len(class_agent_ids))
        for local_idx, agent_id in enumerate(class_agent_ids):
            assigned_scores = np.asarray(assignments[local_idx], dtype=float).reshape(-1)
            scores_by_agent[agent_id].extend(float(value) for value in assigned_scores.tolist())

    for agent_id, agent_class in enumerate(agent_classes):
        if scores_by_agent[agent_id]:
            continue
        class_scores = class_score_arrays.get(agent_class)
        if class_scores is None or class_scores.size == 0:
            raise ValueError(f"No calibration scores found for agent {agent_id} ({agent_class.value})")
        scores_by_agent[agent_id] = [float(value) for value in class_scores.tolist()]

    return (
        {agent_id: np.asarray(scores, dtype=float) for agent_id, scores in scores_by_agent.items()},
        alpha_by_class,
    )


def _run_startup_distributed_classwise_dcp(
    scores_by_agent: dict[int, np.ndarray],
    agent_classes: list[AgentClass],
    alpha_by_class: dict[AgentClass, float],
    comm_edges: list[list[int]],
    num_steps: int,
    step_size: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[AgentClass]]:
    # Algorithm 2 implementation:
    # 1) local classwise subgradient, 2) local descent step, 3) communication, 4) consensus update.
    num_agents = len(agent_classes)
    unique_classes = [normalize_agent_class(agent_class) for agent_class in dict.fromkeys(agent_classes)]
    class_to_idx = {agent_class: idx for idx, agent_class in enumerate(unique_classes)}
    num_classes = len(unique_classes)

    # Local partitioned datasets D_{i,c}. In this scenario each agent contributes its class data,
    # and other class partitions are empty by default.
    local_scores_by_agent_class: dict[int, dict[AgentClass, np.ndarray]] = {}
    for agent_id in range(num_agents):
        local_scores_by_agent_class[agent_id] = {
            agent_class: np.asarray([], dtype=float)
            for agent_class in unique_classes
        }
        own_class = normalize_agent_class(agent_classes[agent_id])
        local_scores_by_agent_class[agent_id][own_class] = np.asarray(scores_by_agent[agent_id], dtype=float).reshape(-1)

    # Initialize Q_{i,0}[c] = 0 for all agents and classes.
    current = np.zeros((num_agents, num_classes), dtype=float)

    own_class_indices = np.array(
        [class_to_idx[normalize_agent_class(agent_class)] for agent_class in agent_classes],
        dtype=int,
    )
    quantile_history = np.zeros((num_agents, int(num_steps) + 1), dtype=float)
    quantile_history[:, 0] = current[np.arange(num_agents), own_class_indices]

    # Doubly stochastic communication weights W.
    mixing_weights = _doubly_stochastic_weights_from_comm_edges(
        agent_ids=[int(agent_id) for agent_id in range(num_agents)],
        comm_edges=comm_edges,
    )

    for step in range(1, int(num_steps) + 1):
        gradients = np.zeros_like(current)
        for agent_id in range(num_agents):
            for agent_class in unique_classes:
                class_idx = class_to_idx[agent_class]
                scores_ic = local_scores_by_agent_class[agent_id][agent_class]
                if scores_ic.size == 0:
                    gradients[agent_id, class_idx] = 0.0
                    continue
                tau_c = 1.0 - float(alpha_by_class[agent_class])
                gradients[agent_id, class_idx] = float(
                    np.mean(
                        np.where(
                            scores_ic > current[agent_id, class_idx],
                            -tau_c,
                            1.0 - tau_c,
                        )
                    )
                )

        # Local descent step: \tilde{Q}_{i,t} = Q_{i,t-1} - eta_t g_i(Q_{i,t-1})
        intermediate = current - float(step_size) * gradients

        # Consensus update: Q_{i,t} = \sum_j W_{ij} \tilde{Q}_{j,t}
        current = np.maximum(mixing_weights @ intermediate, 1.0e-6)
        quantile_history[:, step] = current[np.arange(num_agents), own_class_indices]

    final_quantiles = current[np.arange(num_agents), own_class_indices]
    return final_quantiles, quantile_history, current.copy(), unique_classes


def _agent_class_quantile_maps(
    quantile_matrix: np.ndarray,
    class_order: list[AgentClass],
) -> list[dict[AgentClass, float]]:
    quantile_matrix = np.asarray(quantile_matrix, dtype=float)
    num_agents = int(quantile_matrix.shape[0])
    maps: list[dict[AgentClass, float]] = []
    for agent_id in range(num_agents):
        maps.append(
            {
                normalize_agent_class(agent_class): float(quantile_matrix[agent_id, class_idx])
                for class_idx, agent_class in enumerate(class_order)
            }
        )
    return maps


def _classwise_average_from_agent_maps(
    agent_quantile_maps: list[dict[AgentClass, float]],
    class_order: list[AgentClass],
) -> dict[AgentClass, float]:
    result: dict[AgentClass, float] = {}
    if not agent_quantile_maps:
        return result
    for agent_class in class_order:
        normalized_class = normalize_agent_class(agent_class)
        values = [
            float(agent_quantile_map[normalized_class])
            for agent_quantile_map in agent_quantile_maps
            if normalized_class in agent_quantile_map
        ]
        if values:
            result[normalized_class] = float(np.mean(np.asarray(values, dtype=float)))
    return result


def _sample_class_scores_for_joiner(
    class_scores: np.ndarray,
    sample_count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    class_scores = np.asarray(class_scores, dtype=float).reshape(-1)
    if class_scores.size == 0:
        raise ValueError("Cannot sample joiner scores from an empty class score pool")
    count = max(1, int(sample_count))
    replace = count > class_scores.size
    chosen = rng.choice(class_scores, size=count, replace=replace)
    return np.asarray(chosen, dtype=float).reshape(-1)


def rollout_ground_truth_episode(
    num_steps: int,
    seed: int = 7,
    initial_jitter_std: float = 0.25,
    agent_classes: list[AgentClass] | None = None,
    epsilon_by_class: dict[AgentClass, float] | None = None,
    dt: float | None = None,
) -> tuple[list[Robot], list[dict[str, object]]]:
    """Generate a purely kinematic heterogeneous rollout for visualization."""

    # Build a motion-only rollout for quick visualization/debugging.
    rng = np.random.default_rng(seed)
    dt = float(sim_env.dt if dt is None else dt)
    agent_classes = agent_classes or alternating_agent_classes(sim_env.N)
    epsilon_by_class = epsilon_by_class or default_epsilon_by_class()

    initial_state = sample_initial_state(rng, jitter_std=initial_jitter_std)
    robots = build_ground_truth_team(initial_state, agent_classes, epsilon_by_class)
    frames: list[dict[str, object]] = []

    # Per-step: sample controls, propagate truth, and record a frame.
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
    dcp_calibration_dataset: Path | str | None = None,
    dcp_steps: int = 250,
    dcp_step_size: float = 0.5,
    dcp_mid_join_step: int | None = None,
    dcp_mid_join_dataset: Path | str | None = None,
    dcp_mid_join_samples_per_joiner: int = 48,
) -> dict[str, object]:
    """Run one heterogeneous GS-CI rollout and collect rendering diagnostics."""

    # Full rollout with motion, observation, communication, and diagnostics.
    rng = np.random.default_rng(seed)
    dt = float(sim_env.dt if dt is None else dt)
    observ_prob = float(sim_env.observ_prob if observ_prob is None else observ_prob)
    comm_prob = float(sim_env.comm_prob if comm_prob is None else comm_prob)
    epsilon_by_class = epsilon_by_class or default_epsilon_by_class()
    class_quantiles = _normalized_quantiles(class_quantiles)
    agent_classes = agent_classes or alternating_agent_classes(sim_env.N)
    if motion_mode not in {"random", "formation"}:
        raise ValueError("motion_mode must be 'random' or 'formation'")

    startup_dataset_path = (
        Path(dcp_calibration_dataset).expanduser().resolve()
        if dcp_calibration_dataset is not None
        else None
    )
    mid_join_dataset_path = (
        Path(dcp_mid_join_dataset).expanduser().resolve()
        if dcp_mid_join_dataset is not None
        else startup_dataset_path
    )
    mid_join_step = None if dcp_mid_join_step is None else int(dcp_mid_join_step)
    if mid_join_step is not None and (mid_join_step < 0 or mid_join_step >= int(num_steps)):
        raise ValueError("dcp_mid_join_step must be within [0, num_steps - 1]")

    online_dcp: dict[str, object] = {
        "enabled": bool(startup_dataset_path is not None or mid_join_step is not None),
        "startup": {
            "enabled": bool(startup_dataset_path is not None),
            "dataset_path": None if startup_dataset_path is None else str(startup_dataset_path),
            "steps": int(dcp_steps),
            "step_size": float(dcp_step_size),
            "consensus_topology": "doubly_stochastic_metropolis_over_scenario_comm_neighbors",
        },
        "mid_join": {
            "enabled": bool(mid_join_step is not None),
            "step": None if mid_join_step is None else int(mid_join_step),
            "dataset_path": None if mid_join_dataset_path is None else str(mid_join_dataset_path),
            "samples_per_joiner": int(dcp_mid_join_samples_per_joiner),
            "joiner_classes": [AgentClass.CLASS_A_UGV.value, AgentClass.CLASS_B_UAV.value],
            "consensus_topology": "doubly_stochastic_metropolis_over_augmented_comm_neighbors",
        },
        "events": [],
    }

    initial_state = sample_initial_state(rng, jitter_std=initial_jitter_std)
    truth_robots = build_ground_truth_team(initial_state, agent_classes, epsilon_by_class)
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

    dcp_scores_by_agent: dict[int, np.ndarray] | None = None
    alpha_by_class: dict[AgentClass, float] = {}
    dcp_class_order = [normalize_agent_class(agent_class) for agent_class in dict.fromkeys(agent_classes)]
    agent_quantile_maps = [
        {
            normalize_agent_class(agent_class): float(class_quantiles[normalize_agent_class(agent_class)])
            for agent_class in dcp_class_order
        }
        for _ in range(len(agent_classes))
    ]
    if startup_dataset_path is not None:
        dcp_scores_by_agent, alpha_by_class = _load_online_dcp_calibration_scores(
            dataset_path=startup_dataset_path,
            agent_classes=list(agent_classes),
        )
        final_agent_quantiles, startup_quantile_history, startup_quantile_matrix, startup_class_order = _run_startup_distributed_classwise_dcp(
            scores_by_agent=dcp_scores_by_agent,
            agent_classes=list(agent_classes),
            alpha_by_class=alpha_by_class,
            comm_edges=[edge.copy() for edge in comm_topology.edges],
            num_steps=dcp_steps,
            step_size=dcp_step_size,
        )
        dcp_class_order = [normalize_agent_class(agent_class) for agent_class in startup_class_order]
        agent_quantile_maps = _agent_class_quantile_maps(
            quantile_matrix=startup_quantile_matrix,
            class_order=dcp_class_order,
        )
        class_quantiles.update(
            _classwise_average_from_agent_maps(
                agent_quantile_maps=agent_quantile_maps,
                class_order=dcp_class_order,
            )
        )
        online_dcp["startup"].update(
            {
                "sample_counts": {str(agent_id): int(scores.size) for agent_id, scores in dcp_scores_by_agent.items()},
                "alpha_by_class": {agent_class.value: float(value) for agent_class, value in alpha_by_class.items()},
                "final_agent_quantiles": [float(value) for value in final_agent_quantiles.tolist()],
                "final_agent_quantile_maps": [
                    {agent_class.value: float(value) for agent_class, value in agent_quantile_map.items()}
                    for agent_quantile_map in agent_quantile_maps
                ],
                "final_class_quantiles": {agent_class.value: float(value) for agent_class, value in class_quantiles.items()},
                "quantile_history": startup_quantile_history.tolist(),
                "consensus_comm_edges": [edge.copy() for edge in comm_topology.edges],
            }
        )

    # Baseline team: original GS-CI (no conformal quantile inflation).
    gs_ci_baseline_robots = build_local_filters(
        initial_state=initial_state,
        agent_classes=agent_classes,
        epsilon_by_class=epsilon_by_class,
        class_quantiles=None,
        ci_coeff=ci_coeff,
    )
    # Conformalized team: same GS-CI recursion with local distributed quantile maps.
    gs_ci_conformal_robots = build_local_filters(
        initial_state=initial_state,
        agent_classes=agent_classes,
        epsilon_by_class=epsilon_by_class,
        class_quantiles=None,
        ci_coeff=ci_coeff,
    )
    for agent_id, local_filter in enumerate(gs_ci_conformal_robots):
        local_filter.set_class_quantiles(agent_quantile_maps[agent_id])

    frames: list[dict[str, object]] = []
    baseline_position_error = np.zeros((sim_env.N, num_steps), dtype=float)
    conformal_position_error = np.zeros((sim_env.N, num_steps), dtype=float)
    baseline_cov_trace = np.zeros((sim_env.N, num_steps), dtype=float)
    conformal_cov_trace = np.zeros((sim_env.N, num_steps), dtype=float)
    truth_positions = np.zeros((sim_env.N, num_steps, 2), dtype=float)
    baseline_estimated_positions = np.zeros((sim_env.N, num_steps, 2), dtype=float)
    conformal_estimated_positions = np.zeros((sim_env.N, num_steps, 2), dtype=float)
    baseline_covariances = np.zeros((sim_env.N, num_steps, 2, 2), dtype=float)
    conformal_covariances = np.zeros((sim_env.N, num_steps, 2, 2), dtype=float)
    baseline_formation_position_error = np.full((sim_env.N, num_steps), np.nan, dtype=float)
    conformal_formation_position_error = np.full((sim_env.N, num_steps), np.nan, dtype=float)
    render_altitude = np.zeros((sim_env.N, num_steps), dtype=float)
    current_render_altitude = np.asarray(
        [default_render_altitude(agent_class) for agent_class in agent_classes],
        dtype=float,
    )

    # Optional formation controller adds target geometry and altitude profiles.
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

    mid_join_applied = False
    # Main simulation loop.
    for step_id in range(num_steps):
        if (
            (not mid_join_applied)
            and mid_join_step is not None
            and step_id == mid_join_step
        ):
            if mid_join_dataset_path is None:
                raise ValueError("mid-scenario DCP join requires a calibration dataset path")

            # If startup DCP was skipped, bootstrap baseline participant data now.
            if dcp_scores_by_agent is None:
                dcp_scores_by_agent, alpha_by_class = _load_online_dcp_calibration_scores(
                    dataset_path=mid_join_dataset_path,
                    agent_classes=list(agent_classes),
                )

            join_class_scores, join_alpha_by_class = _load_classwise_calibration_score_pools(mid_join_dataset_path)
            alpha_by_class.update(join_alpha_by_class)

            if AgentClass.CLASS_A_UGV not in join_class_scores or join_class_scores[AgentClass.CLASS_A_UGV].size == 0:
                raise ValueError("Mid-join dataset is missing CLASS_A_UGV scores")
            if AgentClass.CLASS_B_UAV not in join_class_scores or join_class_scores[AgentClass.CLASS_B_UAV].size == 0:
                raise ValueError("Mid-join dataset is missing CLASS_B_UAV scores")

            if AgentClass.CLASS_A_UGV not in alpha_by_class or AgentClass.CLASS_B_UAV not in alpha_by_class:
                raise ValueError("Missing class epsilon values required for DCP re-optimization")

            ugv_join_scores = _sample_class_scores_for_joiner(
                class_scores=join_class_scores[AgentClass.CLASS_A_UGV],
                sample_count=dcp_mid_join_samples_per_joiner,
                rng=rng,
            )
            uav_join_scores = _sample_class_scores_for_joiner(
                class_scores=join_class_scores[AgentClass.CLASS_B_UAV],
                sample_count=dcp_mid_join_samples_per_joiner,
                rng=rng,
            )

            extended_scores_by_agent: dict[int, np.ndarray] = {
                int(agent_id): np.asarray(scores, dtype=float)
                for agent_id, scores in dcp_scores_by_agent.items()
            }
            next_agent_id = max(extended_scores_by_agent.keys(), default=-1) + 1
            ugv_joiner_id = next_agent_id
            uav_joiner_id = next_agent_id + 1
            extended_scores_by_agent[ugv_joiner_id] = ugv_join_scores
            extended_scores_by_agent[uav_joiner_id] = uav_join_scores

            extended_agent_classes = list(agent_classes) + [
                AgentClass.CLASS_A_UGV,
                AgentClass.CLASS_B_UAV,
            ]
            mid_join_comm_edges = _augment_comm_edges_with_joiners(
                base_comm_edges=[edge.copy() for edge in comm_topology.edges],
                base_agent_count=len(agent_classes),
                joiner_agent_ids=[ugv_joiner_id, uav_joiner_id],
                comm_prob=comm_prob,
                rng=rng,
            )
            final_agent_quantiles, mid_join_quantile_history, mid_join_quantile_matrix, mid_join_class_order = _run_startup_distributed_classwise_dcp(
                scores_by_agent=extended_scores_by_agent,
                agent_classes=extended_agent_classes,
                alpha_by_class=alpha_by_class,
                comm_edges=mid_join_comm_edges,
                num_steps=dcp_steps,
                step_size=dcp_step_size,
            )
            previous_class_quantiles = _classwise_average_from_agent_maps(
                agent_quantile_maps=agent_quantile_maps,
                class_order=dcp_class_order,
            )
            dcp_class_order = [normalize_agent_class(agent_class) for agent_class in mid_join_class_order]
            extended_agent_quantile_maps = _agent_class_quantile_maps(
                quantile_matrix=mid_join_quantile_matrix,
                class_order=dcp_class_order,
            )
            agent_quantile_maps = extended_agent_quantile_maps[:len(agent_classes)]
            updated_class_quantiles = _classwise_average_from_agent_maps(
                agent_quantile_maps=extended_agent_quantile_maps,
                class_order=dcp_class_order,
            )
            class_quantiles.update(updated_class_quantiles)

            # Push refreshed per-agent quantile maps into each conformalized local filter.
            for agent_id, local_filter in enumerate(gs_ci_conformal_robots):
                local_filter.set_class_quantiles(agent_quantile_maps[agent_id])

            dcp_scores_by_agent = extended_scores_by_agent
            mid_join_applied = True
            online_dcp["events"].append(
                {
                    "type": "mid_scenario_join_rerun",
                    "step": int(step_id),
                    "joiner_agent_ids": {
                        "CLASS_A_UGV": int(ugv_joiner_id),
                        "CLASS_B_UAV": int(uav_joiner_id),
                    },
                    "samples_per_joiner": int(dcp_mid_join_samples_per_joiner),
                    "sample_counts": {
                        "CLASS_A_UGV": int(ugv_join_scores.size),
                        "CLASS_B_UAV": int(uav_join_scores.size),
                    },
                    "previous_class_quantiles": {
                        agent_class.value: float(value)
                        for agent_class, value in previous_class_quantiles.items()
                    },
                    "updated_class_quantiles": {
                        agent_class.value: float(value)
                        for agent_class, value in updated_class_quantiles.items()
                    },
                    "delta_class_quantiles": {
                        agent_class.value: float(
                            updated_class_quantiles[agent_class]
                            - previous_class_quantiles.get(agent_class, 0.0)
                        )
                        for agent_class in updated_class_quantiles.keys()
                    },
                    "quantile_history": mid_join_quantile_history.tolist(),
                    "final_agent_quantiles": [float(value) for value in final_agent_quantiles.tolist()],
                    "final_agent_quantile_maps": [
                        {agent_class.value: float(value) for agent_class, value in agent_quantile_map.items()}
                        for agent_quantile_map in agent_quantile_maps
                    ],
                    "consensus_comm_edges": [edge.copy() for edge in mid_join_comm_edges],
                }
            )

        for agent_id in range(sim_env.N):
            gs_ci_baseline_robots[agent_id].theta = truth_robots[agent_id].theta
            gs_ci_conformal_robots[agent_id].theta = truth_robots[agent_id].theta

        nominal_inputs = [None] * sim_env.N
        realized_inputs = [None] * sim_env.N
        # Input generation: either random odometry or controller-driven motion.
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
                    gs_ci_baseline_robots[agent_id].s[ii:ii + 2],
                    dtype=float,
                ).reshape(-1)
                nominal_inputs[agent_id] = controller.compute_nominal_input(
                    agent_id=agent_id,
                    estimated_position_xy=estimated_self_position,
                    current_theta=gs_ci_baseline_robots[agent_id].theta,
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

        # Propagate truth and estimator priors.
        for agent_id in range(sim_env.N):
            truth_robots[agent_id].motion_propagation(realized_inputs[agent_id], dt)
            gs_ci_baseline_robots[agent_id].motion_propagation_update(nominal_inputs[agent_id], dt)
            gs_ci_conformal_robots[agent_id].motion_propagation_update(nominal_inputs[agent_id], dt)

        # Observation update for sampled graph edges.
        for observer_idx, observed_idx in observ_topology.edges:
            observer = truth_robots[observer_idx]
            baseline_filter = gs_ci_baseline_robots[observer_idx]
            conformal_filter = gs_ci_conformal_robots[observer_idx]

            if observed_idx == sim_env.N:
                if not sim_env.inRange(observer.position, landmark_position_xy):
                    continue
                measurement = noisy_relative_measurement(
                    observer_position=observer.position,
                    observer_theta=observer.theta,
                    target_position=landmark_position_xy,
                    range_var=baseline_filter.var_dis,
                    bearing_var=baseline_filter.var_phi,
                    rng=rng,
                )
                baseline_filter.ablt_obsv_update(measurement, landmarks[0])
                conformal_filter.ablt_obsv_update(measurement, landmarks[0])
            else:
                target_robot = truth_robots[observed_idx]
                if not sim_env.inRange(observer.position, target_robot.position):
                    continue
                measurement = noisy_relative_measurement(
                    observer_position=observer.position,
                    observer_theta=observer.theta,
                    target_position=target_robot.position,
                    range_var=baseline_filter.var_dis,
                    bearing_var=baseline_filter.var_phi,
                    rng=rng,
                )
                baseline_filter.rela_obsv_update(observed_idx, measurement)
                conformal_filter.rela_obsv_update(observed_idx, measurement)

        # Communication update with pre-comm transmission buffers.
        # This prevents within-step sender overwrite from contaminating later receivers.
        baseline_tx_buffer = [
            {
                "agent_id": int(agent_id),
                "agent_class": gs_ci_baseline_robots[agent_id].agent_class,
                "s": gs_ci_baseline_robots[agent_id].s.copy(),
                "sigma": gs_ci_baseline_robots[agent_id].sigma.copy(),
                "th_sigma": gs_ci_baseline_robots[agent_id].th_sigma.copy(),
            }
            for agent_id in range(sim_env.N)
        ]
        conformal_tx_buffer = [
            {
                "agent_id": int(agent_id),
                "agent_class": gs_ci_conformal_robots[agent_id].agent_class,
                "s": gs_ci_conformal_robots[agent_id].s.copy(),
                "sigma": gs_ci_conformal_robots[agent_id].sigma.copy(),
                "th_sigma": gs_ci_conformal_robots[agent_id].th_sigma.copy(),
            }
            for agent_id in range(sim_env.N)
        ]

        baseline_neighbors_by_receiver = {agent_id: [] for agent_id in range(sim_env.N)}
        conformal_neighbors_by_receiver = {agent_id: [] for agent_id in range(sim_env.N)}
        for sender_idx, receiver_idx in comm_topology.edges:
            baseline_neighbors_by_receiver[receiver_idx].append(baseline_tx_buffer[sender_idx])
            conformal_neighbors_by_receiver[receiver_idx].append(conformal_tx_buffer[sender_idx])

        conformal_received_comm = np.zeros((sim_env.N,), dtype=bool)
        for receiver_idx in range(sim_env.N):
            baseline_neighbors = baseline_neighbors_by_receiver[receiver_idx]
            conformal_neighbors = conformal_neighbors_by_receiver[receiver_idx]
            if baseline_neighbors:
                gs_ci_baseline_robots[receiver_idx].comm_update(
                    neighbor_data=baseline_neighbors,
                    use_quantiles=False,
                )
            if conformal_neighbors:
                gs_ci_conformal_robots[receiver_idx].comm_update(
                    neighbor_data=conformal_neighbors,
                    class_quantiles=None,
                    use_quantiles=True,
                )
                conformal_received_comm[receiver_idx] = True

        # Collect per-agent diagnostics for plotting/rendering.
        frame_poses = []
        for agent_id in range(sim_env.N):
            ii = 2 * agent_id
            truth_position = truth_robots[agent_id].position.copy()
            baseline_estimate = np.asarray(gs_ci_baseline_robots[agent_id].s[ii:ii + 2], dtype=float).reshape(-1)
            conformal_estimate = np.asarray(gs_ci_conformal_robots[agent_id].s[ii:ii + 2], dtype=float).reshape(-1)
            baseline_sigma_self = np.asarray(gs_ci_baseline_robots[agent_id].sigma[ii:ii + 2, ii:ii + 2], dtype=float)
            conformal_sigma_self = np.asarray(gs_ci_conformal_robots[agent_id].sigma[ii:ii + 2, ii:ii + 2], dtype=float)
            quantile = gs_ci_conformal_robots[agent_id].get_class_quantile(agent_class=agent_classes[agent_id])
            if not conformal_received_comm[agent_id]:
                conformal_sigma_self = float(quantile) * conformal_sigma_self
                conformal_sigma_self = 0.5 * (conformal_sigma_self + conformal_sigma_self.T)

            truth_positions[agent_id, step_id] = truth_position
            baseline_estimated_positions[agent_id, step_id] = baseline_estimate
            conformal_estimated_positions[agent_id, step_id] = conformal_estimate
            baseline_covariances[agent_id, step_id] = baseline_sigma_self
            conformal_covariances[agent_id, step_id] = conformal_sigma_self
            baseline_position_error[agent_id, step_id] = float(np.linalg.norm(baseline_estimate - truth_position))
            conformal_position_error[agent_id, step_id] = float(np.linalg.norm(conformal_estimate - truth_position))
            baseline_cov_trace[agent_id, step_id] = float(np.trace(baseline_sigma_self))
            conformal_cov_trace[agent_id, step_id] = float(np.trace(conformal_sigma_self))
            render_altitude[agent_id, step_id] = float(current_render_altitude[agent_id])
            if controller is not None:
                baseline_formation_position_error[agent_id, step_id] = float(
                    np.linalg.norm(baseline_estimate - target_positions_xyz[agent_id, :2])
                )
                conformal_formation_position_error[agent_id, step_id] = float(
                    np.linalg.norm(conformal_estimate - target_positions_xyz[agent_id, :2])
                )

            frame_poses.append(
                {
                    "agent_id": int(agent_id),
                    "agent_class": truth_robots[agent_id].agent_class.value,
                    "position_xy": truth_position.tolist(),
                    "theta": float(truth_robots[agent_id].theta),
                    "estimated_position_xy": baseline_estimate.tolist(),
                    "calibrated_estimated_position_xy": conformal_estimate.tolist(),
                    "target_position_xyz": target_positions_xyz[agent_id].tolist(),
                    "formation_position_error": float(baseline_formation_position_error[agent_id, step_id]),
                    "calibrated_formation_position_error": float(conformal_formation_position_error[agent_id, step_id]),
                    "position_error": float(baseline_position_error[agent_id, step_id]),
                    "calibrated_position_error": float(conformal_position_error[agent_id, step_id]),
                    "raw_cov_trace": float(baseline_cov_trace[agent_id, step_id]),
                    "calibrated_cov_trace": float(conformal_cov_trace[agent_id, step_id]),
                    "quantile": float(quantile),
                    "render_z": float(current_render_altitude[agent_id]),
                    "nominal_input": [float(value) for value in nominal_inputs[agent_id]],
                    "realized_input": [float(value) for value in realized_inputs[agent_id]],
                }
            )

        # Camera focus follows robots (and formation targets in formation mode).
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

    online_dcp["mid_join"]["applied"] = bool(mid_join_applied)
    final_class_quantiles = _classwise_average_from_agent_maps(
        agent_quantile_maps=agent_quantile_maps,
        class_order=dcp_class_order,
    )
    class_quantiles.update(final_class_quantiles)
    online_dcp["final_class_quantiles"] = {
        agent_class.value: float(value)
        for agent_class, value in final_class_quantiles.items()
    }

    # Return full rollout bundle consumed by render and plotting scripts.
    return {
        "frames": frames,
        "truth_robots": truth_robots,
        "filters_baseline": gs_ci_baseline_robots,
        "filters_conformal": gs_ci_conformal_robots,
        "filters": gs_ci_conformal_robots,
        "agent_classes": [agent_class.value for agent_class in agent_classes],
        "epsilon_by_class": {
            agent_class.value: float(epsilon_by_class[agent_class])
            for agent_class in epsilon_by_class.keys()
        },
        "class_quantiles": {agent_class.value: float(q) for agent_class, q in class_quantiles.items()},
        "final_agent_quantile_maps": [
            {agent_class.value: float(value) for agent_class, value in agent_quantile_map.items()}
            for agent_quantile_map in agent_quantile_maps
        ],
        "online_dcp": online_dcp,
        "motion_mode": motion_mode,
        "controller_name": controller_name,
        "formation_targets": formation_targets,
        "static_obstacles": static_obstacles,
        "target_positions_xyz": target_positions_xyz,
        "time": (np.arange(num_steps, dtype=float) * dt),
        "truth_positions": truth_positions,
        "estimated_positions": baseline_estimated_positions,
        "calibrated_estimated_positions": conformal_estimated_positions,
        "raw_covariances": baseline_covariances,
        "calibrated_covariances": conformal_covariances,
        "render_altitude": render_altitude,
        "formation_position_error": baseline_formation_position_error,
        "calibrated_formation_position_error": conformal_formation_position_error,
        "position_error": baseline_position_error,
        "calibrated_position_error": conformal_position_error,
        "raw_cov_trace": baseline_cov_trace,
        "calibrated_cov_trace": conformal_cov_trace,
        "observ_topology_edges": [edge.copy() for edge in observ_topology.edges],
        "comm_topology_edges": [edge.copy() for edge in comm_topology.edges],
    }
