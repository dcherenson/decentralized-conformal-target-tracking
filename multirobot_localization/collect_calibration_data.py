"""Offline calibration dataset collection for class-conditional GS-CI."""

from __future__ import annotations

import argparse
import json
from math import atan2
from pathlib import Path

import numpy as np
from scipy import linalg

import sim_env
from agent_classes import DEFAULT_AGENT_CLASS_PROFILES, AgentClass
from algorithm.gs_ci import GS_CI
from heterogeneous_scenario import (
    alternating_agent_classes,
    build_ground_truth_team,
    capture_team_state,
    default_epsilon_by_class,
    sample_initial_state,
    sample_odometry_input,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Collect an exchangeable offline calibration dataset for "
            "class-conditional distributed conformal prediction."
        )
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("multirobot_localization/calibration_dataset.json"),
        help="Destination JSON file.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=200,
        help="Number of independent simulated episodes.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=150,
        help="Timesteps per episode.",
    )
    parser.add_argument(
        "--burn-in",
        type=int,
        default=25,
        help="Earliest timestep eligible for a logged snapshot.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Base random seed for reproducible dataset generation.",
    )
    parser.add_argument(
        "--relative-observ-prob",
        type=float,
        default=sim_env.observ_prob,
        help="Probability of logging a relative observation edge at a timestep.",
    )
    parser.add_argument(
        "--landmark-observ-prob",
        type=float,
        default=sim_env.observ_prob,
        help="Probability of logging a landmark observation for a robot at a timestep.",
    )
    parser.add_argument(
        "--initial-jitter-std",
        type=float,
        default=0.25,
        help="Gaussian position jitter applied to the default team initialization.",
    )
    parser.add_argument(
        "--epsilon-ugv",
        type=float,
        default=0.10,
        help="Target error rate logged for CLASS_A_UGV agents.",
    )
    parser.add_argument(
        "--epsilon-uav",
        type=float,
        default=0.05,
        help="Target error rate logged for CLASS_B_UAV agents.",
    )
    return parser.parse_args()


def build_local_filters(
    initial_state: np.ndarray,
    agent_classes: list[AgentClass],
    epsilon_by_class: dict[AgentClass, float],
) -> list[GS_CI]:
    initial_state_column = np.matrix(initial_state, dtype=float).reshape((-1, 1))
    return [
        GS_CI(
            _index=agent_id,
            _initial_s=initial_state_column.copy(),
            agent_class=agent_class,
            epsilon=epsilon_by_class[agent_class],
        )
        for agent_id, agent_class in enumerate(agent_classes)
    ]


def flatten_state(state) -> list[float]:
    return np.asarray(state, dtype=float).reshape(-1).tolist()


def flatten_covariance(covariance) -> list[list[float]]:
    return np.asarray(covariance, dtype=float).tolist()


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
    distance = max(1.0e-6, float(linalg.norm(delta) + rng.normal(0.0, np.sqrt(range_var))))
    bearing = atan2(delta[1], delta[0]) + rng.normal(0.0, np.sqrt(bearing_var)) - observer_theta
    return [distance, bearing]


def log_snapshot(
    dataset_by_class: dict[str, list[dict[str, object]]],
    episode_id: int,
    step_id: int,
    agent_id: int,
    agent_class: AgentClass,
    epsilon: float,
    truth_state: np.ndarray,
    local_filter: GS_CI,
):
    dataset_by_class[agent_class.value].append(
        {
            "episode_id": int(episode_id),
            "time_index": int(step_id),
            "agent_id": int(agent_id),
            "agent_class": agent_class.value,
            "epsilon": float(epsilon),
            "ground_truth_state": flatten_state(truth_state),
            "posterior_mean": flatten_state(local_filter.s),
            "posterior_covariance": flatten_covariance(local_filter.sigma),
        }
    )


def collect_dataset(args: argparse.Namespace) -> dict[str, object]:
    rng = np.random.default_rng(args.seed)
    epsilon_by_class = default_epsilon_by_class(
        epsilon_ugv=args.epsilon_ugv,
        epsilon_uav=args.epsilon_uav,
    )
    agent_classes = alternating_agent_classes(sim_env.N)
    landmarks = [
        sim_env.Landmark(
            0,
            np.matrix(sim_env.landmark_position, dtype=float).reshape((2, 1)),
        )
    ]
    landmark_position_xy = np.asarray(landmarks[0].position, dtype=float).reshape(-1)

    dataset_by_class: dict[str, list[dict[str, object]]] = {
        agent_class.value: [] for agent_class in AgentClass
    }

    min_snapshot_step = min(max(args.burn_in, 0), args.steps - 1)

    for episode_id in range(args.episodes):
        episode_rng = np.random.default_rng(rng.integers(0, 2**32 - 1))
        initial_state = sample_initial_state(episode_rng, args.initial_jitter_std)
        truth_robots = build_ground_truth_team(initial_state, agent_classes, epsilon_by_class)
        local_filters = build_local_filters(initial_state, agent_classes, epsilon_by_class)

        snapshot_steps = {
            agent_id: int(episode_rng.integers(min_snapshot_step, args.steps))
            for agent_id in range(sim_env.N)
        }
        pending_snapshots = set(snapshot_steps.keys())

        for step_id in range(args.steps):
            for agent_id in range(sim_env.N):
                local_filters[agent_id].theta = truth_robots[agent_id].theta

            nominal_inputs = [None] * sim_env.N
            realized_inputs = [None] * sim_env.N
            for agent_id, robot in enumerate(truth_robots):
                nominal_inputs[agent_id], realized_inputs[agent_id] = sample_odometry_input(
                    robot=robot,
                    dt=sim_env.dt,
                    rng=episode_rng,
                )

            for agent_id in range(sim_env.N):
                truth_robots[agent_id].motion_propagation(realized_inputs[agent_id], sim_env.dt)
                local_filters[agent_id].motion_propagation_update(nominal_inputs[agent_id], sim_env.dt)

            for observer_id, observer in enumerate(truth_robots):
                local_filter = local_filters[observer_id]

                if (
                    episode_rng.random() < args.landmark_observ_prob
                    and sim_env.inRange(observer.position, landmark_position_xy)
                ):
                    measurement = noisy_relative_measurement(
                        observer_position=observer.position,
                        observer_theta=observer.theta,
                        target_position=landmark_position_xy,
                        range_var=local_filter.var_dis,
                        bearing_var=local_filter.var_phi,
                        rng=episode_rng,
                    )
                    local_filter.ablt_obsv_update(measurement, landmarks[0])

                for target_id, target_robot in enumerate(truth_robots):
                    if observer_id == target_id:
                        continue
                    if episode_rng.random() >= args.relative_observ_prob:
                        continue
                    if not sim_env.inRange(observer.position, target_robot.position):
                        continue
                    measurement = noisy_relative_measurement(
                        observer_position=observer.position,
                        observer_theta=observer.theta,
                        target_position=target_robot.position,
                        range_var=local_filter.var_dis,
                        bearing_var=local_filter.var_phi,
                        rng=episode_rng,
                    )
                    local_filter.rela_obsv_update(target_id, measurement)

            truth_state = capture_team_state(truth_robots)
            sampled_agents = [
                agent_id for agent_id in pending_snapshots if snapshot_steps[agent_id] == step_id
            ]
            for agent_id in sampled_agents:
                agent_class = agent_classes[agent_id]
                log_snapshot(
                    dataset_by_class=dataset_by_class,
                    episode_id=episode_id,
                    step_id=step_id,
                    agent_id=agent_id,
                    agent_class=agent_class,
                    epsilon=epsilon_by_class[agent_class],
                    truth_state=truth_state,
                    local_filter=local_filters[agent_id],
                )
                pending_snapshots.remove(agent_id)

    return {
        "metadata": {
            "format": "class_conditional_calibration_dataset/v1",
            "source": "synthetic_utias_like_offline_gs_ci_rollout",
            "num_agents": int(sim_env.N),
            "episodes": int(args.episodes),
            "steps_per_episode": int(args.steps),
            "dt": float(sim_env.dt),
            "seed": int(args.seed),
            "snapshot_strategy": "one_random_snapshot_per_agent_per_episode_after_burn_in",
            "burn_in_steps": int(args.burn_in),
            "relative_observ_prob": float(args.relative_observ_prob),
            "landmark_observ_prob": float(args.landmark_observ_prob),
            "agent_profiles": {
                agent_class.value: DEFAULT_AGENT_CLASS_PROFILES[agent_class].to_dict()
                for agent_class in AgentClass
            },
            "epsilon_by_class": {
                agent_class.value: float(epsilon_by_class[agent_class])
                for agent_class in AgentClass
            },
        },
        "samples_by_class": dataset_by_class,
    }


def main():
    args = parse_args()
    dataset = collect_dataset(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(dataset, handle, indent=2)

    samples_by_class = dataset["samples_by_class"]
    total_samples = sum(len(samples) for samples in samples_by_class.values())
    print(f"Saved {total_samples} calibration samples to {args.output}")
    for agent_class, samples in samples_by_class.items():
        print(f"  {agent_class}: {len(samples)} samples")


if __name__ == "__main__":
    main()
