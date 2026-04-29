"""Offline calibration dataset collection for class-conditional GS-CI."""

from __future__ import annotations

import argparse
import json
import pickle
from math import atan2
from pathlib import Path

import numpy as np
from scipy import linalg
from scipy.stats import chi2

import sim_env
from agent_classes import DEFAULT_AGENT_CLASS_PROFILES, AgentClass
from algorithm.gs_ci import GS_CI
from formation_control import EstimateDrivenFormationController, default_formation_targets
from heterogeneous_scenario import (
    BEARING_OUTLIER_PROB,
    BEARING_OUTLIER_SCALE,
    RANGE_OUTLIER_PROB,
    RANGE_OUTLIER_SCALE,
    alternating_agent_classes,
    build_ground_truth_team,
    capture_team_state,
    default_epsilon_by_class,
    sample_initial_state,
    sample_odometry_input,
)


def parse_args() -> argparse.Namespace:
    # CLI for generating synthetic calibration samples under class-conditioned dynamics.
    parser = argparse.ArgumentParser(
        description=(
            "Collect an exchangeable offline calibration dataset for "
            "class-conditional distributed conformal prediction."
        )
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("multirobot_localization/calibration_dataset.npz"),
        help="Destination dataset file. Supported suffixes: .npz, .pkl, .pickle, .json.",
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
        default=0.10,
        help="Target error rate logged for CLASS_B_UAV agents.",
    )
    parser.add_argument(
        "--score-histogram-path",
        type=Path,
        default=None,
        help=(
            "Optional output path for score histogram figure. "
            "Defaults to <output_stem>_score_histogram.png next to --output."
        ),
    )
    parser.add_argument(
        "--motion-mode",
        choices=("formation", "random"),
        default="random",
        help="Motion model used to generate calibration trajectories.",
    )
    return parser.parse_args()


def build_local_filters(
    initial_state: np.ndarray,
    agent_classes: list[AgentClass],
    epsilon_by_class: dict[AgentClass, float],
) -> list[GS_CI]:
    # Build one local GS-CI estimator per robot from the same initial stacked state.
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
    # Normalize matrix/array state to a flat JSON-serializable list.
    return np.asarray(state, dtype=float).reshape(-1).tolist()


def flatten_covariance(covariance) -> list[list[float]]:
    # Normalize covariance to nested Python lists for serialization.
    return np.asarray(covariance, dtype=float).tolist()


def mahalanobis_radius(
    mean: np.ndarray,
    covariance: np.ndarray,
    truth: np.ndarray,
    epsilon: float | None = None,
) -> float:
    # Nonconformity score used by CP: Mahalanobis radius normalized by chi-square cutoff.
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


def noisy_relative_measurement(
    observer_position: np.ndarray,
    observer_theta: float,
    target_position,
    range_var: float,
    bearing_var: float,
    rng: np.random.Generator,
) -> list[float]:
    # Generate noisy range-bearing observations from observer frame.
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
    distance = max(1.0e-6, float(linalg.norm(delta) + distance_noise))
    bearing = atan2(delta[1], delta[0]) + bearing_noise - observer_theta
    bearing = (float(bearing) + np.pi) % (2.0 * np.pi) - np.pi
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
    # Record one exchangeable posterior snapshot for offline conformal calibration.
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
    # Global setup shared across episodes.
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
    controller = None
    if args.motion_mode == "formation":
        controller = EstimateDrivenFormationController(
            formation_targets=default_formation_targets(agent_classes),
        )

    dataset_by_class: dict[str, list[dict[str, object]]] = {
        agent_class.value: [] for agent_class in AgentClass
    }

    min_snapshot_step = min(max(args.burn_in, 0), args.steps - 1)

    # Each episode contributes one sampled snapshot per agent after burn-in.
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

        # Simulation loop: synchronize heading, propagate, observe, and log snapshots.
        for step_id in range(args.steps):
            for agent_id in range(sim_env.N):
                local_filters[agent_id].theta = truth_robots[agent_id].theta

            nominal_inputs = [None] * sim_env.N
            realized_inputs = [None] * sim_env.N
            # Sample noisy controls and evolve truth + filter predictions.
            for agent_id, robot in enumerate(truth_robots):
                if controller is None:
                    nominal_inputs[agent_id], realized_inputs[agent_id] = sample_odometry_input(
                        robot=robot,
                        dt=sim_env.dt,
                        rng=episode_rng,
                    )
                else:
                    ii = 2 * agent_id
                    estimated_self_position = np.asarray(
                        local_filters[agent_id].s[ii:ii + 2],
                        dtype=float,
                    ).reshape(-1)
                    nominal_inputs[agent_id] = controller.compute_nominal_input(
                        agent_id=agent_id,
                        estimated_position_xy=estimated_self_position,
                        current_theta=local_filters[agent_id].theta,
                        robot=robot,
                        dt=sim_env.dt,
                    )
                    realized_inputs[agent_id] = controller.sample_realized_input(
                        robot=robot,
                        nominal_input=nominal_inputs[agent_id],
                        dt=sim_env.dt,
                        rng=episode_rng,
                    )

            for agent_id in range(sim_env.N):
                truth_robots[agent_id].motion_propagation(realized_inputs[agent_id], sim_env.dt)
                local_filters[agent_id].motion_propagation_update(nominal_inputs[agent_id], sim_env.dt)

            # Apply stochastic landmark and relative observations.
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

            # Log only agents whose pre-sampled snapshot time matches this step.
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

    # Dataset payload keeps metadata and class-partitioned samples.
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
            "motion_mode": str(args.motion_mode),
            "controller_name": "random_odometry" if controller is None else controller.name,
            "agent_profiles": {
                agent_class.value: DEFAULT_AGENT_CLASS_PROFILES[agent_class].to_dict()
                for agent_class in AgentClass
            },
            "epsilon_by_class": {
                agent_class.value: float(epsilon_by_class[agent_class])
                for agent_class in AgentClass
            },
            "measurement_noise_model": "contaminated_gaussian",
            "measurement_noise_params": {
                "range": {
                    "outlier_prob": float(RANGE_OUTLIER_PROB),
                    "outlier_scale": float(RANGE_OUTLIER_SCALE),
                },
                "bearing": {
                    "outlier_prob": float(BEARING_OUTLIER_PROB),
                    "outlier_scale": float(BEARING_OUTLIER_SCALE),
                },
            },
        },
        "samples_by_class": dataset_by_class,
    }


def _empty_sample_arrays(state_dim: int) -> dict[str, np.ndarray]:
    # Allocate empty typed arrays for classes without any samples.
    return {
        "episode_id": np.zeros((0,), dtype=np.int64),
        "time_index": np.zeros((0,), dtype=np.int64),
        "agent_id": np.zeros((0,), dtype=np.int64),
        "epsilon": np.zeros((0,), dtype=np.float64),
        "ground_truth_state": np.zeros((0, state_dim), dtype=np.float64),
        "posterior_mean": np.zeros((0, state_dim), dtype=np.float64),
        "posterior_covariance": np.zeros((0, state_dim, state_dim), dtype=np.float64),
    }


def _samples_to_numpy_arrays(
    samples: list[dict[str, object]],
    state_dim: int,
) -> dict[str, np.ndarray]:
    # Convert per-snapshot dictionaries into compact vectorized arrays.
    if not samples:
        return _empty_sample_arrays(state_dim)

    return {
        "episode_id": np.asarray([sample["episode_id"] for sample in samples], dtype=np.int64),
        "time_index": np.asarray([sample["time_index"] for sample in samples], dtype=np.int64),
        "agent_id": np.asarray([sample["agent_id"] for sample in samples], dtype=np.int64),
        "epsilon": np.asarray([sample["epsilon"] for sample in samples], dtype=np.float64),
        "ground_truth_state": np.asarray(
            [sample["ground_truth_state"] for sample in samples],
            dtype=np.float64,
        ).reshape((-1, state_dim)),
        "posterior_mean": np.asarray(
            [sample["posterior_mean"] for sample in samples],
            dtype=np.float64,
        ).reshape((-1, state_dim)),
        "posterior_covariance": np.asarray(
            [sample["posterior_covariance"] for sample in samples],
            dtype=np.float64,
        ).reshape((-1, state_dim, state_dim)),
    }


def save_calibration_dataset(path: Path, dataset: dict[str, object]) -> None:
    # Write dataset in one of the supported interchange/storage formats.
    suffix = path.suffix.lower()

    if suffix == ".json":
        with path.open("w", encoding="utf-8") as handle:
            json.dump(dataset, handle, indent=2)
        return

    if suffix in {".pkl", ".pickle"}:
        with path.open("wb") as handle:
            pickle.dump(dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return

    if suffix == ".npz":
        # NPZ path packs metadata as JSON text plus per-class array fields.
        metadata = dict(dataset["metadata"])
        state_dim = 2 * int(metadata["num_agents"])
        arrays: dict[str, np.ndarray] = {
            "metadata_json": np.asarray(json.dumps(metadata), dtype=np.str_),
        }

        samples_by_class = dataset["samples_by_class"]
        for agent_class in AgentClass:
            class_name = agent_class.value
            class_arrays = _samples_to_numpy_arrays(
                samples=samples_by_class.get(class_name, []),
                state_dim=state_dim,
            )
            for field_name, values in class_arrays.items():
                arrays[f"{class_name}__{field_name}"] = values

        np.savez_compressed(path, **arrays)
        return

    raise ValueError(
        f"Unsupported dataset suffix '{path.suffix}'. "
        "Use .npz, .pkl, .pickle, or .json."
    )


def load_calibration_dataset(path: Path) -> dict[str, object]:
    # Read dataset from JSON/Pickle/NPZ into the canonical dict structure.
    suffix = path.suffix.lower()

    if suffix == ".json":
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    if suffix in {".pkl", ".pickle"}:
        with path.open("rb") as handle:
            return pickle.load(handle)

    if suffix == ".npz":
        # Rebuild list-of-dicts samples from array-backed NPZ storage.
        with np.load(path) as archive:
            metadata = json.loads(str(archive["metadata_json"]))
            state_dim = 2 * int(metadata["num_agents"])
            samples_by_class: dict[str, list[dict[str, object]]] = {}

            for agent_class in AgentClass:
                class_name = agent_class.value
                episode_id = archive[f"{class_name}__episode_id"]
                time_index = archive[f"{class_name}__time_index"]
                agent_id = archive[f"{class_name}__agent_id"]
                epsilon = archive[f"{class_name}__epsilon"]
                ground_truth_state = archive[f"{class_name}__ground_truth_state"].reshape((-1, state_dim))
                posterior_mean = archive[f"{class_name}__posterior_mean"].reshape((-1, state_dim))
                posterior_covariance = archive[f"{class_name}__posterior_covariance"].reshape(
                    (-1, state_dim, state_dim)
                )

                samples_by_class[class_name] = [
                    {
                        "episode_id": int(episode_id[idx]),
                        "time_index": int(time_index[idx]),
                        "agent_id": int(agent_id[idx]),
                        "agent_class": class_name,
                        "epsilon": float(epsilon[idx]),
                        "ground_truth_state": ground_truth_state[idx].tolist(),
                        "posterior_mean": posterior_mean[idx].tolist(),
                        "posterior_covariance": posterior_covariance[idx].tolist(),
                    }
                    for idx in range(len(episode_id))
                ]

        return {
            "metadata": metadata,
            "samples_by_class": samples_by_class,
        }

    raise ValueError(
        f"Unsupported dataset suffix '{path.suffix}'. "
        "Use .npz, .pkl, .pickle, or .json."
    )


def print_all_sample_scores(dataset: dict[str, object]) -> None:
    # Print every calibration sample with truth, posterior, covariance, and CP score.
    samples_by_class = dataset["samples_by_class"]
    print("Calibration samples with CP scores (chi-square normalized):")
    for class_name in sorted(samples_by_class.keys()):
        samples = samples_by_class[class_name]
        print(f"[{class_name}] {len(samples)} samples")
        for sample_idx, sample in enumerate(samples):
            agent_id = int(sample["agent_id"])
            ii = 2 * agent_id
            truth_state = np.asarray(sample["ground_truth_state"], dtype=float).reshape(-1)
            posterior_mean = np.asarray(sample["posterior_mean"], dtype=float).reshape(-1)
            posterior_covariance = np.asarray(sample["posterior_covariance"], dtype=float)
            truth_xy = truth_state[ii:ii + 2]
            mean_xy = posterior_mean[ii:ii + 2]
            covariance_xy = posterior_covariance[ii:ii + 2, ii:ii + 2]
            score = mahalanobis_radius(
                mean=mean_xy,
                covariance=covariance_xy,
                truth=truth_xy,
                epsilon=float(sample["epsilon"]),
            )
            truth_repr = np.array2string(truth_xy, precision=6, separator=", ")
            mean_repr = np.array2string(mean_xy, precision=6, separator=", ")
            covariance_repr = np.array2string(covariance_xy, precision=6, separator=", ")
            print(
                "  "
                f"sample={sample_idx} "
                f"episode={int(sample['episode_id'])} "
                f"t={int(sample['time_index'])} "
                f"agent={agent_id} "
                f"score={score:.9f}"
            )
            print(f"    truth_xy={truth_repr}")
            print(f"    mean_xy={mean_repr}")
            print(f"    cov_xy={covariance_repr}")


def save_score_histogram(dataset: dict[str, object], output_path: Path) -> None:
    # Save per-class histograms of CP nonconformity scores.
    mplconfig_dir = output_path.parent / ".mplconfig"
    mplconfig_dir.mkdir(parents=True, exist_ok=True)

    import os

    os.environ.setdefault("MPLCONFIGDIR", str(mplconfig_dir))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    samples_by_class = dataset["samples_by_class"]
    class_names = sorted(samples_by_class.keys())
    if not class_names:
        return

    scores_by_class: dict[str, np.ndarray] = {}
    for class_name in class_names:
        class_scores: list[float] = []
        for sample in samples_by_class[class_name]:
            agent_id = int(sample["agent_id"])
            ii = 2 * agent_id
            truth_state = np.asarray(sample["ground_truth_state"], dtype=float).reshape(-1)
            posterior_mean = np.asarray(sample["posterior_mean"], dtype=float).reshape(-1)
            posterior_covariance = np.asarray(sample["posterior_covariance"], dtype=float)
            score = mahalanobis_radius(
                mean=posterior_mean[ii:ii + 2],
                covariance=posterior_covariance[ii:ii + 2, ii:ii + 2],
                truth=truth_state[ii:ii + 2],
                epsilon=float(sample["epsilon"]),
            )
            class_scores.append(float(score))
        scores_by_class[class_name] = np.asarray(class_scores, dtype=float)

    num_classes = len(class_names)
    cols = min(2, num_classes)
    rows = int(np.ceil(num_classes / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(7.2 * cols, 4.2 * rows), squeeze=False)
    flat_axes = axes.ravel()

    for idx, class_name in enumerate(class_names):
        ax = flat_axes[idx]
        class_scores = scores_by_class[class_name]
        if class_scores.size == 0:
            ax.text(0.5, 0.5, "No samples", ha="center", va="center")
            ax.set_title(class_name)
            ax.set_xlabel("normalized score")
            ax.set_ylabel("count")
            ax.grid(alpha=0.25)
            continue

        bins = int(min(50, max(10, round(np.sqrt(class_scores.size)))))
        ax.hist(
            class_scores,
            bins=bins,
            color=(0.2, 0.4, 0.75),
            edgecolor="black",
            alpha=0.8,
        )
        ax.set_title(f"{class_name} (n={class_scores.size})")
        ax.set_xlabel("normalized score")
        ax.set_ylabel("count")
        ax.grid(alpha=0.25)

    for idx in range(num_classes, len(flat_axes)):
        flat_axes[idx].axis("off")

    fig.suptitle("Calibration Score Histograms by Agent Class (Chi-square Normalized)", fontsize=14)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    # End-to-end entrypoint used from command line.
    args = parse_args()
    dataset = collect_dataset(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_calibration_dataset(args.output, dataset)
    histogram_path = (
        args.score_histogram_path.resolve()
        if args.score_histogram_path is not None
        else args.output.resolve().with_name(f"{args.output.resolve().stem}_score_histogram.png")
    )
    save_score_histogram(dataset, histogram_path)
    print(f"Saved score histogram to {histogram_path}")
    print_all_sample_scores(dataset)

    samples_by_class = dataset["samples_by_class"]
    total_samples = sum(len(samples) for samples in samples_by_class.values())
    print(f"Saved {total_samples} calibration samples to {args.output}")
    for agent_class, samples in samples_by_class.items():
        print(f"  {agent_class}: {len(samples)} samples")


if __name__ == "__main__":
    main()
