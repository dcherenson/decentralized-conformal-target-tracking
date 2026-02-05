"""Plot trajectories from generated calibration datasets."""

from __future__ import annotations

import argparse
from typing import Tuple

import numpy as np
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from scipy.stats import norm
from matplotlib.patches import Ellipse

# Global plot style for readability.
plt.rcParams.update({
    "font.size": 14,
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "legend.fontsize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
})


def load_labels(path: str) -> np.ndarray:
    if path.endswith(".npz"):
        data = np.load(path)
        if "y" not in data:
            raise ValueError(".npz file must contain key 'y' for labels")
        return data["y"]
    if path.endswith(".npy"):
        return np.load(path)
    raise ValueError("Only .npz or .npy files are supported")


def plot_trajectories(
    labels: np.ndarray,
    max_traj: int | None = None,
    save_path: str | None = None,
    show: bool = True,
) -> None:
    if labels.ndim != 3:
        raise ValueError("labels must have shape (N, T, state_dim)")

    num_traj, _, state_dim = labels.shape
    if state_dim < 2:
        raise ValueError("Need at least 2D position in labels to plot")

    limit = num_traj if max_traj is None else min(max_traj, num_traj)

    plt.figure(figsize=(7, 7))
    for i in range(limit):
        x = labels[i, :, 0]
        y = labels[i, :, 1]
        plt.plot(x, y, alpha=0.6)

    plt.title(f"Trajectories (N={limit})")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.axis("equal")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    if show:
        plt.show()
    else:
        plt.close()


def plot_simulation_with_ellipses(
    truth: np.ndarray,
    estimates: dict[int, np.ndarray],
    covariances: dict[int, np.ndarray],
    quantiles: dict[int, float] | None = None,
    nominal_scale: float | None = None,
    nominal_violations: dict[int, np.ndarray] | None = None,
    agent_positions: dict[int, np.ndarray] | None = None,
    save_path: str | None = None,
    show: bool = False,
    ellipse_step: int = 5,
    ellipse_scale: float = 2.0,
) -> None:
    if truth.ndim != 2 or truth.shape[1] < 2:
        raise ValueError("truth must have shape (T, 2)")

    plt.figure(figsize=(7, 7))
    plt.plot(truth[:, 0], truth[:, 1], "k-", label="Truth", linewidth=2.5)

    for agent_id, est in estimates.items():
        plt.plot(est[:, 0], est[:, 1], label=f"Estimate", linewidth=2.0)

    for agent_id, covs in covariances.items():
        violation_idx = (
            set(nominal_violations.get(agent_id, []))
            if nominal_violations is not None
            else set()
        )
        for t in range(covs.shape[0]):
            if (t % ellipse_step != 0) and (t not in violation_idx):
                continue
            cov = covs[t]
            vals, vecs = np.linalg.eigh(cov)
            order = np.argsort(vals)[::-1]
            vals = vals[order]
            vecs = vecs[:, order]
            angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
            center = estimates[agent_id][t]
            # Nominal ellipse.
            nominal_factor = nominal_scale if nominal_scale is not None else ellipse_scale
            width, height = 2 * nominal_factor * np.sqrt(vals)
            ellipse_nominal = Ellipse(
                xy=center,
                width=width,
                height=height,
                angle=angle,
                fill=False,
                alpha=0.4,
                edgecolor="red",
                linestyle="--",
                linewidth=2.0,
                label=f"Uncalibrated Miscoverage" if t == 0 else None,
            )
            plt.gca().add_patch(ellipse_nominal)

            # Calibrated ellipse.
            if quantiles is not None and agent_id in quantiles:
                q = quantiles[agent_id]
                width_c, height_c = 2 * q * np.sqrt(vals)
                ellipse_cal = Ellipse(
                    xy=center,
                    width=width_c,
                    height=height_c,
                    angle=angle,
                    fill=False,
                    alpha=0.4,
                    edgecolor="green",
                    linewidth=2.0,
                    label=f"Calibrated Coverage" if t == 0 else None,
                )
                plt.gca().add_patch(ellipse_cal)

    # Mark nominal violations at truth positions.
    if nominal_violations is not None:
        for agent_id, idx in nominal_violations.items():
            plt.scatter(
                truth[idx, 0],
                truth[idx, 1],
                color="red",
                s=90,
                marker="x",
                label=f"Agent {agent_id} nominal violation",
            )

    # Plot agent positions.
    if agent_positions is not None:
        for agent_id, pos in agent_positions.items():
            plt.scatter(
                pos[0],
                pos[1],
                color="black",
                s=160,
                marker="^",
                label=f"Agent {agent_id} position",
            )

    plt.title("Target Truth, EKF Estimates, and Covariance Ellipses")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.axis("equal")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    if show:
        plt.show()
    else:
        plt.close()


def plot_position_time_with_sigma(
    truth: np.ndarray,
    estimates: dict[int, np.ndarray],
    covariances: dict[int, np.ndarray],
    quantiles: dict[int, float] | None = None,
    alpha: float = 0.1,
    nominal_violations: dict[int, np.ndarray] | None = None,
    save_path: str | None = None,
    show: bool = False,
) -> None:
    if truth.ndim != 2 or truth.shape[1] < 2:
        raise ValueError("truth must have shape (T, 2)")

    time = np.arange(truth.shape[0])

    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    axes[0].plot(time, truth[:, 0], "k-", label="Truth Target x", linewidth=2.5)
    axes[1].plot(time, truth[:, 1], "k-", label="Truth Target y", linewidth=2.5)

    # Use two-sided Gaussian quantile for nominal band.
    z = float(norm.ppf(1 - alpha / 2))

    for agent_id, est in estimates.items():
        axes[0].plot(time, est[:, 0], label=f"Estimated Target x", linewidth=2.0)
        axes[1].plot(time, est[:, 1], label=f"Estimated Target y", linewidth=2.0)

        covs = covariances[agent_id]
        sigma_x = np.sqrt(covs[:, 0, 0])
        sigma_y = np.sqrt(covs[:, 1, 1])
        axes[0].fill_between(
            time,
            est[:, 0] - z * sigma_x,
            est[:, 0] + z * sigma_x,
            alpha=0.15,
            label=f"Uncalibrated Uncertainty",
        )
        axes[1].fill_between(
            time,
            est[:, 1] - z * sigma_y,
            est[:, 1] + z * sigma_y,
            alpha=0.15,
            label=f"Uncalibrated Uncertainty",
        )

        if quantiles is not None and agent_id in quantiles:
            q = quantiles[agent_id]
            axes[0].fill_between(
                time,
                est[:, 0] - q * sigma_x,
                est[:, 0] + q * sigma_x,
                alpha=0.1,
                label=f"Calibrated Uncertainty",
            )
            axes[1].fill_between(
                time,
                est[:, 1] - q * sigma_y,
                est[:, 1] + q * sigma_y,
                alpha=0.1,
                label=f"Calibrated Uncertainty",
            )

        if nominal_violations is not None and agent_id in nominal_violations:
            violation_idx = nominal_violations[agent_id]
            axes[0].scatter(
                time[violation_idx],
                truth[violation_idx, 0],
                color="red",
                s=60,
                marker="x",
                label=f"Uncalibrated Miscoverage",
            )
            axes[1].scatter(
                time[violation_idx],
                truth[violation_idx, 1],
                color="red",
                s=60,
                marker="x",
                label=f"Uncalibrated Miscoverage",
            )

    axes[0].set_ylabel("x")
    axes[1].set_ylabel("y")
    axes[1].set_xlabel("time step")
    axes[0].grid(True, alpha=0.3)
    axes[1].grid(True, alpha=0.3)
    axes[0].legend()
    axes[1].legend()
    fig.suptitle("Position vs Time with Uncalibrated and Calibrated Bands")
    fig.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    if show:
        plt.show()
    else:
        plt.close()


def parse_args() -> Tuple[argparse.Namespace, argparse.ArgumentParser]:
    parser = argparse.ArgumentParser(
        description="Plot trajectories from calibration dataset labels."
    )
    parser.add_argument("path", type=str, help="Path to .npz or .npy file")
    parser.add_argument(
        "--max-traj",
        type=int,
        default=None,
        help="Max number of trajectories to plot",
    )
    parser.add_argument(
        "--save",
        type=str,
        default="",
        help="Optional path to save the figure",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not display the plot window",
    )
    return parser.parse_args(), parser


def main() -> None:
    args, _ = parse_args()
    labels = load_labels(args.path)
    save_path = args.save if args.save else None
    plot_trajectories(labels, max_traj=args.max_traj, save_path=save_path, show=not args.no_show)


if __name__ == "__main__":
    main()
