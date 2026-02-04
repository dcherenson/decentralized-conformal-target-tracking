"""Generate N i.i.d. random target trajectories of length T."""

from __future__ import annotations

from typing import Optional

import numpy as np

from src.targets.target import Target, LaplaceNoise
from src.dynamics.motion_model import ConstantVelocityModel
from src.sensors.sensor import Sensor, GaussianNoise, RangeBearingSensor


def generate_random_trajectories(
    num_trajectories: int,
    length: int,
    dt: float,
    dim: int,
    velocity_mean: float,
    velocity_std: float,
    laplace_scale: float,
    include_velocity_in_label: bool,
    velocity_noise_std: float,
    turn_rate_std: float,
    sensor_model: Sensor,
    motion_model: ConstantVelocityModel,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate i.i.d. trajectories and measurements for conformal calibration.

    Args:
        sensor_model: Sensor model instance to use for measurements.
        motion_model: Motion model instance for target dynamics.

    Returns:
        features: Measurements with shape (num_trajectories, length, measurement_dim).
        labels: Target state with shape (num_trajectories, length, state_dim).
    """
    if length <= 0:
        raise ValueError("length must be positive")
    if num_trajectories <= 0:
        raise ValueError("num_trajectories must be positive")
    if dim <= 0:
        raise ValueError("dim must be positive")
    if sensor_model is None:
        raise ValueError("sensor_model is required")
    if motion_model is None:
        raise ValueError("motion_model is required")

    rng = np.random.default_rng(seed)

    measurement_dim = sensor_model.measurement_dim or dim
    measurements = np.zeros((num_trajectories, length, measurement_dim), dtype=float)
    state_dim = dim * 2 if include_velocity_in_label else dim
    labels = np.zeros((num_trajectories, length, state_dim), dtype=float)

    process_noise_model = None
    if laplace_scale > 0.0:
        process_noise_model = LaplaceNoise(scale=laplace_scale)

    for idx in range(num_trajectories):
        initial_position = np.zeros(dim)
        initial_velocity = rng.normal(velocity_mean, velocity_std, size=dim)

        target = Target(
            target_id=idx,
            initial_position=initial_position,
            initial_velocity=initial_velocity,
            process_noise_model=process_noise_model,
        )
        target.set_dynamics_model(motion_model)
        sensor = sensor_model

        measurements[idx, 0, :] = sensor.measure(target.position)
        if include_velocity_in_label:
            labels[idx, 0, :] = np.concatenate([target.position, target.velocity])
        else:
            labels[idx, 0, :] = target.position

        for step in range(1, length):
            if dim == 2 and turn_rate_std > 0.0:
                dtheta = rng.normal(0.0, turn_rate_std)
                c, s = np.cos(dtheta), np.sin(dtheta)
                vx, vy = target.velocity[0], target.velocity[1]
                target.velocity[0] = c * vx - s * vy
                target.velocity[1] = s * vx + c * vy

            if velocity_noise_std > 0.0:
                target.velocity += rng.normal(0.0, velocity_noise_std, size=dim)

            target.update(dt)
            measurements[idx, step, :] = sensor.measure(target.position)
            if include_velocity_in_label:
                labels[idx, step, :] = np.concatenate([target.position, target.velocity])
            else:
                labels[idx, step, :] = target.position

    return measurements, labels
