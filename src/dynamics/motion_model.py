"""Motion models for target/robot dynamics."""

from __future__ import annotations

import numpy as np


class ConstantVelocityModel:
    """Constant velocity motion model for EKF in 2D or 3D.

    State is [x, y, vx, vy] for 2D or [x, y, z, vx, vy, vz] for 3D.
    """

    def __init__(self, dim: int = 2):
        if dim not in (2, 3):
            raise ValueError("dim must be 2 or 3")
        self.dim = dim
        self.state_dim = dim * 2

    def process_model(
        self, state: np.ndarray, control_input: np.ndarray | None, dt: float
    ) -> np.ndarray:
        pos = state[: self.dim]
        vel = state[self.dim :]
        pos_next = pos + vel * dt
        return np.concatenate([pos_next, vel]).astype(float)

    def update(
        self,
        position: np.ndarray,
        velocity: np.ndarray,
        control_input: np.ndarray | None,
        dt: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Update position/velocity for Target dynamics."""
        position = np.array(position, dtype=float)
        velocity = np.array(velocity, dtype=float)

        if control_input is not None:
            velocity = velocity + np.array(control_input, dtype=float) * dt

        position = position + velocity * dt
        return position, velocity

    def process_jacobian(
        self, state: np.ndarray, control_input: np.ndarray | None, dt: float
    ) -> np.ndarray:
        F = np.eye(self.state_dim)
        F[: self.dim, self.dim :] = np.eye(self.dim) * dt
        return F

