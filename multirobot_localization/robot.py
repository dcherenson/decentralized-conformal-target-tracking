"""Ground-truth robot model used by the cooperative localization simulator."""

from __future__ import annotations

from math import cos, sin

import numpy as np

from agent_classes import DEFAULT_AGENT_CLASS_PROFILES, AgentClass, normalize_agent_class


class Robot:
    """Minimal planar robot used by the original GS-CI simulations."""

    def __init__(
        self,
        _position,
        _theta: float = 0.0,
        agent_class: AgentClass | str = AgentClass.CLASS_A_UGV,
        epsilon: float = 0.1,
    ):
        position = np.asarray(_position, dtype=float).reshape(-1)
        if position.size != 2:
            raise ValueError("Robot ground-truth position must be 2D")

        self.position = position.copy()
        self.theta = float(_theta)
        self.agent_class = normalize_agent_class(agent_class)
        self.epsilon = float(epsilon)
        self.class_profile = DEFAULT_AGENT_CLASS_PROFILES[self.agent_class]

    def motion_propagation(self, odometry_input, dt: float):
        """Advance the true robot state using the supplied odometry input."""

        v, omega = odometry_input
        self.theta += float(omega) * dt
        self.position[0] += cos(self.theta) * float(v) * dt
        self.position[1] += sin(self.theta) * float(v) * dt
