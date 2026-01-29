import numpy as np
from typing import Optional


class Target:
    """
    A target with dynamics model for motion prediction.
    """
    
    def __init__(self, target_id: int, initial_position: np.ndarray, initial_velocity: Optional[np.ndarray] = None):
        """
        Initialize a target.
        
        Args:
            target_id: Unique identifier for this target
            initial_position: Initial position vector (e.g., [x, y] or [x, y, z])
            initial_velocity: Initial velocity vector (defaults to zero)
        """
        self.id = target_id
        self.position = np.array(initial_position, dtype=float)
        self.velocity = np.array(initial_velocity, dtype=float) if initial_velocity is not None else np.zeros_like(self.position)
        self.acceleration = np.zeros_like(self.position)
        
        # Dynamics model
        self.dynamics_model = None  # Can be set to a specific dynamics model
        
        # State history for tracking
        self.trajectory = [self.position.copy()]
        
    def set_dynamics_model(self, dynamics_model):
        """Set the dynamics model for this target."""
        self.dynamics_model = dynamics_model
    
    def update(self, dt: float, control_input: Optional[np.ndarray] = None):
        """
        Update target state based on dynamics model.
        
        Args:
            dt: Time step
            control_input: Optional control/disturbance input
        """
        if self.dynamics_model is not None:
            self.position, self.velocity = self.dynamics_model.update(
                self.position, self.velocity, control_input, dt
            )
        else:
            # Simple constant velocity motion if no dynamics model is set
            self.position += self.velocity * dt
            
        # Store trajectory
        self.trajectory.append(self.position.copy())
    
    def get_state(self) -> dict:
        """Return the current state of the target."""
        return {
            'id': self.id,
            'position': self.position.copy(),
            'velocity': self.velocity.copy()
        }
    
    def get_trajectory(self) -> np.ndarray:
        """Return the trajectory history as a numpy array."""
        return np.array(self.trajectory)
    
    def __repr__(self):
        return f"Target(id={self.id}, pos={self.position}, vel={self.velocity})"
