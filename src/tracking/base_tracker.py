import numpy as np
from typing import Optional
from abc import ABC, abstractmethod


class TrackingModule(ABC):
    """
    Abstract base class for target tracking modules.
    """
    
    def __init__(self, module_id: int, state_dim: int, measurement_dim: int):
        """
        Initialize tracking module.
        
        Args:
            module_id: Unique identifier for this tracking module
            state_dim: Dimension of the state vector
            measurement_dim: Dimension of the measurement vector
        """
        self.id = module_id
        self.state_dim = state_dim
        self.measurement_dim = measurement_dim
        self.state_estimate = np.zeros(state_dim)
        self.covariance = np.eye(state_dim)
        
    @abstractmethod
    def predict(self, dt: float, control_input: Optional[np.ndarray] = None):
        """
        Prediction step for the tracking algorithm.
        
        Args:
            dt: Time step
            control_input: Optional control input
        """
        pass
    
    @abstractmethod
    def update(self, measurement: np.ndarray):
        """
        Update step for the tracking algorithm.
        
        Args:
            measurement: Measurement vector
        """
        pass
    
    def get_state_estimate(self) -> np.ndarray:
        """Return the current state estimate."""
        return self.state_estimate.copy()
    
    def get_covariance(self) -> np.ndarray:
        """Return the current covariance estimate."""
        return self.covariance.copy()
    
    def reset(self, initial_state: Optional[np.ndarray] = None, initial_covariance: Optional[np.ndarray] = None):
        """Reset the tracking module to initial conditions."""
        if initial_state is not None:
            self.state_estimate = np.array(initial_state, dtype=float)
        else:
            self.state_estimate = np.zeros(self.state_dim)
            
        if initial_covariance is not None:
            self.covariance = np.array(initial_covariance, dtype=float)
        else:
            self.covariance = np.eye(self.state_dim)
