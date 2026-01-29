import numpy as np
from typing import Optional, Callable
from src.tracking.base_tracker import TrackingModule


class EKFTracker(TrackingModule):
    """
    Extended Kalman Filter for target tracking.
    """
    
    def __init__(self, module_id: int, state_dim: int, measurement_dim: int,
                 process_model: Callable,
                 measurement_model: Callable,
                 process_jacobian: Callable,
                 measurement_jacobian: Callable,
                 process_noise: np.ndarray,
                 measurement_noise: np.ndarray):
        """
        Initialize EKF tracker.
        
        Args:
            module_id: Unique identifier
            state_dim: Dimension of state
            measurement_dim: Dimension of measurements
            process_model: Function f(x, u, dt) that returns predicted state
            measurement_model: Function h(x) that returns predicted measurement
            process_jacobian: Function F(x, u, dt) that returns Jacobian of f
            measurement_jacobian: Function H(x) that returns Jacobian of h
            process_noise: Process noise covariance matrix Q
            measurement_noise: Measurement noise covariance matrix R
        """
        super().__init__(module_id, state_dim, measurement_dim)
        self.process_model = process_model
        self.measurement_model = measurement_model
        self.process_jacobian = process_jacobian
        self.measurement_jacobian = measurement_jacobian
        self.Q = process_noise
        self.R = measurement_noise
        
    def predict(self, dt: float, control_input: Optional[np.ndarray] = None):
        """
        EKF prediction step.
        
        Args:
            dt: Time step
            control_input: Optional control input
        """
        # Predict state: x_k|k-1 = f(x_k-1|k-1, u_k, dt)
        self.state_estimate = self.process_model(self.state_estimate, control_input, dt)
        
        # Predict covariance: P_k|k-1 = F_k * P_k-1|k-1 * F_k^T + Q_k
        F = self.process_jacobian(self.state_estimate, control_input, dt)
        self.covariance = F @ self.covariance @ F.T + self.Q
        
    def update(self, measurement: np.ndarray):
        """
        EKF update step.
        
        Args:
            measurement: Measurement vector z_k
        """
        measurement = np.array(measurement, dtype=float)
        
        # Innovation: y_k = z_k - h(x_k|k-1)
        predicted_measurement = self.measurement_model(self.state_estimate)
        innovation = measurement - predicted_measurement
        
        # Innovation covariance: S_k = H_k * P_k|k-1 * H_k^T + R_k
        H = self.measurement_jacobian(self.state_estimate)
        S = H @ self.covariance @ H.T + self.R
        
        # Kalman gain: K_k = P_k|k-1 * H_k^T * S_k^-1
        K = self.covariance @ H.T @ np.linalg.inv(S)
        
        # Update state: x_k|k = x_k|k-1 + K_k * y_k
        self.state_estimate = self.state_estimate + K @ innovation
        
        # Update covariance: P_k|k = (I - K_k * H_k) * P_k|k-1
        I = np.eye(self.state_dim)
        self.covariance = (I - K @ H) @ self.covariance
        
    def __repr__(self):
        return f"EKFTracker(id={self.id}, state_dim={self.state_dim})"
