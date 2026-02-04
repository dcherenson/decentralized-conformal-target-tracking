import numpy as np
from typing import Any, Optional
from src.tracking.base_tracker import TrackingModule
from src.sensors.sensor import Sensor


class EKFTracker(TrackingModule):
    """
    Extended Kalman Filter for target tracking.
    """
    
    def __init__(self, module_id: int, state_dim: int, measurement_dim: int,
                 measurement_model: Sensor = None,
                 process_noise: Optional[np.ndarray] = None,
                 measurement_noise: Optional[np.ndarray] = None,
                 motion_model: Any = None):
        """
        Initialize EKF tracker.
        
        Args:
            module_id: Unique identifier
            state_dim: Dimension of state
            measurement_dim: Dimension of measurements
            measurement_model: Measurement model object with h(x) and jacobian(x)
            process_noise: Process noise covariance matrix Q
            measurement_noise: Measurement noise covariance matrix R
            motion_model: Motion model object with process_model and process_jacobian
        """
        super().__init__(module_id, state_dim, measurement_dim)
        if motion_model is None:
            raise ValueError("motion_model is required")
        if measurement_model is None:
            raise ValueError("measurement_model is required")

        # Process model (f, F) from motion model.
        self.process_model = motion_model.process_model
        self.process_jacobian = motion_model.process_jacobian

        # Measurement model (h, H) from measurement model object.
        self.measurement_model = measurement_model
        self.measurement_jacobian = measurement_model.measurement_jacobian
        self.Q = process_noise if process_noise is not None else np.eye(state_dim)
        self.R = (
            measurement_noise
            if measurement_noise is not None
            else np.eye(measurement_dim)
        )
        
    def predict(self, dt: float, control_input: Optional[np.ndarray] = None):
        """
        EKF prediction step.
        
        Args:
            dt: Time step
            control_input: Optional control input
        """
        # Predict state: x_k|k-1 = f(x_k-1|k-1, u_k, dt)
        if self.process_model is None or self.process_jacobian is None:
            raise ValueError("process_model and process_jacobian must be set")

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
        if self.measurement_model is None or self.measurement_jacobian is None:
            raise ValueError("measurement_model and measurement_jacobian must be set")

        # Use noise-free measurement model for EKF prediction.
        noise = np.zeros(self.measurement_dim)
        predicted_measurement = self.measurement_model.measurement_function(
            self.state_estimate,
            noise,
        )

        H = self.measurement_jacobian(self.state_estimate)
        R = self.R
        innovation = measurement - predicted_measurement
        
        # Innovation covariance: S_k = H_k * P_k|k-1 * H_k^T + R_k
        S = H @ self.covariance @ H.T + R
        
        # Kalman gain: K_k = P_k|k-1 * H_k^T * S_k^-1
        K = self.covariance @ H.T @ np.linalg.inv(S)
        
        # Update state: x_k|k = x_k|k-1 + K_k * y_k
        self.state_estimate = self.state_estimate + K @ innovation
        
        # Update covariance: P_k|k = (I - K_k * H_k) * P_k|k-1
        I = np.eye(self.state_dim)
        self.covariance = (I - K @ H) @ self.covariance
        
    def __repr__(self):
        return f"EKFTracker(id={self.id}, state_dim={self.state_dim})"
