import numpy as np
from typing import Optional, Callable


class Sensor:
    """
    A sensor that takes true state and outputs noisy measurements via transformation function h(x,w).
    """
    
    def __init__(self, sensor_id: int, 
                 noise_model: Optional[Callable] = None, 
                 measurement_dim: Optional[int] = None,
                 measurement_function: Optional[Callable] = None,
                 measurement_jacobian: Optional[Callable] = None,
                 position: Optional[np.ndarray] = None):
        """
        Initialize a sensor.
        
        Args:
            sensor_id: Unique identifier for this sensor
            noise_model: Function that generates noise (takes measurement_dim, returns noise vector)
            measurement_dim: Dimension of measurements (if None, inferred from truth value)
            measurement_function: Transformation function h(x, w) that takes state and noise, returns measurement
            measurement_jacobian: Jacobian of measurement model H(x) for EKF
        """
        self.id = sensor_id
        self.noise_model = noise_model
        self.measurement_dim = measurement_dim
        self.measurement_function = measurement_function  # h(x, w)
        self.measurement_jacobian = measurement_jacobian
        self.measurement_history = []
        # Sensor position in world coordinates.
        self.position = (
            np.array(position, dtype=float)
            if position is not None
            else np.zeros(2, dtype=float)
        )
        
    def measure(self, truth_value: np.ndarray) -> np.ndarray:
        """
        Take a measurement of the truth value with noise using transformation h(x,w).
        
        Args:
            truth_value: True state x to be measured
            
        Returns:
            Noisy measurement y = h(x, w)
        """
        truth_value = np.array(truth_value, dtype=float)
        
        # Generate noise w
        noise_dim = self.measurement_dim or truth_value.shape[0]
        if self.noise_model is not None:
            noise = self.noise_model(noise_dim)
        else:
            # Default: Gaussian noise with std=0.1
            noise = np.random.randn(noise_dim) * 0.1
        
        # Apply measurement transformation h(x, w)
        if self.measurement_function is not None:
            measurement = self.measurement_function(truth_value, noise, self.position)
        else:
            # Default: linear additive measurement y = x + w
            measurement = truth_value + noise
        
        self.measurement_history.append(measurement.copy())
        
        return measurement
    
    def set_noise_model(self, noise_model: Callable):
        """Set the noise model for this sensor."""
        self.noise_model = noise_model
    
    def set_measurement_function(self, measurement_function: Callable):
        """
        Set the measurement transformation function h(x, w).
        
        Args:
            measurement_function: Callable that takes (state, noise) and returns measurement
        """
        self.measurement_function = measurement_function

    def set_position(self, position: np.ndarray):
        """Set the sensor position in world coordinates."""
        self.position = np.array(position, dtype=float)

    def set_measurement_jacobian(self, measurement_jacobian: Callable):
        """
        Set the measurement Jacobian H(x) for EKF.

        Args:
            measurement_jacobian: Callable that takes state and returns Jacobian
        """
        self.measurement_jacobian = measurement_jacobian
    
    def get_measurement_history(self) -> np.ndarray:
        """Return the measurement history as a numpy array."""
        return np.array(self.measurement_history) if self.measurement_history else np.array([])
    
    def clear_history(self):
        """Clear the measurement history."""
        self.measurement_history = []
    
    def __repr__(self):
        return f"Sensor(id={self.id}, measurements={len(self.measurement_history)})"


class LaplaceNoise:
    """Laplace noise model for sensors."""
    
    def __init__(self, scale: float = 1.0, mean: float = 0.0):
        """
        Initialize Laplace noise model.
        
        Args:
            scale: Scale parameter (b) of Laplace distribution
            mean: Mean (mu) of Laplace distribution
        """
        self.scale = scale
        self.mean = mean
    
    def __call__(self, dim: int) -> np.ndarray:
        """Generate Laplace noise vector."""
        return np.random.laplace(self.mean, self.scale, dim)

class GaussianNoise:
    """Gaussian noise model for sensors."""
    
    def __init__(self, std: float = 1.0, mean: float = 0.0):
        """
        Initialize Gaussian noise model.
        
        Args:
            std: Standard deviation of noise
            mean: Mean of noise
        """
        self.std = std
        self.mean = mean
    
    def __call__(self, dim: int) -> np.ndarray:
        """Generate Gaussian noise vector."""
        return np.random.randn(dim) * self.std + self.mean


class UniformNoise:
    """Uniform noise model for sensors."""
    
    def __init__(self, low: float = -1.0, high: float = 1.0):
        """
        Initialize uniform noise model.
        
        Args:
            low: Lower bound of uniform distribution
            high: Upper bound of uniform distribution
        """
        self.low = low
        self.high = high
    
    def __call__(self, dim: int) -> np.ndarray:
        """Generate uniform noise vector."""
        return np.random.uniform(self.low, self.high, dim)


class RangeSensor(Sensor):
    """Range-only sensor measuring distance to origin in 2D."""

    def __init__(self, sensor_id: int, noise_model: Optional[Callable] = None,
                 position: Optional[np.ndarray] = None):
        def measurement_function(
            state: np.ndarray,
            noise: np.ndarray,
            sensor_position: np.ndarray,
        ) -> np.ndarray:
            dx, dy = state[0] - sensor_position[0], state[1] - sensor_position[1]
            r = np.sqrt(dx * dx + dy * dy)
            return np.array([r + noise[0]], dtype=float)

        def measurement_jacobian(
            state: np.ndarray,
            sensor_position: np.ndarray,
        ) -> np.ndarray:
            dx, dy = state[0] - sensor_position[0], state[1] - sensor_position[1]
            r = np.sqrt(x * x + y * y)
            if r == 0.0:
                return np.zeros((1, state.shape[0]))
            H = np.zeros((1, state.shape[0]))
            H[0, 0] = dx / r
            H[0, 1] = dy / r
            return H

        super().__init__(
            sensor_id=sensor_id,
            noise_model=noise_model,
            measurement_dim=1,
            measurement_function=measurement_function,
            measurement_jacobian=measurement_jacobian,
            position=position,
        )


class RangeBearingSensor(Sensor):
    """Range-bearing sensor measuring distance and angle to origin in 2D."""

    def __init__(self, sensor_id: int, noise_model: Optional[Callable] = None,
                 position: Optional[np.ndarray] = None,
                 max_range: float = 3.0,
                 far_noise_scale: float = 0.00001):
        # Noise scaling for far targets.
        self.max_range = max_range
        self.far_noise_scale = far_noise_scale

        def measurement_function(
            state: np.ndarray,
            noise: np.ndarray,
            sensor_position: np.ndarray,
        ) -> np.ndarray:
            dx, dy = state[0] - sensor_position[0], state[1] - sensor_position[1]
            r = np.sqrt(dx * dx + dy * dy)
            theta = np.arctan2(dy, dx)
            # Add noise to range and bearing, then emit range, cos(bearing), sin(bearing).
            theta_noisy = theta + noise[1]
            return np.array(
                [
                    r + noise[0],
                    np.cos(theta_noisy),
                    np.sin(theta_noisy),
                ],
                dtype=float,
            )

        def measurement_jacobian(
            state: np.ndarray,
            sensor_position: np.ndarray,
        ) -> np.ndarray:
            dx, dy = state[0] - sensor_position[0], state[1] - sensor_position[1]
            r2 = dx * dx + dy * dy
            r = np.sqrt(r2)
            if r == 0.0:
                return np.zeros((3, state.shape[0]))
            H = np.zeros((3, state.shape[0]))
            dtheta_dx = -dy / r2
            dtheta_dy = dx / r2
            theta = np.arctan2(dy, dx)

            # Range derivative.
            H[0, 0] = dx / r
            H[0, 1] = dy / r

            # cos(theta) derivative.
            H[1, 0] = -np.sin(theta) * dtheta_dx
            H[1, 1] = -np.sin(theta) * dtheta_dy

            # sin(theta) derivative.
            H[2, 0] = np.cos(theta) * dtheta_dx
            H[2, 1] = np.cos(theta) * dtheta_dy
            return H

        super().__init__(
            sensor_id=sensor_id,
            noise_model=noise_model,
            measurement_dim=3,
            measurement_function=measurement_function,
            measurement_jacobian=measurement_jacobian,
            position=position,
        )

    def measure(self, truth_value: np.ndarray) -> np.ndarray:
        """Override measure to generate noise only on range and bearing."""
        truth_value = np.array(truth_value, dtype=float)

        if self.noise_model is not None:
            noise = self.noise_model(2)
        else:
            noise = np.random.randn(2) * 0.1

        # Increase measurement noise for far targets using tanh scaling.
        dx, dy = truth_value[0] - self.position[0], truth_value[1] - self.position[1]
        r = np.sqrt(dx * dx + dy * dy)
        if r > self.max_range:
            noise_scale = 1.0 + self.far_noise_scale * (np.tanh((r - self.max_range) / self.max_range) + 1.0)
            noise = noise * noise_scale

        measurement = self.measurement_function(truth_value, noise, self.position)
        self.measurement_history.append(measurement.copy())
        return measurement


class PositionSensor(Sensor):
    """Position sensor measuring x, y from state [x, y, ...]."""

    def __init__(self, sensor_id: int, noise_model: Optional[Callable] = None,
                 position: Optional[np.ndarray] = None):
        def measurement_function(
            state: np.ndarray,
            noise: np.ndarray,
            sensor_position: np.ndarray,
        ) -> np.ndarray:
            x, y = state[0], state[1]
            return np.array([x + noise[0], y + noise[1]], dtype=float)

        def measurement_jacobian(
            state: np.ndarray,
            sensor_position: np.ndarray,
        ) -> np.ndarray:
            H = np.zeros((2, state.shape[0]))
            H[0, 0] = 1.0
            H[1, 1] = 1.0
            return H

        super().__init__(
            sensor_id=sensor_id,
            noise_model=noise_model,
            measurement_dim=2,
            measurement_function=measurement_function,
            measurement_jacobian=measurement_jacobian,
            position=position,
        )


class VelocitySensor(Sensor):
    """Velocity-only sensor measuring vx, vy from state [x, y, vx, vy]."""

    def __init__(self, sensor_id: int, noise_model: Optional[Callable] = None,
                 position: Optional[np.ndarray] = None):
        def measurement_function(
            state: np.ndarray,
            noise: np.ndarray,
            sensor_position: np.ndarray,
        ) -> np.ndarray:
            if state.shape[0] < 4:
                raise ValueError("VelocitySensor requires state with at least 4 elements")
            vx, vy = state[2], state[3]
            return np.array([vx + noise[0], vy + noise[1]], dtype=float)

        def measurement_jacobian(
            state: np.ndarray,
            sensor_position: np.ndarray,
        ) -> np.ndarray:
            if state.shape[0] < 4:
                raise ValueError("VelocitySensor requires state with at least 4 elements")
            H = np.zeros((2, state.shape[0]))
            H[0, 2] = 1.0
            H[1, 3] = 1.0
            return H

        super().__init__(
            sensor_id=sensor_id,
            noise_model=noise_model,
            measurement_dim=2,
            measurement_function=measurement_function,
            measurement_jacobian=measurement_jacobian,
            position=position,
        )
