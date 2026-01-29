import numpy as np
from typing import Optional, Callable


class Sensor:
    """
    A sensor that takes true state and outputs noisy measurements via transformation function h(x,w).
    """
    
    def __init__(self, sensor_id: int, 
                 noise_model: Optional[Callable] = None, 
                 measurement_dim: Optional[int] = None,
                 measurement_function: Optional[Callable] = None):
        """
        Initialize a sensor.
        
        Args:
            sensor_id: Unique identifier for this sensor
            noise_model: Function that generates noise (takes measurement_dim, returns noise vector)
            measurement_dim: Dimension of measurements (if None, inferred from truth value)
            measurement_function: Transformation function h(x, w) that takes state and noise, returns measurement
        """
        self.id = sensor_id
        self.noise_model = noise_model
        self.measurement_dim = measurement_dim
        self.measurement_function = measurement_function  # h(x, w)
        self.measurement_history = []
        
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
        if self.noise_model is not None:
            noise = self.noise_model(truth_value.shape[0])
        else:
            # Default: Gaussian noise with std=0.1
            noise = np.random.randn(*truth_value.shape) * 0.1
        
        # Apply measurement transformation h(x, w)
        if self.measurement_function is not None:
            measurement = self.measurement_function(truth_value, noise)
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
    
    def get_measurement_history(self) -> np.ndarray:
        """Return the measurement history as a numpy array."""
        return np.array(self.measurement_history) if self.measurement_history else np.array([])
    
    def clear_history(self):
        """Clear the measurement history."""
        self.measurement_history = []
    
    def __repr__(self):
        return f"Sensor(id={self.id}, measurements={len(self.measurement_history)})"


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
