import numpy as np
from typing import List, Optional, Set, Dict
from src.sensors.sensor import Sensor


class Robot:
    """
    A robot agent with dynamics, target tracking module, and graph connectivity.
    """
    
    def __init__(self, robot_id: int, initial_position: np.ndarray, initial_velocity: Optional[np.ndarray] = None):
        """
        Initialize a robot.
        
        Args:
            robot_id: Unique identifier for this robot
            initial_position: Initial position vector (e.g., [x, y] or [x, y, z])
            initial_velocity: Initial velocity vector (defaults to zero)
        """
        self.id = robot_id
        self.position = np.array(initial_position, dtype=float)
        self.velocity = np.array(initial_velocity, dtype=float) if initial_velocity is not None else np.zeros_like(self.position)
        
        # Target tracking module
        self.tracking_module = None  # Can be set to a specific tracking algorithm
        self.tracked_targets = {}  # Dictionary to store tracked target states

        # Conformal prediction module
        self.conformal_module = None
        
        # Dynamics
        self.dynamics_model = None  # Can be set to a specific dynamics model
        self.control_input = np.zeros_like(self.position)
        
        # Sensors
        self.sensors: Dict[int, Sensor] = {}  # Dictionary of sensors by sensor ID
        
        # Graph connectivity
        self.neighbors: Set[int] = set()  # Set of neighbor robot IDs
        
    def add_neighbor(self, neighbor_id: int):
        """Add a neighbor to this robot's communication graph."""
        self.neighbors.add(neighbor_id)
    
    def remove_neighbor(self, neighbor_id: int):
        """Remove a neighbor from this robot's communication graph."""
        self.neighbors.discard(neighbor_id)
    
    def get_neighbors(self) -> List[int]:
        """Return list of neighbor robot IDs."""
        return list(self.neighbors)
    
    def set_tracking_module(self, tracking_module):
        """Set the target tracking module for this robot."""
        self.tracking_module = tracking_module

    def set_conformal_module(self, conformal_module):
        """Set the conformal prediction module for this robot."""
        self.conformal_module = conformal_module
    
    def set_dynamics_model(self, dynamics_model):
        """Set the dynamics model for this robot."""
        self.dynamics_model = dynamics_model
    
    def add_sensor(self, sensor: Sensor):
        """Add a sensor to this robot."""
        # Place the sensor at the robot's current position.
        if hasattr(sensor, "set_position"):
            sensor.set_position(self.position)
        self.sensors[sensor.id] = sensor
    
    def remove_sensor(self, sensor_id: int):
        """Remove a sensor from this robot."""
        if sensor_id in self.sensors:
            del self.sensors[sensor_id]
    
    def get_sensor(self, sensor_id: int) -> Optional[Sensor]:
        """Get a specific sensor by ID."""
        return self.sensors.get(sensor_id)
    
    def measure_target(self, target_position: np.ndarray, sensor_id: Optional[int] = None) -> np.ndarray:
        """
        Measure a target position using a sensor.
        
        Args:
            target_position: True position of the target
            sensor_id: ID of sensor to use (if None, use first available sensor)
            
        Returns:
            Noisy measurement
        """
        if sensor_id is None:
            # Use first available sensor
            if not self.sensors:
                raise ValueError("No sensors available on this robot")
            sensor = next(iter(self.sensors.values()))
        else:
            sensor = self.sensors.get(sensor_id)
            if sensor is None:
                raise ValueError(f"Sensor {sensor_id} not found on this robot")
        
        return sensor.measure(target_position)
    
    def update_dynamics(self, dt: float, control_input: Optional[np.ndarray] = None):
        """
        Update robot state based on dynamics model.
        
        Args:
            dt: Time step
            control_input: Control input vector (optional)
        """
        if control_input is not None:
            self.control_input = np.array(control_input, dtype=float)
        
        if self.dynamics_model is not None:
            self.position, self.velocity = self.dynamics_model.update(
                self.position, self.velocity, self.control_input, dt
            )
        else:
            # Simple integration if no dynamics model is set
            self.position += self.velocity * dt
            self.velocity += self.control_input * dt
    
    def update_tracking(self, measurements):
        """
        Update target tracking based on measurements.
        
        Args:
            measurements: Sensor measurements for target tracking
        """
        if self.tracking_module is not None:
            self.tracked_targets = self.tracking_module.update(measurements)
    
    def get_state(self) -> dict:
        """Return the current state of the robot."""
        return {
            'id': self.id,
            'position': self.position.copy(),
            'velocity': self.velocity.copy(),
            'neighbors': self.get_neighbors(),
            'tracked_targets': self.tracked_targets.copy()
        }
    
    def __repr__(self):
        return f"Robot(id={self.id}, pos={self.position}, neighbors={len(self.neighbors)})"
