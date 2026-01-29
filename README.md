# Decentralized Conformal Prediction for Target Tracking

A modular system for decentralized target tracking with conformal prediction and multi-robot collaboration.

## Features

- **Multi-Robot System**: Robots with dynamics, sensors, and graph connectivity
- **Target Tracking**: Modular tracking modules (EKF, Neural Networks, LSTM)
- **Flexible Sensors**: Configurable sensors with custom noise models and measurement functions h(x,w)
- **Target Dynamics**: Targets with pluggable dynamics models

## Project Structure

```
decentralized-conformal-target-tracking/
├── src/
│   ├── agents/          # Robot/agent implementations
│   ├── tracking/        # Tracking algorithms (EKF, NN, LSTM)
│   ├── sensors/         # Sensor models and noise
│   ├── targets/         # Target dynamics
│   └── dynamics/        # Dynamics models (future)
├── examples/            # Usage examples
├── tests/              # Unit tests
├── notebooks/          # Jupyter notebooks
└── main.py             # Entry point
```

## Installation

```bash
# Install dependencies with uv
uv sync
```

## Quick Start

```python
from src.agents.robot import Robot
from src.targets.target import Target
from src.sensors.sensor import Sensor, GaussianNoise
import numpy as np

# Create a robot
robot = Robot(robot_id=1, initial_position=np.array([0.0, 0.0]))

# Add a sensor
sensor = Sensor(sensor_id=1, noise_model=GaussianNoise(std=0.1))
robot.add_sensor(sensor)

# Create a target
target = Target(target_id=1, 
               initial_position=np.array([10.0, 10.0]),
               initial_velocity=np.array([1.0, 1.0]))

# Simulate
dt = 0.1
target.update(dt)
measurement = robot.measure_target(target.position)
```

## Examples

See the `examples/` directory for detailed examples:

```bash
python examples/simple_tracking.py
```

## Testing

```bash
pytest tests/
```

## Development

This project uses:
- Python 3.x
- NumPy for numerical computations
- PyTorch for neural network tracking modules
- pytest for testing

