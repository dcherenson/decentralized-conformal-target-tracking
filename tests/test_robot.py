"""Unit tests for Robot class."""

import pytest
import numpy as np
from src.agents.robot import Robot
from src.sensors.sensor import Sensor, GaussianNoise


def test_robot_creation():
    """Test basic robot creation."""
    robot = Robot(robot_id=1, initial_position=np.array([0.0, 0.0]))
    assert robot.id == 1
    assert np.allclose(robot.position, [0.0, 0.0])
    assert len(robot.neighbors) == 0


def test_add_neighbor():
    """Test adding neighbors to robot."""
    robot = Robot(robot_id=1, initial_position=np.array([0.0, 0.0]))
    robot.add_neighbor(2)
    robot.add_neighbor(3)
    assert len(robot.neighbors) == 2
    assert 2 in robot.neighbors
    assert 3 in robot.neighbors


def test_add_sensor():
    """Test adding sensors to robot."""
    robot = Robot(robot_id=1, initial_position=np.array([0.0, 0.0]))
    sensor = Sensor(sensor_id=1, noise_model=GaussianNoise(std=0.1))
    robot.add_sensor(sensor)
    assert sensor.id in robot.sensors
    assert robot.get_sensor(1) == sensor


def test_robot_dynamics():
    """Test robot dynamics update."""
    robot = Robot(robot_id=1, 
                 initial_position=np.array([0.0, 0.0]),
                 initial_velocity=np.array([1.0, 1.0]))
    
    # Update with no control input
    robot.update_dynamics(dt=1.0)
    assert np.allclose(robot.position, [1.0, 1.0])
