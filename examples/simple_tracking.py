"""
Simple example of using the decentralized conformal target tracking system.

This example demonstrates:
1. Creating robots with sensors
2. Creating targets with dynamics
3. Taking measurements
4. Tracking targets using different tracking modules
"""

import numpy as np
from src.agents.robot import Robot
from src.targets.target import Target
from src.sensors.sensor import Sensor, GaussianNoise
from src.tracking.ekf_tracker import EKFTracker


def main():
    # Create a robot at position [0, 0]
    robot = Robot(robot_id=1, initial_position=np.array([0.0, 0.0]))
    print(f"Created {robot}")
    
    # Create a sensor with Gaussian noise
    sensor = Sensor(sensor_id=1, noise_model=GaussianNoise(std=0.1))
    robot.add_sensor(sensor)
    print(f"Added sensor to robot")
    
    # Create a target at position [10, 10] with velocity [1, 1]
    target = Target(target_id=1, 
                   initial_position=np.array([10.0, 10.0]),
                   initial_velocity=np.array([1.0, 1.0]))
    print(f"Created {target}")
    
    # Simulate target motion and measurements
    dt = 0.1
    num_steps = 10
    
    print("\nSimulating target tracking:")
    print("-" * 50)
    
    for step in range(num_steps):
        # Update target dynamics
        target.update(dt)
        
        # Robot takes a measurement
        measurement = robot.measure_target(target.position)
        
        print(f"Step {step+1}: Target at {target.position}, Measured at {measurement}")
    
    print("\nTrajectory shape:", target.get_trajectory().shape)


if __name__ == "__main__":
    main()
