"""Main entry point for the decentralized conformal target tracking system."""

import numpy as np
from src.agents.robot import Robot
from src.targets.target import Target
from src.sensors.sensor import Sensor, GaussianNoise
from src.tracking.ekf_tracker import EKFTracker


def main():
    """Main function - run your simulations here."""
    print("Decentralized Conformal Target Tracking System")
    print("=" * 50)
    print("\nSee examples/ directory for usage examples.")
    print("Run: python examples/simple_tracking.py")


if __name__ == "__main__":
    main()

