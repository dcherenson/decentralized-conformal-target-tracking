"""Main entry point for the decentralized conformal target tracking system."""

import numpy as np
from scipy.stats import norm
import random
import networkx as nx

from examples.generate_random_trajectories import generate_random_trajectories
from examples.plot_trajectories import (
    plot_trajectories,
    plot_simulation_with_ellipses,
    plot_position_time_with_sigma,
)
from src.targets.target import Target
from src.agents.robot import Robot
from src.sensors.sensor import RangeBearingSensor
from src.dynamics.motion_model import ConstantVelocityModel
from src.tracking.ekf_tracker import EKFTracker
from src.tracking.conformal_prediction import ConformalPredictionModule


def generate_random_connected_graph(n, p_extra=0.1):
    """
    Generates a connected graph with N nodes and random edges.
    
    Args:
        n (int): Number of nodes.
        p_extra (float): Probability of adding an extra edge between any 
                         unconnected pair after the spanning tree is built.
    """
    if n <= 1:
        return nx.complete_graph(n)

    # 1. Create the graph and add nodes
    G = nx.Graph()
    G.add_nodes_from(range(n))

    # 2. Guarantee Connectivity (Random Spanning Tree)
    # We shuffle nodes to ensure the 'backbone' isn't just 0-1-2-3...
    nodes_list = list(G.nodes())
    random.shuffle(nodes_list)
    
    # Connect node i to i+1 to form a random path (backbone)
    for i in range(n - 1):
        G.add_edge(nodes_list[i], nodes_list[i + 1])

    # 3. Add extra random edges
    # Iterate through all possible pairs
    for i in range(n):
        for j in range(i + 1, n):
            # If edge doesn't exist, add it with probability p_extra
            if not G.has_edge(i, j):
                if random.random() < p_extra:
                    G.add_edge(i, j)
    
    return nx.adjacency_matrix(G).toarray()

def get_metropolis_weights(adj_matrix):
    n = adj_matrix.shape[0]
    degrees = np.sum(adj_matrix, axis=1) # Vector of degrees
    W = np.zeros((n, n))

    # 1. Fill off-diagonal weights
    for i in range(n):
        for j in range(i + 1, n):
            if adj_matrix[i][j] == 1:
                # Weight is 1 / max(degree_i, degree_j)
                weight = 1.0 / max(degrees[i], degrees[j])
                W[i][j] = weight
                W[j][i] = weight # Keep it symmetric!

    # 2. Fill diagonal (Self-loops)
    # The diagonal must make the row sum to 1
    for i in range(n):
        W[i][i] = 1.0 - np.sum(W[i, :])
        
    return W


def main():
    """Main function - create agents and generate calibration data."""
    # Fixed seed for reproducibility across runs.
    base_seed = 1338
    np.random.seed(base_seed)

    # Shared sensor noise model and motion model for all agents.
    range_noise_std = 0.2
    bearing_noise_std = 0.05

    def range_bearing_noise_model(dim: int) -> np.ndarray:
        # Laplace noise model for range and bearing (theta) only.
        if dim < 2:
            raise ValueError("Range-bearing noise model requires dim >= 2")
        noise = np.zeros(dim, dtype=float)
        noise[0] = np.random.randn() * range_noise_std
        noise[1] = np.random.randn() * bearing_noise_std
        return noise

    motion_model = ConstantVelocityModel(dim=2)

    # Process/measurement noise covariances for the EKF.
    process_noise = np.diag([0.01, 0.01, 0.1, 0.1])
    # Approximate noise for cos/sin with bearing noise std.
    measurement_noise = np.diag([
        range_noise_std**2,
        bearing_noise_std**2,
        bearing_noise_std**2,
    ])

    # Create agents with sensors, EKFs, and conformal modules.
    num_agents = 5
    alpha = 0.1

    # make a random fully connected graph adjacency matrix
    adjacency_matrix = generate_random_connected_graph(num_agents, p_extra=0.5)

    print(adjacency_matrix)

    weights = get_metropolis_weights(adjacency_matrix)

    print("Metropolis Weights:")
    print(weights)

    agents : list[Robot] = []
    for agent_id in range(5):
        robot = Robot(
            robot_id=agent_id,
            initial_position=np.array([float(agent_id)+2.5, 1.5], dtype=float)
        )
        # Each agent uses a range-bearing sensor with shared noise model.
        sensor = RangeBearingSensor(sensor_id=agent_id, noise_model=range_bearing_noise_model)
        robot.add_sensor(sensor)
        # EKF uses shared motion model and sensor-specific Jacobian.
        robot.set_tracking_module(
            EKFTracker(
                module_id=agent_id,
                state_dim=motion_model.state_dim,
                measurement_dim=3,
                measurement_model=sensor,
                measurement_noise=measurement_noise,
                motion_model=motion_model,
                process_noise=process_noise,
            )
        )
        # Conformal module per agent for calibration.
        robot.set_conformal_module(ConformalPredictionModule(module_id=agent_id, weights=weights[agent_id], alpha=alpha))
        agents.append(robot)

    for agent in agents:
        print(f"Created {agent} with sensor {list(agent.sensors.keys())}")

    # Generate calibration datasets for each agent.
    calibration_sets = []
    num_cal_data = 50
    dt = 0.1
    T = 50
    initial_velocity_std = 0.5
    initial_velocity_mean = 0.0
    for agent in agents:
        features, labels = generate_random_trajectories(
            num_trajectories=num_cal_data,
            length=T,
            dt=dt,
            dim=2,
            velocity_mean=initial_velocity_mean,
            velocity_std=initial_velocity_std,
            laplace_scale=0.0,
            include_velocity_in_label=False,
            velocity_noise_std=0.05,
            turn_rate_std=0.05,
            # Use the agent's sensor and shared motion model.
            sensor_model=RangeBearingSensor(sensor_id=agent.id, noise_model=range_bearing_noise_model),
            motion_model=motion_model,
            seed=base_seed + agent.id,
        )
        calibration_sets.append((features, labels))
        print(
            f"Agent {agent.id} calibration: features {features.shape}, labels {labels.shape}"
        )

    # Compute conformal quantiles using max Mahalanobis score per trajectory.
    for agent, (_, labels) in zip(agents, calibration_sets):
        ekf = agent.tracking_module
        if ekf is None:
            raise ValueError("Agent is missing EKF tracker")

        scores = []
        for trajectory in labels:
            initial_state = np.array([trajectory[0, 0], trajectory[0, 1], 0.0, 0.0])
            ekf.reset(initial_state=initial_state, initial_covariance=np.eye(4))

            max_score = 0.0
            for truth_pos in trajectory:
                measurement = next(iter(agent.sensors.values())).measure(truth_pos)
                ekf.predict(dt)
                ekf.update(measurement)

                cov_pos = ekf.covariance[:2, :2]
                mean_pos = ekf.state_estimate[:2]
                score = agent.conformal_module.compute_mahalanobis_score(
                    mean_pos,
                    cov_pos,
                    truth_pos,
                )
                if score > max_score:
                    max_score = score

            # Store max Mahalanobis score for this trajectory.
            scores.append(max_score)

        scores = np.array(scores, dtype=float)
        print(f"Agent {agent.id} scores: {scores}")
        quantile = agent.conformal_module.calibrate(scores)
        print(f"Agent {agent.id} conformal quantile (alpha={alpha}): {quantile:.4f}")
        agent.conformal_module.initialize_distributed_subgradient(quantile)

    # Save a plot of the first agent's calibration trajectories.
    plot_trajectories(
        calibration_sets[0][1],
        save_path="trajectories.png",
        show=False,
    )

    # Run a simulation of a single target and track with all agents.
    sim_steps = T
    # Use a fresh RNG for simulation so it changes each run.
    rng = np.random.default_rng()
    turn_rate_std = 0.05
    velocity_noise_std = 0.05
    target = Target(
        target_id=0,
        initial_position=np.zeros(2),
        # Match calibration initial velocity distribution.
        initial_velocity=np.array([0.0, 1.0]) #rng.normal(initial_velocity_mean, initial_velocity_std, size=2),
    )
    target.set_dynamics_model(motion_model)

    # Containers for truth, estimates, and covariance ellipses.
    truth = np.zeros((sim_steps, 2), dtype=float)
    estimates: dict[int, np.ndarray] = {}
    covariances: dict[int, np.ndarray] = {}

    sim_scores: dict[int, list[float]] = {}
    for agent in agents:
        ekf = agent.tracking_module
        if ekf is None:
            raise ValueError("Agent is missing EKF tracker")
        ekf.reset(initial_state=np.array([0.0, 0.0, 0.0, 0.0]), initial_covariance=np.eye(4))
        estimates[agent.id] = np.zeros((sim_steps, 2), dtype=float)
        covariances[agent.id] = np.zeros((sim_steps, 2, 2), dtype=float)
        sim_scores[agent.id] = []

    dcp_steps = 10

    avg_data_per_agent = num_cal_data
    for k in range(dcp_steps):
        agent_values = np.array([agent.conformal_module.dist_quantile_estimate for agent in agents])
        for agent in agents:
            agent.conformal_module.run_distributed_subgradient_step(agent_values, num_cal_data)

    print("Distributed subgradient quantile estimates:")
    for agent in agents:
        print(f"Agent {agent.id}: {agent.conformal_module.dist_quantile_estimate:.4f}")


    exit()

    # Simulate target motion and EKF updates per agent.
    for t in range(sim_steps):
        # Apply random turn to target velocity.
        dtheta = rng.normal(0.0, turn_rate_std)
        c, s = np.cos(dtheta), np.sin(dtheta)
        vx, vy = target.velocity[0], target.velocity[1]
        target.velocity[0] = c * vx - s * vy
        target.velocity[1] = s * vx + c * vy

        # Apply per-step velocity noise to match calibration settings.
        target.velocity += rng.normal(0.0, velocity_noise_std, size=2)

        target.update(dt)
        truth[t] = target.position

        for agent in agents:
            measurement = next(iter(agent.sensors.values())).measure(target.position)
            ekf = agent.tracking_module
            ekf.predict(dt)
            ekf.update(measurement)
            estimates[agent.id][t] = ekf.state_estimate[:2]
            covariances[agent.id][t] = ekf.covariance[:2, :2]
            score = agent.conformal_module.compute_mahalanobis_score(
                ekf.state_estimate[:2],
                ekf.covariance[:2, :2],
                truth[t],
            )
            sim_scores[agent.id].append(score)

    # Save simulation plot with nominal vs calibrated ellipses for first agent.
    agent0_scores = np.array(sim_scores[0], dtype=float)
    nominal_scale = float(norm.ppf(1 - alpha / 2))
    nominal_violation_idx = np.where(agent0_scores > nominal_scale)[0]
    plot_simulation_with_ellipses(
        truth=truth,
        estimates={0: estimates[0]},
        covariances={0: covariances[0]},
        quantiles={0: agents[0].conformal_module.quantile(alpha)},
        nominal_scale=nominal_scale,
        nominal_violations={0: nominal_violation_idx},
        agent_positions={0: agents[0].position},
        ellipse_step=10,
        save_path="simulation.png",
        show=False,
    )

    plot_position_time_with_sigma(
        truth=truth,
        estimates={0: estimates[0]},
        covariances={0: covariances[0]},
        quantiles={0: agents[0].conformal_module.quantile(alpha)},
        alpha=alpha,
        nominal_violations={0: nominal_violation_idx},
        save_path="position_time.png",
        show=False,
    )

    for agent in agents:
        quantile = agent.conformal_module.quantile(alpha)
        scores = np.array(sim_scores[agent.id], dtype=float)
        coverage = float(np.mean(scores <= quantile))
        # Compare nominal Gaussian vs calibrated scaling for this agent.
        nominal_scale = float(norm.ppf(1 - alpha / 2))
        uncalibrated_violations = int(np.sum(scores > nominal_scale))
        calibrated_violations = int(np.sum(scores > quantile))
        print(f"Agent {agent.id} simulation coverage: {coverage:.3f}")
        print(f"Agent {agent.id} sim score min/max: {scores.min():.4f}/{scores.max():.4f}")
        print(f"Agent {agent.id} quantile threshold: {quantile:.4f}")
        print(f"Agent {agent.id} nominal scale (z): {nominal_scale:.4f}")
        print(
            f"Agent {agent.id} violations (uncalibrated/calibrated): "
            f"{uncalibrated_violations}/{calibrated_violations}"
        )


if __name__ == "__main__":
    main()

