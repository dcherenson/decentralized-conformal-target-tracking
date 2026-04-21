# Multirobot Localization Simulation

This is the simulation code for the paper "Resilient and Consistent Multirobot Cooperative Localization with Covariance Intersection" submitted to *IEEE Transaction on Robotics*.



## Multirobot Cooperative Localization Algorithm based on Covariance Intersection

This is our algorithm developed in the paper. The proposed algorithm contains 3 steps:

### Motion propagation update

### Observation update

### Communication update



## Other Multirobot Cooperative Localization Algorithms

We simulate 4 other algorithms for comparision. We rename and classify them to emphasize the structural difference. For algorithms in which each robot only tracks its own spatial state, we call them local state (LS) algorithms, in order to distinguish from our algorithm in which the spatial state of the entire robot team is tracked in each robot.

### LS-Cen

### LS-CI

### LS-SCI

### LS-BDA




## Usage

All the simulation parameters are specified in `sim_env.py`. One can specify the random seed here as well.

For boundedness analysis, just run `boundedness_sim.py`.

For topology analysis, please run `topology_sim.py`.

To generate cooperative-localization plots with top-down uncertainty tubes and DCP quantile histories:

```bash
python multirobot_localization/collect_calibration_data.py \
  --output multirobot_localization/calibration_dataset.npz

python multirobot_localization/plot_dcp_localization.py \
  --calibration-dataset multirobot_localization/calibration_dataset.npz
```

This writes the plots to `multirobot_localization/output/`. The collector also supports `.pkl`, `.pickle`, and `.json`, but `.npz` is the default.

To render the estimate-driven 3D formation scenario with static obstacles:

```bash
uv run python multirobot_localization/render_pybullet_scene.py \
  --motion-mode formation \
  --show-trails
```

The renderer now drives each agent toward a fixed 3D formation slot using its local state estimate, renders translucent target markers for those slots, and includes static 3D obstacles as future inputs to a CBF-style safety filter.



## Covariance Boundedness

![](boundedness_result/performance_dr.png)

One trial with dead reckoning only. With identical odometry inputs, the estimation positions are the same across all algorithms.

![](boundedness_result/performance_obs.png)

One trial with dead reckoning and observation. LS-CI and LS-SCI have close estimation results.


![](boundedness_result/performance.png)

The averaged RMSE and RMTE over 100 trials.


## Observation and Communication Graphs

Deu to the detailed implementation of each algorithm, we first assume that communication is not necessary after the absolute observation. We then investigate the required communication links after the relative observation for each LS algorithms. 

algorithm   | relative observation 
------------ | ------------- 
LS\-Cen | all\-to\-all
LS\-CI, LS\-SCI | unidirectional
LS\-BDA | bidirectional

![](topology_result/topology.png)

The averaged RMSE and RMTE over 50 randomly generated graphs. The observation link is established with probability 0.75.
