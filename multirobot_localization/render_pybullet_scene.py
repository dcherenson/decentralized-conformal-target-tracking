"""Headless PyBullet renderer and diagnostics exporter for the heterogeneous scenario."""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import sim_env
from agent_classes import AgentClass, normalize_agent_class
from heterogeneous_scenario import (
    default_class_quantiles,
    simulate_class_conditional_gs_ci_rollout,
)


@dataclass(frozen=True)
class VehicleVisualSpec:
    color_rgba: tuple[float, float, float, float]
    primitive: str
    box_half_extents: tuple[float, float, float] | None = None
    cylinder_radius: float | None = None
    cylinder_length: float | None = None
    collision_box_half_extents: tuple[float, float, float] | None = None


VISUAL_SPECS: dict[AgentClass, VehicleVisualSpec] = {
    AgentClass.CLASS_A_UGV: VehicleVisualSpec(
        color_rgba=(0.84, 0.39, 0.15, 1.0),
        primitive="box",
        box_half_extents=(0.24, 0.16, 0.10),
        collision_box_half_extents=(0.24, 0.16, 0.10),
    ),
    AgentClass.CLASS_B_UAV: VehicleVisualSpec(
        color_rgba=(0.15, 0.55, 0.84, 0.92),
        primitive="cylinder",
        cylinder_radius=0.20,
        cylinder_length=0.08,
        collision_box_half_extents=(0.20, 0.20, 0.04),
    ),
}


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    output_dir = script_dir / "output"
    mesh_dir = script_dir / "assets" / "meshes"

    parser = argparse.ArgumentParser(
        description="Render the mixed-class UGV/UAV scenario offscreen and export diagnostics."
    )
    parser.add_argument("--steps", type=int, default=500, help="Number of simulated timesteps.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed for the rollout.")
    parser.add_argument(
        "--initial-jitter-std",
        type=float,
        default=0.25,
        help="Initial position jitter applied to the default team state.",
    )
    parser.add_argument("--dt", type=float, default=float(sim_env.dt), help="Scenario timestep in seconds.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=output_dir,
        help="Directory where video and plots will be written.",
    )
    parser.add_argument(
        "--video-name",
        type=str,
        default="heterogeneous_scenario.mp4",
        help="Output MP4 filename.",
    )
    parser.add_argument(
        "--metrics-name",
        type=str,
        default="heterogeneous_scenario_metrics.npz",
        help="Output metrics archive filename.",
    )
    parser.add_argument(
        "--metadata-name",
        type=str,
        default="heterogeneous_scenario_metadata.json",
        help="Output metadata JSON filename.",
    )
    parser.add_argument(
        "--frame-width",
        type=int,
        default=1280,
        help="Rendered video width in pixels.",
    )
    parser.add_argument(
        "--frame-height",
        type=int,
        default=720,
        help="Rendered video height in pixels.",
    )
    parser.add_argument("--fps", type=int, default=12, help="Video frames per second.")
    parser.add_argument(
        "--playback-speed",
        type=float,
        default=1.0,
        help="Playback rate multiplier used only in GUI mode.",
    )

    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--gui",
        dest="headless",
        action="store_false",
        help="Open a PyBullet window instead of rendering offscreen.",
    )
    mode_group.add_argument(
        "--headless",
        dest="headless",
        action="store_true",
        help="Render offscreen without showing the PyBullet GUI.",
    )
    parser.set_defaults(headless=True)

    parser.add_argument(
        "--skip-video",
        action="store_true",
        help="Skip MP4 export and only write diagnostics plots and metrics.",
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Skip plot generation and only export the video and metrics.",
    )
    parser.add_argument(
        "--loop",
        action="store_true",
        help="Loop the animation in GUI mode.",
    )
    parser.add_argument(
        "--hold-on-complete",
        action="store_true",
        help="Keep the GUI open after playback completes.",
    )
    parser.add_argument(
        "--mesh-dir",
        type=Path,
        default=mesh_dir,
        help="Directory containing `ugv.obj` and `uav.obj` meshes.",
    )
    parser.add_argument("--ugv-mesh", type=str, default="ugv.obj", help="OBJ filename for the UGV body mesh.")
    parser.add_argument("--uav-mesh", type=str, default="uav.obj", help="OBJ filename for the UAV body mesh.")
    parser.add_argument(
        "--use-primitives-only",
        action="store_true",
        help="Ignore meshes and render built-in primitive shapes only.",
    )
    parser.add_argument("--ugv-scale", type=float, default=1.0, help="Uniform scale for the UGV mesh.")
    parser.add_argument("--uav-scale", type=float, default=1.0, help="Uniform scale for the UAV mesh.")
    parser.add_argument("--ugv-z", type=float, default=0.10, help="World Z position for the UGV body origin.")
    parser.add_argument(
        "--uav-altitude",
        type=float,
        default=1.60,
        help="World Z position for the UAV body origin.",
    )
    parser.add_argument("--show-labels", action="store_true", help="Show agent IDs and classes above each vehicle.")
    parser.add_argument(
        "--show-trails",
        action="store_true",
        help="Draw recent trajectory segments behind each vehicle.",
    )
    parser.add_argument(
        "--trail-length",
        type=int,
        default=40,
        help="Maximum number of trail segments retained per vehicle.",
    )
    parser.add_argument(
        "--camera-distance",
        type=float,
        default=9.5,
        help="PyBullet camera distance.",
    )
    parser.add_argument("--camera-yaw", type=float, default=45.0, help="PyBullet camera yaw in degrees.")
    parser.add_argument("--camera-pitch", type=float, default=-42.0, help="PyBullet camera pitch in degrees.")
    parser.add_argument(
        "--camera-height",
        type=float,
        default=0.75,
        help="Additional Z offset applied to the camera target.",
    )
    parser.add_argument("--camera-fov", type=float, default=60.0, help="Perspective field-of-view in degrees.")
    parser.add_argument("--observ-prob", type=float, default=float(sim_env.observ_prob), help="Observation link probability.")
    parser.add_argument("--comm-prob", type=float, default=float(sim_env.comm_prob), help="Communication link probability.")
    parser.add_argument("--ci-coeff", type=float, default=0.8, help="Covariance intersection convex coefficient.")
    parser.add_argument(
        "--ugv-quantile",
        type=float,
        default=1.0,
        help="Class-conditional covariance scale for UGV agents.",
    )
    parser.add_argument(
        "--uav-quantile",
        type=float,
        default=1.0,
        help="Class-conditional covariance scale for UAV agents.",
    )
    return parser.parse_args()


def import_pybullet():
    try:
        import pybullet as pybullet
    except ImportError as exc:
        raise ImportError(
            "PyBullet is not installed in the active environment."
        ) from exc
    return pybullet


def import_imageio():
    try:
        import imageio.v2 as imageio
        import imageio_ffmpeg  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "MP4 export requires both `imageio` and `imageio-ffmpeg` in the active environment."
        ) from exc
    return imageio


def connection_is_alive(pybullet, client_id: int) -> bool:
    info = pybullet.getConnectionInfo(physicsClientId=client_id)
    return bool(info.get("isConnected", 0))


def resolve_mesh_paths(args: argparse.Namespace) -> dict[AgentClass, Path]:
    return {
        AgentClass.CLASS_A_UGV: args.mesh_dir / args.ugv_mesh,
        AgentClass.CLASS_B_UAV: args.mesh_dir / args.uav_mesh,
    }


def mesh_scale_for(agent_class: AgentClass, args: argparse.Namespace) -> list[float]:
    scale = args.ugv_scale if agent_class == AgentClass.CLASS_A_UGV else args.uav_scale
    return [float(scale), float(scale), float(scale)]


def body_z_for(agent_class: AgentClass, args: argparse.Namespace) -> float:
    if agent_class == AgentClass.CLASS_A_UGV:
        return float(args.ugv_z)
    return float(args.uav_altitude)


def create_ground(pybullet, client_id: int):
    collision = pybullet.createCollisionShape(
        pybullet.GEOM_BOX,
        halfExtents=[40.0, 40.0, 0.05],
        physicsClientId=client_id,
    )
    visual = pybullet.createVisualShape(
        pybullet.GEOM_BOX,
        halfExtents=[40.0, 40.0, 0.05],
        rgbaColor=[0.94, 0.95, 0.97, 1.0],
        physicsClientId=client_id,
    )
    pybullet.createMultiBody(
        baseMass=0.0,
        baseCollisionShapeIndex=collision,
        baseVisualShapeIndex=visual,
        basePosition=[0.0, 0.0, -0.05],
        physicsClientId=client_id,
    )


def create_visual_shape(pybullet, client_id: int, agent_class: AgentClass, mesh_path: Path, args: argparse.Namespace) -> int:
    spec = VISUAL_SPECS[agent_class]
    if mesh_path.exists() and not args.use_primitives_only:
        return pybullet.createVisualShape(
            shapeType=pybullet.GEOM_MESH,
            fileName=str(mesh_path),
            meshScale=mesh_scale_for(agent_class, args),
            rgbaColor=list(spec.color_rgba),
            specularColor=[0.18, 0.18, 0.18],
            physicsClientId=client_id,
        )

    if spec.primitive == "box":
        return pybullet.createVisualShape(
            shapeType=pybullet.GEOM_BOX,
            halfExtents=list(spec.box_half_extents),
            rgbaColor=list(spec.color_rgba),
            specularColor=[0.18, 0.18, 0.18],
            physicsClientId=client_id,
        )

    return pybullet.createVisualShape(
        shapeType=pybullet.GEOM_CYLINDER,
        radius=float(spec.cylinder_radius),
        length=float(spec.cylinder_length),
        rgbaColor=list(spec.color_rgba),
        specularColor=[0.18, 0.18, 0.18],
        physicsClientId=client_id,
    )


def create_collision_shape(pybullet, client_id: int, agent_class: AgentClass) -> int:
    spec = VISUAL_SPECS[agent_class]
    return pybullet.createCollisionShape(
        pybullet.GEOM_BOX,
        halfExtents=list(spec.collision_box_half_extents),
        physicsClientId=client_id,
    )


def create_vehicle_body(pybullet, client_id: int, agent_id: int, agent_class: AgentClass, mesh_path: Path, args: argparse.Namespace) -> int:
    visual = create_visual_shape(pybullet, client_id, agent_class, mesh_path, args)
    collision = create_collision_shape(pybullet, client_id, agent_class)
    z = body_z_for(agent_class, args)
    body_id = pybullet.createMultiBody(
        baseMass=1.0,
        baseCollisionShapeIndex=collision,
        baseVisualShapeIndex=visual,
        basePosition=[0.0, 0.0, z],
        baseOrientation=pybullet.getQuaternionFromEuler([0.0, 0.0, 0.0]),
        physicsClientId=client_id,
    )
    pybullet.changeDynamics(
        body_id,
        -1,
        linearDamping=0.0,
        angularDamping=0.0,
        lateralFriction=0.8 if agent_class == AgentClass.CLASS_A_UGV else 0.2,
        physicsClientId=client_id,
    )
    return body_id


def update_vehicle_pose(pybullet, client_id: int, body_id: int, pose: dict[str, object], args: argparse.Namespace):
    agent_class = normalize_agent_class(pose["agent_class"])
    x, y = pose["position_xy"]
    yaw = float(pose["theta"])
    z = body_z_for(agent_class, args)
    orientation = pybullet.getQuaternionFromEuler([0.0, 0.0, yaw])
    pybullet.resetBasePositionAndOrientation(
        bodyUniqueId=body_id,
        posObj=[float(x), float(y), z],
        ornObj=orientation,
        physicsClientId=client_id,
    )
    pybullet.resetBaseVelocity(
        objectUniqueId=body_id,
        linearVelocity=[0.0, 0.0, 0.0],
        angularVelocity=[0.0, 0.0, 0.0],
        physicsClientId=client_id,
    )


def update_agent_labels(pybullet, client_id: int, label_ids: dict[int, int], pose: dict[str, object], args: argparse.Namespace):
    if not args.show_labels:
        return

    agent_id = int(pose["agent_id"])
    agent_class = normalize_agent_class(pose["agent_class"])
    x, y = pose["position_xy"]
    z = body_z_for(agent_class, args) + 0.35
    color = VISUAL_SPECS[agent_class].color_rgba[:3]
    label = f"{agent_id} {'UGV' if agent_class == AgentClass.CLASS_A_UGV else 'UAV'}"
    previous_label_id = label_ids.get(agent_id, -1)
    label_ids[agent_id] = pybullet.addUserDebugText(
        text=label,
        textPosition=[float(x), float(y), z],
        textColorRGB=list(color),
        textSize=1.2,
        lifeTime=0,
        replaceItemUniqueId=previous_label_id,
        physicsClientId=client_id,
    )


def update_trails(
    pybullet,
    client_id: int,
    trail_ids: dict[int, deque[int]],
    previous_positions: dict[int, list[float]],
    pose: dict[str, object],
    args: argparse.Namespace,
):
    if not args.show_trails:
        return

    agent_id = int(pose["agent_id"])
    agent_class = normalize_agent_class(pose["agent_class"])
    x, y = pose["position_xy"]
    z = body_z_for(agent_class, args)
    current_position = [float(x), float(y), z]
    previous = previous_positions.get(agent_id)
    previous_positions[agent_id] = current_position
    if previous is None:
        return

    line_id = pybullet.addUserDebugLine(
        lineFromXYZ=previous,
        lineToXYZ=current_position,
        lineColorRGB=list(VISUAL_SPECS[agent_class].color_rgba[:3]),
        lineWidth=2.0,
        lifeTime=0,
        physicsClientId=client_id,
    )
    trail_ids[agent_id].append(line_id)
    if len(trail_ids[agent_id]) > args.trail_length:
        pybullet.removeUserDebugItem(trail_ids[agent_id].popleft(), physicsClientId=client_id)


def frame_camera_target(frame: dict[str, object], args: argparse.Namespace) -> list[float]:
    positions = np.array([pose["position_xy"] for pose in frame["poses"]], dtype=float)
    centroid = positions.mean(axis=0)
    return [float(centroid[0]), float(centroid[1]), float(args.camera_height)]


def reset_camera(pybullet, frame: dict[str, object], args: argparse.Namespace):
    pybullet.resetDebugVisualizerCamera(
        cameraDistance=float(args.camera_distance),
        cameraYaw=float(args.camera_yaw),
        cameraPitch=float(args.camera_pitch),
        cameraTargetPosition=frame_camera_target(frame, args),
    )


def capture_frame(pybullet, client_id: int, frame: dict[str, object], args: argparse.Namespace) -> np.ndarray:
    view_matrix = pybullet.computeViewMatrixFromYawPitchRoll(
        cameraTargetPosition=frame_camera_target(frame, args),
        distance=float(args.camera_distance),
        yaw=float(args.camera_yaw),
        pitch=float(args.camera_pitch),
        roll=0.0,
        upAxisIndex=2,
    )
    projection_matrix = pybullet.computeProjectionMatrixFOV(
        fov=float(args.camera_fov),
        aspect=float(args.frame_width) / float(args.frame_height),
        nearVal=0.05,
        farVal=80.0,
    )
    _, _, rgba, _, _ = pybullet.getCameraImage(
        width=int(args.frame_width),
        height=int(args.frame_height),
        viewMatrix=view_matrix,
        projectionMatrix=projection_matrix,
        renderer=pybullet.ER_TINY_RENDERER,
        physicsClientId=client_id,
    )
    rgba_array = np.asarray(rgba, dtype=np.uint8).reshape((args.frame_height, args.frame_width, 4))
    return rgba_array[:, :, :3].copy()


def color_for_class(agent_class: AgentClass) -> tuple[float, float, float]:
    return VISUAL_SPECS[agent_class].color_rgba[:3]


def save_per_agent_position_error_plot(
    time_axis: np.ndarray,
    position_error: np.ndarray,
    agent_classes: list[str],
    output_path: Path,
):
    mplconfig_dir = output_path.parent / ".mplconfig"
    mplconfig_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mplconfig_dir))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    num_agents = position_error.shape[0]
    fig, axes = plt.subplots(num_agents, 1, figsize=(12, max(2.4 * num_agents, 6)), sharex=True)
    axes = np.atleast_1d(axes)

    for agent_id, ax in enumerate(axes):
        agent_class = normalize_agent_class(agent_classes[agent_id])
        color = color_for_class(agent_class)
        ax.plot(time_axis, position_error[agent_id], color=color, linewidth=1.8)
        ax.grid(alpha=0.25)
        ax.set_ylabel(f"A{agent_id}\nerr [m]")
        ax.set_title(f"Agent {agent_id} ({'UGV' if agent_class == AgentClass.CLASS_A_UGV else 'UAV'})", loc="left", fontsize=10)

    axes[-1].set_xlabel("time [s]")
    fig.suptitle("Per-Agent Position Error", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_per_agent_covariance_plot(
    time_axis: np.ndarray,
    raw_cov_trace: np.ndarray,
    calibrated_cov_trace: np.ndarray,
    agent_classes: list[str],
    output_path: Path,
):
    mplconfig_dir = output_path.parent / ".mplconfig"
    mplconfig_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mplconfig_dir))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    num_agents = raw_cov_trace.shape[0]
    fig, axes = plt.subplots(num_agents, 1, figsize=(12, max(2.6 * num_agents, 6)), sharex=True)
    axes = np.atleast_1d(axes)

    for agent_id, ax in enumerate(axes):
        agent_class = normalize_agent_class(agent_classes[agent_id])
        color = color_for_class(agent_class)
        ax.plot(time_axis, raw_cov_trace[agent_id], linestyle="--", color=(0.4, 0.4, 0.4), linewidth=1.2, label="raw")
        ax.plot(time_axis, calibrated_cov_trace[agent_id], color=color, linewidth=1.8, label="calibrated")
        ax.grid(alpha=0.25)
        ax.set_ylabel(f"A{agent_id}\ntr [m^2]")
        ax.set_title(f"Agent {agent_id} ({'UGV' if agent_class == AgentClass.CLASS_A_UGV else 'UAV'})", loc="left", fontsize=10)
        ax.legend(loc="upper right")

    axes[-1].set_xlabel("time [s]")
    fig.suptitle("Per-Agent Position Covariance Trace", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_metadata(output_path: Path, rollout: dict[str, object], args: argparse.Namespace):
    metadata = {
        "steps": int(args.steps),
        "dt": float(args.dt),
        "fps": int(args.fps),
        "seed": int(args.seed),
        "headless": bool(args.headless),
        "frame_width": int(args.frame_width),
        "frame_height": int(args.frame_height),
        "observ_prob": float(args.observ_prob),
        "comm_prob": float(args.comm_prob),
        "ci_coeff": float(args.ci_coeff),
        "class_quantiles": rollout["class_quantiles"],
        "agent_classes": rollout["agent_classes"],
        "observ_topology_edges": rollout["observ_topology_edges"],
        "comm_topology_edges": rollout["comm_topology_edges"],
    }
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)


def save_metrics(output_path: Path, rollout: dict[str, object]):
    np.savez(
        output_path,
        time=np.asarray(rollout["time"], dtype=float),
        position_error=np.asarray(rollout["position_error"], dtype=float),
        raw_cov_trace=np.asarray(rollout["raw_cov_trace"], dtype=float),
        calibrated_cov_trace=np.asarray(rollout["calibrated_cov_trace"], dtype=float),
        agent_classes=np.asarray(rollout["agent_classes"]),
    )


def build_rollout(args: argparse.Namespace) -> dict[str, object]:
    quantiles = default_class_quantiles(
        ugv_quantile=args.ugv_quantile,
        uav_quantile=args.uav_quantile,
    )
    rollout = simulate_class_conditional_gs_ci_rollout(
        num_steps=args.steps,
        seed=args.seed,
        initial_jitter_std=args.initial_jitter_std,
        class_quantiles=quantiles,
        dt=args.dt,
        observ_prob=args.observ_prob,
        comm_prob=args.comm_prob,
        ci_coeff=args.ci_coeff,
    )
    if not rollout["frames"]:
        raise ValueError("`--steps` must be positive for rendering.")
    return rollout


def ensure_valid_mode(args: argparse.Namespace):
    if args.loop and args.headless:
        raise ValueError("`--loop` is only supported in GUI mode.")
    if args.hold_on_complete and args.headless:
        raise ValueError("`--hold-on-complete` is only supported in GUI mode.")


def run(args: argparse.Namespace):
    ensure_valid_mode(args)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rollout = build_rollout(args)
    frames = rollout["frames"]

    pybullet = import_pybullet()
    imageio = None
    video_path = output_dir / args.video_name
    if not args.skip_video:
        imageio = import_imageio()

    mesh_paths = resolve_mesh_paths(args)
    client_id = pybullet.connect(pybullet.DIRECT if args.headless else pybullet.GUI)
    writer = None

    try:
        pybullet.resetSimulation(physicsClientId=client_id)
        pybullet.setGravity(0.0, 0.0, -9.81, physicsClientId=client_id)
        pybullet.setTimeStep(float(args.dt), physicsClientId=client_id)
        if not args.headless:
            pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_GUI, 0, physicsClientId=client_id)
            pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_SHADOWS, 1, physicsClientId=client_id)

        create_ground(pybullet, client_id)

        body_ids: dict[int, int] = {}
        label_ids: dict[int, int] = {}
        trail_ids: dict[int, deque[int]] = defaultdict(deque)
        previous_positions: dict[int, list[float]] = {}
        step_text_id = -1

        for pose in frames[0]["poses"]:
            agent_class = normalize_agent_class(pose["agent_class"])
            agent_id = int(pose["agent_id"])
            body_ids[agent_id] = create_vehicle_body(
                pybullet=pybullet,
                client_id=client_id,
                agent_id=agent_id,
                agent_class=agent_class,
                mesh_path=mesh_paths[agent_class],
                args=args,
            )

        if imageio is not None:
            writer = imageio.get_writer(
                str(video_path),
                fps=int(args.fps),
                codec="libx264",
                macro_block_size=None,
                ffmpeg_log_level="error",
            )

        playback_delay = max(float(args.dt) / max(float(args.playback_speed), 1.0e-6), 0.0)

        while True:
            for frame in frames:
                if not args.headless:
                    reset_camera(pybullet, frame, args)

                if args.show_labels or not args.headless:
                    step_text_id = pybullet.addUserDebugText(
                        text=f"step {frame['step']}",
                        textPosition=[-3.0, -3.0, 3.0],
                        textColorRGB=[0.15, 0.15, 0.15],
                        textSize=1.4,
                        lifeTime=0,
                        replaceItemUniqueId=step_text_id,
                        physicsClientId=client_id,
                    )

                for pose in frame["poses"]:
                    agent_id = int(pose["agent_id"])
                    update_vehicle_pose(pybullet, client_id, body_ids[agent_id], pose, args)
                    update_agent_labels(pybullet, client_id, label_ids, pose, args)
                    update_trails(pybullet, client_id, trail_ids, previous_positions, pose, args)

                pybullet.stepSimulation(physicsClientId=client_id)

                if writer is not None:
                    writer.append_data(capture_frame(pybullet, client_id, frame, args))

                if (not args.headless) and playback_delay > 0.0:
                    time.sleep(playback_delay)

            if not args.loop:
                break

        if args.hold_on_complete and not args.headless:
            while connection_is_alive(pybullet, client_id):
                pybullet.stepSimulation(physicsClientId=client_id)
                time.sleep(1.0 / 60.0)
    finally:
        if writer is not None:
            writer.close()
        if connection_is_alive(pybullet, client_id):
            pybullet.disconnect(client_id)

    if not args.skip_plots:
        save_per_agent_position_error_plot(
            time_axis=np.asarray(rollout["time"], dtype=float),
            position_error=np.asarray(rollout["position_error"], dtype=float),
            agent_classes=list(rollout["agent_classes"]),
            output_path=output_dir / "position_error_per_agent.png",
        )
        save_per_agent_covariance_plot(
            time_axis=np.asarray(rollout["time"], dtype=float),
            raw_cov_trace=np.asarray(rollout["raw_cov_trace"], dtype=float),
            calibrated_cov_trace=np.asarray(rollout["calibrated_cov_trace"], dtype=float),
            agent_classes=list(rollout["agent_classes"]),
            output_path=output_dir / "calibrated_covariance_per_agent.png",
        )

    save_metrics(output_dir / args.metrics_name, rollout)
    save_metadata(output_dir / args.metadata_name, rollout, args)


def main():
    args = parse_args()
    run(args)


if __name__ == "__main__":
    main()
