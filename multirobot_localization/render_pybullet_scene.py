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
    # CLI for rendering rollout playback plus diagnostic exports.
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
        "--dcp-only",
        action="store_true",
        help="Run startup online DCP only (no localization rollout playback/rendering).",
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
        "--run-online-dcp",
        action="store_true",
        help="Run classwise distributed conformal calibration at scenario start using offline calibration data.",
    )
    parser.add_argument(
        "--dcp-calibration-dataset",
        type=Path,
        default=script_dir / "calibration_dataset.npz",
        help="Calibration dataset used for startup online DCP (.npz/.pkl/.pickle/.json).",
    )
    parser.add_argument(
        "--dcp-steps",
        type=int,
        default=250,
        help="Number of startup distributed subgradient iterations for online DCP.",
    )
    parser.add_argument(
        "--dcp-step-size",
        type=float,
        default=0.5,
        help="Startup online DCP base step size (constant distributed subgradient step).",
    )
    parser.add_argument(
        "--dcp-mid-join-step",
        type=int,
        default=-1,
        help="Timestep when one UGV and one UAV join with new calibration data and trigger a DCP re-run (-1 disables).",
    )
    parser.add_argument(
        "--dcp-mid-join-dataset",
        type=Path,
        default=None,
        help="Optional dataset for mid-scenario joiners. Defaults to --dcp-calibration-dataset.",
    )
    parser.add_argument(
        "--dcp-mid-join-samples-per-joiner",
        type=int,
        default=48,
        help="Number of calibration scores each joining agent contributes during mid-scenario DCP refresh.",
    )
    parser.add_argument(
        "--tube-alpha",
        type=float,
        default=0.18,
        help="Transparency used for cooperative-localization uncertainty tube overlays.",
    )
    parser.add_argument(
        "--motion-mode",
        choices=("formation", "random"),
        default="formation",
        help="Whether to render estimate-driven formation control or the original random odometry motion.",
    )
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
    # Lazy import keeps this module importable without PyBullet installed.
    try:
        import pybullet as pybullet
    except ImportError as exc:
        raise ImportError(
            "PyBullet is not installed in the active environment."
        ) from exc
    return pybullet


def import_imageio():
    # Video writer dependencies are only required when MP4 export is enabled.
    try:
        import imageio.v2 as imageio
        import imageio_ffmpeg  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "MP4 export requires both `imageio` and `imageio-ffmpeg` in the active environment."
        ) from exc
    return imageio


def connection_is_alive(pybullet, client_id: int) -> bool:
    # Defensive connection check for cleanup/GUI hold loops.
    info = pybullet.getConnectionInfo(physicsClientId=client_id)
    return bool(info.get("isConnected", 0))


def resolve_mesh_paths(args: argparse.Namespace) -> dict[AgentClass, Path]:
    # Map class to mesh filename from CLI options.
    return {
        AgentClass.CLASS_A_UGV: args.mesh_dir / args.ugv_mesh,
        AgentClass.CLASS_B_UAV: args.mesh_dir / args.uav_mesh,
    }


def mesh_scale_for(agent_class: AgentClass, args: argparse.Namespace) -> list[float]:
    # Uniform scale factor per class.
    scale = args.ugv_scale if agent_class == AgentClass.CLASS_A_UGV else args.uav_scale
    return [float(scale), float(scale), float(scale)]


def body_z_for(agent_class: AgentClass, args: argparse.Namespace) -> float:
    # Default world z-offset per class body origin.
    if agent_class == AgentClass.CLASS_A_UGV:
        return float(args.ugv_z)
    return float(args.uav_altitude)


def pose_z_for(pose: dict[str, object], args: argparse.Namespace) -> float:
    # Frame-level render altitude overrides class default if provided.
    if "render_z" in pose:
        return float(pose["render_z"])
    return body_z_for(normalize_agent_class(pose["agent_class"]), args)


def create_ground(pybullet, client_id: int):
    # Create a static floor plane represented as a thin box.
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


def create_target_marker(pybullet, client_id: int, target: dict[str, object], agent_class: AgentClass) -> int:
    # Render translucent target slots used in formation mode.
    x, y, z = [float(value) for value in target["position_xyz"]]
    color = list(VISUAL_SPECS[agent_class].color_rgba[:3]) + [0.30]
    visual = pybullet.createVisualShape(
        shapeType=pybullet.GEOM_SPHERE,
        radius=0.16 if agent_class == AgentClass.CLASS_B_UAV else 0.18,
        rgbaColor=color,
        specularColor=[0.16, 0.16, 0.16],
        physicsClientId=client_id,
    )
    return pybullet.createMultiBody(
        baseMass=0.0,
        baseCollisionShapeIndex=-1,
        baseVisualShapeIndex=visual,
        basePosition=[x, y, z],
        physicsClientId=client_id,
    )


def create_static_obstacle_body(pybullet, client_id: int, obstacle: dict[str, object]) -> int:
    # Build static scene obstacles from serialized obstacle dictionaries.
    primitive = str(obstacle["primitive"])
    x, y, z = [float(value) for value in obstacle["position_xyz"]]
    color = list(obstacle["color_rgba"])

    if primitive == "box":
        half_extents = [float(value) for value in obstacle["half_extents_xyz"]]
        collision = pybullet.createCollisionShape(
            pybullet.GEOM_BOX,
            halfExtents=half_extents,
            physicsClientId=client_id,
        )
        visual = pybullet.createVisualShape(
            pybullet.GEOM_BOX,
            halfExtents=half_extents,
            rgbaColor=color,
            specularColor=[0.15, 0.15, 0.15],
            physicsClientId=client_id,
        )
    elif primitive == "cylinder":
        radius = float(obstacle["radius"])
        height = float(obstacle["height"])
        collision = pybullet.createCollisionShape(
            pybullet.GEOM_CYLINDER,
            radius=radius,
            height=height,
            physicsClientId=client_id,
        )
        visual = pybullet.createVisualShape(
            pybullet.GEOM_CYLINDER,
            radius=radius,
            length=height,
            rgbaColor=color,
            specularColor=[0.15, 0.15, 0.15],
            physicsClientId=client_id,
        )
    else:
        raise ValueError(f"Unsupported obstacle primitive '{primitive}'")

    return pybullet.createMultiBody(
        baseMass=0.0,
        baseCollisionShapeIndex=collision,
        baseVisualShapeIndex=visual,
        basePosition=[x, y, z],
        physicsClientId=client_id,
    )


def create_visual_shape(pybullet, client_id: int, agent_class: AgentClass, mesh_path: Path, args: argparse.Namespace) -> int:
    # Prefer mesh assets when available; fall back to class primitives otherwise.
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
    # Keep collision geometry simple/stable across mesh variants.
    spec = VISUAL_SPECS[agent_class]
    return pybullet.createCollisionShape(
        pybullet.GEOM_BOX,
        halfExtents=list(spec.collision_box_half_extents),
        physicsClientId=client_id,
    )


def create_vehicle_body(pybullet, client_id: int, agent_id: int, agent_class: AgentClass, mesh_path: Path, args: argparse.Namespace) -> int:
    # Construct one kinematic body that will be pose-reset every frame.
    visual = create_visual_shape(pybullet, client_id, agent_class, mesh_path, args)
    collision = create_collision_shape(pybullet, client_id, agent_class)
    z = body_z_for(agent_class, args)
    body_id = pybullet.createMultiBody(
        # The rollout already provides authoritative poses, so the render bodies
        # should be kinematic and not participate in gravity-driven simulation.
        baseMass=0.0,
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
    # Apply rollout pose directly to the render body.
    x, y = pose["position_xy"]
    yaw = float(pose["theta"])
    z = pose_z_for(pose, args)
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
    # Maintain one replaceable debug text handle per agent.
    if not args.show_labels:
        return

    agent_id = int(pose["agent_id"])
    agent_class = normalize_agent_class(pose["agent_class"])
    x, y = pose["position_xy"]
    z = pose_z_for(pose, args) + 0.35
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
    # Draw a finite history trail by appending/removing debug lines.
    if not args.show_trails:
        return

    agent_id = int(pose["agent_id"])
    agent_class = normalize_agent_class(pose["agent_class"])
    x, y = pose["position_xy"]
    z = pose_z_for(pose, args)
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
    # Camera tracks team centroid (or explicit focus) with altitude-aware offset.
    if "camera_focus_xy" in frame:
        focus_xy = np.asarray(frame["camera_focus_xy"], dtype=float).reshape(2)
    else:
        positions = np.array([pose["position_xy"] for pose in frame["poses"]], dtype=float)
        focus_xy = positions.mean(axis=0)
    z_values = np.asarray([pose_z_for(pose, args) for pose in frame["poses"]], dtype=float)
    focus_z = float(args.camera_height + 0.20 * np.mean(z_values))
    return [float(focus_xy[0]), float(focus_xy[1]), focus_z]


def reset_camera(pybullet, frame: dict[str, object], args: argparse.Namespace):
    # Update GUI camera each frame in interactive mode.
    pybullet.resetDebugVisualizerCamera(
        cameraDistance=float(args.camera_distance),
        cameraYaw=float(args.camera_yaw),
        cameraPitch=float(args.camera_pitch),
        cameraTargetPosition=frame_camera_target(frame, args),
    )


def capture_frame(pybullet, client_id: int, frame: dict[str, object], args: argparse.Namespace) -> np.ndarray:
    # Render RGB frame via TinyRenderer for deterministic offscreen capture.
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
    # Shared color lookup used by diagnostics plots.
    return VISUAL_SPECS[agent_class].color_rgba[:3]


def extract_agent_quantile_history(rollout: dict[str, object]) -> np.ndarray:
    # Build [num_agents, num_steps] quantile history from per-frame pose diagnostics.
    frames = rollout["frames"]
    num_agents = len(rollout["agent_classes"])
    num_steps = len(frames)
    quantiles = np.zeros((num_agents, num_steps), dtype=float)
    for step_idx, frame in enumerate(frames):
        for pose in frame["poses"]:
            agent_id = int(pose["agent_id"])
            quantiles[agent_id, step_idx] = float(pose.get("quantile", 1.0))
    return quantiles


def extract_startup_quantile_history(rollout: dict[str, object]) -> np.ndarray | None:
    # Return [num_agents, num_iterations] startup DCP history when present.
    num_agents = len(rollout["agent_classes"])
    online_dcp = rollout.get("online_dcp", {})
    startup = online_dcp.get("startup", {}) if isinstance(online_dcp, dict) else {}
    startup_history_raw = startup.get("quantile_history") if isinstance(startup, dict) else None
    if startup_history_raw is None:
        return None

    candidate = np.asarray(startup_history_raw, dtype=float)
    if candidate.ndim != 2 or candidate.shape[0] < num_agents or candidate.shape[1] < 1:
        return None
    return candidate[:num_agents]


def tangent_normals(points: np.ndarray) -> np.ndarray:
    # Estimate per-point curve normals from finite-difference tangents.
    tangents = np.zeros_like(points)
    if points.shape[0] == 1:
        tangents[0] = np.array([1.0, 0.0])
    else:
        tangents[0] = points[1] - points[0]
        tangents[-1] = points[-1] - points[-2]
        if points.shape[0] > 2:
            tangents[1:-1] = points[2:] - points[:-2]

    tangent_norm = np.linalg.norm(tangents, axis=1, keepdims=True)
    tangent_norm = np.where(tangent_norm < 1.0e-9, 1.0, tangent_norm)
    unit_tangent = tangents / tangent_norm
    return np.column_stack((-unit_tangent[:, 1], unit_tangent[:, 0]))


def covariance_tube_radius(
    covariances: np.ndarray,
    normals: np.ndarray,
    scale: float,
) -> np.ndarray:
    # Project covariance along local normal direction to get tube half-width.
    variances = np.einsum("ti,tij,tj->t", normals, covariances, normals)
    variances = np.maximum(variances, 0.0)
    return float(scale) * np.sqrt(variances)


def draw_uncertainty_tube(
    ax,
    estimated_positions: np.ndarray,
    covariances: np.ndarray,
    scale: float,
    color,
    alpha: float,
) -> None:
    # Draw a closed polygon around the estimated trajectory centerline.
    normals = tangent_normals(estimated_positions)
    radius = covariance_tube_radius(covariances, normals, scale=scale)
    upper = estimated_positions + normals * radius[:, None]
    lower = estimated_positions - normals * radius[:, None]
    polygon = np.vstack((upper, lower[::-1]))
    ax.fill(polygon[:, 0], polygon[:, 1], color=color, alpha=alpha, linewidth=0.0)


def save_top_down_uncertainty_tubes_plot(
    rollout: dict[str, object],
    output_path: Path,
    tube_alpha: float = 0.18,
):
    # Side-by-side figure: baseline GS-CI vs conformalized GS-CI uncertainty tubes.
    mplconfig_dir = output_path.parent / ".mplconfig"
    mplconfig_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mplconfig_dir))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    truth_positions = np.asarray(rollout["truth_positions"], dtype=float)
    baseline_estimated_positions = np.asarray(rollout["estimated_positions"], dtype=float)
    baseline_covariances = np.asarray(rollout["raw_covariances"], dtype=float)
    agent_classes = list(rollout["agent_classes"])

    num_agents = truth_positions.shape[0]
    colors = np.array([color_for_class(normalize_agent_class(agent_classes[agent_id])) for agent_id in range(num_agents)])

    calibrated_estimated_positions_raw = rollout.get("calibrated_estimated_positions")
    calibrated_covariances_raw = rollout.get("calibrated_covariances")
    has_explicit_conformal = (
        calibrated_estimated_positions_raw is not None
        and calibrated_covariances_raw is not None
    )
    if has_explicit_conformal:
        calibrated_estimated_positions = np.asarray(calibrated_estimated_positions_raw, dtype=float)
        calibrated_covariances = np.asarray(calibrated_covariances_raw, dtype=float)
    else:
        calibrated_estimated_positions = np.asarray(baseline_estimated_positions, dtype=float)
        calibrated_covariances = np.asarray(baseline_covariances, dtype=float)

    quantile_history = extract_agent_quantile_history(rollout)
    final_scales = quantile_history[:, -1] if quantile_history.shape[1] > 0 else np.ones(num_agents, dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(15, 7), sharex=True, sharey=True)
    subplot_specs = [
        (
            "Baseline GS-CI (No Quantiles)",
            baseline_estimated_positions,
            baseline_covariances,
            np.ones(num_agents, dtype=float),
        ),
        (
            (
                "Conformalized GS-CI"
                if has_explicit_conformal
                else "DCP-Calibrated Covariance (Scaled Baseline)"
            ),
            calibrated_estimated_positions,
            calibrated_covariances,
            np.ones(num_agents, dtype=float) if has_explicit_conformal else final_scales,
        ),
    ]

    for ax, (title, estimate_bank, covariance_bank, scales) in zip(axes, subplot_specs):
        for agent_id in range(num_agents):
            color = colors[agent_id]
            truth_xy = truth_positions[agent_id]
            estimate_xy = estimate_bank[agent_id]
            draw_uncertainty_tube(
                ax=ax,
                estimated_positions=estimate_xy,
                covariances=covariance_bank[agent_id],
                scale=float(scales[agent_id]),
                color=color,
                alpha=tube_alpha,
            )
            ax.plot(truth_xy[:, 0], truth_xy[:, 1], color=color, linewidth=2.2)
            ax.plot(estimate_xy[:, 0], estimate_xy[:, 1], color=color, linewidth=1.7, linestyle="--")
            ax.scatter(truth_xy[0, 0], truth_xy[0, 1], color=color, s=18, zorder=5)
            ax.text(
                estimate_xy[-1, 0],
                estimate_xy[-1, 1],
                f"A{agent_id}",
                color=color,
                fontsize=9,
                ha="left",
                va="bottom",
            )

        ax.set_title(title)
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.grid(alpha=0.25)
        ax.set_aspect("equal", adjustable="box")

    class_handles = [
        Line2D(
            [0],
            [0],
            color=colors[agent_id],
            linewidth=2.2,
            label=f"A{agent_id} {normalize_agent_class(agent_classes[agent_id]).name.split('_')[-1]}",
        )
        for agent_id in range(num_agents)
    ]
    style_handles = [
        Line2D([0], [0], color="black", linewidth=2.2, label="True trajectory"),
        Line2D([0], [0], color="black", linewidth=1.7, linestyle="--", label="Estimated trajectory"),
        Patch(facecolor=(0.5, 0.5, 0.5, tube_alpha), edgecolor="none", label="Uncertainty tube"),
    ]
    fig.legend(handles=style_handles + class_handles, loc="upper center", ncol=4, frameon=False)
    fig.suptitle("Cooperative Localization Trajectories: Baseline vs Conformalized GS-CI", y=0.98)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.90))
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_dcp_quantiles_over_time_plot(
    rollout: dict[str, object],
    output_path: Path,
):
    # Plot startup DCP consensus convergence when available; otherwise fallback to scenario-time quantiles.
    mplconfig_dir = output_path.parent / ".mplconfig"
    mplconfig_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mplconfig_dir))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    agent_classes = list(rollout["agent_classes"])
    num_agents = len(agent_classes)
    online_dcp = rollout.get("online_dcp", {})
    startup = online_dcp.get("startup", {}) if isinstance(online_dcp, dict) else {}
    startup_quantile_history = extract_startup_quantile_history(rollout)

    use_startup_history = False
    x_axis: np.ndarray
    quantile_history: np.ndarray
    x_label: str
    title: str
    if startup_quantile_history is not None:
        quantile_history = startup_quantile_history
        x_axis = np.arange(quantile_history.shape[1], dtype=int)
        x_label = "DCP iteration"
        title = "DCP Quantiles During Startup Consensus"
        use_startup_history = True

    if not use_startup_history:
        x_axis = np.asarray(rollout["time"], dtype=float)
        quantile_history = extract_agent_quantile_history(rollout)
        x_label = "time [s]"
        title = "DCP Quantiles Over Scenario Time"

    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    for agent_id in range(num_agents):
        agent_class = normalize_agent_class(agent_classes[agent_id])
        ax.plot(
            x_axis,
            quantile_history[agent_id],
            color=color_for_class(agent_class),
            linewidth=1.8,
            label=f"A{agent_id}",
        )

    if use_startup_history:
        startup_class_quantiles = startup.get("final_class_quantiles", {}) if isinstance(startup, dict) else {}
        for class_name, value in startup_class_quantiles.items():
            agent_class = normalize_agent_class(class_name)
            linestyle = "--" if agent_class == AgentClass.CLASS_A_UGV else ":"
            ax.axhline(
                float(value),
                color=(0.2, 0.2, 0.2),
                linewidth=1.1,
                linestyle=linestyle,
                label=f"{agent_class.name.split('_')[-1]} final",
            )
    else:
        # Mark mid-scenario re-calibration events only for scenario-time fallback mode.
        for event in online_dcp.get("events", []):
            if event.get("type") != "mid_scenario_join_rerun":
                continue
            step = int(event.get("step", 0))
            step = min(max(step, 0), max(len(x_axis) - 1, 0))
            t = float(x_axis[step]) if x_axis.size else float(step)
            ax.axvline(t, color=(0.2, 0.2, 0.2), linewidth=1.2, linestyle="--", label=f"re-run @ step {step}")

    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Quantile")
    if use_startup_history:
        # Startup consensus axis is discrete iteration count.
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_xlim(0, max(int(x_axis[-1]), 0))
    ax.grid(alpha=0.25)
    ax.legend(ncol=3, frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_topology_graph_plot(
    rollout: dict[str, object],
    output_path: Path,
):
    # Draw directed observation (solid) and communication (dashed) topology.
    mplconfig_dir = output_path.parent / ".mplconfig"
    mplconfig_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mplconfig_dir))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import FancyArrowPatch, RegularPolygon

    agent_classes = list(rollout["agent_classes"])
    num_agents = len(agent_classes)
    landmark_idx = num_agents

    # Use a fixed circular layout so topology is independent of the rollout geometry.
    if num_agents > 0:
        layout_radius = max(2.1, 0.52 * float(num_agents + 1))
        total_nodes = num_agents + 1  # agents + landmark
        angles = np.linspace(0.0, 2.0 * np.pi, total_nodes, endpoint=False) + 0.5 * np.pi
        circle_positions = np.column_stack((layout_radius * np.cos(angles), layout_radius * np.sin(angles)))
        agent_positions = circle_positions[:num_agents]
        landmark_xy = circle_positions[num_agents]
    else:
        layout_radius = 2.1
        agent_positions = np.zeros((0, 2), dtype=float)
        landmark_xy = np.array([0.0, 0.0], dtype=float)

    node_radius = max(0.12, 0.06 * layout_radius)
    landmark_radius = 1.35 * node_radius

    fig, ax = plt.subplots(figsize=(9.5, 5.8))

    def draw_arrow(
        source_xy: np.ndarray,
        target_xy: np.ndarray,
        target_is_landmark: bool,
        linestyle: str,
        curve_rad: float = 0.0,
        linewidth: float = 1.5,
        zorder: int = 2,
    ) -> None:
        delta = target_xy - source_xy
        distance = float(np.linalg.norm(delta))
        if distance < 1.0e-9:
            return
        direction = delta / distance
        source_pad = 1.15 * node_radius
        target_pad = 1.15 * (landmark_radius if target_is_landmark else node_radius)
        start = source_xy + direction * source_pad
        end = target_xy - direction * target_pad
        arrow = FancyArrowPatch(
            posA=start,
            posB=end,
            arrowstyle="-|>",
            mutation_scale=15,
            linewidth=linewidth,
            color="black",
            linestyle=linestyle,
            connectionstyle=f"arc3,rad={curve_rad}",
            zorder=zorder,
        )
        ax.add_patch(arrow)

    def unique_directed_edges(raw_edges) -> list[tuple[int, int]]:
        seen: set[tuple[int, int]] = set()
        unique_edges: list[tuple[int, int]] = []
        for edge in raw_edges:
            if len(edge) != 2:
                continue
            sender = int(edge[0])
            receiver = int(edge[1])
            key = (sender, receiver)
            if key in seen:
                continue
            seen.add(key)
            unique_edges.append(key)
        return unique_edges

    observation_edges = unique_directed_edges(rollout.get("observ_topology_edges", []))
    communication_edges = unique_directed_edges(rollout.get("comm_topology_edges", []))
    observation_edge_set = set(observation_edges)
    comm_edge_set = set(communication_edges)
    overlap_edges = observation_edge_set & comm_edge_set

    # Add curvature for reciprocal directed links to avoid overplotting.
    reciprocal_observation_pairs = {
        edge for edge in observation_edge_set if (edge[1], edge[0]) in observation_edge_set
    }
    reciprocal_communication_pairs = {
        edge for edge in comm_edge_set if (edge[1], edge[0]) in comm_edge_set
    }

    def _edge_sign(sender: int, receiver: int) -> float:
        return 1.0 if sender < receiver else -1.0

    overlap_obs_curve = 0.30
    overlap_comm_curve = 0.52
    reciprocal_curve = 0.16

    # Observation edges (solid).
    for sender, receiver in observation_edges:
        if not (0 <= sender < num_agents):
            continue
        source_xy = agent_positions[sender]
        if receiver == landmark_idx:
            draw_arrow(
                source_xy=source_xy,
                target_xy=landmark_xy,
                target_is_landmark=True,
                linestyle="-",
                curve_rad=0.0,
            )
            continue
        if not (0 <= receiver < num_agents):
            continue
        draw_arrow(
            source_xy=source_xy,
            target_xy=agent_positions[receiver],
            target_is_landmark=False,
            linestyle="-",
            curve_rad=(
                -overlap_obs_curve if (sender, receiver) in overlap_edges
                else (reciprocal_curve if (sender, receiver) in reciprocal_observation_pairs else 0.0)
            ) * _edge_sign(sender, receiver),
            linewidth=1.5,
            zorder=2,
        )

    # Communication edges (dashed). When an edge also exists in observation topology,
    # draw it with opposite curvature so both link types are visible.
    for sender, receiver in communication_edges:
        if not (0 <= sender < num_agents and 0 <= receiver < num_agents):
            continue
        draw_arrow(
            source_xy=agent_positions[sender],
            target_xy=agent_positions[receiver],
            target_is_landmark=False,
            linestyle=(0, (2.6, 2.6)),
            curve_rad=(
                overlap_comm_curve if (sender, receiver) in overlap_edges
                else (reciprocal_curve if (sender, receiver) in reciprocal_communication_pairs else 0.0)
            ) * _edge_sign(sender, receiver),
            linewidth=2.2,
            zorder=4,
        )

    for agent_id in range(num_agents):
        xy = agent_positions[agent_id]
        circle = plt.Circle(
            (float(xy[0]), float(xy[1])),
            radius=node_radius,
            facecolor="white",
            edgecolor="black",
            linewidth=1.6,
            zorder=5,
        )
        ax.add_patch(circle)
        ax.text(
            float(xy[0]),
            float(xy[1]),
            f"{agent_id + 1}",
            ha="center",
            va="center",
            fontsize=12,
            color="black",
            zorder=6,
        )

    landmark = RegularPolygon(
        xy=(float(landmark_xy[0]), float(landmark_xy[1])),
        numVertices=3,
        radius=landmark_radius,
        orientation=np.pi,
        edgecolor="black",
        facecolor="white",
        linewidth=1.7,
        zorder=5,
    )
    ax.add_patch(landmark)
    ax.text(
        float(landmark_xy[0] + 1.35 * landmark_radius),
        float(landmark_xy[1]),
        "landmark",
        ha="left",
        va="center",
        fontsize=13,
    )

    legend_handles = [
        Line2D([0], [0], color="black", linewidth=1.6, linestyle="-", label="observation"),
    ]
    if communication_edges:
        legend_handles.append(
            Line2D([0], [0], color="black", linewidth=2.2, linestyle=(0, (2.6, 2.6)), label="communication")
        )
    ax.legend(handles=legend_handles, loc="lower right", frameon=False, fontsize=12)

    all_points = np.vstack((agent_positions, landmark_xy[None, :]))
    view_center = np.mean(all_points, axis=0)
    base_extent = np.max(np.abs(all_points - view_center), axis=0)
    half_range = max(float(np.max(base_extent)) + 0.85 * node_radius, 1.65 * layout_radius)
    ax.set_xlim(float(view_center[0] - half_range), float(view_center[0] + half_range))
    ax.set_ylim(float(view_center[1] - half_range), float(view_center[1] + half_range))
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_per_agent_position_error_plot(
    time_axis: np.ndarray,
    position_error: np.ndarray,
    agent_classes: list[str],
    output_path: Path,
    calibrated_position_error: np.ndarray | None = None,
):
    # Plot baseline vs conformalized position error time-series per agent.
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
        ax.plot(
            time_axis,
            position_error[agent_id],
            linestyle="--",
            color=(0.4, 0.4, 0.4),
            linewidth=1.2,
            label="baseline",
        )
        if calibrated_position_error is not None:
            ax.plot(
                time_axis,
                calibrated_position_error[agent_id],
                color=color,
                linewidth=1.8,
                label="conformalized",
            )
        ax.grid(alpha=0.25)
        ax.set_ylabel(f"A{agent_id}\nerr [m]")
        ax.set_title(f"Agent {agent_id} ({'UGV' if agent_class == AgentClass.CLASS_A_UGV else 'UAV'})", loc="left", fontsize=10)
        ax.legend(loc="upper right")

    axes[-1].set_xlabel("time [s]")
    fig.suptitle("Per-Agent Position Error: Baseline vs Conformalized GS-CI", fontsize=14)
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
    # Plot baseline GS-CI vs conformalized GS-CI covariance trace for each agent.
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
        ax.plot(time_axis, raw_cov_trace[agent_id], linestyle="--", color=(0.4, 0.4, 0.4), linewidth=1.2, label="baseline")
        ax.plot(time_axis, calibrated_cov_trace[agent_id], color=color, linewidth=1.8, label="conformalized")
        ax.grid(alpha=0.25)
        ax.set_ylabel(f"A{agent_id}\ntr [m^2]")
        ax.set_title(f"Agent {agent_id} ({'UGV' if agent_class == AgentClass.CLASS_A_UGV else 'UAV'})", loc="left", fontsize=10)
        ax.legend(loc="upper right")

    axes[-1].set_xlabel("time [s]")
    fig.suptitle("Per-Agent Covariance Trace: Baseline vs Conformalized GS-CI", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_metadata(output_path: Path, rollout: dict[str, object], args: argparse.Namespace):
    # Export compact run configuration + topology metadata as JSON.
    metadata = {
        "steps": int(np.asarray(rollout["time"], dtype=float).shape[0]),
        "requested_steps": int(args.steps),
        "dt": float(args.dt),
        "fps": int(args.fps),
        "seed": int(args.seed),
        "headless": bool(args.headless),
        "dcp_only": bool(args.dcp_only),
        "frame_width": int(args.frame_width),
        "frame_height": int(args.frame_height),
        "observ_prob": float(args.observ_prob),
        "comm_prob": float(args.comm_prob),
        "ci_coeff": float(args.ci_coeff),
        "run_online_dcp": bool(args.run_online_dcp),
        "dcp_calibration_dataset": str(args.dcp_calibration_dataset.resolve()),
        "dcp_steps": int(args.dcp_steps),
        "dcp_step_size": float(args.dcp_step_size),
        "dcp_mid_join_step": int(args.dcp_mid_join_step),
        "dcp_mid_join_dataset": str(
            (args.dcp_mid_join_dataset.resolve() if args.dcp_mid_join_dataset is not None else args.dcp_calibration_dataset.resolve())
        ),
        "dcp_mid_join_samples_per_joiner": int(args.dcp_mid_join_samples_per_joiner),
        "tube_alpha": float(args.tube_alpha),
        "motion_mode": str(args.motion_mode),
        "controller_name": rollout["controller_name"],
        "class_quantiles": rollout["class_quantiles"],
        "final_agent_quantile_maps": rollout.get("final_agent_quantile_maps"),
        "online_dcp": rollout.get("online_dcp"),
        "agent_classes": rollout["agent_classes"],
        "formation_targets": rollout["formation_targets"],
        "static_obstacles": rollout["static_obstacles"],
        "observ_topology_edges": rollout["observ_topology_edges"],
        "comm_topology_edges": rollout["comm_topology_edges"],
    }
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)


def save_metrics(output_path: Path, rollout: dict[str, object]):
    # Export numeric diagnostics arrays for later analysis.
    scenario_quantile_history = extract_agent_quantile_history(rollout)
    startup_quantile_history = extract_startup_quantile_history(rollout)
    quantile_history = (
        startup_quantile_history
        if startup_quantile_history is not None
        else scenario_quantile_history
    )
    history_mode = "startup_consensus" if startup_quantile_history is not None else "scenario_time"
    empty_startup_shape = (scenario_quantile_history.shape[0], 0)
    np.savez(
        output_path,
        time=np.asarray(rollout["time"], dtype=float),
        position_error=np.asarray(rollout["position_error"], dtype=float),
        calibrated_position_error=np.asarray(rollout.get("calibrated_position_error", []), dtype=float),
        formation_position_error=np.asarray(rollout["formation_position_error"], dtype=float),
        calibrated_formation_position_error=np.asarray(rollout.get("calibrated_formation_position_error", []), dtype=float),
        raw_cov_trace=np.asarray(rollout["raw_cov_trace"], dtype=float),
        calibrated_cov_trace=np.asarray(rollout["calibrated_cov_trace"], dtype=float),
        estimated_positions=np.asarray(rollout["estimated_positions"], dtype=float),
        calibrated_estimated_positions=np.asarray(rollout.get("calibrated_estimated_positions", []), dtype=float),
        raw_covariances=np.asarray(rollout["raw_covariances"], dtype=float),
        calibrated_covariances=np.asarray(rollout.get("calibrated_covariances", []), dtype=float),
        truth_positions=np.asarray(rollout.get("truth_positions", []), dtype=float),
        render_altitude=np.asarray(rollout["render_altitude"], dtype=float),
        target_positions_xyz=np.asarray(rollout["target_positions_xyz"], dtype=float),
        agent_classes=np.asarray(rollout["agent_classes"]),
        dcp_quantile_history=quantile_history,
        dcp_quantile_history_mode=np.asarray(history_mode),
        dcp_quantile_history_scenario=scenario_quantile_history,
        dcp_quantile_history_startup=(
            startup_quantile_history
            if startup_quantile_history is not None
            else np.empty(empty_startup_shape, dtype=float)
        ),
        dcp_final_quantiles=quantile_history[:, -1] if quantile_history.shape[1] > 0 else np.ones((quantile_history.shape[0],), dtype=float),
    )


def print_covariance_miscoverage_report(rollout: dict[str, object]) -> None:
    # Print terminal summary of CP miscoverage: score exceeds quantile threshold.
    truth_positions = np.asarray(rollout.get("truth_positions", []), dtype=float)
    baseline_estimated_positions = np.asarray(rollout.get("estimated_positions", []), dtype=float)
    baseline_covariances = np.asarray(rollout.get("raw_covariances", []), dtype=float)
    calibrated_estimated_positions_raw = rollout.get("calibrated_estimated_positions")
    calibrated_covariances_raw = rollout.get("calibrated_covariances")
    calibrated_estimated_positions = (
        np.asarray(calibrated_estimated_positions_raw, dtype=float)
        if calibrated_estimated_positions_raw is not None
        else baseline_estimated_positions
    )
    calibrated_covariances = (
        np.asarray(calibrated_covariances_raw, dtype=float)
        if calibrated_covariances_raw is not None
        else baseline_covariances
    )
    agent_classes = [str(value) for value in rollout.get("agent_classes", [])]
    epsilon_raw = rollout.get("epsilon_by_class", {})

    if truth_positions.ndim != 3 or baseline_estimated_positions.shape != truth_positions.shape:
        print("[miscoverage] unavailable: rollout does not contain compatible truth/estimate arrays.")
        return
    if baseline_covariances.ndim != 4 or baseline_covariances.shape[:2] != truth_positions.shape[:2]:
        print("[miscoverage] unavailable: rollout does not contain compatible covariance arrays.")
        return
    if calibrated_estimated_positions.shape != truth_positions.shape:
        calibrated_estimated_positions = baseline_estimated_positions
    if calibrated_covariances.shape[:2] != truth_positions.shape[:2]:
        calibrated_covariances = baseline_covariances

    num_agents, num_steps, state_dim = truth_positions.shape
    if num_steps <= 0:
        print("[miscoverage] skipped: no localization timesteps in this run.")
        return

    epsilon_by_class: dict[str, float] = {}
    if isinstance(epsilon_raw, dict):
        for class_name, epsilon in epsilon_raw.items():
            epsilon_by_class[str(class_name)] = float(epsilon)
    if not epsilon_by_class:
        epsilon_by_class = {
            AgentClass.CLASS_A_UGV.value: 0.05,
            AgentClass.CLASS_B_UAV.value: 0.10,
        }

    eval_dim = min(2, int(state_dim))
    if eval_dim <= 0:
        print("[miscoverage] unavailable: invalid state dimension.")
        return

    quantile_history = extract_agent_quantile_history(rollout)
    if quantile_history.shape != (num_agents, num_steps):
        class_quantiles = rollout.get("class_quantiles", {})
        quantile_history = np.ones((num_agents, num_steps), dtype=float)
        if isinstance(class_quantiles, dict):
            for agent_id in range(num_agents):
                class_name = agent_classes[agent_id] if agent_id < len(agent_classes) else ""
                class_q = class_quantiles.get(class_name)
                if class_q is not None:
                    quantile_history[agent_id, :] = float(class_q)

    def evaluate(mode: str) -> tuple[int, int, dict[str, int], dict[str, int], dict[str, float]]:
        mis_by_class: dict[str, int] = {class_name: 0 for class_name in epsilon_by_class.keys()}
        checks_by_class: dict[str, int] = {class_name: 0 for class_name in epsilon_by_class.keys()}
        quantile_sum_by_class: dict[str, float] = {class_name: 0.0 for class_name in epsilon_by_class.keys()}
        total_mis = 0
        total_checks = 0

        for agent_id in range(num_agents):
            class_name = agent_classes[agent_id] if agent_id < len(agent_classes) else ""
            if class_name not in epsilon_by_class:
                continue
            for step_id in range(num_steps):
                if mode == "calibrated":
                    estimate_xy = calibrated_estimated_positions[agent_id, step_id, :eval_dim]
                    covariance = np.asarray(calibrated_covariances[agent_id, step_id, :eval_dim, :eval_dim], dtype=float)
                else:
                    estimate_xy = baseline_estimated_positions[agent_id, step_id, :eval_dim]
                    covariance = np.asarray(baseline_covariances[agent_id, step_id, :eval_dim, :eval_dim], dtype=float)
                diff = truth_positions[agent_id, step_id, :eval_dim] - estimate_xy
                covariance = 0.5 * (covariance + covariance.T) + 1.0e-12 * np.eye(eval_dim)
                inv_covariance = np.linalg.pinv(covariance)
                score = float(np.sqrt(max(float(diff.T @ inv_covariance @ diff), 0.0)))
                threshold = 1.0
                if mode == "calibrated":
                    threshold = max(float(quantile_history[agent_id, step_id]), 1.0e-12)
                miscovered = int(score > threshold)
                total_mis += miscovered
                total_checks += 1
                mis_by_class[class_name] += miscovered
                checks_by_class[class_name] += 1
                quantile_sum_by_class[class_name] += threshold
        return total_mis, total_checks, mis_by_class, checks_by_class, quantile_sum_by_class

    for mode in ("raw", "calibrated"):
        total_miscoverage, total_checks, mis_by_class, checks_by_class, quantile_sum_by_class = evaluate(mode)
        total_rate = (100.0 * total_miscoverage / total_checks) if total_checks > 0 else 0.0
        label = "baseline GS-CI" if mode == "raw" else "conformalized GS-CI"
        print(f"[miscoverage] {label} CP miscoverage (score > quantile)")
        print(f"[miscoverage] total: {total_miscoverage}/{total_checks} ({total_rate:.2f}%)")
        for class_name in sorted(epsilon_by_class.keys()):
            class_checks = checks_by_class.get(class_name, 0)
            class_mis = mis_by_class.get(class_name, 0)
            class_rate = (100.0 * class_mis / class_checks) if class_checks > 0 else 0.0
            epsilon = float(epsilon_by_class[class_name])
            allowable = epsilon * class_checks
            avg_quantile = (
                float(quantile_sum_by_class[class_name]) / float(class_checks)
                if class_checks > 0 else float("nan")
            )
            print(
                f"[miscoverage] {class_name}: {class_mis}/{class_checks} ({class_rate:.2f}%), "
                f"avg_q={avg_quantile:.3f}, epsilon={epsilon:.3f}, allowable~{allowable:.1f}"
            )


def build_rollout(args: argparse.Namespace) -> dict[str, object]:
    # Run simulator once and validate there is at least one frame when rendering is enabled.
    quantiles = default_class_quantiles(
        ugv_quantile=args.ugv_quantile,
        uav_quantile=args.uav_quantile,
    )
    rollout_steps = 0 if args.dcp_only else int(args.steps)
    mid_join_dataset = (
        args.dcp_mid_join_dataset.resolve()
        if args.dcp_mid_join_dataset is not None
        else args.dcp_calibration_dataset.resolve()
    )
    rollout = simulate_class_conditional_gs_ci_rollout(
        num_steps=rollout_steps,
        seed=args.seed,
        initial_jitter_std=args.initial_jitter_std,
        class_quantiles=quantiles,
        dt=args.dt,
        observ_prob=args.observ_prob,
        comm_prob=args.comm_prob,
        ci_coeff=args.ci_coeff,
        motion_mode=args.motion_mode,
        dcp_calibration_dataset=args.dcp_calibration_dataset.resolve() if args.run_online_dcp else None,
        dcp_steps=args.dcp_steps,
        dcp_step_size=args.dcp_step_size,
        dcp_mid_join_step=None if int(args.dcp_mid_join_step) < 0 else int(args.dcp_mid_join_step),
        dcp_mid_join_dataset=mid_join_dataset,
        dcp_mid_join_samples_per_joiner=args.dcp_mid_join_samples_per_joiner,
    )
    if (not args.dcp_only) and (not rollout["frames"]):
        raise ValueError("`--steps` must be positive for rendering.")
    return rollout


def ensure_valid_mode(args: argparse.Namespace):
    # Prevent GUI-only options in headless mode.
    if args.loop and args.headless:
        raise ValueError("`--loop` is only supported in GUI mode.")
    if args.hold_on_complete and args.headless:
        raise ValueError("`--hold-on-complete` is only supported in GUI mode.")
    if args.dcp_only and (not args.run_online_dcp):
        raise ValueError("`--dcp-only` requires `--run-online-dcp`.")
    if args.dcp_only and int(args.dcp_mid_join_step) >= 0:
        raise ValueError("`--dcp-only` does not support mid-scenario join. Set `--dcp-mid-join-step -1`.")


def run(args: argparse.Namespace):
    # Main rendering pipeline: rollout -> pybullet playback -> video/plots/metadata.
    ensure_valid_mode(args)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rollout = build_rollout(args)
    print_covariance_miscoverage_report(rollout)

    if args.dcp_only:
        if not args.skip_plots:
            save_dcp_quantiles_over_time_plot(
                rollout=rollout,
                output_path=output_dir / "dcp_quantiles_over_time.png",
            )
            save_topology_graph_plot(
                rollout=rollout,
                output_path=output_dir / "observation_communication_topology.png",
            )
        save_metrics(output_dir / args.metrics_name, rollout)
        save_metadata(output_dir / args.metadata_name, rollout, args)
        return

    frames = rollout["frames"]

    # Save diagnostics plots before initializing PyBullet/video rendering.
    if not args.skip_plots:
        save_per_agent_position_error_plot(
            time_axis=np.asarray(rollout["time"], dtype=float),
            position_error=np.asarray(rollout["position_error"], dtype=float),
            agent_classes=list(rollout["agent_classes"]),
            output_path=output_dir / "position_error_per_agent.png",
            calibrated_position_error=np.asarray(rollout.get("calibrated_position_error", []), dtype=float)
            if rollout.get("calibrated_position_error") is not None else None,
        )
        save_per_agent_covariance_plot(
            time_axis=np.asarray(rollout["time"], dtype=float),
            raw_cov_trace=np.asarray(rollout["raw_cov_trace"], dtype=float),
            calibrated_cov_trace=np.asarray(rollout["calibrated_cov_trace"], dtype=float),
            agent_classes=list(rollout["agent_classes"]),
            output_path=output_dir / "calibrated_covariance_per_agent.png",
        )
        save_top_down_uncertainty_tubes_plot(
            rollout=rollout,
            output_path=output_dir / "cooperative_localization_uncertainty_tubes.png",
            tube_alpha=float(args.tube_alpha),
        )
        save_dcp_quantiles_over_time_plot(
            rollout=rollout,
            output_path=output_dir / "dcp_quantiles_over_time.png",
        )
        save_topology_graph_plot(
            rollout=rollout,
            output_path=output_dir / "observation_communication_topology.png",
        )

    pybullet = import_pybullet()
    imageio = None
    video_path = output_dir / args.video_name
    if not args.skip_video:
        imageio = import_imageio()

    mesh_paths = resolve_mesh_paths(args)
    client_id = pybullet.connect(pybullet.DIRECT if args.headless else pybullet.GUI)
    writer = None

    try:
        # Scene setup and static geometry creation.
        pybullet.resetSimulation(physicsClientId=client_id)
        pybullet.setGravity(0.0, 0.0, -9.81, physicsClientId=client_id)
        pybullet.setTimeStep(float(args.dt), physicsClientId=client_id)
        if not args.headless:
            pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_GUI, 0, physicsClientId=client_id)
            pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_SHADOWS, 1, physicsClientId=client_id)

        create_ground(pybullet, client_id)
        obstacle_body_ids = [
            create_static_obstacle_body(pybullet, client_id, obstacle)
            for obstacle in rollout.get("static_obstacles", [])
        ]
        target_marker_ids = [
            create_target_marker(
                pybullet=pybullet,
                client_id=client_id,
                target=target,
                agent_class=normalize_agent_class(rollout["agent_classes"][int(target["agent_id"])]),
            )
            for target in rollout.get("formation_targets", [])
        ]

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
        _ = obstacle_body_ids, target_marker_ids

        # Playback loop (optionally repeated in GUI mode when --loop is set).
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

                # Pose application and optional overlays.
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

        # Optional post-playback hold for manual inspection in GUI mode.
        if args.hold_on_complete and not args.headless:
            while connection_is_alive(pybullet, client_id):
                pybullet.stepSimulation(physicsClientId=client_id)
                time.sleep(1.0 / 60.0)
    finally:
        if writer is not None:
            writer.close()
        if connection_is_alive(pybullet, client_id):
            pybullet.disconnect(client_id)

    save_metrics(output_dir / args.metrics_name, rollout)
    save_metadata(output_dir / args.metadata_name, rollout, args)


def main():
    # CLI entrypoint.
    args = parse_args()
    run(args)


if __name__ == "__main__":
    main()
