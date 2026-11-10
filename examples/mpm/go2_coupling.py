# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Compare one-way and two-way Go2 coupling across a shallow MPM sand strip.

Observe how particle resistance affects a policy trained for rigid terrain.
The published ``Isaac-Velocity-Flat-UnitreeGo2`` policy walks along a rigid runway and
through an 8 cm sand layer. Both coupling modes let the robot
displace particles; two-way coupling also returns their reaction forces to the
robot. Compare its progress and balance with ``--coupling one_way`` and
``--coupling two_way``.

.. code-block:: bash

    uv run python examples/mpm/go2_coupling.py \
      --coupling two_way --visualizer newton_gl
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np

from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab.app.logging_utils import configure_console_logging

from isaaclab_tasks.utils import setup_preset_cli

logger = logging.getLogger("isaaclab.examples.mpm.go2_coupling")

TASK = "Isaac-Velocity-Flat-UnitreeGo2"
POLICY_BACKEND_NAMES = ("newtonmjwarp", "none")
ROBOT_PATTERN = r"/World/envs/env_.*/Robot"
RUNWAY_PATTERN = r"/World/envs/env_.*/RigidRunway"
ROBOT_START_X = -0.90
STRIP_START_X = -0.40
STRIP_LENGTH = 1.20
STRIP_HALF_WIDTH = 0.50
STRIP_THICKNESS = 0.08
STRIP_NAMES = ("sand",)
CAMERA_EYE = (0.0, -2.4, 1.1)
CAMERA_TARGET = (0.0, 0.0, 0.3)
CAMERA_FOCAL_LENGTH = 34.0
CAMERA_LEAD = 0.70
CAMERA_TRACK_MIN_X = ROBOT_START_X + CAMERA_LEAD
CAMERA_TRACK_MAX_X = STRIP_START_X + len(STRIP_NAMES) * STRIP_LENGTH + 1.0
PARTICLES_PER_VOXEL_AXIS = 2.0
MJWARP_CONTACT_CAPACITY = 300
STRIP_COLORS = {"sand": (0.76, 0.48, 0.20)}


class StripMaterialPreset(NamedTuple):
    """Material values for one particle strip."""

    label: str
    density: float
    young_modulus: float
    friction: float
    yield_pressure: float
    tensile_yield_ratio: float
    yield_stress: float
    hardening: float
    dilatancy: float
    viscosity: float


STRIP_MATERIAL_PRESETS = {
    "sand": StripMaterialPreset("dry sand", 1600.0, 1.0e15, 0.48, 1.0e15, 0.0, 0.0, 0.0, 0.0, 0.0),
}


parser = argparse.ArgumentParser(description="Go2 one-way versus two-way Newton MPM walking comparison.")
parser.add_argument(
    "--coupling",
    choices=("one_way", "two_way"),
    default="two_way",
    help="Whether particle reaction forces are returned to the robot.",
)
parser.add_argument("--checkpoint", type=str, default=None, help="Go2 RSL-RL checkpoint; defaults to published.")
parser.add_argument("--command_speed", type=float, default=0.3, help="Fixed forward velocity command [m/s].")
parser.add_argument("--seed", type=int, default=42, help="Environment and policy seed.")
parser.add_argument("--voxel_size", type=float, default=0.015, help="MPM grid voxel size [m].")
parser.add_argument(
    "--particle_jitter_fraction",
    type=float,
    default=0.30,
    help="Maximum per-axis particle offset as a fraction of particle spacing.",
)
parser.add_argument("--mpm_substeps", type=int, default=4, help="MPM substeps per coupled simulation tick.")
parser.add_argument("--proxy_mass_scale", type=float, default=1.0, help="Robot proxy inertia multiplier.")
parser.add_argument(
    "--proxy_relaxation",
    type=float,
    default=0.1,
    help="Under-relaxation applied to the two-way proxy feedback force.",
)
parser.add_argument(
    "--proxy_mode",
    choices=("lagged", "staggered"),
    default="staggered",
    help="Newton virtual-proxy scheduling mode; both choices preserve two-way feedback.",
)
parser.add_argument(
    "--max_steps", type=int, default=-1, help="Stop after this many policy steps; negative runs forever."
)
parser.add_argument("--disable_cuda_graph", action="store_true", help="Disable Newton CUDA graphs for debugging.")
add_launcher_args(parser)
parser.add_argument("--record_dir", type=Path, default=None, help="Record an offscreen force-overlay MP4 and trace.")
parser.set_defaults(visualizer=["newton_gl"])
args_cli, overrides = setup_preset_cli(parser)

if not np.isfinite(args_cli.command_speed) or args_cli.command_speed <= 0.0:
    parser.error("--command_speed must be finite and positive")
if not np.isfinite(args_cli.voxel_size) or args_cli.voxel_size <= 0.0:
    parser.error("--voxel_size must be finite and positive")
if not np.isfinite(args_cli.particle_jitter_fraction) or not 0.0 <= args_cli.particle_jitter_fraction < 0.5:
    parser.error("--particle_jitter_fraction must be finite and in [0, 0.5)")
if not np.isfinite(args_cli.proxy_mass_scale) or args_cli.proxy_mass_scale <= 0.0:
    parser.error("--proxy_mass_scale must be finite and positive")
if not np.isfinite(args_cli.proxy_relaxation) or args_cli.proxy_relaxation < 0.0:
    parser.error("--proxy_relaxation must be finite and nonnegative")
if args_cli.mpm_substeps <= 0:
    parser.error("--mpm_substeps must be positive")


def _kinematic_box(
    prim_path: str,
    *,
    size: tuple[float, float, float],
    position: tuple[float, float, float],
    color: tuple[float, float, float],
    contact_margin: float,
) -> Any:
    """Create one visible kinematic collider."""
    from isaaclab_newton.sim.schemas import NewtonCollisionCfg
    from isaaclab_physx.sim.schemas import PhysxRigidBodyCfg

    import isaaclab.sim as sim_utils
    from isaaclab.assets import RigidObjectCfg

    return RigidObjectCfg(
        prim_path=prim_path,
        spawn=sim_utils.CuboidCfg(
            size=size,
            rigid_props=[
                sim_utils.UsdPhysicsRigidBodyCfg(rigid_body_enabled=True, kinematic_enabled=True),
                PhysxRigidBodyCfg(disable_gravity=True),
            ],
            collision_props=[
                sim_utils.UsdPhysicsCollisionCfg(collision_enabled=True),
                NewtonCollisionCfg(contact_margin=contact_margin, contact_gap=0.0),
            ],
            physics_material=sim_utils.NewtonMaterialPropertiesCfg(
                static_friction=0.9,
                dynamic_friction=0.8,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=color,
                roughness=0.48,
                metallic=0.08,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=position),
    )


def _configure_common_environment(env_cfg: Any) -> None:
    """Apply deterministic playback settings shared by both coupling modes."""
    from isaaclab_newton.sim.schemas import NewtonCollisionCfg

    import isaaclab.sim as sim_utils
    from isaaclab.assets import AssetBaseCfg

    env_cfg.scene.num_envs = 1
    env_cfg.scene.env_spacing = 0.0
    env_cfg.scene.terrain = None
    env_cfg.scene.height_scanner = None
    env_cfg.scene.contact_forces = None
    env_cfg.observations.height_scan = None
    env_cfg.scene.robot.init_state.pos = (ROBOT_START_X, 0.0, 0.4)
    particle_spacing = args_cli.voxel_size / PARTICLES_PER_VOXEL_AXIS
    env_cfg.scene.robot.spawn.collision_props = [
        sim_utils.UsdPhysicsCollisionCfg(collision_enabled=True),
        NewtonCollisionCfg(contact_margin=particle_spacing, contact_gap=0.0),
    ]
    env_cfg.episode_length_s = 100.0
    env_cfg.curriculum.terrain_levels = None
    env_cfg.ui_window_class_type = None

    command = env_cfg.commands.base_velocity
    command.resampling_time_range = (100.0, 100.0)
    command.rel_standing_envs = 0.0
    command.rel_heading_envs = 0.0
    command.heading_command = True
    command.debug_vis = False
    command.ranges.lin_vel_x = (args_cli.command_speed, args_cli.command_speed)
    command.ranges.lin_vel_y = (0.0, 0.0)
    command.ranges.ang_vel_z = (0.0, 0.0)
    command.ranges.heading = (0.0, 0.0)

    env_cfg.events.physics_material = None
    env_cfg.events.add_base_mass = None
    env_cfg.events.base_com = None
    env_cfg.events.base_external_force_torque = None
    env_cfg.events.push_robot = None
    env_cfg.events.reset_base.params["pose_range"] = {
        "x": (0.0, 0.0),
        "y": (0.0, 0.0),
        "yaw": (0.0, 0.0),
    }
    env_cfg.events.reset_base.params["velocity_range"] = {
        axis: (0.0, 0.0) for axis in ("x", "y", "z", "roll", "pitch", "yaw")
    }
    env_cfg.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
    env_cfg.terminations.base_contact = None
    env_cfg.rewards.feet_air_time = None
    env_cfg.rewards.feet_slide = None

    for visualizer in env_cfg.sim.visualizer_cfgs:
        visualizer.eye = CAMERA_EYE
        visualizer.lookat = CAMERA_TARGET
        visualizer.focal_length = CAMERA_FOCAL_LENGTH
        if args_cli.record_dir:
            visualizer.headless = True
            visualizer.window.size = (960, 640)
    env_cfg.scene.backdrop = AssetBaseCfg(
        prim_path="/World/PresentationGround",
        spawn=sim_utils.GroundPlaneCfg(size=(20.0, 16.0)),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -0.17)),
    )


def _configure_particle_runway(env_cfg: Any) -> None:
    """Add the rigid runway, the sand strip, and directed coupler."""
    from isaaclab_newton.assets import MPMObjectCfg
    from isaaclab_newton.physics import MPMSolverCfg, NewtonCfg, NewtonCollisionPipelineCfg
    from isaaclab_newton.sim.spawners.mpm import MPMGridCfg, MPMParticleMaterialCfg

    from isaaclab_contrib.coupling import (
        CouplerEntryCfg,
        CouplerProxyCfg,
        CouplerProxyMappingCfg,
    )

    previous_physics = env_cfg.sim.physics
    rigid_solver_cfg = previous_physics.solver_cfg.replace(
        nconmax=max(previous_physics.solver_cfg.nconmax or 0, MJWARP_CONTACT_CAPACITY)
    )
    particle_spacing = args_cli.voxel_size / PARTICLES_PER_VOXEL_AXIS
    jitter_width = 2.0 * args_cli.particle_jitter_fraction * particle_spacing

    env_cfg.scene.rigid_runway = _kinematic_box(
        "{ENV_REGEX_NS}/RigidRunway",
        size=(12.0, 2.40, 0.16),
        position=(1.35, 0.0, -0.08),
        color=(0.20, 0.26, 0.34),
        contact_margin=particle_spacing,
    )

    for index, name in enumerate(STRIP_NAMES):
        material = STRIP_MATERIAL_PRESETS[name]
        lower_x = STRIP_START_X + index * STRIP_LENGTH
        setattr(
            env_cfg.scene,
            f"{name}_strip",
            MPMObjectCfg(
                prim_path=f"{{ENV_REGEX_NS}}/{name.title()}Strip",
                spawn=MPMGridCfg(
                    lower=(lower_x, -STRIP_HALF_WIDTH, 0.0),
                    upper=(lower_x + STRIP_LENGTH, STRIP_HALF_WIDTH, STRIP_THICKNESS),
                    voxel_size=args_cli.voxel_size,
                    particles_per_cell=PARTICLES_PER_VOXEL_AXIS,
                    particle_placement="cell_center",
                    jitter=jitter_width,
                    material=MPMParticleMaterialCfg(
                        density=material.density,
                        young_modulus=material.young_modulus,
                        poisson_ratio=0.3,
                        viscosity=material.viscosity,
                        friction=material.friction,
                        damping=0.01,
                        yield_pressure=material.yield_pressure,
                        tensile_yield_ratio=material.tensile_yield_ratio,
                        yield_stress=material.yield_stress,
                        hardening=material.hardening,
                        dilatancy=material.dilatancy,
                    ),
                    visual_color=STRIP_COLORS[name],
                ),
            ),
        )

    proxy_selectors = [ROBOT_PATTERN, RUNWAY_PATTERN]
    feedback_relaxation = args_cli.proxy_relaxation if args_cli.coupling == "two_way" else 0.0

    coupler_cfg = CouplerProxyCfg(
        entries=[
            CouplerEntryCfg(
                name="rigid",
                solver_cfg=rigid_solver_cfg,
                bodies=[ROBOT_PATTERN, RUNWAY_PATTERN],
                include_static_shapes=False,
                substeps=2,
            ),
            CouplerEntryCfg(
                name="mpm",
                solver_cfg=MPMSolverCfg(
                    voxel_size=args_cli.voxel_size,
                    grid_type="sparse",
                    grid_padding=0,
                    max_active_cell_count=1 << 18,
                    max_leaf_node_count=1 << 15,
                    max_lower_node_count=1 << 11,
                    max_upper_node_count=1 << 9,
                    max_iterations=36,
                    tolerance=1.0e-4,
                    solver="auto",
                    warmstart_mode="auto",
                    transfer_scheme="pic",
                    strain_basis="P0",
                    velocity_basis="Q1",
                    collider_basis="pic27",
                    collider_velocity_mode="forward",
                    air_drag=1.0,
                    project_outside_colliders=False,
                    separate_worlds=True,
                ),
                all_particles=True,
                include_static_shapes=False,
                include_child_joints=False,
                substeps=args_cli.mpm_substeps,
                in_place=True,
            ),
        ],
        proxies=[
            CouplerProxyMappingCfg(
                source="rigid",
                destination="mpm",
                bodies=proxy_selectors,
                mode=args_cli.proxy_mode,
                mass_scale=args_cli.proxy_mass_scale,
                proxy_relaxation=feedback_relaxation,
                collision_pipeline=None,
            )
        ],
        iterations=1,
    )
    env_cfg.sim.physics = NewtonCfg(
        solver_cfg=coupler_cfg,
        collision_cfg=NewtonCollisionPipelineCfg(soft_contact_max=0),
        soft_contact_cfg=previous_physics.soft_contact_cfg,
        default_shape_cfg=previous_physics.default_shape_cfg,
        num_substeps=1,
        use_cuda_graph=not args_cli.disable_cuda_graph,
        load_visual_shapes=previous_physics.load_visual_shapes,
    )


def configure_environment(env_cfg: Any) -> None:
    """Configure the deterministic Go2 sand comparison."""
    _configure_common_environment(env_cfg)
    _configure_particle_runway(env_cfg)


def _tracking_camera_pose(base_x: float) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Return a side-on dolly camera that keeps Go2 and the next strip visible."""
    camera_x = float(np.clip(base_x + CAMERA_LEAD, CAMERA_TRACK_MIN_X, CAMERA_TRACK_MAX_X))
    return (camera_x, CAMERA_EYE[1], CAMERA_EYE[2]), (camera_x, CAMERA_TARGET[1], CAMERA_TARGET[2])


def main() -> None:
    """Load the published policy and play the selected coupling variant."""
    configure_console_logging(vars(args_cli))
    import isaaclab_tasks  # noqa: F401
    from isaaclab_tasks.utils import resolve_task_config

    env_cfg, agent_cfg = resolve_task_config(
        TASK,
        "rsl_rl_cfg_entry_point",
        play_mode=True,
        overrides=("physics=newton_mjwarp", *overrides),
    )

    checkpoint = Path(args_cli.checkpoint).expanduser().resolve() if args_cli.checkpoint else None
    if checkpoint is not None:
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Go2 checkpoint not found: {checkpoint}")
    configure_environment(env_cfg)
    logger.info("Configured Go2 rigid/MPM environment; launching visualizer.")
    env_cfg.seed = args_cli.seed
    agent_cfg.seed = args_cli.seed
    env_cfg.sim.device = args_cli.device

    with launch_simulation(env_cfg, args_cli):
        agent_cfg.device = env_cfg.sim.device
        import torch
        from rsl_rl.runners import OnPolicyRunner

        from isaaclab.envs import ManagerBasedRLEnv
        from isaaclab.utils import to_dict
        from isaaclab.utils.seed import configure_seed

        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

        if checkpoint is None:
            from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

            # Download only after the application is running so remote asset
            # access does not import Kit modules before SimulationApp starts.
            published_checkpoint = get_published_pretrained_checkpoint("rsl_rl", TASK, *POLICY_BACKEND_NAMES)
            if published_checkpoint is None:
                raise FileNotFoundError(f"No published rsl_rl checkpoint is available for {TASK!r}.")
            checkpoint = Path(published_checkpoint).resolve()
        logger.info(f"Resolved Go2 checkpoint: {checkpoint}")

        base_env = ManagerBasedRLEnv(cfg=env_cfg)
        env = RslRlVecEnvWrapper(base_env, clip_actions=agent_cfg.clip_actions)
        runner = OnPolicyRunner(env, to_dict(agent_cfg), log_dir=None, device=agent_cfg.device)
        if args_cli.deterministic:
            configure_seed(args_cli.seed, torch_deterministic=True)
        runner.load(str(checkpoint))
        policy = runner.get_inference_policy(device=env.unwrapped.device)
        obs = env.get_observations()

        strip_particle_counts = {
            name: env.unwrapped.scene[f"{name}_strip"].particles_per_object for name in STRIP_NAMES
        }
        logger.info(
            f"Go2 {args_cli.coupling} playback ready with {sum(strip_particle_counts.values())} particles; "
            f"{1000 * args_cli.voxel_size:g} mm voxels and {100 * STRIP_THICKNESS:g} cm strips.",
        )

        step = 0
        rows = []
        writer = None
        if args_cli.record_dir:
            import imageio_ffmpeg

            args_cli.record_dir.mkdir(parents=True, exist_ok=True)
            writer = imageio_ffmpeg.write_frames(
                str(args_cli.record_dir / "rollout.mp4"),
                (960, 640),
                fps=50,
                codec="libx264",
                output_params=["-crf", "21", "-movflags", "+faststart"],
            )
            writer.send(None)
        try:
            while env.unwrapped.sim.is_running() and (args_cli.max_steps < 0 or step < args_cli.max_steps):
                with torch.inference_mode():
                    actions = policy(obs)
                    obs, _, dones, _ = env.step(actions)
                    if hasattr(policy, "reset"):
                        policy.reset(dones)
                step += 1
                base_x = float(env.unwrapped.scene["robot"].data.root_pos_w.torch[0, 0])
                eye, target = _tracking_camera_pose(base_x)
                env.unwrapped.sim.set_camera_view(eye=eye, target=target)
                robot = env.unwrapped.scene["robot"]
                position = robot.data.root_pos_w.torch[0].cpu().numpy()
                velocity = robot.data.root_lin_vel_w.torch[0].cpu().numpy()
                rows.append([step * env.unwrapped.step_dt, *position, *velocity])
                if writer:
                    import warp as wp
                    from PIL import Image, ImageDraw

                    sim = env.unwrapped.sim
                    solver = sim.backend_manager.backend.solver
                    forces = solver.get_proxy_body_wrenches("rigid", "mpm").numpy()
                    poses = sim.backend_manager.backend.state_0.body_q.numpy()
                    xyz, scalar = poses[:, 3:6], poses[:, 6:7]
                    com = solver.model.body_com.numpy()
                    centers = poses[:, :3] + com + 2 * np.cross(xyz, np.cross(xyz, com) + scalar * com)
                    selected = [i for i, label in enumerate(solver.model.body_label) if "/Robot/" in label]
                    viewer = sim.visualizers[0]._viewer
                    viewer.log_arrows(
                        "granular/robot_feedback",
                        wp.array(centers[selected], dtype=wp.vec3),
                        wp.array(centers[selected] + forces[selected, :3] / 200, dtype=wp.vec3),
                        wp.array(np.tile([0.1, 0.8, 0.2], (len(selected), 1)), dtype=wp.vec3),
                    )
                    frame = Image.fromarray(np.asarray(sim.visualizers[0].render_rgb_array(), dtype=np.uint8)[..., :3])
                    draw = ImageDraw.Draw(frame)
                    draw.rectangle((0, 0, 960, 65), fill=(15, 23, 42))
                    draw.text(
                        (15, 12),
                        (f"Stock Go2 policy | {args_cli.coupling} MPM | t={step * env.unwrapped.step_dt:.2f}s"),
                        fill="white",
                    )
                    draw.text(
                        (15, 35),
                        "Rigid runway + 8 cm sand layer; integration demo, not free-bed validation",
                        fill="white",
                    )
                    writer.send(np.ascontiguousarray(np.asarray(frame)))
                    if step == min(100, args_cli.max_steps):
                        frame.save(args_cli.record_dir / "preview.jpg")
        finally:
            if writer:
                writer.close()
                np.savez_compressed(args_cli.record_dir / "rollout.npz", state=np.asarray(rows))
                (args_cli.record_dir / "summary.json").write_text(
                    json.dumps(
                        dict(
                            task=TASK,
                            checkpoint=str(checkpoint),
                            coupling=args_cli.coupling,
                            steps=step,
                            rigid_supported=True,
                            strip_thickness=STRIP_THICKNESS,
                            interpretation=(
                                "Stock flat-ground policy; integration showcase, not the RFT transfer evaluation."
                            ),
                        ),
                        indent=2,
                    )
                )
            robot = env.unwrapped.scene["robot"]
            root_pos = robot.data.root_pos_w.torch[0]
            root_velocity = robot.data.root_lin_vel_w.torch[0]
            logger.info(
                f"Go2 stopped after {step} steps at x={float(root_pos[0]):.3f} m, "
                f"height={float(root_pos[2]):.3f} m, forward speed={float(root_velocity[0]):.3f} m/s.",
            )
            env.close()


if __name__ == "__main__":
    main()
