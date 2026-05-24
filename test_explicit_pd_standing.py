"""
Robot standing test with EXPLICIT PD torques (mimics DCMotor actuator model).

PD stiffness/damping are set to 0 in the model. Each step we compute:
    torque = ke * (q_target - q) + kd * (0 - qd)
and write it to control.joint_f, exactly like Isaac Lab's DCMotor does.

This tests whether Chrono (and MuJoCo) work correctly with external torque application.

Usage:
  python test_explicit_pd_standing.py --solver chrono --robot go2
  python test_explicit_pd_standing.py --solver mujoco --robot anymal
"""
import argparse
import numpy as np
import yaml

import warp as wp
import newton
import newton.examples
import newton.utils
from newton import JointTargetMode
from newton.solvers import SolverType

# -- CLI -----------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Explicit PD standing test")
parser.add_argument("--solver", type=str, required=True, choices=["chrono", "mujoco"])
parser.add_argument("--robot", type=str, default="go2", choices=["go2", "anymal"])
parser.add_argument("--substeps", type=int, default=1)
parser.add_argument("--steps", type=int, default=1000)
args = parser.parse_args()

# -- Resolve asset paths -------------------------------------------------------
ROBOT_ASSETS = {
    "go2": ("unitree_go2", "usd/go2.usda", "rl_policies/go2.yaml"),
    "anymal": ("anybotics_anymal_c", "usd/anymal_c.usda", "rl_policies/anymal.yaml"),
}

asset_dir_name, asset_path, yaml_path = ROBOT_ASSETS[args.robot]
asset_directory = str(newton.utils.download_asset(asset_dir_name))
yaml_file_path = f"{asset_directory}/{yaml_path}"

with open(yaml_file_path, encoding="utf-8") as f:
    config = yaml.safe_load(f)

num_dofs = config["num_dofs"]
pd_ke = np.array(config["mjw_joint_stiffness"])  # (num_dofs,)
pd_kd = np.array(config["mjw_joint_damping"])    # (num_dofs,)
q_target = np.array(config["mjw_joint_pos"])     # (num_dofs,)

# DCMotor saturation parameters (from UNITREE_GO2_CFG / ANYmal config)
DC_MOTOR_PARAMS = {
    "go2":    {"effort_limit": 23.5, "saturation_effort": 23.5, "velocity_limit": 30.0},
    "anymal": {"effort_limit": 80.0, "saturation_effort": 120.0, "velocity_limit": 15.0},
}
dc = DC_MOTOR_PARAMS[args.robot]

print(f"Robot: {args.robot} | Solver: {args.solver} | substeps={args.substeps}")
print(f"PD ke={pd_ke[0]}, kd={pd_kd[0]} (applied EXPLICITLY via control.joint_f)")
print(f"DC motor: effort_limit={dc['effort_limit']}, saturation={dc['saturation_effort']}, vel_limit={dc['velocity_limit']}")

# -- Build model (PD gains = 0 in model, like Isaac Lab DCMotor) ---------------
fps = 200
frame_dt = 1.0 / fps
sim_substeps = args.substeps
sim_dt = frame_dt / sim_substeps

builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
if args.solver != "chrono":
    newton.solvers.SolverMuJoCo.register_custom_attributes(builder)

builder.default_joint_cfg = newton.ModelBuilder.JointDofConfig(
    armature=0.1,
    limit_ke=1.0e2,
    limit_kd=1.0e0,
)
builder.default_shape_cfg.ke = 5.0e4
builder.default_shape_cfg.kd = 5.0e2
builder.default_shape_cfg.kf = 1.0e3
builder.default_shape_cfg.mu = 0.75
if args.solver == "chrono":
    builder.default_shape_cfg.gap = 0.01
    builder.default_shape_cfg.margin = 0.001

builder.add_usd(
    newton.examples.get_asset(asset_directory + "/" + asset_path),
    xform=wp.transform(wp.vec3(0, 0, 0.8)),
    collapse_fixed_joints=False,
    enable_self_collisions=False,
    joint_ordering="dfs",
    hide_collision_shapes=True,
)
builder.approximate_meshes("convex_hull")
builder.add_ground_plane()

# Set initial pose
builder.joint_q[:3] = [0.0, 0.0, 0.76]
builder.joint_q[3:7] = [0.0, 0.0, 0.7071, 0.7071]
builder.joint_q[7:] = config["mjw_joint_pos"]

# Set armature but PD gains = 0 (DCMotor-style: external torques only)
for i in range(num_dofs):
    builder.joint_target_ke[i + 6] = 0.0    # <-- ZERO stiffness in model
    builder.joint_target_kd[i + 6] = 0.0    # <-- ZERO damping in model
    builder.joint_armature[i + 6] = config["mjw_joint_armature"][i]
    builder.joint_target_mode[i + 6] = int(JointTargetMode.NONE)

model = builder.finalize()
model.set_gravity((0.0, 0.0, -9.81))

# -- Create solver (same as example_robot_policy) ------------------------------
if args.solver == "chrono":
    joint_config = newton.solvers.NumericalSolverConfig(
        solver_type=SolverType.SPARSE_LDL,
        max_iterations=50, omega=0.3, relax=0.8,
        alpha=0.0, recovery_speed=1000000.0, reg=1e-6,
    )
    contact_config = newton.solvers.NumericalSolverConfig(
        solver_type=SolverType.SPARSE_JACOBI,
        max_iterations=50, omega=0.3, relax=0.9,
        alpha=0.0, recovery_speed=1.0, reg=1e-4,
    )
    solver = newton.solvers.SolverChrono(
        model, joint_solver=joint_config, contact_solver=contact_config,
        angular_damping=0.01, enable_contacts=True, enable_timers=False,
        use_implicit_pd=False,  # No implicit PD — we apply torques explicitly
    )
else:
    solver = newton.solvers.SolverMuJoCo(
        model, solver="newton", nconmax=30, njmax=100,
    )

# -- Initialize state -----------------------------------------------------------
state_0 = model.state()
state_1 = model.state()
control = model.control()

if args.solver == "chrono":
    newton.eval_fk(model, state_0.joint_q, state_0.joint_qd, state_0)
    newton.eval_fk(model, state_1.joint_q, state_1.joint_qd, state_1)
    contacts = model.collide(state_0)
    solver.finalize_for_capture(state_0)
else:
    contacts = newton.Contacts(solver.get_max_contact_count(), 0)

# DOF offset: floating base has 6 DOFs (indices 0-5), actuated joints start at 6
DOF_OFFSET = 6


def dc_motor_clip(torque, joint_vel):
    """DC motor saturation model (matches Isaac Lab DCMotor._clip_effort)."""
    effort_limit = dc["effort_limit"]
    saturation_effort = dc["saturation_effort"]
    velocity_limit = dc["velocity_limit"]

    # Torque-speed curve limits
    tau_max = np.clip(
        saturation_effort * (1.0 - joint_vel / velocity_limit),
        -np.inf, effort_limit
    )
    tau_min = np.clip(
        saturation_effort * (-1.0 - joint_vel / velocity_limit),
        -effort_limit, np.inf
    )
    return np.clip(torque, tau_min, tau_max)


# -- Simulation loop -----------------------------------------------------------
print(f"\n{'Step':>6} {'Time':>8} {'BaseH':>8} {'BaseVz':>8} {'MaxTrq':>8}")
print("-" * 48)

min_h = 999.0
max_h = -999.0

for step in range(args.steps):
    # --- Compute explicit PD torques (like DCMotor) ---
    jq = state_0.joint_q.numpy()
    jqd = state_0.joint_qd.numpy()

    q_current = jq[7:]          # actuated joint positions
    qd_current = jqd[DOF_OFFSET:]  # actuated joint velocities

    error_pos = q_target - q_current
    error_vel = 0.0 - qd_current  # vel target = 0
    torque = pd_ke * error_pos + pd_kd * error_vel
    torque = dc_motor_clip(torque, qd_current)

    # Write torques to control.joint_f
    joint_f = control.joint_f.numpy()
    joint_f[DOF_OFFSET:DOF_OFFSET + num_dofs] = torque
    control.joint_f.assign(wp.array(joint_f, dtype=wp.float32, device=model.device))

    # --- Step simulation ---
    for _ in range(sim_substeps):
        if args.solver == "chrono":
            model.collide(state_0, contacts)
            solver.step(state_0, state_1, control, contacts, sim_dt)
            newton.eval_ik(model, state_1, state_1.joint_q, state_1.joint_qd)
        else:
            solver.step(state_0, state_1, control, contacts, sim_dt)
        state_0, state_1 = state_1, state_0
        state_0.clear_forces()

    # Read base height
    jq = state_0.joint_q.numpy()
    jqd = state_0.joint_qd.numpy()
    h = jq[2]
    vz = jqd[2]
    max_trq = np.max(np.abs(torque))

    min_h = min(min_h, h)
    max_h = max(max_h, h)

    if step % 100 == 0 or step == args.steps - 1:
        t = step * frame_dt
        print(f"{step:>6} {t:>8.3f} {h:>8.4f} {vz:>8.4f} {max_trq:>8.2f}")

final_h = jq[2]
stand_threshold = 0.15 if args.robot == "go2" else 0.35
standing = final_h > stand_threshold
print(f"\nFinal height: {final_h:.4f} | Standing: {'YES ✓' if standing else 'NO ✗'}")
print(f"Height range: [{min_h:.4f}, {max_h:.4f}]")
