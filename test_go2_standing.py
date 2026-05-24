"""
Robot standing test: Chrono vs MuJoCo (pure Newton, matching example_robot_policy.py).

Drops a robot from ~0.76m with PD gains holding default pose. No RL policy.
Tests that PD alone keeps the robot standing under both solvers.

Usage:
  python test_go2_standing.py --solver chrono --robot go2
  python test_go2_standing.py --solver mujoco --robot anymal
  python test_go2_standing.py --solver chrono --robot go2 --substeps 2
"""
import argparse
import yaml

import warp as wp
import newton
import newton.examples
import newton.utils
from newton import JointTargetMode
from newton.solvers import SolverType

# -- CLI -----------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Robot standing test: Chrono vs MuJoCo")
parser.add_argument("--solver", type=str, required=True, choices=["chrono", "mujoco"])
parser.add_argument("--robot", type=str, default="go2", choices=["go2", "anymal"])
parser.add_argument("--substeps", type=int, default=1)
parser.add_argument("--steps", type=int, default=1000)
parser.add_argument("--num-envs", type=int, default=1, help="Number of environments (uses replicate)")
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

print(f"Robot: {args.robot} | Solver: {args.solver} | substeps={args.substeps} | num_envs={args.num_envs}")
print(f"PD stiffness: {config['mjw_joint_stiffness'][0]}, damping: {config['mjw_joint_damping'][0]}, "
      f"armature: {config['mjw_joint_armature'][0]}")

# -- Build model (matches example_robot_policy.py exactly) ---------------------
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

robot_handle = builder.add_usd(
    newton.examples.get_asset(asset_directory + "/" + asset_path),
    xform=wp.transform(wp.vec3(0, 0, 0.8)),
    collapse_fixed_joints=False,
    enable_self_collisions=False,
    joint_ordering="dfs",
    hide_collision_shapes=True,
)
builder.approximate_meshes("convex_hull")
builder.add_ground_plane()

# Set initial pose (same as example_robot_policy)
builder.joint_q[:3] = [0.0, 0.0, 0.76]
builder.joint_q[3:7] = [0.0, 0.0, 0.7071, 0.7071]
builder.joint_q[7:] = config["mjw_joint_pos"]

# Set PD gains from YAML (matches example_robot_policy exactly)
for i in range(len(config["mjw_joint_stiffness"])):
    builder.joint_target_ke[i + 6] = config["mjw_joint_stiffness"][i]
    builder.joint_target_kd[i + 6] = config["mjw_joint_damping"][i]
    builder.joint_armature[i + 6] = config["mjw_joint_armature"][i]
    builder.joint_target_mode[i + 6] = int(JointTargetMode.POSITION)

# Replicate for multi-env
if args.num_envs > 1:
    builder.replicate(robot_handle, args.num_envs - 1, spacing=2.0)

model = builder.finalize()
model.set_gravity((0.0, 0.0, -9.81))

# -- Create solver (matches example_robot_policy.py) ---------------------------
if args.solver == "chrono":
    joint_config = newton.solvers.NumericalSolverConfig(
        solver_type=SolverType.SPARSE_LDL,
        max_iterations=50,
        omega=0.3,
        relax=0.8,
        alpha=0.0,
        recovery_speed=1000000.0,
        reg=1e-6,
        position_correction=None,
    )
    contact_config = newton.solvers.NumericalSolverConfig(
        solver_type=SolverType.SPARSE_JACOBI,
        max_iterations=50,
        omega=0.3,
        relax=0.9,
        alpha=0.0,
        recovery_speed=1.0,
        reg=1e-4,
        position_correction=None,
    )
    solver = newton.solvers.SolverChrono(
        model,
        joint_solver=joint_config,
        contact_solver=contact_config,
        angular_damping=0.01,
        enable_contacts=True,
        enable_timers=False,
        use_implicit_pd=True,
    )
else:
    solver = newton.solvers.SolverMuJoCo(
        model,
        solver="newton",
        nconmax=30,
        njmax=100,
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

# -- Simulation loop -----------------------------------------------------------
print(f"\n{'Step':>6} {'Time':>8} {'BaseH':>8} {'BaseVz':>8}")
print("-" * 38)

min_h = 999.0
max_h = -999.0

for step in range(args.steps):
    for _ in range(sim_substeps):
        if args.solver == "chrono":
            model.collide(state_0, contacts)
            solver.step(state_0, state_1, control, contacts, sim_dt)
            newton.eval_ik(model, state_1, state_1.joint_q, state_1.joint_qd)
        else:
            solver.step(state_0, state_1, control, contacts, sim_dt)
        state_0, state_1 = state_1, state_0
        state_0.clear_forces()

    # Read base height/velocity from joint_q/joint_qd (first body = floating base)
    jq = state_0.joint_q.numpy()
    jqd = state_0.joint_qd.numpy()
    h = jq[2]  # z position
    vz = jqd[2]  # z velocity

    min_h = min(min_h, h)
    max_h = max(max_h, h)

    if step % 100 == 0 or step == args.steps - 1:
        t = step * frame_dt
        print(f"{step:>6} {t:>8.3f} {h:>8.4f} {vz:>8.4f}")

final_h = jq[2]
# Go2 PD equilibrium ~0.22m, ANYmal ~0.61m
stand_threshold = 0.15 if args.robot == "go2" else 0.35
standing = final_h > stand_threshold
print(f"\nFinal height: {final_h:.4f} | Standing: {'YES ✓' if standing else 'NO ✗'}")
print(f"Height range: [{min_h:.4f}, {max_h:.4f}]")
