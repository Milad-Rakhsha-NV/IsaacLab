"""
Quick standing diagnostic for ANYmal-C with Chrono solver.
Tests a single config, prints base height over time. 
Run with: python test_anymal_standing.py --rs 1.0 --gap 0.01 --margin 0.001
"""
import argparse
import torch
torch.set_default_device("cuda:0")

import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationCfg
from isaaclab_newton.physics import ChronoSolverCfg, NewtonCfg, NewtonShapeCfg
from isaaclab_newton.physics.newton_collision_cfg import NewtonCollisionPipelineCfg
from isaaclab.scene import InteractiveSceneCfg, InteractiveScene
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab_assets.robots.anymal import ANYMAL_C_CFG
from isaaclab_newton.sensors import ContactSensorCfg as NewtonContactSensorCfg

parser = argparse.ArgumentParser()
parser.add_argument("--rs", type=float, default=1.0, help="contact_recovery_speed")
parser.add_argument("--gap", type=float, default=0.01, help="shape gap")
parser.add_argument("--margin", type=float, default=0.001, help="shape margin")
parser.add_argument("--steps", type=int, default=1000, help="sim steps")
parser.add_argument("--substeps", type=int, default=4, help="num substeps")
args = parser.parse_args()

print(f"Config: recovery_speed={args.rs}, gap={args.gap}, margin={args.margin}, substeps={args.substeps}")

@configclass
class TestSceneCfg(InteractiveSceneCfg):
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
    )
    robot = ANYMAL_C_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    contact_forces = NewtonContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=3,
        track_air_time=True,
    )

physics_cfg = NewtonCfg(
    solver_cfg=ChronoSolverCfg(
        joint_solver_type="sparse_ldl",
        joint_max_iterations=50,
        contact_solver_type="sparse_jacobi",
        contact_max_iterations=50,
        contact_recovery_speed=args.rs,
        angular_damping=0.05,
        use_implicit_pd=True,
        max_velocity=20.0,
    ),
    num_substeps=args.substeps,
    debug_mode=False,
    use_cuda_graph=False,
    default_shape_cfg=NewtonShapeCfg(margin=args.margin, gap=args.gap),
    collision_cfg=NewtonCollisionPipelineCfg(rigid_contact_max=665536),
)

sim_cfg = SimulationCfg(
    dt=0.005,
    physics=physics_cfg,
    physics_material=sim_utils.RigidBodyMaterialCfg(
        friction_combine_mode="multiply",
        restitution_combine_mode="multiply",
        static_friction=1.0,
        dynamic_friction=1.0,
    ),
)

sim = sim_utils.SimulationContext(sim_cfg)
scene_cfg = TestSceneCfg(num_envs=4, env_spacing=2.5)
scene = InteractiveScene(scene_cfg)
sim.reset()
scene.reset()

print(f"\n{'Step':>6} {'Time(s)':>8} {'BaseH':>8} {'MinH':>8} {'MaxH':>8}")
print("-" * 45)

min_h_all = 999.0
max_h_all = -999.0

for step in range(args.steps):
    scene.write_data_to_sim()
    sim.step()
    scene.update(sim.get_physics_dt())
    
    root_pos = scene["robot"].data.root_pos_w.torch
    h_mean = root_pos[:, 2].mean().item()
    h_min = root_pos[:, 2].min().item()
    h_max = root_pos[:, 2].max().item()
    min_h_all = min(min_h_all, h_min)
    max_h_all = max(max_h_all, h_max)
    
    if step % 100 == 0 or step == args.steps - 1:
        t = step * 0.005
        print(f"{step:>6} {t:>8.3f} {h_mean:>8.4f} {h_min:>8.4f} {h_max:>8.4f}")

root_pos = scene["robot"].data.root_pos_w.torch
final_h = root_pos[:, 2].mean().item()
standing = final_h > 0.3
print(f"\nFinal height: {final_h:.4f} | Standing: {'YES' if standing else 'NO'}")
print(f"Height range: [{min_h_all:.4f}, {max_h_all:.4f}]")
sim.clear_instance()
