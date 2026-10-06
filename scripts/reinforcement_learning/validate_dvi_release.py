# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Launch archived DVI locomotion budgets through the Isaac Lab 3.0 CLI.

Use the editable Newton DVI and Isaac Lab installations in the active environment::

    python scripts/reinforcement_learning/validate_dvi_release.py --task all --phase smoke
    python scripts/reinforcement_learning/validate_dvi_release.py --task humanoid --phase train

The Jacobi budgets come from the report's reinforcement_learning/parameters.csv,
and DR Legs from its archived launch.sh. Release task/MDP changes are permitted:
validation means finite state and useful locomotion, rather than equal reward scales.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import shlex
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path


@dataclass(frozen=True)
class Recipe:
    """One paper training budget with the release-3 task name."""

    task: str
    substeps: int
    contacts: int
    coupling: int
    post_stabilize: bool
    iterations: int


RECIPES = {
    "ant": Recipe("Isaac-Ant-Direct", 1, 10, 1, True, 1000),
    "humanoid": Recipe("Isaac-Humanoid-Direct", 2, 10, 1, True, 1000),
    "anymal_c": Recipe("IsaacContrib-Velocity-Flat-AnymalC", 1, 20, 2, False, 500),
    "go2": Recipe("Isaac-Velocity-Flat-UnitreeGo2", 1, 15, 2, False, 500),
    "h1": Recipe("Isaac-Velocity-Flat-H1", 1, 10, 1, False, 1000),
    "g1": Recipe("Isaac-Velocity-Flat-G1", 1, 15, 2, False, 1500),
    "dr_legs": Recipe("Isaac-DrLegs-Walk-v0", 1, 10, 2, False, 1000),
}


def recipe_overrides(recipe: Recipe, solver: str = "jacobi") -> list[str]:
    """Return explicit physics budgets, independent of task preset defaults."""
    contact_solver = {"jacobi": "sparse_jacobi", "apgd": "sparse_apgd", "pspg": "sparse_pspg"}[solver]
    overrides = [
        "presets=newton_dvi",
        "env.sim.physics=newton_dvi",
        f"env.sim.physics.num_substeps={recipe.substeps}",
        f"env.sim.physics.solver_cfg.contact_solver_type={contact_solver}",
        f"env.sim.physics.solver_cfg.contact_max_iterations={recipe.contacts}",
        f"env.sim.physics.solver_cfg.coupling_iterations={recipe.coupling}",
        f"env.sim.physics.solver_cfg.post_stabilize_joints={str(recipe.post_stabilize).lower()}",
        "env.sim.physics.solver_cfg.cache_factorization=true",
        "env.sim.physics.solver_cfg.contact_friction_projection=tangential",
    ]
    if solver == "apgd" or (solver == "pspg" and recipe.task in ("Isaac-Ant-Direct", "Isaac-Humanoid-Direct")):
        overrides.append("env.sim.physics.solver_cfg.contact_tolerance=0.0001")
    return overrides


def main() -> int:
    """Run selected tasks sequentially and save commands, versions, and console logs."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", choices=["all", *RECIPES], default="all")
    parser.add_argument("--phase", choices=["smoke", "train"], default="smoke")
    parser.add_argument("--solver", choices=["jacobi", "apgd", "pspg"], default="jacobi")
    parser.add_argument("--num_envs", type=int)
    parser.add_argument("--iterations", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("logs/dvi_release_validation"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--experiment-name", default="release3_dvi_validation")
    parser.add_argument("--run-name", help="Override the run label when selecting one task")
    args = parser.parse_args()
    if args.run_name and args.task == "all":
        parser.error("--run-name requires one selected task")
    import newton  # noqa: PLC0415
    from newton.solvers import SolverDVI  # noqa: F401, PLC0415

    root = Path(__file__).resolve().parents[2]
    num_envs = args.num_envs or (64 if args.phase == "smoke" else 4096)
    tasks = RECIPES if args.task == "all" else {args.task: RECIPES[args.task]}
    output = args.output.resolve() / datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")
    versions = {name: importlib.metadata.version(name) for name in ("newton", "isaaclab", "warp-lang", "rsl-rl-lib")}
    print(f"Newton DVI: {newton.__file__}; versions: {versions}", flush=True)
    if not args.dry_run:
        output.mkdir(parents=True)
    for name, recipe in tasks.items():
        iterations = args.iterations or (3 if args.phase == "smoke" else recipe.iterations)
        command = [
            sys.executable,
            str(root / "scripts/reinforcement_learning/train.py"),
            "--rl_library",
            "rsl_rl",
            "--task",
            recipe.task,
            "--visualizer",
            "none",
            "--num_envs",
            str(num_envs),
            "--max_iterations",
            str(iterations),
            "--seed",
            str(args.seed),
            *recipe_overrides(recipe, args.solver),
            f"agent.experiment_name={args.experiment_name}",
            f"agent.run_name={args.run_name or f'{name}_{args.solver}{recipe.contacts}'}",
        ]
        print(shlex.join(command), flush=True)
        if args.dry_run:
            continue
        manifest = {"recipe": asdict(recipe), "command": command, "versions": versions, "newton": newton.__file__}
        log_path = output / f"{name}.log"
        print(f"Console: {log_path}", flush=True)
        with log_path.open("w") as log:
            result = subprocess.run(command, cwd=root, stdout=log, stderr=subprocess.STDOUT)
        manifest["exit_code"] = result.returncode
        (output / f"{name}.json").write_text(json.dumps(manifest, indent=2) + "\n")
        if result.returncode:
            print(f"{name} failed; inspect {log_path}", file=sys.stderr)
            return result.returncode
        print(f"{name} completed", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
