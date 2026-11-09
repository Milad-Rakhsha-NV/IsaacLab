# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Run readiness checks, then train only against a matching successful readiness report."""

import argparse
import hashlib
import json
import random
import signal
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from .learner import Learner, TrainConfig
from .motions import MotionLibrary
from .networks import goal


def fingerprint(cfg, library):
    digest = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    digest.update(json.dumps(asdict(cfg), sort_keys=True).encode())
    digest.update(library.sha256.encode())
    digest.update(json.dumps(library.names).encode())
    digest.update(torch.__version__.encode())
    digest.update(torch.cuda.get_device_name().encode())
    return digest.hexdigest()


@torch.no_grad()
def evaluate(learner, steps=250):
    """Fixed first-frame, noise-free evaluation; steps=0 evaluates complete clips."""
    env, library = learner.env, learner.library
    if env.num_envs < len(library.names):
        raise ValueError("Evaluation needs at least one environment per selected clip")
    ids = torch.arange(env.num_envs, device=env.device)
    clip = ids % len(library.names)
    frame = torch.zeros_like(ids)
    env.reset(ids, clip, frame, noise=False)
    active = torch.ones(env.num_envs, device=env.device, dtype=torch.bool)
    failed = torch.zeros_like(active)
    returns = torch.zeros(env.num_envs, device=env.device)
    lengths = library.lengths[clip] - 1
    horizon = int(lengths.max()) if steps == 0 else steps
    for _ in range(horizon):
        target = goal(env.read(), library.future(env.clip, env.frame))
        action = learner.actor(env.history, target)
        result = env.step(action, noise=False)
        returns += active * result["reward"]
        failed |= active & result["failure"]
        complete = env.frame >= library.lengths[env.clip] - 1
        active &= ~(result["failure"] | complete)
        if not active.any():
            break
        # The 500-step training truncation must not restart a full-clip evaluation.
        reset_ids = (result["failure"] | complete).nonzero().flatten()
        env.reset(reset_ids, clip[reset_ids], frame[reset_ids], noise=False)
    denominator = torch.minimum(lengths, torch.full_like(lengths, horizon)) * 0.1
    scores = returns / denominator
    # Every motion receives equal weight even if num_envs is not a multiple of clips.
    scores = [float(scores[clip == i].mean()) for i in range(len(library.names))]
    failures = [bool(failed[clip == i].any()) for i in range(len(library.names))]
    result = {
        "normalized_return": sum(scores) / len(scores),
        "failed_clips": sum(failures),
        "clips": len(scores),
        "horizon": horizon,
        "complete_clips": steps == 0,
        "per_clip": dict(zip(library.names, scores, strict=True)),
    }
    env.reset(ids)
    return result


def write_json(path, content):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(content, indent=2) + "\n")
    temporary.replace(path)


def readiness(learner, out, fit_updates, smoke_updates, log):
    from .sanity import check_environment

    start = time.monotonic()
    report = {
        "passed": False,
        "fingerprint": fingerprint(learner.cfg, learner.library),
        "config": asdict(learner.cfg),
        "motion_sha256": learner.library.sha256,
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "checks": {},
    }
    write_json(out / "readiness.json", report)
    report["checks"]["environment"] = check_environment(learner.env)
    log({"phase": "physical_checks", **report["checks"]["environment"]})
    # Completely separate collection blocks: the validation transitions are cloned
    # out of replay before training data overwrites every replay slot.
    for _ in range(64):
        learner.collect()
    validation = tuple(x.clone() for x in learner.replay.windows(256, learner.cfg.model_horizon))
    for _ in range(learner.cfg.replay_steps):
        learner.collect()
    # Establish output units before comparing normalized prediction errors.
    from .math_utils import features, state_delta

    with torch.no_grad():
        s, a, target = learner.replay.windows(learner.cfg.model_batch, learner.cfg.model_horizon)
        learner.model.inputs.update(torch.cat((features(s), a[:, 0]), -1))
        learner.model.outputs.update(state_delta(s, target[:, 0]))
    initial = learner.prediction_errors(validation)
    log({"phase": "dynamics_before", **initial})
    for update in range(fit_updates):
        model_stats = learner.fit_model()
        if update % 10 == 0:
            log({"phase": "dynamics_fit", "update": update, **model_stats})
    fitted = learner.prediction_errors(validation)
    report["checks"]["held_out_dynamics"] = {"before": initial, "after": fitted}
    log({"phase": "dynamics_after", **fitted})
    if (
        fitted["normalized_error"] >= initial["normalized_error"] * 0.8
        or fitted["normalized_error"] >= fitted["persistence_normalized_error"]
        or fitted["position_rmse_m"] > 0.08
        or fitted["joint_rmse_rad"] > 0.15
    ):
        write_json(out / "readiness.json", report)
        raise RuntimeError("Held-out model validation failed; full training remains disabled")
    before = evaluate(learner)
    repeated = evaluate(learner)
    repeat_error = abs(before["normalized_return"] - repeated["normalized_return"])
    report["checks"]["evaluation_repeat_error"] = repeat_error
    if repeat_error > 1e-4:
        raise RuntimeError(f"Deterministic evaluation differs by {repeat_error}")
    norms = []
    smoke_start = time.monotonic()
    for update in range(smoke_updates):
        for _ in range(3):
            learner.collect()
        stats = learner.fit_model() | learner.improve_policy()
        norms.append(stats["actor_gradient_norm"])
        if update % 5 == 0:
            log({"phase": "training_smoke", "update": update, **stats})
    after = evaluate(learner)
    report["checks"]["seconds_per_smoke_update"] = (time.monotonic() - smoke_start) / smoke_updates
    report["checks"]["simulator_evaluation"] = {"before": before, "after": after}
    report["checks"]["actor_gradient_norm_min"] = min(norms)
    if min(norms) <= 1e-10 or after["normalized_return"] < before["normalized_return"] * 1.1:
        write_json(out / "readiness.json", report)
        raise RuntimeError(
            "Actor gradient or simulator learning gate failed (requires at least 10% return improvement)"
        )
    # Exercise serialization with a fresh learner, not an in-memory state copy.
    checkpoint = out / "sanity.pt"
    learner.save(checkpoint)
    report["checkpoint"] = str(checkpoint.resolve())
    report["checkpoint_sha256"] = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    with torch.no_grad():
        history = learner.env.history.clone()
        target = goal(learner.env.read(), learner.library.future(learner.env.clip, learner.env.frame))
        prediction = learner.actor(history, target).clone()
    restored = Learner(learner.env, learner.cfg)
    restored.load(checkpoint)
    with torch.no_grad():
        torch.testing.assert_close(restored.actor(history, target), prediction, atol=0, rtol=0)
    # One real resumed optimizer cycle catches missing or incompatible optimizer state.
    for _ in range(64):
        restored.collect()
    restored.fit_model()
    resumed_stats = restored.improve_policy()
    if resumed_stats["actor_gradient_norm"] <= 0:
        raise RuntimeError("Resumed actor has no gradient")
    report["checks"]["checkpoint_resume"] = True
    report["checks"]["peak_allocated_gib"] = torch.cuda.max_memory_allocated() / 2**30
    free, total = torch.cuda.mem_get_info()
    report["checks"]["free_vram_gib"] = free / 2**30
    if free < total * 0.1:
        raise RuntimeError("Insufficient memory headroom for a full run")
    report["elapsed_seconds"] = time.monotonic() - start
    report["passed"] = True
    write_json(out / "readiness.json", report)
    log({"phase": "readiness_passed", "report": str(out / "readiness.json"), **report["checks"]})


def train(learner, args, out, log):
    report = json.loads(Path(args.readiness).read_text())
    if not report["passed"] or report["fingerprint"] != fingerprint(learner.cfg, learner.library):
        raise ValueError(
            "Full training requires a successful readiness report matching code, config, data and hardware"
        )
    if args.resume:
        learner.load(args.resume)
    else:
        # Continue the validated learner, including its dynamics prefit. A cold
        # start would not exercise the initialization checked by readiness.
        checkpoint = Path(report["checkpoint"])
        if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != report["checkpoint_sha256"]:
            raise ValueError("The validated warm-start checkpoint has changed")
        learner.load(checkpoint)
    stop = []

    def request_stop(signum, frame):
        stop.append(signum)

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    initial_eval = evaluate(learner)
    log({"phase": "evaluation", "iteration": learner.iteration, **initial_eval})
    learner.save(out / "last.pt")
    for _ in range(64):
        learner.collect()
    start = time.monotonic()
    credit = 0
    stats = {}
    for iteration in range(learner.iteration, args.iterations):
        sim_stats = learner.collect()
        credit += learner.cfg.utd
        if credit >= 1:
            credit -= 1
            # Paper cosine schedule; preserve the same schedule across resume.
            lr = learner.cfg.learning_rate * (0.03 + 0.97 * 0.5 * (1 + np.cos(np.pi * min(iteration / 250000, 1))))
            for optimizer in (learner.model_optimizer, learner.actor_optimizer, learner.critic_optimizer):
                for group in optimizer.param_groups:
                    group["lr"] = lr
            stats = learner.fit_model() | learner.improve_policy()
        learner.iteration = iteration + 1
        if iteration % 25 == 0:
            log(
                {
                    "phase": "training",
                    "iteration": learner.iteration,
                    "updates": learner.updates,
                    "elapsed_seconds": time.monotonic() - start,
                    "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                    **sim_stats,
                    **stats,
                }
            )
        if learner.iteration % args.eval_every == 0:
            log({"phase": "evaluation", "iteration": learner.iteration, **evaluate(learner)})
        if learner.iteration % args.save_every == 0 or stop:
            learner.save(out / "last.pt")
        if stop or (args.max_hours > 0 and time.monotonic() - start >= args.max_hours * 3600):
            break
    learner.save(out / "last.pt")
    log(
        {
            "phase": "finished",
            "iteration": learner.iteration,
            "updates": learner.updates,
            "elapsed_seconds": time.monotonic() - start,
            **evaluate(learner),
        }
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motions", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--mode", choices=("sanity", "train", "evaluate"), default="sanity")
    parser.add_argument("--clips", nargs="*", default=[])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--readiness")
    parser.add_argument("--resume")
    parser.add_argument("--iterations", type=int, default=320000)
    parser.add_argument("--max-hours", type=float, default=0)
    parser.add_argument("--eval-every", type=int, default=4000)
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--fit-updates", type=int, default=30)
    parser.add_argument("--smoke-updates", type=int, default=30)
    parser.add_argument("--evaluation-steps", type=int, default=250)
    for name, field in TrainConfig.__dataclass_fields__.items():
        default = field.default
        if isinstance(default, bool):
            parser.add_argument("--" + name.replace("_", "-"), action=argparse.BooleanOptionalAction, default=default)
        else:
            parser.add_argument("--" + name.replace("_", "-"), type=type(default), default=default)
    args = parser.parse_args()
    if (
        args.fit_updates < 1
        or args.smoke_updates < 1
        or args.iterations < 1
        or min(args.save_every, args.eval_every) < 1
    ):
        parser.error("Iteration and evaluation counts must be positive")
    cfg = TrainConfig(**{name: getattr(args, name) for name in TrainConfig.__dataclass_fields__})
    cfg.validate()
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)
    torch.set_float32_matmul_precision("high")
    torch.set_num_threads(4)
    args.out.mkdir(parents=True, exist_ok=True)
    write_json(
        args.out / "config.json",
        {
            "learner": asdict(cfg),
            "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        },
    )
    with (args.out / "metrics.jsonl").open("a", buffering=1) as metrics:

        def log(record):
            record["time_unix"] = time.time()
            line = json.dumps(record, allow_nan=False)
            metrics.write(line + "\n")
            print(line, flush=True)

        from isaaclab.app import launch_simulation

        from .environment import TrackingEnv, simulation_cfg

        library = MotionLibrary(args.motions, args.device, args.clips)
        sim_cfg = simulation_cfg(args.device)
        with launch_simulation(sim_cfg, {"headless": True, "visualizer": "none"}):
            env = TrackingEnv(sim_cfg, library, cfg.num_envs)
            try:
                learner = Learner(env, cfg)
                if args.mode == "sanity":
                    readiness(learner, args.out, args.fit_updates, args.smoke_updates, log)
                elif args.mode == "evaluate":
                    if not args.resume:
                        parser.error("Evaluation requires --resume")
                    learner.load(args.resume)
                    log({"phase": "evaluation", **evaluate(learner, args.evaluation_steps)})
                else:
                    if not args.readiness:
                        parser.error("Training requires --readiness")
                    train(learner, args, args.out, log)
            except BaseException as exc:
                log({"phase": "failed", "error": str(exc), "exception": type(exc).__name__})
                raise
            finally:
                env.close()


if __name__ == "__main__":
    main()
