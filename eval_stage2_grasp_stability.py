"""Stage 2 feasibility check: does an active power-grasp retain the included object against a
perturbation that the passive open-hand pose can't? Privileged-state (direct qvel injection,
not a vision/sensing problem) sanity check, same role as every sibling project's own Stage 2 --
not the project's eventual deliverable, a reference number for the learned policy to beat.

For each of N trials per push magnitude, the SAME randomly-sampled push direction is applied to
BOTH conditions (open-hand, closed power-grasp) -- a paired comparison, same discipline as
panda_vision_grasp's checkpoint-vs-checkpoint evals on identical seeds.

Usage:
    python3 eval_stage2_grasp_stability.py --magnitudes 0.6,0.8,1.0,1.2,1.5,2.0 --trials 8
"""
import argparse

import mujoco as mj
import numpy as np

from hand_env import (N_FINGER_JOINTS, POWER_GRASP_TARGET, apply_push, object_retained,
                       ramp_to_target, settle_object)


def run_trial(model, push_vel, grasp):
    data = mj.MjData(model)
    mj.mj_forward(model, data)
    settle_object(model, data, settle_seconds=1.0)
    if grasp:
        ramp_to_target(model, data, POWER_GRASP_TARGET, ramp_seconds=1.0, hold_seconds=0.5)
        hold_ctrl = POWER_GRASP_TARGET
    else:
        hold_ctrl = np.zeros(N_FINGER_JOINTS)
    apply_push(data, push_vel)
    dt = model.opt.timestep
    for _ in range(int(1.0 / dt)):
        data.ctrl[:] = hold_ctrl
        mj.mj_step(model, data)
    return object_retained(data)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", type=str, default="assets/scene_right.xml")
    ap.add_argument("--magnitudes", type=str, default="0.6,0.8,1.0,1.2,1.5,2.0",
                     help="Comma-separated push speeds (m/s) to sweep.")
    ap.add_argument("--trials", type=int, default=8, help="Random push directions per magnitude.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--retain-threshold", type=float, default=0.05)
    args = ap.parse_args()

    model = mj.MjModel.from_xml_path(args.scene)
    rng = np.random.default_rng(args.seed)
    magnitudes = [float(m) for m in args.magnitudes.split(",")]

    print(f"{'magnitude':>10} {'open':>12} {'power-grasp':>14}")
    for mag in magnitudes:
        open_ret, grasp_ret = 0, 0
        for _ in range(args.trials):
            angle = rng.uniform(0, 2 * np.pi)
            push = mag * np.array([np.cos(angle), np.sin(angle), 0.0])
            retained_open, _ = run_trial(model, push, grasp=False)
            retained_grasp, _ = run_trial(model, push, grasp=True)
            open_ret += int(retained_open)
            grasp_ret += int(retained_grasp)
        print(f"{mag:>10.2f} {open_ret:>5d}/{args.trials:<5d}  "
              f"{grasp_ret:>6d}/{args.trials:<5d}")
        print(f"SUMMARY_PARSEABLE magnitude={mag} trials={args.trials} "
              f"open_retained={open_ret} grasp_retained={grasp_ret}")


if __name__ == "__main__":
    main()
