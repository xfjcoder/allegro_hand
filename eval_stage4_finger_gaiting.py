"""Stage 4 feasibility check: does releasing-and-recontacting a finger between twist blocks
(a minimal finger-gaiting attempt) extend Stage 3's ~23-25 degree rotation ceiling?

Compares plain repeated twisting (no gait) against the same total number of twist cycles split
into blocks with a `regrasp_finger` call (ring finger, then middle finger variants) between each
block -- same amplitude/cycle-count budget for both, so any gain is attributable to the regrasp
itself, not just more twisting time.

Usage:
    python3 eval_stage4_finger_gaiting.py --amplitude 0.4 --blocks 5 --cycles-per-block 10
"""
import argparse

import mujoco as mj
import numpy as np

from hand_env import (POWER_GRASP_TARGET, object_retained, quat_angle_deg, ramp_to_target,
                       regrasp_finger, settle_object, twist_ctrl)

RING_FINGER_SLICE = slice(8, 12)
MIDDLE_FINGER_SLICE = slice(4, 8)


def twist_cycles(model, data, amp, n_cycles, cycle_seconds=2.0):
    n_steps = int(cycle_seconds / model.opt.timestep)
    for _ in range(n_cycles):
        for t in range(n_steps):
            phase = (t / n_steps) * 2 * np.pi
            data.ctrl[:] = twist_ctrl(POWER_GRASP_TARGET, amp, phase)
            mj.mj_step(model, data)


def run(model, amp, blocks, cycles_per_block, gait_slice):
    data = mj.MjData(model)
    mj.mj_forward(model, data)
    settle_object(model, data)
    ramp_to_target(model, data, POWER_GRASP_TARGET, ramp_seconds=1.0, hold_seconds=0.3)
    q0 = data.body("object").xquat.copy()

    angles = []
    for b in range(blocks):
        twist_cycles(model, data, amp, cycles_per_block)
        angles.append(quat_angle_deg(q0, data.body("object").xquat))
        if gait_slice is not None and b < blocks - 1:
            regrasp_finger(model, data, gait_slice, POWER_GRASP_TARGET)
    retained, dist = object_retained(data)
    return angles, retained, dist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", type=str, default="assets/scene_right.xml")
    ap.add_argument("--amplitude", type=float, default=0.4)
    ap.add_argument("--blocks", type=int, default=5)
    ap.add_argument("--cycles-per-block", type=int, default=10)
    args = ap.parse_args()

    model = mj.MjModel.from_xml_path(args.scene)
    total_cycles = args.blocks * args.cycles_per_block

    conditions = [("no gait", None), ("gait ring finger", RING_FINGER_SLICE),
                  ("gait middle finger", MIDDLE_FINGER_SLICE)]
    for label, gait_slice in conditions:
        angles, retained, dist = run(model, args.amplitude, args.blocks, args.cycles_per_block,
                                      gait_slice)
        print(f"{label} ({total_cycles} total twist cycles): "
              f"angle per block = {[round(a, 1) for a in angles]}, retained={retained} "
              f"(dist={dist:.3f})")
        print(f"SUMMARY_PARSEABLE condition={label.replace(' ', '_')} "
              f"total_cycles={total_cycles} final_angle_deg={angles[-1]:.2f} "
              f"retained={retained}")


if __name__ == "__main__":
    main()
