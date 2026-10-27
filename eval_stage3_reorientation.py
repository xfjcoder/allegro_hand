"""Stage 3 feasibility check: how far can simultaneous finger motion (no lifting/re-contacting
fingers -- that's finger-gaiting, Stage 4) reorient the held object before it saturates?

Sweeps the twist oscillation's amplitude (does a bigger swing reach farther?) and repeated-cycle
count (does repeating it keep accumulating rotation, or hit a limit?) against both the object's
cumulative rotation (quaternion angle from its power-grasp starting orientation) and whether it's
still retained (same criterion as Stage 2) throughout.

Usage:
    python3 eval_stage3_reorientation.py --amplitudes 0.1,0.2,0.3,0.4 --max-cycles 6
"""
import argparse

import mujoco as mj
import numpy as np

from hand_env import (POWER_GRASP_TARGET, object_retained, quat_angle_deg, ramp_to_target,
                       settle_object, twist_ctrl)


def run(model, amp, n_cycles, cycle_seconds=2.0):
    data = mj.MjData(model)
    mj.mj_forward(model, data)
    settle_object(model, data)
    ramp_to_target(model, data, POWER_GRASP_TARGET, ramp_seconds=1.0, hold_seconds=0.3)
    q0 = data.body("object").xquat.copy()

    dt = model.opt.timestep
    n_steps = int(cycle_seconds / dt)
    angles, retained_each_cycle = [], []
    for _ in range(n_cycles):
        for t in range(n_steps):
            phase = (t / n_steps) * 2 * np.pi
            data.ctrl[:] = twist_ctrl(POWER_GRASP_TARGET, amp, phase)
            mj.mj_step(model, data)
        angles.append(quat_angle_deg(q0, data.body("object").xquat))
        retained, _ = object_retained(data)
        retained_each_cycle.append(retained)
    return angles, retained_each_cycle


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", type=str, default="assets/scene_right.xml")
    ap.add_argument("--amplitudes", type=str, default="0.1,0.2,0.3,0.4,0.47")
    ap.add_argument("--max-cycles", type=int, default=50,
                     help="A short sweep (e.g. 6-20 cycles) can look like unbounded linear "
                          "growth at higher amplitudes before it actually flattens -- confirmed "
                          "directly that 50 cycles is enough to see every tested amplitude "
                          "converge. Don't shorten this without re-checking that.")
    args = ap.parse_args()

    model = mj.MjModel.from_xml_path(args.scene)
    amplitudes = [float(a) for a in args.amplitudes.split(",")]

    for amp in amplitudes:
        angles, retained = run(model, amp, args.max_cycles)
        midpoint = angles[len(angles) // 2 - 1]
        print(f"amp={amp}: rotation at cycle {len(angles)//2}={midpoint:.1f}deg, "
              f"at cycle {len(angles)}={angles[-1]:.1f}deg, retained every cycle={all(retained)}")
        print(f"SUMMARY_PARSEABLE amp={amp} cycles={len(angles)} "
              f"midpoint_rotation_deg={midpoint:.2f} final_rotation_deg={angles[-1]:.2f}")


if __name__ == "__main__":
    main()
