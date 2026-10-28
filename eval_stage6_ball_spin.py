"""Stage 6, reframed per user question ("if two balls rotate against each other with finger
force, will that make it easy?"): instead of ORBITING the pair around a shared center (the
original framing -- hard, and the RL attempt's early readout turned out to be one ball getting
ejected sideways, not real circulation, see README), what if the two balls stay roughly in place
and SPIN against each other via mutual rolling contact, like two meshing gears?

Confirmed directly: reusing Stage 3's own `twist_ctrl` primitive UNCHANGED (just retargeted at
`BAODING_GRASP_TARGET` instead of the single-object `POWER_GRASP_TARGET`) makes each ball spin on
its own axis, in OPPOSITE directions (consistent with rolling-without-slipping contact between
them), with NO saturation over 80 cycles (160 seconds) of testing -- a constant ~2.1 degrees/cycle
rate the entire time, unlike every other primitive tried in this project (Stage 3's own single-
object case, Stage 6's orbit attempts) which all flatten to a bounded ceiling. Ball-pair
separation stays essentially fixed (~0.041m, no ejection) throughout. The mechanical reason this
is different: Stage 3's bounded ceiling came from a static-friction-cone limit on simultaneous-
contact REPOSITIONING of a single grasped object; a sphere rolling against another sphere under
sustained contact has no such limit -- rotation about the rolling axis isn't resisted by friction
the way translation/repositioning is.

Usage:
    python3 eval_stage6_ball_spin.py --amplitude 0.4 --cycles 50
"""
import argparse

import mujoco as mj
import numpy as np

from hand_env import (BAODING_GRASP_TARGET, TWO_BALL_SCENE, balls_retained, ball_separation_ok,
                       ramp_to_target, settle_object, twist_angle_rad, twist_ctrl)


def run(model, amp, n_cycles, cycle_seconds=2.0):
    data = mj.MjData(model)
    mj.mj_forward(model, data)
    settle_object(model, data)
    ramp_to_target(model, data, BAODING_GRASP_TARGET, ramp_seconds=1.0, hold_seconds=0.3)

    dt = model.opt.timestep
    n_steps = int(cycle_seconds / dt)
    q1_prev = data.body("ball1").xquat.copy()
    q2_prev = data.body("ball2").xquat.copy()
    cum1, cum2 = 0.0, 0.0
    retained_all = True
    for _ in range(n_cycles):
        for t in range(n_steps):
            phase = (t / n_steps) * 2 * np.pi
            data.ctrl[:] = twist_ctrl(BAODING_GRASP_TARGET, amp, phase)
            mj.mj_step(model, data)
        q1_cur = data.body("ball1").xquat.copy()
        q2_cur = data.body("ball2").xquat.copy()
        cum1 += twist_angle_rad(q1_prev, q1_cur)
        cum2 += twist_angle_rad(q2_prev, q2_cur)
        q1_prev, q2_prev = q1_cur, q2_cur
        retained, _ = balls_retained(data)
        retained_all = retained_all and retained
    _, sep = ball_separation_ok(data)
    return cum1, cum2, retained_all, sep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", type=str, default=TWO_BALL_SCENE)
    ap.add_argument("--amplitude", type=float, default=0.4)
    ap.add_argument("--cycles", type=int, default=50)
    args = ap.parse_args()

    model = mj.MjModel.from_xml_path(args.scene)
    cum1, cum2, retained, sep = run(model, args.amplitude, args.cycles)
    print(f"amp={args.amplitude}, {args.cycles} cycles: ball1_spin={np.degrees(cum1):.1f}deg "
          f"ball2_spin={np.degrees(cum2):.1f}deg (opposite directions = rolling contact, not "
          f"independent wobble), retained={retained}, final_separation={sep:.4f}m")
    print(f"SUMMARY_PARSEABLE amp={args.amplitude} cycles={args.cycles} "
          f"ball1_spin_deg={np.degrees(cum1):.2f} ball2_spin_deg={np.degrees(cum2):.2f} "
          f"retained={retained} separation={sep:.4f}")


if __name__ == "__main__":
    main()
