"""Stage 6 feasibility check: can simultaneous-finger-motion scripted primitives (the kind that
worked, within a bound, for single-object reorientation in Stage 3) make the two-ball pair
(`assets/scene_two_balls.xml`) actually ORBIT around each other -- real Baoding-ball motion --
rather than just jitter in place?

Tries two candidate primitives: `hand_env.twist_ctrl` (Stage 3's antiphase base-joint oscillation,
reused as-is) and `hand_env.orbit_wave_ctrl` (a new phase-shifted traveling-wave curl pattern,
designed specifically to try to "paddle" the balls around). Reports the unwrapped cumulative
change in the ball-pair angle (same accumulate-every-step discipline as Stage 5's
`twist_angle_rad`, for the same reason: a one-shot start-vs-end angle can't tell a real multi-turn
orbit from "wiggled and came back").

Usage:
    python3 eval_stage6_baoding_balls.py --cycles 25
"""
import argparse

import mujoco as mj
import numpy as np

from hand_env import (POWER_GRASP_TARGET, TWO_BALL_SCENE, balls_retained, ball_pair_angle_rad,
                       orbit_wave_ctrl, ramp_to_target, settle_object, twist_ctrl)


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def run(model, ctrl_fn, amp, n_cycles, cycle_seconds=2.0):
    data = mj.MjData(model)
    mj.mj_forward(model, data)
    settle_object(model, data)
    ramp_to_target(model, data, POWER_GRASP_TARGET, ramp_seconds=1.0, hold_seconds=0.3)

    dt = model.opt.timestep
    n_steps = int(cycle_seconds / dt)
    angle_prev = ball_pair_angle_rad(data)
    cum = 0.0
    retained_all = True
    for _ in range(n_cycles):
        for t in range(n_steps):
            phase = (t / n_steps) * 2 * np.pi
            data.ctrl[:] = ctrl_fn(POWER_GRASP_TARGET, amp, phase)
            mj.mj_step(model, data)
        angle_cur = ball_pair_angle_rad(data)
        cum += _wrap(angle_cur - angle_prev)
        angle_prev = angle_cur
        retained, _ = balls_retained(data)
        retained_all = retained_all and retained
    return cum, retained_all


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", type=str, default=TWO_BALL_SCENE)
    ap.add_argument("--amplitude", type=float, default=0.47)
    ap.add_argument("--cycles", type=int, default=30,
                     help="A short sweep can look like real progress (e.g. orbit_wave_ctrl at "
                          "amp=0.47 reads 28.6 degrees at cycle 2) before settling into a small "
                          "BOUNDED repositioning around cycle 15 -- same 'don't judge early' "
                          "lesson as Stage 3. 30 cycles is enough to see every tested amplitude "
                          "flatten.")
    args = ap.parse_args()

    model = mj.MjModel.from_xml_path(args.scene)

    for label, ctrl_fn in (("twist_ctrl (Stage 3, reused)", twist_ctrl),
                            ("orbit_wave_ctrl (new)", orbit_wave_ctrl)):
        cum, retained = run(model, ctrl_fn, args.amplitude, args.cycles)
        print(f"{label}: cumulative orbit = {np.degrees(cum):.1f} degrees over {args.cycles} "
              f"cycles, both balls retained={retained}")
        print(f"SUMMARY_PARSEABLE primitive={ctrl_fn.__name__} cycles={args.cycles} "
              f"cumulative_orbit_deg={np.degrees(cum):.2f} retained={retained}")


if __name__ == "__main__":
    main()
