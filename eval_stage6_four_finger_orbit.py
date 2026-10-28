"""Stage 6 continued: tests `hand_env.four_finger_orbit_ctrl`, built from the user's observation
on the diagonal_stroke_ctrl GIF -- "middle finger and thumb block balls cross each other, can 4
fingers work in a coordinate way". diagonal_stroke_ctrl only moved ff/rf; mf and thumb sat rigid
the whole time, forming a static wall. This primitive phases all four digits a quarter-cycle
apart so a "gap" rotates around all four in sequence instead of staying fixed at ff/rf.

Usage:
    python3 eval_stage6_four_finger_orbit.py --cycles 20
    python3 eval_stage6_four_finger_orbit.py --thumb-amp-base 0.2 --thumb-amp-curl 0.2 --cycles 20
"""
import argparse

import mujoco as mj
import numpy as np

from hand_env import (BAODING_GRASP_TARGET_LARGE, BALL_REST_POS_LARGE, MAX_BALL_SEPARATION_LARGE,
                       MIN_BALL_SEPARATION_LARGE, TWO_BALL_SCENE_LARGE, ball_pair_angle_rad,
                       ball_separation_ok, balls_retained, four_finger_orbit_ctrl, ramp_to_target,
                       settle_object)


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def run(model, amp_lat, amp_curl, thumb_amp_base, thumb_amp_curl, n_cycles, cycle_seconds=2.0,
        verbose=False):
    data = mj.MjData(model)
    mj.mj_forward(model, data)
    settle_object(model, data)
    ramp_to_target(model, data, BAODING_GRASP_TARGET_LARGE, ramp_seconds=1.0, hold_seconds=0.3)

    dt = model.opt.timestep
    n_steps = int(cycle_seconds / dt)
    angle_prev = ball_pair_angle_rad(data)
    cum = 0.0
    retained_all = True
    for cyc in range(n_cycles):
        for t in range(n_steps):
            phase = (t / n_steps) * 2 * np.pi
            data.ctrl[:] = four_finger_orbit_ctrl(BAODING_GRASP_TARGET_LARGE, amp_lat, amp_curl,
                                                   phase, thumb_amp_base, thumb_amp_curl)
            mj.mj_step(model, data)
        angle_cur = ball_pair_angle_rad(data)
        cum += _wrap(angle_cur - angle_prev)
        angle_prev = angle_cur
        retained, _ = balls_retained(data, rest_pos=BALL_REST_POS_LARGE)
        _, sep = ball_separation_ok(data, MIN_BALL_SEPARATION_LARGE, MAX_BALL_SEPARATION_LARGE)
        retained_all = retained_all and retained
        if verbose:
            print(f"  cycle {cyc}: cum_orbit={np.degrees(cum):.1f}deg retained={retained} "
                  f"sep={sep:.4f}")
        if not retained:
            break
    return cum, retained_all


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", type=str, default=TWO_BALL_SCENE_LARGE)
    ap.add_argument("--amp-lat", type=float, default=0.15)
    ap.add_argument("--amp-curl", type=float, default=0.25)
    ap.add_argument("--thumb-amp-base", type=float, default=0.0)
    ap.add_argument("--thumb-amp-curl", type=float, default=0.0)
    ap.add_argument("--cycles", type=int, default=20)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    model = mj.MjModel.from_xml_path(args.scene)
    cum, retained = run(model, args.amp_lat, args.amp_curl, args.thumb_amp_base,
                         args.thumb_amp_curl, args.cycles, verbose=args.verbose)
    print(f"four_finger_orbit_ctrl: cumulative orbit = {np.degrees(cum):.1f} degrees over "
          f"{args.cycles} cycles, both balls retained={retained}")
    print(f"SUMMARY_PARSEABLE amp_lat={args.amp_lat} amp_curl={args.amp_curl} "
          f"thumb_amp_base={args.thumb_amp_base} thumb_amp_curl={args.thumb_amp_curl} "
          f"cycles={args.cycles} cumulative_orbit_deg={np.degrees(cum):.2f} retained={retained}")


if __name__ == "__main__":
    main()
