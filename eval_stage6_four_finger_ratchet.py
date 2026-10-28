"""Stage 6 continued: tests `hand_env.four_finger_ratchet_cycle`, built from the user's request
"try the ratchet version on all four fingers" -- a step/hold sequence (one digit open at a time,
advancing ff->mf->rf->th, no smooth sinusoidal return) instead of `four_finger_orbit_ctrl`'s
continuous sin/cos sweep, which must retrace every digit's own path each cycle.

Usage:
    python3 eval_stage6_four_finger_ratchet.py --cycles 15
    python3 eval_stage6_four_finger_ratchet.py --amp-lat 0.3 --amp-curl 0.3 --dwell 0.4 --cycles 15
"""
import argparse

import mujoco as mj
import numpy as np

from hand_env import (BAODING_GRASP_TARGET_LARGE_UPRIGHT_THUMB, BALL_REST_POS_LARGE,
                       MAX_BALL_SEPARATION_LARGE, MIN_BALL_SEPARATION_LARGE, TWO_BALL_SCENE_LARGE,
                       ball_pair_angle_rad, ball_separation_ok, balls_retained,
                       four_finger_ratchet_cycle, ramp_to_target, settle_object)


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def run(model, amp_lat, amp_curl, dwell_seconds, engage_extra, n_cycles, thumb_amp_curl=None,
        verbose=False):
    data = mj.MjData(model)
    mj.mj_forward(model, data)
    settle_object(model, data)
    ramp_to_target(model, data, BAODING_GRASP_TARGET_LARGE_UPRIGHT_THUMB, ramp_seconds=1.0,
                    hold_seconds=0.3)

    angle_prev = ball_pair_angle_rad(data)
    cum = 0.0
    retained_all = True
    for cyc in range(n_cycles):
        four_finger_ratchet_cycle(model, data, BAODING_GRASP_TARGET_LARGE_UPRIGHT_THUMB, amp_lat,
                                   amp_curl, dwell_seconds, engage_extra,
                                   thumb_amp_curl=thumb_amp_curl)
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
    ap.add_argument("--amp-lat", type=float, default=0.25)
    ap.add_argument("--amp-curl", type=float, default=0.25)
    ap.add_argument("--dwell", type=float, default=0.3)
    ap.add_argument("--engage-extra", type=float, default=0.0)
    ap.add_argument("--thumb-amp-curl", type=float, default=None)
    ap.add_argument("--cycles", type=int, default=15)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    model = mj.MjModel.from_xml_path(args.scene)
    cum, retained = run(model, args.amp_lat, args.amp_curl, args.dwell, args.engage_extra,
                         args.cycles, thumb_amp_curl=args.thumb_amp_curl, verbose=args.verbose)
    print(f"four_finger_ratchet: cumulative orbit = {np.degrees(cum):.1f} degrees over "
          f"{args.cycles} cycles, both balls retained={retained}")
    print(f"SUMMARY_PARSEABLE amp_lat={args.amp_lat} amp_curl={args.amp_curl} "
          f"dwell={args.dwell} engage_extra={args.engage_extra} cycles={args.cycles} "
          f"cumulative_orbit_deg={np.degrees(cum):.2f} retained={retained}")


if __name__ == "__main__":
    main()
