"""Stage 6, the "sequenced contact planning" half of what the user asked for ("proceed with
Stage 6 (reward redesign vs. sequenced contact planning)"): instead of a smooth periodic
oscillation (twist_ctrl, orbit_wave_ctrl -- both settle to a fixed point, see their own
docstrings), alternately release-and-push each side (hand_env.leapfrog_half_cycle) to try to
"leapfrog" the balls past each other, on the larger ball scene.

Gets real progress (~20-28 degrees, tuning-dependent) but every parameter combination tried
eventually ejects a ball after a handful of cycles -- a genuine, repeatable instability, not
matched by the RL approach's own (different) "real but bounded" finding. See README Stage 6.

Usage:
    python3 eval_stage6_leapfrog.py --release-amt 0.2 --push-amt 0.25 --cycles 15
"""
import argparse

import mujoco as mj
import numpy as np

from hand_env import (BAODING_GRASP_TARGET_LARGE, BALL_REST_POS_LARGE, FF_CURL_SLICE,
                       MAX_BALL_SEPARATION_LARGE, MF_PROXIMAL_IDX, MIN_BALL_SEPARATION_LARGE,
                       RF_CURL_SLICE, TWO_BALL_SCENE_LARGE, ball_pair_angle_rad,
                       ball_separation_ok, balls_retained, leapfrog_half_cycle, ramp_to_target,
                       settle_object)


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", type=str, default=TWO_BALL_SCENE_LARGE)
    ap.add_argument("--release-amt", type=float, default=0.2)
    ap.add_argument("--push-amt", type=float, default=0.25)
    ap.add_argument("--cycles", type=int, default=15)
    args = ap.parse_args()

    model = mj.MjModel.from_xml_path(args.scene)
    data = mj.MjData(model)
    mj.mj_forward(model, data)
    settle_object(model, data)
    ramp_to_target(model, data, BAODING_GRASP_TARGET_LARGE, ramp_seconds=1.0, hold_seconds=0.3)

    angle_prev = ball_pair_angle_rad(data)
    cum = 0.0
    for cyc in range(args.cycles):
        leapfrog_half_cycle(model, data, BAODING_GRASP_TARGET_LARGE, RF_CURL_SLICE,
                             MF_PROXIMAL_IDX, args.release_amt, args.push_amt)
        leapfrog_half_cycle(model, data, BAODING_GRASP_TARGET_LARGE, FF_CURL_SLICE,
                             MF_PROXIMAL_IDX, args.release_amt, args.push_amt)
        angle_cur = ball_pair_angle_rad(data)
        cum += _wrap(angle_cur - angle_prev)
        angle_prev = angle_cur
        retained, _ = balls_retained(data, rest_pos=BALL_REST_POS_LARGE)
        _, sep = ball_separation_ok(data, MIN_BALL_SEPARATION_LARGE, MAX_BALL_SEPARATION_LARGE)
        print(f"cycle {cyc}: cum_orbit={np.degrees(cum):.1f}deg retained={retained} sep={sep:.4f}")
        print(f"SUMMARY_PARSEABLE cycle={cyc} cum_orbit_deg={np.degrees(cum):.2f} "
              f"retained={retained} separation={sep:.4f}")
        if not retained:
            break


if __name__ == "__main__":
    main()
