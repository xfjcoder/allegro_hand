"""Stage 6 continued: tests `hand_env.four_finger_orbit_ctrl`, built from the user's observation
on the diagonal_stroke_ctrl GIF -- "middle finger and thumb block balls cross each other, can 4
fingers work in a coordinate way". diagonal_stroke_ctrl only moved ff/rf; mf and thumb sat rigid
the whole time, forming a static wall. This primitive phases all four digits a quarter-cycle
apart so a "gap" rotates around all four in sequence instead of staying fixed at ff/rf.

Default grasp/amplitudes are now the "crossover" variant (see
`hand_env.BAODING_GRASP_TARGET_LARGE_CROSSOVER`'s docstring): the user's next observation,
"finger are moving, but not in a way to push balls pass cross each other, the thumb only hold on
the ball", led to tracking each ball's OWN angular position (not just the pair-vector angle used
before) and auditing `data.contact` with a body-name (not geom-name) match. That found ball2 was
nearly pinned in a persistent 3-digit (mf+rf+thumb) pocket the whole cycle regardless of phase or
amplitude -- the fix was loosening mf's static grasp curl (0.85 -> 0.4), which unlocked real
circulation: ball2_cum reaches -172 degrees and ball1_cum -64 degrees over 20+ cycles, both
retained, instead of the previous -11/-60 split that was mostly just ball1 wobbling. Pass
`--grasp upright_thumb` or `--grasp original` to reproduce the earlier results.

Usage:
    python3 eval_stage6_four_finger_orbit.py --cycles 20 --verbose
    python3 eval_stage6_four_finger_orbit.py --grasp original --amp-lat 0.3 --amp-curl 0.15
"""
import argparse

import mujoco as mj
import numpy as np

from hand_env import (BAODING_GRASP_TARGET_LARGE, BAODING_GRASP_TARGET_LARGE_CROSSOVER,
                       BAODING_GRASP_TARGET_LARGE_UPRIGHT_THUMB, BALL_REST_POS_LARGE,
                       MAX_BALL_SEPARATION_LARGE, MIN_BALL_SEPARATION_LARGE, TWO_BALL_SCENE_LARGE,
                       ball_pair_angle_rad, ball_separation_ok, balls_retained,
                       four_finger_orbit_ctrl, ramp_to_target, settle_object)


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


PALM_CENTER = np.array([0.0, 0.0, 0.035])


def _ball_angle(data, name):
    p = data.body(name).xpos
    return np.arctan2(p[1] - PALM_CENTER[1], p[0] - PALM_CENTER[0])


def run(model, grasp_target, amp_lat, amp_curl, thumb_amp_base, thumb_amp_curl, n_cycles,
        cycle_seconds=2.0, verbose=False):
    """Returns (pair_cum, ball1_cum, ball2_cum, retained_all). The per-ball angles are the real
    cross-over check -- see BAODING_GRASP_TARGET_LARGE_CROSSOVER's docstring: the pair-vector
    metric alone can't distinguish "both balls genuinely circulating" from "one ball wobbles, the
    other sits still", since an asymmetric wobble still drifts the pair angle over time."""
    data = mj.MjData(model)
    mj.mj_forward(model, data)
    settle_object(model, data)
    ramp_to_target(model, data, grasp_target, ramp_seconds=1.0, hold_seconds=0.3)

    dt = model.opt.timestep
    n_steps = int(cycle_seconds / dt)
    angle_prev = ball_pair_angle_rad(data)
    b1_prev, b2_prev = _ball_angle(data, "ball1"), _ball_angle(data, "ball2")
    cum = cum1 = cum2 = 0.0
    retained_all = True
    for cyc in range(n_cycles):
        for t in range(n_steps):
            phase = (t / n_steps) * 2 * np.pi
            data.ctrl[:] = four_finger_orbit_ctrl(grasp_target, amp_lat, amp_curl, phase,
                                                   thumb_amp_base, thumb_amp_curl)
            mj.mj_step(model, data)
        angle_cur = ball_pair_angle_rad(data)
        cum += _wrap(angle_cur - angle_prev)
        angle_prev = angle_cur
        b1_cur, b2_cur = _ball_angle(data, "ball1"), _ball_angle(data, "ball2")
        cum1 += _wrap(b1_cur - b1_prev)
        cum2 += _wrap(b2_cur - b2_prev)
        b1_prev, b2_prev = b1_cur, b2_cur
        retained, _ = balls_retained(data, rest_pos=BALL_REST_POS_LARGE)
        _, sep = ball_separation_ok(data, MIN_BALL_SEPARATION_LARGE, MAX_BALL_SEPARATION_LARGE)
        retained_all = retained_all and retained
        if verbose:
            print(f"  cycle {cyc}: cum_orbit={np.degrees(cum):.1f}deg "
                  f"ball1={np.degrees(cum1):.1f}deg ball2={np.degrees(cum2):.1f}deg "
                  f"retained={retained} sep={sep:.4f}")
        if not retained:
            break
    return cum, cum1, cum2, retained_all


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", type=str, default=TWO_BALL_SCENE_LARGE)
    ap.add_argument("--grasp", choices=("crossover", "upright_thumb", "original"),
                     default="crossover")
    ap.add_argument("--amp-lat", type=float, default=0.4)
    ap.add_argument("--amp-curl", type=float, default=0.15)
    ap.add_argument("--thumb-amp-base", type=float, default=0.3)
    ap.add_argument("--thumb-amp-curl", type=float, default=0.25)
    ap.add_argument("--cycles", type=int, default=20)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    grasp_target = {"crossover": BAODING_GRASP_TARGET_LARGE_CROSSOVER,
                     "upright_thumb": BAODING_GRASP_TARGET_LARGE_UPRIGHT_THUMB,
                     "original": BAODING_GRASP_TARGET_LARGE}[args.grasp]
    model = mj.MjModel.from_xml_path(args.scene)
    cum, cum1, cum2, retained = run(model, grasp_target, args.amp_lat, args.amp_curl,
                                     args.thumb_amp_base, args.thumb_amp_curl, args.cycles,
                                     verbose=args.verbose)
    print(f"four_finger_orbit_ctrl ({args.grasp}): pair_angle cum = {np.degrees(cum):.1f} deg, "
          f"ball1_cum={np.degrees(cum1):.1f} deg, ball2_cum={np.degrees(cum2):.1f} deg over "
          f"{args.cycles} cycles, both balls retained={retained}")
    print(f"SUMMARY_PARSEABLE grasp={args.grasp} amp_lat={args.amp_lat} amp_curl={args.amp_curl} "
          f"thumb_amp_base={args.thumb_amp_base} thumb_amp_curl={args.thumb_amp_curl} "
          f"cycles={args.cycles} cumulative_orbit_deg={np.degrees(cum):.2f} "
          f"ball1_cum_deg={np.degrees(cum1):.2f} ball2_cum_deg={np.degrees(cum2):.2f} "
          f"retained={retained}")


if __name__ == "__main__":
    main()
