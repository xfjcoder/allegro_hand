"""Stage 6, testing the user's follow-up question: "the current baoding .gif show two are
separated with distance, no contact, no push each other" -- does bringing the two balls into
actual geom-geom contact (hand_env.BAODING_CONTACT_TARGET, vs. BAODING_GRASP_TARGET's ~0.041m
gap) let a scripted primitive drive real circulation, now that they can mechanically push on each
other?

Answer: no. `twist_ctrl` settles into a small stable wobble (~4-5 degrees) while contact genuinely
holds (separation pinned at ~0.030m), then after several more cycles the contact becomes unstable
and one ball is violently ejected (separation jumps ~4x in a single cycle). Contact wasn't the
missing ingredient -- circulation still needs a traveling/sequenced disturbance, the same
contact-planning problem Stage 4/5 hit for one object, now compounded by a second.

Usage:
    python3 eval_stage6_contact_orbit.py --amplitude 0.1 --cycles 12
"""
import argparse

import mujoco as mj
import numpy as np

from hand_env import (BAODING_CONTACT_TARGET, TWO_BALL_SCENE, balls_retained, ball_pair_angle_rad,
                       ball_separation_ok, ramp_to_target, settle_object, twist_ctrl)


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", type=str, default=TWO_BALL_SCENE)
    ap.add_argument("--amplitude", type=float, default=0.1)
    ap.add_argument("--cycles", type=int, default=12)
    args = ap.parse_args()

    model = mj.MjModel.from_xml_path(args.scene)
    data = mj.MjData(model)
    mj.mj_forward(model, data)
    settle_object(model, data)
    ramp_to_target(model, data, BAODING_CONTACT_TARGET, ramp_seconds=1.0, hold_seconds=0.3)
    _, sep0 = ball_separation_ok(data)
    print(f"starting separation: {sep0:.4f}m (touching ~0.030m)")

    dt = model.opt.timestep
    n_steps = int(2.0 / dt)
    angle_prev = ball_pair_angle_rad(data)
    cum = 0.0
    for cyc in range(args.cycles):
        for t in range(n_steps):
            phase = (t / n_steps) * 2 * np.pi
            data.ctrl[:] = twist_ctrl(BAODING_CONTACT_TARGET, args.amplitude, phase)
            mj.mj_step(model, data)
        angle_cur = ball_pair_angle_rad(data)
        cum += _wrap(angle_cur - angle_prev)
        angle_prev = angle_cur
        retained, _ = balls_retained(data)
        _, sep = ball_separation_ok(data)
        print(f"cycle {cyc}: cum_orbit={np.degrees(cum):.1f}deg retained={retained} sep={sep:.4f}")
        print(f"SUMMARY_PARSEABLE cycle={cyc} cum_orbit_deg={np.degrees(cum):.2f} "
              f"retained={retained} separation={sep:.4f}")


if __name__ == "__main__":
    main()
