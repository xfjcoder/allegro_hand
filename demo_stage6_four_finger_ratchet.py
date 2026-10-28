"""Renders hand_env.four_finger_ratchet_cycle -- the step/hold (non-periodic) 4-digit sequence
built from the user's request "try the ratchet version on all four fingers", plus a deeper
thumb-only release (`thumb_amp_curl`) from the follow-up "the thumb seems not upright enough,
still block balls pass by each other".

Default config (amp_lat=0.3, amp_curl=0.3, dwell=0.4, thumb_amp_curl=0.6) converges to -62.9
degrees, rock-stable over 60 cycles -- the best result in this entire stage, ~15% better than
the sinusoidal four_finger_orbit_ctrl's -54.6 degree ceiling, found specifically by letting the
thumb release much deeper (0.6 vs the fingers' own 0.3) during its own dwell window than a
uniform amplitude would allow.

Usage:
    python3 demo_stage6_four_finger_ratchet.py --record media/stage6_four_finger_ratchet.gif
"""
import argparse
from pathlib import Path

import mujoco as mj
import numpy as np
from PIL import Image

from hand_env import (BAODING_GRASP_TARGET_LARGE_UPRIGHT_THUMB, BALL_REST_POS_LARGE,
                       MAX_BALL_SEPARATION_LARGE, MIN_BALL_SEPARATION_LARGE, TWO_BALL_SCENE_LARGE,
                       ball_pair_angle_rad, ball_separation_ok, balls_retained,
                       four_finger_ratchet_cycle, ramp_to_target, settle_object)


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", type=str, default=TWO_BALL_SCENE_LARGE)
    ap.add_argument("--amp-lat", type=float, default=0.3)
    ap.add_argument("--amp-curl", type=float, default=0.3)
    ap.add_argument("--dwell", type=float, default=0.4)
    ap.add_argument("--engage-extra", type=float, default=0.0)
    ap.add_argument("--thumb-amp-curl", type=float, default=0.6)
    ap.add_argument("--cycles", type=int, default=20)
    ap.add_argument("--record", type=str, default=None)
    ap.add_argument("--frame-stride", type=int, default=15)
    args = ap.parse_args()

    model = mj.MjModel.from_xml_path(args.scene)
    data = mj.MjData(model)
    mj.mj_forward(model, data)

    renderer = mj.Renderer(model, height=240, width=320)
    cam = mj.MjvCamera()
    cam.type = mj.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = [0.0, 0.0, 0.035]
    cam.distance = 0.18
    cam.elevation = -75
    cam.azimuth = 200

    frames = []
    step_count = [0]

    def step_n(ctrl, n):
        for _ in range(n):
            data.ctrl[:] = ctrl
            mj.mj_step(model, data)
            step_count[0] += 1
            if step_count[0] % args.frame_stride == 0:
                renderer.update_scene(data, camera=cam)
                frames.append(renderer.render().copy())

    settle_object(model, data)
    dt = model.opt.timestep
    n_ramp = int(1.0 / dt)
    start = data.ctrl.copy()
    for i in range(n_ramp):
        alpha = (i + 1) / n_ramp
        step_n(start + alpha * (BAODING_GRASP_TARGET_LARGE_UPRIGHT_THUMB - start), 1)
    step_n(BAODING_GRASP_TARGET_LARGE_UPRIGHT_THUMB, int(0.3 / dt))

    def ratchet_cycle_rendered():
        from hand_env import FOUR_DIGIT_ORDER, four_finger_ratchet_ctrl
        n_dwell = int(args.dwell / dt)
        for digit in FOUR_DIGIT_ORDER:
            ctrl = four_finger_ratchet_ctrl(BAODING_GRASP_TARGET_LARGE_UPRIGHT_THUMB, digit,
                                             args.amp_lat, args.amp_curl, args.engage_extra,
                                             args.thumb_amp_curl)
            step_n(ctrl, n_dwell)
        step_n(BAODING_GRASP_TARGET_LARGE_UPRIGHT_THUMB, 1)

    angle_prev = ball_pair_angle_rad(data)
    cum = 0.0
    for cyc in range(args.cycles):
        ratchet_cycle_rendered()
        angle_cur = ball_pair_angle_rad(data)
        cum += _wrap(angle_cur - angle_prev)
        angle_prev = angle_cur
        retained, _ = balls_retained(data, rest_pos=BALL_REST_POS_LARGE)
        _, sep = ball_separation_ok(data, MIN_BALL_SEPARATION_LARGE, MAX_BALL_SEPARATION_LARGE)
        print(f"cycle {cyc}: cum_orbit={np.degrees(cum):.1f}deg retained={retained} "
              f"sep={sep:.4f}")
        if not retained:
            break

    print(f"Rendered {len(frames)} frames.")
    if args.record:
        Path(args.record).parent.mkdir(parents=True, exist_ok=True)
        imgs = [Image.fromarray(f) for f in frames]
        imgs[0].save(args.record, save_all=True, append_images=imgs[1:], duration=40, loop=0)
        print(f"Saved {len(frames)} frames to {args.record}")

    renderer.close()


if __name__ == "__main__":
    main()
