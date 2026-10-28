"""Renders hand_env.four_finger_orbit_ctrl -- the 4-digit rotating-gate primitive built from the
user's observation on the diagonal_stroke_ctrl GIF: "middle finger and thumb block balls cross
each other, can 4 fingers work in a coordinate way?"

Default grasp/amplitudes are the "upright thumb" variant
(`hand_env.BAODING_GRASP_TARGET_LARGE_UPRIGHT_THUMB`), from the user's next observation: "the
thumb curl all way down, that block the way in hand for balls pass to other side, the thumb
should be more in upright position". Re-tuning the thumb's curl_frac (0.85 -> 0.75) and
re-sweeping amp_lat/amp_curl for that geometry found amp_lat=0.35/amp_curl=0.12 converges to
-54.6 degrees, rock-stable, an 18% improvement over the fully-curled thumb's -46.6 degree
ceiling. Pass `--grasp original` to render the fully-curled-thumb version instead.

Usage:
    python3 demo_stage6_four_finger_orbit.py --record media/stage6_four_finger_orbit.gif
"""
import argparse
from pathlib import Path

import mujoco as mj
import numpy as np
from PIL import Image

from hand_env import (BAODING_GRASP_TARGET_LARGE, BAODING_GRASP_TARGET_LARGE_UPRIGHT_THUMB,
                       BALL_REST_POS_LARGE, MAX_BALL_SEPARATION_LARGE, MIN_BALL_SEPARATION_LARGE,
                       TWO_BALL_SCENE_LARGE, ball_pair_angle_rad, ball_separation_ok,
                       balls_retained, four_finger_orbit_ctrl, ramp_to_target, settle_object)


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", type=str, default=TWO_BALL_SCENE_LARGE)
    ap.add_argument("--grasp", choices=("upright_thumb", "original"), default="upright_thumb")
    ap.add_argument("--amp-lat", type=float, default=0.4)
    ap.add_argument("--amp-curl", type=float, default=0.15)
    ap.add_argument("--thumb-amp-base", type=float, default=0.0)
    ap.add_argument("--thumb-amp-curl", type=float, default=0.0)
    ap.add_argument("--cycles", type=int, default=15)
    ap.add_argument("--cycle-seconds", type=float, default=2.0)
    ap.add_argument("--record", type=str, default=None)
    ap.add_argument("--frame-stride", type=int, default=30)
    args = ap.parse_args()

    grasp_target = (BAODING_GRASP_TARGET_LARGE_UPRIGHT_THUMB if args.grasp == "upright_thumb"
                     else BAODING_GRASP_TARGET_LARGE)
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
        step_n(start + alpha * (grasp_target - start), 1)
    step_n(grasp_target, int(0.3 / dt))

    n_steps = int(args.cycle_seconds / dt)
    angle_prev = ball_pair_angle_rad(data)
    cum = 0.0
    for cyc in range(args.cycles):
        for t in range(n_steps):
            phase = (t / n_steps) * 2 * np.pi
            ctrl = four_finger_orbit_ctrl(grasp_target, args.amp_lat, args.amp_curl, phase,
                                           args.thumb_amp_base, args.thumb_amp_curl)
            step_n(ctrl, 1)
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
