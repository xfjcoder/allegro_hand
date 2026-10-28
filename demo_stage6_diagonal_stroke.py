"""Renders hand_env.diagonal_stroke_ctrl -- the lateral+curl quadrature primitive built from the
user's observation "only three fingers make efforts and the thumb not involved much, and only
make balls spin, not push balls to opposite direction and cross each other, can finger move in
cross direction bottom right to top left and verce vise?"

Uses the amp_lat=0.15/amp_curl=0.25, no-thumb config (see eval_stage6_diagonal_stroke.py): the
best raw progress found (~-34 degrees before ejection around cycle 12), to show what the diagonal
stroke actually looks like before the separation-drift failure kicks in.

Usage:
    python3 demo_stage6_diagonal_stroke.py --record media/stage6_diagonal_stroke.gif
"""
import argparse
from pathlib import Path

import mujoco as mj
import numpy as np
from PIL import Image

from hand_env import (BAODING_GRASP_TARGET_LARGE, BALL_REST_POS_LARGE, MAX_BALL_SEPARATION_LARGE,
                       MIN_BALL_SEPARATION_LARGE, TWO_BALL_SCENE_LARGE, ball_pair_angle_rad,
                       ball_separation_ok, balls_retained, diagonal_stroke_ctrl, ramp_to_target,
                       settle_object)


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", type=str, default=TWO_BALL_SCENE_LARGE)
    ap.add_argument("--amp-lat", type=float, default=0.15)
    ap.add_argument("--amp-curl", type=float, default=0.25)
    ap.add_argument("--thumb-amp-base", type=float, default=0.0)
    ap.add_argument("--thumb-amp-curl", type=float, default=0.0)
    ap.add_argument("--cycles", type=int, default=13)
    ap.add_argument("--cycle-seconds", type=float, default=2.0)
    ap.add_argument("--record", type=str, default=None)
    ap.add_argument("--frame-stride", type=int, default=30)
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
        step_n(start + alpha * (BAODING_GRASP_TARGET_LARGE - start), 1)
    step_n(BAODING_GRASP_TARGET_LARGE, int(0.3 / dt))

    n_steps = int(args.cycle_seconds / dt)
    angle_prev = ball_pair_angle_rad(data)
    cum = 0.0
    for cyc in range(args.cycles):
        for t in range(n_steps):
            phase = (t / n_steps) * 2 * np.pi
            ctrl = diagonal_stroke_ctrl(BAODING_GRASP_TARGET_LARGE, args.amp_lat, args.amp_curl,
                                         phase, args.thumb_amp_base, args.thumb_amp_curl)
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
