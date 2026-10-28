"""Renders the Stage 6 leapfrog scripted attempt (hand_env.leapfrog_half_cycle): real progress,
then eventual ejection. Same "GIF for every notable attempt" convention as every other stage.

Usage:
    python3 demo_stage6_leapfrog.py --record media/stage6_leapfrog.gif
"""
import argparse
from pathlib import Path

import mujoco as mj
import numpy as np
from PIL import Image

from hand_env import (BAODING_GRASP_TARGET_LARGE, FF_CURL_SLICE, MF_PROXIMAL_IDX, RF_CURL_SLICE,
                       TWO_BALL_SCENE_LARGE, ball_pair_angle_rad, ramp_to_target, settle_object)


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", type=str, default=TWO_BALL_SCENE_LARGE)
    ap.add_argument("--release-amt", type=float, default=0.2)
    ap.add_argument("--push-amt", type=float, default=0.25)
    ap.add_argument("--cycles", type=int, default=9)
    ap.add_argument("--record", type=str, default=None)
    ap.add_argument("--frame-stride", type=int, default=30)
    args = ap.parse_args()

    model = mj.MjModel.from_xml_path(args.scene)
    data = mj.MjData(model)
    mj.mj_forward(model, data)
    dt = model.opt.timestep

    renderer = mj.Renderer(model, height=240, width=320)
    cam = mj.MjvCamera()
    cam.type = mj.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = [0.0, 0.0, 0.035]
    cam.distance = 0.2
    cam.elevation = -50
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

    def leapfrog_half_cycle_rendered(release_slice, push_idx, release_amt, push_amt,
                                      duration_seconds=0.6, hold_seconds=0.2):
        n_half = int(duration_seconds / dt)
        for _ in range(n_half):
            c = BAODING_GRASP_TARGET_LARGE.copy()
            c[release_slice] = BAODING_GRASP_TARGET_LARGE[release_slice] * (1 - release_amt)
            c[push_idx] = min(BAODING_GRASP_TARGET_LARGE[push_idx] + push_amt, 1.6)
            step_n(c, 1)
        step_n(BAODING_GRASP_TARGET_LARGE, int(hold_seconds / dt))

    step_n(np.zeros(16), int(1.0 / dt))
    ramp_to_target(model, data, BAODING_GRASP_TARGET_LARGE, ramp_seconds=1.0, hold_seconds=0.3)
    angle_prev = ball_pair_angle_rad(data)
    cum = 0.0

    for cyc in range(args.cycles):
        leapfrog_half_cycle_rendered(RF_CURL_SLICE, MF_PROXIMAL_IDX, args.release_amt,
                                      args.push_amt)
        leapfrog_half_cycle_rendered(FF_CURL_SLICE, MF_PROXIMAL_IDX, args.release_amt,
                                      args.push_amt)
        angle_cur = ball_pair_angle_rad(data)
        cum += _wrap(angle_cur - angle_prev)
        angle_prev = angle_cur

    print(f"Cumulative orbit: {np.degrees(cum):.1f} degrees over {args.cycles} leapfrog cycles. "
          f"Rendered {len(frames)} frames.")

    if args.record:
        Path(args.record).parent.mkdir(parents=True, exist_ok=True)
        imgs = [Image.fromarray(f) for f in frames]
        imgs[0].save(args.record, save_all=True, append_images=imgs[1:], duration=40, loop=0)
        print(f"Saved {len(frames)} frames to {args.record}")

    renderer.close()


if __name__ == "__main__":
    main()
