"""Renders the Stage 6 contact-orbit experiment: the two balls brought into actual geom-geom
contact (hand_env.BAODING_CONTACT_TARGET), then `twist_ctrl` applied -- shows the real but small
stable wobble, followed by the contact becoming unstable and one ball getting violently ejected.
Same "GIF for every notable attempt" convention as every other stage, including this negative one.

Usage:
    python3 demo_stage6_contact_orbit.py --record media/stage6_contact_orbit.gif
"""
import argparse
from pathlib import Path

import mujoco as mj
import numpy as np
from PIL import Image

from hand_env import (BAODING_CONTACT_TARGET, TWO_BALL_SCENE, ball_pair_angle_rad,
                       ball_separation_ok, ramp_to_target, settle_object, twist_ctrl)


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", type=str, default=TWO_BALL_SCENE)
    ap.add_argument("--amplitude", type=float, default=0.1)
    ap.add_argument("--cycles", type=int, default=10)
    ap.add_argument("--cycle-seconds", type=float, default=2.0)
    ap.add_argument("--record", type=str, default=None)
    ap.add_argument("--frame-stride", type=int, default=100)
    args = ap.parse_args()

    model = mj.MjModel.from_xml_path(args.scene)
    data = mj.MjData(model)
    mj.mj_forward(model, data)
    dt = model.opt.timestep

    renderer = mj.Renderer(model, height=240, width=320)
    cam = mj.MjvCamera()
    cam.type = mj.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = [0.0, 0.0, 0.03]
    cam.distance = 0.15
    cam.elevation = -35
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

    step_n(np.zeros(16), int(1.0 / dt))
    ramp_to_target(model, data, BAODING_CONTACT_TARGET, ramp_seconds=1.0, hold_seconds=0.3)
    _, sep0 = ball_separation_ok(data)
    angle_prev = ball_pair_angle_rad(data)
    cum = 0.0

    n_steps = int(args.cycle_seconds / dt)
    for cyc in range(args.cycles):
        for t in range(n_steps):
            phase = (t / n_steps) * 2 * np.pi
            step_n(twist_ctrl(BAODING_CONTACT_TARGET, args.amplitude, phase), 1)
        angle_cur = ball_pair_angle_rad(data)
        cum += _wrap(angle_cur - angle_prev)
        angle_prev = angle_cur
        _, sep = ball_separation_ok(data)
        print(f"cycle {cyc}: cum_orbit={np.degrees(cum):.1f}deg separation={sep:.4f}m "
              f"(start was {sep0:.4f}m)")

    if args.record:
        Path(args.record).parent.mkdir(parents=True, exist_ok=True)
        imgs = [Image.fromarray(f) for f in frames]
        imgs[0].save(args.record, save_all=True, append_images=imgs[1:], duration=40, loop=0)
        print(f"Saved {len(frames)} frames to {args.record}")

    renderer.close()


if __name__ == "__main__":
    main()
