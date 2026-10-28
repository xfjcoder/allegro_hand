"""Renders the Stage 6 ball-vs-ball rolling-spin finding: Stage 3's `twist_ctrl` primitive,
retargeted at `BAODING_GRASP_TARGET`, makes the two balls spin against each other continuously
(no saturation, unlike every other primitive in this project) rather than orbiting a shared
center. Each ball has a small white marker dot (see assets/scene_two_balls.xml) so the spin is
actually visible -- a plain-colored sphere's rotation is otherwise invisible in a render even
when it's really happening.

Usage:
    python3 demo_stage6_ball_spin.py --record media/stage6_ball_spin.gif
"""
import argparse
from pathlib import Path

import mujoco as mj
import numpy as np
from PIL import Image

from hand_env import (BAODING_GRASP_TARGET, TWO_BALL_SCENE, ramp_to_target, settle_object,
                       twist_angle_rad, twist_ctrl)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", type=str, default=TWO_BALL_SCENE)
    ap.add_argument("--amplitude", type=float, default=0.4)
    ap.add_argument("--cycles", type=int, default=15)
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
    ramp_to_target(model, data, BAODING_GRASP_TARGET, ramp_seconds=1.0, hold_seconds=0.3)
    q1_0 = data.body("ball1").xquat.copy()
    q2_0 = data.body("ball2").xquat.copy()

    n_steps = int(args.cycle_seconds / dt)
    for _ in range(args.cycles):
        for t in range(n_steps):
            phase = (t / n_steps) * 2 * np.pi
            step_n(twist_ctrl(BAODING_GRASP_TARGET, args.amplitude, phase), 1)

    q1_f = data.body("ball1").xquat.copy()
    q2_f = data.body("ball2").xquat.copy()
    print(f"ball1 net spin: {np.degrees(twist_angle_rad(q1_0, q1_f)):.1f} deg "
          f"(NOTE: quat_angle-style readout from start wraps at 180 -- see eval script's "
          f"per-cycle accumulation for the real unwrapped total)")
    print(f"ball2 net spin: {np.degrees(twist_angle_rad(q2_0, q2_f)):.1f} deg")
    print(f"Rendered {len(frames)} frames over {args.cycles} cycles.")

    if args.record:
        Path(args.record).parent.mkdir(parents=True, exist_ok=True)
        imgs = [Image.fromarray(f) for f in frames]
        imgs[0].save(args.record, save_all=True, append_images=imgs[1:], duration=40, loop=0)
        print(f"Saved {len(frames)} frames to {args.record}")

    renderer.close()


if __name__ == "__main__":
    main()
