"""Renders a Stage 6 Baoding-balls attempt: the `orbit_wave_ctrl` traveling-wave primitive (the
better of the two scripted candidates tested, see README/eval_stage6_baoding_balls.py) applied to
the two-ball scene -- visually documents the (negative, bounded) result that it settles into a
small repositioning rather than sustained orbiting. Same "GIF for every notable attempt"
convention as every other stage in this project family.

Usage:
    python3 demo_stage6_baoding_balls.py --record media/stage6_baoding_balls.gif
"""
import argparse
from pathlib import Path

import mujoco as mj
import numpy as np
from PIL import Image

from hand_env import (POWER_GRASP_TARGET, TWO_BALL_SCENE, ball_pair_angle_rad, orbit_wave_ctrl,
                       ramp_to_target, settle_object)


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", type=str, default=TWO_BALL_SCENE)
    ap.add_argument("--amplitude", type=float, default=0.47)
    ap.add_argument("--cycles", type=int, default=20)
    ap.add_argument("--cycle-seconds", type=float, default=2.0)
    ap.add_argument("--record", type=str, default=None)
    ap.add_argument("--frame-stride", type=int, default=150)
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

    step_n(np.zeros(16), int(1.0 / dt))
    ramp_to_target(model, data, POWER_GRASP_TARGET, ramp_seconds=1.0, hold_seconds=0.3)
    angle_prev = ball_pair_angle_rad(data)
    cum = 0.0

    n_steps = int(args.cycle_seconds / dt)
    for _ in range(args.cycles):
        for t in range(n_steps):
            phase = (t / n_steps) * 2 * np.pi
            step_n(orbit_wave_ctrl(POWER_GRASP_TARGET, args.amplitude, phase), 1)
        angle_cur = ball_pair_angle_rad(data)
        cum += _wrap(angle_cur - angle_prev)
        angle_prev = angle_cur

    print(f"Cumulative ball-pair orbit: {np.degrees(cum):.1f} degrees over {args.cycles} cycles "
          f"(amplitude={args.amplitude}) -- settles to a small bounded repositioning, not "
          f"sustained orbiting. Rendered {len(frames)} frames.")

    if args.record:
        Path(args.record).parent.mkdir(parents=True, exist_ok=True)
        imgs = [Image.fromarray(f) for f in frames]
        imgs[0].save(args.record, save_all=True, append_images=imgs[1:], duration=40, loop=0)
        print(f"Saved {len(frames)} frames to {args.record}")

    renderer.close()


if __name__ == "__main__":
    main()
