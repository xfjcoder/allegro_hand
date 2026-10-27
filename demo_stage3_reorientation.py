"""Renders the Stage 3 twist primitive reorienting the held object toward its ~23-25 degree
saturation ceiling (see README Stage 3 / hand_env.py's twist_ctrl docstring).

Usage:
    python3 demo_stage3_reorientation.py --amplitude 0.4 --cycles 50 --record media/stage3_reorientation.gif
"""
import argparse
from pathlib import Path

import mujoco as mj
import numpy as np
from PIL import Image

from hand_env import POWER_GRASP_TARGET, quat_angle_deg, ramp_to_target, settle_object, twist_ctrl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", type=str, default="assets/scene_right.xml")
    ap.add_argument("--amplitude", type=float, default=0.4)
    ap.add_argument("--cycles", type=int, default=25,
                     help="25 (not the full 50-cycle sweep to the asymptote) already covers "
                          "~22.8 of amp=0.4's ~23.4 degree ceiling -- plenty to show visually, "
                          "and keeps the GIF a reasonable size/length.")
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
    cam.lookat[:] = [0.0, 0.0, 0.03]
    cam.distance = 0.35
    cam.elevation = -20
    cam.azimuth = 120

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
    q0 = data.body("object").xquat.copy()

    n_steps = int(args.cycle_seconds / dt)
    for _ in range(args.cycles):
        for t in range(n_steps):
            phase = (t / n_steps) * 2 * np.pi
            step_n(twist_ctrl(POWER_GRASP_TARGET, args.amplitude, phase), 1)

    final_angle = quat_angle_deg(q0, data.body("object").xquat)
    print(f"Final cumulative rotation: {final_angle:.1f} degrees over {args.cycles} cycles "
          f"(amplitude={args.amplitude}). Rendered {len(frames)} frames.")

    if args.record:
        Path(args.record).parent.mkdir(parents=True, exist_ok=True)
        imgs = [Image.fromarray(f) for f in frames]
        imgs[0].save(args.record, save_all=True, append_images=imgs[1:], duration=40, loop=0)
        print(f"Saved {len(frames)} frames to {args.record}")

    renderer.close()


if __name__ == "__main__":
    main()
