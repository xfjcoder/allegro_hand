"""Renders a Stage 4 finger-gaiting attempt: twist blocks interleaved with ring-finger
release-and-recontact regrasps -- visually documents the (negative) result that this doesn't
extend Stage 3's rotation ceiling, same "GIF for every notable attempt" convention as every
other stage in this project family.

Usage:
    python3 demo_stage4_finger_gaiting.py --record media/stage4_finger_gaiting.gif
"""
import argparse
from pathlib import Path

import mujoco as mj
import numpy as np
from PIL import Image

from hand_env import POWER_GRASP_TARGET, quat_angle_deg, ramp_to_target, settle_object, twist_ctrl

RING_FINGER_SLICE = slice(8, 12)


def regrasp_with_render(data, step_n, finger_slice, hold_target, release_seconds=0.4,
                         hold_seconds=0.3, dt=0.002):
    """Same release-then-recontact sequence as hand_env.regrasp_finger, but routed through the
    caller's render-aware step_n so the regrasp itself is visible in the GIF."""
    release = hold_target.copy()
    release[finger_slice] = 0.0
    n = int(release_seconds / dt)
    start = data.ctrl.copy()
    for i in range(n):
        alpha = (i + 1) / n
        step_n(start + alpha * (release - start), 1)
    step_n(release, int(hold_seconds / dt))
    start2 = data.ctrl.copy()
    for i in range(n):
        alpha = (i + 1) / n
        step_n(start2 + alpha * (hold_target - start2), 1)
    step_n(hold_target, int(hold_seconds / dt))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", type=str, default="assets/scene_right.xml")
    ap.add_argument("--amplitude", type=float, default=0.4)
    ap.add_argument("--blocks", type=int, default=3)
    ap.add_argument("--cycles-per-block", type=int, default=10)
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

    n_steps = int(2.0 / dt)
    for b in range(args.blocks):
        for _ in range(args.cycles_per_block):
            for t in range(n_steps):
                phase = (t / n_steps) * 2 * np.pi
                step_n(twist_ctrl(POWER_GRASP_TARGET, args.amplitude, phase), 1)
        if b < args.blocks - 1:
            regrasp_with_render(data, step_n, RING_FINGER_SLICE, POWER_GRASP_TARGET, dt=dt)

    final_angle = quat_angle_deg(q0, data.body("object").xquat)
    print(f"Final rotation after {args.blocks} twist-blocks with {args.blocks - 1} ring-finger "
          f"regrasps between them: {final_angle:.1f} degrees (compare to Stage 3's ~23-25 "
          f"degree ceiling WITHOUT any regrasping -- gaiting doesn't extend it). "
          f"Rendered {len(frames)} frames.")

    if args.record:
        Path(args.record).parent.mkdir(parents=True, exist_ok=True)
        imgs = [Image.fromarray(f) for f in frames]
        imgs[0].save(args.record, save_all=True, append_images=imgs[1:], duration=40, loop=0)
        print(f"Saved {len(frames)} frames to {args.record}")

    renderer.close()


if __name__ == "__main__":
    main()
