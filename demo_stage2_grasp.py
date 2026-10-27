"""Renders the Stage 2 grasp-stability comparison: the passive open-hand pose losing the object
to a push, side by side with the active power-grasp retaining it under the identical push.

Usage:
    python3 demo_stage2_grasp.py --magnitude 1.0 --record media/stage2_grasp_stability.gif
"""
import argparse
from pathlib import Path

import mujoco as mj
import numpy as np
from PIL import Image

from hand_env import N_FINGER_JOINTS, POWER_GRASP_TARGET, apply_push, ramp_to_target, settle_object


def rollout(model, push_vel, grasp, renderer, cam, frame_stride):
    """Runs the full setup (settle, then the grasp ramp if `grasp`) WITHOUT rendering (the two
    conditions take different amounts of setup time, which would otherwise misalign the push
    moment between them), then renders only a short pre-push lead-in plus the full post-push
    hold -- so the combined side-by-side GIF shows both conditions' push response starting at
    the same frame, not each condition's own setup phase."""
    data = mj.MjData(model)
    mj.mj_forward(model, data)
    dt = model.opt.timestep

    def step_n(ctrl, n, render_list=None):
        for i in range(n):
            data.ctrl[:] = ctrl
            mj.mj_step(model, data)
            if render_list is not None and i % frame_stride == 0:
                renderer.update_scene(data, camera=cam)
                render_list.append(renderer.render().copy())

    step_n(np.zeros(N_FINGER_JOINTS), int(1.0 / dt))  # settle, no render
    if grasp:
        start = data.ctrl.copy()
        n_ramp = int(1.0 / dt)
        for i in range(n_ramp):
            alpha = (i + 1) / n_ramp
            step_n(start + alpha * (POWER_GRASP_TARGET - start), 1)
        step_n(POWER_GRASP_TARGET, int(0.5 / dt))  # no render
        hold_ctrl = POWER_GRASP_TARGET
    else:
        hold_ctrl = np.zeros(N_FINGER_JOINTS)

    frames = []
    step_n(hold_ctrl, int(0.2 / dt), render_list=frames)  # short pre-push lead-in, rendered
    apply_push(data, push_vel)
    step_n(hold_ctrl, int(1.0 / dt), render_list=frames)  # push + aftermath, rendered
    return frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", type=str, default="assets/scene_right.xml")
    ap.add_argument("--magnitude", type=float, default=1.0)
    ap.add_argument("--angle", type=float, default=0.3, help="Push direction, radians.")
    ap.add_argument("--record", type=str, default=None)
    args = ap.parse_args()

    model = mj.MjModel.from_xml_path(args.scene)
    renderer = mj.Renderer(model, height=240, width=320)
    cam = mj.MjvCamera()
    cam.type = mj.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = [0.0, 0.0, 0.03]
    cam.distance = 0.35
    cam.elevation = -20
    cam.azimuth = 120

    push = args.magnitude * np.array([np.cos(args.angle), np.sin(args.angle), 0.0])
    open_frames = rollout(model, push, grasp=False, renderer=renderer, cam=cam, frame_stride=20)
    renderer.close()
    renderer = mj.Renderer(model, height=240, width=320)
    grasp_frames = rollout(model, push, grasp=True, renderer=renderer, cam=cam, frame_stride=20)
    renderer.close()

    n = min(len(open_frames), len(grasp_frames))
    combined = [np.concatenate([open_frames[i], grasp_frames[i]], axis=1) for i in range(n)]
    print(f"open-hand frames={len(open_frames)}, power-grasp frames={len(grasp_frames)}, "
          f"combined={n} (left=open-hand loses the object, right=power-grasp retains it)")

    if args.record:
        Path(args.record).parent.mkdir(parents=True, exist_ok=True)
        imgs = [Image.fromarray(f) for f in combined]
        imgs[0].save(args.record, save_all=True, append_images=imgs[1:], duration=40, loop=0)
        print(f"Saved {len(combined)} frames to {args.record}")


if __name__ == "__main__":
    main()
