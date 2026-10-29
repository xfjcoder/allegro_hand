"""Renders the Stage 7 toss-and-catch cycle (`hand_env.TossCatchController`, `TOSS_CATCH_PARAMS`) as a
GIF: a side view (toss height is visible) next to a view from above the palm. Real-time playback:
each frame is `frame_stride` physics steps, duration = frame_stride * timestep.

Usage:
    python3 demo_stage7_toss_catch.py --seconds 8 --record media/stage7_toss_catch.gif
"""
import argparse
from pathlib import Path

import mujoco as mj
import numpy as np
from PIL import Image

from hand_env import TOSS_CATCH_PARAMS, TOSS_SCENE, TOSS_SPAWN, TossCatchController, place_balls


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=12.0)
    ap.add_argument("--record", type=str, default=None)
    ap.add_argument("--frame-stride", type=int, default=25)
    args = ap.parse_args()

    m = mj.MjModel.from_xml_path(TOSS_SCENE)
    d = mj.MjData(m)
    dt = m.opt.timestep
    place_balls(m, d, TOSS_SPAWN)
    ctl = TossCatchController(**TOSS_CATCH_PARAMS)
    d.ctrl[:] = ctl.cup
    for _ in range(int(0.5 / dt)):
        mj.mj_step(m, d)

    renderer = mj.Renderer(m, height=260, width=320)
    cams = []
    for elevation, azimuth, lookat in ((-5, 90, (0.015, 0.0, 0.07)), (-60, 200, (0.02, 0.0, 0.04))):
        cam = mj.MjvCamera()
        cam.type = mj.mjtCamera.mjCAMERA_FREE
        cam.lookat[:] = lookat
        cam.distance = 0.32
        cam.elevation, cam.azimuth = elevation, azimuth
        cams.append(cam)

    frames = []
    lo, hi = m.actuator_ctrlrange[:, 0], m.actuator_ctrlrange[:, 1]
    for k in range(int(args.seconds / dt)):
        d.ctrl[:] = np.clip(ctl.step(d, dt), lo, hi)
        mj.mj_step(m, d)
        if k % args.frame_stride == 0:
            views = []
            for cam in cams:
                renderer.update_scene(d, camera=cam)
                views.append(renderer.render().copy())
            frames.append(np.concatenate(views, axis=1))
        if ctl.failed:
            break
    renderer.close()
    print(f"{ctl.cycles} toss+catch cycles in {args.seconds}s, toss heights (cm): "
          f"{np.round(np.array(ctl.apexes) * 100, 1).tolist()}" + (f", FAILED: {ctl.failed}" if ctl.failed else ""))

    if args.record:
        Path(args.record).parent.mkdir(parents=True, exist_ok=True)
        imgs = [Image.fromarray(f) for f in frames]
        duration_ms = round(args.frame_stride * dt * 1000)
        imgs[0].save(args.record, save_all=True, append_images=imgs[1:], duration=duration_ms, loop=0)
        print(f"Saved {len(frames)} frames to {args.record} at {duration_ms}ms/frame")


if __name__ == "__main__":
    main()
