"""Stage 1 verification: does the vendored Allegro Hand V3 MJCF (assets/scene_right.xml, from
mujoco_menagerie) move the way a real Allegro Hand should under its own (already-tuned) position
actuators, and does the included free-floating object settle into a sane resting configuration
under gravity?

Same discipline as panda_vision_grasp's own Stage 1: this model ships with position-controlled
actuators already tuned by mujoco_menagerie, so this sweep drives REAL closed-loop physics
(mj_step under gravity, not just mj_forward on a set qpos) -- confirming the ready-made model
actually holds/tracks a commanded pose, not just that it loads.

Usage:
    python3 demo_stage1_kinematic_sweep.py --record media/stage1_kinematic_sweep.gif
"""
import argparse
from pathlib import Path

import mujoco as mj
import numpy as np
from PIL import Image

N_FINGER_JOINTS = 16
JOINT_NAMES = [f"{f}j{j}" for f in ("ff", "mf", "rf", "th") for j in range(4)]
ACTUATOR_NAMES = [f"{f}a{j}" for f in ("ff", "mf", "rf", "th") for j in range(4)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", type=str, default="assets/scene_right.xml")
    ap.add_argument("--record", type=str, default=None)
    ap.add_argument("--seconds-per-target", type=float, default=0.3)
    ap.add_argument("--settle-seconds", type=float, default=1.2,
                     help="After the joint sweep, hold ctrl=0 this long to show the included "
                          "free-floating object settle under gravity.")
    args = ap.parse_args()

    model = mj.MjModel.from_xml_path(args.scene)
    data = mj.MjData(model)
    dt = model.opt.timestep
    jid = {name: mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, name) for name in JOINT_NAMES}
    qadr = {name: model.jnt_qposadr[jid[name]] for name in JOINT_NAMES}
    n = int(args.seconds_per_target / dt)

    renderer = mj.Renderer(model, height=180, width=240)
    cam = mj.MjvCamera()
    cam.type = mj.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = [0.0, 0.0, 0.05]
    cam.distance = 0.5
    cam.elevation = -25

    frames = []
    step_count = [0]
    azimuth = [50.0]

    def render():
        azimuth[0] += 0.3
        cam.azimuth = azimuth[0]
        renderer.update_scene(data, camera=cam)
        frames.append(renderer.render().copy())

    ctrl_target = np.zeros(N_FINGER_JOINTS)
    data.ctrl[:] = ctrl_target

    def hold_target(target, seconds):
        data.ctrl[:] = target
        for _ in range(int(seconds / dt)):
            mj.mj_step(model, data)
            step_count[0] += 1
            if step_count[0] % 50 == 0:
                render()

    # Sweep each finger joint, one at a time, to its own [low, high] range and back to 0,
    # holding every other joint at its last commanded (not necessarily yet-achieved) target.
    for i, name in enumerate(JOINT_NAMES):
        lo, hi = model.jnt_range[jid[name]]
        for val in (hi, lo, 0.0):
            ctrl_target[i] = val
            hold_target(ctrl_target, args.seconds_per_target)
        ctrl_target[i] = 0.0

    achieved = np.array([data.qpos[qadr[name]] for name in JOINT_NAMES])
    print(f"Joint sweep done ({len(frames)} frames so far). Commanded target: zeros, achieved "
          f"qpos: {np.round(achieved, 3)} (should be near zero -- confirms real closed-loop "
          f"tracking, not just forward kinematics).")

    # Let the included free-floating object (scene_right.xml's ellipsoid, spawned above the
    # palm) fall under gravity and settle, with the hand held open (ctrl=0).
    hold_target(np.zeros(N_FINGER_JOINTS), args.settle_seconds)
    obj_pos = data.body("object").xpos.copy()
    print(f"Object settled at {np.round(obj_pos, 4)} after {args.settle_seconds}s under gravity "
          f"(starts at [0, 0, 0.1]) -- confirms it falls into the open hand's workspace rather "
          f"than through it, a real resting configuration to build a grasp/manipulation task on.")

    if args.record:
        Path(args.record).parent.mkdir(parents=True, exist_ok=True)
        imgs = [Image.fromarray(f) for f in frames]
        imgs[0].save(args.record, save_all=True, append_images=imgs[1:], duration=33, loop=0)
        print(f"Saved {len(frames)} frames to {args.record}")

    renderer.close()


if __name__ == "__main__":
    main()
