"""Renders a Stage 5 RL-gaiting rollout: the trained GaitingEnv policy's own chosen finger
motions (not a scripted twist/regrasp primitive) holding and reorienting the object. Same "GIF
for every notable attempt" convention as every other stage in this project family.

Usage:
    python3 demo_stage5_gaiting_rl.py --run-name gait01 --seconds 20 --record media/stage5_gaiting_rl.gif
"""
import argparse
from pathlib import Path

import mujoco as mj
import numpy as np
from PIL import Image
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from rl_env import CONTROL_DT, GaitingEnv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", type=str, required=True)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--seed", type=int, default=2000)
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--record", type=str, default=None)
    ap.add_argument("--frame-stride", type=int, default=5,
                     help="In control-step units (each control step = 10 physics steps) -- a "
                          "stride of 5 control steps = 50 physics steps, comparable visual "
                          "density to Stage 3/4's frame-stride=150 physics-step convention.")
    args = ap.parse_args()

    run_dir = Path(__file__).parent / "runs" / args.run_name
    if args.steps is not None:
        model_stem = f"ppo_{args.steps}_steps"
        vecnorm_name = f"ppo_vecnormalize_{args.steps}_steps.pkl"
    else:
        model_stem = "final"
        vecnorm_name = "vecnormalize_final.pkl"
    policy = PPO.load(run_dir / "checkpoints" / model_stem)

    env = GaitingEnv()
    vec_env = DummyVecEnv([lambda: env])
    norm_env = VecNormalize.load(str(run_dir / "checkpoints" / vecnorm_name), vec_env)
    norm_env.training = False
    norm_env.norm_reward = False

    renderer = mj.Renderer(env.model, height=240, width=320)
    cam = mj.MjvCamera()
    cam.type = mj.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = [0.0, 0.0, 0.03]
    cam.distance = 0.35
    cam.elevation = -20
    cam.azimuth = 120

    obs, _ = env.reset(seed=args.seed)
    n_control_steps = int(args.seconds / CONTROL_DT)

    frames = []
    cum_rotation_rad = 0.0
    retained = True
    for t in range(n_control_steps):
        obs_norm = norm_env.normalize_obs(obs)
        action, _ = policy.predict(obs_norm, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        cum_rotation_rad += info["rot_step_rad"]
        if t % args.frame_stride == 0:
            renderer.update_scene(env.data, camera=cam)
            frames.append(renderer.render().copy())
        if terminated:
            retained = False
            break

    cum_deg = np.degrees(cum_rotation_rad)
    print(f"Cumulative rotation: {cum_deg:.1f} degrees over {args.seconds}s (stopped early at "
          f"t={t} control steps due to drop)" if not retained else
          f"Cumulative rotation: {cum_deg:.1f} degrees over {args.seconds}s, retained throughout. "
          f"Rendered {len(frames)} frames.")

    if args.record:
        Path(args.record).parent.mkdir(parents=True, exist_ok=True)
        imgs = [Image.fromarray(f) for f in frames]
        imgs[0].save(args.record, save_all=True, append_images=imgs[1:], duration=40, loop=0)
        print(f"Saved {len(frames)} frames to {args.record}")

    renderer.close()


if __name__ == "__main__":
    main()
