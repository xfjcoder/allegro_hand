"""Renders a Stage 6 RL-orbiting rollout: the trained BaodingEnv policy's own chosen finger
motions (not a scripted primitive) holding and (if it worked) orbiting the two-ball pair. Same
"GIF for every notable attempt" convention as every other stage in this project family. Uses the
closer, steeper camera angle fixed for demo_stage6_baoding_balls.py so the balls aren't hidden
behind the fingers.

Usage:
    python3 demo_stage6_baoding_rl.py --run-name baoding01 --seconds 20 --record media/stage6_baoding_rl.gif
"""
import argparse
from pathlib import Path

import mujoco as mj
import numpy as np
from PIL import Image
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from rl_env_baoding import CONTROL_DT, BaodingEnv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", type=str, required=True)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--seed", type=int, default=2000)
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--record", type=str, default=None)
    ap.add_argument("--frame-stride", type=int, default=5)
    args = ap.parse_args()

    run_dir = Path(__file__).parent / "runs" / args.run_name
    if args.steps is not None:
        model_stem = f"ppo_{args.steps}_steps"
        vecnorm_name = f"ppo_vecnormalize_{args.steps}_steps.pkl"
    else:
        model_stem = "final"
        vecnorm_name = "vecnormalize_final.pkl"
    policy = PPO.load(run_dir / "checkpoints" / model_stem)

    env = BaodingEnv()
    vec_env = DummyVecEnv([lambda: env])
    norm_env = VecNormalize.load(str(run_dir / "checkpoints" / vecnorm_name), vec_env)
    norm_env.training = False
    norm_env.norm_reward = False

    renderer = mj.Renderer(env.model, height=240, width=320)
    cam = mj.MjvCamera()
    cam.type = mj.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = [0.0, 0.0, 0.035]
    cam.distance = 0.2
    cam.elevation = -50
    cam.azimuth = 200

    obs, _ = env.reset(seed=args.seed)
    n_control_steps = int(args.seconds / CONTROL_DT)

    frames = []
    cum_orbit_rad = 0.0
    retained = True
    for t in range(n_control_steps):
        obs_norm = norm_env.normalize_obs(obs)
        action, _ = policy.predict(obs_norm, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        cum_orbit_rad += info["orbit_step_rad"]
        if t % args.frame_stride == 0:
            renderer.update_scene(env.data, camera=cam)
            frames.append(renderer.render().copy())
        if terminated:
            retained = False
            break

    cum_deg = np.degrees(cum_orbit_rad)
    print(f"Cumulative orbit: {cum_deg:.1f} degrees over {args.seconds}s (stopped early at "
          f"t={t} control steps due to failure)" if not retained else
          f"Cumulative orbit: {cum_deg:.1f} degrees over {args.seconds}s, retained throughout. "
          f"Rendered {len(frames)} frames.")

    if args.record:
        Path(args.record).parent.mkdir(parents=True, exist_ok=True)
        imgs = [Image.fromarray(f) for f in frames]
        imgs[0].save(args.record, save_all=True, append_images=imgs[1:], duration=40, loop=0)
        print(f"Saved {len(frames)} frames to {args.record}")

    renderer.close()


if __name__ == "__main__":
    main()
