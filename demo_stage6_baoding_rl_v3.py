"""Renders a Stage 6 BaodingEnvV3 rollout (crossover grasp start, per-ball same-direction-matched
orbit reward -- see rl_env_baoding_v3.py). Same "GIF for every notable attempt" convention as
every other stage, and the same real-time-playback discipline as demo_stage6_baoding_rl_v2.py
(duration-ms defaults to frame_stride*CONTROL_DT*1000, not a hardcoded value).

Usage:
    python3 demo_stage6_baoding_rl_v3.py --run-name baoding_v3_01 --seconds 20 --record media/stage6_baoding_rl_v3.gif
"""
import argparse
from pathlib import Path

import mujoco as mj
import numpy as np
from PIL import Image
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from rl_env_baoding_v3 import CONTROL_DT, BaodingEnvV3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", type=str, required=True)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--seed", type=int, default=2000)
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--record", type=str, default=None)
    ap.add_argument("--frame-stride", type=int, default=5)
    ap.add_argument("--duration-ms", type=int, default=None)
    args = ap.parse_args()
    duration_ms = args.duration_ms
    if duration_ms is None:
        duration_ms = round(args.frame_stride * CONTROL_DT * 1000)

    run_dir = Path(__file__).parent / "runs" / args.run_name
    if args.steps is not None:
        model_stem = f"ppo_{args.steps}_steps"
        vecnorm_name = f"ppo_vecnormalize_{args.steps}_steps.pkl"
    else:
        model_stem = "final"
        vecnorm_name = "vecnormalize_final.pkl"
    policy = PPO.load(run_dir / "checkpoints" / model_stem)

    env = BaodingEnvV3()
    vec_env = DummyVecEnv([lambda: env])
    norm_env = VecNormalize.load(str(run_dir / "checkpoints" / vecnorm_name), vec_env)
    norm_env.training = False
    norm_env.norm_reward = False

    renderer = mj.Renderer(env.model, height=240, width=320)
    cam = mj.MjvCamera()
    cam.type = mj.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = [0.0, 0.0, 0.035]
    cam.distance = 0.18
    cam.elevation = -75
    cam.azimuth = 200

    obs, _ = env.reset(seed=args.seed)
    n_control_steps = int(args.seconds / CONTROL_DT)

    frames = []
    retained = True
    info = {}
    t = 0
    for t in range(n_control_steps):
        obs_norm = norm_env.normalize_obs(obs)
        action, _ = policy.predict(obs_norm, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        if t % args.frame_stride == 0:
            renderer.update_scene(env.data, camera=cam)
            frames.append(renderer.render().copy())
        if terminated:
            retained = False
            break

    cum1_deg = np.degrees(info["cum1_rad"])
    cum2_deg = np.degrees(info["cum2_rad"])
    print(f"ball1_cum={cum1_deg:.1f}deg ball2_cum={cum2_deg:.1f}deg over {args.seconds}s" +
          (" (stopped early due to failure)" if not retained else ", retained throughout") +
          f". Rendered {len(frames)} frames.")

    if args.record:
        Path(args.record).parent.mkdir(parents=True, exist_ok=True)
        imgs = [Image.fromarray(f) for f in frames]
        imgs[0].save(args.record, save_all=True, append_images=imgs[1:], duration=duration_ms,
                     loop=0)
        print(f"Saved {len(frames)} frames to {args.record} at {duration_ms}ms/frame")

    renderer.close()


if __name__ == "__main__":
    main()
