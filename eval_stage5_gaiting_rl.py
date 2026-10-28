"""Stage 5 feasibility check: does a trained GaitingEnv policy sustain rotation past Stage 3/4's
scripted ~23-25 degree ceiling, over the SAME ~100s sim-time budget those stages used (50 twist
cycles * 2s/cycle)?

Runs a long deterministic rollout (ignoring the env's own 6s training-episode truncation --
that's just a sample-efficiency choice for PPO, not a real episode boundary; this eval only stops
early on an actual drop) and reports cumulative unwrapped rotation (sum of
hand_env.twist_angle_rad per control step -- NOT quat_angle_deg from start, which saturates at
180 degrees and can't detect a genuine multi-turn spin) alongside retention, directly comparable
to eval_stage3/4's own numbers.

Usage:
    python3 eval_stage5_gaiting_rl.py --run-name gait01 --seconds 100
"""
import argparse
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from rl_env import CONTROL_DT, GaitingEnv

HERE = Path(__file__).parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", type=str, required=True)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--base-seed", type=int, default=1000)
    ap.add_argument("--seconds", type=float, default=100.0,
                     help="Matches eval_stage3/4's 50-cycle*2s=100s budget for a direct "
                          "rotation-ceiling comparison.")
    args = ap.parse_args()

    run_dir = HERE / "runs" / args.run_name
    if args.steps is not None:
        model_stem = f"ppo_{args.steps}_steps"
        vecnorm_name = f"ppo_vecnormalize_{args.steps}_steps.pkl"
    else:
        model_stem = "final"
        vecnorm_name = "vecnormalize_final.pkl"
    policy = PPO.load(run_dir / "checkpoints" / model_stem)

    n_control_steps = int(args.seconds / CONTROL_DT)

    for i in range(args.trials):
        seed = args.base_seed + i
        env = GaitingEnv()
        vec_env = DummyVecEnv([lambda: env])
        norm_env = VecNormalize.load(str(run_dir / "checkpoints" / vecnorm_name), vec_env)
        norm_env.training = False
        norm_env.norm_reward = False

        obs, _ = env.reset(seed=seed)
        cum_rotation_rad = 0.0
        retained_all = True
        dist = 0.0
        t_dropped = None
        for t in range(n_control_steps):
            obs_norm = norm_env.normalize_obs(obs)
            action, _ = policy.predict(obs_norm, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            cum_rotation_rad += info["rot_step_rad"]
            dist = info["dist"]
            if terminated:
                retained_all = False
                t_dropped = t
                break

        cum_deg = np.degrees(cum_rotation_rad)
        print(f"trial {i} seed={seed}: cumulative_rotation={cum_deg:.1f}deg "
              f"retained={retained_all}" +
              (f" (dropped at t={t_dropped} control steps)" if t_dropped is not None else
               f" (dist={dist:.3f})"))
        print(f"SUMMARY_PARSEABLE seed={seed} cumulative_rotation_deg={cum_deg:.2f} "
              f"retained={retained_all}")


if __name__ == "__main__":
    main()
