"""Stage 6 RL feasibility check: does a trained BaodingEnv policy sustain real orbiting of the
two-ball pair, where both scripted primitives (eval_stage6_baoding_balls.py) only managed a
small bounded repositioning before flatlining?

Runs a long deterministic rollout (ignoring the env's own 6s training-episode truncation -- a
sample-efficiency choice for PPO, not a real episode boundary; this eval only stops early on an
actual failure) and reports cumulative unwrapped orbit (sum of hand_env.ball_pair_angle_rad's
wrap-deltas per control step -- NOT a one-shot start-to-now angle, which can't tell a real
multi-turn orbit from "wiggled and came back"), alongside per-ball retention, fingertip contact
counts, and separation -- directly comparable to eval_stage6_baoding_balls.py's own numbers.

Usage:
    python3 eval_stage6_baoding_rl.py --run-name baoding01 --seconds 60
"""
import argparse
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from rl_env_baoding import CONTROL_DT, BaodingEnv

HERE = Path(__file__).parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", type=str, required=True)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--base-seed", type=int, default=1000)
    ap.add_argument("--seconds", type=float, default=60.0)
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
        env = BaodingEnv()
        vec_env = DummyVecEnv([lambda: env])
        norm_env = VecNormalize.load(str(run_dir / "checkpoints" / vecnorm_name), vec_env)
        norm_env.training = False
        norm_env.norm_reward = False

        obs, _ = env.reset(seed=seed)
        cum_orbit_rad = 0.0
        retained_all = True
        info = {}
        t_dropped = None
        for t in range(n_control_steps):
            obs_norm = norm_env.normalize_obs(obs)
            action, _ = policy.predict(obs_norm, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            cum_orbit_rad += info["orbit_step_rad"]
            if terminated:
                retained_all = False
                t_dropped = t
                break

        cum_deg = np.degrees(cum_orbit_rad)
        print(f"trial {i} seed={seed}: cumulative_orbit={cum_deg:.1f}deg "
              f"retained={retained_all}" +
              (f" (failed at t={t_dropped * CONTROL_DT:.1f}s)" if t_dropped is not None else
               f" (dist1={info['dist1']:.3f} dist2={info['dist2']:.3f} sep={info['separation']:.3f} "
               f"contacts={info['n_contacts_1']}/{info['n_contacts_2']})"))
        print(f"SUMMARY_PARSEABLE seed={seed} cumulative_orbit_deg={cum_deg:.2f} "
              f"retained={retained_all}")


if __name__ == "__main__":
    main()
