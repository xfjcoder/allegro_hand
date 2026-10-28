"""Stage 6, third RL design iteration: does a trained BaodingEnvV3 policy (crossover grasp start,
per-ball same-direction-matched orbit reward -- see rl_env_baoding_v3.py's own docstring) achieve
BALANCED circulation of both balls, where BaodingEnvV2's pair-vector reward could in principle be
satisfied by one ball wobbling alone?

Runs a long deterministic rollout and reports each ball's own cumulative orbit (not just the pair
metric), number of milestone bonuses hit, and failure diagnostics.

Usage:
    python3 eval_stage6_baoding_rl_v3.py --run-name baoding_v3_01 --seconds 60
"""
import argparse
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from rl_env_baoding_v3 import CONTROL_DT, MILESTONE_RAD, BaodingEnvV3

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
        env = BaodingEnvV3()
        vec_env = DummyVecEnv([lambda: env])
        norm_env = VecNormalize.load(str(run_dir / "checkpoints" / vecnorm_name), vec_env)
        norm_env.training = False
        norm_env.norm_reward = False

        obs, _ = env.reset(seed=seed)
        n_milestones = 0
        retained_all = True
        info = {}
        t_dropped = None
        for t in range(n_control_steps):
            obs_norm = norm_env.normalize_obs(obs)
            action, _ = policy.predict(obs_norm, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            if info["milestone_hit"]:
                n_milestones += 1
            if terminated:
                retained_all = False
                t_dropped = t
                break

        cum1_deg = np.degrees(info["cum1_rad"])
        cum2_deg = np.degrees(info["cum2_rad"])
        print(f"trial {i} seed={seed}: ball1_cum={cum1_deg:.1f}deg ball2_cum={cum2_deg:.1f}deg "
              f"({n_milestones} milestones x {np.degrees(MILESTONE_RAD):.0f}deg, keyed to the "
              f"LAGGING ball) retained={retained_all}" +
              (f" (failed at t={t_dropped * CONTROL_DT:.1f}s)" if t_dropped is not None else
               f" (dist1={info['dist1']:.3f} dist2={info['dist2']:.3f} sep={info['separation']:.3f} "
               f"contacts={info['n_contacts_1']}/{info['n_contacts_2']})"))
        print(f"SUMMARY_PARSEABLE seed={seed} ball1_cum_deg={cum1_deg:.2f} "
              f"ball2_cum_deg={cum2_deg:.2f} milestones={n_milestones} retained={retained_all}")


if __name__ == "__main__":
    main()
