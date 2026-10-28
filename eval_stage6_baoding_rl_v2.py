"""Stage 6, second RL design iteration: does a trained BaodingEnvV2 policy (larger balls, grace
periods on transient contact/rate violations, milestone bonus for new rotation thresholds -- see
rl_env_baoding_v2.py's own docstring) sustain real, repeated circulation, where `baoding02`
(rl_env_baoding.py) only managed one partial shift that then decayed?

Runs a long deterministic rollout and reports cumulative unwrapped orbit, number of milestone
bonuses hit (a direct signal of "did it go around more than once"), and failure diagnostics.

Usage:
    python3 eval_stage6_baoding_rl_v2.py --run-name baoding_v2_01 --seconds 60
"""
import argparse
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from rl_env_baoding_v2 import CONTROL_DT, MILESTONE_RAD, BaodingEnvV2

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
        env = BaodingEnvV2()
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

        cum_deg = np.degrees(info["cum_orbit_rad"])
        print(f"trial {i} seed={seed}: cumulative_orbit={cum_deg:.1f}deg "
              f"({n_milestones} milestones x {np.degrees(MILESTONE_RAD):.0f}deg) "
              f"retained={retained_all}" +
              (f" (failed at t={t_dropped * CONTROL_DT:.1f}s)" if t_dropped is not None else
               f" (dist1={info['dist1']:.3f} dist2={info['dist2']:.3f} sep={info['separation']:.3f} "
               f"contacts={info['n_contacts_1']}/{info['n_contacts_2']})"))
        print(f"SUMMARY_PARSEABLE seed={seed} cumulative_orbit_deg={cum_deg:.2f} "
              f"milestones={n_milestones} retained={retained_all}")


if __name__ == "__main__":
    main()
