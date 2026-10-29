"""Stage 6, fifth RL design iteration: does a trained BaodingEnvV5 policy massage the 60mm balls with
the INSIDE of its fingers -- palmar medial/distal/tip contact on all 4 digits, no back-of-finger
contact, little finger-root contact -- while actually circulating them?

Reports per-ball cumulative orbit, matched (same-direction) progress, per-digit good-contact %,
and the fraction of steps with any back-of-finger / finger-root contact.

Usage:
    python3 eval_stage6_baoding_rl_v5.py --run-name baoding_v5_01 --seconds 90
"""
import argparse
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from hand_env import DIGIT_PREFIXES
from rl_env_baoding_v5 import CONTROL_DT, BaodingEnvV5

HERE = Path(__file__).parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", type=str, required=True)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--seconds", type=float, default=90.0)
    ap.add_argument("--print-every", type=float, default=0.0)
    args = ap.parse_args()

    ckpt = HERE / "runs" / args.run_name / "checkpoints"
    stem, vn = (("final", "vecnormalize_final.pkl") if args.steps is None else
                (f"ppo_{args.steps}_steps", f"ppo_vecnormalize_{args.steps}_steps.pkl"))
    policy = PPO.load(ckpt / stem)
    env = BaodingEnvV5()
    norm = VecNormalize.load(str(ckpt / vn), DummyVecEnv([lambda: env]))
    norm.training = False
    norm.norm_reward = False

    obs, _ = env.reset(seed=args.seed)
    good_steps = {p: 0 for p in DIGIT_PREFIXES}
    dorsal_steps = root_steps = milestones = 0
    t_fail = None
    info = {}
    n = int(args.seconds / CONTROL_DT)
    every = int(args.print_every / CONTROL_DT) if args.print_every else 0
    for t in range(n):
        action, _ = policy.predict(norm.normalize_obs(obs), deterministic=True)
        obs, _, terminated, _, info = env.step(action)
        for p in info["good_digits"]:
            good_steps[p] += 1
        dorsal_steps += info["n_dorsal"] > 0
        root_steps += info["n_root"] > 0
        milestones += info["milestone_hit"]
        if every and t % every == 0:
            print(f"t={t * CONTROL_DT:5.1f}s cum1={np.degrees(info['cum1_rad']):6.1f} "
                  f"cum2={np.degrees(info['cum2_rad']):6.1f} matched={np.degrees(info['matched_cum_rad']):6.1f} "
                  f"r1={info['radial1']:.3f} r2={info['radial2']:.3f} sep={info['separation']:.3f} "
                  f"good={info['good_digits']} dorsal={info['n_dorsal']} root={info['n_root']} "
                  f"moving={info['actively_moving']}")
        if terminated:
            t_fail = t
            break
    steps = (t_fail if t_fail is not None else n - 1) + 1
    print(f"{args.run_name}: ball1_cum={np.degrees(info['cum1_rad']):.1f}deg "
          f"ball2_cum={np.degrees(info['cum2_rad']):.1f}deg matched={np.degrees(info['matched_cum_rad']):.1f}deg "
          f"milestones={milestones} " +
          (f"FAILED at t={t_fail * CONTROL_DT:.1f}s" if t_fail is not None else f"retained {args.seconds:.0f}s"))
    print("  good (palmar medial/distal/tip) contact, % of steps: " +
          " ".join(f"{p[:2]}={good_steps[p] / steps * 100:.0f}%" for p in DIGIT_PREFIXES))
    print(f"  any back-of-finger contact: {dorsal_steps / steps * 100:.0f}% of steps; "
          f"any finger-root contact: {root_steps / steps * 100:.0f}% of steps")


if __name__ == "__main__":
    main()
