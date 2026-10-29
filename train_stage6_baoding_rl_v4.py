"""Trains a PPO policy on BaodingEnvV4 (rl_env_baoding_v4.py): BaodingEnvV3's per-ball
same-direction-matched orbit reward, but with the static starting grasp actually reaching thumb
contact (hold_seconds 0.3->1.0) and a per-digit idle-timeout termination + engagement bonus so the
policy can't quietly abandon the thumb (or any digit) the way `baoding_v3_05` did -- see
rl_env_baoding_v4.py's own docstring.

Usage:
    python3 train_stage6_baoding_rl_v4.py --run-name baoding_v4_01 --timesteps 4000000 --n-envs 16
"""
import argparse
import json
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, VecNormalize

from rl_env_baoding_v4 import BaodingEnvV4

HERE = Path(__file__).parent


def linear_schedule(initial):
    def fn(progress_remaining):
        return progress_remaining * initial
    return fn


def make_env(seed):
    def _init():
        env = BaodingEnvV4()
        env.reset(seed=seed)
        return env
    return _init


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", type=str, required=True)
    ap.add_argument("--timesteps", type=int, default=4_000_000)
    ap.add_argument("--n-envs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--n-steps", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=2048)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--ent-coef", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", type=str, default=None)
    ap.add_argument("--resume-vecnormalize", type=str, default=None)
    ap.add_argument("--checkpoint-freq", type=int, default=400_000)
    args = ap.parse_args()

    run_dir = HERE / "runs" / args.run_name
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)
    (run_dir / "args.json").write_text(json.dumps(vars(args), indent=2))

    env_fns = [make_env(args.seed + i) for i in range(args.n_envs)]
    env = SubprocVecEnv(env_fns)
    env = VecMonitor(env, info_keywords=("success",))
    if args.resume_vecnormalize:
        env = VecNormalize.load(args.resume_vecnormalize, env)
        env.training = True
    else:
        env = VecNormalize(env, norm_obs=True, norm_reward=True, gamma=args.gamma)

    n_steps = args.n_steps
    batch_size = args.batch_size
    if (n_steps * args.n_envs) % batch_size != 0:
        batch_size = n_steps * args.n_envs
        print(f"n_steps*n_envs not divisible by batch_size, using batch_size={batch_size}")

    if args.resume:
        model = PPO.load(args.resume, env=env, device="cpu", verbose=1,
                          learning_rate=linear_schedule(args.lr))
        print(f"resumed from {args.resume}")
    else:
        model = PPO(
            "MlpPolicy", env,
            learning_rate=linear_schedule(args.lr),
            n_steps=n_steps,
            batch_size=batch_size,
            gamma=args.gamma,
            ent_coef=args.ent_coef,
            device="cpu",
            tensorboard_log=str(run_dir / "logs"),
            seed=args.seed,
            verbose=1,
        )

    save_freq = max(args.checkpoint_freq // args.n_envs, 1)
    checkpoint_cb = CheckpointCallback(
        save_freq=save_freq,
        save_path=str(run_dir / "checkpoints"),
        name_prefix="ppo",
        save_vecnormalize=True,
    )

    model.learn(total_timesteps=args.timesteps, callback=CallbackList([checkpoint_cb]))
    model.save(run_dir / "checkpoints" / "final")
    env.save(str(run_dir / "checkpoints" / "vecnormalize_final.pkl"))
    print(f"done, saved to {run_dir}/checkpoints/")


if __name__ == "__main__":
    main()
