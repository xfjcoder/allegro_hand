"""Trains a PPO policy on GaitingEnv (rl_env.py): full privileged state in, 16-dim delta-joint-
target actions out, dense signed-rotation reward. Mirrors panda_vision_grasp/train.py's own PPO +
SubprocVecEnv + VecNormalize + VecMonitor conventions, minus the curriculum callback (no obvious
difficulty axis to schedule here -- single fixed start config, same as Stage 3/4's own eval).

Usage:
    python3 train_stage5_gaiting_rl.py --run-name gait01 --timesteps 2000000 --n-envs 16
"""
import argparse
import json
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, VecNormalize

from rl_env import GaitingEnv

HERE = Path(__file__).parent


def linear_schedule(initial):
    def fn(progress_remaining):
        return progress_remaining * initial
    return fn


def make_env(seed):
    def _init():
        env = GaitingEnv()
        env.reset(seed=seed)
        return env
    return _init


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", type=str, required=True)
    ap.add_argument("--timesteps", type=int, default=2_000_000)
    ap.add_argument("--n-envs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--n-steps", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--ent-coef", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", type=str, default=None)
    ap.add_argument("--resume-vecnormalize", type=str, default=None)
    ap.add_argument("--checkpoint-freq", type=int, default=None)
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

    if args.checkpoint_freq is not None:
        save_freq = max(args.checkpoint_freq // args.n_envs, 1)
    else:
        save_freq = max(args.timesteps // (args.n_envs * 10), 1)
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
