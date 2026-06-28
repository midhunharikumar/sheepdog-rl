"""Evaluate a trained Stable-Baselines3 model on the Sheepdog env.

    # after training (examples/train_sb3.py saved ppo_sheepdog.zip):
    python examples/eval_sb3.py --model ppo_sheepdog --episodes 50

    python examples/eval_sb3.py --model models/<run_id>/model --obs pixel --wolves
    python examples/eval_sb3.py --model ppo_sheepdog --gif eval.gif   # record rollouts

Reports success rate (>= target fraction penned), mean sheep penned, mean
episode return and length. Use --stochastic to sample actions instead of using
the deterministic (mean) policy.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sheepdog_env import SheepdogHerdingEnv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    help="path to the saved model (.zip), with or without extension")
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--obs", choices=["vector", "pixel"], default="vector",
                    help="must match what the model was trained on")
    ap.add_argument("--wolves", action="store_true")
    ap.add_argument("--stochastic", action="store_true",
                    help="sample actions instead of using the deterministic policy")
    ap.add_argument("--seed", type=int, default=10_000,
                    help="base seed; episode k uses seed+k")
    ap.add_argument("--gif", default=None, help="optional path to save a rollout gif")
    ap.add_argument("--gif-episodes", type=int, default=4)
    ap.add_argument("--vecnorm", default=None,
                    help="path to VecNormalize stats (auto-detected if omitted)")
    ap.add_argument("--target-fraction", type=float, default=None,
                    help="success threshold (match what you trained with)")
    ap.add_argument("--gap-width", type=float, default=None,
                    help="pen opening width (match what you trained with)")
    ap.add_argument("--pen-frac", type=float, nargs=4, default=None,
                    metavar=("X0", "Y0", "X1", "Y1"),
                    help="pen rectangle fractions (match what you trained with)")
    ap.add_argument("--action-mode", choices=["polar", "cartesian"], default=None,
                    help="must match training")
    ap.add_argument("--n-sheep", type=int, default=None,
                    help="flock CAPACITY (the obs size). MUST equal what the model "
                    "trained with, else the policy input won't match. Auto-detected "
                    "from the model if omitted.")
    ap.add_argument("--n-active", type=int, default=None,
                    help="how many sheep actually spawn (<= --n-sheep). Raise this "
                    "to eval a denser flock without changing the obs size.")
    ap.add_argument("--n-bushes", type=int, default=0,
                    help="bush obstacles (must match training; changes the obs size)")
    ap.add_argument("--n-rocks", type=int, default=0,
                    help="rock obstacles (must match training; changes the obs size)")
    ap.add_argument("--dr-active", type=int, nargs=2, default=None, metavar=("MIN", "MAX"),
                    help="randomize active flock size per episode (robustness eval)")
    ap.add_argument("--dr-spawn", type=float, nargs=2, default=None, metavar=("MIN", "MAX"),
                    help="randomize spawn x-fraction per episode")
    ap.add_argument("--dr-gap", type=float, nargs=2, default=None, metavar=("MIN", "MAX"),
                    help="randomize entrance width per episode")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    args = ap.parse_args()

    from stable_baselines3 import PPO
    from _policy_utils import load_obs_normalizer
    model = PPO.load(args.model, device=args.device)
    normalize, msg = load_obs_normalizer(args.model, args.vecnorm)
    print(msg)

    # The policy input is fixed by training: obs_dim = 14 + n_sheep*5 + M*3.
    model_dim = model.observation_space.shape[0]
    M = args.n_bushes + args.n_rocks
    if args.n_sheep is not None:
        n_sheep = args.n_sheep
    elif M == 0:
        n_sheep = (model_dim - 14) // 5         # unambiguous only without obstacles
    else:
        sys.exit("with obstacles, pass --n-sheep to match the trained capacity")
    if args.n_active is not None and args.n_active > n_sheep:
        sys.exit(f"--n-active {args.n_active} exceeds capacity {n_sheep}")

    env_kwargs = {"n_sheep": n_sheep, "n_bushes": args.n_bushes, "n_rocks": args.n_rocks}
    if args.n_active is not None:
        env_kwargs["n_active"] = args.n_active
    if args.target_fraction is not None:
        env_kwargs["target_fraction"] = args.target_fraction
    if args.gap_width is not None:
        env_kwargs["pen_opening_width"] = args.gap_width
    if args.pen_frac is not None:
        env_kwargs["pen_frac"] = tuple(args.pen_frac)
    if args.action_mode is not None:
        env_kwargs["action_mode"] = args.action_mode
    if args.dr_active is not None:
        env_kwargs["dr_active_range"] = tuple(args.dr_active)
    if args.dr_spawn is not None:
        env_kwargs["dr_spawn_x_range"] = tuple(args.dr_spawn)
    if args.dr_gap is not None:
        env_kwargs["dr_gap_range"] = tuple(args.dr_gap)
    env = SheepdogHerdingEnv(obs_mode=args.obs, enable_wolves=args.wolves,
                             render_mode="rgb_array" if args.gif else None, **env_kwargs)
    env_dim = env.observation_space.shape[0]
    if env_dim != model_dim:
        sys.exit(f"obs mismatch: env={env_dim} vs model={model_dim}. Set "
                 f"--n-sheep/--n-bushes/--n-rocks to match what the model trained with "
                 f"(the policy input is fixed to {model_dim}).")

    penned, returns, lengths, wins = [], [], [], 0
    frames = []
    deterministic = not args.stochastic

    for ep in range(args.episodes):
        obs, info = env.reset(seed=args.seed + ep)
        ep_ret = 0.0
        steps = 0
        record = args.gif and ep < args.gif_episodes
        while True:
            action, _ = model.predict(normalize(obs), deterministic=deterministic)
            obs, r, term, trunc, info = env.step(action)
            ep_ret += r
            steps += 1
            if record and steps % 3 == 0:
                frames.append(env.render())
            if term or trunc:
                break
        penned.append(info["n_penned"])
        returns.append(ep_ret)
        lengths.append(steps)
        wins += int(info.get("is_success", False))

    n = args.episodes
    tgt = int(env.cfg.target_fraction * env.cfg.n_sheep)
    print(f"\nEvaluated {n} episodes  (policy={'stochastic' if args.stochastic else 'deterministic'})")
    print(f"  success rate : {wins}/{n} = {wins / n:.0%}   (>= {tgt}/{env.cfg.n_sheep} penned)")
    print(f"  penned       : mean {np.mean(penned):.1f} / {env.cfg.n_sheep}"
          f"   (min {min(penned)}, max {max(penned)})")
    print(f"  return       : mean {np.mean(returns):.1f}  +/- {np.std(returns):.1f}")
    print(f"  ep length    : mean {np.mean(lengths):.0f} steps")

    if args.gif and frames:
        try:
            import imageio
            imageio.mimsave(args.gif, frames, fps=30, loop=0)
            print(f"  saved {len(frames)} frames -> {args.gif}")
        except ImportError:
            print("  install imageio to save the gif: pip install imageio")

    env.close()


if __name__ == "__main__":
    main()
