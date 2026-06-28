"""Inspect what a trained policy actually predicts (is it frozen? does it bark?).

    python examples/inspect_policy.py --model ppo_sheepdog

Rolls out a few episodes with the model and reports the action statistics that
matter for the "no movement / no exploration" diagnosis:
  * mean and spread of the move vector, and mean movement magnitude (0 = frozen)
  * how often it barks (action[2] > 0)
  * the policy's own action std (log_std) — i.e. how much exploration noise it
    still has left
  * how far the dog actually travels over an episode
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
    ap.add_argument("--model", default="ppo_sheepdog")
    ap.add_argument("--obs", choices=["vector", "pixel"], default="vector")
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--wolves", action="store_true")
    ap.add_argument("--vecnorm", default=None,
                    help="path to VecNormalize stats (auto-detected if omitted)")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    args = ap.parse_args()

    from stable_baselines3 import PPO
    from _policy_utils import load_obs_normalizer
    model = PPO.load(args.model, device=args.device)
    normalize, msg = load_obs_normalizer(args.model, args.vecnorm)
    print(msg)

    # Policy action std (state-independent DiagGaussian for PPO MlpPolicy).
    try:
        import torch
        log_std = model.policy.log_std.detach().cpu().numpy()
        print(f"policy action std (exp(log_std)) = {np.exp(log_std).round(4)}  "
              f"[a0, a1, bark]   (near 0 => no exploration noise)")
    except Exception as e:
        print(f"(could not read log_std: {e})")

    env = SheepdogHerdingEnv(obs_mode=args.obs, enable_wolves=args.wolves)
    all_actions, dog_paths = [], []

    for ep in range(args.episodes):
        obs, info = env.reset(seed=1000 + ep)
        start = env.dog_pos.copy()
        path_len = 0.0
        prev = start.copy()
        det_actions = []
        while True:
            action, _ = model.predict(normalize(obs), deterministic=True)
            det_actions.append(np.asarray(action, dtype=np.float32))
            obs, r, term, trunc, info = env.step(action)
            path_len += float(np.linalg.norm(env.dog_pos - prev))
            prev = env.dog_pos.copy()
            if term or trunc:
                break
        a = np.array(det_actions)
        all_actions.append(a)
        dog_paths.append(path_len)
        print(f"ep {ep}: penned={info['n_penned']:2d}/{env.cfg.n_sheep}  "
              f"dog_travelled={path_len:6.1f} world-units  steps={len(a)}")

    A = np.concatenate(all_actions, axis=0)
    move = A[:, :2]
    mag = np.linalg.norm(np.clip(move, -1, 1), axis=1)
    bark_rate = float((A[:, 2] > 0).mean())
    print("\n--- deterministic action summary (all steps) ---")
    print(f"  move mean   : [{move[:,0].mean():+.3f}, {move[:,1].mean():+.3f}]")
    print(f"  move std    : [{move[:,0].std():.3f}, {move[:,1].std():.3f}]")
    print(f"  |move| mean : {mag.mean():.3f}   (0 = dog frozen, 1 = full speed)")
    print(f"  |move| <0.05 on {(mag < 0.05).mean():.0%} of steps   (frozen fraction)")
    print(f"  bark action : a[2] mean {A[:,2].mean():+.3f},  fires (>0) {bark_rate:.0%} of steps")
    env.close()


if __name__ == "__main__":
    main()
