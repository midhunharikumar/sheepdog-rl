"""CEM -> Behavior Cloning warmup.

1. Use the CEM planner to solve several seeds, recording the (obs, action) pairs
   of the executed plans -> expert demonstrations.
2. Fit observation-normalization stats on those demos (so the policy expects the
   same normalized inputs PPO will feed it via VecNormalize).
3. Behavior-clone an SB3 PPO policy on the demos (supervised regression of the
   policy's mean action onto the expert action).
4. Save the model + normalizer. Fine-tune with PPO:
       python examples/train_sb3.py --init-from bc_sheepdog

This bootstraps the agent past the hard exploration: it starts out already
herding, and RL only has to polish.

    python examples/cem_to_bc.py --seeds 0 1 2 3 4 5 --iters 8 --epochs 40

CEM is slow, so collecting demos for many seeds takes a while; start small.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sheepdog_env import SheepdogHerdingEnv
from cem_solver import CEMPlanner


def collect_demos(seeds, source, cem_kwargs, min_penned):
    """Roll out an expert (CEM or the heuristic) per seed; keep (obs, action)
    pairs only from rollouts that pen at least ``min_penned`` sheep, so the
    policy is cloned on *good* demonstrations."""
    obs_buf, act_buf, kept = [], [], []
    for s in seeds:
        env = SheepdogHerdingEnv()
        obs, info = env.reset(seed=s)

        if source == "cem":
            base = SheepdogHerdingEnv()
            base.reset(seed=s)
            plan, _, _ = CEMPlanner(seed=s, **cem_kwargs).plan(base, verbose=False)
            plan_iter = iter(plan)
            policy = lambda o: next(plan_iter)
        else:  # heuristic
            from heuristic_agent import HeuristicShepherd
            agent = HeuristicShepherd(env)
            policy = agent.act

        ep_obs, ep_act = [], []
        for _ in range(env.cfg.max_steps):
            try:
                a = policy(obs)
            except StopIteration:
                break
            ep_obs.append(np.asarray(obs, dtype=np.float32))
            ep_act.append(np.asarray(a, dtype=np.float32))
            obs, r, term, trunc, info = env.step(a)
            if term or trunc:
                break

        if info["n_penned"] >= min_penned:
            obs_buf.extend(ep_obs)
            act_buf.extend(ep_act)
            kept.append(info["n_penned"])
            tag = "kept"
        else:
            tag = f"dropped (<{min_penned})"
        print(f"  seed {s} [{source}]: penned {info['n_penned']}/{env.n_active}  "
              f"{len(ep_act)} steps -> {tag}", flush=True)
    return np.array(obs_buf), np.array(act_buf), kept


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3])
    ap.add_argument("--source", choices=["cem", "heuristic"], default="cem",
                    help="demo source: CEM planner (strong but slow) or the heuristic (fast)")
    ap.add_argument("--min-penned", type=int, default=1,
                    help="only clone rollouts that pen at least this many sheep")
    ap.add_argument("--save", default="bc_sheepdog")
    # CEM budget (per seed)
    ap.add_argument("--segments", type=int, default=22)
    ap.add_argument("--samples", type=int, default=40)
    ap.add_argument("--elite", type=int, default=8)
    ap.add_argument("--iters", type=int, default=6)
    ap.add_argument("--plan-horizon", type=int, default=350)
    # BC training
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()

    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    cem_kwargs = dict(n_segments=args.segments, n_samples=args.samples,
                      n_elite=args.elite, n_iters=args.iters,
                      plan_horizon=args.plan_horizon)
    print(f"Collecting {args.source} demos for seeds {args.seeds} ...")
    X, Y, penned = collect_demos(args.seeds, args.source, cem_kwargs, args.min_penned)
    if len(X) == 0:
        sys.exit("no demos passed the --min-penned filter; lower it or improve the expert.")
    print(f"kept {len(penned)} rollouts, {len(X)} (obs, action) pairs  "
          f"(mean penned {np.mean(penned):.1f}/40)")

    # Observation normalization stats from the demos.
    mean = X.mean(axis=0)
    var = X.var(axis=0) + 1e-8
    Xn = np.clip((X - mean) / np.sqrt(var), -10.0, 10.0).astype(np.float32)

    # Build a PPO model whose VecNormalize carries these obs stats, so eval/play
    # /fine-tune all use the same normalization the policy was cloned on.
    venv = VecNormalize(DummyVecEnv([lambda: SheepdogHerdingEnv()]),
                        norm_obs=True, norm_reward=True, clip_obs=10.0, gamma=0.999)
    venv.obs_rms.mean[:] = mean
    venv.obs_rms.var[:] = var
    venv.obs_rms.count = float(len(X))

    model = PPO("MlpPolicy", venv, policy_kwargs=dict(log_std_init=-1.0),
                ent_coef=0.003, verbose=0)

    # --- behavior cloning: regress the policy mean action onto expert actions --
    device = model.device
    Xt = torch.as_tensor(Xn, device=device)
    Yt = torch.as_tensor(Y, device=device)
    opt = torch.optim.Adam(model.policy.parameters(), lr=args.lr)
    n = len(Xt)
    print(f"behavior cloning: {args.epochs} epochs over {n} samples ...")
    for epoch in range(args.epochs):
        perm = torch.randperm(n, device=device)
        tot = 0.0
        for i in range(0, n, args.batch_size):
            idx = perm[i:i + args.batch_size]
            dist = model.policy.get_distribution(Xt[idx])
            pred = dist.distribution.mean            # policy's mean action
            loss = ((pred - Yt[idx]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)
        if epoch % 5 == 0 or epoch == args.epochs - 1:
            print(f"  epoch {epoch:3d}  bc_mse={tot / n:.4f}", flush=True)

    model.save(args.save)
    venv.save(f"{args.save}_vecnorm.pkl")
    print(f"\nsaved BC model -> {args.save}.zip  and normalizer -> {args.save}_vecnorm.pkl")

    # --- quick eval of the cloned policy --------------------------------------
    def norm(o):
        return np.clip((o - mean) / np.sqrt(var), -10, 10).astype(np.float32)

    res = []
    for s in [100, 101, 102, 103]:
        env = SheepdogHerdingEnv()
        obs, info = env.reset(seed=s)
        while True:
            a, _ = model.predict(norm(obs), deterministic=True)
            obs, r, term, trunc, info = env.step(a)
            if term or trunc:
                break
        res.append(info["n_penned"])
    print(f"cloned policy eval (held-out seeds): penned {res}  mean {np.mean(res):.1f}/40")


if __name__ == "__main__":
    main()
