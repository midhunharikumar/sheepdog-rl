"""CEM planner: evaluate candidate action *paths* before committing to one.

The Cross-Entropy Method (CEM) optimizes an open-loop action sequence for a
whole episode. Because the environment is its own simulator, we *clone* it
(``copy.deepcopy``) and roll candidate sequences forward to score them — i.e. we
evaluate paths before making them — then refine the sampling distribution toward
the best (elite) ones.

The action sequence is parameterized as ``n_segments`` piecewise-constant
actions (each held for several steps), which keeps the search dimension small.
All candidates in an iteration are evaluated from the *same* cloned state with
the *same* RNG (common random numbers), so their returns are directly comparable.

    python examples/cem_solver.py --seed 0 --gif cem.gif

Tune the search with --segments / --samples / --elite / --iters.
"""

from __future__ import annotations

import argparse
import copy
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sheepdog_env import SheepdogHerdingEnv


def rollout_return(base_env, action_seq):
    """Clone the env and roll the action sequence forward; return (return, info)."""
    sim = copy.deepcopy(base_env)
    total = 0.0
    info = {}
    for a in action_seq:
        _, r, term, trunc, info = sim.step(a)
        total += r
        if term or trunc:
            break
    return total, info


class CEMPlanner:
    def __init__(self, n_segments=24, n_samples=48, n_elite=8, n_iters=6,
                 init_std=0.6, min_std=0.1, alpha=0.7, plan_horizon=350, seed=0):
        self.n_segments = n_segments
        self.n_samples = n_samples
        self.n_elite = n_elite
        self.n_iters = n_iters
        self.init_std = init_std
        self.min_std = min_std            # std floor: keep exploring, avoid collapse
        self.alpha = alpha               # momentum on the distribution update
        self.plan_horizon = plan_horizon  # cap rollout length (herding is fast)
        self.rng = np.random.default_rng(seed)

    def plan(self, base_env, verbose=True):
        """Optimize an action sequence for ``base_env``'s current state.
        Returns (action_sequence [H,3], best_return, best_info)."""
        H = min(self.plan_horizon, base_env.cfg.max_steps)
        repeat = int(np.ceil(H / self.n_segments))
        A = base_env.action_space.shape[0]

        mean = np.zeros((self.n_segments, A), dtype=np.float32)
        std = np.full((self.n_segments, A), self.init_std, dtype=np.float32)
        best = (-np.inf, None, {})

        for it in range(self.n_iters):
            noise = self.rng.standard_normal((self.n_samples, self.n_segments, A))
            samples = np.clip(mean[None] + std[None] * noise, -1.0, 1.0).astype(np.float32)

            rets = np.empty(self.n_samples, dtype=np.float32)
            infos = []
            for i, s in enumerate(samples):
                seq = np.repeat(s, repeat, axis=0)[:H]
                rets[i], info = rollout_return(base_env, seq)
                infos.append(info)

            elite_idx = np.argsort(rets)[-self.n_elite:]
            elites = samples[elite_idx]
            # Momentum update + std floor: refine toward the elites without
            # collapsing onto an early mediocre solution.
            mean = self.alpha * elites.mean(axis=0) + (1 - self.alpha) * mean
            std = np.maximum(self.alpha * elites.std(axis=0) + (1 - self.alpha) * std,
                             self.min_std)

            bi = int(np.argmax(rets))
            if rets[bi] > best[0]:
                best = (float(rets[bi]), samples[bi].copy(), infos[bi])
            if verbose:
                print(f"  iter {it}: best_return={rets.max():7.1f}  "
                      f"elite_mean={rets[elite_idx].mean():7.1f}  "
                      f"penned={infos[bi].get('n_penned', 0)}/{base_env.n_active}",
                      flush=True)

        best_seq = np.repeat(best[1], repeat, axis=0)[:H]
        return best_seq, best[0], best[2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--segments", type=int, default=30)
    ap.add_argument("--samples", type=int, default=48)
    ap.add_argument("--elite", type=int, default=8)
    ap.add_argument("--iters", type=int, default=6)
    ap.add_argument("--init-std", type=float, default=0.6)
    ap.add_argument("--plan-horizon", type=int, default=350,
                    help="max steps per evaluated path (herding finishes fast)")
    ap.add_argument("--gif", default=None, help="render the best plan to this gif")
    args = ap.parse_args()

    # Planning env (no rendering — kept cheap for cloning).
    base = SheepdogHerdingEnv()
    base.reset(seed=args.seed)

    planner = CEMPlanner(n_segments=args.segments, n_samples=args.samples,
                         n_elite=args.elite, n_iters=args.iters,
                         init_std=args.init_std, plan_horizon=args.plan_horizon,
                         seed=args.seed)
    print(f"CEM planning (seed={args.seed}, {args.samples} samples x {args.iters} iters "
          f"= {args.samples * args.iters} path evaluations)...")
    plan, ret, info = planner.plan(base)
    print(f"\nbest plan: return={ret:.1f}  penned={info.get('n_penned')}/{base.n_active}")

    # Re-run the plan in a fresh env with the SAME seed (identical dynamics) to
    # confirm and optionally render.
    env = SheepdogHerdingEnv(render_mode="rgb_array" if args.gif else None)
    obs, _ = env.reset(seed=args.seed)
    frames, total = [], 0.0
    for t, a in enumerate(plan):
        obs, r, term, trunc, info = env.step(a)
        total += r
        if args.gif and t % 2 == 0:
            frames.append(env.render())
        if term or trunc:
            break
    print(f"executed plan: return={total:.1f}  penned={info['n_penned']}/{env.n_active}  "
          f"success={info.get('is_success')}  steps={t + 1}")

    if args.gif and frames:
        try:
            import imageio
            frames += [env.render()] * 20
            imageio.mimsave(args.gif, frames, fps=30, loop=0)
            print(f"saved {len(frames)} frames -> {args.gif}")
        except ImportError:
            print("install imageio to save the gif: pip install imageio")
    env.close()


if __name__ == "__main__":
    main()
