"""Run an episode and save it as a GIF (or show a live window).

Usage:
    python examples/demo.py                  # heuristic agent -> demo.gif
    python examples/demo.py --random         # random actions
    python examples/demo.py --human          # live pygame window (needs pygame)
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sheepdog_env import SheepdogHerdingEnv
from examples.heuristic_agent import HeuristicShepherd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--random", action="store_true", help="random policy")
    ap.add_argument("--human", action="store_true", help="live window")
    ap.add_argument("--wolves", action="store_true", help="enable dusk wolves")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="demo.gif")
    ap.add_argument("--max-frames", type=int, default=900)
    args = ap.parse_args()

    env = SheepdogHerdingEnv(
        render_mode="human" if args.human else "rgb_array",
        enable_wolves=args.wolves,
    )
    agent = None if args.random else HeuristicShepherd(env)
    obs, info = env.reset(seed=args.seed)

    frames = []
    total = 0.0
    for i in range(min(args.max_frames, env.cfg.max_steps)):
        if agent is None:
            action = env.action_space.sample()
        else:
            action = agent.act(obs)
        obs, r, term, trunc, info = env.step(action)
        total += r
        frame = env.render()
        if not args.human and frame is not None:
            frames.append(frame)
        if term or trunc:
            break

    print(f"return={total:.1f}  penned={info['n_penned']}/{env.cfg.n_sheep}  "
          f"success={info.get('is_success')}")

    if not args.human and frames:
        try:
            import imageio
            imageio.mimsave(args.out, frames, fps=30)
            print(f"saved {len(frames)} frames -> {args.out}")
        except ImportError:
            print("install imageio to save a gif: pip install imageio")
    env.close()


if __name__ == "__main__":
    main()
