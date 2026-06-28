"""Render several rollouts side-by-side as a single GIF.

    python examples/rollouts_gif.py                 # 2x2 heuristic rollouts
    python examples/rollouts_gif.py --seeds 0 2 3 5 --out rollouts.gif
    python examples/rollouts_gif.py --random        # random policy
    python examples/rollouts_gif.py --wolves

Each panel shows one episode (different seed) playing simultaneously, with a
live "penned / total" counter and a WIN flash when the target is reached.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sheepdog_env import SheepdogHerdingEnv
from examples.heuristic_agent import HeuristicShepherd

PANEL_W, PANEL_H = 336, 224          # per-rollout render size
PAD = 6                               # gap between panels


def _label(img, text, color=(255, 255, 255)):
    """Stamp a small caption top-left using PIL if available (else skip)."""
    try:
        from PIL import Image, ImageDraw
    except Exception:
        return img
    im = Image.fromarray(img)
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, 150, 18], fill=(0, 0, 0))
    d.text((4, 4), text, fill=color)
    return np.asarray(im)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[3, 5, 2, 6])
    ap.add_argument("--random", action="store_true")
    ap.add_argument("--wolves", action="store_true")
    ap.add_argument("--out", default="rollouts.gif")
    ap.add_argument("--stride", type=int, default=3, help="record every Nth step")
    args = ap.parse_args()

    n = len(args.seeds)
    cols = 2 if n > 1 else 1
    rows = (n + cols - 1) // cols

    envs, agents, states = [], [], []
    for s in args.seeds:
        e = SheepdogHerdingEnv(render_mode="rgb_array", enable_wolves=args.wolves,
                               render_size=(PANEL_W, PANEL_H))
        ag = None if args.random else HeuristicShepherd(e)
        obs, info = e.reset(seed=s)
        envs.append(e); agents.append(ag)
        states.append({"obs": obs, "done": False, "info": info, "win": False,
                       "hold": 0})

    max_steps = max(e.cfg.max_steps for e in envs)
    frames = []
    for t in range(max_steps):
        for e, ag, st in zip(envs, agents, states):
            if st["done"]:
                continue
            action = e.action_space.sample() if ag is None else ag.act(st["obs"])
            obs, r, term, trunc, info = e.step(action)
            st.update(obs=obs, info=info)
            if info.get("is_success"):
                st["win"] = True
            if term or trunc:
                st["done"] = True
                st["hold"] = 40            # linger on the final frame

        if t % args.stride == 0 or all(s["done"] for s in states):
            panels = []
            for e, st, seed in zip(envs, states, args.seeds):
                img = e.render()
                tag = "WIN" if st["win"] else f"{st['info']['n_penned']}/{e.cfg.n_sheep}"
                col = (120, 255, 120) if st["win"] else (255, 255, 255)
                panels.append(_label(img, f"seed {seed}  pen {tag}", col))
            frames.append(_tile(panels, rows, cols))

        if all(s["done"] and s["hold"] <= 0 for s in states):
            break
        for st in states:
            if st["done"]:
                st["hold"] -= 1

    # Hold the final frame a moment.
    frames += [frames[-1]] * 25

    try:
        import imageio
        imageio.mimsave(args.out, frames, fps=30, loop=0)
        size_kb = os.path.getsize(args.out) / 1024
        print(f"saved {len(frames)} frames -> {args.out} ({size_kb:.0f} KB)")
    except ImportError:
        print("install imageio to save the gif: pip install imageio")

    for e in envs:
        e.close()


def _tile(panels, rows, cols):
    h, w, _ = panels[0].shape
    grid = np.full((rows * h + (rows - 1) * PAD, cols * w + (cols - 1) * PAD, 3),
                   30, dtype=np.uint8)
    for i, p in enumerate(panels):
        r, c = divmod(i, cols)
        y, x = r * (h + PAD), c * (w + PAD)
        grid[y:y + h, x:x + w] = p
    return grid


if __name__ == "__main__":
    main()
