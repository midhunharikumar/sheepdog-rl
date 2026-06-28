"""Diagnostic: curated states -> check reward (score), gradient, and dynamics.

Builds hand-crafted scenarios and verifies that:
  * the reward components are what we expect,
  * "good" behaviour (drive the flock through the gap) scores higher than
    doing nothing  -> the reward gradient points the right way,
  * the bark dynamics actually move sheep toward the gap from a good position,
  * even the trivial 1-sheep case is solvable.

If the simplest cases don't score/solve correctly, the problem is the env/reward.
If they do, the problem is RL exploration/optimization.

    python examples/diagnose.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sheepdog_env import SheepdogHerdingEnv
from heuristic_agent import HeuristicShepherd


def make_env(n_active):
    e = SheepdogHerdingEnv(n_bushes=0, n_rocks=0)   # no obstacles, isolate the herding
    e.cfg.n_active = n_active
    e.reset(seed=0)
    return e


def set_state(e, sheep_xy, dog_xy, penned_idx=()):
    """Force a curated state: explicit sheep positions, dog position."""
    n = len(sheep_xy)
    f = e.flock
    f.pos[:n] = np.array(sheep_xy, dtype=np.float32)
    f.penned[:] = False
    f.alive[:n] = True
    f.present[:] = False
    f.present[:n] = True
    for i in penned_idx:
        f.penned[i] = True
    e.dog_pos = np.array(dog_xy, dtype=np.float32)
    h = e.opening_point - e.dog_pos
    e.dog_heading = (h / max(np.linalg.norm(h), 1e-6)).astype(np.float32)
    e._bark_cd = 0
    e._bark_timer = 0
    e.t = 0
    return e._get_obs()


def rollout(e, policy, steps):
    obs = e._get_obs()
    total = 0.0
    info = {}
    for _ in range(steps):
        obs, r, term, trunc, info = e.step(policy(obs))
        total += r
        if term or trunc:
            break
    return total, info


def noop(_obs):
    # do nothing: speed 0 (a0=-1 -> 0), heading 0, no bark
    return np.array([-1.0, 0.0, -1.0], dtype=np.float32)


def banner(t):
    print("\n" + "=" * 72 + f"\n{t}\n" + "=" * 72)


def main():
    banner("Geometry")
    e = make_env(1)
    gx, gy = e.opening_point
    x0, y0, x1, y1 = e.pen
    print(f"pen x[{x0:.0f},{x1:.0f}] y[{y0:.0f},{y1:.0f}]  entrance=({gx:.0f},{gy:.0f})  "
          f"gap_width={e.cfg.pen_opening_width}  dog_max_speed={e.cfg.dog_max_speed}  "
          f"sheep delta={e.cfg.flock.delta}")

    # ---- 1) reward components for a few curated single-sheep states ---------
    banner("1) Reward components (single active sheep, target=1)")
    for label, sxy, dxy in [
        ("sheep AT entrance (just outside)", (gx - 4, gy), (gx - 12, gy)),
        ("sheep FAR left of entrance",        (gx - 60, gy), (gx - 70, gy)),
        ("sheep BEHIND the pen",              (x1 + 4, gy), (x1 + 10, gy)),
        ("sheep already PENNED (inside)",     (gx + 8, gy), (gx - 12, gy)),
    ]:
        e = make_env(1)
        penned = (3,) if "PENNED" in label else ()
        if penned:
            set_state(e, [(gx + 8, gy)], (gx - 12, gy))
            e.flock.penned[0] = True
        else:
            set_state(e, [sxy], dxy)
        _, r, *_ , info = e.step(noop(None))
        rb = info["reward_breakdown"]
        print(f"  {label:38s} reward={r:+7.3f}  " +
              "  ".join(f"{k}={v:+.2f}" for k, v in rb.items() if abs(v) > 1e-9 or k in ("entrance",)))

    # ---- 2) does the reward gradient favour driving over doing nothing? -----
    banner("2) Good policy (heuristic) vs do-nothing  [60-step return, then penned]")
    scenarios = [
        ("1 sheep, dog directly behind, aligned with gap", 1, [(gx - 8, gy)], (gx - 18, gy)),
        ("1 sheep, dog behind but OFF-axis (gy+8)",        1, [(gx - 8, gy + 8)], (gx - 18, gy + 8)),
        ("5 sheep clustered just left of gap",             5,
         [(gx - 8, gy), (gx - 10, gy + 3), (gx - 10, gy - 3), (gx - 13, gy + 1), (gx - 13, gy - 2)],
         (gx - 22, gy)),
        ("full flock (40), default spawn",                 40, None, None),
    ]
    for label, n, sxy, dxy in scenarios:
        # heuristic
        e = make_env(n); ag = HeuristicShepherd(e)
        if sxy is not None:
            set_state(e, sxy, dxy)
        rh, ih = rollout(e, ag.act, 150)
        # do-nothing
        e2 = make_env(n)
        if sxy is not None:
            set_state(e2, sxy, dxy)
        rn, ins = rollout(e2, noop, 150)
        print(f"  {label:48s}")
        print(f"      heuristic : return={rh:+8.1f}  penned={ih.get('n_penned')}/{n}")
        print(f"      do-nothing: return={rn:+8.1f}  penned={ins.get('n_penned')}/{n}")
        print(f"      => good behaviour scores higher: {rh > rn}")

    # ---- 3) does a radial bark from directly behind move the sheep to the gap?
    banner("3) Bark dynamics: 1 sheep at (gx-8, gy), dog at (gx-16, gy), bark once")
    e = make_env(1)
    set_state(e, [(gx - 8, gy)], (gx - 16, gy))
    print(f"  before: sheep=({e.flock.pos[0,0]:.1f},{e.flock.pos[0,1]:.1f})  dog=({e.dog_pos[0]:.1f},{e.dog_pos[1]:.1f})")
    # action: stay put, bark
    a = np.array([-1.0, 0.0, 1.0], dtype=np.float32)
    for t in range(6):
        e.step(a)
        print(f"  t={t}: sheep=({e.flock.pos[0,0]:.2f},{e.flock.pos[0,1]:.2f})  "
              f"barking={e._bark_timer>0}  penned={bool(e.flock.penned[0])}  "
              f"dx_toward_gap={e.flock.pos[0,0]-(gx-8):+.2f}")


if __name__ == "__main__":
    main()
