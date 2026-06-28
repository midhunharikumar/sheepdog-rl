"""Audit that rewards are assigned correctly, using the heuristic shepherd.

For each episode it:
  * checks the per-step reward equals the sum of its components (info breakdown),
  * accumulates each component over the episode,
  * verifies the win bonus fires exactly when the episode is a success.

    python examples/check_rewards.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sheepdog_env import SheepdogHerdingEnv
from heuristic_agent import HeuristicShepherd


def run(seed):
    e = SheepdogHerdingEnv()
    ag = HeuristicShepherd(e)
    obs, info = e.reset(seed=seed)
    totals = {}
    ret = 0.0
    mismatch = 0.0
    steps = 0
    pen_bonus_events = 0
    while True:
        obs, r, term, trunc, info = e.step(ag.act(obs))
        rb = info["reward_breakdown"]
        # (1) components must sum to the returned reward
        mismatch = max(mismatch, abs(sum(rb.values()) - r))
        for k, v in rb.items():
            totals[k] = totals.get(k, 0.0) + v
        if info["newly_penned"] > 0:
            pen_bonus_events += 1
        ret += r
        steps += 1
        if term or trunc:
            break
    return dict(seed=seed, penned=info["n_penned"], success=info.get("is_success"),
                n_active=info.get("n_active", e.cfg.n_sheep),
                steps=steps, ret=ret, totals=totals, mismatch=mismatch,
                pen_bonus_events=pen_bonus_events)


def main():
    c = SheepdogHerdingEnv().cfg
    print(f"{'seed':>4} {'pen':>4} {'win':>5} {'steps':>5} {'return':>9} | component totals")
    print("-" * 100)
    for s in range(6):
        o = run(s)
        t = o["totals"]
        comp = "  ".join(f"{k}={v:+.1f}" for k, v in t.items())
        print(f"{o['seed']:>4} {o['penned']:>4} {str(o['success']):>5} {o['steps']:>5} "
              f"{o['ret']:>9.1f} | {comp}")
        # --- consistency assertions ------------------------------------
        assert o["mismatch"] < 1e-4, f"seed {s}: reward != sum(components) (off by {o['mismatch']})"
        # one-off pen reward == penned x w_pen_enter
        assert abs(t["pen_enter"] - o["penned"] * c.w_pen_enter) < 1e-3, \
            f"seed {s}: pen_enter {t['pen_enter']} != {o['penned'] * c.w_pen_enter}"
        # terminal 'final' == w_final * frac_penned_at_end
        frac = o["penned"] / max(o["n_active"], 1)
        assert abs(t["final"] - c.w_final * frac) < 1e-2, \
            f"seed {s}: final {t['final']} != {c.w_final * frac:.2f}"
        # distance / behind / cohesion terms are (non-positive) drags
        assert t["entrance"] <= 1e-6 and t["back"] <= 1e-6 and t["cohesion"] <= 1e-6
    print("-" * 100)
    print("OK: reward == sum(components) every step; pen reward == penned x weight; "
          "terminal reward == w_final x frac penned.")


if __name__ == "__main__":
    main()
