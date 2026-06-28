"""Smoke + API-conformance tests. Run: python -m pytest -q  (or run directly)."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sheepdog_env import SheepdogHerdingEnv


def test_penned_sheep_zeroed_in_obs():
    """Once a sheep is penned, its position/velocity in the observation are
    zeroed (only the status flag remains) so the policy stops attending to it."""
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "examples"))
    from heuristic_agent import HeuristicShepherd
    env = SheepdogHerdingEnv(enable_bark=True)   # heuristic herds via barking
    agent = HeuristicShepherd(env)
    obs, info = env.reset(seed=1)
    base, N = 14, env.cfg.n_sheep        # per-sheep block starts after dog+pen+opening+time
    for _ in range(env.cfg.max_steps):
        obs, r, term, trunc, info = env.step(agent.act(obs))
        penned = env.flock.penned & env.flock.alive
        if penned.sum() >= 3:
            sheep = obs[base:base + N * 5].reshape(N, 5)
            assert np.abs(sheep[penned][:, :4]).max() < 1e-6, "penned sheep not zeroed in obs"
            assert (sheep[penned][:, 4] == 1.0).all(), "penned status flag should be 1"
            break
        if term or trunc:
            break
    env.close()


def _rollout(env, steps=200, seed=0):
    obs, info = env.reset(seed=seed)
    assert env.observation_space.contains(obs), "reset obs out of space"
    for _ in range(steps):
        a = env.action_space.sample()
        obs, r, term, trunc, info = env.step(a)
        assert env.observation_space.contains(obs), "step obs out of space"
        assert np.isfinite(r)
        if term or trunc:
            obs, info = env.reset()
    return True


def test_vector_obs():
    env = SheepdogHerdingEnv(obs_mode="vector")
    assert _rollout(env)
    env.close()


def test_pixel_obs():
    env = SheepdogHerdingEnv(obs_mode="pixel")
    obs, _ = env.reset(seed=1)
    assert obs.shape == (84, 84, 3) and obs.dtype == np.uint8
    assert _rollout(env, steps=50)
    env.close()


def test_wolves():
    env = SheepdogHerdingEnv(enable_wolves=True, max_steps=200)
    assert _rollout(env, steps=200)
    env.close()


def test_determinism():
    e1 = SheepdogHerdingEnv(); e2 = SheepdogHerdingEnv()
    o1, _ = e1.reset(seed=42); o2, _ = e2.reset(seed=42)
    assert np.allclose(o1, o2)
    a = np.array([0.5, -0.2, 1.0], dtype=np.float32)
    for _ in range(20):
        o1, r1, *_ = e1.step(a); o2, r2, *_ = e2.step(a)
    assert np.allclose(o1, o2) and r1 == r2
    e1.close(); e2.close()


def test_penning_is_reachable():
    """Sanity check that the mechanics allow penning: the scripted shepherd
    should pen a meaningful number of sheep on at least one of several seeds.

    (It is only a partial baseline — it does not reliably hit the 80% target —
    so we check that penning *works*, not that it wins.)"""
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "examples"))
    from heuristic_agent import HeuristicShepherd
    best = 0
    for seed in (6, 7, 8):
        env = SheepdogHerdingEnv(enable_bark=True)   # heuristic herds via barking
        agent = HeuristicShepherd(env)
        obs, info = env.reset(seed=seed)
        for _ in range(env.cfg.max_steps):
            obs, r, term, trunc, info = env.step(agent.act(obs))
            if term or trunc:
                break
        best = max(best, info["n_penned"])
        env.close()
    assert best >= 10, f"heuristic penned at most {best} sheep across seeds"


def test_dog_cannot_enter_pen():
    """The dog must stay fenced out of the pen, and sheep must only enter
    through the opening (no wall leaks)."""
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "examples"))
    from heuristic_agent import HeuristicShepherd
    env = SheepdogHerdingEnv(enable_bark=True)   # heuristic herds via barking
    agent = HeuristicShepherd(env)
    x0, y0, x1, y1 = env.pen
    obs, info = env.reset(seed=3)
    dog_inside = 0
    for _ in range(env.cfg.max_steps):
        obs, r, term, trunc, info = env.step(agent.act(obs))
        d = env.dog_pos
        if x0 <= d[0] <= x1 and y0 <= d[1] <= y1:
            dog_inside += 1
        if term or trunc:
            break
    assert dog_inside == 0, f"dog entered the pen on {dog_inside} steps"
    # Every penned sheep must be inside the pen bounds.
    f = env.flock
    pen = f.penned & f.alive
    pos = f.pos[pen]
    if len(pos):
        assert (pos[:, 0] >= x0 - 0.6).all() and (pos[:, 0] <= x1 + 0.6).all()
        assert (pos[:, 1] >= y0 - 0.6).all() and (pos[:, 1] <= y1 + 0.6).all()
    env.close()


if __name__ == "__main__":
    test_vector_obs(); print("vector OK")
    test_pixel_obs(); print("pixel OK")
    test_wolves(); print("wolves OK")
    test_determinism(); print("determinism OK")
    test_penning_is_reachable(); print("heuristic penning OK")
    test_dog_cannot_enter_pen(); print("dog-containment OK")
    test_penned_sheep_zeroed_in_obs(); print("penned-zeroed-in-obs OK")
    print("all smoke tests passed")
