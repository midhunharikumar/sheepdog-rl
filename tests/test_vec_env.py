"""Tests for the batched vector env: numerical parity + VecEnv contract."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sheepdog_env.env import SheepdogHerdingEnv
from sheepdog_env.flocking import FlockParams
from sheepdog_env.vec_env import BatchedSheepdogVecEnv


def _det_cfg():
    """Config with all RNG (noise + grazing) removed -> deterministic dynamics."""
    fp = FlockParams()
    fp.noise = 0.0
    fp.graze_prob = 0.0
    return dict(n_sheep=8, enable_bark=True, action_mode="cartesian", flock=fp)


def test_batched_matches_single_env():
    """With noise/grazing off and identical initial state + actions, one batched
    world reproduces the single env trajectory to float precision."""
    kw = _det_cfg()
    single = SheepdogHerdingEnv(**kw)
    single.reset(seed=3)
    vec = BatchedSheepdogVecEnv(1, **kw)
    vec.pos[0] = single.flock.pos
    vec.heading[0] = single.flock.heading
    vec.penned[0] = single.flock.penned
    vec.alive[0] = single.flock.alive
    vec.present[0] = single.flock.present
    vec.staged[0] = False
    vec.dog_pos[0] = single.dog_pos
    vec.dog_heading[0] = single.dog_heading
    vec.t[0] = 0
    vec.bark_cd[0] = 0
    vec.bark_timer[0] = 0
    vec.active_n[0] = 8

    rng = np.random.default_rng(0)
    for _ in range(400):
        a = rng.uniform(-1, 1, size=3).astype(np.float32)
        _, r1, term, trunc, _ = single.step(a)
        vec.step_async(a[None, :])
        _, r2, _, _ = vec.step_wait()
        assert np.abs(single.flock.pos - vec.pos[0]).max() < 1e-3
        assert abs(r1 - r2[0]) < 1e-4
        assert int(single.flock.penned.sum()) == int(vec.penned[0].sum())
        if term or trunc:
            break


def test_vecenv_contract_and_autoreset():
    B = 8
    vec = BatchedSheepdogVecEnv(B, n_sheep=8, enable_bark=True,
                                action_mode="cartesian", max_steps=50)
    obs = vec.reset()
    assert obs.shape == (B, vec._obs_dim)
    rng = np.random.default_rng(1)
    dones_total, ep_infos = 0, 0
    for _ in range(300):
        a = rng.uniform(-1, 1, size=(B, 3)).astype(np.float32)
        vec.step_async(a)
        obs, rew, dones, infos = vec.step_wait()
        assert obs.shape == (B, vec._obs_dim)
        assert rew.shape == (B,) and dones.shape == (B,)
        for b, info in enumerate(infos):
            assert {"flock_spread", "frac_penned", "is_success"} <= set(info)
            if dones[b]:
                assert "terminal_observation" in info and "episode" in info
                ep_infos += 1
        dones_total += int(dones.sum())
    assert dones_total > 0 and ep_infos == dones_total


def test_no_sheep_leak_through_fence():
    """Penned sheep stay inside the pen rectangle (fence containment holds)."""
    B = 6
    vec = BatchedSheepdogVecEnv(B, n_sheep=8, enable_bark=True, action_mode="cartesian")
    vec.reset()
    rng = np.random.default_rng(2)
    x0, y0, x1, y1 = vec.pen
    for _ in range(400):
        a = rng.uniform(-1, 1, size=(B, 3)).astype(np.float32)
        vec.step_async(a)
        vec.step_wait()
        pen = vec.penned & vec.present
        if pen.any():
            px = vec.pos[..., 0][pen]
            py = vec.pos[..., 1][pen]
            assert (px >= x0 - 1e-3).all() and (px <= x1 + 1e-3).all()
            assert (py >= y0 - 1e-3).all() and (py <= y1 + 1e-3).all()


def test_curriculum_hooks():
    vec = BatchedSheepdogVecEnv(4, n_sheep=12)
    assert vec.env_method("set_active", 6) == [6, 6, 6, 6]
    vec.reset()
    assert int(vec.active_n[0]) == 6
    assert (vec.present.sum(1) == 6).all()


def test_dr_gap_parity_with_single_env():
    """A batched world with a randomized (but fixed-range) gate width must match
    the single env at that gap, proving the per-world fence geometry is correct."""
    G = 20.0
    fp = FlockParams()
    fp.noise = 0.0
    fp.graze_prob = 0.0
    single = SheepdogHerdingEnv(n_sheep=10, pen_opening_width=G,
                                action_mode="cartesian", enable_bark=True, flock=fp)
    single.reset(seed=5)
    vec = BatchedSheepdogVecEnv(2, n_sheep=10, action_mode="cartesian",
                                enable_bark=True, flock=fp, dr_gap_range=(G, G))
    assert vec._per_world_walls
    vec.pos[0] = single.flock.pos
    vec.heading[0] = single.flock.heading
    vec.penned[0] = single.flock.penned
    vec.alive[0] = single.flock.alive
    vec.present[0] = single.flock.present
    vec.staged[0] = False
    vec.dog_pos[0] = single.dog_pos
    vec.dog_heading[0] = single.dog_heading
    vec.t[0] = 0
    vec.active_n[0] = 10
    rng = np.random.default_rng(7)
    for _ in range(300):
        a = rng.uniform(-1, 1, size=3).astype(np.float32)
        _, _, term, trunc, _ = single.step(a)
        vec.step_async(np.tile(a, (2, 1)))
        vec.step_wait()
        assert np.abs(single.flock.pos - vec.pos[0]).max() < 1e-3
        if term or trunc:
            break


def test_dr_ranges_and_no_leak():
    vec = BatchedSheepdogVecEnv(16, n_sheep=32, action_mode="cartesian",
                                enable_bark=True, dr_active_range=(8, 32),
                                dr_spawn_x_range=(0.2, 0.6), dr_gap_range=(16, 30))
    vec.reset()
    assert vec._obs_dim == 14 + 32 * 5            # obs size fixed despite DR
    x0, y0, x1, y1 = vec.pen
    rng = np.random.default_rng(0)
    gaps, acts = set(), set()
    for _ in range(300):
        vec.step_async(rng.uniform(-1, 1, size=(16, 3)).astype(np.float32))
        vec.step_wait()
        pen = vec.penned & vec.present
        if pen.any():
            px, py = vec.pos[..., 0][pen], vec.pos[..., 1][pen]
            assert (px >= x0 - 1e-2).all() and (px <= x1 + 1e-2).all()
            assert (py >= y0 - 1e-2).all() and (py <= y1 + 1e-2).all()
        gaps.update(np.round(vec.gap_w, 1).tolist())
        acts.update(vec.active_n.tolist())
    assert min(gaps) >= 16 - 1e-6 and max(gaps) <= 30 + 1e-6
    assert min(acts) >= 8 and max(acts) <= 32 and len(acts) > 3
