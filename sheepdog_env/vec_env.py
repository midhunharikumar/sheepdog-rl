"""Batched, fully-vectorised Sheepdog environment.

This is a drop-in Stable-Baselines3 ``VecEnv`` that simulates ``num_envs`` copies
of :class:`~sheepdog_env.env.SheepdogHerdingEnv` *simultaneously* in NumPy. Every
quantity carries a leading batch axis ``(B, N, ...)`` and a single ``step`` call
advances all B worlds in a handful of vectorised NumPy ops -- no per-env Python
loop, no subprocess IPC.

Why this is fast: the env is cheap (tiny flock) so per-env Python overhead and
SubprocVecEnv pickling dominate. Batching removes both. The one subtlety -- sheep
in different worlds must not interact -- is handled by computing the Strombom
cohesion/separation forces as per-world ``(B, N, N)`` tensors. The pen geometry is
identical across worlds, so wall/containment ops simply flatten ``(B, N) -> (B*N)``
and reuse the exact same tested geometry code.

Scope: vector observations only, no wolves, no pixel/render (training uses a
separate env for rollout GIFs). It is numerically faithful to the single env --
with noise/grazing disabled and identical initial state it reproduces the
single-env trajectory to float precision (see tests/test_vec_env.py).
"""

from __future__ import annotations

from typing import Optional

import numpy as np

try:
    from gymnasium import spaces
    from stable_baselines3.common.vec_env.base_vec_env import VecEnv
except Exception as exc:  # pragma: no cover
    raise ImportError("vec_env needs gymnasium + stable_baselines3") from exc

from .env import EnvConfig
from .geometry import build_pen_walls, resolve_walls, contain_pen, wall_repulsion


def _unit(v, eps=1e-8):
    """Normalise along the last axis, leaving (near-)zero vectors as zero."""
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, eps)


# --- per-world batched geometry (used only when the gate width is randomized
# per episode, so each world has its own (K,4) wall set) ------------------------

def _wall_repulsion_pw(P, walls, reach):
    """Soft repulsion (B,N,2) from per-world walls (B,K,4) within ``reach``."""
    A = walls[:, :, :2]
    Bb = walls[:, :, 2:]
    AB = Bb - A
    L2 = np.einsum('bkd,bkd->bk', AB, AB)
    diff = P[:, :, None, :] - A[:, None, :, :]                 # (B,N,K,2)
    t = np.clip(np.einsum('bnkd,bkd->bnk', diff, AB)
                / np.where(L2 < 1e-9, 1.0, L2)[:, None, :], 0.0, 1.0)
    proj = A[:, None, :, :] + t[..., None] * AB[:, None, :, :]
    v = P[:, :, None, :] - proj
    d = np.linalg.norm(v, axis=3)
    strength = np.clip((reach - d) / max(reach, 1e-6), 0.0, 1.0) * (d < reach)
    u = v / np.maximum(np.linalg.norm(v, axis=3, keepdims=True), 1e-6)
    return (u * strength[..., None]).sum(2)


def _resolve_walls_pw(P0, P1, walls, eps=0.3):
    """Block movements P0->P1 that cross a per-world wall (B,K,4); slide along it.
    Earliest crossing wins per point (non-overlapping fence -> at most one)."""
    A = walls[:, :, :2]
    Bb = walls[:, :, 2:]
    s = Bb - A
    sl = np.linalg.norm(s, axis=2)
    good = sl > 1e-9
    sl_s = np.where(good, sl, 1.0)
    sdir = s / sl_s[..., None]
    nrm = np.stack([-sdir[..., 1], sdir[..., 0]], axis=2)      # (B,K,2)
    An = np.einsum('bkd,bkd->bk', A, nrm)
    d0 = np.einsum('bnd,bkd->bnk', P0, nrm) - An[:, None, :]
    d1 = np.einsum('bnd,bkd->bnk', P1, nrm) - An[:, None, :]
    denom = d0 - d1
    safe = np.abs(denom) > 1e-9
    t = np.where(safe, d0 / np.where(safe, denom, 1.0), -1.0)
    cross = P0[:, :, None, :] + t[..., None] * (P1 - P0)[:, :, None, :]
    u = np.einsum('bnkd,bkd->bnk', cross - A[:, None, :, :], sdir) / sl_s[:, None, :]
    changed = np.sign(d0) != np.sign(d1)
    crossed = (safe & changed & (t >= -1e-4) & (t <= 1.0 + 1e-4)
               & (u >= -0.02) & (u <= 1.02) & good[:, None, :])
    side = np.sign(d0)
    side = np.where(side == 0.0, -np.sign(d1), side)
    tt = np.where(crossed, t, np.inf)
    kbest = np.argmin(tt, axis=2)
    any_cross = np.isfinite(tt.min(axis=2))
    B, N = P1.shape[0], P1.shape[1]
    g = kbest[:, :, None, None]
    cp = np.take_along_axis(cross, g, axis=2)[:, :, 0, :]
    sdir_e = np.broadcast_to(sdir[:, None], (B, N, sdir.shape[1], 2))
    nrm_e = np.broadcast_to(nrm[:, None], (B, N, nrm.shape[1], 2))
    sdb = np.take_along_axis(sdir_e, g, axis=2)[:, :, 0, :]
    nb = np.take_along_axis(nrm_e, g, axis=2)[:, :, 0, :]
    sb = np.take_along_axis(side, kbest[:, :, None], axis=2)[:, :, 0]
    rem = P1 - cp
    tang = np.einsum('bnd,bnd->bn', rem, sdb)[..., None] * sdb
    new = cp + tang + nb * (sb[..., None] * eps)
    return np.where(any_cross[..., None], new, P1)


def _contain_pen_pw(P0, P1, pen, side, gap_lo, gap_hi):
    """Hard pen-rect containment with a PER-WORLD gap interval (gap_lo/hi: (B,)).
    A point may only change inside/outside through the opening; else move cancels."""
    x0, y0, x1, y1 = pen
    inside = lambda P: ((P[..., 0] >= x0) & (P[..., 0] <= x1) &
                        (P[..., 1] >= y0) & (P[..., 1] <= y1))
    in0, in1 = inside(P0), inside(P1)
    flip = in0 != in1
    if not flip.any():
        return P1
    dx = (P1 - P0)[..., 0]
    dy = (P1 - P0)[..., 1]
    INF = np.inf

    def vert(xe):           # crossing a vertical edge x=xe
        with np.errstate(divide='ignore', invalid='ignore'):
            t = (xe - P0[..., 0]) / np.where(np.abs(dx) > 1e-12, dx, 1.0)
        yc = P0[..., 1] + t * dy
        ok = (np.abs(dx) > 1e-12) & (t > 1e-9) & (t <= 1.0) & (yc >= y0 - 1e-6) & (yc <= y1 + 1e-6)
        return np.where(ok, t, INF)

    def horiz(ye):          # crossing a horizontal edge y=ye
        with np.errstate(divide='ignore', invalid='ignore'):
            t = (ye - P0[..., 1]) / np.where(np.abs(dy) > 1e-12, dy, 1.0)
        xc = P0[..., 0] + t * dx
        ok = (np.abs(dy) > 1e-12) & (t > 1e-9) & (t <= 1.0) & (xc >= x0 - 1e-6) & (xc <= x1 + 1e-6)
        return np.where(ok, t, INF)

    ts = np.stack([vert(x0), vert(x1), horiz(y0), horiz(y1)], axis=0)  # left,right,top,bottom
    edge = np.argmin(ts, axis=0)
    tmin = np.min(ts, axis=0)
    tsafe = np.where(np.isfinite(tmin), tmin, 0.0)   # avoid inf*0 where no crossing
    cx = P0[..., 0] + tsafe * dx
    cy = P0[..., 1] + tsafe * dy
    opening_idx = {"left": 0, "right": 1, "top": 2, "bottom": 3}[side]
    coord = cy if side in ("left", "right") else cx
    legal = (edge == opening_idx) & (coord >= gap_lo[:, None] - 1e-3) & (coord <= gap_hi[:, None] + 1e-3)
    cancel = flip & ~(legal & np.isfinite(tmin))
    return np.where(cancel[..., None], P0, P1)


class BatchedSheepdogVecEnv(VecEnv):
    """``num_envs`` Sheepdog worlds stepped together in batched NumPy."""

    def __init__(self, num_envs: int, config: Optional[EnvConfig] = None,
                 seed: Optional[int] = None, **overrides):
        cfg = config or EnvConfig()
        for k, v in overrides.items():
            if not hasattr(cfg, k):
                raise TypeError(f"Unknown config field: {k!r}")
            setattr(cfg, k, v)
        if cfg.obs_mode != "vector":
            raise ValueError("BatchedSheepdogVecEnv supports obs_mode='vector' only")
        if cfg.enable_wolves:
            raise ValueError("BatchedSheepdogVecEnv does not support wolves")
        self.cfg = cfg
        self.B = int(num_envs)
        self.N = int(cfg.n_sheep)
        self.fp = cfg.flock
        self.render_mode = None

        # --- static geometry (shared by every world) ----------------------
        self.bounds = np.array([cfg.width, cfg.height], dtype=np.float32)
        self.pen = np.array([cfg.pen_frac[0] * cfg.width, cfg.pen_frac[1] * cfg.height,
                             cfg.pen_frac[2] * cfg.width, cfg.pen_frac[3] * cfg.height],
                            dtype=np.float32)
        self.pen_center = np.array([(self.pen[0] + self.pen[2]) / 2,
                                    (self.pen[1] + self.pen[3]) / 2], dtype=np.float32)
        self.walls, self.opening_point, self.opening_outer = build_pen_walls(
            self.pen, cfg.pen_opening_side, cfg.pen_opening_center, cfg.pen_opening_width)
        self.fulcrum = (self.opening_point + self.opening_outer * cfg.fulcrum_dist).astype(np.float32)
        x0, y0, x1, y1 = [float(v) for v in self.pen]
        self.dog_walls = np.array([[x0, y0, x1, y0], [x1, y0, x1, y1],
                                   [x0, y1, x1, y1], [x0, y0, x0, y1]], dtype=np.float32)
        gc = self.opening_point[1] if cfg.pen_opening_side in ("left", "right") else self.opening_point[0]
        self._gap_lo = float(gc - cfg.pen_opening_width / 2)
        self._gap_hi = float(gc + cfg.pen_opening_width / 2)
        self.diag = float(np.hypot(cfg.width, cfg.height))

        # --- domain randomization ----------------------------------------
        self._dr_active = cfg.dr_active_range
        self._dr_spawn = cfg.dr_spawn_x_range
        self._dr_gap = cfg.dr_gap_range
        # When the gate width is randomized, each world carries its own fence.
        self._per_world_walls = self._dr_gap is not None
        if self._per_world_walls:
            self._setup_pw_geometry(float(gc))

        # --- spaces --------------------------------------------------------
        self.M = cfg.n_bushes + cfg.n_rocks            # obstacle count (per world)
        m = self.M
        self._obs_dim = 5 + 4 + 3 + 2 + self.N * 5 + m * 3
        obs_space = spaces.Box(-1.0, 1.0, shape=(self._obs_dim,), dtype=np.float32)
        act_space = spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32)
        super().__init__(self.B, obs_space, act_space)

        # --- curriculum knobs (applied at each world's next reset) ---------
        self._pending_active = int(cfg.n_active) if cfg.n_active is not None else self.N
        self._pending_spawn_x = cfg.spawn_x_frac

        self.rng = np.random.default_rng(seed)
        self._alloc()
        self._actions = None
        self.reset()

    # --------------------------------------------------- per-world gate setup
    def _setup_pw_geometry(self, gc):
        """Precompute the fixed parts of the fence so per-episode gate widths only
        need to rebuild the two segments around the opening."""
        x0, y0, x1, y1 = [float(v) for v in self.pen]
        side = self.cfg.pen_opening_side
        if side == "left":
            es, ee, axis = (x0, y0), (x0, y1), 1
            others = [[x0, y0, x1, y0], [x0, y1, x1, y1], [x1, y0, x1, y1]]
        elif side == "right":
            es, ee, axis = (x1, y0), (x1, y1), 1
            others = [[x0, y0, x1, y0], [x0, y1, x1, y1], [x0, y0, x0, y1]]
        elif side == "top":
            es, ee, axis = (x0, y0), (x1, y0), 0
            others = [[x0, y1, x1, y1], [x0, y0, x0, y1], [x1, y0, x1, y1]]
        else:  # bottom
            es, ee, axis = (x0, y1), (x1, y1), 0
            others = [[x0, y0, x1, y0], [x0, y0, x0, y1], [x1, y0, x1, y1]]
        self._pw_es = np.array(es, np.float32)
        self._pw_ee = np.array(ee, np.float32)
        self._pw_axis = axis
        self._pw_gc = float(gc)
        self._pw_others = np.array(others, np.float32)        # (3,4)

    def _build_walls_pw(self, widths):
        """Build per-world walls (k,5,4) + gap intervals for the given widths (k,)."""
        k = len(widths)
        ax, gc = self._pw_axis, self._pw_gc
        half = np.asarray(widths, np.float32) / 2.0
        es = np.tile(self._pw_es, (k, 1))
        ee = np.tile(self._pw_ee, (k, 1))
        lo_pt = es.copy(); lo_pt[:, ax] = gc - half
        hi_pt = es.copy(); hi_pt[:, ax] = gc + half
        seg_lo = np.concatenate([es, lo_pt], axis=1)          # (k,4): es -> lo_pt
        seg_hi = np.concatenate([hi_pt, ee], axis=1)          # (k,4): hi_pt -> ee
        others = np.tile(self._pw_others, (k, 1, 1))          # (k,3,4)
        walls = np.concatenate([seg_lo[:, None, :], seg_hi[:, None, :], others], axis=1)
        return walls.astype(np.float32), gc - half, gc + half

    # ------------------------------------------------------------------ state
    def _alloc(self):
        B, N = self.B, self.N
        self.pos = np.zeros((B, N, 2), np.float32)
        self.heading = np.zeros((B, N, 2), np.float32)
        self.vel = np.zeros((B, N, 2), np.float32)
        self.penned = np.zeros((B, N), bool)
        self.alive = np.zeros((B, N), bool)
        self.present = np.zeros((B, N), bool)
        self.staged = np.zeros((B, N), bool)
        self.active_n = np.full(B, self._pending_active, np.int64)
        self.obstacles = np.zeros((B, self.M, 3), np.float32)   # [x, y, radius] per world
        self.obstacle_kinds = np.zeros((B, self.M), np.int32)
        if self._per_world_walls:
            w0, glo, ghi = self._build_walls_pw(np.full(B, self.cfg.pen_opening_width))
            self.walls_b = w0                       # (B,5,4)
            self.gap_lo, self.gap_hi = glo.astype(np.float32), ghi.astype(np.float32)
            self.gap_w = np.full(B, self.cfg.pen_opening_width, np.float32)
        self.dog_pos = np.zeros((B, 2), np.float32)
        self.dog_vel = np.zeros((B, 2), np.float32)
        self.dog_heading = np.tile(np.array([1.0, 0.0], np.float32), (B, 1))
        self.t = np.zeros(B, np.int64)
        self.bark_cd = np.zeros(B, np.int64)
        self.bark_timer = np.zeros(B, np.int64)
        self.ep_ret = np.zeros(B, np.float64)
        self.ep_len = np.zeros(B, np.int64)

    def _spawn(self, idx):
        """(Re)initialise the worlds in ``idx`` (array of env indices)."""
        c = self.cfg
        k = len(idx)
        if k == 0:
            return
        # spawn distance: DR range > curriculum value > random far range
        if self._dr_spawn is not None:
            sx = self.rng.uniform(self._dr_spawn[0], self._dr_spawn[1], size=k)
            cx = np.minimum(sx * c.width, self.pen[0] - 12.0).astype(np.float32)
        elif self._pending_spawn_x is not None:
            cx = np.full(k, min(self._pending_spawn_x * c.width, self.pen[0] - 12.0), np.float32)
        else:
            cx = self.rng.uniform(0.15, 0.30, size=k).astype(np.float32) * c.width
        cy = self.rng.uniform(0.35, 0.65, size=k).astype(np.float32) * c.height
        pos = np.stack([
            self.rng.normal(cx[:, None], c.spawn_spread, size=(k, self.N)),
            self.rng.normal(cy[:, None], c.spawn_spread, size=(k, self.N)),
        ], axis=2).astype(np.float32)
        pos[..., 0] = np.clip(pos[..., 0], 2, c.width - 2)
        pos[..., 1] = np.clip(pos[..., 1], 2, c.height - 2)
        self.pos[idx] = pos
        ang = self.rng.uniform(0, 2 * np.pi, size=(k, self.N))
        self.heading[idx] = np.stack([np.cos(ang), np.sin(ang)], axis=2).astype(np.float32)
        self.vel[idx] = 0.0
        self.penned[idx] = False
        self.staged[idx] = False
        # active flock size: DR range (per world) > curriculum value
        if self._dr_active is not None:
            active = self.rng.integers(int(self._dr_active[0]),
                                       int(self._dr_active[1]) + 1, size=k)
        else:
            active = np.full(k, int(self._pending_active))
        active = np.clip(active, 1, self.N)
        self.active_n[idx] = active
        present = np.arange(self.N)[None, :] < active[:, None]       # (k,N) per world
        self.present[idx] = present
        self.alive[idx] = present.copy()
        # gate width: per-world fence
        if self._per_world_walls:
            widths = self.rng.uniform(self._dr_gap[0], self._dr_gap[1], size=k)
            wls, glo, ghi = self._build_walls_pw(widths)
            self.walls_b[idx] = wls
            self.gap_lo[idx] = glo
            self.gap_hi[idx] = ghi
            self.gap_w[idx] = widths.astype(np.float32)
        # dog starts behind the flock, facing the pen
        dog = np.stack([np.maximum(2.0, cx - 0.18 * c.width), cy], axis=1).astype(np.float32)
        self.dog_pos[idx] = dog
        self.dog_vel[idx] = 0.0
        head = self.pen_center[None, :] - dog
        self.dog_heading[idx] = _unit(head).astype(np.float32)
        self.t[idx] = 0
        self.bark_cd[idx] = 0
        self.bark_timer[idx] = 0
        self.ep_ret[idx] = 0.0
        self.ep_len[idx] = 0
        if self.M > 0:
            for b in idx:
                self.obstacles[b], self.obstacle_kinds[b] = self._spawn_obstacles()

    def _spawn_obstacles(self):
        """Place M non-overlapping obstacles in the central field (one world).
        Bushes first (kind 0), then rocks (kind 1) -- matching the single env."""
        c = self.cfg
        obs, kinds, attempts = [], [], 0
        while len(obs) < self.M and attempts < 500:
            attempts += 1
            x = self.rng.uniform(0.25 * c.width, 0.75 * c.width)
            y = self.rng.uniform(0.1 * c.height, 0.9 * c.height)
            r = self.rng.uniform(*c.obstacle_radius_range)
            if any(np.hypot(x - ox, y - oy) < r + orr + 4 for ox, oy, orr in obs):
                continue
            kinds.append(0 if len(obs) < c.n_bushes else 1)
            obs.append((x, y, r))
        # if rejection stalled, pad the remainder without the spacing constraint
        while len(obs) < self.M:
            x = self.rng.uniform(0.25 * c.width, 0.75 * c.width)
            y = self.rng.uniform(0.1 * c.height, 0.9 * c.height)
            kinds.append(0 if len(obs) < c.n_bushes else 1)
            obs.append((x, y, float(np.mean(c.obstacle_radius_range))))
        return (np.array(obs, np.float32).reshape(self.M, 3),
                np.array(kinds, np.int32))

    # ------------------------------------------------------------------ VecEnv
    def reset(self):
        self._spawn(np.arange(self.B))
        return self._obs()

    def step_async(self, actions):
        self._actions = np.asarray(actions, dtype=np.float32).reshape(self.B, 3)

    def step_wait(self):
        rewards, rb = self._dynamics(self._actions)

        n_alive = (self.present & self.alive).sum(1)
        n_penned = (self.present & self.penned & self.alive).sum(1)
        frac = n_penned / np.maximum(self.active_n, 1)
        success = frac >= self.cfg.target_fraction

        terminated = np.zeros(self.B, bool)
        if self.cfg.terminate_on_target:
            terminated |= success
        terminated |= (n_alive > 0) & (n_penned == n_alive)
        terminated |= (n_alive == 0)

        self.t += 1
        truncated = (self.t >= self.cfg.max_steps) & ~terminated
        done = terminated | truncated

        final = np.where(done, self.cfg.w_final * frac, 0.0)
        rewards = rewards + final
        self.ep_ret += rewards
        self.ep_len += 1

        # spread (world units) over free sheep, for metrics
        free = self.present & self.alive & ~self.penned
        spread = self._free_spread(free)

        obs = self._obs()
        infos = []
        for b in range(self.B):
            info = {
                "t": int(self.t[b]),
                "n_penned": int(n_penned[b]),
                "n_active": int(self.active_n[b]),
                "frac_penned": float(frac[b]),
                "flock_spread": float(spread[b]),
                "is_success": bool(success[b]),
            }
            if done[b]:
                info["terminal_observation"] = obs[b].copy()
                if truncated[b]:
                    info["TimeLimit.truncated"] = True
                info["episode"] = {"r": float(self.ep_ret[b]), "l": int(self.ep_len[b])}
            infos.append(info)

        # auto-reset finished worlds (after stashing their terminal obs)
        if done.any():
            self._spawn(np.where(done)[0])
            obs = self._obs()
        return obs, rewards.astype(np.float32), done, infos

    def close(self):
        pass

    def env_method(self, method_name, *args, indices=None, **kwargs):
        n = self.B if indices is None else len(self._idx(indices))
        if method_name == "set_active":
            self._pending_active = int(max(1, min(int(args[0]), self.N)))
            return [self._pending_active] * n
        if method_name == "set_spawn_x":
            self._pending_spawn_x = float(np.clip(args[0], 0.1, 0.72))
            return [self._pending_spawn_x] * n
        raise NotImplementedError(f"env_method({method_name!r}) not supported")

    def get_attr(self, attr_name, indices=None):
        idx = self._idx(indices)
        if hasattr(self, attr_name):
            return [getattr(self, attr_name)] * len(idx)
        if attr_name == "n_active":
            return [int(self.active_n[i]) for i in idx]
        raise AttributeError(attr_name)

    def set_attr(self, attr_name, value, indices=None):
        setattr(self, attr_name, value)

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False] * len(self._idx(indices))

    def seed(self, seed=None):
        self.rng = np.random.default_rng(seed)
        return [seed] * self.B

    def _idx(self, indices):
        if indices is None:
            return list(range(self.B))
        if isinstance(indices, int):
            return [indices]
        return list(indices)

    # --------------------------------------------------------------- dynamics
    def _dynamics(self, action):
        """Advance every world one tick; return (reward (B,), breakdown dict)."""
        c = self.cfg
        # epsilon-random: per-world, occasionally pick a random move direction
        if c.dog_random_action_prob > 0.0:
            flip = self.rng.random(self.B) < c.dog_random_action_prob
            if flip.any():
                action = action.copy()
                action[flip, :2] = self.rng.uniform(-1.0, 1.0, size=(int(flip.sum()), 2))
        # ---- dog ----------------------------------------------------------
        if c.action_mode == "polar":
            speed = (np.clip(action[:, 0], -1, 1) + 1.0) * 0.5 * c.dog_max_speed
            theta = np.clip(action[:, 1], -1, 1) * np.pi
            self.dog_heading = np.stack([np.cos(theta), np.sin(theta)], axis=1).astype(np.float32)
            self.dog_vel = self.dog_heading * speed[:, None]
        else:
            move = action[:, :2].copy()
            mag = np.linalg.norm(move, axis=1)
            big = mag > 1.0
            move[big] /= mag[big, None]
            self.dog_vel = move * c.dog_max_speed
            sp = np.linalg.norm(self.dog_vel, axis=1)
            moving = sp > 1e-6
            self.dog_heading[moving] = (self.dog_vel[moving] / sp[moving, None]).astype(np.float32)
        new_dog = self.dog_pos + self.dog_vel
        new_dog[:, 0] = np.clip(new_dog[:, 0], 0, c.width)
        new_dog[:, 1] = np.clip(new_dog[:, 1], 0, c.height)
        new_dog = self._avoid_obstacles_dog(new_dog)
        fence = self.dog_walls if c.dog_blocked_from_pen else self.walls
        new_dog = resolve_walls(self.dog_pos, new_dog, fence, slide=True)
        if c.dog_blocked_from_pen:
            new_dog = contain_pen(self.dog_pos, new_dog, self.pen, c.pen_opening_side,
                                  self._gap_lo, self._gap_hi, allow_opening=False)
            new_dog = self._eject_dog(new_dog)
        self.dog_pos = new_dog.astype(np.float32)

        # ---- bark timers (per world) -------------------------------------
        want = (action[:, 2] > 0.0) if c.enable_bark else np.zeros(self.B, bool)
        self.bark_timer = np.maximum(self.bark_timer - 1, 0)
        self.bark_cd = np.maximum(self.bark_cd - 1, 0)
        fire = want & (self.bark_cd == 0)
        self.bark_timer[fire] = c.bark_duration
        self.bark_cd[fire] = c.bark_cooldown
        barking = self.bark_timer > 0                      # (B,)

        # ---- flock --------------------------------------------------------
        pre = self.pos.copy()
        self._flock_step(barking)
        if self._per_world_walls:
            self.pos = _contain_pen_pw(pre, self.pos, self.pen, c.pen_opening_side,
                                       self.gap_lo, self.gap_hi)
        else:
            flat1 = contain_pen(pre.reshape(-1, 2), self.pos.reshape(-1, 2), self.pen,
                                c.pen_opening_side, self._gap_lo, self._gap_hi,
                                allow_opening=True)
            self.pos = flat1.reshape(self.B, self.N, 2)
        self.vel = (self.pos - pre).astype(np.float32)

        # ---- capture ------------------------------------------------------
        newly_penned = self._update_pen()

        return self._reward(newly_penned)

    def _flock_step(self, barking):
        p = self.fp
        B, N = self.B, self.N
        mask = self.present & self.alive & ~self.penned          # (B,N) active
        pos = self.pos
        to_dog = pos - self.dog_pos[:, None, :]                   # (B,N,2)
        dist_dog = np.linalg.norm(to_dog, axis=2)                 # (B,N)

        near_bark = barking[:, None] & (dist_dog < p.bark_radius)
        det_radius = np.where(near_bark, max(p.r_s, p.bark_radius), p.r_s)
        repulse_dog = np.where(near_bark, p.rho_s * p.bark_repulsion_mult, p.rho_s)
        delta = np.where(near_bark, p.delta * p.bark_speed_mult, p.delta)
        aware = mask & (dist_dog < det_radius)

        # pairwise distances within each world (B,N,N)
        diff = pos[:, :, None, :] - pos[:, None, :, :]           # (B,N,N,2): i - j
        dd = np.linalg.norm(diff, axis=3)                        # (B,N,N)
        eye = np.eye(N, dtype=bool)[None]
        valid_j = mask[:, None, :] & ~eye                        # neighbour j active & not self

        # --- cohesion: pull to local centre of mass of k nearest active ----
        coh = np.zeros((B, N, 2), np.float32)
        K = min(p.n_neighbours, N - 1)
        if K >= 1:
            Dn = np.where(valid_j, dd, np.inf)                   # (B,N,N)
            nn = np.argpartition(Dn, K - 1, axis=2)[:, :, :K]    # (B,N,K) nearest
            nb_pos = pos[np.arange(B)[:, None, None], nn]        # (B,N,K,2)
            nb_d = np.take_along_axis(Dn, nn, axis=2)            # (B,N,K)
            good = np.isfinite(nb_d)                             # real neighbours
            cnt = good.sum(2, keepdims=True)                    # (B,N,1)
            lcm = (nb_pos * good[..., None]).sum(2) / np.maximum(cnt, 1)
            coh = _unit(lcm - pos)
            coh[(cnt[..., 0] == 0)] = 0.0

        # --- separation: push from neighbours closer than r_a --------------
        too_close = valid_j & (dd < p.r_a)                      # (B,N,N)
        contrib = _unit(diff) * too_close[..., None]
        sep = _unit(contrib.sum(2))                             # (B,N,2)

        push_dir = _unit(to_dog)                                # radial flee

        # --- obstacle repulsion (per world, B,N,M) -------------------------
        obj = np.zeros((B, N, 2), np.float32)
        if self.M:
            ov = pos[:, :, None, :] - self.obstacles[:, None, :, :2]      # (B,N,M,2)
            od = np.linalg.norm(ov, axis=3)                              # (B,N,M)
            reach = self.obstacles[:, :, 2] + p.obstacle_margin          # (B,M)
            reach = reach[:, None, :]                                    # (B,1,M)
            close = od < reach
            strength = np.clip((reach - od) / np.maximum(reach, 1e-6), 0, 1)
            push = _unit(ov) * (strength * close)[..., None]
            obj = _unit(push.sum(2))

        # --- pen-fence repulsion -------------------------------------------
        if self._per_world_walls:
            wall = _unit(_wall_repulsion_pw(pos, self.walls_b, p.wall_margin))
        else:
            wall = wall_repulsion(pos.reshape(-1, 2), self.walls, p.wall_margin)
            wall = _unit(wall).reshape(B, N, 2)

        ang = self.rng.uniform(0, 2 * np.pi, size=(B, N))
        noise = np.stack([np.cos(ang), np.sin(ang)], axis=2).astype(np.float32)

        force = (p.inertia * self.heading
                 + p.c * coh
                 + p.rho_a * sep
                 + repulse_dog[..., None] * push_dir * aware[..., None]
                 + p.rho_o * obj
                 + p.rho_w * wall
                 + p.noise * noise)
        new_heading = _unit(force)

        graze = mask & ~aware
        graze_roll = self.rng.random((B, N)) < p.graze_prob
        step_len = np.where(aware, delta, 0.0).astype(np.float32)
        step_len = np.where(graze & graze_roll, p.graze_step, step_len)
        disp = new_heading * step_len[..., None]
        if barking.any():
            shove = (dist_dog < p.bark_radius) & mask & barking[:, None]
            disp += push_dir * (shove[..., None] * p.bark_impulse)

        new_pos = pos + disp
        new_pos[..., 0] = np.clip(new_pos[..., 0], 0, self.bounds[0])
        new_pos[..., 1] = np.clip(new_pos[..., 1], 0, self.bounds[1])
        if self.M:
            new_pos = self._push_out_of_circles(new_pos)
        if self._per_world_walls:
            new_pos = _resolve_walls_pw(pos, new_pos, self.walls_b)
        else:
            new_pos = resolve_walls(pos.reshape(-1, 2), new_pos.reshape(-1, 2),
                                    self.walls, slide=True).reshape(B, N, 2)

        m3 = mask[..., None]
        self.pos = np.where(m3, new_pos, pos).astype(np.float32)
        self.heading = np.where(m3, new_heading, self.heading).astype(np.float32)
        # self.vel is recomputed in _dynamics() after pen containment.

    def _avoid_obstacles_dog(self, dog):
        """Pop each world's dog just outside any obstacle it would enter."""
        if self.M == 0:
            return dog
        out = dog.copy()
        for j in range(self.M):
            oc = self.obstacles[:, j, :2]
            orad = self.obstacles[:, j, 2]
            v = out - oc
            d = np.linalg.norm(v, axis=1)
            inside = d < orad + 1.0
            if inside.any():
                u = v[inside] / np.maximum(d[inside, None], 1e-6)
                out[inside] = oc[inside] + u * (orad[inside, None] + 1.0)
        return out

    def _push_out_of_circles(self, pos):
        """Push any sheep sitting inside an obstacle out to its rim (per world)."""
        out = pos
        for j in range(self.M):
            oc = self.obstacles[:, j, :2]                       # (B,2)
            orad = self.obstacles[:, j, 2]                      # (B,)
            v = out - oc[:, None, :]                            # (B,N,2)
            d = np.linalg.norm(v, axis=2)                       # (B,N)
            inside = d < orad[:, None]
            if inside.any():
                u = v / np.maximum(d[..., None], 1e-6)
                pushed = oc[:, None, :] + u * orad[:, None, None]
                out = np.where(inside[..., None], pushed, out)
        return out.astype(np.float32)

    def _update_pen(self):
        x0, y0, x1, y1 = self.pen
        inside = ((self.pos[..., 0] >= x0) & (self.pos[..., 0] <= x1) &
                  (self.pos[..., 1] >= y0) & (self.pos[..., 1] <= y1) & self.alive)
        newly = inside & ~self.penned
        if self.cfg.pen_capture:
            self.penned |= inside
        else:
            self.penned = inside
        return newly.sum(1)                                     # (B,)

    def _eject_dog(self, dog):
        x0, y0, x1, y1 = self.pen
        inside = (dog[:, 0] >= x0) & (dog[:, 0] <= x1) & (dog[:, 1] >= y0) & (dog[:, 1] <= y1)
        if not inside.any():
            return dog
        out = dog.copy()
        eps = 0.5
        for b in np.where(inside)[0]:
            px, py = dog[b]
            cands = {x0 - eps - px: (x0 - eps, py), x1 + eps - px: (x1 + eps, py),
                     y0 - eps - py: (px, y0 - eps), y1 + eps - py: (px, y1 + eps)}
            out[b] = cands[min(cands, key=lambda d: abs(d))]
        return out

    # ----------------------------------------------------------------- reward
    def _free_spread(self, free):
        cnt = free.sum(1)                                       # (B,)
        denom = np.maximum(cnt, 1)[:, None]
        cen = (self.pos * free[..., None]).sum(1) / denom       # (B,2)
        d = np.linalg.norm(self.pos - cen[:, None, :], axis=2)  # (B,N)
        spread = (d * free).sum(1) / np.maximum(cnt, 1)
        return np.where(cnt > 1, spread, 0.0)                   # world units

    def _reward(self, newly_penned):
        c = self.cfg
        free = self.present & self.alive & ~self.penned         # (B,N)
        cnt = free.sum(1)                                       # (B,)
        denom = np.maximum(cnt, 1)

        d_gate = np.linalg.norm(self.pos - self.opening_point[None, None, :], axis=2)
        mean_dist = (d_gate * free).sum(1) / denom / self.diag

        # behind the pen (opening faces the field): which side is "behind"
        x0, y0, x1, y1 = self.pen
        side = c.pen_opening_side
        if side == "left":
            behind = self.pos[..., 0] > x1
        elif side == "right":
            behind = self.pos[..., 0] < x0
        elif side == "top":
            behind = self.pos[..., 1] > y1
        else:
            behind = self.pos[..., 1] < y0
        frac_behind = (behind & free).sum(1) / denom

        spread_world = self._free_spread(free)
        spread = spread_world / self.diag
        gate_relax = np.clip((mean_dist - 0.03) / (0.15 - 0.03), 0.0, 1.0)
        spread = spread * gate_relax
        spread = np.where(cnt > 1, spread, 0.0)

        # fulcrum staging (one-time, latched), polar score in the gate cone
        to_gate = -self.opening_outer
        v = self.pos - self.fulcrum[None, None, :]
        along = v @ to_gate                                    # (B,N)
        perp = np.linalg.norm(v - along[..., None] * to_gate[None, None, :], axis=2)
        polar = np.sqrt(along ** 2 + (1.8 * perp) ** 2) / 50.0
        in_cone = (1.0 - polar) > 0.5
        newly_staged = in_cone & free & ~self.staged
        self.staged |= (in_cone & free)
        n_staged = newly_staged.sum(1)

        rb = {
            "pen_enter": c.w_pen_enter * newly_penned,
            "fulcrum": c.w_fulcrum * n_staged,
            "entrance": -c.w_entrance * mean_dist,
            "back": -c.w_back * frac_behind,
            "cohesion": -c.w_cohesion * spread,
        }
        reward = sum(rb.values())
        return reward.astype(np.float64), rb

    # ----------------------------------------------------------------- obs
    def _obs(self):
        c = self.cfg
        b = self.bounds
        diag = self.diag
        B, N = self.B, self.N

        def npos(p):
            return (p / b) * 2.0 - 1.0

        dog = npos(self.dog_pos)                                # (B,2)
        dvel = np.clip(self.dog_vel / c.dog_max_speed, -1, 1)
        bark_ready = np.where(self.bark_cd[:, None] == 0, 1.0, -1.0)
        pen_lo = np.tile(npos(self.pen[:2]), (B, 1))
        pen_hi = np.tile(npos(self.pen[2:]), (B, 1))
        open_rel = np.clip((self.opening_point[None, :] - self.dog_pos) / diag, -1, 1)
        if self._per_world_walls:
            open_w = np.minimum(self.gap_w[:, None] / diag, 1.0).astype(np.float32)
        else:
            open_w = np.full((B, 1), min(c.pen_opening_width / diag, 1.0), np.float32)
        present = self.present
        frac_pen = (present & self.penned & self.alive).sum(1) / np.maximum(self.active_n, 1)
        tfrac = np.stack([2.0 * self.t / c.max_steps - 1.0, 2.0 * frac_pen - 1.0], axis=1)

        rel = np.clip((self.pos - self.dog_pos[:, None, :]) / diag, -1, 1)
        vel = np.clip(self.vel / max(c.flock.delta * c.flock.bark_speed_mult, 1e-6), -1, 1)
        status = np.where(~self.alive, -1.0, np.where(self.penned, 1.0, 0.0))[..., None]
        free = present & self.alive & ~self.penned
        rel = rel * free[..., None]
        vel = vel * free[..., None]
        status[~present, 0] = -1.0
        sheep = np.concatenate([rel, vel, status], axis=2).reshape(B, N * 5)

        parts = [dog, dvel, bark_ready, pen_lo, pen_hi, open_rel, open_w, tfrac, sheep]
        if self.M:
            orel = np.clip((self.obstacles[:, :, :2] - self.dog_pos[:, None, :]) / diag, -1, 1)
            orad = (self.obstacles[:, :, 2] / diag)[..., None]
            obuf = np.concatenate([orel, orad], axis=2).reshape(B, self.M * 3)
            parts.append(obuf)
        flat = np.concatenate(parts, axis=1)
        return np.clip(flat, -1, 1).astype(np.float32)
