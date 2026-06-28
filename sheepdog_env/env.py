"""Sheepdog Herding — a Gymnasium environment for RL.

A faithful re-creation of the "Shepherd's Dog" browser game as a single-agent
reinforcement-learning environment.

The agent controls a **sheepdog** with limited speed.  It can **bark**, which
makes nearby sheep bolt away faster.  Sheep behave as a Strombom flock (cohesion
+ separation + flee-from-dog + obstacle avoidance).  The goal is to herd at
least ``target_fraction`` (default 80%) of the flock into the **pen** before
nightfall.  Optionally, **wolves** appear at dusk and pick off stray sheep.

Action space (Box, shape (3,), all in [-1, 1]); interpretation set by
``action_mode``:
    "polar" (default):
        a[0] : speed   in [-1,1] -> [0, dog_max_speed]
        a[1] : heading in [-1,1] -> [-pi, pi]   (also the dog's orientation)
        a[2] : bark when > 0 (subject to a cooldown)
    "cartesian":
        a[0], a[1] : velocity vector (magnitude clamped to the dog's max speed)
        a[2]       : bark when > 0
When the dog barks, nearby sheep are driven along the dog's *orientation* (its
heading), so the heading both steers the dog and aims the herd.

Observation space:
    obs_mode="vector" -> flat Box of dog state, flock state and obstacles
    obs_mode="pixel"  -> (H, W, 3) uint8 top-down image

Reward (all weights configurable):
    + potential-based shaping for moving the un-penned flock toward the pen
    + bonus each time a sheep newly enters the pen
    + large bonus on success (target fraction penned)
    - small per-step time cost
    - penalty per sheep lost to a wolf
    - small control / bark cost
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except Exception as exc:  # pragma: no cover
    raise ImportError(
        "gymnasium is required: pip install gymnasium"
    ) from exc

from .flocking import Flock, FlockParams
from .geometry import build_pen_walls, resolve_walls, contain_pen
from .rendering import Renderer


@dataclass
class EnvConfig:
    # --- world ------------------------------------------------------------
    width: float = 150.0
    height: float = 100.0
    n_sheep: int = 40                   # observation CAPACITY (max flock size)
    # Active flock size for curriculum learning. None -> all n_sheep are active.
    # If set (<= n_sheep), only this many sheep spawn; the rest are masked out as
    # "absent" so the observation dimension stays fixed while the effective flock
    # size changes. Win/accounting are relative to the active count.
    n_active: Optional[int] = None
    target_fraction: float = 0.8        # fraction penned to count as a "success"
    terminate_on_target: bool = False   # if True, end the episode the instant
                                        # target_fraction is reached. Default
                                        # False so every extra sheep keeps
                                        # paying (episode runs until all
                                        # survivors are penned or it times out).
    max_steps: int = 1200               # one "day"; episode horizon
    # Where the flock spawns, as a fraction of the field width. None -> random
    # in the far range (0.15-0.30); set a value (or use the spawn curriculum) to
    # fix it. Larger = nearer the pen = easier.
    spawn_x_frac: Optional[float] = None
    # Std (world units) of the initial flock cluster. Small = the herd starts
    # tightly together.
    spawn_spread: float = 4.0

    # --- dog --------------------------------------------------------------
    dog_max_speed: float = 2.6          # world units / step (much faster than sheep)
    # Epsilon-random exploration: probability per step that the dog ignores the
    # policy and moves in a random direction (its bark choice is left untouched).
    # 0 = always follow the policy.
    dog_random_action_prob: float = 0.0
    enable_bark: bool = False           # if False, the dog never barks (herds by presence only)
    bark_cooldown: int = 18             # steps between barks
    bark_duration: int = 4              # steps a bark stays "active"
    # How the 3-D action is interpreted (shape is Box(3,) in [-1,1] either way):
    #   "polar"     -> [speed, heading, bark]  (speed in [0,max], heading in [-pi,pi])
    #   "cartesian" -> [vel_x, vel_y, bark]    (velocity vector, magnitude clamped)
    action_mode: str = "polar"

    # --- pen --------------------------------------------------------------
    # Pen rectangle as fractions of the field (right side, like the game).
    pen_frac: tuple = (0.80, 0.30, 0.96, 0.70)
    pen_capture: bool = True            # freeze sheep once inside the pen
    # The pen is a fenced box with a single opening of fixed width. Sheep can
    # only get in (or out) through the gap.
    pen_opening_side: str = "left"      # which wall holds the gap (faces field)
    pen_opening_center: float = 0.5     # gap centre along that wall, as a fraction
    pen_opening_width: float = 14.0     # fixed gap width, in world units
    # The dog works the flock from the field; it is fenced out of the pen
    # entirely (it cannot pass through the opening — only sheep can).
    dog_blocked_from_pen: bool = True

    # --- domain randomization (per-episode) ------------------------------
    # When set, each episode samples the value uniformly from the (lo, hi) range
    # at reset, for robustness to flock size / start distance / gate width.
    # None -> use the fixed value above. The opening CENTRE stays fixed; only the
    # WIDTH varies, so the gate/fulcrum geometry is otherwise unchanged.
    dr_active_range: Optional[tuple] = None     # (min, max) active sheep
    dr_spawn_x_range: Optional[tuple] = None    # (min, max) spawn x fraction
    dr_gap_range: Optional[tuple] = None        # (min, max) opening width

    # --- obstacles --------------------------------------------------------
    n_bushes: int = 0                   # obstacles removed for now (set >0 to re-add)
    n_rocks: int = 0
    obstacle_radius_range: tuple = (2.5, 4.5)

    # --- wolves (dusk hazard) --------------------------------------------
    enable_wolves: bool = False
    dusk_fraction: float = 0.7          # time fraction when wolves appear
    n_wolves: int = 2
    wolf_speed: float = 1.3
    wolf_catch_dist: float = 2.5
    wolf_flee_from_dog: float = 25.0    # wolves avoid the dog within this range

    # --- observation / render --------------------------------------------
    obs_mode: str = "vector"            # "vector" | "pixel"
    pixel_shape: tuple = (84, 84)       # (H, W) for pixel obs
    render_size: tuple = (672, 448)     # (W, H) for rgb_array / human

    # --- reward (dense, simple) ------------------------------------------
    # Strong reward for sheep entering / being home, a per-step pull toward the
    # entrance (the gap), and a penalty for sheep driven behind the pen.
    w_pen_enter: float = 10.0           # one-off, per sheep that enters the pen
    w_final: float = 100.0              # terminal: w_final * (fraction penned at end)
    w_entrance: float = 0.08            # per-step cost ~ mean dist of free sheep to entrance
    w_back: float = 0.15                # per-step penalty ~ fraction of free sheep behind the pen
    w_cohesion: float = 1.5             # per-step penalty ~ how spread out the free flock is (keep together!)
    # One-time bonus per sheep the first time it reaches the staging cone in
    # front of the gate (a "fulcrum" point just outside the opening), scored by
    # polar distance. Event reward (NOT per-step) so winning never forfeits it.
    w_fulcrum: float = 3.0              # bonus per sheep staged in front of the gate
    fulcrum_dist: float = 12.0          # how far in front of the gate the fulcrum sits (world units)

    flock: FlockParams = field(default_factory=FlockParams)


class SheepdogHerdingEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array", "human"], "render_fps": 30}

    def __init__(self, config: Optional[EnvConfig] = None,
                 render_mode: Optional[str] = None, **overrides):
        super().__init__()
        self.cfg = config or EnvConfig()
        # Allow quick keyword overrides, e.g. SheepdogHerdingEnv(n_sheep=20).
        for k, v in overrides.items():
            if not hasattr(self.cfg, k):
                raise TypeError(f"Unknown config field: {k!r}")
            setattr(self.cfg, k, v)

        self.render_mode = render_mode
        self.bounds = np.array([self.cfg.width, self.cfg.height], dtype=np.float32)

        c = self.cfg
        self.pen = np.array([
            c.pen_frac[0] * c.width, c.pen_frac[1] * c.height,
            c.pen_frac[2] * c.width, c.pen_frac[3] * c.height,
        ], dtype=np.float32)
        self.pen_center = np.array([
            (self.pen[0] + self.pen[2]) / 2,
            (self.pen[1] + self.pen[3]) / 2,
        ], dtype=np.float32)

        # Fenced pen with a single fixed-width opening (used by the sheep).
        self.walls, self.opening_point, self.opening_outer = build_pen_walls(
            self.pen, c.pen_opening_side, c.pen_opening_center, c.pen_opening_width)

        # Fulcrum: a staging point just outside the gate (in the field). Sheep are
        # rewarded for gathering here, lined up to be funnelled through.
        self.fulcrum = (self.opening_point + self.opening_outer * c.fulcrum_dist
                        ).astype(np.float32)

        # The dog gets the *full* perimeter (no gap) so it is fenced out of the
        # pen entirely — it cannot follow sheep in through the opening.
        x0, y0, x1, y1 = [float(v) for v in self.pen]
        self.dog_walls = np.array([
            [x0, y0, x1, y0],   # top
            [x1, y0, x1, y1],   # right
            [x0, y1, x1, y1],   # bottom
            [x0, y0, x0, y1],   # left (solid — no opening for the dog)
        ], dtype=np.float32)

        # Gap interval along the opening edge (used by the containment guard).
        if c.pen_opening_side in ("left", "right"):
            gc = self.opening_point[1]
        else:
            gc = self.opening_point[0]
        self._gap_lo = float(gc - c.pen_opening_width / 2)
        self._gap_hi = float(gc + c.pen_opening_width / 2)

        self.flock = Flock(c.n_sheep, c.flock, np.random.default_rng())

        # --- action space -------------------------------------------------
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,),
                                       dtype=np.float32)

        # --- observation space -------------------------------------------
        if c.obs_mode == "vector":
            self._obs_dim = self._vector_obs_dim()
            self.observation_space = spaces.Box(
                low=-1.0, high=1.0, shape=(self._obs_dim,), dtype=np.float32)
        elif c.obs_mode == "pixel":
            h, w = c.pixel_shape
            self.observation_space = spaces.Box(
                low=0, high=255, shape=(h, w, 3), dtype=np.uint8)
        else:
            raise ValueError(f"obs_mode must be 'vector' or 'pixel', got {c.obs_mode!r}")

        # Renderers (lazy for the big one).
        self._obs_renderer = Renderer(c.pixel_shape[1], c.pixel_shape[0], self.bounds) \
            if c.obs_mode == "pixel" else None
        self._big_renderer = None

        # Episode state placeholders.
        self.dog_pos = np.zeros(2, dtype=np.float32)
        self.dog_vel = np.zeros(2, dtype=np.float32)
        self.dog_heading = np.array([1.0, 0.0], dtype=np.float32)
        self.obstacles = np.zeros((0, 3), dtype=np.float32)
        self.obstacle_kinds = np.zeros((0,), dtype=np.int32)
        self.wolves = np.zeros((0, 2), dtype=np.float32)
        self.t = 0
        self._bark_cd = 0
        self._bark_timer = 0
        self._staged = np.zeros(self.cfg.n_sheep, dtype=bool)  # sheep that reached the fulcrum
        self.n_active = (int(self.cfg.n_active) if self.cfg.n_active is not None
                         else self.cfg.n_sheep)

    def _rebuild_pen(self, width: float):
        """Rebuild the fence for a new opening WIDTH (centre unchanged). Used by
        per-episode gate-width randomization."""
        c = self.cfg
        c.pen_opening_width = float(width)
        self.walls, self.opening_point, self.opening_outer = build_pen_walls(
            self.pen, c.pen_opening_side, c.pen_opening_center, c.pen_opening_width)
        self.fulcrum = (self.opening_point + self.opening_outer * c.fulcrum_dist
                        ).astype(np.float32)
        gc = (self.opening_point[1] if c.pen_opening_side in ("left", "right")
              else self.opening_point[0])
        self._gap_lo = float(gc - c.pen_opening_width / 2)
        self._gap_hi = float(gc + c.pen_opening_width / 2)

    def set_active(self, n_active: int):
        """Set the active flock size for *future* episodes (curriculum hook).

        Capacity (the observation size) is fixed at ``n_sheep``; this only
        changes how many sheep spawn next reset. Returns the clamped value.
        """
        self.cfg.n_active = int(max(1, min(int(n_active), self.cfg.n_sheep)))
        return self.cfg.n_active

    def set_spawn_x(self, frac: float):
        """Set where the flock spawns (fraction of width) for future episodes.
        Larger = nearer the pen = easier. Curriculum hook."""
        self.cfg.spawn_x_frac = float(np.clip(frac, 0.1, 0.72))
        return self.cfg.spawn_x_frac

    # ------------------------------------------------------------- spaces
    def _vector_obs_dim(self) -> int:
        # dog(5) + pen(4) + opening(3) + time/frac(2) + sheep(n*5) + obstacles(m*3)
        m = self.cfg.n_bushes + self.cfg.n_rocks
        return 5 + 4 + 3 + 2 + self.cfg.n_sheep * 5 + m * 3

    # -------------------------------------------------------------- reset
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        rng = self.np_random
        self.flock.rng = rng
        c = self.cfg

        # --- domain randomization (per-episode) --------------------------
        # Sample the gate width (rebuilding the fence), spawn distance and flock
        # size from their ranges. Centre of the opening is unchanged, so only the
        # fence segments + gap interval move.
        if c.dr_gap_range is not None:
            self._rebuild_pen(float(rng.uniform(*c.dr_gap_range)))
        ep_spawn_x = c.spawn_x_frac
        if c.dr_spawn_x_range is not None:
            ep_spawn_x = float(rng.uniform(*c.dr_spawn_x_range))
        ep_active = c.n_active
        if c.dr_active_range is not None:
            ep_active = int(rng.integers(int(c.dr_active_range[0]),
                                         int(c.dr_active_range[1]) + 1))

        # Obstacles, kept clear of the pen and the spawn strip.
        obs_list, kinds = [], []
        n_obs = c.n_bushes + c.n_rocks
        attempts = 0
        while len(obs_list) < n_obs and attempts < 500:
            attempts += 1
            x = rng.uniform(0.25 * c.width, 0.75 * c.width)
            y = rng.uniform(0.1 * c.height, 0.9 * c.height)
            r = rng.uniform(*c.obstacle_radius_range)
            if any(np.hypot(x - ox, y - oy) < r + orr + 4 for ox, oy, orr in obs_list):
                continue
            kinds.append(0 if len(obs_list) < c.n_bushes else 1)
            obs_list.append((x, y, r))
        self.obstacles = np.array(obs_list, dtype=np.float32).reshape(-1, 3)
        self.obstacle_kinds = np.array(kinds, dtype=np.int32)

        # Flock spawns at a random position but always as a TIGHT cluster
        # (spawn_spread, world units). spawn_x_frac, if set, fixes the x position
        # (used by the optional spawn curriculum); otherwise x is random.
        if ep_spawn_x is not None:
            cx = min(ep_spawn_x * c.width, self.pen[0] - 12.0)
        else:
            cx = rng.uniform(0.15, 0.30) * c.width
        cy = rng.uniform(0.35, 0.65) * c.height
        spread = c.spawn_spread
        pos = np.stack([
            rng.normal(cx, spread, size=c.n_sheep),
            rng.normal(cy, spread, size=c.n_sheep),
        ], axis=1).astype(np.float32)
        pos[:, 0] = np.clip(pos[:, 0], 2, c.width - 2)
        pos[:, 1] = np.clip(pos[:, 1], 2, c.height - 2)
        # Active flock size (curriculum / DR). Capacity is c.n_sheep; the rest are
        # masked out as absent so the observation dimension never changes.
        self.n_active = int(ep_active) if ep_active is not None else c.n_sheep
        self.n_active = max(1, min(self.n_active, c.n_sheep))
        self.flock.reset(pos, n_active=self.n_active)

        # Dog starts behind (to the left of) the flock, facing the pen.
        self.dog_pos = np.array([max(2.0, cx - 0.18 * c.width), cy], dtype=np.float32)
        self.dog_vel[:] = 0.0
        head = self.pen_center - self.dog_pos
        self.dog_heading = (head / max(np.linalg.norm(head), 1e-6)).astype(np.float32)

        self.wolves = np.zeros((0, 2), dtype=np.float32)
        self.t = 0
        self._bark_cd = 0
        self._bark_timer = 0
        self._staged[:] = False

        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    # --------------------------------------------------------------- step
    def step(self, action):
        c = self.cfg
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        # Epsilon-random: occasionally override the dog's movement with a random
        # direction (bark choice kept), to inject exploration / a distractible dog.
        if c.dog_random_action_prob > 0.0 and self.np_random.random() < c.dog_random_action_prob:
            action = action.copy()
            action[:2] = self.np_random.uniform(-1.0, 1.0, size=2).astype(np.float32)
        # Barking can be disabled entirely (enable_bark=False): the dog then
        # herds purely by its presence (sheep flee proximity), never barks.
        want_bark = bool(c.enable_bark) and float(action[2]) > 0.0

        # --- move the dog (limited speed) --------------------------------
        # self.dog_heading is the dog's orientation; it both steers movement and,
        # while barking, aims the herd.
        if c.action_mode == "polar":
            # a[0] = speed in [-1,1] -> [0, max] ; a[1] = heading in [-1,1] -> [-pi, pi]
            speed = (np.clip(action[0], -1.0, 1.0) + 1.0) * 0.5 * c.dog_max_speed
            theta = np.clip(action[1], -1.0, 1.0) * np.pi
            self.dog_heading = np.array([np.cos(theta), np.sin(theta)], dtype=np.float32)
            self.dog_vel = self.dog_heading * speed
        else:  # "cartesian": a[0], a[1] = velocity vector (magnitude clamped to 1)
            move = action[:2]
            mag = np.linalg.norm(move)
            if mag > 1.0:
                move = move / mag
            self.dog_vel = move * c.dog_max_speed
            sp = float(np.linalg.norm(self.dog_vel))
            if sp > 1e-6:                       # keep last heading when standing still
                self.dog_heading = (self.dog_vel / sp).astype(np.float32)
        new_dog = self.dog_pos + self.dog_vel
        new_dog[0] = np.clip(new_dog[0], 0, c.width)
        new_dog[1] = np.clip(new_dog[1], 0, c.height)
        new_dog = self._avoid_obstacles(new_dog)
        # The dog is fenced out of the pen: block it against the full perimeter
        # (no opening) so it can't enter, then hard-clamp out of the interior as
        # a safety net against fast-step tunnelling.
        dog_fence = self.dog_walls if c.dog_blocked_from_pen else self.walls
        new_dog = resolve_walls(self.dog_pos[None, :], new_dog[None, :],
                                dog_fence, slide=True)
        if c.dog_blocked_from_pen:
            # Backstop: the dog may never enter the pen (no opening for it).
            new_dog = contain_pen(self.dog_pos[None, :], new_dog, self.pen,
                                  c.pen_opening_side, self._gap_lo, self._gap_hi,
                                  allow_opening=False)
            new_dog[0] = self._eject_from_pen(new_dog[0])
        self.dog_pos = new_dog[0].astype(np.float32)

        # --- barking ------------------------------------------------------
        barked = False
        if self._bark_timer > 0:
            self._bark_timer -= 1
        if self._bark_cd > 0:
            self._bark_cd -= 1
        if want_bark and self._bark_cd == 0:
            self._bark_timer = c.bark_duration
            self._bark_cd = c.bark_cooldown
            barked = True
        barking = self._bark_timer > 0

        # --- flock dynamics ----------------------------------------------
        pre_sheep = self.flock.pos.copy()
        self.flock.step(self.dog_pos, barking, self.obstacles, self.bounds,
                        self.walls, bark_dir=self.dog_heading)
        # Backstop: a sheep may only cross the pen boundary through the opening
        # (catches corner-slide leaks the per-wall resolver can miss).
        self.flock.pos = contain_pen(
            pre_sheep, self.flock.pos, self.pen, c.pen_opening_side,
            self._gap_lo, self._gap_hi, allow_opening=True)
        self.flock.vel = (self.flock.pos - pre_sheep).astype(np.float32)

        # --- capture sheep that reached the pen --------------------------
        newly_penned = self._update_pen()

        # --- wolves -------------------------------------------------------
        lost = 0
        if c.enable_wolves:
            lost = self._update_wolves()

        # --- reward (dense, simple) --------------------------------------
        rb = self._reward_simple(newly_penned)

        # --- termination --------------------------------------------------
        n_alive = int((self.flock.present & self.flock.alive).sum())
        n_penned = int((self.flock.present & self.flock.penned & self.flock.alive).sum())
        frac = n_penned / max(self.n_active, 1)
        terminated = False
        truncated = False
        # `success` is a logged metric -- reaching target_fraction no longer ends
        # the episode (unless terminate_on_target is set). We keep going so each
        # additional sheep keeps paying via pen_enter and a larger proportional
        # `final`. The episode ends only when every surviving sheep is home, or
        # none are left -- i.e. when no further progress is possible.
        success = frac >= c.target_fraction
        if c.terminate_on_target and success:
            terminated = True
        elif n_alive > 0 and n_penned == n_alive:
            terminated = True       # all surviving sheep are penned -> done
        elif n_alive == 0:
            terminated = True

        self.t += 1
        if self.t >= c.max_steps and not terminated:
            truncated = True

        # Terminal reward proportional to how many sheep are home at the end
        # ("more sheep in the pen at the end"). Continuous in `frac`, so every
        # extra penned sheep raises the terminal payout by w_final / n_active.
        if terminated or truncated:
            rb["final"] = c.w_final * frac

        reward = float(sum(rb.values()))

        obs = self._get_obs()
        info = self._get_info()
        info["is_success"] = success
        info["newly_penned"] = int(newly_penned)
        info["sheep_lost"] = int(lost)
        info["reward_breakdown"] = rb
        return obs, float(reward), terminated, truncated, info

    # ----------------------------------------------------------- helpers
    def _avoid_obstacles(self, p):
        for ox, oy, orad in self.obstacles:
            v = p - np.array([ox, oy], dtype=np.float32)
            d = np.linalg.norm(v)
            if d < orad + 1.0:
                u = v / max(d, 1e-6)
                p = np.array([ox, oy], dtype=np.float32) + u * (orad + 1.0)
        return p

    def _eject_from_pen(self, p):
        """Safety net: if the dog is inside the pen rect, pop it to nearest edge."""
        x0, y0, x1, y1 = self.pen
        if x0 <= p[0] <= x1 and y0 <= p[1] <= y1:
            eps = 0.5
            dists = {x0 - eps - p[0]: (x0 - eps, p[1]),   # exit left
                     x1 + eps - p[0]: (x1 + eps, p[1]),   # exit right
                     y0 - eps - p[1]: (p[0], y0 - eps),   # exit top
                     y1 + eps - p[1]: (p[0], y1 + eps)}   # exit bottom
            nearest = min(dists, key=lambda d: abs(d))
            p = np.array(dists[nearest], dtype=np.float32)
        return p

    # ------------------------------------------------------------- rewards
    def _behind_pen_mask(self):
        """Per-sheep bool: True if the sheep is on the *far* side of the pen from
        the entrance (i.e. it has been driven 'behind' the pen)."""
        pos = self.flock.pos
        x0, y0, x1, y1 = self.pen
        side = self.cfg.pen_opening_side
        if side == "left":
            return pos[:, 0] > x1
        if side == "right":
            return pos[:, 0] < x0
        if side == "top":
            return pos[:, 1] > y1
        return pos[:, 1] < y0          # bottom

    def _reward_simple(self, newly_penned):
        """Simple, dense reward: strong reward for sheep entering/being in the
        pen, a one-time bonus for staging them at the fulcrum, a per-step pull
        toward the entrance, a penalty for sheep driven behind the pen, and a
        penalty for letting the herd spread out. (Terminal ``final`` in step.)"""
        c = self.cfg
        diag = float(np.hypot(c.width, c.height))
        f = self.flock
        free = f.present & f.alive & ~f.penned
        mean_dist = 0.0
        frac_behind = 0.0
        spread = 0.0
        newly_staged = 0
        if free.any():
            fp = f.pos[free]
            d = np.linalg.norm(fp - self.opening_point[None, :], axis=1)
            mean_dist = float(np.mean(d) / diag)            # ~[0,1], 0 = at entrance
            frac_behind = float(np.mean(self._behind_pen_mask()[free]))
            if fp.shape[0] > 1:
                # mean distance of free sheep to their centroid -> how dispersed
                # the herd is. Penalize spread so the dog keeps the flock together
                # WHILE DRIVING -- but relax it as the flock nears the gate, since
                # funnelling through a narrow opening requires the flock to fan
                # out (otherwise the cohesion penalty fights the funnel and the
                # reward dips right before penning).
                spread = float(np.mean(np.linalg.norm(fp - fp.mean(0), axis=1)) / diag)
                gate_relax = float(np.clip((mean_dist - 0.03) / (0.15 - 0.03), 0.0, 1.0))
                spread *= gate_relax     # ~0 at the gate, full far from it
            # Fulcrum: ONE-TIME bonus the first time a sheep reaches the staging
            # cone in front of the gate, scored by polar distance (radial +
            # lateral, off-axis weighted worse). It is an event reward (like the
            # pen bonus), NOT a per-step stream -- a per-step positive reward
            # would make winning (which ends the episode) forfeit future reward,
            # so the agent would learn to never pen. Latched so it can't be
            # farmed by oscillating in and out of the cone.
            to_gate = -self.opening_outer
            v = fp - self.fulcrum[None, :]
            along = v @ to_gate
            perp = np.linalg.norm(v - along[:, None] * to_gate[None, :], axis=1)
            polar = np.sqrt(along ** 2 + (1.8 * perp) ** 2) / 50.0
            in_cone = (1.0 - polar) > 0.5             # well-staged sheep
            idx = np.where(free)[0][in_cone]
            newly = idx[~self._staged[idx]]
            self._staged[idx] = True
            newly_staged = int(len(newly))
        return {
            "pen_enter": c.w_pen_enter * newly_penned,
            "fulcrum": c.w_fulcrum * newly_staged,
            "entrance": -c.w_entrance * mean_dist,
            "back": -c.w_back * frac_behind,
            "cohesion": -c.w_cohesion * spread,
            "final": 0.0,
        }

    def _update_pen(self) -> int:
        pos = self.flock.pos
        inside = (
            (pos[:, 0] >= self.pen[0]) & (pos[:, 0] <= self.pen[2]) &
            (pos[:, 1] >= self.pen[1]) & (pos[:, 1] <= self.pen[3]) &
            self.flock.alive
        )
        newly = inside & ~self.flock.penned
        count = int(newly.sum())
        if self.cfg.pen_capture:
            self.flock.penned |= inside
        else:
            self.flock.penned = inside
        return count

    def _update_wolves(self) -> int:
        c = self.cfg
        if self.t < c.dusk_fraction * c.max_steps:
            return 0
        if len(self.wolves) == 0:
            # Spawn wolves at field edges.
            ang = self.np_random.uniform(0, 2 * np.pi, size=c.n_wolves)
            self.wolves = np.stack([
                np.where(np.cos(ang) > 0, c.width - 2, 2.0),
                self.np_random.uniform(0.1, 0.9, size=c.n_wolves) * c.height,
            ], axis=1).astype(np.float32)

        lost = 0
        prey_mask = self.flock.alive & ~self.flock.penned
        for w in range(len(self.wolves)):
            wp = self.wolves[w]
            # Flee the dog if it is close, else chase nearest free sheep.
            to_dog = wp - self.dog_pos
            if np.linalg.norm(to_dog) < c.wolf_flee_from_dog:
                step = to_dog / max(np.linalg.norm(to_dog), 1e-6)
            elif prey_mask.any():
                idx = np.where(prey_mask)[0]
                d = np.linalg.norm(self.flock.pos[idx] - wp[None, :], axis=1)
                tgt = idx[int(np.argmin(d))]
                step = self.flock.pos[tgt] - wp
                step = step / max(np.linalg.norm(step), 1e-6)
            else:
                step = np.zeros(2, dtype=np.float32)
            wp = wp + step * c.wolf_speed
            wp[0] = np.clip(wp[0], 0, c.width)
            wp[1] = np.clip(wp[1], 0, c.height)
            self.wolves[w] = wp

            # Eat any free sheep within catch distance.
            if prey_mask.any():
                idx = np.where(prey_mask)[0]
                d = np.linalg.norm(self.flock.pos[idx] - wp[None, :], axis=1)
                caught = idx[d < c.wolf_catch_dist]
                if caught.size:
                    self.flock.alive[caught] = False
                    prey_mask = self.flock.alive & ~self.flock.penned
                    lost += int(caught.size)
        return lost

    # --------------------------------------------------------- observation
    def _get_obs(self):
        if self.cfg.obs_mode == "pixel":
            return self._obs_renderer.render(self._render_state())
        return self._vector_obs()

    def _vector_obs(self):
        c = self.cfg
        b = self.bounds
        diag = float(np.hypot(c.width, c.height))

        def npos(p):       # normalise a position to [-1, 1]
            return (p / b) * 2.0 - 1.0

        parts = []
        # Dog: pos, vel, bark-ready flag.
        parts.append(npos(self.dog_pos))
        parts.append(np.clip(self.dog_vel / c.dog_max_speed, -1, 1))
        parts.append([1.0 if self._bark_cd == 0 else -1.0])
        # Pen rectangle (normalised corners).
        parts.append(npos(self.pen[:2]))
        parts.append(npos(self.pen[2:]))
        # Opening: gap centre (relative to dog) and half-width — tells the agent
        # where it must funnel the flock.
        parts.append(np.clip((self.opening_point - self.dog_pos) / diag, -1, 1))
        parts.append([min(self.cfg.pen_opening_width / diag, 1.0)])
        # Time fraction and penned fraction (over the active flock).
        present = self.flock.present
        frac_pen = (present & self.flock.penned & self.flock.alive).sum() / max(self.n_active, 1)
        parts.append([2.0 * self.t / c.max_steps - 1.0, 2.0 * frac_pen - 1.0])
        # Each sheep: position relative to dog, velocity, status flag.
        # status: 1 penned, 0 free, -1 dead/absent. A sheep that is "done"
        # (penned, dead, or absent) has its position/velocity ZEROED so the
        # policy doesn't keep attending to it — once a sheep is in the pen the
        # dog shouldn't care where it is, only that it's home (the count is in
        # frac_penned and the status flag).
        rel = (self.flock.pos - self.dog_pos[None, :]) / diag
        rel = np.clip(rel, -1, 1)
        vel = np.clip(self.flock.vel / max(c.flock.delta * c.flock.bark_speed_mult, 1e-6), -1, 1)
        status = np.where(~self.flock.alive, -1.0,
                          np.where(self.flock.penned, 1.0, 0.0))[:, None]
        free = present & self.flock.alive & ~self.flock.penned
        rel[~free] = 0.0           # only free sheep carry a position
        vel[~free] = 0.0
        status[~present, 0] = -1.0
        sheep = np.concatenate([rel, vel, status], axis=1).reshape(-1)
        parts.append(sheep)
        # Obstacles: position relative to dog + radius (padded to fixed count).
        m = c.n_bushes + c.n_rocks
        obuf = np.zeros((m, 3), dtype=np.float32)
        if self.obstacles.size:
            k = self.obstacles.shape[0]
            obuf[:k, :2] = np.clip(
                (self.obstacles[:, :2] - self.dog_pos[None, :]) / diag, -1, 1)
            obuf[:k, 2] = self.obstacles[:, 2] / diag
        parts.append(obuf.reshape(-1))

        flat = np.concatenate([np.asarray(p, dtype=np.float32).reshape(-1) for p in parts])
        return np.clip(flat, -1, 1).astype(np.float32)

    def _render_state(self):
        c = self.cfg
        dusk_t = 0.0
        if c.enable_wolves:
            d0 = c.dusk_fraction
            dusk_t = float(np.clip((self.t / c.max_steps - d0) / max(1 - d0, 1e-6), 0, 1))
        return {
            "pen": self.pen,
            "pen_center": self.pen_center,
            "walls": self.walls,
            "opening_point": self.opening_point,
            "fulcrum": self.fulcrum,
            "obstacles_typed": [
                (self.obstacles[i, 0], self.obstacles[i, 1], self.obstacles[i, 2],
                 int(self.obstacle_kinds[i])) for i in range(len(self.obstacles))
            ],
            "sheep_pos": self.flock.pos,
            "sheep_penned": self.flock.penned,
            "sheep_alive": self.flock.alive,
            "dog_pos": self.dog_pos,
            "dog_heading": self.dog_heading,
            "barking": self._bark_timer > 0,
            "bark_radius": c.flock.bark_radius,
            "wolves": [tuple(w) for w in self.wolves],
            "dusk_t": dusk_t,
        }

    def _get_info(self):
        present = self.flock.present
        n_penned = int((present & self.flock.penned & self.flock.alive).sum())
        free = present & self.flock.alive & ~self.flock.penned
        spread = 0.0
        if free.sum() > 1:
            fp = self.flock.pos[free]
            spread = float(np.mean(np.linalg.norm(fp - fp.mean(0), axis=1)))  # world units
        return {
            "t": self.t,
            "n_penned": n_penned,
            "n_active": int(self.n_active),
            "n_alive": int((present & self.flock.alive).sum()),
            "frac_penned": n_penned / max(self.n_active, 1),
            "flock_spread": spread,        # mean dist of free sheep to flock centroid
            "dog_pos": self.dog_pos.copy(),
        }

    # ------------------------------------------------------------- render
    def render(self):
        if self.render_mode is None:
            return None
        if self._big_renderer is None:
            w, h = self.cfg.render_size
            self._big_renderer = Renderer(w, h, self.bounds)
        img = self._big_renderer.render(self._render_state())
        if self.render_mode == "human":
            # Returns False if the viewer window was closed (Esc/Q/close button).
            return self._big_renderer.show(img, fps=self.metadata["render_fps"])
        return img

    def close(self):
        if self._big_renderer is not None:
            self._big_renderer.close()
            self._big_renderer = None
