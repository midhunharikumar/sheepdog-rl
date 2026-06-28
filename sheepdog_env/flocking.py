"""Sheep flocking dynamics.

The model is a vectorised implementation of the Strombom et al. (2014)
"shepherding" agent model ("Solving the shepherding problem: heuristics for
herding autonomous, interacting agents", J. R. Soc. Interface), which is the
canonical mathematical model for exactly the behaviour seen in the original
"Shepherd's Dog" game: a flock that "thinks as one", bolts away from the dog,
clumps together, and can be pushed toward a goal.

All of the weights below are exposed as plain attributes so they are trivial to
retune to match a specific game build.  Distances are expressed in *world units*
(the env uses a 150 x 100 field by default, matching the wide pasture).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .geometry import wall_repulsion, resolve_walls


@dataclass
class FlockParams:
    # --- Strombom interaction weights -------------------------------------
    rho_s: float = 1.0      # strength of repulsion away from the dog
    c: float = 1.05         # attraction to local centre of mass (cohesion)
    rho_a: float = 2.0      # repulsion from neighbours that are too close
    inertia: float = 0.5    # tendency to keep previous heading
    noise: float = 0.3      # angular noise strength

    # --- interaction distances (world units) ------------------------------
    r_s: float = 30.0       # dog detection distance: sheep let the dog get close
    r_a: float = 2.5        # neighbour separation distance
    n_neighbours: int = 12  # topological neighbourhood for the cohesion term

    # --- per-step displacement --------------------------------------------
    delta: float = 0.6      # step length of an *active* (fleeing) sheep (slow)
    graze_step: float = 0.12    # step length while idly grazing
    graze_prob: float = 0.06    # probability a grazing sheep takes a step

    # --- obstacles ---------------------------------------------------------
    rho_o: float = 3.0      # repulsion from bushes / rocks
    obstacle_margin: float = 6.0  # how far the repulsion field extends

    # --- pen fence ---------------------------------------------------------
    rho_w: float = 2.0      # repulsion from the pen fence
    wall_margin: float = 3.0    # how far the fence repulsion field extends

    # --- bark amplification ------------------------------------------------
    # When the dog barks, nearby sheep flee FASTER and HARDER. By default they
    # bolt *radially outward from the dog's position* (the classic sheepdog
    # behaviour). Set bark_directional=True to instead drive them along the dog's
    # heading (the dog "aims" the herd).
    bark_radius: float = 38.0
    bark_speed_mult: float = 2.0    # delta multiplier inside the bark radius (gentle)
    bark_repulsion_mult: float = 1.8  # push strength multiplier inside bark radius
    bark_impulse: float = 2.5       # one-off shove on the bark frame (soft, keeps flock together)
    bark_directional: bool = False  # False = radial away from dog; True = along heading


def _safe_unit(vecs: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Row-wise normalise an (N, 2) array, leaving zero rows as zero."""
    norm = np.linalg.norm(vecs, axis=-1, keepdims=True)
    return vecs / np.maximum(norm, eps)


class Flock:
    """Holds and integrates the sheep state.

    Positions/velocities are stored as (N, 2) float arrays.  ``penned`` and
    ``alive`` are boolean masks so the env can freeze captured sheep and remove
    sheep eaten by wolves without changing array shapes (important for a fixed
    observation size).
    """

    def __init__(self, n_sheep: int, params: FlockParams, rng: np.random.Generator):
        self.n = int(n_sheep)
        self.p = params
        self.rng = rng
        self.pos = np.zeros((self.n, 2), dtype=np.float32)
        self.heading = np.zeros((self.n, 2), dtype=np.float32)
        self.vel = np.zeros((self.n, 2), dtype=np.float32)
        self.penned = np.zeros(self.n, dtype=bool)
        self.alive = np.ones(self.n, dtype=bool)
        self.present = np.ones(self.n, dtype=bool)   # False = masked-out (absent)

    # ------------------------------------------------------------------ init
    def reset(self, pos: np.ndarray, n_active: int = None):
        self.pos = pos.astype(np.float32).copy()
        ang = self.rng.uniform(0, 2 * np.pi, size=self.n)
        self.heading = np.stack([np.cos(ang), np.sin(ang)], axis=1).astype(np.float32)
        self.vel[:] = 0.0
        self.penned[:] = False
        self.alive[:] = True
        self.present[:] = True
        if n_active is not None and n_active < self.n:
            self.present[n_active:] = False     # mask out the surplus slots
            self.alive[n_active:] = False        # ...so they never participate

    @property
    def active_mask(self) -> np.ndarray:
        """Sheep that still participate in the dynamics."""
        return self.present & self.alive & ~self.penned

    # ----------------------------------------------------------------- step
    def step(self, dog_pos: np.ndarray, barking: bool, obstacles: np.ndarray,
             bounds: np.ndarray, walls: np.ndarray = None,
             bark_dir: np.ndarray = None):
        """Advance the flock one tick.

        Parameters
        ----------
        dog_pos    : (2,) dog position.
        barking    : whether the dog is barking this frame.
        obstacles  : (M, 3) array of [x, y, radius] (may be empty).
        bounds     : (2,) [width, height] of the field.
        walls      : (K, 4) pen-fence segments [ax, ay, bx, by] (may be None).
        bark_dir   : (2,) unit vector of the dog's orientation. While barking,
                     nearby sheep are driven along this direction (the dog "aims"
                     the herd) instead of radially away from the dog.
        """
        p = self.p
        n = self.n
        mask = self.active_mask
        if not mask.any():
            self.vel[:] = 0.0
            return

        prev_pos = self.pos.copy()

        # Distance from each sheep to the dog.
        to_dog = self.pos - dog_pos[None, :]          # vector dog -> sheep
        dist_dog = np.linalg.norm(to_dog, axis=1)

        # Effective detection radius / strengths grow while barking.
        det_radius = np.full(n, p.r_s, dtype=np.float32)
        repulse_dog = np.full(n, p.rho_s, dtype=np.float32)
        delta = np.full(n, p.delta, dtype=np.float32)
        if barking:
            near_bark = dist_dog < p.bark_radius
            det_radius[near_bark] = max(p.r_s, p.bark_radius)
            repulse_dog[near_bark] = p.rho_s * p.bark_repulsion_mult
            delta[near_bark] = p.delta * p.bark_speed_mult

        aware = mask & (dist_dog < det_radius)

        # --- cohesion: attraction to local centre of mass --------------------
        coh = np.zeros((n, 2), dtype=np.float32)
        active_idx = np.where(mask)[0]
        if active_idx.size > 1:
            sub = self.pos[active_idx]
            d = np.linalg.norm(sub[:, None, :] - sub[None, :, :], axis=2)
            np.fill_diagonal(d, np.inf)
            k = min(p.n_neighbours, active_idx.size - 1)
            nn = np.argpartition(d, kth=k - 1, axis=1)[:, :k]
            lcm = sub[nn].mean(axis=1)               # local centre of mass
            coh_sub = _safe_unit(lcm - sub)
            coh[active_idx] = coh_sub

        # --- separation: push away from neighbours closer than r_a -----------
        sep = np.zeros((n, 2), dtype=np.float32)
        if active_idx.size > 1:
            diff = sub[:, None, :] - sub[None, :, :]
            dd = np.linalg.norm(diff, axis=2)
            np.fill_diagonal(dd, np.inf)
            too_close = dd < p.r_a
            contrib = np.where(too_close[..., None], _row_unit(diff), 0.0)
            sep_sub = contrib.sum(axis=1)
            sep[active_idx] = _safe_unit(sep_sub)

        # --- push direction from the dog ------------------------------------
        # Sheep flee radially away from the dog's position. (Optionally, while
        # barking they can instead be driven along the dog's heading.)
        push_dir = _safe_unit(to_dog)                # radial: points away from dog
        if barking and p.bark_directional and bark_dir is not None:
            bd = np.asarray(bark_dir, dtype=np.float32)
            bd = bd / max(np.linalg.norm(bd), 1e-6)
            near_bark = mask & (dist_dog < p.bark_radius)
            push_dir[near_bark] = bd
        rep_dog = push_dir

        # --- obstacle avoidance ---------------------------------------------
        obj = np.zeros((n, 2), dtype=np.float32)
        if obstacles.size:
            ov = self.pos[:, None, :] - obstacles[None, :, :2]   # (n, M, 2)
            od = np.linalg.norm(ov, axis=2)                      # (n, M)
            reach = obstacles[None, :, 2] + p.obstacle_margin
            close = od < reach
            strength = np.clip((reach - od) / np.maximum(reach, 1e-6), 0, 1)
            push = _row_unit(ov) * (strength * close)[..., None]
            obj = _safe_unit(push.sum(axis=1))

        # --- pen fence avoidance --------------------------------------------
        wall = np.zeros((n, 2), dtype=np.float32)
        if walls is not None and len(walls):
            wall = _safe_unit(wall_repulsion(self.pos, walls, p.wall_margin))

        # --- noise -----------------------------------------------------------
        ang = self.rng.uniform(0, 2 * np.pi, size=n)
        noise = np.stack([np.cos(ang), np.sin(ang)], axis=1).astype(np.float32)

        # --- combine ---------------------------------------------------------
        force = (
            p.inertia * self.heading
            + p.c * coh
            + p.rho_a * sep
            + repulse_dog[:, None] * rep_dog * aware[:, None]
            + p.rho_o * obj
            + p.rho_w * wall
            + p.noise * noise
        )
        new_heading = _safe_unit(force)

        # Active (aware or fleeing) sheep take a full step; the rest graze.
        moving = aware.copy()
        graze = mask & ~aware
        graze_roll = self.rng.random(n) < p.graze_prob
        step_len = np.where(moving, delta, 0.0).astype(np.float32)
        step_len = np.where(graze & graze_roll, p.graze_step, step_len)

        disp = new_heading * step_len[:, None]

        if barking:
            # One-off shove away from the dog for sheep inside the bark ring.
            shove = (dist_dog < p.bark_radius) & mask
            disp += rep_dog * (shove[:, None] * p.bark_impulse)

        new_pos = self.pos + disp

        # Keep inside the field.
        new_pos[:, 0] = np.clip(new_pos[:, 0], 0, bounds[0])
        new_pos[:, 1] = np.clip(new_pos[:, 1], 0, bounds[1])

        # Resolve hard overlap with obstacles (don't let sheep sit inside one).
        if obstacles.size:
            new_pos = _push_out_of_circles(new_pos, obstacles)

        # Block any movement that would cross the pen fence (slide along it).
        if walls is not None and len(walls):
            new_pos = resolve_walls(prev_pos, new_pos, walls, slide=True)

        # Commit only for active sheep.
        self.pos[mask] = new_pos[mask]
        self.heading[mask] = new_heading[mask]
        self.vel = (self.pos - prev_pos).astype(np.float32)
        self.vel[~mask] = 0.0


def _row_unit(vecs: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Normalise the last axis of an arbitrarily shaped array."""
    norm = np.linalg.norm(vecs, axis=-1, keepdims=True)
    return vecs / np.maximum(norm, eps)


def _push_out_of_circles(pos: np.ndarray, obstacles: np.ndarray) -> np.ndarray:
    out = pos.copy()
    for ox, oy, orad in obstacles:
        v = out - np.array([ox, oy], dtype=np.float32)
        d = np.linalg.norm(v, axis=1)
        inside = d < orad
        if inside.any():
            u = v[inside] / np.maximum(d[inside, None], 1e-6)
            out[inside] = np.array([ox, oy], dtype=np.float32) + u * orad
    return out
