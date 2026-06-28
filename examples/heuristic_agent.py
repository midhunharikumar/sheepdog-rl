"""A simple scripted shepherd — a *non-trivial baseline*, not an expert policy.

Strategy (the classic Strombom collect/drive heuristic):
  * If the flock is too spread out, "collect": go behind the sheep furthest
    from the flock centre and push it back toward the group.
  * Otherwise "drive": position behind the flock centre on the line to the gap
    and push the whole group toward the opening (flanking around the flock
    rather than cutting through it).
  * Bark while driving from close behind to shove the cluster through the gap.

It demonstrates the dynamics and pens a portion of the flock, but it does NOT
reliably hit the 80% target: herding a 40-sheep flock through a single narrow
opening while the dog is fenced out of the pen is a hard control problem — a
naive point-shepherd tends to fragment the flock against the field edges. That
difficulty is the point: it's what makes this a worthwhile RL benchmark rather
than something a few lines of geometry can solve. Treat this agent as a sanity
check and a starting point, not a solution.
"""

from __future__ import annotations

import numpy as np


class HeuristicShepherd:
    """Canonical Strombom collect/drive controller, adapted so the dog (which is
    fenced out of the pen) presses the flock through the single opening.

    Each step it either *collects* the sheep furthest from the flock's centre of
    mass (when the flock is too spread to fit through the gap) or *drives* the
    whole flock toward a point just inside the gap. The dog always positions
    itself on the far side of the flock from its goal, so it stays in the field.
    """

    def __init__(self, env):
        self.env = env
        c = env.cfg
        # Use the active flock size (curriculum) so the Strombom distances scale.
        N = getattr(env, "n_active", None) or c.n_active or c.n_sheep
        ra = c.flock.r_a
        # Aim a little way *inside* the pen, straight through the gap.
        self.target = env.opening_point - env.opening_outer * 8.0
        # Strombom distances.
        self.drive_off = ra * np.sqrt(N)        # how far behind the GCM to drive
        self.collect_thresh = ra * (N ** (2.0 / 3.0))   # "flock is too spread"
        self.collect_off = ra                   # how far behind a stray to stand
        self.bark_dist = 6.0

    def act(self, _obs):
        env = self.env
        f = env.flock
        mask = f.alive & ~f.penned
        if not mask.any():
            return self._encode(np.array([1.0, 0.0], np.float32), 0.0, False)

        pos = f.pos[mask]
        gcm = pos.mean(axis=0)
        to_t = self.target - gcm
        to_t_u = to_t / max(np.linalg.norm(to_t), 1e-6)

        # Furthest sheep from the centre of mass.
        d = np.linalg.norm(pos - gcm, axis=1)
        far_i = int(np.argmax(d))
        far = pos[far_i]
        collecting = d[far_i] > self.collect_thresh

        if collecting:
            anchor = far
            adir = (far - gcm) / max(np.linalg.norm(far - gcm), 1e-6)
            drive_pt = far + adir * self.collect_off
        else:
            anchor = gcm
            drive_pt = gcm - to_t_u * self.drive_off   # behind flock toward gap

        # Reach the driving position by flanking *around* the flock when not
        # already behind it, so the straight path doesn't cut through the group.
        dog = env.dog_pos
        behind = np.dot(dog - anchor, -to_t_u) > self.drive_off * 0.3
        if behind:
            target_pt = drive_pt
        else:
            perp = np.array([-to_t_u[1], to_t_u[0]], dtype=np.float32)
            side = np.sign(np.dot(dog - anchor, perp)) or 1.0
            target_pt = anchor + perp * side * self.drive_off - to_t_u * self.drive_off

        dist_to_slot = np.linalg.norm(drive_pt - dog)
        in_position = behind and dist_to_slot < self.bark_dist

        # Always steer to the driving slot (behind the flock, on the line to the
        # gap). The bark is *radial* — it pushes the flock away from the dog — so
        # standing behind the flock and barking drives it toward the opening.
        # Slow down near the slot so we hold position instead of overrunning.
        heading_vec = target_pt - dog
        speed01 = float(np.clip(dist_to_slot / self.drive_off, 0.2, 1.0))
        bark = (not collecting) and in_position
        heading_vec = heading_vec / max(np.linalg.norm(heading_vec), 1e-6)

        return self._encode(heading_vec, speed01, bark)

    def _encode(self, heading_vec, speed01, bark):
        """Pack (heading, speed, bark) into the env's action convention."""
        b = 1.0 if bark else -1.0
        if self.env.cfg.action_mode == "polar":
            speed_a = 2.0 * float(np.clip(speed01, 0, 1)) - 1.0   # [0,1] -> [-1,1]
            theta = float(np.arctan2(heading_vec[1], heading_vec[0])) / np.pi
            return np.array([speed_a, theta, b], dtype=np.float32)
        # cartesian: velocity vector = heading * speed
        v = heading_vec * float(np.clip(speed01, 0, 1))
        return np.array([v[0], v[1], b], dtype=np.float32)


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from sheepdog_env import SheepdogHerdingEnv

    env = SheepdogHerdingEnv(render_mode="rgb_array", enable_wolves=False)
    agent = HeuristicShepherd(env)
    obs, info = env.reset(seed=0)
    total = 0.0
    for step in range(env.cfg.max_steps):
        obs, r, term, trunc, info = env.step(agent.act(obs))
        total += r
        if term or trunc:
            break
    print(f"steps={step+1}  penned={info['n_penned']}/{env.cfg.n_sheep}  "
          f"success={info.get('is_success')}  return={total:.1f}")
