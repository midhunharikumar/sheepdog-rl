"""Pen-wall geometry: build the fence, repel from it, and block crossings.

The pen is an axis-aligned box with a solid fence on every side except a single
**opening of fixed width**. Sheep (and the dog) cannot pass through the fence —
they can only enter/leave through the gap, exactly like the original game.

Walls are stored as axis-aligned segments ``[ax, ay, bx, by]``.
"""

from __future__ import annotations

import numpy as np


def build_pen_walls(pen, opening_side: str, opening_center_frac: float,
                    opening_width: float):
    """Return (walls, opening_point, opening_outer_dir).

    ``walls``         : (K, 4) array of segments making up the fence (gap omitted).
    ``opening_point`` : (2,) centre of the gap, on the fence line.
    ``opening_outer`` : (2,) unit vector pointing from the gap *out* into the field
                        (the direction a shepherd approaches from).
    """
    x0, y0, x1, y1 = [float(v) for v in pen]
    w = float(opening_width)
    walls = []

    def split(p_start, p_end, gap_lo, gap_hi, axis):
        """Split a side into two segments leaving a gap along ``axis`` (0=x,1=y)."""
        a = np.array(p_start, dtype=np.float32)
        b = np.array(p_end, dtype=np.float32)
        lo = a.copy(); lo[axis] = gap_lo
        hi = a.copy(); hi[axis] = gap_hi
        segs = []
        if gap_lo > min(a[axis], b[axis]) + 1e-3:
            segs.append([a[0], a[1], lo[0], lo[1]])
        if gap_hi < max(a[axis], b[axis]) - 1e-3:
            segs.append([hi[0], hi[1], b[0], b[1]])
        return segs

    top = [x0, y0, x1, y0]
    bottom = [x0, y1, x1, y1]
    left = [x0, y0, x0, y1]
    right = [x1, y0, x1, y1]

    if opening_side == "left":
        oc = y0 + opening_center_frac * (y1 - y0)
        walls += split((x0, y0), (x0, y1), oc - w / 2, oc + w / 2, axis=1)
        walls += [top, bottom, right]
        opening_point = np.array([x0, oc], dtype=np.float32)
        opening_outer = np.array([-1.0, 0.0], dtype=np.float32)
    elif opening_side == "right":
        oc = y0 + opening_center_frac * (y1 - y0)
        walls += split((x1, y0), (x1, y1), oc - w / 2, oc + w / 2, axis=1)
        walls += [top, bottom, left]
        opening_point = np.array([x1, oc], dtype=np.float32)
        opening_outer = np.array([1.0, 0.0], dtype=np.float32)
    elif opening_side == "top":
        oc = x0 + opening_center_frac * (x1 - x0)
        walls += split((x0, y0), (x1, y0), oc - w / 2, oc + w / 2, axis=0)
        walls += [bottom, left, right]
        opening_point = np.array([oc, y0], dtype=np.float32)
        opening_outer = np.array([0.0, -1.0], dtype=np.float32)
    elif opening_side == "bottom":
        oc = x0 + opening_center_frac * (x1 - x0)
        walls += split((x0, y1), (x1, y1), oc - w / 2, oc + w / 2, axis=0)
        walls += [top, left, right]
        opening_point = np.array([oc, y1], dtype=np.float32)
        opening_outer = np.array([0.0, 1.0], dtype=np.float32)
    else:
        raise ValueError(f"opening_side must be left/right/top/bottom, got {opening_side!r}")

    return np.array(walls, dtype=np.float32).reshape(-1, 4), opening_point, opening_outer


def inside_rect(P, pen):
    x0, y0, x1, y1 = pen
    return ((P[:, 0] >= x0) & (P[:, 0] <= x1) &
            (P[:, 1] >= y0) & (P[:, 1] <= y1))


def _rect_first_crossing(p0, p1, pen):
    """First boundary point where segment p0->p1 crosses the rectangle, plus the
    edge it crossed ('left'/'right'/'top'/'bottom'). Returns (point, edge) or
    (None, None)."""
    x0, y0, x1, y1 = pen
    d = p1 - p0
    best_t, best = 2.0, (None, None)
    # vertical edges (constant x)
    for xe, name in ((x0, "left"), (x1, "right")):
        if abs(d[0]) > 1e-12:
            t = (xe - p0[0]) / d[0]
            if 1e-9 < t < best_t:
                y = p0[1] + t * d[1]
                if y0 - 1e-6 <= y <= y1 + 1e-6:
                    best_t, best = t, (np.array([xe, y], np.float32), name)
    # horizontal edges (constant y)
    for ye, name in ((y0, "top"), (y1, "bottom")):
        if abs(d[1]) > 1e-12:
            t = (ye - p0[1]) / d[1]
            if 1e-9 < t < best_t:
                x = p0[0] + t * d[0]
                if x0 - 1e-6 <= x <= x1 + 1e-6:
                    best_t, best = t, (np.array([x, ye], np.float32), name)
    return best


def contain_pen(P0, P1, pen, opening_side, gap_lo, gap_hi, allow_opening=True):
    """Hard containment for the pen rectangle (robust at corners).

    A point may only change between outside/inside the pen by passing through the
    opening gap; any other boundary crossing cancels the move (P1 := P0). This is
    a backstop that catches corner-slide leaks the per-wall resolver can miss.
    """
    in0 = inside_rect(P0, pen)
    in1 = inside_rect(P1, pen)
    flip = np.where(in0 != in1)[0]
    if flip.size == 0:
        return P1
    out = P1.copy()
    for k in flip:
        pt, edge = _rect_first_crossing(P0[k], P1[k], pen)
        legal = False
        if allow_opening and edge == opening_side and pt is not None:
            coord = pt[1] if opening_side in ("left", "right") else pt[0]
            legal = (gap_lo - 1e-3) <= coord <= (gap_hi + 1e-3)
        if not legal:
            out[k] = P0[k]            # cancel illegal entry/exit
    return out


def _seg_point_distance(P, A, B):
    """Distance from each point in P (N,2) to segment A-B; also the closest point."""
    AB = B - A
    L2 = float(AB @ AB)
    if L2 < 1e-9:
        d = np.linalg.norm(P - A[None, :], axis=1)
        return d, np.broadcast_to(A, P.shape).copy()
    t = np.clip(((P - A[None, :]) @ AB) / L2, 0.0, 1.0)
    proj = A[None, :] + t[:, None] * AB[None, :]
    return np.linalg.norm(P - proj, axis=1), proj


def _none_near(P, walls, margin):
    """True if every point in P is outside the walls' bounding box by `margin`.
    Cheap pre-check so we can skip the per-segment loops when the flock is
    mid-field (the common case) -- exact, since far points feel no wall."""
    xs = walls[:, 0::2]
    ys = walls[:, 1::2]
    xmin = xs.min() - margin
    xmax = xs.max() + margin
    ymin = ys.min() - margin
    ymax = ys.max() + margin
    px, py = P[:, 0], P[:, 1]
    return not bool(np.any((px >= xmin) & (px <= xmax) & (py >= ymin) & (py <= ymax)))


def wall_repulsion(P, walls, reach):
    """Unit-ish repulsion (N,2) pushing points away from any wall within ``reach``."""
    force = np.zeros_like(P)
    if walls is None or len(walls) == 0:
        return force
    walls = np.asarray(walls, dtype=np.float32)
    if _none_near(P, walls, reach):       # far from every wall -> zero force
        return force
    for ax, ay, bx, by in walls:
        d, proj = _seg_point_distance(P, np.array([ax, ay], np.float32),
                                      np.array([bx, by], np.float32))
        close = d < reach
        if close.any():
            v = P - proj
            u = v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-6)
            strength = np.clip((reach - d) / max(reach, 1e-6), 0, 1)
            force += u * (strength * close)[:, None]
    return force


def resolve_walls(P0, P1, walls, slide=True):
    """Stop movements P0->P1 that cross a wall; optionally slide along it.

    P0, P1 : (N,2) old/new positions. Returns adjusted (N,2).
    """
    if walls is None or len(walls) == 0:
        return P1
    walls = np.asarray(walls, dtype=np.float32)
    # Skip the per-segment work when both the old and new positions are well
    # clear of the fence (margin > any single-step displacement). Exact: a
    # crossing requires a point within ~one step of a wall.
    if _none_near(np.concatenate([P0, P1], axis=0), walls, 8.0):
        return P1
    out = P1.copy()
    for ax, ay, bx, by in walls:
        out = _resolve_one(P0, out, np.array([ax, ay], np.float32),
                           np.array([bx, by], np.float32), slide)
    return out


def _resolve_one(P0, P1, A, B, slide, eps=0.3):
    """Block movements that switch sides of the (infinite) wall line within the
    segment's extent.

    Uses a signed-distance side-change test instead of a swept-segment
    intersection. This is robust to the degenerate case where a point starts
    *on* the wall (crossing parameter ~0) and gets shoved through — which a
    plain segment-segment test misses.
    """
    s = B - A
    sl = float(np.linalg.norm(s))
    if sl < 1e-9:
        return P1
    sdir = s / sl
    nrm = np.array([-sdir[1], sdir[0]], dtype=np.float32)   # unit wall normal

    d0 = (P0 - A[None, :]) @ nrm        # signed distance, old position
    d1 = (P1 - A[None, :]) @ nrm        # signed distance, new position
    denom = d0 - d1
    safe = np.abs(denom) > 1e-9
    t = np.where(safe, d0 / np.where(safe, denom, 1.0), -1.0)   # path frac at the line
    cross_pt = P0 + t[:, None] * (P1 - P0)
    u = ((cross_pt - A[None, :]) @ sdir) / sl                   # frac along the wall

    changed_side = np.sign(d0) != np.sign(d1)
    crossed = (safe & changed_side
               & (t >= -1e-4) & (t <= 1.0 + 1e-4)
               & (u >= -0.02) & (u <= 1.02))
    if not crossed.any():
        return P1

    # Which side did the point come from? Keep it there. If it started exactly
    # on the wall, treat the *opposite* of the new side as "old".
    side = np.sign(d0)
    side = np.where(side == 0.0, -np.sign(d1), side)

    out = P1.copy()
    if slide:
        rem = P1 - cross_pt
        tang = (rem @ sdir)[:, None] * sdir[None, :]           # parallel component
        new = cross_pt + tang + nrm[None, :] * (side[:, None] * eps)
    else:
        new = cross_pt + nrm[None, :] * (side[:, None] * eps)
    out[crossed] = new[crossed]
    return out
