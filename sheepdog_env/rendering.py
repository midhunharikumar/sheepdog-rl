"""Dependency-light top-down renderer.

Produces an (H, W, 3) uint8 image with pure NumPy so it works headless (used
both for `render_mode="rgb_array"` and for pixel observations).  A `human`
viewer is provided via pygame *if it is installed*; otherwise human mode falls
back to writing frames you can inspect.

Colours roughly echo the original game: green pasture, sandy pen, white sheep,
dark dog, a bark ring, and red wolves.
"""

from __future__ import annotations

import numpy as np

# Palette (RGB)
C_GRASS = np.array([124, 169, 86], dtype=np.uint8)
C_GRASS_DUSK = np.array([70, 92, 70], dtype=np.uint8)
C_PEN = np.array([196, 178, 122], dtype=np.uint8)
C_PEN_EDGE = np.array([120, 88, 52], dtype=np.uint8)
C_BUSH = np.array([57, 120, 54], dtype=np.uint8)
C_ROCK = np.array([150, 150, 152], dtype=np.uint8)
C_SHEEP = np.array([245, 245, 240], dtype=np.uint8)
C_SHEEP_PEN = np.array([180, 220, 170], dtype=np.uint8)
C_DOG = np.array([40, 35, 40], dtype=np.uint8)
C_BARK = np.array([255, 255, 255], dtype=np.uint8)
C_WOLF = np.array([170, 40, 40], dtype=np.uint8)


class Renderer:
    def __init__(self, width_px: int, height_px: int, world_bounds):
        self.w = int(width_px)
        self.h = int(height_px)
        self.bounds = np.asarray(world_bounds, dtype=np.float32)
        self.sx = self.w / self.bounds[0]
        self.sy = self.h / self.bounds[1]
        self._yy, self._xx = np.mgrid[0:self.h, 0:self.w]
        self._pygame_screen = None

    # --------------------------------------------------------------- helpers
    def _to_px(self, p):
        return np.array([p[0] * self.sx, p[1] * self.sy])

    def _disc(self, img, cx, cy, r, color, alpha=1.0):
        r = max(r, 0.6)
        m = (self._xx - cx) ** 2 + (self._yy - cy) ** 2 <= r * r
        if alpha >= 1.0:
            img[m] = color
        else:
            img[m] = (img[m] * (1 - alpha) + color * alpha).astype(np.uint8)

    def _ring(self, img, cx, cy, r, color, thickness=1.5, alpha=1.0):
        d2 = (self._xx - cx) ** 2 + (self._yy - cy) ** 2
        m = (d2 <= (r + thickness) ** 2) & (d2 >= (r - thickness) ** 2)
        if alpha >= 1.0:
            img[m] = color
        else:
            img[m] = (img[m] * (1 - alpha) + color * alpha).astype(np.uint8)

    def _line(self, img, x0, y0, x1, y1, color, width=1.0):
        """Draw a thick line segment by stamping discs along it."""
        x0, y0, x1, y1 = float(x0), float(y0), float(x1), float(y1)
        length = max(np.hypot(x1 - x0, y1 - y0), 1e-6)
        n = int(length) + 1
        ts = np.linspace(0.0, 1.0, n)
        r = max(width * 0.5, 0.6)
        for t in ts:
            self._disc(img, x0 + t * (x1 - x0), y0 + t * (y1 - y0), r, color)

    def _rect(self, img, x0, y0, x1, y1, color):
        x0, x1 = sorted((int(x0), int(x1)))
        y0, y1 = sorted((int(y0), int(y1)))
        x0 = max(x0, 0); y0 = max(y0, 0)
        x1 = min(x1, self.w); y1 = min(y1, self.h)
        img[y0:y1, x0:x1] = color

    # --------------------------------------------------------------- render
    def render(self, state) -> np.ndarray:
        dusk = float(state.get("dusk_t", 0.0))  # 0 day .. 1 night
        bg = (C_GRASS * (1 - dusk) + C_GRASS_DUSK * dusk).astype(np.uint8)
        img = np.empty((self.h, self.w, 3), dtype=np.uint8)
        img[:] = bg

        # Pen floor (sandy rectangle).
        pen = state["pen"]  # [x0, y0, x1, y1]
        px0, py0 = self._to_px((pen[0], pen[1]))
        px1, py1 = self._to_px((pen[2], pen[3]))
        self._rect(img, px0, py0, px1, py1, C_PEN)

        # Fence: draw each wall segment as a thick post line, leaving the gap.
        fence_t = max(2, int(0.6 * self.sx))
        for ax, ay, bx, by in state.get("walls", []):
            a = self._to_px((ax, ay))
            b = self._to_px((bx, by))
            self._rect(img, min(a[0], b[0]) - fence_t / 2, min(a[1], b[1]) - fence_t / 2,
                       max(a[0], b[0]) + fence_t / 2, max(a[1], b[1]) + fence_t / 2,
                       C_PEN_EDGE)

        # Flag at the top-right corner of the pen.
        fx, fy = self._to_px((pen[2], pen[1]))
        self._rect(img, fx - 1, fy - 14, fx + 1, fy, C_PEN_EDGE)
        self._rect(img, fx + 1, fy - 14, fx + 9, fy - 8, np.array([200, 40, 40], np.uint8))

        # Fulcrum (staging point in front of the gate): a small hollow marker.
        fdot = state.get("fulcrum")
        if fdot is not None:
            fp = self._to_px((fdot[0], fdot[1]))
            self._ring(img, fp[0], fp[1], max(3.0, 1.4 * self.sx),
                       np.array([235, 235, 180], np.uint8), thickness=1.2, alpha=0.7)

        # Obstacles.
        for ox, oy, orad, kind in state["obstacles_typed"]:
            c = self._to_px((ox, oy))
            color = C_BUSH if kind == 0 else C_ROCK
            self._disc(img, c[0], c[1], orad * self.sx, color)

        # Sheep.
        pos = state["sheep_pos"]
        penned = state["sheep_penned"]
        alive = state["sheep_alive"]
        rr = max(1.6, 0.9 * self.sx)
        for i in range(len(pos)):
            if not alive[i]:
                continue
            c = self._to_px(pos[i])
            self._disc(img, c[0], c[1], rr,
                       C_SHEEP_PEN if penned[i] else C_SHEEP)

        # Wolves.
        for wx, wy in state.get("wolves", []):
            c = self._to_px((wx, wy))
            self._disc(img, c[0], c[1], rr * 1.3, C_WOLF)

        # Dog + bark ring + heading indicator.
        dog = self._to_px(state["dog_pos"])
        if state.get("barking"):
            self._ring(img, dog[0], dog[1],
                       state["bark_radius"] * self.sx, C_BARK,
                       thickness=1.5, alpha=0.5)
        r_dog = max(2.0, 1.1 * self.sx)
        # Orientation tick: a short line from the dog in its heading direction
        # (brightened to the bark colour while barking, since that's the push dir).
        head = state.get("dog_heading")
        if head is not None:
            h = np.asarray(head, dtype=np.float32)
            h = h / max(float(np.linalg.norm(h)), 1e-6)
            tip = dog + h * (r_dog + max(5.0, 2.2 * self.sx))
            self._line(img, dog[0], dog[1], tip[0], tip[1], C_DOG,
                       width=max(1.0, 0.5 * self.sx))
        self._disc(img, dog[0], dog[1], r_dog, C_DOG)

        return img

    # ----------------------------------------------------------- human view
    def show(self, img: np.ndarray, fps: int = 30):
        try:
            import pygame
        except Exception:
            return False
        if self._pygame_screen is None:
            pygame.init()
            self._pygame_screen = pygame.display.set_mode((self.w, self.h))
            pygame.display.set_caption("Sheepdog Herding")
            self._clock = pygame.time.Clock()
        for event in pygame.event.get():
            quit_req = event.type == pygame.QUIT
            quit_req |= (event.type == pygame.KEYDOWN and
                         event.key in (pygame.K_ESCAPE, pygame.K_q))
            if quit_req:
                pygame.quit()
                self._pygame_screen = None
                return False
        surf = pygame.surfarray.make_surface(np.transpose(img, (1, 0, 2)))
        self._pygame_screen.blit(surf, (0, 0))
        pygame.display.flip()
        self._clock.tick(fps)
        return True

    def close(self):
        if self._pygame_screen is not None:
            try:
                import pygame
                pygame.quit()
            except Exception:
                pass
            self._pygame_screen = None
