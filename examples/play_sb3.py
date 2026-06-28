"""Live viewer with a real-time reward graph (pygame window).

    # play it yourself with the mouse (dog follows the cursor, click to bark):
    python examples/play_sb3.py --mouse

    # watch a trained model / the scripted shepherd / a random policy:
    python examples/play_sb3.py --model ppo_sheepdog
    python examples/play_sb3.py --heuristic
    python examples/play_sb3.py --random --wolves --fps 20

The top band plots cumulative return over time and shows the live per-step
reward broken down by component (entrance / back / cohesion / pen / final).
Disable the graph with --no-plot.

Requires pygame (`pip install pygame`). Episodes auto-reset; close the window or
press Esc/Q to quit. Each episode's result is also printed to the terminal.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sheepdog_env import SheepdogHerdingEnv

GRAPH_H = 150          # height of the reward-graph band (px); 0 if --no-plot


def mouse_action(env, y_offset):
    """Action that steers the dog toward the mouse cursor; bark while LMB held."""
    import pygame
    W, H = env.cfg.render_size
    sx, sy = W / env.cfg.width, H / env.cfg.height
    mx, my = pygame.mouse.get_pos()
    target = np.array([mx / sx, (my - y_offset) / sy], dtype=np.float32)
    d = target - env.dog_pos
    dist = float(np.linalg.norm(d))
    u = d / max(dist, 1e-6)
    speed01 = float(np.clip(dist / (2.0 * env.cfg.dog_max_speed), 0.0, 1.0))
    bark = 1.0 if pygame.mouse.get_pressed()[0] else -1.0
    if env.cfg.action_mode == "polar":
        theta = float(np.arctan2(u[1], u[0])) / np.pi
        return np.array([2.0 * speed01 - 1.0, theta, bark], dtype=np.float32)
    v = u * speed01
    return np.array([v[0], v[1], bark], dtype=np.float32)


def build_policy(args, env):
    """Return (policy(obs)->action, label) for the non-mouse modes."""
    if args.heuristic:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from heuristic_agent import HeuristicShepherd
        agent = HeuristicShepherd(env)
        return (lambda obs: agent.act(obs)), "heuristic"
    if args.random:
        return (lambda obs: env.action_space.sample()), "random"
    from stable_baselines3 import PPO
    from _policy_utils import load_obs_normalizer
    model = PPO.load(args.model, device=args.device)
    normalize, msg = load_obs_normalizer(args.model, args.vecnorm)
    print(msg)
    det = not args.stochastic
    return (lambda obs: model.predict(normalize(obs), deterministic=det)[0]), f"model:{args.model}"


def draw_graph(pygame, screen, font, W, steps, returns, info, rb, comp_totals, max_steps):
    """Draw the cumulative-return curve + cumulative component readout.

    Everything shown is cumulative over the episode so the components stay in
    sync with the return curve (pen_enter is an event reward, so per-step it
    would just read 0.0 between pens)."""
    screen.fill((24, 26, 30), (0, 0, W, GRAPH_H))
    pad = 26
    lo = min(returns + [0.0]); hi = max(returns + [0.0])
    rng = (hi - lo) or 1.0

    def ymap(v):
        return int((GRAPH_H - pad) - (v - lo) / rng * (GRAPH_H - 2 * pad))

    # zero baseline + curve
    y0 = ymap(0.0)
    pygame.draw.line(screen, (70, 74, 80), (0, y0), (W, y0), 1)
    if len(steps) > 1:
        pts = [(int(s / max_steps * (W - 1)), ymap(r)) for s, r in zip(steps, returns)]
        pygame.draw.lines(screen, (120, 210, 120), False, pts, 2)

    ret = returns[-1] if returns else 0.0
    step_r = sum(rb.values()) if rb else 0.0
    head = (f"return {ret:7.1f}   step {step_r:+5.2f}   "
            f"penned {info.get('n_penned', 0)}/{info.get('n_active', '?')}   t={info.get('t', 0)}")
    screen.blit(font.render(head, True, (235, 235, 235)), (8, 4))

    # cumulative per-component readout (color-coded) — sums to the return
    colors = {"entrance": (240, 200, 120), "back": (235, 130, 130),
              "cohesion": (130, 180, 235), "pen_enter": (140, 235, 140),
              "fulcrum": (235, 235, 150), "final": (220, 220, 140)}
    x = 8
    for k in ["pen_enter", "fulcrum", "entrance", "back", "cohesion", "final"]:
        v = comp_totals.get(k, 0.0)
        txt = font.render(f"{k}={v:+.1f}", True, colors.get(k, (200, 200, 200)))
        screen.blit(txt, (x, GRAPH_H - 20))
        x += txt.get_width() + 14


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--model", help="path to a saved SB3 model (.zip)")
    g.add_argument("--heuristic", action="store_true", help="use the scripted shepherd")
    g.add_argument("--random", action="store_true", help="use a random policy")
    g.add_argument("--mouse", action="store_true",
                   help="play it yourself: dog follows the mouse, click to bark")
    ap.add_argument("--obs", choices=["vector", "pixel"], default="vector")
    ap.add_argument("--vecnorm", default=None, help="VecNormalize stats (auto-detected)")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    ap.add_argument("--action-mode", choices=["polar", "cartesian"], default=None,
                    help="must match training")
    ap.add_argument("--wolves", action="store_true")
    ap.add_argument("--stochastic", action="store_true")
    ap.add_argument("--no-plot", action="store_true", help="hide the reward graph")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--episodes", type=int, default=0, help="0 = run until window closed")
    args = ap.parse_args()
    if not (args.model or args.heuristic or args.random or args.mouse):
        ap.error("pass one of --model / --heuristic / --random / --mouse")

    try:
        import pygame
    except ImportError:
        sys.exit("pygame is required: pip install pygame")

    env_kwargs = {"action_mode": args.action_mode} if args.action_mode else {}
    env = SheepdogHerdingEnv(obs_mode=args.obs, enable_wolves=args.wolves,
                             render_mode="rgb_array", **env_kwargs)
    W, H = env.cfg.render_size
    band = 0 if args.no_plot else GRAPH_H
    if args.mouse:
        policy, label = None, "you (mouse)"
    else:
        policy, label = build_policy(args, env)

    pygame.init()
    screen = pygame.display.set_mode((W, H + band))
    pygame.display.set_caption(f"Sheepdog — {label}")
    font = pygame.font.SysFont("monospace", 15)
    clock = pygame.time.Clock()
    print(f"Live viewer — {label}.  Close the window or press Esc/Q to quit.")
    if args.mouse:
        print("Move the mouse to steer the dog; hold the left button to bark.")

    ep, running = 0, True
    while running and (args.episodes == 0 or ep < args.episodes):
        obs, info = env.reset(seed=args.seed + ep)
        steps, returns, ret = [], [], 0.0
        rb, comp_totals = {}, {}
        done = False
        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (
                        event.type == pygame.KEYDOWN and event.key in (pygame.K_ESCAPE, pygame.K_q)):
                    running = False
                    done = True
            if not running:
                break

            action = mouse_action(env, band) if args.mouse else policy(obs)
            obs, r, term, trunc, info = env.step(action)
            rb = info.get("reward_breakdown", {})
            for k, v in rb.items():
                comp_totals[k] = comp_totals.get(k, 0.0) + v
            ret += r
            steps.append(info["t"]); returns.append(ret)

            frame = env.render()                      # (H, W, 3) uint8
            surf = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
            screen.blit(surf, (0, band))
            if band:
                draw_graph(pygame, screen, font, W, steps, returns, info, rb,
                           comp_totals, env.cfg.max_steps)
            pygame.display.flip()
            clock.tick(args.fps)

            if term or trunc:
                done = True

        if running:
            print(f"episode {ep:2d}: penned {info['n_penned']:2d}/{env.cfg.n_sheep}  "
                  f"success={info.get('is_success')}  return={ret:7.1f}")
        ep += 1

    pygame.quit()
    env.close()


if __name__ == "__main__":
    main()
