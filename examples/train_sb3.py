"""Train a PPO agent on the vector-observation env with Stable-Baselines3.

Experiment tracking uses Weights & Biases (wandb).

    pip install "stable-baselines3>=2.0" wandb
    wandb login                      # once, or set WANDB_API_KEY
    python examples/train_sb3.py --timesteps 1_000_000

    # flock-size curriculum: start with a big dense flock (more pen events =
    # denser reward) and anneal down to the game's 40 over the first half:
    python examples/train_sb3.py --n-sheep 80 --active-start 80 --active-end 40 \
        --curriculum-frac 0.5 --timesteps 2_000_000

    # tune the reward shaping weights:
    python examples/train_sb3.py --w-back 0.15 --w-entrance 0.3

For pixel observations use --obs pixel (PPO with CnnPolicy; much slower).
Run without tracking via --no-wandb, or fully offline with --wandb-mode offline.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sheepdog_env import SheepdogHerdingEnv


def linear_schedule(initial, final=0.0):
    """LR schedule: decays linearly from `initial` to `final` over training.
    SB3 passes progress_remaining in [1, 0] (1 at the start, 0 at the end)."""
    return lambda pr: final + pr * (initial - final)


def cosine_schedule(initial, final=0.0):
    """LR schedule: cosine decay from `initial` to `final` over training.
    Gentler early, flattens out near the end -- often more stable than linear."""
    import math

    return lambda pr: final + 0.5 * (initial - final) * (1.0 + math.cos(math.pi * (1.0 - pr)))


def make_env(obs_mode, wolves, n_sheep, n_active, env_kwargs):
    def _f():
        return SheepdogHerdingEnv(
            obs_mode=obs_mode,
            enable_wolves=wolves,
            n_sheep=n_sheep,
            n_active=n_active,
            **env_kwargs,
        )

    return _f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timesteps", type=int, default=1_000_000)
    ap.add_argument("--obs", choices=["vector", "pixel"], default="vector")
    ap.add_argument("--n-envs", type=int, default=8)
    ap.add_argument(
        "--vec-backend",
        choices=["subproc", "batched"],
        default="subproc",
        help="'batched' runs all envs in one process as vectorised NumPy "
        "(3-6x faster for the vector env; lets you push --n-envs much higher). "
        "'subproc' is the classic one-process-per-env (needed for pixel/wolves).",
    )
    ap.add_argument("--wolves", action="store_true")
    ap.add_argument("--save", default="ppo_sheepdog")
    ap.add_argument(
        "--init-from",
        default=None,
        help="warm-start from a saved model (e.g. a BC checkpoint); "
        "loads its weights and VecNormalize stats into a FRESH run",
    )
    ap.add_argument(
        "--resume",
        default=None,
        help="resume a saved .zip checkpoint, continuing optimizer state, "
        "timestep count and LR schedule (give path WITHOUT the .zip)",
    )
    # --- flock-size curriculum -------------------------------------------
    ap.add_argument(
        "--n-sheep", type=int, default=40, help="observation capacity / max flock size"
    )
    ap.add_argument(
        "--active-start",
        type=int,
        default=None,
        help="curriculum: initial active flock (default = n_sheep)",
    )
    ap.add_argument(
        "--active-end",
        type=int,
        default=None,
        help="curriculum: final active flock (default = n_sheep, i.e. no curriculum)",
    )
    ap.add_argument(
        "--curriculum-frac",
        type=float,
        default=0.5,
        help="default fraction of training over which to anneal each curriculum "
        "(both start at step 0 unless an independent window is set below)",
    )
    ap.add_argument(
        "--active-frac",
        type=float,
        default=None,
        help="independent anneal length for the FLOCK-SIZE curriculum, as a "
        "fraction of training (window [0, active_frac]). Defaults to --curriculum-frac.",
    )
    ap.add_argument(
        "--spawn-frac",
        type=float,
        default=None,
        help="independent anneal length for the SPAWN-DISTANCE curriculum. "
        "Defaults to --curriculum-frac.",
    )
    ap.add_argument(
        "--spawn-begin-frac",
        type=float,
        default=0.0,
        help="when the spawn-distance curriculum STARTS, as a fraction of training. "
        "Set this to --active-frac to stage them: grow the flock first, THEN push "
        "it farther.",
    )
    # --- spawn-distance curriculum (start the flock near the gap, move it out) --
    ap.add_argument(
        "--spawn-start",
        type=float,
        default=None,
        help="curriculum: initial flock spawn x as a fraction of width "
        "(e.g. 0.65 = near the gap = easy)",
    )
    ap.add_argument(
        "--spawn-end",
        type=float,
        default=None,
        help="curriculum: final flock spawn x fraction (e.g. 0.22 = far)",
    )
    # --- reward knobs (override EnvConfig defaults) ---------------------
    ap.add_argument(
        "--w-back",
        type=float,
        default=None,
        help="penalty weight for sheep driven behind the pen",
    )
    ap.add_argument(
        "--w-entrance",
        type=float,
        default=None,
        help="distance-to-entrance shaping weight",
    )
    ap.add_argument(
        "--w-cohesion",
        type=float,
        default=None,
        help="weight on keeping the flock together (penalty for spread)",
    )
    ap.add_argument(
        "--w-fulcrum",
        type=float,
        default=None,
        help="reward for staging the flock in front of the gate",
    )
    ap.add_argument(
        "--net-arch",
        default="256,256",
        help="MLP hidden layer sizes, comma-separated (default 256,256; "
        "SB3's own default is 64,64)",
    )
    # --- optimizer / stability knobs ------------------------------------
    ap.add_argument(
        "--lr",
        type=float,
        default=1.5e-4,
        help="learning rate (SB3 default 3e-4; lowered for stability)",
    )
    ap.add_argument(
        "--lr-schedule",
        choices=["linear", "cosine", "constant"],
        default="linear",
        help="decay the LR over training: linear or cosine (toward --lr-final), "
        "or 'constant' to keep it fixed",
    )
    ap.add_argument(
        "--lr-final",
        type=float,
        default=0.0,
        help="floor the LR schedule ends at (linear/cosine). Default 0; set a "
        "small value (e.g. 1e-5) to avoid the LR collapsing fully late in training",
    )
    ap.add_argument(
        "--n-epochs",
        type=int,
        default=6,
        help="PPO epochs per rollout (SB3 default 10; fewer = less drift)",
    )
    ap.add_argument(
        "--target-kl",
        type=float,
        default=0.03,
        help="KL cap per update; PPO stops epochs when KL > 1.5x this. Higher = "
        "looser (bigger, riskier updates). Lowering --lr is usually better.",
    )
    ap.add_argument(
        "--device", default="auto", choices=["auto", "cpu", "cuda", "mps"],
        help="torch device. 'mps' = Apple Silicon GPU (note: the env sim is CPU, "
        "and the MLP is small, so MPS may not be faster than 'cpu' here).",
    )
    ap.add_argument(
        "--log-std-init",
        type=float,
        default=-1.0,
        help="initial policy log std (std=exp); -1.0->0.37 (default), "
        "-0.5->0.61, 0->1.0. Higher = wider exploration cone.",
    )
    ap.add_argument(
        "--ent-coef",
        type=float,
        default=0.003,
        help="entropy bonus; higher keeps exploration alive longer",
    )
    ap.add_argument(
        "--clip-range",
        type=float,
        default=0.2,
        help="PPO clip range (smaller = more conservative updates)",
    )
    # --- watch training (record rollouts) -------------------------------
    ap.add_argument(
        "--watch",
        action="store_true",
        help="every --watch-every rollouts, play an episode with the current "
        "policy and save it as a GIF (+ W&B video). Headless-safe.",
    )
    ap.add_argument(
        "--watch-every",
        type=int,
        default=5,
        help="record an episode every N rollouts (1 = every rollout)",
    )
    ap.add_argument("--watch-dir", default="watch_rollouts", help="where to save the GIFs")
    ap.add_argument(
        "--checkpoint-freq",
        type=int,
        default=0,
        help="save a model + VecNormalize checkpoint every N env steps "
        "(0 = off). Saved under --checkpoint-dir; resume with --resume.",
    )
    ap.add_argument(
        "--checkpoint-dir",
        default="checkpoints",
        help="directory for periodic checkpoints (see --checkpoint-freq)",
    )
    ap.add_argument("--watch-fps", type=int, default=30)
    ap.add_argument("--watch-stochastic", action="store_true",
                    help="record the SAMPLED policy (shows exploration) instead of the "
                         "deterministic mean (early on the mean is a constant 'move forward')")
    # --- difficulty of the WIN condition (the success-rate ceiling) ----------
    ap.add_argument(
        "--target-fraction",
        type=float,
        default=None,
        help="fraction penned to count as success (env default 0.8; "
        "0.8 is above the achievable ceiling with the narrow gap)",
    )
    ap.add_argument(
        "--terminate-on-target",
        action="store_true",
        help="end the episode the instant target_fraction is reached. Default "
        "off: the episode runs until every sheep is penned (or it times out), "
        "so each extra sheep keeps paying via pen_enter + a larger final reward.",
    )
    ap.add_argument(
        "--gap-width",
        type=float,
        default=None,
        help="pen opening width (env default 14; wider = more winnable)",
    )
    # --- environment realism: obstacles + how scattered the flock is ---------
    ap.add_argument("--n-bushes", type=int, default=0,
                    help="number of bush obstacles the flock must herd around")
    ap.add_argument("--n-rocks", type=int, default=0,
                    help="number of rock obstacles (same dynamics, different render)")
    ap.add_argument("--spawn-spread", type=float, default=None,
                    help="std (world units) of the initial flock cluster "
                    "(env default 4; larger = the herd starts more scattered)")
    ap.add_argument("--flock-cohesion", type=float, default=None,
                    help="Strombom cohesion weight (default 1.05; LOWER = sheep "
                    "clump less and scatter more)")
    ap.add_argument("--flock-noise", type=float, default=None,
                    help="angular noise in sheep motion (default 0.3; HIGHER = "
                    "more erratic, scattered wandering)")
    ap.add_argument("--dog-random-prob", type=float, default=None,
                    help="probability per step the dog ignores the policy and moves "
                    "in a random direction (epsilon-random exploration; 0 = off)")
    # --- domain randomization (per-episode ranges; for a robust policy) ------
    ap.add_argument("--dr-active", type=int, nargs=2, default=None, metavar=("MIN", "MAX"),
                    help="randomize active flock size in [MIN,MAX] each episode "
                    "(overrides the active curriculum)")
    ap.add_argument("--dr-spawn", type=float, nargs=2, default=None, metavar=("MIN", "MAX"),
                    help="randomize spawn x-fraction in [MIN,MAX] each episode "
                    "(overrides the spawn curriculum). Higher = nearer the pen.")
    ap.add_argument("--dr-gap", type=float, nargs=2, default=None, metavar=("MIN", "MAX"),
                    help="randomize the entrance width in [MIN,MAX] each episode")
    ap.add_argument(
        "--pen-frac",
        type=float,
        nargs=4,
        default=None,
        metavar=("X0", "Y0", "X1", "Y1"),
        help="pen rectangle as fractions of the field (default 0.80 0.30 0.96 0.70); "
        "a bigger rectangle = a larger, easier-to-hit pen",
    )
    ap.add_argument(
        "--enable-bark",
        action="store_true",
        help="let the dog bark (off by default); barking is the strongest herding tool",
    )
    ap.add_argument(
        "--action-mode",
        choices=["polar", "cartesian"],
        default=None,
        help="polar [speed,heading,bark] (env default) biases a fresh policy to "
        "drive forward; cartesian [vx,vy,bark] has unbiased isotropic exploration",
    )
    # --- Weights & Biases -------------------------------------------------
    ap.add_argument("--no-wandb", action="store_true", help="disable wandb tracking")
    ap.add_argument("--wandb-project", default="sheepdog-rl")
    ap.add_argument("--wandb-entity", default=None, help="team/user (optional)")
    ap.add_argument(
        "--wandb-mode", default="online", choices=["online", "offline", "disabled"]
    )
    ap.add_argument("--run-name", default=None)
    args = ap.parse_args()

    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import (
        BaseCallback,
        CallbackList,
        CheckpointCallback,
    )
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

    # Resolve the flock-size curriculum.
    active_start = args.active_start if args.active_start is not None else args.n_sheep
    active_end = args.active_end if args.active_end is not None else args.n_sheep
    active_start = max(1, min(active_start, args.n_sheep))
    active_end = max(1, min(active_end, args.n_sheep))
    use_curriculum = active_start != active_end
    use_spawn_curriculum = args.spawn_start is not None and args.spawn_end is not None
    spawn_init = args.spawn_start if use_spawn_curriculum else None

    # Optional reward overrides (only set what was passed; else env defaults).
    env_kwargs = {}
    if use_spawn_curriculum:
        env_kwargs["spawn_x_frac"] = spawn_init
    if args.w_back is not None:
        env_kwargs["w_back"] = args.w_back
    if args.w_entrance is not None:
        env_kwargs["w_entrance"] = args.w_entrance
    if args.w_cohesion is not None:
        env_kwargs["w_cohesion"] = args.w_cohesion
    if args.w_fulcrum is not None:
        env_kwargs["w_fulcrum"] = args.w_fulcrum
    if args.target_fraction is not None:
        env_kwargs["target_fraction"] = args.target_fraction
    if args.terminate_on_target:
        env_kwargs["terminate_on_target"] = True
    if args.gap_width is not None:
        env_kwargs["pen_opening_width"] = args.gap_width
    if args.pen_frac is not None:
        env_kwargs["pen_frac"] = tuple(args.pen_frac)
    if args.enable_bark:
        env_kwargs["enable_bark"] = True
    if args.action_mode is not None:
        env_kwargs["action_mode"] = args.action_mode
    if args.n_bushes:
        env_kwargs["n_bushes"] = args.n_bushes
    if args.n_rocks:
        env_kwargs["n_rocks"] = args.n_rocks
    if args.spawn_spread is not None:
        env_kwargs["spawn_spread"] = args.spawn_spread
    if args.dog_random_prob is not None:
        env_kwargs["dog_random_action_prob"] = args.dog_random_prob
    if args.dr_active is not None:
        env_kwargs["dr_active_range"] = tuple(args.dr_active)
    if args.dr_spawn is not None:
        env_kwargs["dr_spawn_x_range"] = tuple(args.dr_spawn)
    if args.dr_gap is not None:
        env_kwargs["dr_gap_range"] = tuple(args.dr_gap)

    config = dict(
        algo="PPO",
        obs_mode=args.obs,
        n_envs=args.n_envs,
        wolves=args.wolves,
        n_sheep=args.n_sheep,
        active_start=active_start,
        active_end=active_end,
        curriculum_frac=args.curriculum_frac if use_curriculum else 0.0,
        reward_overrides=dict(env_kwargs),   # snapshot (flock obj added below)
        timesteps=args.timesteps,
        n_steps=4096 // max(args.n_envs, 1),
        batch_size=256,
        gae_lambda=0.95,
        gamma=0.999,
        learning_rate=args.lr,
        lr_schedule=args.lr_schedule,
        n_epochs=args.n_epochs,
        clip_range=args.clip_range,
        # Entropy bonus + initial policy std control how much the agent explores.
        # Defaults are modest (std=exp(-1)~0.37) so heading is controllable; raise
        # --ent-coef / --log-std-init to widen the exploration cone.
        ent_coef=args.ent_coef,
        target_kl=args.target_kl,
        log_std_init=args.log_std_init,
        net_arch=[int(x) for x in args.net_arch.split(",") if x],
    )

    # Scatter knobs: override the Strombom cohesion/noise (sub-fields of the
    # nested FlockParams). Kept out of the W&B reward_overrides snapshot above;
    # logged as plain scalars instead.
    if args.flock_cohesion is not None or args.flock_noise is not None:
        from sheepdog_env.flocking import FlockParams
        fp = FlockParams()
        if args.flock_cohesion is not None:
            fp.c = args.flock_cohesion
        if args.flock_noise is not None:
            fp.noise = args.flock_noise
        env_kwargs["flock"] = fp
        config["flock_cohesion"] = fp.c
        config["flock_noise"] = fp.noise

    # Set up wandb (the run owns the tensorboard dir that SB3 writes to, so all
    # of SB3's scalars are synced to W&B automatically).
    run = None
    callback = None
    tb_log = None
    use_wandb = not args.no_wandb and args.wandb_mode != "disabled"
    if use_wandb:
        try:
            import wandb
            from wandb.integration.sb3 import WandbCallback
        except ImportError:
            sys.exit("wandb not installed. `pip install wandb` or pass --no-wandb.")
        run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.run_name,
            config=config,
            sync_tensorboard=True,  # auto-upload SB3's scalar logs to W&B
            monitor_gym=True,  # auto-log rendered episodes if recorded
            save_code=True,
            mode=args.wandb_mode,
        )
        tb_log = f"runs/{run.id}"
        callback = WandbCallback(
            model_save_path=f"models/{run.id}",
            model_save_freq=50_000,
            gradient_save_freq=0,
            verbose=2,
        )

    def _window_frac(t, begin, finish):
        """Progress in [0,1] across a [begin, finish] step window (held at the
        ends). Lets each curriculum have its own start time + length so they can
        be staged sequentially."""
        if finish <= begin:
            return 1.0 if t >= finish else 0.0
        return float(min(1.0, max(0.0, (t - begin) / (finish - begin))))

    class ActiveSheepCurriculum(BaseCallback):
        """Anneal the active flock size start -> end across a [begin, finish]
        window, pushing it to every worker via ``set_active``."""

        def __init__(self, start, end, begin_steps, finish_steps, verbose=0):
            super().__init__(verbose)
            self.start, self.end = start, end
            self.begin, self.finish = int(begin_steps), int(finish_steps)
            self._last = None

        def _set(self, k):
            if k != self._last:
                self.training_env.env_method("set_active", k)
                self._last = k
                self.logger.record("curriculum/n_active", k)

        def _on_training_start(self):
            self._set(self.start)

        def _on_step(self):
            f = _window_frac(self.num_timesteps, self.begin, self.finish)
            self._set(int(round(self.start + (self.end - self.start) * f)))
            return True

    class SpawnCurriculum(BaseCallback):
        """Anneal the flock spawn distance start -> end (near the gap -> far)
        across its own [begin, finish] window."""

        def __init__(self, start, end, begin_steps, finish_steps, verbose=0):
            super().__init__(verbose)
            self.start, self.end = start, end
            self.begin, self.finish = int(begin_steps), int(finish_steps)
            self._last = None

        def _set(self, frac):
            frac = round(frac, 3)
            if frac != self._last:
                self.training_env.env_method("set_spawn_x", frac)
                self._last = frac
                self.logger.record("curriculum/spawn_x_frac", frac)

        def _on_training_start(self):
            self._set(self.start)

        def _on_step(self):
            f = _window_frac(self.num_timesteps, self.begin, self.finish)
            self._set(self.start + (self.end - self.start) * f)
            return True

    class FlockMetrics(BaseCallback):
        """Log mean flock spread and penned fraction per rollout (-> W&B)."""

        def __init__(self, verbose=0):
            super().__init__(verbose)
            self._spread, self._fpen = [], []

        def _on_step(self):
            for info in self.locals.get("infos", []):
                if "flock_spread" in info:
                    self._spread.append(info["flock_spread"])
                    self._fpen.append(info["frac_penned"])
            return True

        def _on_rollout_end(self):
            if self._spread:
                self.logger.record(
                    "flock/spread_mean", sum(self._spread) / len(self._spread)
                )
                self.logger.record(
                    "flock/frac_penned_mean", sum(self._fpen) / len(self._fpen)
                )
                self._spread.clear()
                self._fpen.clear()

    class WatchRollouts(BaseCallback):
        """Every `every` rollouts, play one greedy episode with the current
        policy and save it as a GIF (+ log to W&B as a video). Headless-safe:
        renders to rgb_array, no window required."""

        def __init__(self, every, fps, out_dir, run, verbose=0):
            super().__init__(verbose)
            self.every, self.fps = max(1, every), fps
            self.out_dir, self.run = out_dir, run
            self.env, self.count, self.ok = None, 0, True

        def _on_training_start(self):
            try:
                import imageio  # noqa: F401
            except ImportError:
                print("[watch] DISABLED: imageio not installed (pip install imageio)")
                self.ok = False
                return
            # Watch the GREEDY policy: drop epsilon-random exploration so the GIF
            # reflects what the policy does, not the random-direction jitter.
            wk = dict(env_kwargs)
            wk.pop("dog_random_action_prob", None)
            self.env = make_env(args.obs, args.wolves, args.n_sheep, active_end, wk)()
            self.env.render_mode = "rgb_array"
            env_dim = self.env.observation_space.shape[0]
            model_dim = self.model.observation_space.shape[0]
            if env_dim != model_dim:
                print(f"[watch] DISABLED: watch-env obs ({env_dim}) != policy obs "
                      f"({model_dim}). The rollout env must match the model — check "
                      f"--n-sheep / --n-bushes / --n-rocks (and any --resume model).")
                self.ok = False
                return
            os.makedirs(self.out_dir, exist_ok=True)
            print(f"[watch] recording GIFs to {os.path.abspath(self.out_dir)} "
                  f"(every {self.every} rollouts)")

        def _on_step(self):
            return True

        def _on_rollout_end(self):
            if not self.ok:
                return
            self.count += 1
            # Always record the FIRST rollout (immediate confirmation it works),
            # then every `every` rollouts after that.
            if self.count != 1 and self.count % self.every != 0:
                return
            import imageio
            import numpy as np
            try:
                vn = self.model.get_vec_normalize_env()
                obs, info = self.env.reset()
                frames, t = [], 0
                while True:
                    o = vn.normalize_obs(obs) if vn is not None else obs
                    action, _ = self.model.predict(o, deterministic=not args.watch_stochastic)
                    obs, r, term, trunc, info = self.env.step(action)
                    if t % 2 == 0:
                        frames.append(self.env.render())
                    t += 1
                    if term or trunc:
                        break
                if not frames:
                    print("[watch] no frames captured (episode ended immediately); skipped")
                    return
                path = os.path.join(self.out_dir, f"rollout_{self.num_timesteps}.gif")
                imageio.mimsave(path, frames, fps=self.fps, loop=0)
                print(f"[watch] saved {path}  (penned {info['n_penned']}/{self.env.n_active})")
            except Exception as e:
                import traceback
                print(f"[watch] rollout FAILED, training continues: {e}")
                traceback.print_exc()
                return
            if self.run is not None:
                try:
                    import wandb
                    vid = np.stack(frames).transpose(0, 3, 1, 2)   # T,H,W,C -> T,C,H,W
                    self.run.log({"rollout": wandb.Video(vid, fps=self.fps, format="gif")},
                                 step=self.num_timesteps)
                except Exception as e:
                    print(f"[watch] W&B video log skipped ({e})")

        def _on_training_end(self):
            if self.env is not None:
                self.env.close()

    if args.vec_backend == "batched":
        from sheepdog_env.env import EnvConfig
        from sheepdog_env.vec_env import BatchedSheepdogVecEnv
        if args.obs != "vector" or args.wolves:
            sys.exit("--vec-backend batched supports vector obs only and no wolves; "
                     "use --vec-backend subproc for pixel/wolves.")
        bcfg = EnvConfig(n_sheep=args.n_sheep, n_active=active_start, **env_kwargs)
        venv = BatchedSheepdogVecEnv(args.n_envs, config=bcfg)
        print(f"vec backend: batched ({args.n_envs} worlds in one process)")
    else:
        venv = SubprocVecEnv(
            [
                lambda: Monitor(
                    make_env(
                        args.obs, args.wolves, args.n_sheep, active_start, env_kwargs
                    )()
                )
                for _ in range(args.n_envs)
            ]
        )
    # Normalize rewards (and obs for the vector env). On resume, restore the
    # saved running stats so normalization continues seamlessly.
    if args.resume and os.path.exists(f"{args.resume}_vecnorm.pkl"):
        venv = VecNormalize.load(f"{args.resume}_vecnorm.pkl", venv)
        venv.training = True
    else:
        venv = VecNormalize(
            venv,
            norm_obs=(args.obs == "vector"),
            norm_reward=True,
            clip_obs=10.0,
            clip_reward=10.0,
            gamma=config["gamma"],
        )

    if args.lr_schedule == "linear":
        lr = linear_schedule(args.lr, args.lr_final)
    elif args.lr_schedule == "cosine":
        lr = cosine_schedule(args.lr, args.lr_final)
    else:
        lr = args.lr

    if args.resume:
        # True resume: restore optimizer state, timestep count and schedules.
        model = PPO.load(args.resume, env=venv, device=args.device,
                         tensorboard_log=tb_log)
        print(f"resuming from {args.resume}.zip at {model.num_timesteps} timesteps")
    else:
        policy = "MlpPolicy" if args.obs == "vector" else "CnnPolicy"
        model = PPO(
            policy,
            venv,
            verbose=1,
            device=args.device,
            learning_rate=lr,
            n_epochs=config["n_epochs"],
            clip_range=config["clip_range"],
            n_steps=config["n_steps"],
            batch_size=config["batch_size"],
            gae_lambda=config["gae_lambda"],
            gamma=config["gamma"],
            ent_coef=config["ent_coef"],
            target_kl=config["target_kl"],
            policy_kwargs=dict(log_std_init=config["log_std_init"],
                               net_arch=config["net_arch"]),
            tensorboard_log=tb_log,
        )

    # Warm-start from a checkpoint (e.g. the CEM->BC model): copy its policy
    # weights and seed VecNormalize with its observation stats.
    if args.init_from and not args.resume:
        import pickle

        bc = PPO.load(args.init_from, device=model.device)
        model.policy.load_state_dict(bc.policy.state_dict())
        vp = f"{args.init_from}_vecnorm.pkl"
        if os.path.exists(vp):
            with open(vp, "rb") as f:
                bcvn = pickle.load(f)
            venv.obs_rms = bcvn.obs_rms
            venv.ret_rms = bcvn.ret_rms
        print(f"warm-started policy + normalizer from {args.init_from}")

    # Assemble callbacks: flock metrics + curricula (if any) + watch + W&B.
    callbacks = [FlockMetrics()]
    if args.checkpoint_freq > 0:
        # CheckpointCallback fires once per rollout-step across all envs, so to
        # checkpoint every N *env* steps we divide by the number of envs.
        per_env = max(args.checkpoint_freq // args.n_envs, 1)
        callbacks.append(
            CheckpointCallback(
                save_freq=per_env,
                save_path=args.checkpoint_dir,
                name_prefix=os.path.basename(args.save),
                save_vecnormalize=True,   # save the VecNormalize stats too
                verbose=1,
            )
        )
        print(f"checkpoints: every {args.checkpoint_freq} env steps -> "
              f"{args.checkpoint_dir}/ (model + vecnormalize)")
    if args.watch:
        callbacks.append(WatchRollouts(args.watch_every, args.watch_fps,
                                       args.watch_dir, run))
        print(f"watch: recording a rollout GIF every {args.watch_every} rollouts "
              f"-> {args.watch_dir}/")
    T = args.timesteps
    active_len = (args.active_frac if args.active_frac is not None else args.curriculum_frac)
    spawn_len = (args.spawn_frac if args.spawn_frac is not None else args.curriculum_frac)
    spawn_begin = args.spawn_begin_frac
    if use_curriculum:
        callbacks.append(
            ActiveSheepCurriculum(
                active_start, active_end, 0.0, active_len * T, verbose=1,
            )
        )
        print(
            f"curriculum: active flock {active_start} -> {active_end} over "
            f"[0%, {active_len:.0%}] of training"
        )
    if use_spawn_curriculum:
        callbacks.append(
            SpawnCurriculum(
                args.spawn_start, args.spawn_end,
                spawn_begin * T, (spawn_begin + spawn_len) * T,
            )
        )
        print(
            f"curriculum: flock spawn x {args.spawn_start} -> {args.spawn_end} "
            f"(near gap -> far) over [{spawn_begin:.0%}, {spawn_begin + spawn_len:.0%}] "
            f"of training"
        )
    if callback is not None:
        callbacks.append(callback)
    cb = CallbackList(callbacks) if callbacks else None

    model.learn(total_timesteps=args.timesteps, callback=cb,
                reset_num_timesteps=not bool(args.resume))
    model.save(args.save)
    # Save the normalization statistics next to the model — eval/play need them.
    vecnorm_path = f"{args.save}_vecnorm.pkl"
    venv.save(vecnorm_path)
    if run is not None:
        try:
            venv.save(os.path.join(f"models/{run.id}", "vecnorm.pkl"))
        except Exception:
            pass
    print(f"saved model -> {args.save}.zip  and normalizer -> {vecnorm_path}")

    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()
