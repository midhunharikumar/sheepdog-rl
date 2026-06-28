"""Sheepdog Herding RL environment."""

from .env import SheepdogHerdingEnv, EnvConfig
from .flocking import FlockParams

__all__ = ["SheepdogHerdingEnv", "EnvConfig", "FlockParams"]

# Batched, fully-vectorised SB3 VecEnv. Optional: it needs stable-baselines3,
# so importing the package stays cheap and dependency-free when it's absent.
try:
    from .vec_env import BatchedSheepdogVecEnv

    __all__.append("BatchedSheepdogVecEnv")
except Exception:  # pragma: no cover - stable-baselines3 not installed
    pass

# Register with Gymnasium so you can do gym.make("SheepdogHerding-v0").
try:
    from gymnasium.envs.registration import register

    register(
        id="SheepdogHerding-v0",
        entry_point="sheepdog_env.env:SheepdogHerdingEnv",
        max_episode_steps=None,  # the env handles its own day/night horizon
    )
except Exception:  # pragma: no cover
    pass
