"""Shared helpers for running a trained policy with the right obs normalization.

When training uses VecNormalize, the policy expects *normalized* observations.
At eval/play time we step the raw env (so we keep access to info, dog_pos, etc.)
and normalize each observation with the saved statistics before `predict`.
"""

from __future__ import annotations

import os
import pickle

import numpy as np


def find_vecnorm(model_path: str, explicit: str | None = None):
    """Locate the saved VecNormalize stats for a model, or None."""
    if explicit:
        return explicit if os.path.exists(explicit) else None
    base = model_path[:-4] if model_path.endswith(".zip") else model_path
    candidates = [
        f"{base}_vecnorm.pkl",                      # train_sb3 --save sibling
        os.path.join(os.path.dirname(base), "vecnorm.pkl"),  # run-dir layout
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return None


def load_obs_normalizer(model_path: str, explicit: str | None = None):
    """Return (normalize_fn, info_str).

    normalize_fn maps a raw observation to the normalized form the policy was
    trained on. If no stats are found (or obs weren't normalized), it's identity.
    """
    path = find_vecnorm(model_path, explicit)
    if path is None:
        return (lambda o: o), "no VecNormalize stats found -> using raw observations"
    with open(path, "rb") as f:
        vn = pickle.load(f)            # VecNormalize.__getstate__ strips the venv
    if not getattr(vn, "norm_obs", False):
        return (lambda o: o), f"loaded {path} (obs not normalized)"
    mean = vn.obs_rms.mean.astype(np.float32)
    var = vn.obs_rms.var.astype(np.float32)
    eps = float(vn.epsilon)
    clip = float(vn.clip_obs)

    def norm(o):
        o = np.asarray(o, dtype=np.float32)
        return np.clip((o - mean) / np.sqrt(var + eps), -clip, clip).astype(np.float32)

    return norm, f"loaded obs normalizer from {path}"
