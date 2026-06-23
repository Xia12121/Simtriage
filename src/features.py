"""Task 3 — instance-INVARIANT decision-context features phi(s) (M1 input).

METHOD_DESIGN §4.2:
    phi(s) = [ margin, H_prior, log|valid|, phase, irrev, ... ]
    margin    = top-1 minus top-2 prior preference
    H_prior   = entropy of the prior over the candidate set A(s)
    irrev     = irreversibility proxy (does the action shrink the future legal set?)

CRITICAL CONSTRAINT (SPEC Task 3 / METHOD_DESIGN desideratum 4):
phi MUST NOT contain instance-specific content (no card names, board coordinates,
resource totals). Otherwise the QVL keys leak instance info and transfer fails.
Every feature below is a dimensionless scalar derived from prior *dispersion* and
white-box *structure* — portable across seeds/boards/games.

The ``instance_dependent=True`` switch is ONLY for the ablation that demonstrates where
transferability comes from (SPEC Task 8); it appends leaky features and is never used in
the main method.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence

import numpy as np


def _softmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return x
    z = x - np.max(x)
    e = np.exp(z)
    s = e.sum()
    return e / s if s > 0 else np.full_like(e, 1.0 / e.size)


# Canonical, instance-INVARIANT feature schema. "bias" is appended last as the intercept.
INVARIANT_FEATURES = [
    "margin",          # raw preference gap pref[0]-pref[1] (METHOD §4.2 / SPEC Task 3)
    "entropy",         # normalized entropy of the prior over A(s) in [0,1]
    "log_valid",       # log(1+|valid|) / 10  (branching factor, scale-free-ish)
    "phase",           # categorical game phase mapped to [0,1] (generic, not content)
    "irrev",           # irreversibility proxy in [0,1]
    "progress",        # normalized episode progress in [0,1]
    "branch_balance",  # 1 - p_top1 : how much prior mass is off the favorite
]


class FeatureExtractor:
    """Builds a fixed-dimension phi vector. ``dim`` includes the bias term."""

    def __init__(self, cfg: Optional[dict] = None):
        cfg = cfg or {}
        fcfg = cfg.get("features", {}) if "features" in cfg else cfg
        use = fcfg.get("use", INVARIANT_FEATURES)
        # keep only known invariant features, preserving canonical order
        self.use = [f for f in INVARIANT_FEATURES if f in use]
        if not self.use:
            self.use = list(INVARIANT_FEATURES)
        self.instance_dependent = bool(fcfg.get("instance_dependent", False))
        # ablation-only leaky features (instance-specific). Never used in the main method.
        self._inst_names = ["inst_progress_raw", "inst_valid_raw", "inst_agent_id_hash"]
        self.names = list(self.use)
        if self.instance_dependent:
            self.names = self.names + self._inst_names
        self.names = self.names + ["bias"]

    @property
    def dim(self) -> int:
        return len(self.names)

    # ------------------------------------------------------------------ build
    def __call__(self, state, A: Sequence, prefs: Sequence[float], env=None) -> np.ndarray:
        return self.features(state, A, prefs, env)

    def features(self, state, A: Sequence, prefs: Sequence[float], env=None) -> np.ndarray:
        prefs = np.asarray(list(prefs), dtype=float) if prefs is not None else np.zeros(len(A))
        probs = _softmax(prefs) if prefs.size else np.array([1.0])

        # margin = raw top-1 minus top-2 preference (METHOD §4.2 / SPEC Task 3). Use the RAW
        # prefs, not softmax probs (softmax saturates on logit-scale inputs and collapses the
        # dispersion signal). Softmax is reserved for entropy / branch_balance below.
        if prefs.size >= 2:
            sp = np.sort(prefs)[::-1][:2]
            margin = float(sp[0] - sp[1])
        else:
            margin = 1.0

        # entropy normalized to [0,1] by log(|A|)
        if probs.size >= 2:
            ent = float(-(probs * np.log(probs + 1e-12)).sum())
            entropy = ent / math.log(probs.size)
        else:
            entropy = 0.0

        n_valid = self._n_valid(state, A, env)
        log_valid = math.log(1.0 + n_valid) / 10.0

        phase = self._phase(state, env)
        progress = self._progress(state, env)
        irrev = self._irrev(state, A, env)
        branch_balance = float(1.0 - probs.max()) if probs.size else 0.0

        values = {
            "margin": margin,
            "entropy": entropy,
            "log_valid": log_valid,
            "phase": phase,
            "irrev": irrev,
            "progress": progress,
            "branch_balance": branch_balance,
        }
        vec = [values[name] for name in self.use]

        if self.instance_dependent:
            vec += self._instance_dependent(state, n_valid)

        vec.append(1.0)  # bias / intercept
        return np.asarray(vec, dtype=float)

    # --------------------------------------------------------------- pieces
    def _n_valid(self, state, A, env) -> int:
        if env is not None:
            try:
                return len(env.valid_actions(state))
            except Exception:
                pass
        meta_n = state.meta.get("n_valid") if hasattr(state, "meta") else None
        return int(meta_n) if meta_n else len(A)

    def _phase(self, state, env) -> float:
        # Generic phase in [0,1]. Prefer an env-provided categorical phase; fall back to progress.
        if hasattr(state, "meta") and "phase" in state.meta:
            return float(state.meta["phase"])
        return self._progress(state, env)

    def _progress(self, state, env) -> float:
        if env is not None and hasattr(env, "phase_progress"):
            try:
                return float(env.phase_progress(state))
            except Exception:
                pass
        if hasattr(state, "meta"):
            return float(state.meta.get("progress", 0.0))
        return 0.0

    def _irrev(self, state, A, env) -> float:
        if env is not None and len(A) > 0:
            try:
                return float(env.action_irreversibility(state, A[0]))
            except Exception:
                return 0.0
        return 0.0

    def _instance_dependent(self, state, n_valid) -> list:
        # ABLATION ONLY: deliberately leaky, instance-specific values.
        progress_raw = float(state.meta.get("turn", 0)) if hasattr(state, "meta") else 0.0
        agent_hash = (hash(str(getattr(state, "agent_id", 0))) % 997) / 997.0
        return [progress_raw, float(n_valid), agent_hash]
