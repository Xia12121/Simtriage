"""Task 4 — VoQ predictor g_hat_theta(phi) with online (training-free) update (M1).

METHOD_DESIGN §4.2:
    g_hat_theta(phi(s)) ~ g(s)            # predicted value of querying at s
    label y(s) ~ Qhat_ref(s, a_ref) - Qhat_ref(s, a_0)
    theta <- online regression / bandit update(phi, y)        # e.g. incremental ridge

This is a cheap online regressor. **No LLM / metacontroller training** ever happens.
Cold start returns an OPTIMISTIC value so the agent queries (and collects labels) early
(SPEC §2.2). Optionally consults a QVL cold-start prior (M3) for cross-instance transfer.

``theta`` (the only evolved object) = this regressor's parameters + (via simtriage) the QVL.
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np


class VoQRidge:
    """Incremental ridge regression (closed-form normal equations). SPEC §2.2."""

    def __init__(self, dim: int, l2: float = 1.0, optimistic: float = 1.0, warmup_n: int = 20):
        self.dim = int(dim)
        self.l2 = float(l2)
        self.opt = float(optimistic)
        self.warmup_n = int(warmup_n)
        self.A = self.l2 * np.eye(self.dim)
        self.b = np.zeros(self.dim)
        self.n = 0
        self.cold_start_prior: Optional[Callable[[np.ndarray], Optional[float]]] = None

    def _weights(self) -> np.ndarray:
        return np.linalg.solve(self.A, self.b)

    def predict(self, phi: np.ndarray) -> float:
        phi = np.asarray(phi, dtype=float)
        if self.n < self.warmup_n:
            # Try a transferred prior first (M3); else stay optimistic to encourage querying.
            if self.cold_start_prior is not None:
                p = self.cold_start_prior(phi)
                if p is not None:
                    return float(p)
            return self.opt
        return float(self._weights() @ phi)

    def online_update(self, phi: np.ndarray, y: float) -> None:
        phi = np.asarray(phi, dtype=float)
        self.A += np.outer(phi, phi)
        self.b += float(y) * phi
        self.n += 1

    # introspection (calibration metrics)
    def state_dict(self) -> dict:
        return {"A": self.A.copy(), "b": self.b.copy(), "n": self.n, "type": "ridge"}


class VoQKNN:
    """Prototype / kNN regressor over phi (the alternative in METHOD_DESIGN §4.2)."""

    def __init__(self, dim: int, k: int = 8, optimistic: float = 1.0, warmup_n: int = 20):
        self.dim = int(dim)
        self.k = int(k)
        self.opt = float(optimistic)
        self.warmup_n = int(warmup_n)
        self._X: list[np.ndarray] = []
        self._y: list[float] = []
        self.cold_start_prior: Optional[Callable[[np.ndarray], Optional[float]]] = None

    @property
    def n(self) -> int:
        return len(self._y)

    def predict(self, phi: np.ndarray) -> float:
        phi = np.asarray(phi, dtype=float)
        if self.n < self.warmup_n:
            if self.cold_start_prior is not None:
                p = self.cold_start_prior(phi)
                if p is not None:
                    return float(p)
            return self.opt
        X = np.vstack(self._X)
        d = np.linalg.norm(X - phi[None, :], axis=1)
        idx = np.argsort(d)[: self.k]
        w = 1.0 / (d[idx] + 1e-6)
        return float(np.average(np.asarray(self._y)[idx], weights=w))

    def online_update(self, phi: np.ndarray, y: float) -> None:
        self._X.append(np.asarray(phi, dtype=float))
        self._y.append(float(y))

    def state_dict(self) -> dict:
        return {"n": self.n, "type": "knn"}


def make_voq(dim: int, cfg: dict):
    vcfg = cfg.get("simtriage", {}).get("voq", {}) if "simtriage" in cfg else cfg.get("voq", {})
    vcfg = vcfg or {}
    typ = vcfg.get("type", "ridge")
    if typ == "ridge":
        return VoQRidge(
            dim,
            l2=vcfg.get("l2", 1.0),
            optimistic=vcfg.get("optimistic", 1.0),
            warmup_n=vcfg.get("warmup_n", 20),
        )
    if typ == "knn":
        return VoQKNN(
            dim,
            k=vcfg.get("knn_k", 8),
            optimistic=vcfg.get("optimistic", 1.0),
            warmup_n=vcfg.get("warmup_n", 20),
        )
    raise ValueError(f"unknown voq type {typ!r}")
