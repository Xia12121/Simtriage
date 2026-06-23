"""Task 6 — self-evolving, transferable Query-Value Library (M3).

METHOD_DESIGN §3 M3 / §4:
* Aggregate (phi-prototype -> empirical VoQ statistics + confidence) into a library.
* Keys live on instance-INVARIANT phi, so a QVL built on a set of seeds/boards transfers
  to held-out procedural instances.
* Retention uses a PUBLIC Hoeffding confidence bound (hygiene, not the contribution):
  keep a prototype as "worth querying" only when the VoQ lower-confidence-bound stays
  above its cost; mark "don't query" when the upper bound is below cost.

Two roles in the pipeline:
1. ``predict_prior(phi)`` -> optional VoQ estimate that seeds cold-start VoQ predictions
   (M1) on a fresh instance — this is the transfer mechanism.
2. ``verdict(phi, cost)`` -> {"worth_query","dont_query","uncertain"} for library hygiene
   and as an optional confident gate override.

Hoeffding (two-sided, level alpha):  eps = R * sqrt( ln(2/alpha) / (2 n) ),  R = VoQ range.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class Prototype:
    centroid: np.ndarray            # running mean of phi (the instance-invariant key)
    count: int = 0
    voq_mean: float = 0.0           # running mean of y (VoQ label)
    voq_m2: float = 0.0             # Welford M2 for variance
    cost_mean: float = 0.0          # running mean of lam*c at this prototype
    verdict: str = "uncertain"

    def var(self) -> float:
        return self.voq_m2 / self.count if self.count > 1 else 0.0


class QVL:
    def __init__(self, cfg: Optional[dict] = None):
        qcfg = (cfg or {}).get("simtriage", {}).get("qvl", {}) if cfg and "simtriage" in cfg else (cfg or {}).get("qvl", {})
        qcfg = qcfg or {}
        self.alpha = float(qcfg.get("hoeffding_alpha", 0.1))
        self.min_count = int(qcfg.get("min_count", 8))
        self.radius = float(qcfg.get("proto_radius", 0.75))
        self.R = float(qcfg.get("voq_range", 2.0))
        self.prototypes: list[Prototype] = []

    # ----------------------------------------------------------------- update
    def _nearest(self, phi: np.ndarray) -> tuple[Optional[int], float]:
        if not self.prototypes:
            return None, math.inf
        d = [float(np.linalg.norm(p.centroid - phi)) for p in self.prototypes]
        i = int(np.argmin(d))
        return i, d[i]

    def update(self, phi: np.ndarray, y: float, cost: float = 0.0) -> int:
        """Assign (phi, y) to the nearest prototype within ``radius`` or spawn a new one.
        Returns the prototype index. Welford running mean/var for VoQ."""
        phi = np.asarray(phi, dtype=float)
        i, dist = self._nearest(phi)
        if i is None or dist > self.radius:
            p = Prototype(centroid=phi.copy(), count=1, voq_mean=float(y), voq_m2=0.0, cost_mean=float(cost))
            self.prototypes.append(p)
            return len(self.prototypes) - 1
        p = self.prototypes[i]
        p.count += 1
        # online centroid
        p.centroid += (phi - p.centroid) / p.count
        # Welford for VoQ
        delta = y - p.voq_mean
        p.voq_mean += delta / p.count
        p.voq_m2 += delta * (y - p.voq_mean)
        # running cost mean
        p.cost_mean += (cost - p.cost_mean) / p.count
        return i

    # ----------------------------------------------------------- Hoeffding
    def _eps(self, n: int) -> float:
        if n <= 0:
            return math.inf
        return self.R * math.sqrt(math.log(2.0 / self.alpha) / (2.0 * n))

    def confidence_bounds(self, p: Prototype) -> tuple[float, float]:
        eps = self._eps(p.count)
        return p.voq_mean - eps, p.voq_mean + eps

    def consolidate(self, prune: bool = False) -> None:
        """Recompute verdicts via Hoeffding. ``verdict`` compares VoQ bounds against the
        prototype's own observed cost (mean lam*c). Optionally prune low-evidence prototypes.
        """
        for p in self.prototypes:
            lcb, ucb = self.confidence_bounds(p)
            thr = p.cost_mean
            if p.count >= self.min_count and lcb > thr:
                p.verdict = "worth_query"
            elif p.count >= self.min_count and ucb < thr:
                p.verdict = "dont_query"
            else:
                p.verdict = "uncertain"
        if prune:
            # hygiene: drop prototypes that stay uncertain with very wide intervals and low count
            self.prototypes = [
                p for p in self.prototypes
                if not (p.count < max(2, self.min_count // 2) and self._eps(p.count) > 2 * self.R)
            ]

    # --------------------------------------------------------------- queries
    def predict_prior(self, phi: np.ndarray) -> Optional[float]:
        """Transferred cold-start VoQ estimate: mean VoQ of the nearest TRUSTED prototype."""
        phi = np.asarray(phi, dtype=float)
        i, dist = self._nearest(phi)
        if i is None or dist > self.radius:
            return None
        p = self.prototypes[i]
        if p.count < self.min_count:
            return None
        return float(p.voq_mean)

    def verdict(self, phi: np.ndarray, cost: float) -> str:
        """Confident library verdict for this phi at the given cost (= lam*c). For cold-start
        gating: returns 'worth_query'/'dont_query' only when Hoeffding is confident."""
        phi = np.asarray(phi, dtype=float)
        i, dist = self._nearest(phi)
        if i is None or dist > self.radius:
            return "uncertain"
        p = self.prototypes[i]
        if p.count < self.min_count:
            return "uncertain"
        lcb, ucb = self.confidence_bounds(p)
        if lcb > cost:
            return "worth_query"
        if ucb < cost:
            return "dont_query"
        return "uncertain"

    # ------------------------------------------------------------ persistence
    def to_dict(self) -> dict:
        return {
            "alpha": self.alpha,
            "min_count": self.min_count,
            "radius": self.radius,
            "R": self.R,
            "prototypes": [
                {
                    "centroid": p.centroid.tolist(),
                    "count": p.count,
                    "voq_mean": p.voq_mean,
                    "voq_m2": p.voq_m2,
                    "cost_mean": p.cost_mean,
                    "verdict": p.verdict,
                }
                for p in self.prototypes
            ],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "QVL":
        q = cls()
        q.alpha = d["alpha"]
        q.min_count = d["min_count"]
        q.radius = d["radius"]
        q.R = d["R"]
        q.prototypes = [
            Prototype(
                centroid=np.asarray(pd["centroid"], dtype=float),
                count=pd["count"],
                voq_mean=pd["voq_mean"],
                voq_m2=pd["voq_m2"],
                cost_mean=pd.get("cost_mean", 0.0),
                verdict=pd.get("verdict", "uncertain"),
            )
            for pd in d["prototypes"]
        ]
        return q

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "QVL":
        with open(path) as f:
            return cls.from_dict(json.load(f))

    def __len__(self) -> int:
        return len(self.prototypes)
