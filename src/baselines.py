"""Task 8 — baselines and ablations.

Baselines (METHOD_DESIGN §5 / SPEC §1 Task 8):
  * never_query      : always act on the prior a_0  (~ Voyager: environment as black box)
  * always_query     : always run the depth-d top-m search  (always-search; burns budget)
  * fixed_threshold  : query when the prior margin is small (hand-tuned threshold gate)
  * random_gate      : query with a fixed probability matched to SimTriage's query rate

Ablations:
  * minus_m2_imagine : SimTriage with imagine=True (real rollout -> prior imagination)
  * minus_m3_from_scratch : SimTriage with transfer=False and a fresh QVL per level
  * phi_instance_dependent : SimTriage with features.instance_dependent=True

All baselines share SimTriage's accounting (sim steps, return, U) so they land as comparable
points on the (return vs sim-steps) frontier (SPEC Task 8 / Task 9).
"""

from __future__ import annotations

import copy
import time
from typing import Optional

import numpy as np

from .features import FeatureExtractor
from .policy import make_policy, make_rollout_policy
from .query import run_query
from .simtriage import EpisodeResult, SimTriage


def _margin(prefs) -> float:
    p = np.asarray(list(prefs), dtype=float)
    if p.size < 2:
        return 1.0
    e = np.exp(p - p.max())
    s = e / e.sum()
    top2 = np.sort(s)[::-1][:2]
    return float(top2[0] - top2[1])


class BaselineLoop:
    """Shared gated loop with no VoQ learning and no reference labels."""

    name = "baseline"

    def __init__(self, cfg: dict, lam: float = 0.05, policy=None, game: str = "catan", seed: int = 12345):
        self.cfg = cfg
        self.game = game
        scfg = cfg.get("simtriage", {})
        self.lam = float(lam)
        self.m = int(scfg.get("m", 3))
        self.K = int((scfg.get("K", {}) or {}).get(game, (scfg.get("K", {}) or {}).get("catan", 5)))
        self.d = (scfg.get("d", {}) or {}).get(game, (scfg.get("d", {}) or {}).get("catan", "to_turn_end"))
        self.use_crn = bool(scfg.get("reference", {}).get("use_crn", True))
        self.policy = policy if policy is not None else make_policy(cfg)
        self.rollout_policy = make_rollout_policy(cfg, prior=self.policy)
        self.feat = FeatureExtractor(cfg)
        self._rng = np.random.default_rng(seed)
        self._crn_rng = np.random.default_rng(seed + 7)

    def should_query(self, env, state, A, prefs, phi) -> bool:
        raise NotImplementedError

    def run_episode(self, env, seed: int) -> EpisodeResult:
        env.set_query_budget(self.m, self.K, self.d)
        t0 = time.time()
        env.reset(seed)
        env.reset_sim_steps()
        n_dec = n_q = n_hit = 0
        done = env.done()
        guard = 0
        max_guard = int(self.cfg.get("env", {}).get("catan", {}).get("max_ticks", 2000))
        while not done:
            state = env.current_state()
            A, prefs = self.policy.top_candidates(env, state, self.m)
            if not A:
                break
            phi = self.feat(state, A, prefs, env)
            a0 = A[0]
            n_dec += 1
            if self.should_query(env, state, A, prefs, phi):
                n_q += 1
                crn_base = int(self._crn_rng.integers(0, 2**31 - 1))
                a_hat, _q = run_query(env, state, A, self.rollout_policy, self.K, self.d, self.use_crn, crn_base)
                action = a_hat if a_hat is not None else a0
                if action is not a0 and action != a0:
                    n_hit += 1
            else:
                action = a0
            _s, _r, done, _i = env.step(action)
            guard += 1
            if guard > max_guard:
                break
        G = env.episode_return()
        C = env.sim_steps
        return EpisodeResult(
            seed=seed, lam=self.lam, G=float(G), C=int(C), U=float(G - self.lam * C),
            n_decisions=n_dec, n_queries=n_q, n_hits=n_hit, n_reference=0,
            calib=[], wallclock_s=time.time() - t0,
        )

    def evolve(self, env, seeds, **kw) -> list[EpisodeResult]:
        return [self.run_episode(env, s) for s in seeds]


class NeverQuery(BaselineLoop):
    name = "never_query"

    def should_query(self, env, state, A, prefs, phi) -> bool:
        return False


class AlwaysQuery(BaselineLoop):
    name = "always_query"

    def should_query(self, env, state, A, prefs, phi) -> bool:
        return True


class FixedThreshold(BaselineLoop):
    name = "fixed_threshold"

    def __init__(self, cfg, lam=0.05, margin_thresh: float = 0.2, **kw):
        super().__init__(cfg, lam=lam, **kw)
        self.margin_thresh = float(margin_thresh)

    def should_query(self, env, state, A, prefs, phi) -> bool:
        # query only when the prior is uncertain (small top-1/top-2 margin)
        return _margin(prefs) < self.margin_thresh


class RandomGate(BaselineLoop):
    name = "random_gate"

    def __init__(self, cfg, lam=0.05, query_prob: float = 0.3, **kw):
        super().__init__(cfg, lam=lam, **kw)
        self.query_prob = float(query_prob)

    def should_query(self, env, state, A, prefs, phi) -> bool:
        return self._rng.random() < self.query_prob


# ------------------------------------------------------------------ ablations
def make_minus_m2(cfg, lam, **kw) -> SimTriage:
    """-M2: replace real rollout in the acting query with prior imagination."""
    return SimTriage(cfg, lam=lam, imagine=True, transfer=True, **kw)


def make_minus_m3(cfg, lam, **kw) -> SimTriage:
    """-M3: no QVL transfer prior; fresh library per level (from-scratch)."""
    return SimTriage(cfg, lam=lam, imagine=False, transfer=False, **kw)


def make_phi_instance_dependent(cfg, lam, **kw) -> SimTriage:
    """phi-instance-dependent ablation: leak instance info into features."""
    cfg2 = copy.deepcopy(cfg)
    cfg2.setdefault("features", {})["instance_dependent"] = True
    return SimTriage(cfg2, lam=lam, imagine=False, transfer=True, **kw)


def make_baseline(name: str, cfg: dict, lam: float, **kw):
    name = name.lower()
    table = {
        "never_query": NeverQuery,
        "always_query": AlwaysQuery,
        "fixed_threshold": FixedThreshold,
        "random_gate": RandomGate,
    }
    if name in table:
        return table[name](cfg, lam=lam, **kw)
    if name == "minus_m2_imagine":
        return make_minus_m2(cfg, lam, **kw)
    if name == "minus_m3_from_scratch":
        return make_minus_m3(cfg, lam, **kw)
    if name == "phi_instance_dependent":
        return make_phi_instance_dependent(cfg, lam, **kw)
    raise ValueError(f"unknown baseline/ablation {name!r}")
