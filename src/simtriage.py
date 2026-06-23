"""Task 7 — SimTriage integration: Algorithm 1 (decide) + Algorithm 2 (self-evolution).

Ties together M1 (VoQ predictor), M2 (gate + real-simulator query + reference label) and
M3 (transferable QVL). Frozen policy; the ONLY evolved object is ``theta`` = VoQ predictor
+ QVL (METHOD_DESIGN desideratum 1).

Ablation flags (so the same class expresses the method and its ablations, SPEC Task 8):
  * ``imagine=True``  -> -M2: the acting query uses the prior's imagination instead of real
    rollout (reference labels stay real, to keep the VoQ learning signal honest).
  * ``transfer=False``-> -M3: do not seed cold-start VoQ from the QVL (from-scratch).
  * feature ``instance_dependent`` -> phi-instance-dependent ablation (set in FeatureExtractor).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .features import FeatureExtractor
from .policy import Policy, RolloutPolicy, make_policy, make_rollout_policy
from .qvl import QVL
from .query import gate, reference_voq, run_query
from .voq import make_voq


@dataclass
class StepInfo:
    queried: bool
    changed: bool          # query hit: a_hat != a_0
    g_hat: float
    cost: float
    n_valid: int
    reference: bool


@dataclass
class EpisodeResult:
    seed: int
    lam: float
    G: float                       # game return
    C: int                         # total simulator steps (cost term)
    U: float                       # utility G - lam*C
    n_decisions: int
    n_queries: int
    n_hits: int                    # queries that changed the decision
    n_reference: int
    calib: list = field(default_factory=list)   # (g_hat_pred_before_update, y) pairs
    wallclock_s: float = 0.0

    def to_row(self) -> dict:
        return {
            "seed": self.seed,
            "lam": self.lam,
            "return": self.G,
            "sim_steps": self.C,
            "utility": self.U,
            "n_decisions": self.n_decisions,
            "n_queries": self.n_queries,
            "n_hits": self.n_hits,
            "n_reference": self.n_reference,
            "query_rate": self.n_queries / max(1, self.n_decisions),
            "hit_rate": self.n_hits / max(1, self.n_queries),
            "wallclock_s": self.wallclock_s,
        }


class SimTriage:
    def __init__(
        self,
        cfg: dict,
        policy: Optional[Policy] = None,
        lam: Optional[float] = None,
        qvl: Optional[QVL] = None,
        imagine: bool = False,
        transfer: bool = True,
        game: str = "catan",
        seed: int = 12345,
    ):
        self.cfg = cfg
        self.game = game
        scfg = cfg.get("simtriage", {})
        self.lam = float(scfg.get("lam_default", 0.05)) if lam is None else float(lam)
        self.m = int(scfg.get("m", 3))
        self.K = int((scfg.get("K", {}) or {}).get(game, (scfg.get("K", {}) or {}).get("catan", 5)))
        self.d = (scfg.get("d", {}) or {}).get(game, (scfg.get("d", {}) or {}).get("catan", "to_turn_end"))
        ref = scfg.get("reference", {})
        self.K_ref = int((ref.get("K", {}) or {}).get(game, (ref.get("K", {}) or {}).get("catan", 20)))
        self.d_ref = (ref.get("d", {}) or {}).get(game, (ref.get("d", {}) or {}).get("catan", "to_game_end"))
        self.use_crn = bool(ref.get("use_crn", True))
        self.rho_start = float(scfg.get("rho_start", 0.10))
        self.rho_end = float(scfg.get("rho_end", 0.02))
        self.rho_anneal = int(scfg.get("rho_anneal_instances", 50))

        self.imagine = bool(imagine)
        self.transfer = bool(transfer)

        self.policy = policy if policy is not None else make_policy(cfg)
        self.rollout_policy = make_rollout_policy(cfg, prior=self.policy)
        self.feat = FeatureExtractor(cfg)
        self.voq = make_voq(self.feat.dim, cfg)
        self.qvl = qvl if qvl is not None else QVL(cfg)

        # M3 transfer: seed cold-start VoQ from the QVL prior unless ablated.
        if self.transfer and scfg.get("qvl", {}).get("enable_transfer_prior", True):
            self.voq.cold_start_prior = self.qvl.predict_prior

        self._rng = np.random.default_rng(seed)
        self._crn_rng = np.random.default_rng(seed + 1)
        self._instance_idx = 0  # for rho annealing

    # ---- theta = the evolved object (VoQ predictor + QVL) --------------------
    @property
    def theta(self):
        return {"voq": self.voq, "qvl": self.qvl}

    def rho(self) -> float:
        if self.rho_anneal <= 0:
            return self.rho_end
        frac = min(1.0, self._instance_idx / self.rho_anneal)
        return self.rho_start + (self.rho_end - self.rho_start) * frac

    # =============================================== Algorithm 1: single decide
    def decide(self, env, state, learn_rho: Optional[float] = None) -> tuple[object, Optional[tuple], StepInfo]:
        A, prefs = self.policy.top_candidates(env, state, self.m)
        if not A:
            return None, None, StepInfo(False, False, 0.0, 0.0, 0, False)

        phi = self.feat(state, A, prefs, env)
        a0 = A[0]
        cost = env.query_cost(state)
        g_hat = self.voq.predict(phi)

        # gate (M2). g_hat already reflects the QVL transfer prior during cold start (M3).
        queried = gate(g_hat, self.lam, cost)
        if queried:
            crn_base = int(self._crn_rng.integers(0, 2**31 - 1))
            a_hat = self._act_query(env, state, A, crn_base)
            action = a_hat if a_hat is not None else a0
        else:
            action = a0
        changed = queried and (action is not a0) and (action != a0)

        # reference label (self-supervised). Real high-budget rollout even under -M2.
        label = None
        reference = False
        rho = self.rho() if learn_rho is None else learn_rho
        if self._rng.random() < rho:
            reference = True
            crn_base = int(self._crn_rng.integers(0, 2**31 - 1))
            y, _a_ref, _q = reference_voq(
                env, state, A, self.rollout_policy, self.K_ref, self.d_ref, self.use_crn, crn_base
            )
            # QVL stores the PRICE-WEIGHTED cost lam*c (value units), so its Hoeffding
            # retain/eject compares like-with-like against the VoQ label y (METHOD §3 M3).
            # StepInfo.cost below keeps the raw step count for accounting (U = G - lam*C).
            label = (phi, float(y), float(self.lam * cost))

        info = StepInfo(
            queried=queried, changed=bool(changed), g_hat=float(g_hat),
            cost=float(cost), n_valid=len(env.valid_actions(state)), reference=reference,
        )
        return action, label, info

    def _act_query(self, env, state, A, crn_base):
        if self.imagine:
            # -M2 ablation: imagine instead of real rollout (no simulator steps consumed).
            q = [self.policy.imagine_q(env, state, a) for a in A]
            return A[int(np.argmax(q))]
        a_hat, _q = run_query(env, state, A, self.rollout_policy, self.K, self.d, self.use_crn, crn_base)
        return a_hat

    # =============================================== Algorithm 2: run + evolve
    def run_episode(self, env, seed: int, learn: bool = True,
                    reference_rate: Optional[float] = None) -> EpisodeResult:
        """``reference_rate`` overrides the rho schedule. Set 0.0 for frozen evaluation
        (no further label collection / reference cost) when tracing the deployed frontier."""
        env.set_query_budget(self.m, self.K, self.d)
        t0 = time.time()
        state = env.reset(seed)
        env.reset_sim_steps()  # reset (reset() already did, but be explicit)

        n_dec = n_q = n_hit = n_ref = 0
        calib = []
        guard = 0
        max_guard = int(self.cfg.get("env", {}).get("catan", {}).get("max_ticks", 2000))

        done = env.done()
        while not done:
            state = env.current_state()
            action, label, info = self.decide(env, state, learn_rho=reference_rate)
            if action is None:
                break
            n_dec += 1
            n_q += int(info.queried)
            n_hit += int(info.changed)
            if info.reference and label is not None:
                n_ref += 1
                phi, y, cost = label
                pred_before = self.voq.predict(phi)  # for calibration BEFORE the update
                calib.append((float(pred_before), float(y)))
                if learn:
                    self.voq.online_update(phi, y)         # evolve M1
                    self.qvl.update(phi, y, cost)          # evolve M3
            _s, _r, done, _i = env.step(action)
            guard += 1
            if guard > max_guard:
                break

        G = env.episode_return()
        C = env.sim_steps
        U = G - self.lam * C
        return EpisodeResult(
            seed=seed, lam=self.lam, G=float(G), C=int(C), U=float(U),
            n_decisions=n_dec, n_queries=n_q, n_hits=n_hit, n_reference=n_ref,
            calib=calib, wallclock_s=time.time() - t0,
        )

    def evolve(self, env, seeds, learn: bool = True, consolidate: bool = True) -> list[EpisodeResult]:
        """Algorithm 2: stream of instances, online-evolve theta, consolidate QVL per instance."""
        results = []
        for seed in seeds:
            res = self.run_episode(env, seed, learn=learn)
            results.append(res)
            if consolidate:
                self.qvl.consolidate(prune=False)   # Hoeffding retain/eject (M3)
            self._instance_idx += 1
        return results
