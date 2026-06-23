"""Task 1 — unified game-environment interface with clone + forward rollout.

The whole method rests on one capability: **branch the real simulator and play it
forward** (that is what a "query" is — never an LLM imagination; see METHOD_DESIGN §3 M2).

Design notes / deviations from the literal SPEC pseudo-code (documented on purpose):

* The env is *stateful* (gym-like): it holds the current ``State``. ``clone(state)``
  returns an **independent** env positioned at ``state`` so two branches can be played
  forward without interfering (SPEC Task-1 acceptance test #1).
* ``rollout(state, action, policy, depth)`` clones *internally* and never mutates the
  caller's env. The SPEC §2.1 snippet writes ``env.rollout(env.clone(state), ...)``
  which would double-clone; cloning inside ``rollout`` is equivalent and cleaner. The
  public ``clone`` is still exposed and tested.
* ``query_cost(state)`` returns the **full** estimated cost (in simulator steps) of one
  gate-triggered query at ``state`` (= m*K*expected_rollout_len), so the gate check
  ``g_hat > lam * env.query_cost(state)`` matches SPEC §2.1 directly. Budgets m,K,d are
  injected at construction via ``set_query_budget``.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Optional


@dataclass
class State:
    """A decision-point snapshot.

    ``raw`` is the underlying engine object (e.g. a Catanatron ``Game``). ``agent_id`` is
    whose decision this is (the perspective for value/return). ``meta`` carries cheap
    cached scalars used by features (kept instance-INVARIANT). ``State`` itself is a thin
    handle; deep copying happens in :meth:`GameEnv.clone`.
    """

    raw: Any
    agent_id: Any = None
    meta: dict = field(default_factory=dict)


class GameEnv(ABC):
    """Abstract environment. Concrete adapters: CatanEnv, STSEnv, MinecraftEnv."""

    # ---- query budget (used only to *estimate* query_cost; actual steps are counted) ----
    def __init__(self) -> None:
        self._m = 3
        self._k = 5
        self._depth: Any = "to_turn_end"
        self._sim_steps_total = 0  # C: total simulator steps consumed this episode (rollouts + reference)
        self._done_flag = False    # set True by step() at terminal; read by done()
        self._rollout_len_ema = None  # running mean of observed single-rollout length (self-calibrates query_cost)

    def set_query_budget(self, m: int, k: int, depth: Any) -> None:
        self._m, self._k, self._depth = m, k, depth

    # ---------------------------------------------------------------- core API
    @abstractmethod
    def reset(self, seed: int) -> State:
        """Reset to instance ``seed`` and return the first decision-point State."""

    @abstractmethod
    def step(self, action: Any) -> tuple[State, float, bool, dict]:
        """Advance the *real* episode with ``action``; return (next_state, reward, done, info).

        Counts toward game progress, NOT toward the simulator-query budget C.
        """

    @abstractmethod
    def current_state(self) -> State:
        ...

    @abstractmethod
    def valid_actions(self, state: Optional[State] = None) -> list:
        """Enumerable legal actions (white-box). Defaults to current state."""

    @abstractmethod
    def clone(self, state: Optional[State] = None) -> "GameEnv":
        """Return an INDEPENDENT env positioned at ``state`` (deep copy). The query core."""

    def done(self) -> bool:
        """Whether the real episode has terminated. Default reads the flag set by step();
        adapters may override with a direct terminal check."""
        return bool(self._done_flag)

    @abstractmethod
    def episode_return(self) -> float:
        """Scalar game return G of the finished (or current) real episode."""

    @property
    @abstractmethod
    def supports_cheap_clone(self) -> bool:
        """Catan=True; StS depends on a fast forward sim; Minecraft=False."""

    # --------------------------------------------------------- query / rollout
    def rollout(
        self,
        state: State,
        action: Any,
        policy: Any,
        depth: Any = None,
        crn_seed: Optional[int] = None,
    ) -> float:
        """One bounded rollout: clone @ state, apply ``action``, play forward with
        ``policy`` to ``depth``, return a scalar value estimate from ``state.agent_id``'s
        perspective. Uses the REAL simulator (clone+forward), never imagination.

        ``crn_seed`` (common random numbers): when set, the stochastic draws of this
        rollout are made reproducible so that two candidate actions are compared under the
        *same* luck (METHOD_DESIGN §4.2). Honored only where ``supports_cheap_clone``.
        """
        depth = self._depth if depth is None else depth
        # Isolate ALL randomness this query consumes so it NEVER perturbs the real episode's
        # dice/opponent stream. Without this, query rollouts (esp. CRN reseeding of the global
        # RNG used by the simulator, and clone() drawing from self._rng) corrupt the main game
        # and make queried vs non-queried episodes incomparable (observed: always-query results
        # collapsing to identical values across seeds). Snapshot before, restore in finally.
        snap = self._rng_snapshot()
        try:
            branch = self.clone(state)
            if crn_seed is not None:
                branch._set_crn(crn_seed)
            steps = branch._apply_and_playout(action, policy, depth)
            self._sim_steps_total += steps
            self._update_rollout_len_ema(steps)
            return branch._value_estimate(perspective=state.agent_id)
        finally:
            self._rng_restore(snap)

    # ---- RNG isolation: a query must not change the real episode's random stream ----
    def _rng_snapshot(self):
        import random as _random

        import numpy as _np
        try:
            rng_state = self._rng.bit_generator.state
        except Exception:
            rng_state = None
        return (_random.getstate(), _np.random.get_state(), rng_state)

    def _rng_restore(self, snap) -> None:
        import random as _random

        import numpy as _np
        py_state, np_state, rng_state = snap
        try:
            _random.setstate(py_state)
            _np.random.set_state(np_state)
        except Exception:
            pass
        if rng_state is not None:
            try:
                self._rng.bit_generator.state = rng_state
            except Exception:
                pass

    def _update_rollout_len_ema(self, steps: int, alpha: float = 0.1) -> None:
        s = float(steps)
        self._rollout_len_ema = s if self._rollout_len_ema is None else (1 - alpha) * self._rollout_len_ema + alpha * s

    def query_cost(self, state: Optional[State] = None) -> float:
        """Estimated full cost (sim steps) of ONE gate query at ``state``: m * K * E[rollout len].
        E[rollout len] uses the running EMA of observed rollout lengths once warm (self-
        calibrating), falling back to the adapter's static estimate before any rollout."""
        state = self.current_state() if state is None else state
        est = self._rollout_len_ema if self._rollout_len_ema is not None else self._expected_rollout_len(state)
        return float(self._m) * float(self._k) * float(est)

    # ---- bookkeeping for C (total simulator steps; the cost term in U = G - lam*C) ----
    @property
    def sim_steps(self) -> int:
        return self._sim_steps_total

    def reset_sim_steps(self) -> None:
        self._sim_steps_total = 0

    def add_sim_steps(self, n: int) -> None:
        self._sim_steps_total += int(n)

    # ----------------------------------------- helpers with sensible defaults
    def action_irreversibility(self, state: State, action: Any) -> float:
        """White-box irreversibility proxy (METHOD_DESIGN §4.2): does ``action`` shrink the
        future legal-action set? Returns a value in [0,1]; 0 if not cheaply computable.

        irrev = max(0, 1 - |valid(s')| / |valid(s)|) where s' = s after ``action``.
        Computed on a throwaway clone so the real env is untouched.
        """
        snap = self._rng_snapshot()
        try:
            before = max(1, len(self.valid_actions(state)))
            branch = self.clone(state)
            branch._apply_one(action)
            after = len(branch.valid_actions(branch.current_state()))
            return float(max(0.0, 1.0 - (after / before)))
        except Exception:
            return 0.0
        finally:
            self._rng_restore(snap)

    def phase_progress(self, state: State) -> float:
        """Normalized game progress in [0,1] (instance-invariant). Override per game."""
        return float(state.meta.get("progress", 0.0))

    # -------------------------------- abstract internals used by rollout/cost
    @abstractmethod
    def _apply_and_playout(self, action: Any, policy: Any, depth: Any) -> int:
        """On THIS (already-cloned) env: execute ``action`` then play forward with
        ``policy`` until ``depth`` / turn-end / game-end. Return number of sim steps taken."""

    @abstractmethod
    def _apply_one(self, action: Any) -> None:
        """Execute a single action on THIS env (used by irreversibility probe)."""

    @abstractmethod
    def _value_estimate(self, perspective: Any) -> float:
        """Value of THIS env's current state from ``perspective`` (terminal => true return)."""

    @abstractmethod
    def _expected_rollout_len(self, state: State) -> float:
        """Cheap estimate of steps in one bounded rollout from ``state`` (for query_cost)."""

    def _set_crn(self, seed: int) -> None:
        """Seed this branch's RNG for common-random-numbers. Override where supported."""
        import random as _random

        import numpy as _np

        _random.seed(seed)
        _np.random.seed(seed % (2**31 - 1))


def make_env(game: str, cfg: dict) -> GameEnv:
    """Factory. ``cfg`` is the full parsed sim.yaml dict."""
    game = game.lower()
    if game == "mock":
        from .mock_env import MockEnv

        return MockEnv(cfg)
    if game == "catan":
        from .catan_env import CatanEnv

        return CatanEnv(cfg)
    if game == "sts":
        from .sts_env import STSEnv

        return STSEnv(cfg)
    if game == "minecraft":
        from .minecraft_env import MinecraftEnv

        return MinecraftEnv(cfg)
    raise ValueError(f"unknown game: {game!r}")
