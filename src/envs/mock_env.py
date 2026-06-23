"""A dependency-free synthetic game implementing the full GameEnv interface.

Purpose: validate the M1/M2/M3 machinery (gate, real-rollout query, reference labels, VoQ
learning, QVL transfer) end-to-end WITHOUT catanatron / a GPU. It is also the testbed for
the unit/integration tests and a smoke target for run_experiment (game == "mock").

Construction (deliberately makes querying valuable and VoQ learnable):
  * Each decision point t has hidden true action values ``true[t]``.
  * The frozen prior sees noisy scores ``prior = true + noise``; with enough noise the prior
    favorite a_0 differs from the true best a*, so a real rollout (which reveals ``true``)
    changes the decision -> query has value.
  * The reference label y = max(true) - true[a_0] >= 0 anti-correlates with the prior margin,
    so a regressor on phi (which contains margin/entropy) can learn it -> calibration falls.
This mirrors the Catan motivation (high-branch decisions where the prior misjudges) in a
controllable, reproducible way.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from .base_env import GameEnv, State


class MockEnv(GameEnv):
    def __init__(self, cfg: Optional[dict] = None):
        super().__init__()
        cfg = cfg or {}
        mcfg = cfg.get("env", {}).get("mock", {}) if "env" in cfg else cfg
        self.n_steps = int(mcfg.get("n_steps", 30))
        self.n_actions = int(mcfg.get("n_actions", 5))
        self.prior_sigma = float(mcfg.get("prior_sigma", 0.35))  # prior noise => query value
        self.rollout_noise = float(mcfg.get("rollout_noise", 0.0))  # 0 => CRN-exact rollouts
        # imagination bias for the -M2 ablation: a SEPARATE, biased value model distinct from
        # both the prior ranking and the true value, so imagined-best differs from a_0 yet is
        # strictly worse than a real rollout (which sees the true value).
        self.imagine_sigma = float(mcfg.get("imagine_sigma", 0.5))

        scfg = cfg.get("simtriage", {})
        self._m = int(scfg.get("m", 3))
        self._k = int((scfg.get("K", {}) or {}).get("catan", 5)) if isinstance(scfg.get("K"), dict) else 5
        self._depth = 1  # one-step lookahead suffices for the synthetic value

        self._t = 0
        self._accum = 0.0
        self._true = None       # (n_steps, n_actions) hidden values
        self._prior = None      # (n_steps, n_actions) noisy prior scores
        self._imagined = None   # (n_steps, n_actions) biased imagination model (-M2 ablation)
        self._rng = np.random.default_rng(0)
        self._seed = 0
        # branch bookkeeping (set on clones)
        self._branch_accum = 0.0
        self._branch_t = 0

    # ------------------------------------------------------------- lifecycle
    def reset(self, seed: int) -> State:
        self._seed = int(seed)
        self._rng = np.random.default_rng(seed)
        gen = np.random.default_rng(seed)
        self._true = gen.random((self.n_steps, self.n_actions))
        # Difficulty rises with game progress: late decisions get a noisier prior, so the
        # prior misjudges more there (mirrors the Catan motivation that high-branch / late
        # decisions are where querying pays off). This makes VoQ learnable from phi (the
        # 'progress' feature and the resulting smaller margin / higher entropy).
        t = np.arange(self.n_steps)[:, None]
        sigma_t = self.prior_sigma * (0.2 + 1.6 * (t / max(1, self.n_steps)))
        noise = gen.normal(0.0, 1.0, (self.n_steps, self.n_actions)) * sigma_t
        self._prior = self._true + noise
        # imagination = true + an INDEPENDENT bias (different draw/scale from the prior noise),
        # so argmax(imagined) can differ from a_0 but is an unreliable guess vs the real value.
        self._imagined = self._true + gen.normal(0.0, self.imagine_sigma, (self.n_steps, self.n_actions))
        self._t = 0
        self._accum = 0.0
        self.reset_sim_steps()
        self._done_flag = False
        return self.current_state()

    def step(self, action: Any) -> tuple[State, float, bool, dict]:
        r = float(self._true[self._t, int(action)])
        self._accum += r
        self._t += 1
        done = self._t >= self.n_steps
        self._done_flag = done
        return self.current_state(), r, bool(done), {"t": self._t}

    def current_state(self) -> State:
        return State(
            raw={"t": self._t, "accum": self._accum},
            agent_id=0,
            meta={"progress": self._t / self.n_steps, "phase": min(1.0, self._t / self.n_steps),
                  "n_valid": self.n_actions, "turn": self._t},
        )

    def valid_actions(self, state: Optional[State] = None) -> list:
        return list(range(self.n_actions))

    def episode_return(self) -> float:
        return float(self._accum)

    def done(self) -> bool:
        return self._t >= self.n_steps

    @property
    def supports_cheap_clone(self) -> bool:
        return True

    # ----------------------------------------------------------------- clone
    def clone(self, state: Optional[State] = None) -> "MockEnv":
        env = MockEnv.__new__(MockEnv)
        GameEnv.__init__(env)
        env.n_steps, env.n_actions = self.n_steps, self.n_actions
        env.prior_sigma, env.rollout_noise = self.prior_sigma, self.rollout_noise
        env._m, env._k, env._depth = self._m, self._k, self._depth
        env._true, env._prior = self._true, self._prior  # shared immutable schedule
        env._imagined = self._imagined
        env.imagine_sigma = self.imagine_sigma
        env._seed = self._seed
        env._rng = np.random.default_rng(self._rng.integers(0, 2**31 - 1))
        t = (state.raw["t"] if state is not None else self._t)
        env._t = t
        env._branch_t = t
        env._branch_accum = 0.0
        env._accum = (state.raw["accum"] if state is not None else self._accum)
        env._done_flag = False
        return env

    # ----------------------------------------- internals for rollout/cost
    def _apply_and_playout(self, action: Any, policy: Any, depth: Any) -> int:
        t = self._t  # branch was positioned here by clone()
        val = float(self._true[t, int(action)])
        if self.rollout_noise > 0:
            val += float(self._rng.normal(0, self.rollout_noise))
        self._branch_accum = val
        steps = 1
        # optional deeper playout for reference (depth > 1): play forward with the policy
        if isinstance(depth, int):
            horizon = max(0, depth - 1)
        elif depth == "to_game_end":
            horizon = self.n_steps - t - 1
        else:  # "to_turn_end" or anything else -> single-step value
            horizon = 0
        self._t = t + 1  # advance the cursor so heuristic_scores tracks the current step
        while horizon > 0 and self._t < self.n_steps:
            acts = list(range(self.n_actions))
            a = policy(self, None, acts) if policy is not None else acts[0]
            self._branch_accum += float(self._true[self._t, int(a)])
            self._t += 1
            steps += 1
            horizon -= 1
        return steps

    def _set_crn(self, seed: int) -> None:
        super()._set_crn(seed)
        self._rng = np.random.default_rng(seed)  # make deeper playout draws reproducible

    def _apply_one(self, action: Any) -> None:
        # used by irreversibility probe; mock actions never shrink the valid set
        if self._t < self.n_steps:
            self._t += 1

    def _value_estimate(self, perspective: Any) -> float:
        return float(self._branch_accum)

    def _expected_rollout_len(self, state: State) -> float:
        d = self._depth
        return float(d) if isinstance(d, (int, float)) else 1.0

    # ------------------------------------------------------- policy/action API
    def heuristic_scores(self, state, actions) -> list[float]:
        t = state.raw["t"] if hasattr(state, "raw") else self._t
        return [float(self._prior[t, int(a)]) for a in actions]

    def heuristic_value(self, state, action) -> float:
        # -M2 imagination: a biased value model distinct from the prior ranking and the true
        # value (so imagined-best can differ from a_0 yet be worse than a real rollout).
        t = state.raw["t"] if hasattr(state, "raw") else self._t
        return float(self._imagined[t, int(action)])

    def random_action(self, state, actions):
        return actions[int(self._rng.integers(0, len(actions)))]

    def weighted_random_action(self, state, actions):
        s = np.asarray(self.heuristic_scores(state if state is not None else self.current_state(), actions))
        w = np.exp(s - s.max()); w /= w.sum()
        return actions[int(self._rng.choice(len(actions), p=w))]

    def greedy_action(self, state, actions):
        s = self.heuristic_scores(state if state is not None else self.current_state(), actions)
        return actions[int(np.argmax(s))]

    def render_action(self, action) -> str:
        return f"action#{action}"

    def render_text(self, state) -> str:
        t = state.raw["t"] if hasattr(state, "raw") else self._t
        return f"step={t}/{self.n_steps} accum={state.raw.get('accum', 0):.2f}"

    def phase_progress(self, state: State) -> float:
        t = state.raw["t"] if hasattr(state, "raw") else self._t
        return float(t / self.n_steps)
