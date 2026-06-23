"""Task 1 — Slay the Spire adapter (scaffold).

Risk R1 (METHOD_DESIGN §7): the real client (CommunicationMod) is hard to branch cheaply.
Two query modes:
  * If a fast forward simulator (e.g. an ``sts_lightspeed``-style engine) is wired in,
    ``supports_cheap_clone`` becomes True and clone+rollout works like Catan.
  * Otherwise ``supports_cheap_clone=False`` and a query must use seed-replay of the real
    branch (expensive); reference rate ρ should be lowered (handled by simtriage).

This file deliberately ships as an interface-complete scaffold: it raises informative
errors until a backend is connected, so the Catan pipeline is unblocked (SPEC §5).
"""

from __future__ import annotations

from typing import Any, Optional

from .base_env import GameEnv, State


class STSEnv(GameEnv):
    def __init__(self, cfg: Optional[dict] = None):
        super().__init__()
        cfg = cfg or {}
        self.cfg = cfg
        ecfg = cfg.get("env", {}).get("sts", {}) if "env" in cfg else cfg
        self._cheap_clone = bool(ecfg.get("supports_cheap_clone", False))
        self.forward_sim = None  # wire a fast forward simulator here to enable cheap clone
        scfg = cfg.get("simtriage", {})
        self._m = int(scfg.get("m", 3))
        self._k = int((scfg.get("K", {}) or {}).get("sts", 3)) if isinstance(scfg.get("K"), dict) else 3
        d = scfg.get("d", {})
        self._depth = d.get("sts", 10) if isinstance(d, dict) else 10

    @property
    def supports_cheap_clone(self) -> bool:
        return self._cheap_clone and self.forward_sim is not None

    def reset(self, seed: int) -> State:
        raise NotImplementedError(
            "STSEnv requires spirecomm + CommunicationMod (and ideally a fast forward "
            "simulator). Connect a backend before use; see METHOD_DESIGN §7 R1."
        )

    def step(self, action: Any) -> tuple[State, float, bool, dict]:
        raise NotImplementedError("Connect spirecomm backend.")

    def current_state(self) -> State:
        raise NotImplementedError

    def valid_actions(self, state: Optional[State] = None) -> list:
        raise NotImplementedError

    def clone(self, state: Optional[State] = None) -> "STSEnv":
        if not self.supports_cheap_clone:
            raise NotImplementedError(
                "No cheap clone for StS without a fast forward simulator. Query must use "
                "seed-replay of the real branch (expensive)."
            )
        raise NotImplementedError("Wire forward_sim.clone here.")

    def episode_return(self) -> float:
        raise NotImplementedError

    def _apply_and_playout(self, action: Any, policy: Any, depth: Any) -> int:
        raise NotImplementedError

    def _apply_one(self, action: Any) -> None:
        raise NotImplementedError

    def _value_estimate(self, perspective: Any) -> float:
        raise NotImplementedError

    def _expected_rollout_len(self, state: State) -> float:
        return float(self._depth if isinstance(self._depth, (int, float)) else 10)
