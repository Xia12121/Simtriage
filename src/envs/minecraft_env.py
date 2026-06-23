"""Task 1 — Minecraft adapter (scaffold; query is expensive, run small-scale).

METHOD_DESIGN §4.5 / §7: no cheap exact forward model. ``supports_cheap_clone=False``;
rollout = real/approximate and costly. Reference labels are biased and must be declared in
limitations. Run only small-scale demos here (SPEC §5: catan first, minecraft last).

Interface-complete scaffold; raises until MineDojo/Voyager + Mineflayer are connected.
"""

from __future__ import annotations

from typing import Any, Optional

from .base_env import GameEnv, State


class MinecraftEnv(GameEnv):
    def __init__(self, cfg: Optional[dict] = None):
        super().__init__()
        cfg = cfg or {}
        self.cfg = cfg
        scfg = cfg.get("simtriage", {})
        self._m = int(scfg.get("m", 3))
        self._k = int((scfg.get("K", {}) or {}).get("minecraft", 1)) if isinstance(scfg.get("K"), dict) else 1
        d = scfg.get("d", {})
        self._depth = d.get("minecraft", 20) if isinstance(d, dict) else 20

    @property
    def supports_cheap_clone(self) -> bool:
        return False

    def reset(self, seed: int) -> State:
        raise NotImplementedError(
            "MinecraftEnv requires MineDojo/Voyager + Node.js + Mineflayer. Query is "
            "expensive; run small-scale only (METHOD_DESIGN §7)."
        )

    def step(self, action: Any) -> tuple[State, float, bool, dict]:
        raise NotImplementedError("Connect Mineflayer backend.")

    def current_state(self) -> State:
        raise NotImplementedError

    def valid_actions(self, state: Optional[State] = None) -> list:
        raise NotImplementedError

    def clone(self, state: Optional[State] = None) -> "MinecraftEnv":
        raise NotImplementedError(
            "No cheap clone in Minecraft. Query uses real/approximate rollout; cost is high."
        )

    def episode_return(self) -> float:
        raise NotImplementedError

    def _apply_and_playout(self, action: Any, policy: Any, depth: Any) -> int:
        raise NotImplementedError

    def _apply_one(self, action: Any) -> None:
        raise NotImplementedError

    def _value_estimate(self, perspective: Any) -> float:
        raise NotImplementedError

    def _expected_rollout_len(self, state: State) -> float:
        return float(self._depth if isinstance(self._depth, (int, float)) else 20)
