"""Task 1 — Catanatron adapter (the primary, sim-native environment).

Catanatron *is* a simulator, so clone+rollout is cleanest here — implement & validate this
first (SPEC §5). The agent controls one color; opponents are driven by a cheap policy. A
"query" = clone the ``Game`` and play it forward with the rollout policy; this is the REAL
simulator, never imagination (METHOD_DESIGN §3 M2).

All Catanatron-specific introspection is funnelled through small ``_*`` helpers with
defensive fallbacks and ``# CONFIRM-ON-ENV`` markers, because exact accessor names vary a
little across Catanatron versions and there is no local env to check against yet.
"""

from __future__ import annotations

import copy
import math
import random as _pyrandom
from typing import Any, Optional

import numpy as np

from .base_env import GameEnv, State

# ---- optional import: keep this module importable even without catanatron installed ----
# Accessor names move a little across Catanatron versions, so locate each symbol defensively.
CATAN_AVAILABLE = False
Color = None
Game = None
RandomPlayer = None
try:  # pragma: no cover - import availability depends on the test environment
    import catanatron  # noqa: F401

    try:
        from catanatron import Color, Game  # type: ignore
    except Exception:
        from catanatron.game import Game  # type: ignore
        from catanatron.models.enums import Color  # type: ignore

    for _modpath in (
        "catanatron",
        "catanatron.players.search",
        "catanatron.models.player",
        "catanatron.players.playouts",
    ):
        try:
            _mod = __import__(_modpath, fromlist=["RandomPlayer"])
            RandomPlayer = getattr(_mod, "RandomPlayer")
            break
        except Exception:
            continue
    CATAN_AVAILABLE = Game is not None and Color is not None and RandomPlayer is not None
except Exception:  # pragma: no cover
    CATAN_AVAILABLE = False

# ActionType enum (used for the myopic prior); imported defensively.
try:  # pragma: no cover
    from catanatron.models.enums import ActionType  # type: ignore
except Exception:  # pragma: no cover
    try:
        from catanatron import ActionType  # type: ignore
    except Exception:
        ActionType = None


# Myopic action-TYPE preference for the heuristic prior. Deliberately shallow so the real
# simulator rollout has room to correct it (the regime SimTriage targets). Keyed by the
# ActionType *name* string to avoid enum-version fragility.
_TYPE_PRIORITY = {
    "ROLL": 5.0,
    "BUILD_CITY": 6.0,
    "BUILD_SETTLEMENT": 5.0,
    "BUY_DEVELOPMENT_CARD": 4.0,
    "PLAY_KNIGHT_CARD": 3.5,
    "MOVE_ROBBER": 3.0,
    "BUILD_ROAD": 3.0,
    "PLAY_MONOPOLY": 2.5,
    "PLAY_YEAR_OF_PLENTY": 2.5,
    "PLAY_ROAD_BUILDING": 2.5,
    "MARITIME_TRADE": 2.0,
    "OFFER_TRADE": 1.5,
    "ACCEPT_TRADE": 1.2,
    "DISCARD": 1.0,
    "REJECT_TRADE": 1.0,
    "CONFIRM_TRADE": 1.0,
    "CANCEL_TRADE": 1.0,
    "END_TURN": 0.5,
}
_DEFAULT_PRIORITY = 1.0

# dice pips (ways to roll each number) — production desirability of an adjacent tile
_PIPS = {2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 0, 8: 5, 9: 4, 10: 3, 11: 2, 12: 1}


def _color_by_name(name: str):
    name = (name or "RED").upper()
    return getattr(Color, name)


class CatanEnv(GameEnv):
    def __init__(self, cfg: Optional[dict] = None, _skip_reset: bool = False):
        super().__init__()
        cfg = cfg or {}
        self.cfg = cfg
        ecfg = cfg.get("env", {}).get("catan", {}) if "env" in cfg else cfg
        self.num_players = int(ecfg.get("num_players", 4))
        self.our_color_name = ecfg.get("our_color", "RED")
        self.opp_kind = ecfg.get("opponents", "weighted_random")
        self.return_kind = ecfg.get("return_kind", "victory_points")
        self.max_ticks = int(ecfg.get("max_ticks", 2000))

        # query budget (used by query_cost). Filled from sim config if present.
        scfg = cfg.get("simtriage", {})
        self._m = int(scfg.get("m", 3))
        self._k = int((scfg.get("K", {}) or {}).get("catan", 5)) if isinstance(scfg.get("K"), dict) else 5
        d = scfg.get("d", {})
        self._depth = d.get("catan", "to_turn_end") if isinstance(d, dict) else "to_turn_end"

        self.game = None
        self._our_color = None
        self._rng = np.random.default_rng(0)
        self._seed = 0

        if CATAN_AVAILABLE and not _skip_reset:
            self._our_color = _color_by_name(self.our_color_name)

    # =============================================================== lifecycle
    def reset(self, seed: int) -> State:
        if not CATAN_AVAILABLE:
            raise RuntimeError(
                "catanatron is not installed. `pip install catanatron` to use CatanEnv."
            )
        self._seed = int(seed)
        self._rng = np.random.default_rng(self._seed)
        _pyrandom.seed(self._seed)
        np.random.seed(self._seed % (2**31 - 1))

        colors = self._make_colors()
        self._our_color = _color_by_name(self.our_color_name)
        players = [RandomPlayer(c) for c in colors]  # placeholders; we drive the loop via execute()
        # Try to pass a seed to Game for reproducibility (CONFIRM-ON-ENV: kw name may differ).
        try:
            self.game = Game(players, seed=self._seed)
        except TypeError:
            self.game = Game(players)
            self._try_seed_game(self._seed)

        self.reset_sim_steps()
        self._done_flag = False
        self._advance_to_our_turn()
        self._done_flag = self._game_over()
        return self.current_state()

    def step(self, action: Any) -> tuple[State, float, bool, dict]:
        self.game.execute(action)
        self._advance_to_our_turn()
        done = self._game_over()
        self._done_flag = done
        reward = self.episode_return() if done else 0.0
        info = {"turn": self._num_turns(), "winner": self._winner()}
        return self.current_state(), float(reward), bool(done), info

    def done(self) -> bool:
        return self._game_over()

    def current_state(self) -> State:
        return State(
            raw=self.game,
            agent_id=self._our_color,
            meta={
                "progress": self.phase_progress_value(),
                "phase": self._phase_value(),
                "n_valid": len(self._playable()),
                "turn": self._num_turns(),
            },
        )

    def valid_actions(self, state: Optional[State] = None) -> list:
        g = state.raw if state is not None else self.game
        return list(self._playable(g))

    def episode_return(self) -> float:
        g = self.game
        me = self._vp(g, self._our_color)
        if self.return_kind == "win":
            w = self._winner(g)
            return 1.0 if w == self._our_color else 0.0
        if self.return_kind == "vp_margin":
            opp = self._max_opp_vp(g, self._our_color)
            return float(me - opp)
        return float(me)  # victory_points

    @property
    def supports_cheap_clone(self) -> bool:
        return True

    # =================================================================== clone
    def clone(self, state: Optional[State] = None) -> "CatanEnv":
        src_game = state.raw if state is not None else self.game
        try:
            g2 = src_game.copy()  # Catanatron's efficient deep copy (CONFIRM-ON-ENV)
        except Exception:
            g2 = copy.deepcopy(src_game)
        env = CatanEnv(self.cfg, _skip_reset=True)
        env.game = g2
        env._our_color = self._our_color
        env._seed = self._seed
        env._m, env._k, env._depth = self._m, self._k, self._depth
        # independent RNG so the two branches do not share a stream (unless CRN reseeds them)
        env._rng = np.random.default_rng(self._rng.integers(0, 2**31 - 1))
        return env

    # ============================================== internals for rollout/cost
    def _apply_and_playout(self, action: Any, policy: Any, depth: Any) -> int:
        """On THIS (cloned) env: execute ``action`` (our candidate), then play forward with
        ``policy`` per ``depth``. Returns sim steps taken."""
        start_turns = self._num_turns()
        self.game.execute(action)
        steps = 1
        steps += self._playout(policy, depth, start_turns, already=steps)
        return steps

    def _apply_one(self, action: Any) -> None:
        self.game.execute(action)

    def _playout(self, policy: Any, depth: Any, start_turns: int, already: int = 0) -> int:
        """Drive ALL acting colors with ``policy`` until the depth condition ends.

        depth semantics:
          * "to_turn_end" : until ``state.num_turns`` increments past ``start_turns``. This is
            turn-monotonic and is robust to 7-roll DISCARD interrupts (which MOVE
            current_player_index to each discarder but leave num_turns unchanged until the
            true END_TURN). A 40-step safety cap guards against a Catanatron version where
            num_turns does not advance as assumed.
          * "to_game_end" : until a winner / cap.
          * int N         : at most N total steps (including the candidate already counted).
        """
        steps = 0
        is_int = isinstance(depth, int) or (isinstance(depth, str) and depth.isdigit())
        budget = int(depth) if is_int else None

        while not self._game_over():
            if depth == "to_turn_end":
                if self._num_turns() != start_turns:
                    break
                if steps >= 40:  # safety: a single turn never legitimately exceeds this
                    break
            if budget is not None and (already + steps) >= budget:
                break
            acts = self._playable()
            if not acts:
                break
            a = policy(self, None, acts)
            if a is None:
                a = acts[0]
            self.game.execute(a)
            steps += 1
            if (already + steps) > self.max_ticks:
                break
        return steps

    def _player_strength(self, g, color) -> float:
        """Continuous board strength = VP (dominant) + small-weight LEADING indicators that
        are below the VP threshold: longest-road length, settlements/cities, hand resources,
        largest-army holder. These break ties between equal-VP states so a rollout can actually
        distinguish candidate actions (fixes the hit_rate~=0 / VP-only coarseness, audit-finding
        from the first Catan run). Degrades to VP-only if any accessor is unavailable."""
        score = float(self._vp(g, color))
        try:
            from catanatron.models.enums import CITY, SETTLEMENT
            from catanatron.state_functions import (get_largest_army,
                                                    get_longest_road_length,
                                                    get_player_buildings,
                                                    player_num_resource_cards)
            score += 0.20 * float(get_longest_road_length(g.state, color))
            score += 0.30 * len(get_player_buildings(g.state, color, SETTLEMENT))
            score += 0.60 * len(get_player_buildings(g.state, color, CITY))
            score += 0.08 * float(player_num_resource_cards(g.state, color))
            la = get_largest_army(g.state)
            holder = la[0] if isinstance(la, (tuple, list)) else la
            if holder == color:
                score += 0.50
        except Exception:
            pass
        return score

    def _value_estimate(self, perspective: Any) -> float:
        """Competitive value ~[-2, 2]: (my strength - best opponent strength)/10 + terminal bonus."""
        g = self.game
        colors = list(self._colors(g))
        me = self._player_strength(g, perspective)
        opp = max((self._player_strength(g, c) for c in colors if c != perspective), default=0.0)
        val = (me - opp) / 10.0
        w = self._winner(g)
        if w is not None:
            val += 1.0 if w == perspective else -1.0
        return float(val)

    def _expected_rollout_len(self, state: State) -> float:
        d = self._depth
        if d == "to_turn_end":
            return 4.0  # cold estimate only; the base-env EMA self-calibrates to the real length
        if d == "to_game_end":
            prog = self.phase_progress(state)
            return max(10.0, (1.0 - prog) * 120.0)
        try:
            return float(d)
        except Exception:
            return 10.0

    def _set_crn(self, seed: int) -> None:
        """Common random numbers: make this branch's stochastic draws reproducible so two
        candidates are compared under the same luck.

        In Catanatron 3.2.x the engine's dice / dev-card draws come from the GLOBAL ``random``
        module (there is no per-Game / per-State RNG), so ``random.seed(seed)`` below is the
        actual CRN mechanism; our rollout policy draws from ``self._rng``. ``_try_seed_game``
        is a best-effort no-op kept for other Catanatron versions that may expose a private
        RNG. CRN fidelity is validated by scripts/verify_query_determinism.py. NOTE: if a
        future Catanatron switches to a private RNG that isn't reseeded here, CRN would
        silently weaken — the verify script will catch it."""
        _pyrandom.seed(seed)
        np.random.seed(seed % (2**31 - 1))
        self._rng = np.random.default_rng(seed)
        self._try_seed_game(seed)

    # ====================================================== policy/action API
    def heuristic_scores(self, state, actions) -> list[float]:
        """Myopic, instance-INVARIANT-ranking prior scores (per action). Type priority plus
        a small deterministic perturbation so margins/entropy vary across decision points."""
        scores = []
        for a in actions:
            base = _TYPE_PRIORITY.get(self._atype_name(a), _DEFAULT_PRIORITY)
            noise = ((hash(repr(a)) % 1000) / 1000.0 - 0.5) * 0.4
            scores.append(base + noise)
        return scores

    def heuristic_value(self, state, action) -> float:
        """Prior's *imagined* value of one action for the -M2 ablation: a cheap 1-ply
        lookahead (clone, apply, evaluate). This is a shallow approximate forecast that is
        DISTINCT from both the myopic type-priority ranking (so imagined-best can differ from
        a_0) and the method's depth-d multi-step real rollout (so it is strictly weaker than a
        true query). No multi-step simulator use; the single peek is not counted toward C."""
        try:
            branch = self.clone(state)
            branch._apply_one(action)
            return branch._value_estimate(perspective=getattr(state, "agent_id", self._our_color))
        except Exception:
            base = _TYPE_PRIORITY.get(self._atype_name(action), _DEFAULT_PRIORITY)
            return (base / 6.0) - 0.3

    def random_action(self, state, actions):
        i = int(self._rng.integers(0, len(actions)))
        return actions[i]

    def weighted_random_action(self, state, actions):
        scores = np.asarray(self.heuristic_scores(state, actions), dtype=float)
        z = scores - scores.max()
        w = np.exp(z)
        w = w / w.sum()
        i = int(self._rng.choice(len(actions), p=w))
        return actions[i]

    def greedy_action(self, state, actions):
        scores = self.heuristic_scores(state, actions)
        return actions[int(np.argmax(scores))]

    # opponent action selection during the REAL episode
    def _opponent_action(self, actions):
        if self.opp_kind == "random":
            return self.random_action(None, actions)
        if self.opp_kind in ("value", "greedy"):
            return self.greedy_action(None, actions)
        return self.weighted_random_action(None, actions)

    # ----------------------------------------------------------- LLM rendering
    def _node_production_str(self, node_id) -> str:
        """Board-aware description of a node: adjacent resources + dice numbers + total pips,
        so the LLM can judge settlement/city spot quality instead of staring at a node id."""
        try:
            tiles = self.game.state.board.map.adjacent_tiles[node_id]
            parts, pips = [], 0
            for t in tiles:
                r = getattr(t, "resource", None)
                n = getattr(t, "number", None)
                if r is None:
                    continue  # desert
                parts.append(f"{r}{n}")
                pips += _PIPS.get(n, 0)
            return (" ".join(parts) + f" (pips {pips})") if parts else "desert-only"
        except Exception:
            return "?"

    def render_action(self, action) -> str:
        at = self._atype_name(action)
        val = getattr(action, "value", None)
        if at in ("BUILD_SETTLEMENT", "BUILD_CITY") and isinstance(val, int):
            return f"{at} at node {val} [{self._node_production_str(val)}]"
        if at == "BUILD_ROAD":
            return f"BUILD_ROAD {val}"
        if at == "MARITIME_TRADE":
            return f"MARITIME_TRADE {val}"
        if at == "MOVE_ROBBER":
            return f"MOVE_ROBBER {val}"
        if val is None:
            return at
        return f"{at} {val}"

    def render_text(self, state) -> str:
        g = state.raw if hasattr(state, "raw") else self.game
        you = getattr(self._our_color, "value", self._our_color)
        out = [f"Catan, turn {self._num_turns(g)}. You are {you}. Standings:"]
        try:
            from catanatron.state_functions import (get_longest_road_length,
                                                    player_num_resource_cards)
            for c in self._colors(g):
                vp = self._vp(g, c)
                try:
                    road = get_longest_road_length(g.state, c)
                    hand = player_num_resource_cards(g.state, c)
                    extra = f", longest_road {road}, {hand} cards"
                except Exception:
                    extra = ""
                me = "  <-- you" if c == self._our_color else ""
                out.append(f"  {getattr(c,'value',c)}: {vp} VP{extra}{me}")
        except Exception:
            pass
        return "\n".join(out)

    # ======================================================== phase / progress
    def phase_progress(self, state: State) -> float:
        g = state.raw if hasattr(state, "raw") else self.game
        return self.phase_progress_value(g)

    def phase_progress_value(self, g=None) -> float:
        g = g or self.game
        max_vp = self._max_vp(g)
        return float(min(1.0, max_vp / 10.0))

    def _phase_value(self, g=None) -> float:
        g = g or self.game
        max_vp = self._max_vp(g)
        if max_vp <= 2:
            return 0.0  # initial / early
        if max_vp < 8:
            return 0.5  # midgame
        return 1.0  # endgame

    # ============================================ Catanatron introspection ====
    # All accessors below are the version-sensitive surface. Marked CONFIRM-ON-ENV.
    def _make_colors(self):
        order = ["RED", "BLUE", "ORANGE", "WHITE"][: self.num_players]
        if self.our_color_name.upper() not in order:
            order[0] = self.our_color_name.upper()
        return [_color_by_name(n) for n in order]

    def _playable(self, g=None):
        g = g or self.game
        try:
            return list(g.state.playable_actions)
        except Exception:
            return []

    def _acting_color(self, g=None):
        acts = self._playable(g)
        return getattr(acts[0], "color", None) if acts else None

    def _cur_index(self, g=None) -> int:
        g = g or self.game
        try:
            return int(g.state.current_player_index)
        except Exception:
            # fall back to the acting color's index
            try:
                c = self._acting_color(g)
                return list(self._colors(g)).index(c)
            except Exception:
                return 0

    def _colors(self, g=None):
        g = g or self.game
        try:
            return list(g.state.colors)
        except Exception:
            return [p.color for p in g.state.players]

    def _winner(self, g=None):
        g = g or self.game
        try:
            return g.winning_color()
        except Exception:
            return None

    def _game_over(self, g=None) -> bool:
        g = g or self.game
        if self._winner(g) is not None:
            return True
        if not self._playable(g):
            return True
        try:
            if self._num_turns(g) > self.max_ticks:
                return True
        except Exception:
            pass
        return False

    def _num_turns(self, g=None) -> int:
        g = g or self.game
        for attr in ("num_turns",):
            try:
                return int(getattr(g.state, attr))
            except Exception:
                continue
        return 0

    def _vp(self, g, color) -> int:
        try:
            from catanatron.state_functions import get_actual_victory_points

            return int(get_actual_victory_points(g.state, color))
        except Exception:
            pass
        try:
            from catanatron.state_functions import player_key

            key = player_key(g.state, color)
            ps = g.state.player_state
            for suffix in ("_ACTUAL_VICTORY_POINTS", "_VICTORY_POINTS"):
                if f"{key}{suffix}" in ps:
                    return int(ps[f"{key}{suffix}"])
        except Exception:
            pass
        return 0

    def _max_vp(self, g) -> int:
        try:
            return max(self._vp(g, c) for c in self._colors(g))
        except Exception:
            return 0

    def _max_opp_vp(self, g, perspective) -> int:
        vals = [self._vp(g, c) for c in self._colors(g) if c != perspective]
        return max(vals) if vals else 0

    def _atype_name(self, action) -> str:
        at = getattr(action, "action_type", None)
        if at is None and isinstance(action, (tuple, list)) and len(action) >= 2:
            at = action[1]
        name = getattr(at, "name", None)
        return name if name is not None else str(at)

    def _try_seed_game(self, seed: int) -> None:
        # best-effort: set whatever RNG/seed handle the installed Catanatron exposes
        for attr in ("seed",):
            try:
                setattr(self.game, attr, seed)
            except Exception:
                pass
        try:
            rng = getattr(self.game.state, "rng", None)
            if rng is not None and hasattr(rng, "seed"):
                rng.seed(seed)
        except Exception:
            pass

    # ----- drive opponents in the REAL episode until it is our turn (or game over) -----
    def _advance_to_our_turn(self) -> None:
        guard = 0
        while not self._game_over():
            color = self._acting_color()
            if color == self._our_color:
                return
            acts = self._playable()
            if not acts:
                return
            self.game.execute(self._opponent_action(acts))
            guard += 1
            if guard > self.max_ticks:
                return
