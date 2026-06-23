"""Task 1 acceptance (Catan): clone independence, a full random episode, finite value.
Plus a Task 7 smoke: a few SimTriage episodes on Catan. Skips if catanatron is absent.
"""

import numpy as np
import pytest

catanatron = pytest.importorskip("catanatron")  # noqa: F841

from src.envs.catan_env import CatanEnv  # noqa: E402
from src.policy import HeuristicPolicy, RolloutPolicy  # noqa: E402
from src.simtriage import SimTriage  # noqa: E402


def _cfg():
    return {
        "env": {"catan": {"num_players": 4, "our_color": "RED", "opponents": "weighted_random",
                          "return_kind": "victory_points", "max_ticks": 2000}},
        "simtriage": {
            "lam_default": 0.01, "m": 3, "K": {"catan": 2}, "d": {"catan": "to_turn_end"},
            "reference": {"K": {"catan": 3}, "d": {"catan": "to_game_end"}, "use_crn": True},
            "rho_start": 0.2, "rho_end": 0.1, "rho_anneal_instances": 5,
            "voq": {"type": "ridge", "l2": 1.0, "optimistic": 1.0, "warmup_n": 5},
            "qvl": {"hoeffding_alpha": 0.1, "min_count": 4, "proto_radius": 0.75,
                    "voq_range": 2.0, "enable_transfer_prior": True},
        },
        "policy": {"backend": "heuristic", "rollout_policy": "weighted_random"},
        "features": {"instance_dependent": False},
    }


def test_supports_cheap_clone():
    env = CatanEnv(_cfg())
    assert env.supports_cheap_clone is True


def test_clone_independence():
    env = CatanEnv(_cfg())
    state = env.reset(0)
    acts = env.valid_actions(state)
    assert len(acts) > 0
    b1 = env.clone(state)
    b3 = env.clone(state)
    n_before = len(b3.valid_actions(b3.current_state()))
    b1._apply_one(acts[0])  # mutate branch 1
    # a fresh clone of the same source state must be unaffected by b1's mutation
    assert len(b3.valid_actions(b3.current_state())) == n_before


def test_full_random_episode_returns_scalar():
    cfg = _cfg()
    env = CatanEnv(cfg)
    pol = HeuristicPolicy(cfg)
    state = env.reset(1)
    done = env.done()
    steps = 0
    while not done and steps < 5000:
        acts = env.valid_actions(env.current_state())
        if not acts:
            break
        a = env.weighted_random_action(None, acts)
        _s, _r, done, _i = env.step(a)
        steps += 1
    G = env.episode_return()
    assert np.isfinite(G)


def test_rollout_value_finite():
    cfg = _cfg()
    env = CatanEnv(cfg)
    pol = HeuristicPolicy(cfg)
    rp = RolloutPolicy("weighted_random", prior=pol)
    state = env.reset(2)
    A, _ = pol.top_candidates(env, state, 3)
    v = env.rollout(state, A[0], rp, "to_turn_end", crn_seed=11)
    assert np.isfinite(v)
    assert env.sim_steps > 0


def test_simtriage_smoke_catan():
    cfg = _cfg()
    agent = SimTriage(cfg, lam=0.01, game="catan", seed=5)
    env = CatanEnv(cfg)
    res = agent.evolve(env, seeds=[0, 1], learn=True)
    assert len(res) == 2
    for r in res:
        assert np.isfinite(r.G)
        assert r.C >= 0
