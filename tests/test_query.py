"""Task 5 acceptance: gate, query argmax, and reference-label correctness.

* When the prior is already optimal (sigma=0) the reference label y ~= 0.
* When the prior is wrong (large sigma) the query finds a better action and y > 0.
"""

import numpy as np

from src.envs.mock_env import MockEnv
from src.policy import HeuristicPolicy, RolloutPolicy
from src.query import gate, reference_voq, run_query


def _cfg(sigma):
    return {"env": {"mock": {"n_steps": 10, "n_actions": 6, "prior_sigma": sigma, "rollout_noise": 0.0}},
            "simtriage": {"m": 4, "K": {"catan": 5}}}


def test_gate_rule():
    assert gate(1.0, lam=0.1, cost=5.0) is True       # 1.0 > 0.5
    assert gate(0.4, lam=0.1, cost=5.0) is False      # 0.4 < 0.5
    assert gate(0.0, lam=0.0, cost=100.0) is False    # strict >


def test_reference_label_zero_when_prior_optimal():
    cfg = _cfg(sigma=0.0)  # prior == true => a_0 is optimal everywhere
    env = MockEnv(cfg)
    pol = HeuristicPolicy(cfg)
    rp = RolloutPolicy("weighted_random", prior=pol)
    state = env.reset(0)
    A, prefs = pol.top_candidates(env, state, 4)
    y, a_ref, q = reference_voq(env, state, A, rp, K_ref=3, depth_ref=1, use_crn=True, crn_seed=1)
    assert abs(y) < 1e-6, f"prior optimal => y~=0, got {y}"


def test_query_finds_better_action_when_prior_wrong():
    cfg = _cfg(sigma=1.5)  # noisy prior => often suboptimal a_0
    env = MockEnv(cfg)
    pol = HeuristicPolicy(cfg)
    rp = RolloutPolicy("weighted_random", prior=pol)
    found_improvement = 0
    n = 8
    for seed in range(n):
        state = env.reset(seed)
        A, prefs = pol.top_candidates(env, state, cfg["simtriage"]["m"])
        a_hat, q = run_query(env, state, A, rp, K=3, depth=1, use_crn=True, crn_seed=7)
        y, _, _ = reference_voq(env, state, A, rp, K_ref=3, depth_ref=1, use_crn=True, crn_seed=7)
        assert y >= -1e-9          # value of query is non-negative by construction
        if q[int(np.argmax(q))] > q[0] + 1e-9:
            found_improvement += 1
    assert found_improvement > 0, "with a noisy prior, querying should sometimes change/improve the choice"


def test_crn_reduces_to_deterministic_with_zero_noise():
    cfg = _cfg(sigma=0.5)
    env = MockEnv(cfg)
    pol = HeuristicPolicy(cfg)
    rp = RolloutPolicy("weighted_random", prior=pol)
    state = env.reset(3)
    A, _ = pol.top_candidates(env, state, 4)
    _, q1 = run_query(env, state, A, rp, K=3, depth=1, use_crn=True, crn_seed=42)
    _, q2 = run_query(env, state, A, rp, K=3, depth=1, use_crn=True, crn_seed=42)
    assert np.allclose(q1, q2)
