"""Task 7 acceptance (on the dependency-free mock game):
* the loop runs end-to-end and evolves theta (VoQ.n grows, QVL fills);
* lambda controls the gate: more querying (steps) when cheap, less when expensive, and
  more querying does not hurt return (the frontier direction);
* VoQ generalizes: predicted g_hat correlates with the reference label y on HELD-OUT
  decisions (learnability, the M1 premise).
"""

import numpy as np

from src.envs.mock_env import MockEnv
from src.metrics import aggregate
from src.policy import HeuristicPolicy
from src.simtriage import SimTriage


def _cfg():
    return {
        "env": {"mock": {"n_steps": 30, "n_actions": 6, "prior_sigma": 0.6, "rollout_noise": 0.0},
                "catan": {"max_ticks": 2000}},
        "simtriage": {
            # the mock is an immediate-reward game: depth-1 IS the ground-truth value, so the
            # high-budget reference uses depth 1 (deeper would only add future-step noise).
            "lam_default": 0.05, "m": 3, "K": {"catan": 5}, "d": {"catan": 1},
            "reference": {"K": {"catan": 8}, "d": {"catan": 1}, "use_crn": True},
            "rho_start": 0.5, "rho_end": 0.2, "rho_anneal_instances": 20,
            "voq": {"type": "ridge", "l2": 1.0, "optimistic": 1.0, "warmup_n": 20},
            "qvl": {"hoeffding_alpha": 0.1, "min_count": 8, "proto_radius": 0.75,
                    "voq_range": 2.0, "enable_transfer_prior": True},
        },
        "policy": {"backend": "heuristic", "rollout_policy": "weighted_random"},
        "features": {"instance_dependent": False},
    }


def _agent(lam):
    cfg = _cfg()
    return SimTriage(cfg, lam=lam, game="catan", seed=123)


def test_end_to_end_evolves_theta():
    agent = _agent(0.0)
    env = MockEnv(_cfg())
    res = agent.evolve(env, seeds=list(range(8)), learn=True)
    assert len(res) == 8
    assert all(np.isfinite(r.G) for r in res)
    assert agent.voq.n > 0, "VoQ should receive labels and evolve"
    assert len(agent.qvl) > 0, "QVL should accumulate prototypes"


def test_lambda_controls_querying_and_does_not_hurt_return():
    train = list(range(12))
    heldout = list(range(100, 110))

    a0 = _agent(0.0)
    a0.evolve(MockEnv(_cfg()), train, learn=True)
    r0 = [a0.run_episode(MockEnv(_cfg()), s, learn=False, reference_rate=0.0) for s in heldout]
    agg0 = aggregate(r0)

    ah = _agent(0.5)
    ah.evolve(MockEnv(_cfg()), train, learn=True)
    rh = [ah.run_episode(MockEnv(_cfg()), s, learn=False, reference_rate=0.0) for s in heldout]
    aggh = aggregate(rh)

    assert agg0["query_rate_mean"] >= aggh["query_rate_mean"]
    assert agg0["sim_steps_mean"] >= aggh["sim_steps_mean"]
    # querying more should not reduce return (it can only switch to a simulator-verified action)
    assert agg0["return_mean"] >= aggh["return_mean"] - 0.5


def test_minus_m2_imagination_is_nondegenerate():
    # -M2 must act on a DISTINCT imagined estimate, not reproduce the prior's a_0 (audit H4).
    cfg = _cfg()
    env = MockEnv(cfg)
    pol = HeuristicPolicy(cfg)
    disagree, total = 0, 0
    for s in range(6):
        st = env.reset(s)
        done = False
        while not done:
            A, _ = pol.top_candidates(env, st, 3)
            if len(A) >= 2:
                total += 1
                imag = [pol.imagine_q(env, st, a) for a in A]
                if A[int(np.argmax(imag))] != A[0]:  # imagined-best != prior favorite a_0
                    disagree += 1
            _s, _r, done, _i = env.step(A[0])
            st = env.current_state()
    assert total > 0
    assert disagree > 0, "imagination must sometimes disagree with the prior, else -M2 == never_query"


def test_voq_generalizes_on_heldout():
    cfg = _cfg()
    agent = SimTriage(cfg, lam=0.0, game="catan", seed=7)
    agent.evolve(MockEnv(cfg), seeds=list(range(40)), learn=True)
    # held-out calibration: force reference everywhere, no learning, collect (pred, y)
    pairs = []
    for s in range(200, 215):
        r = agent.run_episode(MockEnv(cfg), s, learn=False, reference_rate=1.0)
        pairs.extend(r.calib)
    pred = np.array([p for p, _ in pairs])
    y = np.array([t for _, t in pairs])
    assert len(pairs) > 50
    # The gate only needs the RANKING of decisions by query value, not exact regression.
    # Operational learnability: decisions VoQ rates high must have higher actual y on average
    # than decisions it rates low (robust to the per-sample label noise).
    med = float(np.median(pred))
    high = y[pred >= med]
    low = y[pred < med]
    assert high.size > 0 and low.size > 0
    assert high.mean() > low.mean(), (
        f"VoQ should rank decisions by query value: high={high.mean():.3f} low={low.mean():.3f}"
    )
    # and it should beat a constant (mean) predictor in MAE (positive skill)
    voq_mae = float(np.mean(np.abs(pred - y)))
    base_mae = float(np.mean(np.abs(y.mean() - y)))
    assert voq_mae <= base_mae + 1e-6, f"VoQ MAE {voq_mae:.3f} should not exceed baseline {base_mae:.3f}"
