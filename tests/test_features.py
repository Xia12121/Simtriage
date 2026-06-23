"""Task 3 acceptance: phi is instance-INVARIANT (no instance leakage) by default."""

import numpy as np

from src.envs.base_env import State
from src.features import FeatureExtractor


def _state(turn, progress, n_valid):
    return State(raw={"x": object()}, agent_id=turn,
                 meta={"turn": turn, "progress": progress, "phase": progress, "n_valid": n_valid})


def test_invariance_same_decision_same_phi():
    fx = FeatureExtractor({"features": {"instance_dependent": False}})
    prefs = [2.0, 1.0, 0.5]
    A = ["a", "b", "c"]
    # two states with identical dispersion/structure but different raw content + turn index
    s1 = _state(turn=3, progress=0.4, n_valid=3)
    s2 = _state(turn=99, progress=0.4, n_valid=3)
    phi1 = fx(s1, A, prefs, env=None)
    phi2 = fx(s2, A, prefs, env=None)
    assert np.allclose(phi1, phi2), "instance-invariant phi must ignore raw/turn identity"


def test_instance_dependent_ablation_leaks():
    fx = FeatureExtractor({"features": {"instance_dependent": True}})
    prefs = [2.0, 1.0, 0.5]
    A = ["a", "b", "c"]
    s1 = _state(turn=3, progress=0.4, n_valid=3)
    s2 = _state(turn=99, progress=0.4, n_valid=3)
    phi1 = fx(s1, A, prefs, env=None)
    phi2 = fx(s2, A, prefs, env=None)
    assert not np.allclose(phi1, phi2), "instance-dependent ablation should leak turn identity"
    assert fx.dim > FeatureExtractor({"features": {"instance_dependent": False}}).dim


def test_margin_entropy_semantics():
    fx = FeatureExtractor({"features": {"instance_dependent": False}})
    A = ["a", "b", "c"]
    s = _state(turn=1, progress=0.2, n_valid=3)
    confident = fx(s, A, [10.0, 0.0, 0.0], env=None)
    uncertain = fx(s, A, [0.0, 0.0, 0.0], env=None)
    mi = fx.names.index("margin"); ei = fx.names.index("entropy")
    assert confident[mi] > uncertain[mi]      # confident prior => larger margin
    assert confident[ei] < uncertain[ei]      # confident prior => lower entropy
    assert abs(uncertain[ei] - 1.0) < 1e-6    # uniform prior => max normalized entropy


def test_dim_includes_bias():
    fx = FeatureExtractor({"features": {"instance_dependent": False}})
    assert fx.names[-1] == "bias"
    s = _state(1, 0.1, 2)
    assert fx(s, ["a", "b"], [1.0, 0.0], env=None).shape[0] == fx.dim
