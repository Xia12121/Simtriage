"""Task 6 acceptance: QVL prototype aggregation, Hoeffding verdicts, transfer prior, persistence."""

import numpy as np

from src.qvl import QVL


def _cfg(min_count=8, radius=0.75):
    return {"qvl": {"hoeffding_alpha": 0.1, "min_count": min_count, "proto_radius": radius,
                    "voq_range": 2.0}}


def test_prototype_aggregation_and_prior():
    q = QVL(_cfg(min_count=5, radius=0.5))
    rng = np.random.default_rng(0)
    # cluster A near origin with high VoQ ~1.0; cluster B far with low VoQ ~0.0
    cA = np.zeros(4); cB = np.array([5.0, 5.0, 0.0, 0.0])
    for _ in range(30):
        q.update(cA + rng.normal(0, 0.05, 4), 1.0 + rng.normal(0, 0.05), cost=0.2)
    for _ in range(30):
        q.update(cB + rng.normal(0, 0.05, 4), 0.0 + rng.normal(0, 0.05), cost=0.2)
    assert len(q) == 2
    pa = q.predict_prior(cA)
    pb = q.predict_prior(cB)
    assert pa is not None and abs(pa - 1.0) < 0.2
    assert pb is not None and abs(pb - 0.0) < 0.2


def test_predict_prior_none_when_untrusted():
    q = QVL(_cfg(min_count=8))
    q.update(np.zeros(4), 1.0, cost=0.1)  # only 1 sample -> not trusted
    assert q.predict_prior(np.zeros(4)) is None


def test_hoeffding_verdicts():
    q = QVL(_cfg(min_count=5, radius=0.5))
    rng = np.random.default_rng(1)
    c = np.zeros(4)
    for _ in range(200):
        q.update(c + rng.normal(0, 0.02, 4), 1.0 + rng.normal(0, 0.02), cost=0.0)
    q.consolidate()
    # high VoQ (~1.0), tiny cost (0.0) => worth_query with confidence
    assert q.verdict(c, cost=0.0) == "worth_query"
    # cost far above VoQ => dont_query
    assert q.verdict(c, cost=5.0) == "dont_query"


def test_persistence_roundtrip(tmp_path):
    q = QVL(_cfg())
    rng = np.random.default_rng(2)
    for _ in range(20):
        q.update(rng.normal(0, 0.1, 4), rng.random(), cost=0.1)
    p = tmp_path / "qvl.json"
    q.save(str(p))
    q2 = QVL.load(str(p))
    assert len(q2) == len(q)
    assert np.allclose(q2.prototypes[0].centroid, q.prototypes[0].centroid)
