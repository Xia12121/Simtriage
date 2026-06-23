"""Task 5 — gate + real-simulator query operator + reference (self-supervised) label (M2).

METHOD_DESIGN §4.3 / Algorithm 1:
    gate  : query iff  g_hat_theta(phi) > lam * c(s)
    query : for each a in top-m candidates, K bounded depth-d rollouts; a_hat = argmax mean
    label : with prob rho, a high-budget reference query gives y = refQ(a_ref) - refQ(a_0)

HARD CONSTRAINT (SPEC §1 / METHOD_DESIGN §3 M2): rollouts MUST be real simulator
clone+forward (``env.rollout``), never LLM imagination. The imagination variant lives in
baselines.py as the -M2 ablation, on purpose.

Common random numbers (CRN): in seedable envs the same rollout seed is reused across
candidates so they are compared under identical luck (variance reduction, METHOD_DESIGN
§4.2). Honored only where ``env.supports_cheap_clone``.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


def gate(g_hat: float, lam: float, cost: float) -> bool:
    """The myopic value-of-computation gate (Russell-Wefald form). Strictly ``>``."""
    return float(g_hat) > float(lam) * float(cost)


def run_query(
    env,
    state,
    A: list,
    rollout_policy,
    K: int,
    depth,
    use_crn: bool = True,
    crn_seed: Optional[int] = None,
    track_len: bool = True,
) -> tuple[object, list[float]]:
    """For each candidate in A run K bounded rollouts (depth d) and average. Return
    (a_hat, q) where q[i] is the mean value of A[i] and a_hat = A[argmax q].

    ``track_len`` calibrates the query-cost EMA (True for acting queries, False for the
    high-budget reference query so it doesn't inflate the estimate)."""
    if not A:
        return None, []
    crn_seeds = None
    if use_crn and env.supports_cheap_clone:
        base = int(crn_seed) if crn_seed is not None else 0
        crn_seeds = [base + j for j in range(K)]  # shared across candidates => same luck

    q = []
    for a in A:
        vals = []
        for j in range(K):
            s = crn_seeds[j] if crn_seeds is not None else None
            vals.append(env.rollout(state, a, rollout_policy, depth, crn_seed=s, track_len=track_len))
        q.append(float(np.mean(vals)) if vals else 0.0)
    best = int(np.argmax(q))
    return A[best], q


def reference_voq(
    env,
    state,
    A: list,
    rollout_policy,
    K_ref: int,
    depth_ref,
    use_crn: bool = True,
    crn_seed: Optional[int] = None,
) -> tuple[float, object, list[float]]:
    """High-budget reference query producing the self-supervised label.

    y(s) = Qhat_ref(a_ref) - Qhat_ref(a_0)  with a_0 = A[0]  (METHOD_DESIGN §4.2).
    y >= 0 by construction (the query can only help under the reference estimator), matching
    g(s) = E[Q(a_hat) - Q(a_0)] >= 0.

    NOTE on bias: selecting a_ref by the same rollouts used to evaluate it introduces a
    small max-selection optimism. We mitigate with a high K_ref + CRN + deeper depth_ref
    (per spec). This stays faithful to the Algorithm-1 definition.
    """
    a_ref, q = run_query(env, state, A, rollout_policy, K_ref, depth_ref, use_crn, crn_seed,
                         track_len=False)  # reference is high-budget; don't let it skew query_cost
    if not q:
        return 0.0, None, []
    y = float(max(q) - q[0])  # improvement of the simulator's pick over the prior's pick
    return y, a_ref, q
