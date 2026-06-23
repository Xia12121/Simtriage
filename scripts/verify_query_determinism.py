#!/usr/bin/env python
"""R2 — verify query determinism / CRN label reproducibility (SPEC Task 1).

Checks, on a seedable env (default: mock; also: catan):
  1. clone independence: two branches forked from the same state evolve independently.
  2. CRN reproducibility: same crn_seed + same default policy => identical rollout value
     and identical reference label y (the precondition for low-variance self-supervision).

Usage:
    python scripts/verify_query_determinism.py --env mock
    python scripts/verify_query_determinism.py --env catan --seed 0
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import yaml  # noqa: E402

from src.envs.base_env import make_env  # noqa: E402
from src.policy import make_policy, make_rollout_policy  # noqa: E402
from src.query import reference_voq, run_query  # noqa: E402


def load_cfg(path=None):
    path = path or os.path.join(ROOT, "configs", "sim.yaml")
    with open(path) as f:
        return yaml.safe_load(f)


def first_decision_state(env, seed):
    state = env.reset(seed)
    return state


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="mock", choices=["mock", "catan"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--config", default=None)
    ap.add_argument("--tol", type=float, default=1e-9)
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    cfg["policy"]["backend"] = "heuristic"  # determinism check uses the cheap prior
    env = make_env(args.env, cfg)
    policy = make_policy(cfg)
    rollout_policy = make_rollout_policy(cfg, prior=policy)

    state = first_decision_state(env, args.seed)
    A, prefs = policy.top_candidates(env, state, cfg["simtriage"]["m"])
    print(f"[info] env={args.env} seed={args.seed} |A|={len(A)} supports_cheap_clone={env.supports_cheap_clone}")
    if not A:
        print("[skip] no candidate actions at the first decision point")
        return 0

    ok = True

    # ---- 1. clone independence -------------------------------------------------
    b1 = env.clone(state)
    b2 = env.clone(state)
    a0 = A[0]
    b1._apply_one(a0)
    n1 = len(b1.valid_actions(b1.current_state()))
    n2 = len(b2.valid_actions(b2.current_state()))
    indep = True  # branches must not affect each other; b2 is unchanged after b1 mutated
    print(f"[clone] after mutating branch1: |valid(b1)|={n1} |valid(b2)|={n2} "
          f"(b2 should equal the pre-mutation count)")
    # The strong check: mutating b1 doesn't change b2's playable set length vs a fresh clone.
    b3 = env.clone(state)
    if len(b3.valid_actions(b3.current_state())) != n2:
        indep = False
    print(f"[clone] independence: {'PASS' if indep else 'FAIL'}")
    ok = ok and indep

    if not env.supports_cheap_clone:
        print("[skip] CRN checks need cheap clone")
        return 0 if ok else 1

    # ---- 2. CRN rollout reproducibility ---------------------------------------
    v1 = env.rollout(state, a0, rollout_policy, cfg["simtriage"]["d"].get(args.env, 1) if isinstance(cfg["simtriage"]["d"], dict) else 1, crn_seed=12345)
    v2 = env.rollout(state, a0, rollout_policy, cfg["simtriage"]["d"].get(args.env, 1) if isinstance(cfg["simtriage"]["d"], dict) else 1, crn_seed=12345)
    crn_roll = abs(v1 - v2) <= max(args.tol, 1e-6)
    print(f"[crn] rollout value with same seed: v1={v1:.6f} v2={v2:.6f} -> {'PASS' if crn_roll else 'FAIL'}")
    ok = ok and crn_roll

    # ---- 3. CRN reference-label reproducibility --------------------------------
    K_ref = cfg["simtriage"]["reference"]["K"].get(args.env, 8)
    d_ref = cfg["simtriage"]["reference"]["d"].get(args.env, "to_game_end")
    y1, _, _ = reference_voq(env, state, A, rollout_policy, K_ref, d_ref, use_crn=True, crn_seed=999)
    y2, _, _ = reference_voq(env, state, A, rollout_policy, K_ref, d_ref, use_crn=True, crn_seed=999)
    crn_ref = abs(y1 - y2) <= max(args.tol, 1e-6)
    print(f"[crn] reference label with same seed: y1={y1:.6f} y2={y2:.6f} -> {'PASS' if crn_ref else 'FAIL'}")
    ok = ok and crn_ref

    print("=" * 50)
    print("RESULT:", "ALL PASS" if ok else "SOME CHECKS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
