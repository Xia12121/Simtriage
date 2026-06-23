#!/usr/bin/env python
"""Small-scale REAL-LLM SimTriage experiment on Catan.

Uses the frozen Venus LLM (deepseek-v4-pro) as the prior. Because each LLM call is ~10s and
an episode has ~80-120 multi-candidate decisions (~15 min/episode), a full frontier is
infeasible; this runs a focused, complete comparison at a single lambda:
  SimTriage(λ) [trained] vs never-query vs always-query vs fixed-threshold, on held-out seeds.
Prints per-episode progress (so the long run is monitorable) and saves a JSON summary.

Disk cache is ON: methods share the same early states on a given seed (before their decisions
diverge), so those LLM calls are reused across methods.

Usage:
    python scripts/llm_experiment.py            # train [0,1], heldout [100,101,102]
    python scripts/llm_experiment.py --train 0 1 2 --heldout 100 101 102 103
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from scripts.run_experiment import _load_dotenv, load_cfg  # noqa: E402
from src.baselines import make_baseline  # noqa: E402
from src.envs.base_env import make_env  # noqa: E402
from src.simtriage import SimTriage  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", default="catan")
    ap.add_argument("--provider", default="venus")
    ap.add_argument("--train", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--heldout", type=int, nargs="+", default=[100, 101, 102])
    ap.add_argument("--lam", type=float, default=None)
    args = ap.parse_args()

    cfg = load_cfg()
    _load_dotenv(os.path.join(ROOT, ".env"))
    cfg["policy"]["backend"] = "llm"
    cfg["policy"]["provider"] = args.provider
    cfg["policy"]["cache"] = True  # reuse shared-prefix LLM calls across methods/seeds
    game = args.game
    lam = float(args.lam) if args.lam is not None else float(cfg["simtriage"]["lam_default"])
    seed0 = int(cfg["experiment"]["seed_global"])
    t_all = time.time()

    print(f"=== REAL-LLM experiment: game={game} provider={args.provider} "
          f"model={cfg['policy']['providers'][args.provider]['model']} lam={lam} ===")
    print(f"train={args.train} heldout={args.heldout}\n", flush=True)

    def run_episodes(agent, seeds, is_simtriage, learn=False):
        rows = []
        for s in seeds:
            t = time.time()
            if is_simtriage:
                r = agent.run_episode(make_env(game, cfg), s, learn=learn, reference_rate=(None if learn else 0.0))
            else:
                r = agent.run_episode(make_env(game, cfg), s)
            rows.append(r)
            print(f"   seed {s}: return={r.G:.1f} steps={r.C} q_rate={r.n_queries/max(1,r.n_decisions):.2f} "
                  f"hit={r.n_hits/max(1,r.n_queries):.2f} dec={r.n_decisions} ({time.time()-t:.0f}s)", flush=True)
        return rows

    results = {}

    # ---- SimTriage: train then frozen eval ----
    print("[SimTriage] training on", args.train, flush=True)
    agent = SimTriage(cfg, lam=lam, game=game, seed=seed0)
    run_episodes(agent, args.train, is_simtriage=True, learn=True)
    print("[SimTriage] frozen eval on heldout", flush=True)
    results["simtriage"] = run_episodes(agent, args.heldout, is_simtriage=True, learn=False)

    # ---- baselines ----
    for name in ["never_query", "always_query", "fixed_threshold"]:
        print(f"[{name}] eval on heldout", flush=True)
        bag = make_baseline(name, cfg, lam, game=game)
        results[name] = run_episodes(bag, args.heldout, is_simtriage=False)

    # ---- summary ----
    print("\n=== SUMMARY (real LLM prior) ===", flush=True)
    summary = {"game": game, "provider": args.provider, "lam": lam,
               "train": args.train, "heldout": args.heldout, "methods": {}}
    for name, rs in results.items():
        G = np.mean([r.G for r in rs]); C = np.mean([r.C for r in rs])
        qr = np.mean([r.n_queries / max(1, r.n_decisions) for r in rs])
        hit = np.mean([r.n_hits / max(1, r.n_queries) for r in rs])
        summary["methods"][name] = {"return_mean": float(G), "sim_steps_mean": float(C),
                                    "query_rate": float(qr), "hit_rate": float(hit),
                                    "returns": [float(r.G) for r in rs]}
        print(f"  {name:16s} return={G:.2f} sim_steps={C:.0f} q_rate={qr:.2f} hit={hit:.2f}", flush=True)

    outdir = os.path.join(ROOT, cfg["experiment"].get("output_dir", "outputs"), game)
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "llm_experiment.json"), "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"\n=== done in {(time.time()-t_all)/60:.1f} min -> {outdir}/llm_experiment.json ===", flush=True)


if __name__ == "__main__":
    main()
