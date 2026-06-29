#!/usr/bin/env python
"""96-core per-seed PARALLEL frontier experiment — nail significance + smooth curves.

Trains the VoQ predictor ONCE (sequential), then fans out (method, lambda, seed) episodes
across a process Pool. With ~96 cores this runs dozens of seeds in minutes, which is what we
need to (a) push query_no_value below p<0.05 and (b) smooth the Pareto frontier.

Outputs: outputs/<game>/parallel_results.json + parallel_frontier.png + a significance test.

Usage:
    python scripts/parallel_experiment.py --game mock --seeds 12 --workers 8   # local smoke
    python scripts/parallel_experiment.py --game catan --seeds 48 --workers 48 # remote
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import yaml  # noqa: E402

from src import metrics  # noqa: E402
from src.baselines import make_baseline  # noqa: E402
from src.envs.base_env import make_env  # noqa: E402
from src.qvl import QVL  # noqa: E402
from src.simtriage import SimTriage  # noqa: E402

# module globals (inherited by forked workers on Linux)
_CFG = None
_GAME = None
_THETA = None  # dict: {"A","b","n","qvl_path"}
_DENSE_LAMS = [0.0, 0.0002, 0.0004, 0.0006, 0.0008, 0.001, 0.0013, 0.0016,
               0.002, 0.0025, 0.003, 0.004, 0.006, 0.01]


def _init_worker(cfg, game, theta):
    """Set worker-process globals explicitly (robust to spawn on macOS and fork on Linux)."""
    global _CFG, _GAME, _THETA
    _CFG, _GAME, _THETA = cfg, game, theta


def _load_simtriage_frozen(lam):
    ag = SimTriage(_CFG, lam=lam, game=_GAME)
    ag.voq.A = _THETA["A"].copy()
    ag.voq.b = _THETA["b"].copy()
    ag.voq.n = int(_THETA["n"])
    ag.voq.cold_start_prior = None  # frozen, past warmup
    ag.qvl = QVL.load(_THETA["qvl_path"])
    return ag


def worker(task):
    """task = (method, lam_or_None, seed, query_prob_or_None). Returns a metrics dict."""
    method, lam, seed, qprob = task
    if method == "simtriage":
        ag = _load_simtriage_frozen(lam)
        r = ag.run_episode(make_env(_GAME, _CFG), seed, learn=False, reference_rate=0.0)
    else:
        kw = {"game": _GAME}
        if method == "random_gate" and qprob is not None:
            kw["query_prob"] = float(qprob)
        ag = make_baseline(method, _CFG, _CFG["simtriage"]["lam_default"], **kw)
        r = ag.run_episode(make_env(_GAME, _CFG), seed)
    return {"method": method, "lam": lam, "seed": seed, "G": float(r.G), "C": int(r.C),
            "nq": r.n_queries, "nd": r.n_decisions, "nh": r.n_hits}


def main():
    global _CFG, _GAME, _THETA
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", default="catan")
    ap.add_argument("--seeds", type=int, default=48)
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--train", type=int, default=12)
    args = ap.parse_args()

    _CFG = yaml.safe_load(open(os.path.join(ROOT, "configs", "sim.yaml")))
    _GAME = args.game
    heldout = [1000 + i for i in range(args.seeds)]      # fresh held-out block
    train = list(range(args.train))
    outdir = os.path.join(ROOT, _CFG["experiment"].get("output_dir", "outputs"), _GAME)
    os.makedirs(outdir, exist_ok=True)
    t0 = time.time()
    print(f"=== parallel frontier: game={_GAME} seeds={args.seeds} workers={args.workers} ===", flush=True)

    # 1) train VoQ + QVL ONCE (sequential)
    print(f"[train] SimTriage on {len(train)} seeds (sequential)...", flush=True)
    base = SimTriage(_CFG, lam=_DENSE_LAMS[0], game=_GAME, seed=_CFG["experiment"]["seed_global"])
    base.evolve(make_env(_GAME, _CFG), train, learn=True)
    base.qvl.consolidate(prune=False)
    qvl_path = os.path.join(outdir, "parallel_qvl.json")
    base.qvl.save(qvl_path)
    _THETA = {"A": base.voq.A, "b": base.voq.b, "n": base.voq.n, "qvl_path": qvl_path}
    print(f"[train] done in {time.time()-t0:.0f}s; VoQ.n={base.voq.n}, |QVL|={len(base.qvl)}", flush=True)

    # 2) build task list
    tasks = []
    for m in ["never_query", "always_query", "fixed_threshold", "random_gate"]:
        for s in heldout:
            tasks.append((m, None, s, 0.4))
    for lam in _DENSE_LAMS:
        for s in heldout:
            tasks.append(("simtriage", lam, s, None))
    print(f"[fanout] {len(tasks)} episodes across {args.workers} workers...", flush=True)

    # 3) parallel map
    with Pool(processes=args.workers, initializer=_init_worker, initargs=(_CFG, _GAME, _THETA)) as pool:
        rows = pool.map(worker, tasks)
    print(f"[fanout] done in {time.time()-t0:.0f}s total", flush=True)

    # 4) aggregate
    def agg(rs):
        G = np.array([r["G"] for r in rs], float)
        C = np.array([r["C"] for r in rs], float)
        qr = np.array([r["nq"] / max(1, r["nd"]) for r in rs], float)
        hr = np.array([r["nh"] / max(1, r["nq"]) for r in rs], float)
        return {"return_mean": float(G.mean()), "return_std": float(G.std()),
                "sim_steps_mean": float(C.mean()), "query_rate": float(qr.mean()),
                "hit_rate": float(hr.mean()), "n": len(rs), "returns": G.tolist()}

    base_rows = {m: [r for r in rows if r["method"] == m] for m in
                 ["never_query", "always_query", "fixed_threshold", "random_gate"]}
    baselines = {m: agg(rs) for m, rs in base_rows.items()}
    st = {}
    for lam in _DENSE_LAMS:
        rs = [r for r in rows if r["method"] == "simtriage" and r["lam"] == lam]
        st[lam] = agg(rs)

    # 5) significance: always vs never
    nv = baselines["never_query"]["returns"]
    av = baselines["always_query"]["returns"]
    tstat, pval = metrics.welch_ttest(av, nv)
    rel = (np.mean(av) - np.mean(nv)) / max(abs(np.mean(nv)), 1.0)
    print(f"\n[significance] always vs never: mean {np.mean(av):.2f} vs {np.mean(nv):.2f} | "
          f"rel_gain={rel:.3f} | p={pval:.4f} | n={len(nv)} per arm", flush=True)
    best = max(st.items(), key=lambda kv: kv[1]["return_mean"])
    print(f"[frontier] best SimTriage: lam={best[0]} return={best[1]['return_mean']:.2f} "
          f"(query {best[1]['query_rate']:.2f}); always={baselines['always_query']['return_mean']:.2f} "
          f"fixed={baselines['fixed_threshold']['return_mean']:.2f} never={baselines['never_query']['return_mean']:.2f}", flush=True)

    # 6) plot
    st_points = [(st[l]["sim_steps_mean"], st[l]["return_mean"]) for l in _DENSE_LAMS]
    method_points = {"simtriage": st_points}
    for m, a in baselines.items():
        method_points[m] = [(a["sim_steps_mean"], a["return_mean"])]
    try:
        metrics.plot_frontier(method_points, os.path.join(outdir, "parallel_frontier.png"),
                              title=f"SimTriage frontier ({_GAME}, {args.seeds} seeds, parallel)")
    except Exception as e:
        print("[plot] skipped:", e)

    # 7) dump
    summary = {"game": _GAME, "seeds": args.seeds, "train": len(train),
               "significance": {"rel_gain": float(rel), "p": float(pval), "n": len(nv)},
               "simtriage": {str(k): v for k, v in st.items()}, "baselines": baselines,
               "wallclock_s": time.time() - t0}
    with open(os.path.join(outdir, "parallel_results.json"), "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"\n=== done in {(time.time()-t0)/60:.1f} min -> {outdir}/parallel_results.json ===", flush=True)


if __name__ == "__main__":
    main()
