#!/usr/bin/env python
"""Produce the SimTriage main figure: (game return vs total simulator steps) Pareto frontier.

Focused & fast: trains the VoQ predictor ONCE, sweeps a DENSE lambda grid at the gate (frozen
eval), and plots SimTriage against the baselines. Skips the expensive transfer/calibration
phases of run_experiment so the headline figure comes out in ~10-15 min instead of ~40.

Usage:
    python scripts/make_main_figure.py --game catan
    python scripts/make_main_figure.py --game catan --quick
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from scripts.run_experiment import (_load_dotenv, load_cfg, minus_m2_point,  # noqa: E402
                                    run_pure_baselines, simtriage_frontier)


def dense_lambda_grid():
    # dense in the active region (~[0, 0.003] for Catan's value/cost scale), plus a tail to never
    return [0.0, 0.0002, 0.0004, 0.0006, 0.0008, 0.0010, 0.0013, 0.0016,
            0.0020, 0.0025, 0.0030, 0.0040, 0.0060, 0.0100]


def pareto(points):
    pts = sorted(set((float(c), float(v)) for c, v in points), key=lambda t: (t[0], -t[1]))
    fr, best = [], -np.inf
    for c, v in pts:
        if v > best:
            fr.append((c, v)); best = v
    return fr


def plot(simtriage_pts, baselines, m2, outpath, game):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5.5))

    # SimTriage frontier (the method) as a connected line; points are (sim_steps, return)
    sp = sorted(simtriage_pts)
    ax.plot([c for c, v in sp], [v for c, v in sp], "-o", color="#1f77b4",
            lw=2, ms=6, label="SimTriage (sweep λ)", zorder=5)

    # baselines + ablation as distinct scatter points
    style = {
        "never_query":     ("#2ca02c", "s", "never-query (≈ prior-only)"),
        "always_query":    ("#d62728", "s", "always-query (full search)"),
        "fixed_threshold": ("#9467bd", "^", "fixed-threshold gate"),
        "random_gate":     ("#8c564b", "x", "random gate"),
    }
    for name, (color, marker, label) in style.items():
        if name in baselines:
            a = baselines[name]["agg"]
            ax.scatter([a["sim_steps_mean"]], [a["return_mean"]], c=color, marker=marker, s=110,
                       label=label, zorder=6, edgecolors="k", linewidths=0.5)
    if m2 is not None:
        ax.scatter([m2["sim_steps_mean"]], [m2["return_mean"]], c="#ff7f0e", marker="D", s=90,
                   label="−M2 (imagine, no real sim)", zorder=6, edgecolors="k", linewidths=0.5)

    # Pareto frontier over everything achievable
    allpts = list(simtriage_pts)
    for name in baselines:
        a = baselines[name]["agg"]; allpts.append((a["sim_steps_mean"], a["return_mean"]))
    fr = pareto(allpts)
    ax.plot([c for c, _ in fr], [v for _, v in fr], "--", color="gray", alpha=0.6,
            label="Pareto frontier", zorder=4)

    ax.set_xlabel("total simulator steps  C  (cost)")
    ax.set_ylabel("game return  G  (Catan victory points)")
    ax.set_title(f"SimTriage: learning when to query the simulator ({game})")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", default="catan")
    ap.add_argument("--config", default=None)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--backend", default=None, choices=["heuristic", "llm", "random"])
    ap.add_argument("--provider", default=None)
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    _load_dotenv(os.path.join(ROOT, ".env"))
    if args.backend:
        cfg["policy"]["backend"] = args.backend
    if args.provider:
        cfg["policy"]["provider"] = args.provider; cfg["policy"]["backend"] = "llm"

    game = args.game
    seed0 = int(cfg["experiment"]["seed_global"])
    train = cfg["experiment"]["seeds_train"]
    heldout = cfg["experiment"]["seeds_heldout"]
    lam_grid = dense_lambda_grid()
    if args.quick:
        train = train[:5]; heldout = heldout[:5]
        lam_grid = [0.0, 0.0005, 0.001, 0.0016, 0.0025, 0.004, 0.01]

    outdir = os.path.join(ROOT, cfg["experiment"].get("output_dir", "outputs"), game)
    os.makedirs(outdir, exist_ok=True)
    print(f"=== main figure: game={game} backend={cfg['policy']['backend']} ===")
    print(f"dense lam_grid={lam_grid}\ntrain={train} heldout={heldout}\n")

    st_points, st_per_lam = simtriage_frontier(cfg, game, lam_grid, train, heldout, seed0)
    st_qrate = max((a["query_rate_mean"] for a in st_per_lam.values()), default=0.3)
    print()
    baselines = run_pure_baselines(cfg, game, heldout, cfg["simtriage"]["lam_default"], st_qrate)
    print()
    m2 = minus_m2_point(cfg, game, cfg["simtriage"]["lam_default"], train, heldout, seed0)

    outpath = os.path.join(outdir, "main_figure.png")
    plot(st_points, baselines, m2, outpath, game)

    summary = {
        "game": game, "lam_grid": lam_grid,
        "simtriage": {str(k): v for k, v in st_per_lam.items()},
        "simtriage_points": st_points,
        "baselines": {k: v["agg"] for k, v in baselines.items()},
        "minus_m2_imagine": m2,
        "figure": outpath,
    }
    with open(os.path.join(outdir, "main_figure.json"), "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"\n=== main figure saved: {outpath} ===")


if __name__ == "__main__":
    main()
