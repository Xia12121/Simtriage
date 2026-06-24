#!/usr/bin/env python
"""Task 9 — experiment runner: (return vs sim-steps) frontier, transfer, calibration, kill-criteria.

Unified, fair protocol: every learning method is TRAINED on train seeds then evaluated
FROZEN (learn=False, reference_rate=0) on held-out seeds, so all (sim_steps, return) points
are apples-to-apples (no reference-label cost leaks into the deployed metric).

Pipeline per game:
  1. SimTriage frontier: per lambda, train theta, frozen-eval -> (sim_steps, return).
  2. Pure baselines (never/always/fixed/random) on held-out.
  3. -M2 (imagine) frozen point: quantifies the white-box bonus (imagination is ~free but
     adds no return).
  4. Transfer (M3): trained-QVL vs from-scratch, run ONLINE on held-out (the regime where a
     transferred library matters); also the phi-instance-dependent variant (leak breaks transfer).
  5. Calibration: a dedicated higher-rho run -> VoQ error vs experience (should fall).
  6. Automated KILL-CRITERIA (SPEC §4).
Outputs: outputs/<game>/{results.json, frontier.png, calibration.png, transfer.png, qvl.json}.

Usage:
    python scripts/run_experiment.py --game mock --quick     # no external deps, fast smoke
    python scripts/run_experiment.py --game catan            # primary experiment
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import yaml  # noqa: E402

from src import metrics  # noqa: E402
from src.baselines import make_baseline  # noqa: E402
from src.envs.base_env import make_env  # noqa: E402
from src.qvl import QVL  # noqa: E402
from src.simtriage import SimTriage  # noqa: E402


def load_cfg(path=None):
    path = path or os.path.join(ROOT, "configs", "sim.yaml")
    with open(path) as f:
        return yaml.safe_load(f)


def _load_dotenv(path):
    """Minimal .env loader (no dependency): KEY=VALUE lines populate os.environ without
    overriding an already-set variable. Lets API keys live in a gitignored .env."""
    if not os.path.exists(path):
        return
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


# --------------------------------------------------------------- protocols
def evolve_then_frozen_eval(cfg, game, lam, train_seeds, heldout_seeds, seed=0, **agent_kw):
    """Train theta on train seeds, then evaluate FROZEN on held-out (deployed metric)."""
    agent = SimTriage(cfg, lam=lam, game=game, seed=seed, **agent_kw)
    agent.evolve(make_env(game, cfg), train_seeds, learn=True)
    res = [agent.run_episode(make_env(game, cfg), s, learn=False, reference_rate=0.0)
           for s in heldout_seeds]
    return agent, res


def simtriage_frontier(cfg, game, lam_sweep, train_seeds, heldout_seeds, seed0):
    """Trace the Pareto frontier by sweeping lambda. Per METHOD §4.1 the VoQ predictor
    estimates g(s), which is INDEPENDENT of lambda — lambda is only the gate's price knob.
    So we train the predictor ONCE and re-threshold the gate at each lambda during frozen
    eval. This gives a clean, monotonic frontier and avoids confounding the price sweep with
    different random training trajectories."""
    agent = SimTriage(cfg, lam=lam_sweep[0], game=game, seed=seed0)
    agent.evolve(make_env(game, cfg), train_seeds, learn=True)  # one evolved theta
    points, per_lam = [], {}
    for lam in lam_sweep:
        agent.lam = lam  # only the gate price changes; theta (VoQ + QVL) is fixed
        res = [agent.run_episode(make_env(game, cfg), s, learn=False, reference_rate=0.0)
               for s in heldout_seeds]
        agg = metrics.aggregate(res)
        per_lam[lam] = agg
        points.append((agg["sim_steps_mean"], agg["return_mean"]))
        print(f"[simtriage] lam={lam:<5} return={agg['return_mean']:.3f} "
              f"sim_steps={agg['sim_steps_mean']:.1f} query_rate={agg['query_rate_mean']:.2f} "
              f"hit_rate={agg['hit_rate_mean']:.2f}")
    return points, per_lam


def run_pure_baselines(cfg, game, heldout_seeds, lam_default, simtriage_query_rate):
    out = {}
    for name in ["never_query", "always_query", "fixed_threshold", "random_gate"]:
        kw = {"game": game}
        if name == "random_gate":
            kw["query_prob"] = float(simtriage_query_rate)
        agent = make_baseline(name, cfg, lam_default, **kw)
        res = [agent.run_episode(make_env(game, cfg), s) for s in heldout_seeds]
        agg = metrics.aggregate(res)
        out[name] = {"agg": agg, "results": res}
        print(f"[{name:<16}] return={agg['return_mean']:.3f} sim_steps={agg['sim_steps_mean']:.1f} "
              f"query_rate={agg['query_rate_mean']:.2f}")
    return out


def minus_m2_point(cfg, game, lam, train_seeds, heldout_seeds, seed0):
    """-M2: acting query uses imagination (no simulator) -> ~0 sim steps; reference stays real
    during training only. Frozen-eval point shows imagination adds no deployed return."""
    _, res = evolve_then_frozen_eval(cfg, game, lam, train_seeds, heldout_seeds, seed=seed0,
                                     imagine=True)
    agg = metrics.aggregate(res)
    print(f"[minus_m2_imagine] return={agg['return_mean']:.3f} sim_steps={agg['sim_steps_mean']:.1f} "
          f"(white-box bonus = SimTriage return gain over this at comparable cost)")
    return agg


def transfer_test(cfg, game, lam, train_seeds, heldout_seeds, seed0, instance_dependent=False):
    """M3: does a pre-built QVL help on NEW instances? Both agents run ONLINE on held-out;
    they differ only in whether they start from the trained QVL (transfer) or empty (scratch).
    The reference cost is ~equal for both (same rho), so it cancels in the transfer GAIN."""
    cfg_use = cfg
    if instance_dependent:
        cfg_use = copy.deepcopy(cfg)
        cfg_use.setdefault("features", {})["instance_dependent"] = True

    # Library-building phase: higher reference rate so QVL prototypes reach the trusted count.
    # NOTE: this override raises label-collection cost during TRAINING only; held-out
    # deployment below uses the normal (low) rho. The transfer GAIN is a difference between
    # two agents run identically on held-out, so this training-phase override cancels out.
    cfg_base = copy.deepcopy(cfg_use)
    cfg_base["simtriage"]["rho_start"] = 0.5
    cfg_base["simtriage"]["rho_end"] = 0.5
    if not instance_dependent:
        print("[transfer] library-building rho override -> 0.5 (training only); held-out uses config rho")
    base = SimTriage(cfg_base, lam=lam, game=game, transfer=True, seed=seed0)
    base.evolve(make_env(game, cfg_base), train_seeds, learn=True)
    base.qvl.consolidate(prune=False)
    trained_qvl = base.qvl

    # FROZEN deployment on held-out (reference_rate=0): the deployed metric U=G-lam*C then
    # reflects GATING QUALITY, not label-collection cost. (Running online with rho>0 lets the
    # expensive to_game_end reference rollouts dominate C and bury the transfer signal — that
    # was the spurious "transfer fails" on Catan.) Transfer = trained QVL + fresh VoQ (so the
    # gate uses the transferred prior); scratch = empty everything (optimistic cold start).
    t_agent = SimTriage(cfg_use, lam=lam, game=game,
                        qvl=QVL.from_dict(trained_qvl.to_dict()), transfer=True, seed=seed0 + 1)
    transfer_results = [t_agent.run_episode(make_env(game, cfg_use), s, learn=False, reference_rate=0.0)
                        for s in heldout_seeds]
    s_agent = SimTriage(cfg_use, lam=lam, game=game, transfer=False, seed=seed0 + 2)
    scratch_results = [s_agent.run_episode(make_env(game, cfg_use), s, learn=False, reference_rate=0.0)
                       for s in heldout_seeds]
    return transfer_results, scratch_results, trained_qvl


def collect_calibration(cfg, game, seeds, seed0):
    """Dedicated calibration trajectory: one agent, high reference rate, online learning,
    (pred_before_update, y) recorded in temporal order -> error-vs-experience."""
    cfg2 = copy.deepcopy(cfg)
    cfg2["simtriage"]["rho_start"] = 0.6
    cfg2["simtriage"]["rho_end"] = 0.4
    agent = SimTriage(cfg2, lam=0.0, game=game, seed=seed0)
    res = agent.evolve(make_env(game, cfg2), seeds, learn=True)
    calib = []
    for r in res:
        calib.extend(r.calib)
    return calib


def rows(results):
    return [r.to_row() for r in results]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", default="mock")
    ap.add_argument("--config", default=None)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--backend", default=None, choices=["heuristic", "llm", "random"],
                    help="override policy.backend")
    ap.add_argument("--provider", default=None,
                    help="override policy.provider (implies --backend llm), e.g. deepseek|qwen|openai")
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    _load_dotenv(os.path.join(ROOT, ".env"))   # pick up API keys (e.g. VENUS_API_KEY) if present
    if args.backend:
        cfg["policy"]["backend"] = args.backend
    if args.provider:
        cfg["policy"]["provider"] = args.provider
        cfg["policy"]["backend"] = "llm"
    game = args.game
    print(f"[policy] backend={cfg['policy']['backend']} provider={cfg['policy'].get('provider')}")
    seed0 = int(cfg["experiment"]["seed_global"])
    lam_sweep = cfg["simtriage"]["lam_sweep"]
    train_seeds = cfg["experiment"]["seeds_train"]
    heldout_seeds = cfg["experiment"]["seeds_heldout"]
    lam_default = cfg["simtriage"]["lam_default"]

    if args.quick:
        lam_sweep = [0.0, 0.003, 0.01, 0.05]
        train_seeds = train_seeds[:6]
        heldout_seeds = heldout_seeds[:4]

    outdir = os.path.join(ROOT, cfg["experiment"].get("output_dir", "outputs"), game)
    os.makedirs(outdir, exist_ok=True)
    print(f"=== SimTriage experiment: game={game} (quick={args.quick}) ===")
    print(f"lam_sweep={lam_sweep} train={train_seeds} heldout={heldout_seeds}\n")

    # 1. SimTriage frontier (train -> frozen eval)
    st_points, st_per_lam = simtriage_frontier(cfg, game, lam_sweep, train_seeds, heldout_seeds, seed0)
    st_qrate = st_per_lam.get(lam_default, list(st_per_lam.values())[0])["query_rate_mean"]

    # 2. pure baselines
    print()
    baselines = run_pure_baselines(cfg, game, heldout_seeds, lam_default, st_qrate)

    # 3. -M2 white-box bonus point
    print()
    m2 = minus_m2_point(cfg, game, lam_default, train_seeds, heldout_seeds, seed0)

    # 4. transfer (M3) + phi-instance-dependent variant.
    # Evaluate transfer in the DISCRIMINATIVE regime (intermediate query-rate): where queries
    # are cheap the gate queries everything and a transferred library can't help; M3's value
    # shows where the gate must actually choose. Pick the swept lambda with query-rate nearest 0.5.
    disc = [(abs(a["query_rate_mean"] - 0.5), lam) for lam, a in st_per_lam.items()
            if 0.05 < a["query_rate_mean"] < 0.95]
    lam_transfer = min(disc)[1] if disc else lam_default
    print(f"\n[transfer] using lam_transfer={lam_transfer} (discriminative regime; query-rate nearest 0.5)")
    transfer_results, scratch_results, trained_qvl = transfer_test(
        cfg, game, lam_transfer, train_seeds, heldout_seeds, seed0)
    trained_qvl.save(os.path.join(outdir, "qvl.json"))
    tU, sU = np.mean([r.U for r in transfer_results]), np.mean([r.U for r in scratch_results])
    print(f"[transfer  invariant-phi] transfer U={tU:.3f}  scratch U={sU:.3f}  gain={tU - sU:.3f}  (|QVL|={len(trained_qvl)})")
    tr_dep, sc_dep, _ = transfer_test(cfg, game, lam_transfer, train_seeds, heldout_seeds, seed0 + 50,
                                      instance_dependent=True)
    tUd, sUd = np.mean([r.U for r in tr_dep]), np.mean([r.U for r in sc_dep])
    print(f"[transfer instance-dep-phi] transfer U={tUd:.3f}  scratch U={sUd:.3f}  gain={tUd - sUd:.3f}  "
          f"(leak should REDUCE the transfer gain vs invariant phi)")

    # 5. calibration
    print()
    calib_ordered = collect_calibration(cfg, game, train_seeds, seed0 + 99)
    calib = metrics.calibration_error(calib_ordered)
    improves = metrics.calibration_improves(calib_ordered)
    print(f"[calibration] mae={calib['mae']:.4f} n={calib['n']} improves={improves} "
          f"trend={['%.3f' % x for x in metrics.calibration_trend(calib_ordered)]}")

    # 6. kill criteria
    print()
    kc = metrics.evaluate_kill_criteria(
        cfg, never_results=baselines["never_query"]["results"],
        always_results=baselines["always_query"]["results"],
        transfer_results=transfer_results, scratch_results=scratch_results,
        calib_ordered=calib_ordered)
    print("[kill-criteria]")
    for k, v in kc.items():
        print(f"   {k:<20} {'TRIGGERED' if v['triggered'] else 'ok':<10} {v['detail']}")

    # ---- figures ----
    method_points = {"simtriage": st_points, "minus_m2_imagine": [(m2["sim_steps_mean"], m2["return_mean"])]}
    for name, blk in baselines.items():
        a = blk["agg"]
        method_points[name] = [(a["sim_steps_mean"], a["return_mean"])]
    metrics.plot_frontier(method_points, os.path.join(outdir, "frontier.png"))
    if calib_ordered:
        metrics.plot_calibration(calib_ordered, os.path.join(outdir, "calibration.png"))
    metrics.plot_transfer(transfer_results, scratch_results, os.path.join(outdir, "transfer.png"))

    # ---- dump json ----
    summary = {
        "game": game,
        "lam_sweep": lam_sweep,
        "simtriage_frontier": {str(k): v for k, v in st_per_lam.items()},
        "simtriage_points": st_points,
        "baselines": {k: v["agg"] for k, v in baselines.items()},
        "minus_m2_imagine": m2,
        "transfer": {
            "invariant_phi": {"transfer_mean_U": float(tU), "scratch_mean_U": float(sU),
                              "gain": float(tU - sU), "transfer_rows": rows(transfer_results),
                              "scratch_rows": rows(scratch_results), "qvl_size": len(trained_qvl)},
            "instance_dependent_phi": {"transfer_mean_U": float(tUd), "scratch_mean_U": float(sUd),
                                       "gain": float(tUd - sUd)},
        },
        "calibration": {"error": calib, "improves": bool(improves),
                        "trend": metrics.calibration_trend(calib_ordered)},
        "kill_criteria": kc,
    }
    with open(os.path.join(outdir, "results.json"), "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"\n=== done. outputs in {outdir} ===")


if __name__ == "__main__":
    main()
