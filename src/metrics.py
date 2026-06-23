"""Task 9 — metrics & main figures.

Reports (SPEC §1 Task 9 / METHOD_DESIGN §6):
  * (return vs total sim-steps) Pareto frontier, swept over lambda — SimTriage should
    DOMINATE the never-query / always-query endpoints.
  * Transfer curve: QVL-transfer vs from-scratch on held-out instances.
  * VoQ calibration error vs experience (should fall).
  * Query hit-rate: fraction of queries that changed the decision.

Plus automated KILL-CRITERIA checks (SPEC §4 / METHOD_DESIGN §7).
"""

from __future__ import annotations

from typing import Iterable, Optional

import numpy as np


# --------------------------------------------------------------- aggregation
def aggregate(results) -> dict:
    if not results:
        return {}
    G = np.array([r.G for r in results], float)
    C = np.array([r.C for r in results], float)
    U = np.array([r.U for r in results], float)
    qr = np.array([r.n_queries / max(1, r.n_decisions) for r in results], float)
    hr = np.array([r.n_hits / max(1, r.n_queries) for r in results], float)
    return {
        "lam": results[0].lam,
        "return_mean": float(G.mean()), "return_std": float(G.std()),
        "sim_steps_mean": float(C.mean()), "sim_steps_std": float(C.std()),
        "utility_mean": float(U.mean()), "utility_std": float(U.std()),
        "query_rate_mean": float(qr.mean()),
        "hit_rate_mean": float(hr.mean()),
        "n_episodes": len(results),
    }


# ------------------------------------------------------------ Pareto frontier
def pareto_frontier(points: Iterable[tuple[float, float]]) -> list[tuple[float, float]]:
    """Non-dominated set for (cost=sim_steps ↓, value=return ↑). Returns sorted by cost."""
    pts = sorted(set((float(c), float(v)) for c, v in points), key=lambda t: (t[0], -t[1]))
    frontier, best_v = [], -np.inf
    for c, v in pts:
        if v > best_v:
            frontier.append((c, v))
            best_v = v
    return frontier


def dominates_endpoints(simtriage_points, never_point, always_point) -> dict:
    """Check the main-figure claim: at matched sim-steps SimTriage >= never on return, and
    at matched return SimTriage <= always on sim-steps."""
    fr = pareto_frontier(simtriage_points)
    out = {"frontier": fr}
    if never_point is not None:
        nc, nv = never_point
        # best return achievable at cost <= ~never cost (never cost ~ 0)
        cand = [v for c, v in fr if c <= nc + 1e-6] or [v for _, v in fr]
        out["beats_never_return_gain"] = float(max(cand) - nv) if cand else 0.0
    if always_point is not None:
        ac, av = always_point
        cand = [c for c, v in fr if v >= av - 1e-6]
        out["beats_always_cost_saving"] = float(ac - min(cand)) if cand else 0.0
    return out


# --------------------------------------------------------- calibration error
def calibration_error(calib_pairs) -> dict:
    """MSE/MAE between predicted g_hat (before update) and reference label y."""
    if not calib_pairs:
        return {"mse": float("nan"), "mae": float("nan"), "n": 0}
    pred = np.array([p for p, _ in calib_pairs], float)
    y = np.array([t for _, t in calib_pairs], float)
    err = pred - y
    return {"mse": float(np.mean(err**2)), "mae": float(np.mean(np.abs(err))), "n": len(y)}


def calibration_trend(calib_pairs_ordered, n_bins: int = 5) -> list[float]:
    """MAE per temporal bin (earliest -> latest). Should trend DOWN as theta evolves."""
    if not calib_pairs_ordered:
        return []
    arr = np.array(calib_pairs_ordered, float)
    bins = np.array_split(arr, min(n_bins, len(arr)))
    return [float(np.mean(np.abs(b[:, 0] - b[:, 1]))) for b in bins if len(b)]


def calibration_improves(calib_pairs_ordered, tol: float = 0.0) -> bool:
    """True if late-window calibration error < early-window error (learnability check)."""
    trend = calibration_trend(calib_pairs_ordered)
    if len(trend) < 2:
        return False
    return (trend[0] - trend[-1]) > tol


# --------------------------------------------------------------- significance
def welch_ttest(a, b) -> tuple[float, float]:
    a = np.asarray(a, float); b = np.asarray(b, float)
    try:
        from scipy import stats

        t, p = stats.ttest_ind(a, b, equal_var=False)
        return float(t), float(p)
    except Exception:
        # fallback: crude z on mean difference
        na, nb = len(a), len(b)
        if na < 2 or nb < 2:
            return 0.0, 1.0
        se = np.sqrt(a.var(ddof=1) / na + b.var(ddof=1) / nb)
        if se == 0:
            return 0.0, 1.0
        z = (a.mean() - b.mean()) / se
        from math import erf, sqrt

        p = 2 * (1 - 0.5 * (1 + erf(abs(z) / sqrt(2))))
        return float(z), float(p)


# ------------------------------------------------------------ kill criteria
def evaluate_kill_criteria(cfg: dict, *, never_results=None, always_results=None,
                           transfer_results=None, scratch_results=None,
                           calib_ordered=None) -> dict:
    """Automated stop-loss checks (SPEC §4). Returns {criterion: {triggered, detail, action}}."""
    kc = cfg.get("kill_criteria", {})
    out = {}

    # 1) query_no_value: always-query must beat never-query return significantly
    if never_results and always_results:
        nv = [r.G for r in never_results]
        av = [r.G for r in always_results]
        # floor the denominator at 1.0 so the relative gain doesn't explode / flip sign when
        # the never-query mean return is near zero or negative (e.g. win-rate or VP-margin returns).
        denom = max(abs(np.mean(nv)), 1.0)
        rel = (np.mean(av) - np.mean(nv)) / denom
        _, p = welch_ttest(av, nv)
        triggered = not (rel >= kc.get("query_value_min_delta", 0.10)
                         and p <= kc.get("query_value_significance_p", 0.05))
        out["query_no_value"] = {
            "triggered": bool(triggered),
            "detail": {"rel_gain": float(rel), "p": float(p)},
            "action": "premise fails (querying has no value) -> pivot",
        }

    # 2) voq_not_learnable: held-out calibration error must trend down
    if calib_ordered is not None:
        improves = calibration_improves(calib_ordered, tol=kc.get("calib_no_improve_tol", 0.0))
        out["voq_not_learnable"] = {
            "triggered": bool(not improves),
            "detail": {"trend": calibration_trend(calib_ordered)},
            "action": "change phi features or use a stronger LLM prior",
        }

    # 3) no_cheap_simulator: any episode wall-clock over the limit
    limit = kc.get("episode_wallclock_limit_s", 600)
    longest = 0.0
    for rs in (never_results, always_results, transfer_results, scratch_results):
        if rs:
            longest = max(longest, max(r.wallclock_s for r in rs))
    out["no_cheap_simulator"] = {
        "triggered": bool(longest > limit),
        "detail": {"longest_episode_s": float(longest), "limit_s": float(limit)},
        "action": "downgrade that game to a small-scale demo (expected: Minecraft)",
    }

    # 4) instance_overfit: QVL transfer must beat from-scratch on held-out
    if transfer_results and scratch_results:
        tg = np.mean([r.U for r in transfer_results])
        sg = np.mean([r.U for r in scratch_results])
        gain = float(tg - sg)
        out["instance_overfit"] = {
            "triggered": bool(gain <= kc.get("transfer_min_gain", 0.0)),
            "detail": {"transfer_utility": float(tg), "scratch_utility": float(sg), "gain": gain},
            "action": "check phi for instance leakage (must be instance-invariant)",
        }
    return out


# ------------------------------------------------------------------ plotting
def plot_frontier(method_to_points: dict, outpath: str, title: str = "Return vs sim-steps") -> None:
    """method_to_points: {method_name: [(sim_steps, return), ...]}."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5))
    for name, pts in method_to_points.items():
        if not pts:
            continue
        pts = sorted(pts)
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ax.plot(xs, ys, marker="o", label=name)
    fr = pareto_frontier([p for pts in method_to_points.values() for p in pts])
    if fr:
        ax.plot([c for c, _ in fr], [v for _, v in fr], "k--", alpha=0.5, label="Pareto frontier")
    ax.set_xlabel("total simulator steps (C)")
    ax.set_ylabel("game return (G)")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outpath, dpi=140)
    plt.close(fig)


def plot_calibration(calib_ordered, outpath: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    trend = calibration_trend(calib_ordered)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(range(1, len(trend) + 1), trend, marker="o")
    ax.set_xlabel("experience bin (early -> late)")
    ax.set_ylabel("VoQ calibration MAE")
    ax.set_title("VoQ calibration error vs experience")
    fig.tight_layout()
    fig.savefig(outpath, dpi=140)
    plt.close(fig)


def plot_transfer(transfer_results, scratch_results, outpath: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def cum(results):
        u = [r.U for r in results]
        return np.cumsum(u) / (np.arange(len(u)) + 1)

    fig, ax = plt.subplots(figsize=(7, 4))
    if transfer_results:
        ax.plot(cum(transfer_results), marker="o", label="QVL transfer")
    if scratch_results:
        ax.plot(cum(scratch_results), marker="s", label="from-scratch")
    ax.set_xlabel("held-out instance index")
    ax.set_ylabel("running mean utility U")
    ax.set_title("Transfer: QVL vs from-scratch (held-out)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outpath, dpi=140)
    plt.close(fig)
