# SimTriage — Learning *when* to query the exact game simulator

A faithful implementation of the method in `METHOD_DESIGN_SimTriage.md` /
`IMPLEMENTATION_SPEC_SimTriage.md`.

**One sentence:** a *frozen* LLM/heuristic agent learns a cheap, training-free, transferable
rule for *when it is worth spending one real-simulator rollout to verify its prior* instead
of either always guessing (Voyager-style) or always searching. The only thing that evolves is
`theta` = a VoQ predictor (M1) + a prototype memory library (QVL, M3). The model is never
fine-tuned; **zero gradient updates**.

```
gate  : query iff  ĝ_θ(φ(s)) > λ·c(s)              # M1 predicts value-of-query; myopic VoC gate
query : top-m candidates × K depth-d REAL rollouts  # M2: clone+forward the actual simulator
label : with prob ρ, a high-budget reference rollout gives y(s)=refQ(â_ref)−refQ(a_0)  # self-supervised
evolve: θ ← online ridge/kNN update(φ,y); consolidate into QVL prototypes (Hoeffding)   # M3 transfer
```

## Layout (maps 1:1 to the spec's Tasks)

| File | Spec Task | Role |
|---|---|---|
| `src/envs/base_env.py` | 1 | unified `reset/step/valid_actions/clone/rollout/episode_return/query_cost` |
| `src/envs/catan_env.py` | 1 | **Catanatron adapter — the primary, sim-native env (do this first)** |
| `src/envs/mock_env.py` | 1 | dependency-free synthetic game (runs the whole pipeline with no GPU/catanatron) |
| `src/envs/sts_env.py`, `minecraft_env.py` | 1 | scaffolds; `supports_cheap_clone=False` (R1) |
| `src/policy.py` | 2 | frozen prior: `top_candidates → (A, prefs)`; heuristic / LLM / random backends |
| `src/features.py` | 3 | **instance-invariant** φ(s): margin, entropy, log\|valid\|, phase, irrev, … |
| `src/voq.py` | 4 | VoQ predictor ĝ_θ + `online_update` (incremental ridge / kNN) — M1 |
| `src/query.py` | 5 | gate + real-rollout query operator + reference label (CRN) — M2 |
| `src/qvl.py` | 6 | transferable prototype library + Hoeffding retain/eject + cold-start prior — M3 |
| `src/simtriage.py` | 7 | Algorithm 1 (decide) + Algorithm 2 (self-evolution loop) |
| `src/baselines.py` | 8 | never/always/fixed-threshold/random-gate + −M2/−M3/φ-instance-dependent ablations |
| `src/metrics.py` | 9 | Pareto frontier, transfer, calibration, **automated kill-criteria** |
| `scripts/verify_query_determinism.py` | 1 | R2: clone independence + CRN reproducibility |
| `scripts/run_experiment.py` | 9 | end-to-end runner → `outputs/<game>/{results.json,frontier.png,…}` |

## Quickstart

```bash
pip install -r requirements.txt          # numpy/scipy/sklearn/pyyaml/matplotlib (+ catanatron)

# 1) sanity: run the entire method with NO external deps (synthetic game)
python scripts/verify_query_determinism.py --env mock
python scripts/run_experiment.py --game mock --quick
pytest -q                                # mock/unit tests run; catan tests skip if not installed

# 2) primary experiment on Catan (sim-native; cleanest clone+rollout)
python scripts/verify_query_determinism.py --env catan --seed 0
python scripts/run_experiment.py --game catan
```

### Using a real frozen LLM as the prior (API, no GPU needed)
The prior is the only swappable part; the gate/VoQ/QVL contribution is policy-agnostic. The
default `heuristic` backend exists so M1/M2/M3 can be validated cheaply without any LLM.

**Cost structure (why API is cheap here):** rollouts and reference queries run on the LOCAL
catanatron simulator, *not* the LLM — so the only paid calls are the per-decision prior
(`top_candidates`) and the `-M2` ablation's `imagine_q`. API volume ≈ number of decision
points, independent of the rollout budget (m/K/depth).

**Multiple providers, switchable per run.** `configs/sim.yaml` `policy.providers` holds named
profiles (all OpenAI-compatible except Anthropic); pick one with `policy.provider` or the
scripts' `--provider`:

```bash
export DEEPSEEK_API_KEY=...   DASHSCOPE_API_KEY=...   OPENAI_API_KEY=...
pip install openai                                   # (or `anthropic` for Claude)

python scripts/run_experiment.py --game catan --provider deepseek
python scripts/run_experiment.py --game catan --provider qwen      # Qwen2.5 via DashScope (spec model)
python scripts/run_experiment.py --game catan --provider openai
```

Built for batch runs: exponential-backoff retry on rate-limit/transient errors, request
timeout, and an optional disk cache (`policy.cache: true`). Misconfiguration (missing package,
bad key, 4xx) **fails fast with a clear message**; persistent failure raises rather than
silently degrading to uniform priors. Frozen, zero-update throughout.

## Key faithfulness guarantees (and where they live)
- **Frozen model, zero training.** Only `voq.py` (regressor) and `qvl.py` (library) change.
- **Query = REAL simulator** (`env.clone` + `env.rollout`), never imagination. The imagination
  variant is isolated as the `−M2` ablation (`baselines.make_minus_m2`).
- **Gate is exactly** `ĝ_θ(φ) > λ·c(s)` (`query.gate`, used in `simtriage.decide`).
- **φ is instance-invariant** (`features.py`): dispersion + white-box structure only; no card
  names / coordinates. `test_features.py` asserts this; the leaky version is the
  `phi-instance-dependent` ablation.
- **CRN labels** in seedable envs (same seed for both branches) — `query.reference_voq` +
  `env._set_crn`; validated by `verify_query_determinism.py`.

## `# CONFIRM-ON-ENV` markers
Because there is no local Catan env yet, all version-sensitive Catanatron accessors are
funnelled through `_*` helpers in `catan_env.py` (search `CONFIRM-ON-ENV`). On the real
environment, confirm: `Game.copy()`, `state.playable_actions`, `state.current_player_index`,
`state.num_turns` (used as the turn-end signal for `to_turn_end` rollouts — see below),
`state.colors`, `winning_color()`, `state_functions.get_actual_victory_points`, and the
`Game(seed=...)` / RNG handle used for CRN. Everything else is engine-agnostic.

## Calibration notes (from an internal adversarial audit)
- **λ is on the simulator-step scale.** The gate is `g_hat > λ·c(s)` with `c(s) = m·K·E[len]`
  (tens of steps) and `g_hat` in value units (~[0,2]), so the *active* λ range is ~[0, 0.02].
  `lam_sweep` in `sim.yaml` is calibrated to this; **re-tune per game/backend**. `E[len]` is a
  self-calibrating running EMA of observed rollout length, so the cost tracks reality.
- **`to_turn_end` uses `num_turns`, not `current_player_index`.** On a 7-roll, Catanatron moves
  `current_player_index` to each discarder mid-turn; `num_turns` is the turn-monotonic signal.
- **−M2 (imagine) is backend-specific.** LLM backend = the model's own value imagination (the
  true paper ablation). Heuristic backend = a cheap 1-ply lookahead. Mock = a separate biased
  value array. All three are distinct from the method's multi-step **real** rollout, so −M2 is a
  genuine ablation (not trivially equal to never-query).
- **The transfer (M3) test runs in the discriminative λ regime** (query-rate ≈ 0.5); where
  queries are nearly free the gate queries everything and a transferred library can't help.
- The QVL stores the **price-weighted** cost `λ·c` so its Hoeffding retain/eject is commensurate
  with the VoQ labels.

## Kill-criteria (auto-checked, SPEC §4 / METHOD §7)
`metrics.evaluate_kill_criteria` flags: query-has-no-value (always≈never), VoQ-not-learnable
(held-out calibration doesn't fall), no-cheap-simulator (episode wall-clock over limit), and
instance-overfit (QVL transfer ≤ from-scratch ⇒ φ is leaking). `run_experiment.py` prints them.
```
