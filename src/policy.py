"""Task 2 — frozen base-policy prior (top-m candidates + margin + entropy).

The policy is FROZEN: inference only, zero updates (METHOD_DESIGN desideratum 1). It must
expose, for a state, the top-m preferred legal actions together with *preference scores*
from which the gate features margin / entropy are computed (Task 3).

Backends
--------
* ``heuristic`` : preference scores come from a cheap, game-specific, deliberately myopic
  scorer (``env.heuristic_scores``). Lets the ENTIRE SimTriage pipeline run with no GPU/LLM,
  so M1/M2/M3 (the actual contribution) can be validated cheaply. This is the default for
  bring-up. The myopic prior leaves room for the real simulator to correct it — exactly the
  regime the method targets.
* ``llm``       : the paper configuration. A frozen instruct model scores the candidates.
* ``random``    : uniform prior (margin ~ 0 everywhere) — a sanity/ablation baseline.

Two extra hooks the method needs:
* ``imagine_q`` : the prior's *imagined* value of an action WITHOUT touching the simulator.
  Used ONLY by the -M2 ablation (replace real rollout with imagination) to quantify the
  white-box bonus (METHOD_DESIGN §3 Occam check). The main method never calls it.
* :class:`RolloutPolicy` : the cheap default policy that drives query rollouts (random /
  weighted-random / value / prior). Game-specific action choices live on the env.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Optional, Sequence

import numpy as np


class Policy(ABC):
    name = "base"

    @abstractmethod
    def top_candidates(self, env, state, m: int) -> tuple[list, list[float]]:
        """Return (A, prefs): up to m legal candidate actions and their preference scores
        (higher = more preferred). A[0] is the prior's favorite a_0. ``prefs`` are on a
        logit-like scale so softmax(prefs) is a meaningful prior distribution."""

    def imagine_q(self, env, state, action) -> float:
        """Prior's imagined value of ``action`` WITHOUT the simulator (-M2 ablation only)."""
        raise NotImplementedError


# --------------------------------------------------------------------------- heuristic
class HeuristicPolicy(Policy):
    name = "heuristic"

    def __init__(self, cfg: Optional[dict] = None):
        self.cfg = cfg or {}

    def top_candidates(self, env, state, m: int):
        actions = list(env.valid_actions(state))
        if not actions:
            return [], []
        scores = np.asarray(env.heuristic_scores(state, actions), dtype=float)
        order = np.argsort(scores)[::-1]
        topk = order[: max(1, m)]
        A = [actions[i] for i in topk]
        prefs = [float(scores[i]) for i in topk]
        return A, prefs

    def imagine_q(self, env, state, action) -> float:
        # The prior's internal one-shot guess (no multi-step real rollout): the same myopic
        # scorer it uses to rank. This is "imagination", not a simulator query.
        return float(env.heuristic_value(state, action))


# --------------------------------------------------------------------------- random
class RandomPolicy(Policy):
    name = "random"

    def __init__(self, cfg: Optional[dict] = None, rng: Optional[np.random.Generator] = None):
        self.cfg = cfg or {}
        self.rng = rng or np.random.default_rng(0)

    def top_candidates(self, env, state, m: int):
        actions = list(env.valid_actions(state))
        if not actions:
            return [], []
        idx = self.rng.permutation(len(actions))[: max(1, m)]
        A = [actions[i] for i in idx]
        prefs = [0.0] * len(A)  # uniform => margin 0, max entropy
        return A, prefs

    def imagine_q(self, env, state, action) -> float:
        return 0.0


# --------------------------------------------------------------------------- LLM
class LLMPolicy(Policy):
    """Frozen instruct-model prior. NO weight updates ever happen here.

    Multi-provider: ``policy.providers`` is a dict of named profiles and ``policy.provider``
    selects one — so DeepSeek / Qwen(DashScope) / OpenAI (all OpenAI-compatible) and Anthropic
    can be configured side-by-side and switched per run (``--provider`` on the scripts).

    Built for batch experiments (thousands of calls): exponential-backoff retry, request
    timeout, and an optional disk cache keyed on (provider, model, temperature, prompt).
    Robust JSON parsing with a uniform fallback keeps the loop alive on any API hiccup.
    """

    name = "llm"

    def __init__(self, cfg: dict):
        pcfg = cfg.get("policy", cfg)
        self.cfg = pcfg
        self.temperature = float(pcfg.get("temperature", 0.7))
        self.max_retries = int(pcfg.get("max_retries", 5))
        self.timeout_s = float(pcfg.get("timeout_s", 60))
        self.max_tokens = int(pcfg.get("max_tokens", 512))

        # Resolve the provider profile (new style); fall back to flat fields (back-compat).
        providers = pcfg.get("providers") or {}
        prov = pcfg.get("provider")
        if prov and prov in providers:
            prof = providers[prov]
        else:
            prof = {
                "api": pcfg.get("api", "openai"),
                "model": pcfg.get("model", "gpt-4o-mini"),
                "base_url": pcfg.get("base_url"),
                "api_key_env": pcfg.get("api_key_env", "OPENAI_API_KEY"),
            }
            prov = prov or "default"
        self.provider = prov
        self.api = prof.get("api", "openai")
        self.model = prof.get("model")
        self.base_url = prof.get("base_url")
        self.api_key_env = prof.get("api_key_env", "OPENAI_API_KEY")
        self._client = None  # lazy

        # optional disk cache (frozen model => identical prompt can be reused; saves $ and is
        # reproducible). Off by default so it doesn't silently freeze temperature sampling.
        self.cache_enabled = bool(pcfg.get("cache", False))
        self.cache_dir = pcfg.get("cache_dir", "outputs/llm_cache")
        self._mem_cache: dict = {}

    def _client_obj(self):
        if self._client is not None:
            return self._client
        if self.api in ("openai", "vllm"):
            from openai import OpenAI

            self._client = OpenAI(
                base_url=self.base_url or None,
                api_key=os.environ.get(self.api_key_env, "EMPTY"),
                timeout=self.timeout_s,
            )
        elif self.api == "anthropic":
            from anthropic import Anthropic

            self._client = Anthropic(
                api_key=os.environ.get(self.api_key_env, ""), timeout=self.timeout_s
            )
        else:
            raise ValueError(f"unknown api {self.api!r}")
        return self._client

    def _chat(self, system: str, user: str) -> str:
        client = self._client_obj()
        if self.api in ("openai", "vllm"):
            r = client.chat.completions.create(
                model=self.model,
                temperature=self.temperature,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            )
            return r.choices[0].message.content or ""
        else:  # anthropic
            r = client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
            return "".join(b.text for b in r.content if getattr(b, "type", "") == "text")

    # ---- robustness: cache + retry/backoff ----------------------------------
    def _cache_key(self, system: str, user: str) -> str:
        import hashlib

        raw = f"{self.provider}|{self.model}|{self.temperature}|{system}|{user}"
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()

    def _cache_get(self, key: str):
        if key in self._mem_cache:
            return self._mem_cache[key]
        path = os.path.join(self.cache_dir, key + ".txt")
        if os.path.exists(path):
            try:
                with open(path, encoding="utf-8") as f:
                    val = f.read()
                self._mem_cache[key] = val
                return val
            except Exception:
                return None
        return None

    def _cache_put(self, key: str, val: str) -> None:
        self._mem_cache[key] = val
        try:
            os.makedirs(self.cache_dir, exist_ok=True)
            with open(os.path.join(self.cache_dir, key + ".txt"), "w", encoding="utf-8") as f:
                f.write(val)
        except Exception:
            pass

    def _sleep_backoff(self, attempt: int) -> None:
        import time

        time.sleep(min(2.0 ** attempt, 30.0))

    @staticmethod
    def _is_fatal(e: Exception) -> bool:
        """Non-retryable: misconfiguration (missing package, bad key, bad request). Retrying
        these just wastes time, and silently degrading would corrupt the experiment."""
        if isinstance(e, (ImportError, ModuleNotFoundError)):
            return True
        name = type(e).__name__.lower()
        if any(k in name for k in ("authentication", "permission", "notfound", "badrequest",
                                   "invalidrequest", "unprocessable")):
            return True
        status = getattr(e, "status_code", None) or getattr(e, "status", None)
        return status in (400, 401, 403, 404, 422)

    def _safe_chat(self, system: str, user: str) -> str:
        """Return the model text. Cache hit short-circuits. Transient errors (rate limit /
        timeout / connection) are retried with exponential backoff; fatal misconfig errors
        fail FAST; persistent failure RAISES (never silently returns a degenerate prior that
        would invalidate the run). A garbage *response* is handled downstream by _parse_scores."""
        key = self._cache_key(system, user) if self.cache_enabled else None
        if key is not None:
            hit = self._cache_get(key)
            if hit is not None:
                return hit
        last: Exception | None = None
        for attempt in range(max(1, self.max_retries)):
            try:
                out = self._chat(system, user)
                if key is not None:
                    self._cache_put(key, out)
                return out
            except Exception as e:
                if self._is_fatal(e):
                    raise RuntimeError(
                        f"LLM call failed (non-retryable) for provider={self.provider!r} "
                        f"model={self.model!r}: {e}. Check the API key ({self.api_key_env}), "
                        f"base_url ({self.base_url}), and that the client package is installed."
                    ) from e
                last = e
                if attempt < self.max_retries - 1:
                    self._sleep_backoff(attempt)
        raise RuntimeError(
            f"LLM call failed after {self.max_retries} retries for provider={self.provider!r} "
            f"model={self.model!r}: {last}"
        )

    def top_candidates(self, env, state, m: int):
        actions = list(env.valid_actions(state))
        if not actions:
            return [], []
        # Cap candidate set for prompting cost; rank the white-box legal actions.
        cand = actions[:50]
        rendered = "\n".join(f"[{i}] {env.render_action(a)}" for i, a in enumerate(cand))
        sys = (
            "You are an expert game player. Rate each candidate action by long-run value. "
            "Reply ONLY with a JSON object mapping the action index (as a string) to a float "
            "score in [0,10]; higher is better. No prose."
        )
        usr = f"State:\n{env.render_text(state)}\n\nCandidate actions:\n{rendered}\n\nJSON scores:"
        scores = self._parse_scores(self._safe_chat(sys, usr), len(cand))
        order = np.argsort(scores)[::-1][: max(1, m)]
        A = [cand[i] for i in order]
        prefs = [float(scores[i]) for i in order]
        return A, prefs

    def imagine_q(self, env, state, action) -> float:
        sys = "Estimate the long-run value of taking the given action. Reply ONLY a single float."
        usr = f"State:\n{env.render_text(state)}\n\nAction: {env.render_action(action)}\n\nValue:"
        txt = self._safe_chat(sys, usr).strip()
        try:
            return float(txt.split()[0])
        except Exception:
            return 0.0

    @staticmethod
    def _parse_scores(text: str, n: int) -> np.ndarray:
        import json
        import re

        scores = np.zeros(n, dtype=float)
        try:
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                d = json.loads(match.group(0))
                for k, v in d.items():
                    i = int(k)
                    if 0 <= i < n:
                        scores[i] = float(v)
                return scores
        except Exception:
            pass
        # fallback: uniform (max entropy, margin 0) so the gate treats it as uncertain
        return np.zeros(n, dtype=float)


# --------------------------------------------------------------------------- rollout policy
class RolloutPolicy:
    """Cheap default policy that drives query rollouts (METHOD_DESIGN §4.3). Game-specific
    action selection is delegated to the env so this stays game-agnostic."""

    def __init__(self, kind: str = "weighted_random", prior: Optional[Policy] = None):
        self.kind = kind
        self.prior = prior

    def __call__(self, env, state, actions):
        return self.action(env, state, actions)

    def action(self, env, state, actions):
        if not actions:
            return None
        if self.kind == "random":
            return env.random_action(state, actions)
        if self.kind == "weighted_random":
            return env.weighted_random_action(state, actions)
        if self.kind == "value":
            return env.greedy_action(state, actions)
        if self.kind == "prior" and self.prior is not None:
            A, _ = self.prior.top_candidates(env, state, 1)
            return A[0] if A else env.random_action(state, actions)
        return env.random_action(state, actions)


def make_policy(cfg: dict, rng: Optional[np.random.Generator] = None) -> Policy:
    backend = cfg.get("policy", {}).get("backend", "heuristic")
    if backend == "heuristic":
        return HeuristicPolicy(cfg)
    if backend == "random":
        return RandomPolicy(cfg, rng=rng)
    if backend == "llm":
        return LLMPolicy(cfg)
    raise ValueError(f"unknown policy backend {backend!r}")


def make_rollout_policy(cfg: dict, prior: Optional[Policy] = None) -> RolloutPolicy:
    kind = cfg.get("policy", {}).get("rollout_policy", "weighted_random")
    return RolloutPolicy(kind=kind, prior=prior)
