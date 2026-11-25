"""LLM wrapper (Gemini / Anthropic / OpenAI over plain REST) with token accounting, retry/backoff and rate limiting.

LLM_PROVIDER=gemini|anthropic|openai selects the provider; LLM_MODEL overrides its default model.

Several Gemini keys form a pool (`KeyPool`). Every call goes to the fastest key that has a free request slot:
each key's response time is tracked (moving average; untried keys are tried first), each key has its own per-minute
limit (LLM_RPM, default 15), a key that answers "rate limited" rests for 60 s while the call moves to the next key at
once (no back-off), and a key whose daily quota is spent is dropped for the run.

Rate limiting is done BEFORE a question starts (`wait_for_budget`), so a free-tier wait never lands inside a
question's latency: the runner waits until enough request slots are free across the keys for a whole question
(LLM_CALLS_RESERVED, default 6) before starting the clock. If a question still needs more, the wait inside it is
counted in `WAITS` so it can be reported.
"""
from __future__ import annotations

import collections
import json
import os
import random
import re
import threading
import time
from dataclasses import dataclass

import requests

from agr.config import SETTINGS

GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
OPENAI_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/") + "/chat/completions"

_lock = threading.Lock()
RPM = int(os.getenv("LLM_RPM", "15" if SETTINGS.llm_provider == "gemini" else "0"))  # per key; 0 = no limit
WAITS = {"in_question_s": 0.0}


# One or more Gemini keys: GEMINI_API_KEYS=key1,key2 and/or GEMINI_API_KEY, GEMINI_API_KEY2, GEMINI_API_KEY3 ...
# (duplicates removed, order kept).
def _gemini_keys() -> list[str]:
    found = [k for k in (os.getenv("GEMINI_API_KEYS") or "").split(",")] + [SETTINGS.gemini_key]
    found += [os.environ[n] for n in sorted((n for n in os.environ if re.fullmatch(r"GEMINI_API_KEY\d+", n)),
                                            key=lambda n: int(n[14:]))]
    return list(dict.fromkeys(k.strip() for k in found if k and k.strip()))


_KEYS = _gemini_keys()


class KeyPool:
    """Chooses the key for each call: the fastest usable key with a free per-minute slot."""

    def __init__(self, n: int, rpm: int, clock=time.time, sleep=time.sleep):
        self.n, self.rpm, self.clock, self.sleep = max(1, n), rpm, clock, sleep
        self.calls = [collections.deque() for _ in range(self.n)]
        self.cool_until = [0.0] * self.n
        self.ewma: list[float | None] = [None] * self.n
        self.exhausted: set[int] = set()
        self.tried: set[int] = set()

    def _free(self, i: int, now: float) -> int:
        if not self.rpm:
            return 1 << 30
        q = self.calls[i]
        while q and now - q[0] > 60:
            q.popleft()
        return self.rpm - len(q)

    def usable(self) -> list[int]:
        return [i for i in range(self.n) if i not in self.exhausted]

    def free_total(self) -> int:
        now = self.clock()
        return sum(max(0, self._free(i, now)) for i in self.usable() if self.cool_until[i] <= now)

    def _next_free_in(self, now: float) -> float:
        waits = []
        for i in self.usable():
            w = max(0.0, self.cool_until[i] - now)
            if self.rpm and self._free(i, now) <= 0:
                w = max(w, 60 - (now - self.calls[i][0]))
            waits.append(w)
        return max(0.05, min(waits) + 0.05)

    def acquire(self, count_wait: bool = True) -> int:
        while True:
            with _lock:
                if not self.usable():
                    raise RuntimeError("Gemini daily quota exhausted on every configured key")
                now = self.clock()
                ready = [i for i in self.usable() if self.cool_until[i] <= now and self._free(i, now) > 0]
                if ready:   # untried keys first, then the fastest so far
                    i = min(ready, key=lambda k: (k in self.tried, self.ewma[k] or 0.0, k))
                    self.tried.add(i)
                    self.calls[i].append(now)
                    return i
                wait = self._next_free_in(now)
            if count_wait:
                WAITS["in_question_s"] += wait
            self.sleep(wait)

    def report(self, i: int, ms: float) -> None:
        with _lock:
            self.ewma[i] = ms if self.ewma[i] is None else 0.7 * self.ewma[i] + 0.3 * ms

    def rate_limited(self, i: int) -> None:
        with _lock:
            self.cool_until[i] = self.clock() + 60

    def spent(self, i: int) -> None:
        with _lock:
            if i not in self.exhausted:
                self.exhausted.add(i)
                print(f"[llm] daily quota reached on key #{i + 1}; {len(self.usable())} key(s) left", flush=True)


POOL = KeyPool(len(_KEYS) if SETTINGS.llm_provider == "gemini" else 1, RPM)


def wait_for_budget(calls: int | None = None) -> float:
    """Block until `calls` requests can be made within the per-minute limits of the usable keys. Called by the
    runner BEFORE a question's clock starts. Returns the seconds waited."""
    if not RPM:
        return 0.0
    t0 = time.time()
    while True:
        need = min(RPM * max(1, len(POOL.usable())), calls or int(os.getenv("LLM_CALLS_RESERVED", "6")))
        if POOL.free_total() >= need:
            return time.time() - t0
        time.sleep(1.0)


def _is_daily_limit(body: str) -> bool:
    """A daily quota error asks to retry in hours; a per-minute one in seconds."""
    m = re.search(r'"retryDelay":\s*"(\d+)s"', body)
    if m:
        return int(m.group(1)) > 3600
    return bool(re.search(r"retry in \d+h", body))


@dataclass
class LLMResult:
    text: str
    input_tokens: int
    output_tokens: int  # includes thinking tokens
    ms: float
    model: str
    key: int = 0          # which configured key served the call (1-based index, never the key itself)
    attempts: int = 1     # requests made, including retries
    total_ms: float = 0.0 # wall time including retries and back-off

    def json(self) -> dict:
        d = self._json()
        if isinstance(d, list):  # some replies wrap the object in a list
            d = next((x for x in d if isinstance(x, dict)), {})
        return d if isinstance(d, dict) else {}

    def _json(self):
        t = self.text.strip()
        t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t)
        try:
            return json.loads(t)
        except json.JSONDecodeError:
            m = re.search(r"\{.*\}", t, re.S)
            if m:
                try:
                    return json.loads(m.group(0))
                except json.JSONDecodeError:
                    pass
        return {}


# ----------------------------------------------------------------------------- providers
def _gemini(prompt, system, model, json_mode, temperature, max_out, key=0):
    body = {"contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": temperature, "maxOutputTokens": max_out}}
    if system:
        body["systemInstruction"] = {"parts": [{"text": system}]}
    if json_mode:
        body["generationConfig"]["responseMimeType"] = "application/json"
    r = requests.post(GEMINI_URL.format(model=model), params={"key": _KEYS[key]}, json=body, timeout=int(os.getenv("LLM_TIMEOUT_S", "45")))
    if r.status_code == 429:
        return ("daily" if _is_daily_limit(r.text) else "rate_limited"), r
    if r.status_code != 200:
        return "retry", r
    data = r.json()
    um = data.get("usageMetadata", {})
    cand = (data.get("candidates") or [{}])[0]
    text = "".join(p.get("text", "") for p in cand.get("content", {}).get("parts", []) if not p.get("thought"))
    return "ok", (text, um.get("promptTokenCount", 0),
                  um.get("candidatesTokenCount", 0) + um.get("thoughtsTokenCount", 0))


def _anthropic(prompt, system, model, json_mode, temperature, max_out, key=0):
    if json_mode:
        prompt += "\n\nReply with a single JSON object only."
    body = {"model": model, "max_tokens": max_out, "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}]}
    if system:
        body["system"] = system
    r = requests.post(ANTHROPIC_URL, json=body, timeout=120,
                      headers={"x-api-key": SETTINGS.anthropic_key, "anthropic-version": "2023-06-01"})
    if r.status_code != 200:
        return "retry", r
    data = r.json()
    text = "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")
    u = data.get("usage", {})
    return "ok", (text, u.get("input_tokens", 0) + u.get("cache_read_input_tokens", 0)
                  + u.get("cache_creation_input_tokens", 0), u.get("output_tokens", 0))


def _openai(prompt, system, model, json_mode, temperature, max_out, key=0):
    msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
    body = {"model": model, "messages": msgs, "max_completion_tokens": max_out}
    if json_mode:
        body["response_format"] = {"type": "json_object"}
    r = requests.post(OPENAI_URL, json=body, timeout=120, headers={"Authorization": f"Bearer {SETTINGS.openai_key}"})
    if r.status_code != 200:
        return "retry", r
    data = r.json()
    u = data.get("usage", {})
    text = (data.get("choices") or [{}])[0].get("message", {}).get("content") or ""
    return "ok", (text, u.get("prompt_tokens", 0), u.get("completion_tokens", 0))


PROVIDERS = {"gemini": _gemini, "anthropic": _anthropic, "openai": _openai}


def generate(prompt: str, system: str | None = None, model: str | None = None, json_mode: bool = False,
             temperature: float = 0.0, max_output_tokens: int = 1024) -> LLMResult:
    model = model or SETTINGS.model
    call = PROVIDERS[SETTINGS.llm_provider]
    delay, last, t_start = 2.0, None, time.time()
    for attempt in range(10):
        key = POOL.acquire()
        t0 = time.time()
        try:
            status, out = call(prompt, system, model, json_mode, temperature, max_output_tokens, key)
        except requests.RequestException as e:
            last = e
            time.sleep(delay); delay *= 2; continue
        if status == "ok":
            ms = (time.time() - t0) * 1000
            POOL.report(key, ms)
            text, tin, tout = out
            return LLMResult(text=text, input_tokens=tin, output_tokens=tout, ms=ms, model=model, key=key + 1,
                             attempts=attempt + 1, total_ms=(time.time() - t_start) * 1000)
        last = out
        if status == "daily":          # this key is spent for the day: the next call goes to another key
            POOL.spent(key)
            continue
        if status == "rate_limited":   # this key rests for a minute; retry at once on another key
            POOL.rate_limited(key)
            continue
        if out.status_code in (408, 429, 500, 502, 503, 504, 529):
            time.sleep(delay + random.random()); delay = min(delay * 2, 60); continue
        raise RuntimeError(f"{SETTINGS.llm_provider} HTTP {out.status_code}: {out.text[:300]}")
    detail = f"{last.status_code} {last.text[:300]}" if isinstance(last, requests.Response) else str(last)
    raise RuntimeError(f"{SETTINGS.llm_provider} call failed after retries: {detail}")


def approx_tokens(text: str) -> int:
    return max(1, len(text) // 4)
