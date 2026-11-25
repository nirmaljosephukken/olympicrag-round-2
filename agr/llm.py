"""LLM wrapper (Gemini / Anthropic / OpenAI over plain REST) with token accounting, retry/backoff and rate limiting.

LLM_PROVIDER=gemini|anthropic|openai selects the provider; LLM_MODEL overrides its default model.

Rate limiting is done BEFORE a question starts (`wait_for_budget`), so a free-tier wait never lands inside a
question's latency: with LLM_RPM=15 the runner waits until enough request slots are free in the last 60 s for a whole
question (LLM_CALLS_RESERVED, default 6) before starting the clock. A guard inside `generate` still prevents a 429 if a
question uses more calls than reserved; such waits are counted in `WAITS` so they can be reported.
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
RPM = int(os.getenv("LLM_RPM", "15" if SETTINGS.llm_provider == "gemini" else "0"))  # 0 = no limit
_calls: collections.deque = collections.deque()   # start times of the calls in the last 60 s
WAITS = {"in_question_s": 0.0}

# One or more Gemini keys, rotated when a key's daily quota is spent: GEMINI_API_KEYS=key1,key2 and/or
# GEMINI_API_KEY, GEMINI_API_KEY2, GEMINI_API_KEY3 ... (duplicates removed, order kept).
def _gemini_keys() -> list[str]:
    found = [k for k in (os.getenv("GEMINI_API_KEYS") or "").split(",")] + [SETTINGS.gemini_key]
    found += [os.environ[n] for n in sorted((n for n in os.environ if re.fullmatch(r"GEMINI_API_KEY\d+", n)),
                                            key=lambda n: int(n[14:]))]
    return list(dict.fromkeys(k.strip() for k in found if k and k.strip()))


_KEYS = _gemini_keys()
_key_idx = [0]


def _slots_free(now: float) -> int:
    while _calls and now - _calls[0] > 60:
        _calls.popleft()
    return RPM - len(_calls)


def wait_for_budget(calls: int | None = None) -> float:
    """Block until `calls` requests can be made within the provider's per-minute limit. Called by the runner
    BEFORE a question's clock starts. Returns the seconds waited."""
    if not RPM:
        return 0.0
    calls = min(RPM, calls or int(os.getenv("LLM_CALLS_RESERVED", "6")))
    t0 = time.time()
    while True:
        with _lock:
            now = time.time()
            if _slots_free(now) >= calls:
                return time.time() - t0
            wait = 60 - (now - _calls[0]) + 0.05
        time.sleep(max(0.05, wait))


def _take_slot() -> None:
    """Guard inside a question: only waits if the question used more calls than were reserved for it."""
    if not RPM:
        return
    while True:
        with _lock:
            now = time.time()
            if _slots_free(now) > 0:
                _calls.append(now)
                return
            wait = 60 - (now - _calls[0]) + 0.05
        WAITS["in_question_s"] += wait
        time.sleep(max(0.05, wait))


def _is_daily_limit(body: str) -> bool:
    """A daily quota error asks to retry in hours; a per-minute one in seconds."""
    m = re.search(r'"retryDelay":\s*"(\d+)s"', body)
    if m:
        return int(m.group(1)) > 3600
    return bool(re.search(r"retry in \d+h", body))


def _rotate_key(failed: int | None = None) -> bool:
    with _lock:
        if failed is not None and _key_idx[0] != failed:
            return True
        if _key_idx[0] + 1 < len(_KEYS):
            _key_idx[0] += 1
            print(f"[llm] daily quota reached, switching to key #{_key_idx[0] + 1}", flush=True)
            return True
    return False


@dataclass
class LLMResult:
    text: str
    input_tokens: int
    output_tokens: int  # includes thinking tokens
    ms: float
    model: str

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
def _gemini(prompt, system, model, json_mode, temperature, max_out):
    body = {"contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": temperature, "maxOutputTokens": max_out}}
    if system:
        body["systemInstruction"] = {"parts": [{"text": system}]}
    if json_mode:
        body["generationConfig"]["responseMimeType"] = "application/json"
    used = _key_idx[0]
    r = requests.post(GEMINI_URL.format(model=model), params={"key": _KEYS[used]}, json=body, timeout=120)
    if r.status_code == 429 and _is_daily_limit(r.text):
        if _rotate_key(used):
            return "retry_now", r
        raise RuntimeError("Gemini daily quota exhausted on every configured key")
    if r.status_code != 200:
        return "retry", r
    data = r.json()
    um = data.get("usageMetadata", {})
    cand = (data.get("candidates") or [{}])[0]
    text = "".join(p.get("text", "") for p in cand.get("content", {}).get("parts", []) if not p.get("thought"))
    return "ok", (text, um.get("promptTokenCount", 0),
                  um.get("candidatesTokenCount", 0) + um.get("thoughtsTokenCount", 0))


def _anthropic(prompt, system, model, json_mode, temperature, max_out):
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


def _openai(prompt, system, model, json_mode, temperature, max_out):
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
    delay, last = 2.0, None
    for attempt in range(8):
        _take_slot()
        t0 = time.time()
        try:
            status, out = call(prompt, system, model, json_mode, temperature, max_output_tokens)
        except requests.RequestException as e:
            last = e
            time.sleep(delay); delay *= 2; continue
        if status == "ok":
            text, tin, tout = out
            return LLMResult(text=text, input_tokens=tin, output_tokens=tout, ms=(time.time() - t0) * 1000,
                             model=model)
        last = out
        if status == "retry_now":
            continue
        if out.status_code in (408, 429, 500, 502, 503, 504, 529):
            time.sleep(delay + random.random()); delay = min(delay * 2, 60); continue
        raise RuntimeError(f"{SETTINGS.llm_provider} HTTP {out.status_code}: {out.text[:300]}")
    detail = f"{last.status_code} {last.text[:300]}" if isinstance(last, requests.Response) else str(last)
    raise RuntimeError(f"{SETTINGS.llm_provider} call failed after retries: {detail}")


def approx_tokens(text: str) -> int:
    return max(1, len(text) // 4)
