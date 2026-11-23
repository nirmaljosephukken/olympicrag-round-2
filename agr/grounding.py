"""
Grounding check (Round 2): is every name and number in an answer stated in the evidence it cites?

The check is done in code, not by the LLM, before the agent may stop. Round 2 answers often combine several documents
(a career across Games, a list of winners, a decision and the provisional record it replaced), so an answer may rest on
several cited evidence items, but nothing in it may be invented:

  1. a bare number answer must be the number stated in the right field of a cited item (a count from a count field or an
     aggregation, never a number that only appears inside a date);
  2. every number in a longer answer must appear in some cited item;
  3. every name (a run of capitalised words, e.g. a person, nation, venue or event) must appear in some cited item, or in
     the question itself (restating what was asked is not a new claim).

Words that open sentences or are generic ("The", "Gold", "Olympic", month names ...) are not names. Matching is
accent-, case- and punctuation-insensitive, and team names stored run together ("Loredana DinuSimona Gherman") match.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from agr.parse import norm


@dataclass
class Grounding:
    ok: bool
    reason: str
    supporting: list[str] = field(default_factory=list)   # evidence ids that state part of the answer
    events: list[str] = field(default_factory=list)       # distinct documents the answer draws on
    problem: str = ""                                      # unknown | uncited | unsupported


def _squash(s: str) -> str:
    return norm(s).replace(" ", "")


def answer_items(answer: str) -> list[str]:
    """Split an answer into the claims it makes: 'A, B and C' -> [A, B, C]; strips '(...)' and 'Label: ' parts."""
    parts = []
    for seg in re.split(r";|\n", answer):
        seg = re.sub(r"\([^)]*\)", "", seg)
        if ": " in seg:
            seg = seg.split(": ")[-1]
        parts += [x.strip(" .") for x in re.split(r",|\band\b|&|/", seg) if x.strip(" .")]
    return parts or [answer.strip()]


def _contains(text: str, item: str) -> bool:
    t, i = norm(text), norm(item)
    if not i:
        return False
    if re.search(r"(?<![a-z0-9])" + re.escape(i) + r"(?![a-z0-9])", t):
        return True
    return len(_squash(item)) >= 6 and _squash(item) in _squash(text)  # team names stored run together


GENERIC = set("""a an the it its no yes not none neither nor this that these those he she they his her their there here
both only each all any some every either one two three four five six seven eight nine ten gold silver bronze medal medals
medallist medalist men women mixed men's women's olympic olympics games summer winter final finals semifinal heat however
according because after before during while although though on in at of for from by with and or but as if when where which
who whom why how what than so then also official officially provisional unverified confirmed status answer note notes utc
pdt pst cet est gmt corpus source sources record records event events team teams it's he's she's we i you our we're
january february march april may june july august september october november december monday tuesday wednesday thursday
friday saturday sunday mr mrs ms dr jr sr st""".split())
NAME_RE = re.compile(r"[A-ZÀ-ÖØ-Þ][\w'’.\-]*(?:\s+(?:(?:de|da|del|der|di|du|van|von|la|le|bin|al|y|of|the)\s+)?"
                     r"[A-ZÀ-ÖØ-Þ][\w'’.\-]*)*")
NUM_RE = re.compile(r"(?<![\w.])\d[\d,]*(?:\.\d+)?")


def names_in(answer: str) -> list[str]:
    """Runs of capitalised words, minus generic words (a run made only of generic words is dropped)."""
    out = []
    for m in NAME_RE.finditer(answer or ""):
        words = [w for w in m.group(0).split() if norm(w) and norm(w) not in GENERIC]
        if words and any(re.match(r"[A-ZÀ-ÖØ-Þ]", w) for w in words):
            out.append(" ".join(words))
    return out


def numbers_in(answer: str) -> list[str]:
    return [n.rstrip(".,").replace(",", "") for n in NUM_RE.findall(answer or "")]


def _has_number(text: str, n: str) -> bool:
    flat = re.sub(r"(?<=\d),(?=\d{3})", "", text)
    return re.search(r"(?<![\d.])" + re.escape(n) + r"(?![\d])(?!\.\d)", flat) is not None


def check(answer: str, cited: list, question: str = "") -> Grounding:
    """`cited` is a list of objects/dicts with id, doc_id, title, text, kind."""
    get = lambda e, k: (e.get(k) if isinstance(e, dict) else getattr(e, k, "")) or ""
    a = (answer or "").strip()
    if norm(a) in ("", "unknown", "none", "n a", "not in corpus", "not in the corpus"):
        return Grounding(False, "answer is unknown", problem="unknown")
    if not cited:
        return Grounding(False, "no evidence cited", problem="uncited")
    texts = {get(e, "id"): get(e, "title") + " | " + get(e, "text") for e in cited}
    docs = {get(e, "id"): get(e, "doc_id") for e in cited}
    kinds = {get(e, "id"): get(e, "kind") for e in cited}
    alldocs = lambda ids: sorted({d for i in ids for d in str(docs[i]).split(",") if d})

    # 1. a bare number must be the number stated in the right field of a cited item
    if re.fullmatch(r"\d[\d,]*", a):
        n = a.replace(",", "")
        count_field = r"(?:competitors|nations): " + n + r"\b"   # structured event facts: only a count field counts
        sup = [i for i, t in texts.items()
               if (re.search(count_field, t) if kinds[i] == "graph" else _has_number(t, n))]
        return (Grounding(True, "number stated in cited evidence", sup, alldocs(sup)) if sup else
                Grounding(False, f"the number {a} is not stated in the cited evidence", problem="unsupported"))

    # 2-3. every number and every name must be stated somewhere in the cited evidence (or in the question)
    missing, used = [], set()
    for n in numbers_in(a):
        sup = [i for i, t in texts.items() if _has_number(t, n)]
        if sup:
            used.update(sup)
        elif not _has_number(question, n):
            missing.append(n)
    for name in names_in(a):
        sup = [i for i, t in texts.items() if _contains(t, name)]
        if not sup:  # a run of names ("Kim Woo-jin Brady Ellison"): accept when every word is stated
            words = [w for w in name.split() if norm(w) not in GENERIC and len(norm(w)) > 1]
            sup = [i for i, t in texts.items() if words and all(_contains(t, w) for w in words)]
        if sup:
            used.update(sup)
        elif not _contains(question, name):
            missing.append(name)
    if missing:
        return Grounding(False, f"not in the cited evidence: {', '.join(dict.fromkeys(missing))}"[:400],
                         problem="unsupported")
    if not used:  # nothing checkable in the answer (e.g. 'No'): it must at least rest on cited evidence
        used = set(texts)
    ids = [i for i in texts if i in used]
    return Grounding(True, f"every name and number is stated in the cited evidence ({len(ids)} item(s))", ids,
                     alldocs(ids))
