"""
Create the graph named by TG_GRAPH on TigerGraph, load documents/events/chunks(+vectors)/source records,
install queries, and write graph_counts.json (vertex and edge counts by type).

    python -m agr.parse data/corpus/corpus.jsonl build      # parse corpus
    python -m agr.embed build                                # local embeddings -> build/chunk_emb.npy
    python -m graph.setup_graph                              # schema + load + install
Flags: --skip-schema --skip-load --skip-queries
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

from agr.config import ROOT, SETTINGS
from agr.parse import norm, person_names, split_names
from agr.tg import TG

BUILD = ROOT / "build"
CORPUS = Path(__import__("os").getenv("CORPUS", str(ROOT / "data/corpus/corpus.jsonl")))


def log(*a):
    print("[setup]", *a, flush=True)


def batched(items, n):
    for i in range(0, len(items), n):
        yield items[i:i + n]


GRAPH = SETTINGS.tg_graph


def gsql_file(name: str) -> str:
    """A GSQL file with the @GRAPH@ placeholder replaced by the configured graph name (TG_GRAPH)."""
    return (ROOT / "graph" / name).read_text(encoding="utf-8").replace("@GRAPH@", GRAPH)


def create_schema(tg: TG):
    out = tg.gsql(f"SHOW GRAPH {GRAPH}")
    if f"Graph {GRAPH}(" in out and "OfficeTerm" in out:
        log(f"graph {GRAPH} exists, schema step skipped")
        return
    if f"Graph {GRAPH}(" not in out:
        log(tg.gsql(f"CREATE GRAPH {GRAPH}()"))
    log(tg.gsql(gsql_file("schema.gsql"))[-800:])


def load(tg: TG):
    docs = json.loads((BUILD / "documents.json").read_text(encoding="utf-8"))
    events = json.loads((BUILD / "events.json").read_text(encoding="utf-8"))
    chunks = [json.loads(l) for l in open(BUILD / "chunks.jsonl", encoding="utf-8")]
    emb = np.load(BUILD / "chunk_emb.npy")
    assert len(emb) == len(chunks), "embeddings out of date: rerun python -m agr.embed"

    for b in batched(docs, 1000):
        tg.upsert({"Doc": {d["id"]: {k: {"value": d[k]} for k in ("title", "url", "kind", "tokens")} for d in b}})
    log(f"docs {len(docs)}")

    V = {"Event": {}, "Games": {}, "Sport": {}, "Venue": {}, "Athlete": {}, "Nation": {}}
    E = {"Doc": {}, "Event": {}, "Games": {}}
    def edge(src_t, src, et, dst_t, dst, attrs=None):
        E.setdefault(src_t, {}).setdefault(src, {}).setdefault(et, {}).setdefault(dst_t, {})[dst] = \
            {k: {"value": v} for k, v in (attrs or {}).items()}

    for e in events:
        V["Event"][e["id"]] = {k: {"value": v} for k, v in {
            "title": e["title"], "title_norm": norm(e["title"]), "sport": e["sport"], "sport_norm": norm(e["sport"]),
            "event": e["event"], "event_norm": norm(e["event"]), "year": e["year"], "season": e["season"],
            "venue": e["venue"], "venue_norm": norm(e["venue"]), "date_text": e["date"], "date_norm": norm(e["date"]),
            "competitors": e["competitors"], "nations": e["nations"], "gold": e["gold"], "silver": e["silver"],
            "bronze": e["bronze"], "gold_noc": e["gold_noc"], "silver_noc": e["silver_noc"],
            "bronze_noc": e["bronze_noc"], "win_value": e["win_value"]}.items()}
        V["Games"][e["games"]] = {"year": {"value": e["year"]}, "season": {"value": e["season"]}}
        V["Sport"][norm(e["sport"])] = {"name": {"value": e["sport"]}}
        edge("Doc", e["id"], "DESCRIBES", "Event", e["id"])
        edge("Event", e["id"], "IN_GAMES", "Games", e["games"])
        edge("Event", e["id"], "OF_SPORT", "Sport", norm(e["sport"]))
        if e["venue"]:
            V["Venue"][norm(e["venue"])] = {"name": {"value": e["venue"]}}
            edge("Event", e["id"], "AT_VENUE", "Venue", norm(e["venue"]))
        for medal in ("gold", "silver", "bronze"):
            names = person_names(e[medal], e["event"])
            if medal == "bronze":
                for extra in e["extra_bronze"]:
                    names += person_names(extra, e["event"])
            for name in names:
                V["Athlete"][norm(name)] = {"name": {"value": name}}
                edge("Event", e["id"], "WON_MEDAL", "Athlete", norm(name), {"medal": medal})
            noc = e[f"{medal}_noc"]
            if noc:
                V["Nation"][noc] = {}
                edge("Event", e["id"], "MEDAL_FOR", "Nation", noc, {"medal": medal})
        if e.get("next_id"):
            edge("Event", e["id"], "NEXT_EDITION", "Event", e["next_id"])
    # Games chain per season
    for season in ("Summer", "Winter"):
        ys = sorted({v["year"]["value"] for k, v in V["Games"].items() if v["season"]["value"] == season})
        for a, b in zip(ys, ys[1:]):
            edge("Games", f"{a} {season}", "NEXT_GAMES", "Games", f"{b} {season}")

    for vt, items in V.items():
        for b in batched(list(items.items()), 2000):
            tg.upsert({vt: dict(b)})
        log(f"{vt} {len(items)}")
    for st, srcs in E.items():
        for b in batched(list(srcs.items()), 1500):
            tg.upsert(edges={st: dict(b)})
    log("edges loaded")

    t0 = time.time()
    for i, b in enumerate(batched(list(range(len(chunks))), 400)):
        verts = {chunks[j]["id"]: {"doc_id": {"value": chunks[j]["doc_id"]}, "idx": {"value": chunks[j]["idx"]},
                                   "text": {"value": chunks[j]["text"]},
                                   "emb": {"value": [round(float(x), 6) for x in emb[j]]}} for j in b}
        eds = {"Doc": {}}
        for j in b:
            eds["Doc"].setdefault(chunks[j]["doc_id"], {"HAS_CHUNK": {"Chunk": {}}})["HAS_CHUNK"]["Chunk"][chunks[j]["id"]] = {}
        tg.upsert(verts and {"Chunk": verts}, eds)
        if i % 10 == 0:
            log(f"chunks {min((i + 1) * 400, len(chunks))}/{len(chunks)} ({time.time() - t0:.0f}s)")
    log("chunks loaded")


def reload_athletes(tg: TG):
    """Delete every Athlete vertex (and its WON_MEDAL edges) and load them again with the current name rules."""
    r = tg._req("DELETE", f"{tg.base}/restpp/graph/{tg.graph}/vertices/Athlete", headers=tg._h(), timeout=600)
    log("deleted athletes:", r.json().get("results"))
    events = json.loads((BUILD / "events.json").read_text(encoding="utf-8"))
    V, E = {}, {}
    for e in events:
        for medal in ("gold", "silver", "bronze"):
            names = person_names(e[medal], e["event"])
            if medal == "bronze":
                for extra in e["extra_bronze"]:
                    names += person_names(extra, e["event"])
            for name in names:
                V[norm(name)] = {"name": {"value": name}}
                E.setdefault(e["id"], {}).setdefault("WON_MEDAL", {}).setdefault("Athlete", {})[norm(name)] = \
                    {"medal": {"value": medal}}
    for b in batched(list(V.items()), 2000):
        tg.upsert({"Athlete": dict(b)})
    for b in batched(list(E.items()), 1500):
        tg.upsert(edges={"Event": dict(b)})
    log(f"athletes {len(V)}")


def build_claims() -> tuple[dict, dict]:
    """Claim vertices + edges for reasoning over time (agr/temporal.py): the infobox record, article statements of the
    totals, and sentences describing a result change. The official record SUPERSEDES the original-result claims."""
    from agr.temporal import event_versions
    events = json.loads((BUILD / "events.json").read_text(encoding="utf-8"))
    corpus = {json.loads(l)["doc_id"]: json.loads(l)["text"]
              for l in open(CORPUS, encoding="utf-8")}
    V, E = {}, {}
    val = lambda **kw: {k: {"value": v} for k, v in kw.items()}
    for e in events:
        v = event_versions(e, corpus.get(e["id"], ""))
        if not (v["counts"] or v["changes"]):
            continue
        rid = f"{e['id']}#record"
        r = v["record"]
        V[rid] = val(event_id=e["id"], kind="official_record", attribute="record", claim_value="",
                     source="infobox (structured, current official record)",
                     sentence=f"gold: {r['gold']}; silver: {r['silver']}; bronze: {r['bronze']}; "
                              f"competitors: {r['competitors']}; nations: {r['nations']}", dates="", status="current")
        E.setdefault(e["id"], {}).setdefault("HAS_CLAIM", {}).setdefault("Claim", {})[rid] = {}
        for i, c in enumerate(v["counts"]):
            cid = f"{e['id']}#count{i}"
            conflict = [x for x in v["count_conflicts"] if x["sentence"] == c["sentence"]]
            V[cid] = val(event_id=e["id"], kind="count_statement", attribute="competitors/nations",
                         claim_value=f"{c['competitors']}/{c['nations']}", source="article text",
                         sentence=c["sentence"], dates="",
                         status="conflicts with the official record" if conflict else "agrees with the official record")
            E[e["id"]]["HAS_CLAIM"]["Claim"][cid] = {}
        for i, c in enumerate(v["changes"]):
            cid = f"{e['id']}#change{i}"
            V[cid] = val(event_id=e["id"], kind=c["kind"], attribute="result", claim_value="", source="article text",
                         sentence=c["sentence"], dates="; ".join(c["dates"]),
                         status="describes a result that was changed later; the current result is the official record")
            E[e["id"]]["HAS_CLAIM"]["Claim"][cid] = {}
            if c["kind"] == "original_result":
                E.setdefault(rid, {}).setdefault("SUPERSEDES", {}).setdefault("Claim", {})[cid] = {}
    return V, E


def load_claims(tg: TG):
    V, E = build_claims()
    for b in batched(list(V.items()), 1500):
        tg.upsert({"Claim": dict(b)})
    ev_edges = {k: v for k, v in E.items() if "#" not in k}
    cl_edges = {k: v for k, v in E.items() if "#" in k}
    for b in batched(list(ev_edges.items()), 1500):
        tg.upsert(edges={"Event": dict(b)})
    for b in batched(list(cl_edges.items()), 1500):
        tg.upsert(edges={"Claim": dict(b)})
    log(f"claims {len(V)}")


def load_records(tg: TG):
    """Source records: who published them and when, linked to the one event their title names (agr.parse)."""
    recs = json.loads((BUILD / "records.json").read_text(encoding="utf-8"))
    V, E, R = {}, {}, {}
    for r in recs:
        utc = r["published_utc"].replace("T", " ").rstrip("Z") if r["published_utc"] else "1970-01-01 00:00:00"
        V[r["id"]] = {"title": {"value": r["title"]}, "source": {"value": r["source"]},
                      "published": {"value": r["published"]}, "published_utc": {"value": utc},
                      "body": {"value": r["body"]}, "tier": {"value": r.get("tier", 0)},
                      "tier_label": {"value": r.get("tier_label", "")}}
        E.setdefault(r["id"], {}).setdefault("HAS_RECORD", {}).setdefault("SourceRecord", {})[r["id"]] = {}
        if r.get("event_id"):
            R.setdefault(r["id"], {}).setdefault("ABOUT_EVENT", {}).setdefault("Event", {})[r["event_id"]] = \
                {"how": {"value": r.get("link_how", "")}}
    if V:
        tg.upsert({"SourceRecord": V})
        tg.upsert(edges={"Doc": E})
    if R:
        tg.upsert(edges={"SourceRecord": R})
    log(f"source records {len(V)}, linked to an event {len(R)}")


def split_people(s: str) -> list[str]:
    """Predecessor / successor names: only clear separators (agr.parse.split_names); placeholders are dropped."""
    return [n for n in split_names(s or "") if n.lower() not in ("none", "vacant", "position established",
                                                                  "position abolished", "office established")]


def entity_payload() -> tuple[dict, dict]:
    """Film / Person / OfficeTerm vertices and their edges from build/films.json, people.json, terms.json."""
    val = lambda d, ks: {k: {"value": d[k]} for k in ks}
    films = json.loads((BUILD / "films.json").read_text(encoding="utf-8"))
    people = {p["id"]: p for p in json.loads((BUILD / "people.json").read_text(encoding="utf-8"))}
    terms = json.loads((BUILD / "terms.json").read_text(encoding="utf-8"))
    holder_of = {p["doc_id"]: p["id"] for p in people.values()}
    V = {"Film": {}, "Person": {}, "OfficeTerm": {}}
    E: dict = {}

    def edge(st, s, et, dt, d, attrs=None):
        E.setdefault(st, {}).setdefault(s, {}).setdefault(et, {}).setdefault(dt, {})[d] = \
            {k: {"value": v} for k, v in (attrs or {}).items()}

    def person(name: str) -> str:
        pid = norm(name)
        if pid and pid not in V["Person"]:
            p = people.get(pid, {"name": name, "doc_id": "", "kind": "mentioned", "birth_date": "", "party": "",
                                 "occupation": ""})
            V["Person"][pid] = val(p, ("name", "doc_id", "kind", "birth_date", "party", "occupation"))
        return pid

    for f in films:
        V["Film"][f["id"]] = {**val(f, ("title", "name", "year", "released", "director", "starring", "music",
                                        "country", "language", "runtime_min", "budget", "gross", "budget_currency",
                                        "budget_low", "budget_high", "gross_currency", "gross_low", "gross_high",
                                        "gross_note")), "title_norm": {"value": norm(f["title"])}}
        edge("Doc", f["id"], "ABOUT_FILM", "Film", f["id"])
        roles: dict[str, list[str]] = {}
        for role, names in f["credits"].items():
            for n in names:
                pid = person(n)
                if pid and role not in roles.setdefault(pid, []):
                    roles[pid].append(role)
        for pid, rs in roles.items():   # one edge per (film, person); several roles are kept together
            edge("Film", f["id"], "FILM_CREDIT", "Person", pid, {"role": ",".join(rs)})
    for p in people.values():
        person(p["name"])
        edge("Doc", p["doc_id"], "ABOUT_PERSON", "Person", p["id"])
    for tm in terms:
        V["OfficeTerm"][tm["id"]] = val(tm, ("doc_id", "office", "office_key", "term_start", "term_end",
                                             "start_year", "end_year", "predecessor", "successor"))
        if holder_of.get(tm["doc_id"]):
            edge("Person", holder_of[tm["doc_id"]], "HELD", "OfficeTerm", tm["id"])
        for et, names in (("PRECEDED_BY", tm["predecessor"]), ("SUCCEEDED_BY", tm["successor"])):
            for n in split_people(names):
                pid = person(n)
                if pid:
                    edge("OfficeTerm", tm["id"], et, "Person", pid)
        if tm.get("next_id"):
            edge("OfficeTerm", tm["id"], "NEXT_TERM", "OfficeTerm", tm["next_id"])
    return V, E


def load_entities(tg: TG):
    V, E = entity_payload()
    for vt, items in V.items():
        for b in batched(list(items.items()), 1500):
            tg.upsert({vt: dict(b)})
    for st, srcs in E.items():
        for b in batched(list(srcs.items()), 1000):
            tg.upsert(edges={st: dict(b)})
    log(f"films {len(V['Film'])}, people {len(V['Person'])}, office terms {len(V['OfficeTerm'])}")


def install_queries(tg: TG):
    out = tg.gsql(gsql_file("queries.gsql"), timeout=3600)
    log(out[-1500:])


VERTEX_TYPES = ("Doc", "Chunk", "Event", "Games", "Sport", "Venue", "Athlete", "Nation", "Claim", "SourceRecord",
                "Film", "Person", "OfficeTerm")
EDGE_TYPES = ("HAS_CHUNK", "DESCRIBES", "IN_GAMES", "OF_SPORT", "AT_VENUE", "WON_MEDAL", "MEDAL_FOR", "NEXT_EDITION",
              "NEXT_GAMES", "HAS_CLAIM", "SUPERSEDES", "HAS_RECORD", "ABOUT_EVENT", "ABOUT_FILM", "ABOUT_PERSON",
              "FILM_CREDIT", "HELD", "PRECEDED_BY", "SUCCEEDED_BY", "NEXT_TERM")


def write_counts(tg: TG, path: Path = ROOT / "graph_counts.json") -> dict:
    """Vertex and edge counts by type, read from TigerGraph's built-in stat endpoints."""
    def stat(kind: str) -> dict:
        r = tg._req("POST", f"{tg.base}/restpp/builtins/{tg.graph}", headers=tg._h({"Content-Type": "application/json"}),
                    data=json.dumps({"function": f"stat_{kind}_number", "type": "*"}), timeout=120)
        body = r.json()
        key = "v_type" if kind == "vertex" else "e_type"
        return {x[key]: x["count"] for x in body.get("results", [])}
    v, e = stat("vertex"), stat("edge")
    out = {"graph": tg.graph, "vertices": {k: v.get(k, 0) for k in VERTEX_TYPES},
           "edges": {k: e.get(k, 0) for k in EDGE_TYPES},
           "edges_reverse": {k: e[k] for k in ("PREV_EDITION", "PREV_GAMES", "SUPERSEDED_BY", "PREV_TERM") if k in e}}
    out["vertices_total"] = sum(out["vertices"].values())
    out["edges_total"] = sum(out["edges"].values())
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    log(json.dumps(out))
    return out


if __name__ == "__main__":
    tg = TG()
    if "--skip-schema" not in sys.argv:
        create_schema(tg)
    if "--athletes-only" in sys.argv:
        reload_athletes(tg)
    elif "--claims-only" in sys.argv:
        load_claims(tg)
    elif "--records-only" in sys.argv:
        load_records(tg)
    elif "--entities-only" in sys.argv:
        load_entities(tg)
    elif "--skip-load" not in sys.argv:
        load(tg)
        load_claims(tg)
        load_records(tg)
        load_entities(tg)
    if "--skip-queries" not in sys.argv:
        install_queries(tg)
    write_counts(tg)
