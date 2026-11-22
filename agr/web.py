"""
A small local web page for asking OlympicRAG questions (standard library only, nothing to install).

    python -m agr.web            ->  open http://localhost:8000

Type a question and press Ask. The Agentic GraphRAG pipeline answers it and the page shows the answer, its cited
evidence, the code-written notes, the reasoning chain, every step of the investigation and every call made to the
TigerGraph MCP server. The MCP server is started once and stays running, so questions after the first are fast.
"""
from __future__ import annotations

import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

os.environ.setdefault("TG_TRANSPORT", "mcp")

from agr import tg as tgmod  # noqa: E402
from agr.pipelines import run_agentic  # noqa: E402
from agr.tools import Toolbox  # noqa: E402

PORT = int(os.getenv("PORT", "8000"))
_lock = threading.Lock()   # one question at a time: keeps the MCP call log per question
_tb: Toolbox | None = None
_live: list = []           # MCP calls of the question being answered, read by /progress

PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>OlympicRAG</title>
<style>
:root{--bg:#0b1222;--panel:#111c33;--panel2:#0f1830;--line:#24324f;--ink:#e2e8f0;--muted:#94a3b8;--accent:#f97316;--ok:#22c55e;--blue:#60a5fa}
*{box-sizing:border-box}body{margin:0;background:radial-gradient(1200px 500px at 50% -200px,#1b2a4d,var(--bg));color:var(--ink);font:15px/1.5 "Segoe UI",system-ui,sans-serif;min-height:100vh}
.wrap{max-width:1000px;margin:0 auto;padding:26px 18px 60px}
.top{display:flex;justify-content:space-between;align-items:center;gap:12px;flex-wrap:wrap}
h1{margin:0;font-size:28px;letter-spacing:-.01em}h1 span{color:var(--accent)}
.badge{display:inline-flex;align-items:center;gap:8px;border:1px solid #7c2d12;background:#2a1408;color:#fed7aa;border-radius:999px;padding:5px 12px;font-size:13px}
.dot{width:8px;height:8px;border-radius:50%;background:var(--ok);box-shadow:0 0 0 3px #22c55e33}
.sub{color:var(--muted);margin:4px 0 18px}
.ask{background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:12px}
form{display:flex;gap:10px}textarea{flex:1;min-height:62px;resize:vertical;background:var(--panel2);color:var(--ink);
border:1px solid var(--line);border-radius:10px;padding:12px;font:inherit;outline:none}textarea:focus{border-color:var(--accent)}
button{background:var(--accent);color:#fff;border:0;border-radius:10px;padding:0 28px;font-weight:700;font-size:16px;cursor:pointer}
button:disabled{opacity:.5;cursor:wait}
.examples{margin-top:10px;display:flex;flex-wrap:wrap;gap:8px;align-items:center;color:var(--muted);font-size:13px}
.chip{border:1px solid var(--line);background:var(--panel2);color:var(--blue);border-radius:999px;padding:3px 11px;cursor:pointer}
.chip:hover{border-color:var(--blue)}.hint{margin-left:auto;font-size:12px}
.flow{display:flex;align-items:center;gap:6px;margin:16px 0 0;flex-wrap:wrap;font-size:13px}
.node{border:1px solid var(--line);background:var(--panel);border-radius:8px;padding:5px 10px;color:var(--muted);transition:.25s}
.node.on{border-color:var(--accent);color:#fff;background:#2a1408;box-shadow:0 0 12px #f9731655}.node.done{border-color:var(--ok);color:#bbf7d0}
.arr{color:#475569}
.card{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:16px 18px;margin-top:16px}
.card h2{margin:0 0 10px;font-size:12px;letter-spacing:.09em;text-transform:uppercase;color:var(--muted);display:flex;justify-content:space-between}
.ans{border-left:4px solid var(--ok)}.ans.bad{border-left-color:#fbbf24}
.answer{font-size:26px;font-weight:700}.meta{color:var(--muted);font-size:13px;margin-top:6px}
.ok{color:var(--ok)}.warn{color:#fbbf24}a.src{color:var(--blue)}
.stats{display:grid;grid-template-columns:repeat(5,1fr);gap:10px;margin-top:16px}
.stat{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:10px 12px}
.stat b{display:block;font-size:22px}.stat span{color:var(--muted);font-size:12px}.stat.hl{border-color:#7c2d12}.stat.hl b{color:var(--accent)}
.call{display:grid;grid-template-columns:26px 1fr 150px;gap:10px;align-items:center;padding:7px 0;border-top:1px solid var(--line);font:13px/1.45 Consolas,monospace;animation:in .3s}
.call:first-child{border-top:0}.n{color:var(--accent);font-weight:700}.q{color:#fff}.pr{color:var(--muted)}
.bar{height:6px;background:#1e293b;border-radius:4px;overflow:hidden}.bar i{display:block;height:100%;background:var(--accent)}
.ms{font-size:12px;color:var(--muted);text-align:right}
@keyframes in{from{opacity:0;transform:translateY(4px)}to{opacity:1}}
table{width:100%;border-collapse:collapse;font-size:13px}td,th{border-top:1px solid var(--line);padding:7px 6px;text-align:left;vertical-align:top}
th{color:var(--muted);font-weight:600}.tag{display:inline-block;padding:1px 7px;border-radius:6px;background:#1e293b;font-size:12px;white-space:nowrap}
.tag.m{background:#7c2d12;color:#fed7aa}.agent{color:var(--blue);white-space:nowrap}ul{margin:0;padding-left:18px}li{margin:3px 0}
.spin{display:inline-block;width:14px;height:14px;border:2px solid #334155;border-top-color:var(--accent);border-radius:50%;animation:r 1s linear infinite;vertical-align:-2px;margin-right:8px}
@keyframes r{to{transform:rotate(360deg)}}
details summary{cursor:pointer;color:var(--muted);font-size:12px;letter-spacing:.09em;text-transform:uppercase}
@media(max-width:700px){.stats{grid-template-columns:repeat(2,1fr)}.call{grid-template-columns:22px 1fr}.call .ms{text-align:left;grid-column:2}}
</style></head><body><div class="wrap">
<div class="top"><h1>Olympic<span>RAG</span></h1><div class="badge"><span class="dot"></span>Graph queries via TigerGraph MCP server</div></div>
<div class="sub">Agentic GraphRAG on TigerGraph: an orchestrator plans each step, specialised agents query the graph through MCP, and code checks the evidence before any answer is released.</div>
<div class="ask"><form id="f"><textarea id="q" placeholder="Ask a multi-step question about the corpus"></textarea><button id="b">Ask</button></form>
<div class="examples">Try:
<span class="chip" data-q="The event held at Deer Valley on February 23, 2002: who won the gold medal in that same event at the previous Olympics?">Deer Valley, previous Games</span>
<span class="chip" data-q="According to the provided corpus, how many biathlon events at the 2018 Winter Olympics had more than 73 competitors?">2018 biathlon count</span>
<span class="chip" data-q="Who won the gold medal in the women's 200 metres athletics event at the Summer Olympics held immediately before 2016?">200 m before 2016</span>
<span class="hint">Ctrl+Enter to ask</span></div></div>
<div class="flow" id="flow"><span class="node" id="n0">Question</span><span class="arr">&rarr;</span><span class="node" id="n1">Orchestrator (Gemini)</span><span class="arr">&rarr;</span>
<span class="node" id="n2">TigerGraph MCP server</span><span class="arr">&rarr;</span><span class="node" id="n3">Evidence check (code)</span><span class="arr">&rarr;</span><span class="node" id="n4">Answer</span></div>
<div id="out"></div></div>
<script>
const $=s=>document.querySelector(s), esc=s=>String(s??"").replace(/[&<>"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
function flow(i,done){for(let k=0;k<5;k++){const e=$("#n"+k);e.className="node"+(done||k<i?" done":k==i?" on":"");}}
function callRows(calls){if(!calls.length)return'<div class="pr">none yet</div>';const mx=Math.max(...calls.map(c=>c.ms),1);
 return calls.map((c,i)=>'<div class="call"><span class="n">'+(i+1)+'</span><div><span class="q">'+esc(c.query)+'</span><span class="pr">('+
 Object.entries(c.params).map(([k,v])=>esc(k)+'='+esc(v)).join(", ")+')</span><br><span class="pr">'+esc(c.tool)+' &middot; '+
 (c.error?'<span class="warn">failed, RESTPP fallback</span>':c.rows+' row(s)')+'</span></div><div><div class="bar"><i style="width:'+
 Math.max(4,100*c.ms/mx)+'%"></i></div><div class="ms">'+c.ms+' ms</div></div></div>').join("");}
document.querySelectorAll(".chip").forEach(a=>a.onclick=()=>{$("#q").value=a.dataset.q;$("#q").focus();});
$("#q").addEventListener("keydown",e=>{if(e.key=="Enter"&&(e.ctrlKey||e.metaKey)){e.preventDefault();$("#f").requestSubmit();}});
$("#f").onsubmit=async e=>{e.preventDefault();const q=$("#q").value.trim();if(!q)return;
 $("#b").disabled=true;const t0=Date.now();flow(1);
 $("#out").innerHTML='<div class="card"><h2><span><span class="spin"></span>Investigating</span><span id="secs">0 s</span></h2>'+
  '<div class="meta" style="margin-bottom:8px">Live TigerGraph MCP calls (the first question also starts the MCP server):</div><div id="live">'+callRows([])+'</div></div>';
 let busy=false;const tick=setInterval(async()=>{const s=$("#secs");if(s)s.textContent=Math.round((Date.now()-t0)/1000)+" s";
  if(busy)return;busy=true;try{const p=await (await fetch("/progress")).json();const l=$("#live");if(l&&p.mcp){if(p.mcp.length)flow(2);
  if(l.dataset.n!=p.mcp.length){l.dataset.n=p.mcp.length;l.innerHTML=callRows(p.mcp);}}}catch(_){}busy=false;},700);
 try{const r=await fetch("/ask",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({question:q})});
  const d=await r.json();render(d);}catch(err){$("#out").innerHTML='<div class="card warn">Error: '+esc(err)+'</div>';flow(-1);}
 clearInterval(tick);$("#b").disabled=false;};
function render(d){if(d.error){$("#out").innerHTML='<div class="card warn">Error: '+esc(d.error)+'</div>';flow(-1);return;}
 flow(5,true);const viaMcp=d.steps.filter(s=>(s.via||"").startsWith("mcp")).length;
 let h='<div class="card ans'+(d.grounded?'':' bad')+'"><h2><span>Answer</span></h2><div class="answer">'+esc(d.answer)+'</div><div class="meta">'+
  (d.grounded?'<span class="ok">&#10003; Supported by one cited source (checked in code)</span>':'<span class="warn">Not released as grounded</span>')+'</div>';
 if(d.citations.length)h+='<div class="meta">Source: '+d.citations.map(c=>'<a class="src" target="_blank" href="'+esc(c.url)+'">'+esc(c.title)+'</a>').join(", ")+'</div>';
 h+='</div><div class="stats">'+[[d.steps.length,"agent steps"],[d.mcp.length,"MCP calls",1],[d.llm_calls,"LLM calls"],
  [d.total_tokens.toLocaleString(),"tokens"],[d.seconds+" s","total time"]].map(x=>'<div class="stat'+(x[2]?' hl':'')+'"><b>'+x[0]+'</b><span>'+x[1]+'</span></div>').join("")+'</div>';
 if(d.notes.length)h+='<div class="card"><h2><span>Notes written by code</span></h2><ul>'+d.notes.map(n=>'<li>'+esc(n)+'</li>').join("")+'</ul></div>';
 h+='<div class="card"><h2><span>TigerGraph MCP calls ('+d.mcp.length+')</span><span>tool: run_installed_query</span></h2>'+
  (d.mcp.length?callRows(d.mcp):'<div class="pr">None: this answer needed no graph query.</div>')+'</div>';
 h+='<div class="card"><h2><span>Reasoning chain</span></h2><ul>'+d.reasoning_chain.map(x=>'<li>'+esc(x)+'</li>').join("")+'</ul></div>';
 h+='<div class="card"><details open><summary>Investigation steps ('+d.steps.length+', '+viaMcp+' via MCP)</summary><table style="margin-top:10px"><tr><th>#</th><th>Agent</th><th>Action</th><th>Why</th><th>Time</th></tr>'+
  d.steps.map(s=>'<tr><td>'+s.n+'</td><td class="agent">'+esc(s.agent)+'</td><td>'+esc(s.action)+(s.via?' <span class="tag'+(s.via.startsWith("mcp")?' m':'')+'">via '+esc(s.via)+'</span>':'')+
  '</td><td>'+esc(s.why)+'</td><td style="white-space:nowrap">'+s.ms+' ms</td></tr>').join("")+'</table></details></div>';
 $("#out").innerHTML=h;}
</script></body></html>"""


def answer(question: str) -> dict:
    global _tb, _live
    with _lock:
        if _tb is None:
            _tb = Toolbox()
        tgmod.MCP_CALLS = _live = []
        t0 = time.time()
        try:
            r = run_agentic(_tb, "web", question)
        finally:
            calls, tgmod.MCP_CALLS = tgmod.MCP_CALLS, None
        return {"answer": r.answer, "grounded": r.grounded, "notes": r.notes, "reasoning_chain": r.reasoning_chain,
                "citations": [{"title": c["title"], "url": c.get("url", "")} for c in r.citations],
                "total_tokens": r.total_tokens, "llm_calls": r.llm_calls, "seconds": round(time.time() - t0, 1),
                "mcp": calls,
                "steps": [{"n": s.n, "agent": s.agent, "action": s.action, "why": s.reasoning, "ms": round(s.ms),
                           "via": getattr(s, "via", "")} for s in r.steps]}


class Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, body: bytes, ctype: str):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.startswith("/progress"):
            return self._send(200, json.dumps({"mcp": list(_live)}).encode("utf-8"), "application/json")
        self._send(200, PAGE.encode("utf-8"), "text/html; charset=utf-8")

    def do_POST(self):
        if self.path != "/ask":
            return self._send(404, b"{}", "application/json")
        try:
            q = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}").get("question", "")
            out = answer(q.strip()) if q.strip() else {"error": "empty question"}
        except Exception as e:  # show the problem on the page instead of failing silently
            out = {"error": f"{type(e).__name__}: {e}"}
        self._send(200, json.dumps(out, ensure_ascii=False).encode("utf-8"), "application/json")

    def log_message(self, *args):  # keep the terminal clean for the MCP log
        pass


if __name__ == "__main__":
    print(f"OlympicRAG is running: open http://localhost:{PORT}  (Ctrl+C to stop)", flush=True)
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
