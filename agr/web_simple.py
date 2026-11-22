"""
A small local web page for asking OlympicRAG questions (standard library only, nothing to install).

    python -m agr.web_simple     ->  open http://localhost:8000

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

PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>OlympicRAG</title>
<style>
:root{--bg:#0f172a;--panel:#111c33;--line:#24324f;--ink:#e2e8f0;--muted:#94a3b8;--accent:#f97316;--ok:#22c55e;--blue:#60a5fa}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 "Segoe UI",system-ui,sans-serif}
.wrap{max-width:980px;margin:0 auto;padding:28px 18px 60px}
h1{margin:0;font-size:26px}h1 span{color:var(--accent)}.sub{color:var(--muted);margin:4px 0 20px}
form{display:flex;gap:10px}textarea{flex:1;min-height:64px;resize:vertical;background:var(--panel);color:var(--ink);
border:1px solid var(--line);border-radius:10px;padding:12px;font:inherit}
button{background:var(--accent);color:#fff;border:0;border-radius:10px;padding:0 26px;font-weight:700;font-size:16px;cursor:pointer}
button:disabled{opacity:.5;cursor:wait}
.examples{margin:10px 0 0;color:var(--muted);font-size:13px}.examples a{color:var(--blue);cursor:pointer;text-decoration:none;margin-right:14px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:16px 18px;margin-top:16px}
.card h2{margin:0 0 10px;font-size:13px;letter-spacing:.08em;text-transform:uppercase;color:var(--muted)}
.answer{font-size:24px;font-weight:700}.meta{color:var(--muted);font-size:13px;margin-top:6px}
.ok{color:var(--ok)}.warn{color:#fbbf24}
.mcp{font:13px/1.6 Consolas,monospace;white-space:pre-wrap}.mcp b{color:var(--accent)}
table{width:100%;border-collapse:collapse;font-size:13px}td,th{border-top:1px solid var(--line);padding:7px 6px;text-align:left;vertical-align:top}
th{color:var(--muted);font-weight:600}.tag{display:inline-block;padding:1px 7px;border-radius:6px;background:#1e293b;font-size:12px}
.tag.mcp-t{background:#7c2d12;color:#fed7aa}ul{margin:0;padding-left:18px}a.src{color:var(--blue)}
.spin{color:var(--muted)}
</style></head><body><div class="wrap">
<h1>Olympic<span>RAG</span></h1>
<div class="sub">Agentic GraphRAG on TigerGraph. Every graph query goes through the TigerGraph MCP server.</div>
<form id="f"><textarea id="q" placeholder="Ask a question about the corpus"></textarea><button id="b">Ask</button></form>
<div class="examples">Try:
<a data-q="The event held at Deer Valley on February 23, 2002: who won the gold medal in that same event at the previous Olympics?">Deer Valley, previous Games</a>
<a data-q="According to the provided corpus, how many biathlon events at the 2018 Winter Olympics had more than 73 competitors?">2018 biathlon count</a>
<a data-q="Who won the gold medal in the women's 200 metres athletics event at the Summer Olympics held immediately before 2016?">200 m before 2016</a></div>
<div id="out"></div></div>
<script>
const $=s=>document.querySelector(s), esc=s=>String(s??"").replace(/[&<>"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
document.querySelectorAll(".examples a").forEach(a=>a.onclick=()=>{$("#q").value=a.dataset.q;});
$("#f").onsubmit=async e=>{e.preventDefault();const q=$("#q").value.trim();if(!q)return;
 $("#b").disabled=true;const t0=Date.now();$("#out").innerHTML='<div class="card spin" id="sp">Investigating... (the first question also starts the MCP server)</div>';
 const tick=setInterval(()=>{const s=$("#sp");if(s)s.textContent="Investigating... "+Math.round((Date.now()-t0)/1000)+" s";},500);
 try{const r=await fetch("/ask",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({question:q})});
  const d=await r.json();render(d);}catch(err){$("#out").innerHTML='<div class="card warn">Error: '+esc(err)+'</div>';}
 clearInterval(tick);$("#b").disabled=false;};
function render(d){if(d.error){$("#out").innerHTML='<div class="card warn">Error: '+esc(d.error)+'</div>';return;}
 const viaMcp=d.mcp.length&&d.mcp.every(c=>!c.error);
 let h='<div class="card"><h2>Answer</h2><div class="answer">'+esc(d.answer)+'</div><div class="meta">'+
  (d.grounded?'<span class="ok">&#10003; supported by one cited source</span>':'<span class="warn">not grounded</span>')+
  ' &middot; '+d.llm_calls+' LLM calls &middot; '+d.total_tokens.toLocaleString()+' tokens &middot; '+d.seconds+' s</div>';
 if(d.citations.length)h+='<div class="meta">Source: '+d.citations.map(c=>'<a class="src" target="_blank" href="'+esc(c.url)+'">'+esc(c.title)+'</a>').join(", ")+'</div>';
 h+='</div>';
 if(d.notes.length)h+='<div class="card"><h2>Notes written by code</h2><ul>'+d.notes.map(n=>'<li>'+esc(n)+'</li>').join("")+'</ul></div>';
 h+='<div class="card"><h2>TigerGraph MCP calls ('+d.mcp.length+')</h2><div class="mcp">'+(d.mcp.length?d.mcp.map(c=>
  '<b>'+esc(c.tool)+'</b> &rarr; '+esc(c.query)+'('+Object.entries(c.params).map(([k,v])=>esc(k)+'='+esc(v)).join(", ")+')  '+c.ms+' ms, '+
  (c.error?'<span class="warn">failed, RESTPP fallback</span>':c.rows+' row(s)')).join("\n"):'none (this answer needed no graph query)')+'</div></div>';
 h+='<div class="card"><h2>Reasoning chain</h2><ul>'+d.reasoning_chain.map(x=>'<li>'+esc(x)+'</li>').join("")+'</ul></div>';
 h+='<div class="card"><h2>Investigation steps</h2><table><tr><th>#</th><th>Agent</th><th>Action</th><th>Why</th><th>Time</th></tr>'+
  d.steps.map(s=>'<tr><td>'+s.n+'</td><td>'+esc(s.agent)+'</td><td>'+esc(s.action)+(s.via?' <span class="tag'+(s.via.startsWith("mcp")?' mcp-t':'')+'">via '+esc(s.via)+'</span>':'')+
  '</td><td>'+esc(s.why)+'</td><td>'+s.ms+' ms</td></tr>').join("")+'</table></div>';
 $("#out").innerHTML=h;}
</script></body></html>"""


def answer(question: str) -> dict:
    global _tb
    with _lock:
        if _tb is None:
            _tb = Toolbox()
        tgmod.MCP_CALLS = []
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
