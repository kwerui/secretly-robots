"""A live window onto what the robot hears: speech-to-text output and command state.

`VoiceEvents` collects events (status, transcript, command) in memory and appends
them to `logs/voice_events.jsonl`. `serve()` shows them as a small web page on
http://127.0.0.1:8765 (loopback only, so nothing leaves the laptop):

    events = VoiceEvents()
    url = serve(events)            # background thread
    events.add("transcript", **asdict(transcript))

Or view a log written by an earlier run:

    sim/.venv/bin/python -m brain.voice_window logs/voice_events.jsonl
"""
from __future__ import annotations

import argparse
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

HOST = "127.0.0.1"
PORT = 8765
LOG = Path("logs/voice_events.jsonl")


def _jsonable(v):
    if isinstance(v, dict):
        return {k: _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    if hasattr(v, "__dataclass_fields__"):
        from dataclasses import asdict
        return _jsonable(asdict(v))
    return str(v)


class VoiceEvents:
    """Thread-safe, append-only event list, mirrored to a JSONL file."""

    def __init__(self, log: Path | str | None = LOG):
        self._lock = threading.Lock()
        self._events: list[dict] = []
        self.log = Path(log) if log else None
        if self.log:
            self.log.parent.mkdir(parents=True, exist_ok=True)

    def add(self, kind: str, **fields) -> dict:
        with self._lock:
            ev = {"seq": len(self._events), "kind": kind, "wall": round(time.time(), 3), **_jsonable(fields)}
            self._events.append(ev)
            if self.log:
                with self.log.open("a") as f:
                    f.write(json.dumps(ev) + "\n")
            return ev

    def since(self, seq: int) -> list[dict]:
        with self._lock:
            return self._events[max(0, seq):]

    @classmethod
    def from_log(cls, path: Path | str) -> "VoiceEvents":
        ev = cls(log=None)
        for line in Path(path).read_text().splitlines():
            if line.strip():
                d = json.loads(line)
                d["seq"] = len(ev._events)
                ev._events.append(d)
        return ev


def _handler(events: VoiceEvents):
    class H(BaseHTTPRequestHandler):
        def do_GET(self):
            u = urlparse(self.path)
            if u.path == "/events":
                after = int(parse_qs(u.query).get("after", ["0"])[0])
                body, ctype = json.dumps(events.since(after)).encode(), "application/json"
            elif u.path == "/":
                body, ctype = PAGE.encode(), "text/html; charset=utf-8"
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):  # keep the terminal for the robot's own output
            pass
    return H


def serve(events: VoiceEvents, port: int = PORT, open_browser: bool = False) -> str:
    """Start the window's server in a daemon thread; return its URL."""
    srv = ThreadingHTTPServer((HOST, port), _handler(events))
    threading.Thread(target=srv.serve_forever, daemon=True, name="voice-window").start()
    url = f"http://{HOST}:{srv.server_address[1]}/"
    if open_browser:
        import webbrowser
        webbrowser.open(url)
    return url


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Robot Hearing</title>
<style>
:root{--bg:#f6f5f2;--card:#fff;--ink:#1c1c1a;--muted:#6b6a65;--line:#e4e2dc;
  --ok:#1f7a4d;--okbg:#e3f3ea;--bad:#a33a2b;--badbg:#f8e5e1;--wait:#8a6a12;--waitbg:#f6efd9;--accent:#2d5bd7}
@media (prefers-color-scheme:dark){:root{--bg:#141413;--card:#1e1e1c;--ink:#ecebe6;--muted:#9a988f;
  --line:#2f2e2b;--ok:#6fd3a0;--okbg:#17352a;--bad:#f09a8b;--badbg:#3a1f1a;--wait:#e8c868;--waitbg:#352c12;--accent:#8fb0ff}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);
  font:15px/1.45 -apple-system,BlinkMacSystemFont,"Segoe UI",system-ui,sans-serif}
header{position:sticky;top:0;background:var(--bg);border-bottom:1px solid var(--line);
  padding:14px 16px;display:flex;gap:12px;align-items:center;flex-wrap:wrap}
h1{font-size:17px;margin:0;font-weight:650}
.status{font-size:13px;color:var(--muted);display:flex;align-items:center;gap:6px}
.dot{width:9px;height:9px;border-radius:50%;background:var(--muted)}
.dot.live{background:var(--ok);box-shadow:0 0 0 0 var(--ok);animation:p 1.6s infinite}
@keyframes p{0%{box-shadow:0 0 0 0 color-mix(in srgb,var(--ok) 60%,transparent)}100%{box-shadow:0 0 0 8px transparent}}
main{max-width:760px;margin:0 auto;padding:16px}
.empty{color:var(--muted);text-align:center;padding:48px 0}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px 16px;margin-bottom:12px}
.card.new{animation:f .6s ease-out}@keyframes f{from{transform:translateY(-6px);opacity:0}}
.text{font-size:22px;font-weight:600;margin:0 0 8px}
.text.none{color:var(--muted);font-weight:400;font-style:italic;font-size:17px}
.meta{display:flex;flex-wrap:wrap;gap:6px 14px;font-size:12.5px;color:var(--muted);font-variant-numeric:tabular-nums}
.meta b{color:var(--ink);font-weight:600}
.cmd{margin-top:10px;padding:7px 10px;border-radius:7px;font-size:13.5px;display:inline-block}
.cmd.understood{background:var(--okbg);color:var(--ok)}.cmd.rejected{background:var(--badbg);color:var(--bad)}
.cmd.heard{background:var(--waitbg);color:var(--wait)}
footer{color:var(--muted);font-size:12px;text-align:center;padding:8px 16px 24px}
</style></head><body>
<header><h1>Robot hearing</h1><div class="status"><span class="dot" id="dot"></span><span id="status">connecting…</span></div></header>
<main><div id="list"><p class="empty" id="empty">Nothing heard yet. Press V in the sim, or speak to the mic.</p></div></main>
<footer>Onboard speech-to-text. This page is served from 127.0.0.1 and nothing leaves the laptop.</footer>
<script>
let after=0, byStart={};
const $=s=>document.querySelector(s), list=$("#list");
const fmt=(v,d=2)=>v==null?"–":Number(v).toFixed(d);
function card(t){
  const c=document.createElement("div"); c.className="card new";
  const p=document.createElement("p"); p.className="text"+(t.text?"":" none");
  p.textContent=t.text?`“${t.text}”`:"(not speech)"; c.appendChild(p);
  const m=document.createElement("div"); m.className="meta";
  const bits=[["heard at",`t=${fmt(t.start_s)} s`],["audio",`${fmt(t.audio_s)} s`],["speech-to-text",`${fmt(t.stt_ms,0)} ms`],
    ["level",`${fmt(t.level_dbfs,1)} dBFS`]];
  if(t.distance_m!=null) bits.push(["distance",`${fmt(t.distance_m)} m`]);
  bits.push(["source",t.source],["model",t.model]);
  for(const [k,v] of bits){const s=document.createElement("span");s.append(k+" ");const b=document.createElement("b");b.textContent=v;s.appendChild(b);m.appendChild(s)}
  c.appendChild(m); return c;
}
function cmdText(e){
  if(e.status==="understood"){const t=e.task||{};return `Command: fetch ${t.item_id} → ${t.destination}  ·  ${e.parser_source}, ${fmt(e.intent_ms,0)} ms`}
  if(e.status==="rejected") return `Not a command: ${e.reason||"not understood"}`;
  return `Command: ${e.status}`;
}
function apply(e){
  if(e.kind==="status"){$("#status").textContent=e.message; $("#dot").className="dot"+(e.listening?" live":""); return}
  if(e.kind==="transcript"){$("#empty")?.remove(); const c=card(e); byStart[e.start_s]=c; list.prepend(c); return}
  if(e.kind==="command"){const c=byStart[e.start_s]||list.firstElementChild; if(!c||c.id==="empty") return;
    let d=c.querySelector(".cmd"); if(!d){d=document.createElement("div"); c.appendChild(d)}
    d.className="cmd "+e.status; d.textContent=cmdText(e)}
}
async function tick(){
  try{const r=await fetch(`/events?after=${after}`,{cache:"no-store"}); const evs=await r.json();
    for(const e of evs){apply(e); after=e.seq+1}}
  catch(_){$("#status").textContent="robot not running"; $("#dot").className="dot"}
  setTimeout(tick,300);
}
tick();
</script></body></html>
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("log", nargs="?", default=str(LOG), help="JSONL log to show")
    ap.add_argument("--port", type=int, default=PORT)
    args = ap.parse_args()
    url = serve(VoiceEvents.from_log(args.log), args.port, open_browser=True)
    print(f"showing {args.log} at {url}  (Ctrl+C to stop)")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
