#!/usr/bin/env python3
"""
Run ATLAS with Claude Opus as the answer model, through Claude Code subagents.

Retrieval is deterministic code. Opus only reads the evidence it is given and
answers. Packets never contain gold answers, answer types, pillars or
supporting items.

  index      Build and cache retrieval indexes for every world (Mycelic
             BM25 + bge-small, and the naive_rag all-MiniLM per-turn baseline).
  packets    Write answer batches for three systems:
               mycelic_loop  round-1 evidence (whole sessions); the Verifier
                             may ask the server for more (round 2)
               naive_rag     top-10 turns, repo naive_rag prompt
               no_rag        full history, repo no_rag prompt (one packet per world)
             mycelic/naive_rag batches hold at most ONE probe per world, so
             an agent can never pile up evidence from the same world.
  serve      Local HTTP retrieval for the Verifier's follow-up queries:
               GET /retrieve?world=<w>&q=<query>&shown=<id,id>
  assemble   Turn agent outputs into result JSON files in the evaluator's
             format, so opus_judge.py and the comparison can score them.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "opus_run"
SETS = {"simple": ROOT / "simple_test_set", "complex": ROOT / "complex_test_set"}

NAIVE_SYSTEM = """You are a helpful assistant with access to retrieved documents.
Answer the question based on the documents provided below.
If the information is not available in the documents, say "Unknown" or "I don't know".

IMPORTANT: Give the shortest possible answer. Just provide the direct answer without any explanation or context.
For example, if asked "What is the user's dog's name?" just answer "Buddy" not "The user's dog's name is Buddy."

Retrieved Documents:
{context}"""

NORAG_SYSTEM = """You are a helpful assistant with access to the user's full conversation history.
Answer the question based on the conversation history provided below.
If the information is not available in the conversations, say "Unknown" or "I don't know".

IMPORTANT: Give the shortest possible answer. Just provide the direct answer without any explanation or context.
For example, if asked "What is the user's dog's name?" just answer "Buddy" not "The user's dog's name is Buddy."

=== CONVERSATION HISTORY ===
{context}
=== END OF HISTORY ==="""


def worlds():
    for ts, d in SETS.items():
        for p in sorted(d.glob("*.json")):
            yield ts, p


def wid(ts: str, p: Path) -> str:
    return f"{ts}/{p.stem}"


def loop_adapter():
    os.environ.setdefault("ATLAS_LLM_BACKEND", "openai_compat")  # adapter is only used for retrieval here
    from benchmark.evaluation.mycelic_loop import MycelicLoopAdapter

    return MycelicLoopAdapter(user_id="opus_run")


def load_index(ad, w: str):
    import numpy as np

    d = OUT / "index" / w
    meta = json.loads((d / "meta.json").read_text())
    ad.clear()
    ad.sessions = meta["sessions"]
    ad.turn_docs = [tuple(x) for x in meta["turn_docs"]]
    from rank_bm25 import BM25Okapi
    from benchmark.evaluation.mycelic_loop import _tokens

    ad._bm25 = BM25Okapi([_tokens(t[2]) or ["_"] for t in ad.turn_docs])
    ad._emb = np.load(d / "bge.npy")
    return meta


def cmd_index(a):
    import numpy as np
    from fastembed import TextEmbedding

    ad = loop_adapter()
    mini = TextEmbedding("sentence-transformers/all-MiniLM-L6-v2")
    for ts, p in worlds():
        d = OUT / "index" / wid(ts, p)
        if (d / "mini.npy").exists():
            continue
        d.mkdir(parents=True, exist_ok=True)
        bench = json.loads(p.read_text())
        ad.ingest_sessions(bench["sessions"])
        docs, owner = [], []
        for s in bench["sessions"]:
            for i, t in enumerate(s["turns"]):
                docs.append(f"[Session {s['id']}, Turn {i}] {t['speaker']}: {t['text']}")
        E = np.array(list(mini.embed(docs)), dtype=np.float32)
        E /= np.linalg.norm(E, axis=1, keepdims=True)
        np.save(d / "bge.npy", ad._emb)
        np.save(d / "mini.npy", E)
        (d / "meta.json").write_text(json.dumps({"sessions": ad.sessions, "turn_docs": ad.turn_docs, "naive_docs": docs}))
        print(f"[index] {wid(ts, p)}", flush=True)


def cmd_packets(a):
    import numpy as np
    from fastembed import TextEmbedding

    ad = loop_adapter()
    mini = TextEmbedding("sentence-transformers/all-MiniLM-L6-v2")
    probes: Dict[str, List[Dict[str, Any]]] = {"mycelic_loop": [], "naive_rag": []}
    norag: List[Dict[str, Any]] = []
    for ts, p in worlds():
        w = wid(ts, p)
        bench = json.loads(p.read_text())
        meta = load_index(ad, w)
        M = np.load(OUT / "index" / w / "mini.npy")
        questions = [{"pid": f"{w}|{i}", "question": pr["question"]} for i, pr in enumerate(bench["probes"])]
        for q in questions:
            sids = ad._retrieve_sessions([q["question"]])
            probes["mycelic_loop"].append({**q, "world": w, "shown": sids, "evidence": ad._render(sids)})
            qv = np.array(list(mini.embed([q["question"]]))[0], dtype=np.float32)
            qv /= np.linalg.norm(qv)
            top = np.argsort(-(M @ qv))[:10]
            ctx = "\n\n".join(meta["naive_docs"][i] for i in top)
            probes["naive_rag"].append({**q, "world": w, "system": NAIVE_SYSTEM.format(context=ctx)})
        full = "\n\n---\n\n".join(ad.format_session_for_ingestion(s) for s in bench["sessions"])
        norag.append({"world": w, "system": NORAG_SYSTEM.format(context=full), "questions": questions})
        print(f"[packets] {w}", flush=True)
    rng = random.Random(0)
    bdir = OUT / "batches"
    bdir.mkdir(parents=True, exist_ok=True)
    for name, items in probes.items():
        # Round-robin by world: batch k gets the k-th probe of every world, so
        # no batch holds two probes from the same world.
        by_world: Dict[str, List] = {}
        for it in items:
            by_world.setdefault(it["world"], []).append(it)
        layers: List[List] = []
        for its in by_world.values():
            for k, it in enumerate(its):
                while len(layers) <= k:
                    layers.append([])
                layers[k].append(it)
        size = a.mycelic_batch if name == "mycelic_loop" else a.naive_batch
        batches = []
        for layer in layers:
            rng.shuffle(layer)
            batches += [layer[i : i + size] for i in range(0, len(layer), size)]
        for i, b in enumerate(batches):
            (bdir / f"{name}_{i:02d}.jsonl").write_text("".join(json.dumps(x) + "\n" for x in b))
        print(f"{name}: {len(items)} probes in {len(batches)} batches")
    rng.shuffle(norag)
    per = a.norag_worlds
    for i in range(0, len(norag), per):
        (bdir / f"no_rag_{i // per:02d}.jsonl").write_text("".join(json.dumps(x) + "\n" for x in norag[i : i + per]))
    print(f"no_rag: {len(norag)} worlds in {-(-len(norag) // per)} batches")


def cmd_serve(a):
    ad = loop_adapter()
    cache: Dict[str, Any] = {}

    class H(BaseHTTPRequestHandler):
        def do_GET(self):
            u = urlparse(self.path)
            q = {k: v[0] for k, v in parse_qs(u.query).items()}
            try:
                w = q["world"]
                if cache.get("w") != w:
                    load_index(ad, w)
                    cache["w"] = w
                shown = [s for s in q.get("shown", "").split(",") if s]
                new = [s for s in ad._retrieve_sessions([q["q"]]) if s not in shown][: a.extra_sessions]
                body = json.dumps({"new_sessions": new, "evidence": ad._render(new) if new else ""})
                self.send_response(200)
            except Exception as e:  # noqa: BLE001
                body = json.dumps({"error": str(e)})
                self.send_response(400)
            self.send_header("content-type", "application/json")
            self.end_headers()
            self.wfile.write(body.encode())
            with open(OUT / "server_log.jsonl", "a") as fh:
                fh.write(json.dumps({"query": q}) + "\n")

        def log_message(self, *args):
            pass

    ThreadingHTTPServer.daemon_threads = True
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), H)
    # One shared adapter: serialize requests.
    srv.socket.listen(64)
    print(f"serving on {a.port}", flush=True)
    import threading

    lock = threading.Lock()
    orig = H.do_GET

    def locked(self):
        with lock:
            orig(self)

    H.do_GET = locked
    srv.serve_forever()


def cmd_assemble(a):
    from benchmark.evaluation.metrics import aggregate_scores

    answers: Dict[str, Dict[str, str]] = {}
    for f in sorted((OUT / "out").glob("*.json")):
        system = f.stem.rsplit("_", 1)[0]
        answers.setdefault(system, {}).update(json.loads(f.read_text()))
    for system, ans in answers.items():
        for ts, d in SETS.items():
            detailed, missing = [], 0
            for p in sorted(d.glob("*.json")):
                bench = json.loads(p.read_text())
                prs = []
                for i, pr in enumerate(bench["probes"]):
                    pid = f"{ts}/{p.stem}|{i}"
                    if pid not in ans:
                        missing += 1
                    prs.append({
                        "probe_id": pr["id"], "pillar": pr["pillar"], "subpillar": pr["subpillar"],
                        "question": pr["question"], "answer_type": pr["answer_type"],
                        "gold_answer": pr["gold_answer"]["text"], "predicted_answer": str(ans.get(pid, "")),
                        "retrieved_context": [], "score": 0, "match_type": "pending_opus_judge",
                        "search_latency_ms": 0, "generation_latency_ms": 0, "total_latency_ms": 0,
                    })
                detailed.append({"benchmark_file": str(p.relative_to(ROOT)), "adapter_type": system, "num_sessions": len(bench["sessions"]),
                                 "num_probes": len(prs), "probe_results": prs, "summary": {}})
            rep = {"adapter_type": system, "answer_model": "claude-opus (Claude Code subagents)", "missing_answers": missing,
                   "overall_summary": aggregate_scores([{"score": 0, "answer_type": x["answer_type"], "pillar": x["pillar"]} for b in detailed for x in b["probe_results"]]),
                   "detailed_results": detailed}
            path = OUT / "results" / f"results_opus_{system}_{ts}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(rep, indent=1))
            print(f"{path.name}: missing {missing}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("index")
    p = sub.add_parser("packets")
    p.add_argument("--mycelic-batch", type=int, default=35)
    p.add_argument("--naive-batch", type=int, default=100)
    p.add_argument("--norag-worlds", type=int, default=10)
    s = sub.add_parser("serve")
    s.add_argument("--port", type=int, default=8799)
    s.add_argument("--extra-sessions", type=int, default=4)
    sub.add_parser("assemble")
    a = ap.parse_args()
    {"index": cmd_index, "packets": cmd_packets, "serve": cmd_serve, "assemble": cmd_assemble}[a.cmd](a)


if __name__ == "__main__":
    main()
