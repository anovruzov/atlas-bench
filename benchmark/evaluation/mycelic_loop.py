"""
Mycelic three-agent memory loop for ATLAS.

  Agent 1  Retriever  hybrid BM25 + dense retrieval over conversation turns,
                      fused with Reciprocal Rank Fusion and lifted to whole
                      sessions (kept in chronological order with timestamps).
  Agent 2  Answerer   reads the retrieved sessions and returns a short answer
                      plus the session ids and quote it relied on.
  Agent 3  Verifier   checks the answer against the quoted evidence and the
                      question (latest value, ordering, missing information).
                      It accepts, corrects, or sends the Retriever a new query.
                      The loop runs at most `max_rounds` times.

Honesty constraints (enforced by construction):
  * The adapter only ever sees the sessions and the question text, exactly
    like every other adapter. It never sees gold answers, answer_type,
    pillar, or supporting_items.
  * Scoring is untouched: the evaluator's own judge in metrics.py scores it.
  * The answer model is configurable. A controlled comparison needs the same
    answer model and judge for each arm.
  * Every probe's full trace (queries, sessions used, drafts, verdicts, context
    size) is kept in `self.traces` and written into the results JSON.
"""

from __future__ import annotations

import json
import math
import os
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import requests

from .memory_adapters import MemoryAdapter, call_openrouter, OPENROUTER_MODEL

try:
    from rank_bm25 import BM25Okapi

    BM25_AVAILABLE = True
except ImportError:  # pragma: no cover
    BM25_AVAILABLE = False

try:
    from fastembed import TextEmbedding
    import numpy as np

    FASTEMBED_AVAILABLE = True
except ImportError:  # pragma: no cover
    FASTEMBED_AVAILABLE = False


DEFAULT_CONFIG: Dict[str, Any] = {
    "embed_model": "BAAI/bge-small-en-v1.5",
    "rrf_k": 60,
    "turn_candidates": 40,  # per channel, before fusion
    "max_sessions": 8,  # whole sessions handed to the Answerer
    "max_rounds": 2,  # Retriever -> Answerer -> Verifier iterations
    "temperature": 0.0,
    "max_tokens": 500,
}

_STOP = set(
    "a an the of to in on at for by with from and or but is are was were be been being "
    "do does did what which who whom whose when where why how that this these those it its "
    "i me my we our you your he she they them their user assistant about as into than then "
    "currently current".split()
)


def _tokens(text: str) -> List[str]:
    return [t for t in re.findall(r"[a-z0-9][a-z0-9.\-_]*", text.lower()) if t not in _STOP]


def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    if fence:
        text = fence.group(1)
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        return None
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None


ANSWERER_SYSTEM = """You are the Answerer in a memory system. You receive excerpts of past conversations between a user and an assistant, shown in chronological order (higher t = later).

Answer the question using ONLY these conversations.
Rules:
- If facts changed over time, the latest statement wins unless the question asks about an earlier time.
- For "before/after" or ordering questions, compare the t values of the sessions involved.
- If the conversations do not contain the information, the answer is "Unknown". Do not guess.
- The answer must be as short as possible: a name, number, phrase, or yes/no. No explanation.

Reply with JSON only:
{"answer": "<short answer>", "sessions": ["<session id>", ...], "quote": "<the exact sentence(s) that support the answer, or empty>"}"""

VERIFIER_SYSTEM = """You are the Verifier in a memory system. Another agent drafted an answer from past conversations. Check it strictly against the conversations shown.

Check:
1. Is the draft directly supported by the conversations (not just plausible)?
2. If the value changed over time, is it the latest value (or the value at the time the question asks about)?
3. Does it answer exactly what was asked (right entity, right attribute)?
4. If the conversations do not contain the answer, the correct answer is "Unknown".

Reply with JSON only:
{"verdict": "accept" | "revise" | "retrieve", "final_answer": "<short answer>", "search_query": "<new search query if verdict is retrieve, else empty>", "reason": "<one short sentence>"}
- accept: the draft is correct; final_answer = draft.
- revise: you can correct it from the conversations shown; give the corrected short answer.
- retrieve: the needed information may be in other conversations not shown; give a better search query. final_answer = your best answer so far.
Keep final_answer as short as possible: a name, number, phrase, or yes/no."""


class MycelicLoopAdapter(MemoryAdapter):
    """Three-agent Retriever -> Answerer -> Verifier loop."""

    def __init__(self, user_id: str = "benchmark_user", config: Optional[Dict[str, Any]] = None, **_: Any):
        super().__init__(user_id)
        if not (BM25_AVAILABLE and FASTEMBED_AVAILABLE):
            raise ImportError("mycelic_loop needs: pip install rank_bm25 fastembed")
        self.config = dict(DEFAULT_CONFIG)
        self.config.update(config or {})
        self.backend = os.environ.get("ATLAS_LLM_BACKEND", "openrouter")
        self.model = (
            os.environ.get("ATLAS_LLM_MODEL", "local")
            if self.backend == "openai_compat"
            else OPENROUTER_MODEL
        )
        self._embedder = TextEmbedding(self.config["embed_model"])
        self.traces: List[Dict[str, Any]] = []
        self.clear()

    # ------------------------------------------------------------------ LLM
    def _llm(self, system: str, user: str) -> str:
        t, mt = self.config["temperature"], self.config["max_tokens"]
        if self.backend == "openai_compat":
            base = os.environ.get("ATLAS_LLM_BASE_URL", "http://127.0.0.1:8080/v1").rstrip("/")
            r = requests.post(
                f"{base}/chat/completions",
                json={
                    "model": self.model,
                    "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                    "temperature": t,
                    "max_tokens": mt,
                },
                timeout=600,
            )
            r.raise_for_status()
            return (r.json()["choices"][0]["message"]["content"] or "").strip()
        return call_openrouter(user, system, max_tokens=mt, temperature=t)

    # ------------------------------------------------------------ ingestion
    def clear(self) -> None:
        self.sessions: Dict[str, Dict[str, Any]] = {}
        self.turn_docs: List[Tuple[str, int, str]] = []  # (session_id, turn_idx, text)
        self._bm25 = None
        self._emb = None

    def ingest_sessions(self, sessions: List[Dict[str, Any]]) -> None:
        self.clear()
        for order, s in enumerate(sorted(sessions, key=lambda x: x.get("timestamp", 0))):
            sid = str(s.get("id", f"s{order}"))
            self.sessions[sid] = {"t": s.get("timestamp", order), "turns": s.get("turns", [])}
            turns = s.get("turns", [])
            for i, turn in enumerate(turns):
                # Each unit is a turn plus its neighbour, so answers that span a
                # question/reply pair stay retrievable.
                prev = turns[i - 1]["text"] if i > 0 else ""
                text = f"{turn.get('speaker', 'user')}: {turn.get('text', '')}"
                if prev:
                    text = f"{turns[i - 1].get('speaker', 'user')}: {prev}\n{text}"
                self.turn_docs.append((sid, i, text))
        corpus = [d[2] for d in self.turn_docs]
        self._bm25 = BM25Okapi([_tokens(c) or ["_"] for c in corpus])
        self._emb = np.array(list(self._embedder.embed(corpus)), dtype=np.float32)
        self._emb /= np.linalg.norm(self._emb, axis=1, keepdims=True) + 1e-9

    # ------------------------------------------------------ Agent 1: Retriever
    def _retrieve_sessions(self, queries: List[str]) -> List[str]:
        k, n = self.config["rrf_k"], self.config["turn_candidates"]
        session_score: Dict[str, float] = defaultdict(float)
        for q in queries:
            fused: Dict[int, float] = defaultdict(float)
            bm = self._bm25.get_scores(_tokens(q) or ["_"])
            for rank, idx in enumerate(np.argsort(-bm)[:n]):
                if bm[idx] > 0:
                    fused[int(idx)] += 1.0 / (k + rank + 1)
            qv = np.array(list(self._embedder.query_embed([q]))[0], dtype=np.float32)
            qv /= np.linalg.norm(qv) + 1e-9
            cos = self._emb @ qv
            for rank, idx in enumerate(np.argsort(-cos)[:n]):
                fused[int(idx)] += 1.0 / (k + rank + 1)
            # Lift turn scores to sessions: best turn plus a small bonus for
            # additional matching turns in the same session.
            per_session: Dict[str, List[float]] = defaultdict(list)
            for idx, sc in fused.items():
                per_session[self.turn_docs[idx][0]].append(sc)
            for sid, scs in per_session.items():
                scs.sort(reverse=True)
                session_score[sid] += scs[0] + 0.25 * sum(scs[1:3])
        ranked = sorted(session_score, key=lambda s: -session_score[s])
        return ranked[: self.config["max_sessions"]]

    def _render(self, session_ids: List[str]) -> str:
        blocks = []
        for sid in sorted(session_ids, key=lambda s: self.sessions[s]["t"]):
            s = self.sessions[sid]
            lines = [f"[Session {sid} | t={s['t']}]"]
            lines += [f"{t.get('speaker', 'user')}: {t.get('text', '')}" for t in s["turns"]]
            blocks.append("\n".join(lines))
        return "\n\n".join(blocks)

    def search(self, query: str, top_k: int = 10) -> List[str]:
        # The evaluator logs this as retrieved context; the loop itself runs in
        # generate_answer and may retrieve again.
        sids = self._retrieve_sessions([query])
        return [self._render([s]) for s in sids][:top_k]

    # ---------------------------------------- Agents 2+3: answer and verify
    def generate_answer(self, question: str, context: List[str]) -> str:
        queries = [question]
        shown: List[str] = []
        trace: Dict[str, Any] = {"question": question, "model": self.model, "rounds": []}
        final = "Unknown"
        for rnd in range(self.config["max_rounds"]):
            for sid in self._retrieve_sessions(queries):
                if sid not in shown:
                    shown.append(sid)
            evidence = self._render(shown)
            draft_raw = self._llm(ANSWERER_SYSTEM, f"Conversations:\n{evidence}\n\nQuestion: {question}")
            draft = _extract_json(draft_raw) or {"answer": draft_raw.strip()[:200], "sessions": [], "quote": ""}
            ver_raw = self._llm(
                VERIFIER_SYSTEM,
                f"Conversations:\n{evidence}\n\nQuestion: {question}\n\n"
                f"Draft answer: {draft.get('answer', '')}\n"
                f"Cited sessions: {draft.get('sessions', [])}\nCited quote: {draft.get('quote', '')}",
            )
            ver = _extract_json(ver_raw) or {"verdict": "accept", "final_answer": draft.get("answer", "")}
            final = str(ver.get("final_answer") or draft.get("answer") or "Unknown").strip()
            trace["rounds"].append(
                {
                    "queries": list(queries),
                    "sessions_shown": list(shown),
                    "context_chars": len(evidence),
                    "draft": draft,
                    "verifier": ver,
                }
            )
            nq = str(ver.get("search_query") or "").strip()
            if ver.get("verdict") != "retrieve" or not nq or nq in queries:
                break
            queries.append(nq)
        trace["final_answer"] = final
        self.traces.append(trace)
        return final
