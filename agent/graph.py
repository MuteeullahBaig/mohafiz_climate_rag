"""Mohafiz agentic RAG graph (LangGraph).

Flow (roadmap 1.6):
  classify -> {emergency | refuse | retrieve | live | both}
  retrieve -> grade -> (rewrite -> retrieve) | continue
  continue -> live (if route == both) -> generate
  generate -> groundedness -> (regenerate) | (abstain) | END

Cheap 8B model runs classify/grade/groundedness; 70B runs generate only.
Retrieval uses the W2 hybrid + reranker path.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from langgraph.graph import END, START, StateGraph

from agent import llm, prompts, quota
from agent.helplines import emergency_block
from agent.state import AgentState
from agent.tools import live_data

LANG_NAME = {"en": "English", "ur": "Urdu"}
MAX_REWRITES = 1
MAX_REGENS = 1

_retriever = None


def _get_retriever():
    global _retriever
    if _retriever is None:
        from retrieval.search import Retriever
        _retriever = Retriever(mode="hybrid", rerank=True)
    return _retriever


# ─── nodes ────────────────────────────────────────────────────────────────────
def classify(state: AgentState) -> dict:
    out = llm.chat_json([{"role": "user", "content": prompts.fill(prompts.ROUTER, question=state["question"])}])
    route = out.get("route", "retrieve")
    if route not in ("emergency", "refuse", "retrieve", "live", "both"):
        route = "retrieve"
    domain = out.get("domain", "other")
    if domain not in ("disaster", "agriculture", "policy", "other"):
        domain = "other"
    return {"route": route, "domain": domain, "tool": out.get("tool"),
            "language": "ur" if out.get("language") == "ur" else "en",
            "query_used": state["question"], "rewrites": 0, "regens": 0}


def emergency(state: AgentState) -> dict:
    ur = state.get("language") == "ur"
    header = ("Fauri madad ke liye abhi call karein:" if ur
              else "For immediate help, call now:")
    tail = ("Mehfooz jagah par jayen aur maqami DDMA/intizamia ki hidayat par amal karein."
            if ur else
            "Move to safety and follow instructions from your local DDMA/district authorities.")
    return {"answer": f"{header}\n\n{emergency_block()}\n\n{tail}", "grounded": True}


def refuse(state: AgentState) -> dict:
    ur = state.get("language") == "ur"
    msg = ("Main Mohafiz hoon — Pakistan ki aafat, mausam aur climate se mutalliq madad ke liye. "
           "Is se hat kar sawal ka jawab nahi de sakta." if ur else
           "I'm Mohafiz — I help with Pakistan's disaster preparedness, weather, and climate "
           "topics, so I can't help with that request.")
    return {"answer": msg, "grounded": True}


def retrieve(state: AgentState) -> dict:
    hits = _get_retriever().search(state["query_used"], k=5)
    return {"contexts": hits}


def grade(state: AgentState) -> dict:
    ctx = "\n\n".join(f"- {c['text'][:300]}" for c in state.get("contexts", []))
    out = llm.chat_json([{"role": "user", "content": prompts.fill(
        prompts.GRADE_DOCS, question=state["question"], contexts=ctx or "(none)")}])
    return {"docs_relevant": bool(out.get("relevant"))}


def rewrite(state: AgentState) -> dict:
    q = llm.chat([{"role": "user", "content": prompts.fill(prompts.REWRITE, question=state["question"])}],
                 max_tokens=128).strip()
    return {"query_used": q, "rewrites": state.get("rewrites", 0) + 1}


def live(state: AgentState) -> dict:
    tool = state.get("tool")
    try:
        if tool == "weather":
            data = live_data.get_weather("lahore")  # city extraction is a later refinement
        elif tool == "earthquake":
            data = live_data.get_earthquakes()
        elif tool == "alerts":
            data = live_data.get_gdacs_alerts()
        elif tool == "sitreps":
            data = live_data.get_sitreps()
        else:
            data = {"note": "no specific live tool selected"}
    except Exception as e:
        data = {"error": f"{type(e).__name__}: {e}"}
    return {"tool_data": data}


def _context_block(state: AgentState) -> str:
    parts = []
    for c in state.get("contexts", []):
        parts.append(f"[{c['doc_id']} p.{c['pages']}] {c['text']}")
    if state.get("tool_data"):
        import json
        parts.append("LIVE DATA:\n" + json.dumps(state["tool_data"], ensure_ascii=False)[:1500])
    return "\n\n".join(parts) if parts else "(no context available)"


def _retrieval_only(state: AgentState) -> str:
    ur = state.get("language") == "ur"
    note = ("Demo ki rozana had puri ho gayi. Neeche mutalliqa iqtebaasat hain:" if ur else
            "The demo's daily answer limit has been reached — here are the most relevant "
            "source passages:")
    cites = "\n".join(f"- [{c['doc_id']} p.{c['pages']}] {c['text'][:200]}..."
                      for c in state.get("contexts", [])[:3])
    return f"{note}\n\n{cites}" if cites else (
        "The demo's daily answer limit has been reached; please try again tomorrow.")


def generate(state: AgentState) -> dict:
    # Budget guard: the 70B is the scarce resource. When the daily cap is hit, degrade
    # to retrieval-only (cited passages, no generation) instead of erroring.
    if not quota.try_consume_budget(1):
        return {"answer": _retrieval_only(state), "grounded": True, "degraded": True}
    prompt = prompts.fill(
        prompts.GENERATE, lang_name=LANG_NAME.get(state.get("language", "en"), "English"),
        context_block=_context_block(state), question=state["question"])
    ans = llm.chat([{"role": "user", "content": prompt}], model=llm.BIG_MODEL, max_tokens=512)
    return {"answer": ans}


def groundedness(state: AgentState) -> dict:
    # Secondary safety net (generation already enforces cite-only). The 8B judge is noisy,
    # so fail OPEN: only an explicit False counts as ungrounded; parse errors / missing key
    # default to grounded, avoiding false abstentions on correct answers.
    try:
        out = llm.chat_json([{"role": "user", "content": prompts.fill(
            prompts.GROUNDEDNESS, context_block=_context_block(state),
            answer=state.get("answer", ""))}])
        val = out.get("grounded", True)
        return {"grounded": val if isinstance(val, bool) else True}
    except Exception:
        return {"grounded": True}


def abstain(state: AgentState) -> dict:
    ur = state.get("language") == "ur"
    note = ("Mujhe mustanad maloomat nahi mili. Neeche mutalliqa iqtebaasat hain:" if ur else
            "I couldn't ground a confident answer. Here are the most relevant source passages:")
    cites = "\n".join(f"- [{c['doc_id']} p.{c['pages']}] {c['text'][:150]}..."
                      for c in state.get("contexts", [])[:3])
    return {"answer": f"{note}\n\n{cites}" if cites else note, "abstained": True}


# ─── edges ────────────────────────────────────────────────────────────────────
def route_from_classify(state: AgentState) -> str:
    return {"emergency": "emergency", "refuse": "refuse", "live": "live"}.get(
        state["route"], "retrieve")  # retrieve and both both start at retrieve


def after_grade(state: AgentState) -> str:
    if not state.get("docs_relevant") and state.get("rewrites", 0) < MAX_REWRITES:
        return "rewrite"
    return "live" if state["route"] == "both" else "generate"


def after_groundedness(state: AgentState) -> str:
    if state.get("grounded"):
        return END
    if state.get("regens", 0) < MAX_REGENS:
        return "regenerate"
    return "abstain"


def regenerate(state: AgentState) -> dict:
    return {"regens": state.get("regens", 0) + 1}


def build_graph():
    g = StateGraph(AgentState)
    for name, fn in [("classify", classify), ("emergency", emergency), ("refuse", refuse),
                     ("retrieve", retrieve), ("grade", grade), ("rewrite", rewrite),
                     ("live", live), ("generate", generate), ("groundedness", groundedness),
                     ("abstain", abstain), ("regenerate", regenerate)]:
        g.add_node(name, fn)

    g.add_edge(START, "classify")
    g.add_conditional_edges("classify", route_from_classify,
                            ["emergency", "refuse", "live", "retrieve"])
    g.add_edge("emergency", END)
    g.add_edge("refuse", END)
    g.add_edge("retrieve", "grade")
    g.add_conditional_edges("grade", after_grade, ["rewrite", "live", "generate"])
    g.add_edge("rewrite", "retrieve")
    g.add_edge("live", "generate")
    g.add_edge("generate", "groundedness")
    g.add_conditional_edges("groundedness", after_groundedness,
                            {"regenerate": "regenerate", "abstain": "abstain", END: END})
    g.add_edge("regenerate", "generate")
    g.add_edge("abstain", END)
    return g.compile()


# ─── public entry point (used by API + UI) ─────────────────────────────────────
_app = None


def get_app():
    global _app
    if _app is None:
        _app = build_graph()
    return _app


def _citations(result: dict) -> list[str]:
    seen = []
    for c in result.get("contexts", [])[:5]:
        cid = f"{c['doc_id']} p.{c['pages']}"
        if cid not in seen:
            seen.append(cid)
    return seen


STAGE_MESSAGE = {
    "classify": "Understanding your question…",
    "retrieve": "Searching official documents…",
    "grade": "Checking that the sources are relevant…",
    "rewrite": "Refining the search…",
    "live": "Fetching live data…",
    "generate": "Composing a grounded answer…",
    "groundedness": "Verifying the answer against its sources…",
    "emergency": "Preparing emergency guidance…",
    "refuse": "Checking scope…",
    "abstain": "No confident answer — gathering source passages…",
}


def _build_result(result: dict, cached: bool = False) -> dict:
    return {
        "answer": result.get("answer", ""),
        "route": result.get("route"),
        "domain": result.get("domain"),
        "language": result.get("language", "en"),
        "grounded": result.get("grounded"),
        "abstained": result.get("abstained", False),
        "degraded": result.get("degraded", False),
        "citations": _citations(result),
        "cached": cached,
    }


def _maybe_cache(question: str, out: dict):
    # Cache only stable, successful answers: skip live/both (time-sensitive), skip
    # degraded (budget) and abstained (low-quality — may succeed on a later retry).
    if (out["route"] in ("retrieve", "refuse", "emergency")
            and not out["degraded"] and not out["abstained"]):
        quota.cache_put(question, {k: v for k, v in out.items() if k != "cached"})


def run_agent(question: str) -> dict:
    """Cache-wrapped agent call returning a clean, serializable result."""
    question = (question or "").strip()
    if not question:
        return {"answer": "Please enter a question.", "route": None, "cached": False}
    cached = quota.cache_get(question)
    if cached:
        return {**cached, "cached": True}
    result = get_app().invoke({"question": question})
    out = _build_result(result)
    _maybe_cache(question, out)
    return out


def stream_agent(question):
    """Yield {'type': 'stage'|'final', ...} events — real per-node progress via
    LangGraph .stream(), used by the SSE endpoint and the Gradio UI."""
    question = (question or "").strip()
    if not question:
        yield {"type": "final", **_build_result({"answer": "Please enter a question."})}
        return
    cached = quota.cache_get(question)
    if cached:
        yield {"type": "final", **cached, "cached": True}
        return
    final_state = {}
    for update in get_app().stream({"question": question}):
        for node, delta in update.items():
            if delta:
                final_state.update(delta)
            yield {"type": "stage", "node": node,
                   "message": STAGE_MESSAGE.get(node, node)}
    out = _build_result(final_state)
    _maybe_cache(question, out)
    yield {"type": "final", **out}


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "What should families do to prepare before a flood?"
    r = run_agent(q)
    print(f"\nroute={r.get('route')} domain={r.get('domain')} lang={r.get('language')} "
          f"grounded={r.get('grounded')} cached={r.get('cached')} degraded={r.get('degraded')}")
    print("citations:", r.get("citations"))
    print("-" * 70)
    print(r.get("answer", "(no answer)"))
