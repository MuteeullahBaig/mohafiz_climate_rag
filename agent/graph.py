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

from agent import llm, prompts
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


def generate(state: AgentState) -> dict:
    prompt = prompts.fill(
        prompts.GENERATE, lang_name=LANG_NAME.get(state.get("language", "en"), "English"),
        context_block=_context_block(state), question=state["question"])
    ans = llm.chat([{"role": "user", "content": prompt}], model=llm.BIG_MODEL, max_tokens=512)
    return {"answer": ans}


def groundedness(state: AgentState) -> dict:
    out = llm.chat_json([{"role": "user", "content": prompts.fill(
        prompts.GROUNDEDNESS, context_block=_context_block(state), answer=state.get("answer", ""))}])
    return {"grounded": bool(out.get("grounded"))}


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


if __name__ == "__main__":
    app = build_graph()
    q = " ".join(sys.argv[1:]) or "What should families do to prepare before a flood?"
    result = app.invoke({"question": q})
    print(f"\nroute={result.get('route')} lang={result.get('language')} "
          f"grounded={result.get('grounded')} abstained={result.get('abstained', False)}")
    print("-" * 70)
    print(result.get("answer", "(no answer)"))
