"""Shared state for the Mohafiz LangGraph agent."""
from typing import Optional, TypedDict


class AgentState(TypedDict, total=False):
    # input
    question: str
    # router output
    route: str                 # retrieve | live | both | emergency | refuse
    domain: str                # disaster | agriculture | policy | other
    tool: Optional[str]        # weather | earthquake | alerts | sitreps | None
    language: str              # en | ur
    # retrieval
    contexts: list             # list[dict] retrieved chunks
    docs_relevant: bool        # CRAG grade
    rewrites: int              # query-rewrite attempts used
    query_used: str            # current (possibly rewritten) retrieval query
    # live tools
    tool_data: dict
    # generation
    answer: str
    grounded: Optional[bool]   # Self-RAG groundedness verdict
    regens: int                # regeneration attempts used
    abstained: bool
    degraded: bool             # budget exhausted -> retrieval-only degradation
