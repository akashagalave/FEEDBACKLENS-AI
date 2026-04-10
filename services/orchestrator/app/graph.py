import httpx
from langgraph.graph import StateGraph, END
from .state import AgentState
from .config import settings
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "../../../"))
from shared.logger import get_logger

logger = get_logger("orchestrator")


# ─── NODE 1: UNDERSTANDING ────────────────────────────────────
async def understanding_node(state: AgentState) -> AgentState:
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{settings.understanding_agent_url}/understand",
                json={"query": state["query"], "company": state.get("company")}
            )
            response.raise_for_status()
            data = response.json()

        state["company"] = data.get("company", state.get("company", "unknown"))
        state["intent"] = data.get("intent", "analyze")
        state["focus"] = data.get("focus")
        logger.info(f"Understanding: company={state['company']} intent={state['intent']}")
        return state

    except Exception as e:
        logger.error(f"Understanding agent error: {e}")
        state["error"] = str(e)
        return state


# ─── NODE 2: INSIGHT (RAG) ────────────────────────────────────
async def insight_node(state: AgentState) -> AgentState:
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{settings.insight_agent_url}/insights",
                json={
                    "query": state["query"],
                    "company": state["company"],
                    "focus": state.get("focus"),
                    "top_k": state.get("top_k", 10)
                }
            )
            response.raise_for_status()
            data = response.json()

        state["top_issues"] = data.get("top_issues", [])
        state["patterns"] = data.get("patterns", [])
        state["sample_reviews"] = data.get("sample_reviews", [])
        state["confidence_score"] = data.get("confidence_score")
        logger.info(f"Insights: {len(state['top_issues'])} issues found")
        return state

    except Exception as e:
        logger.error(f"Insight agent error: {e}")
        state["error"] = str(e)
        return state


# ─── NODE 3: RECOMMENDATION ───────────────────────────────────
async def recommendation_node(state: AgentState) -> AgentState:
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{settings.recommendation_agent_url}/recommend",
                json={
                    "company": state["company"],
                    "top_issues": state.get("top_issues", []),
                    "patterns": state.get("patterns", [])
                }
            )
            response.raise_for_status()
            data = response.json()

        state["recommendations"] = data.get("recommendations", [])
        logger.info(f"Recommendations: {len(state['recommendations'])} generated")
        return state

    except Exception as e:
        logger.error(f"Recommendation agent error: {e}")
        state["error"] = str(e)
        return state


# ─── CONDITIONAL ROUTING ──────────────────────────────────────
def should_continue(state: AgentState) -> str:
    if state.get("error"):
        return "end"
    if not state.get("top_issues"):
        logger.warning("No issues found — ending early")
        return "end"
    return "recommend"


# ─── BUILD GRAPH ──────────────────────────────────────────────
def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("understanding", understanding_node)
    graph.add_node("insight", insight_node)
    graph.add_node("recommendation", recommendation_node)

    graph.set_entry_point("understanding")
    graph.add_edge("understanding", "insight")
    graph.add_conditional_edges(
        "insight",
        should_continue,
        {
            "recommend": "recommendation",
            "end": END
        }
    )
    graph.add_edge("recommendation", END)

    return graph.compile()


workflow = build_graph()