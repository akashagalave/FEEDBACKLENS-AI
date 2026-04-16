import httpx
from langgraph.graph import StateGraph, END
from .state import AgentState
from .config import settings
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "../../../"))
from shared.logger import get_logger

logger = get_logger("orchestrator")


def is_retryable(exception):
    """Only retry on network/timeout errors, not on HTTP 4xx errors."""
    if isinstance(exception, httpx.TimeoutException):
        return True
    if isinstance(exception, httpx.ConnectError):
        return True
    if isinstance(exception, httpx.HTTPStatusError):
    
        return exception.response.status_code in (429, 503)
    return False


@retry(
    retry=retry_if_exception_type((httpx.TimeoutException, httpx.ConnectError)),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=8),
    reraise=True
)
async def _call_understanding_agent(query: str, company: str) -> dict:
    """
    Makes HTTP call to understanding agent with retry logic.
    Retries up to 3 times on timeout/connection errors.
    Wait: 1s → 2s → 4s (exponential backoff, max 8s).
    """
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            f"{settings.understanding_agent_url}/understand",
            json={"query": query, "company": company}
        )
        response.raise_for_status()
        return response.json()


async def understanding_node(state: AgentState) -> AgentState:
    try:
        data = await _call_understanding_agent(
            query=state["query"],
            company=state.get("company")
        )

        state["company"] = data.get("company", state.get("company", "unknown"))
        state["intent"]  = data.get("intent", "analyze")
        state["focus"]   = data.get("focus")
        logger.info(f"Understanding: company={state['company']} intent={state['intent']}")
        return state

    except Exception as e:
        logger.error(f"Understanding agent error after retries: {e}")
        state["error"] = str(e)
        return state

@retry(
    retry=retry_if_exception_type((httpx.TimeoutException, httpx.ConnectError)),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    reraise=True
)
async def _call_insight_agent(query: str, company: str, focus: str, top_k: int) -> dict:
    """
    Makes HTTP call to insight agent with retry logic.
    Insight agent has 120s timeout — heavy operation (Qdrant + BM25 + LLM).
    Retries up to 3 times. Wait: 1s → 2s → 4s (max 10s between retries).
    """
    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(
            f"{settings.insight_agent_url}/insights",
            json={
                "query":   query,
                "company": company,
                "focus":   focus,
                "top_k":   top_k
            }
        )
        response.raise_for_status()
        return response.json()


async def insight_node(state: AgentState) -> AgentState:
    try:
        data = await _call_insight_agent(
            query=state["query"],
            company=state["company"],
            focus=state.get("focus"),
            top_k=int(state.get("top_k") or 10)
        )

        state["top_issues"]      = data.get("top_issues", [])
        state["patterns"]        = data.get("patterns", [])
        state["sample_reviews"]  = data.get("sample_reviews", [])
        state["confidence_score"]= data.get("confidence_score")
        logger.info(f"Insights: {len(state['top_issues'])} issues found")
        return state

    except Exception as e:
        logger.error(f"Insight agent error after retries: {e}")
        state["error"] = str(e)
        return state

@retry(
    retry=retry_if_exception_type((httpx.TimeoutException, httpx.ConnectError)),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=8),
    reraise=True
)
async def _call_recommendation_agent(company: str, top_issues: list, patterns: list) -> dict:
    """
    Makes HTTP call to recommendation agent with retry logic.
    Retries up to 3 times on timeout/connection errors.
    """
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            f"{settings.recommendation_agent_url}/recommend",
            json={
                "company":    company,
                "top_issues": top_issues,
                "patterns":   patterns
            }
        )
        response.raise_for_status()
        return response.json()


async def recommendation_node(state: AgentState) -> AgentState:
    try:
        data = await _call_recommendation_agent(
            company=state["company"],
            top_issues=state.get("top_issues", []),
            patterns=state.get("patterns", [])
        )

        state["recommendations"] = data.get("recommendations", [])
        logger.info(f"Recommendations: {len(state['recommendations'])} generated")
        return state

    except Exception as e:
        logger.error(f"Recommendation agent error after retries: {e}")
        state["error"] = str(e)
        return state


def should_continue(state: AgentState) -> str:
    """
    Decides whether to call recommendation agent or end the workflow.
    
    End conditions:
    1. Any agent reported an error
    2. No issues found at all
    3. Insight agent returned the fallback 'no data' message
    """
    if state.get("error"):
        logger.warning(f"Ending workflow due to error: {state['error']}")
        return "end"

    if not state.get("top_issues"):
        logger.warning("No issues found — ending early")
        return "end"

    if state.get("top_issues") == ["No data found for this company"]:
        logger.warning("No data found — skipping recommendations")
        return "end"

    return "recommend"


def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("understanding",  understanding_node)
    graph.add_node("insight",        insight_node)
    graph.add_node("recommendation", recommendation_node)

    graph.set_entry_point("understanding")
    graph.add_edge("understanding", "insight")
    graph.add_conditional_edges(
        "insight",
        should_continue,
        {
            "recommend": "recommendation",
            "end":       END
        }
    )
    graph.add_edge("recommendation", END)

    return graph.compile()

workflow = build_graph()
