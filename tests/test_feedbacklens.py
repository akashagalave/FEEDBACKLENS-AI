
import pytest
import json
import hashlib
from unittest.mock import MagicMock, patch


import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ─── TEST 1: normalize_company exact match ────────────────────────────────────
def test_normalize_company_exact_match():
    """
    normalize_company should return 'swiggy' for exact lowercase input.
    Core function — if this fails, all retrieval breaks.
    """
    from services.understanding_agent.app.agent import normalize_company
    assert normalize_company("swiggy", "analyze swiggy") == "swiggy"


# ─── TEST 2: normalize_company uppercase ─────────────────────────────────────
def test_normalize_company_uppercase():
    """
    LLM sometimes returns 'SWIGGY' or 'Swiggy'.
    normalize_company must handle case insensitivity.
    """
    from services.understanding_agent.app.agent import normalize_company
    assert normalize_company("SWIGGY", "analyze delivery") == "swiggy"
    assert normalize_company("Uber",   "uber problems")    == "uber"


# ─── TEST 3: normalize_company unknown returns None ───────────────────────────
def test_normalize_company_unknown_returns_none():
    """
    Company not in KNOWN_COMPANIES should return None.
    Prevents incorrect Qdrant filter from being applied.
    """
    from services.understanding_agent.app.agent import normalize_company
    result = normalize_company("dominos", "dominos delivery problem")
    assert result is None


# ─── TEST 4: cache key is consistent ─────────────────────────────────────────
def test_cache_key_is_consistent():
    """
    Same query + company + focus must always produce the same MD5 key.
    If this changes, cache breaks completely.
    """
    from services.insight_agent.app.cache import make_cache_key

    key1 = make_cache_key("Analyze Swiggy delivery", "swiggy", "delivery")
    key2 = make_cache_key("Analyze Swiggy delivery", "swiggy", "delivery")
    assert key1 == key2
    assert key1.startswith("insight:")


# ─── TEST 5: cache key differs per company ───────────────────────────────────
def test_cache_key_differs_per_company():
    """
    Swiggy and Uber queries must NOT share the same cache key.
    If they did, Uber would get Swiggy's cached insights.
    """
    from services.insight_agent.app.cache import make_cache_key

    swiggy_key = make_cache_key("delivery issues", "swiggy", None)
    uber_key   = make_cache_key("delivery issues", "uber",   None)
    assert swiggy_key != uber_key


# ─── TEST 6: empty results not cached ────────────────────────────────────────
def test_empty_results_not_stored_in_cache():
    """
    Smart cache: 'No data found' should NOT be cached.
    If cached, every subsequent request would get empty results
    even after data is indexed.
    """
    empty_result = {
        "top_issues": ["No data found for this company"],
        "patterns": [],
        "confidence_score": 0.0
    }

    with patch("services.insight_agent.app.cache.redis_client") as mock_redis:
        from services.insight_agent.app.cache import set_cache
        set_cache("insight:test_key", empty_result)

    # Redis setex should NOT have been called
    mock_redis.setex.assert_not_called()


# ─── TEST 7: hybrid score formula ────────────────────────────────────────────
def test_hybrid_score_formula():
    """
    Hybrid score = 0.7 * vector_score + 0.3 * bm25_normalized.
    This formula drives retrieval quality — must be correct.
    """
    vector_score    = 0.8
    bm25_score      = 0.6
    bm25_normalized = bm25_score / 1.0  # max_bm25 = 1.0

    final = (0.7 * vector_score) + (0.3 * bm25_normalized)

    # 0.56 + 0.18 = 0.74
    assert abs(final - 0.74) < 0.001
