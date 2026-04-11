import json
from openai import AsyncOpenAI
from .config import settings
from .prompts import INSIGHT_PROMPT
from .hybrid_search import hybrid_search
from .cache import make_cache_key, get_cached, set_cache
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "../../../"))
from shared.logger import get_logger

logger = get_logger("insight-agent")

client = AsyncOpenAI(api_key=settings.openai_api_key)


async def generate_insights(
    query: str,
    company: str,
    focus: str = None,
    top_k: int = 10
) -> dict:

    cache_key = make_cache_key(query, company, focus)
    cached = get_cached(cache_key)
    if cached:
        return cached


    chunks = await hybrid_search(query, company, focus, top_k)

    if not chunks:
        logger.warning(f"No chunks found for company={company}")
        return {
            "top_issues": ["No data found for this company"],
            "patterns": [],
            "sample_reviews": [],
            "confidence_score": 0.0
        }


    reviews_text = "\n".join([
        f"- [{c.issue}] {c.review}" for c in chunks
    ])

    sample_reviews = [c.review for c in chunks[:3]]
    logger.info(f"Retrieved {len(chunks)} chunks for {company}")

   
    try:
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": INSIGHT_PROMPT},
                {"role": "user", "content": f"Company: {company}\n\nReviews:\n{reviews_text}"}
            ],
            temperature=0.2,
            max_tokens=500
        )

        content = response.choices[0].message.content.strip()
        result = json.loads(content)
        result["sample_reviews"] = sample_reviews

        
        set_cache(cache_key, result)

        logger.info(f"Insights generated for {company}")
        return result

    except json.JSONDecodeError as e:
        logger.error(f"JSON parse error: {e}, content: {content}")
        return {
            "top_issues": ["Error parsing insights"],
            "patterns": [],
            "sample_reviews": sample_reviews,
            "confidence_score": 0.0
        }
    except Exception as e:
        logger.error(f"OpenAI error: {e}")
        raise