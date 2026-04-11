import json
from openai import AsyncOpenAI
from .config import settings
from .prompts import UNDERSTANDING_PROMPT
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "../../../"))
from shared.logger import get_logger

logger = get_logger("understanding-agent")

client = AsyncOpenAI(api_key=settings.openai_api_key)


# ---------------------------
# KNOWN कंपनियाँ
# ---------------------------
KNOWN_COMPANIES = ["swiggy", "zomato", "uber"]


# ---------------------------
# NORMALIZATION LOGIC (CRITICAL)
# ---------------------------
def normalize_company(company: str, query: str):
    if not company:
        return None

    company = company.lower().strip()

    # ✅ Exact match
    if company in KNOWN_COMPANIES:
        return company

    # ✅ Partial match (LLM errors fix)
    for c in KNOWN_COMPANIES:
        if c in company:
            return c

    # ✅ Fallback from query
    query = query.lower()
    for c in KNOWN_COMPANIES:
        if c in query:
            return c

    return None


# ---------------------------
# MAIN FUNCTION
# ---------------------------
async def understand_query(query: str, company: str = None) -> dict:
    logger.info(f"Understanding query: {query}")

    user_message = f"Query: {query}"
    if company:
        user_message += f"\nHint - company might be: {company}"

    try:
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": UNDERSTANDING_PROMPT},
                {"role": "user", "content": user_message}
            ],
            temperature=0.1,
            max_tokens=100
        )

        content = response.choices[0].message.content.strip()
        result = json.loads(content)

        # 🔥 RAW OUTPUT
        logger.info(f"Raw LLM output: {result}")

        # 🔥 CRITICAL FIX: Normalize company
        extracted_company = normalize_company(result.get("company"), query)

        logger.info(f"Final normalized company: {extracted_company}")

        final_response = {
            "company": extracted_company or "unknown",
            "intent": result.get("intent", "analyze"),
            "focus": result.get("focus")
        }

        logger.info(f"Final structured output: {final_response}")

        return final_response

    except json.JSONDecodeError as e:
        logger.error(f"JSON parse error: {e}")
        return {
            "company": normalize_company(company, query) or "unknown",
            "intent": "analyze",
            "focus": None
        }

    except Exception as e:
        logger.error(f"OpenAI error: {e}")
        return {
            "company": normalize_company(company, query) or "unknown",
            "intent": "analyze",
            "focus": None
        }