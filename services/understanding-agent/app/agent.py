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

        logger.info(f"Extracted: {result}")
        return result

    except json.JSONDecodeError as e:
        logger.error(f"JSON parse error: {e}")
        return {
            "company": company or "unknown",
            "intent": "analyze",
            "focus": None
        }
    except Exception as e:
        logger.error(f"OpenAI error: {e}")
        return {
            "company": company or "unknown",
            "intent": "analyze",
            "focus": None
        }