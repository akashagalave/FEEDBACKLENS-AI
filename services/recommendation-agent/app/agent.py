import json
from openai import AsyncOpenAI
from .config import settings
from .prompts import RECOMMENDATION_PROMPT
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "../../../"))
from shared.logger import get_logger

logger = get_logger("recommendation-agent")

client = AsyncOpenAI(api_key=settings.openai_api_key)


async def generate_recommendations(
    company: str,
    top_issues: list[str],
    patterns: list[str]
) -> dict:

    context = f"""
Company: {company}

Top Issues:
{chr(10).join(f"- {issue}" for issue in top_issues)}

Patterns:
{chr(10).join(f"- {pattern}" for pattern in patterns)}
"""

    try:
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": RECOMMENDATION_PROMPT},
                {"role": "user", "content": context}
            ],
            temperature=0.3,
            max_tokens=500
        )

        content = response.choices[0].message.content.strip()
        result = json.loads(content)

        logger.info(f"Recommendations generated for {company}")
        return result

    except json.JSONDecodeError as e:
        logger.error(f"JSON parse error: {e}")
        return {"recommendations": ["Unable to generate recommendations"]}
    except Exception as e:
        logger.error(f"OpenAI error: {e}")
        raise