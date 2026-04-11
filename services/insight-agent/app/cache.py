import redis
import json
import hashlib
from .config import settings
from shared.logger import get_logger

logger = get_logger("insight-agent")

redis_client = redis.Redis(
    host=settings.redis_host,
    port=settings.redis_port,
    decode_responses=True
)


def make_cache_key(query: str, company: str, focus: str = None) -> str:
    raw = f"{query}:{company}:{focus}"
    return "insight:" + hashlib.md5(raw.encode()).hexdigest()


def get_cached(key: str) -> dict | None:
    try:
        data = redis_client.get(key)
        if data:
            result = json.loads(data)
            if not result.get("top_issues") or result["top_issues"] == ["No data found for this company"]:
                logger.info(f"Cache SKIP (empty result): {key}")
                return None
            logger.info(f"Cache HIT: {key}")
            return result
    except Exception as e:
        logger.warning(f"Redis get error: {e}")
    return None


def set_cache(key: str, value: dict, ttl: int = 3600):
    try:
        if not value.get("top_issues") or value["top_issues"] == ["No data found for this company"]:
            logger.info(f"Cache SKIP (not caching empty result): {key}")
            return
        redis_client.setex(key, ttl, json.dumps(value))
        logger.info(f"Cache SET: {key}")
    except Exception as e:
        logger.warning(f"Redis set error: {e}")